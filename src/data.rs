use {
    super::{ANALYSING_DATA_FILE_NAME, ANALYSIS_CHUNK_SIZE, CORRELATION_THRESHOLD, SCANNING_DATA_FILE_NAME, TILE_COUNT_PER_SIDE},
    crate::{chunk, util},
    color_eyre::eyre::Result,
    eframe::egui::mutex::{Mutex, MutexGuard},
    image::GenericImageView,
    rayon::iter::{IntoParallelRefIterator, ParallelIterator},
    serde::{Deserialize, Serialize},
    std::{
        fs::{File, OpenOptions},
        io::{self, BufRead, Write},
        iter::zip,
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Instant,
    },
    tracing::{error, info},
};

pub trait DbPathProvider {
    fn db_path_name() -> &'static str;
}

impl DbPathProvider for DataEntry {
    fn db_path_name() -> &'static str {
        SCANNING_DATA_FILE_NAME
    }
}

impl DbPathProvider for CorrelationEntry {
    fn db_path_name() -> &'static str {
        ANALYSING_DATA_FILE_NAME
    }
}

pub fn get_db_path<T>(folder_path: &str) -> String
where
    T: DbPathProvider,
{
    format!("{folder_path}/{}", T::db_path_name())
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct DataEntry {
    pub(crate) filename: String,
    pub(crate) data: Vec<u8>,
}

impl DataEntry {
    /// Creates a new data entry, with the given name and the data slice.
    pub(crate) fn new(file_path: impl Into<String>, data: &[u8]) -> Self {
        Self { filename: file_path.into(), data: data.into() }
    }

    /// Generates a list of all potential image files (recursively), that are within the given folder.
    ///
    /// Will only check for jpg/jpeg & png files. `WebP is currently not supported.`
    pub(crate) fn generate_file_list(folder_path: &str, recursive: bool) -> Vec<String> {
        let mut file_list = Vec::new();
        util::visit_dirs(Path::new(folder_path), recursive, &mut |entry| {
            if let Some(extension) = entry.extension() {
                if let Some(ext) = extension.to_ascii_lowercase().to_str() {
                    match ext {
                        "png" | "jpg" | "jpeg" | "webp" => {
                            if let Some(filepath) = entry.to_str() {
                                file_list.push(filepath.to_string());
                            } else {
                                error!("Could not convert file path to string for entry: {entry:?}");
                            }
                        }
                        _ => {}
                    }
                } else {
                    error!("Could not get file extension as a string for entry: {entry:?}");
                }
            }
        });
        file_list.sort();
        file_list
    }

    /// Will open the file at the given path and then return the data vector with the average tile brightness for each of the tiles.
    #[inline]
    pub(crate) fn calculate_image_tile_data(file_path: &str) -> Vec<u8> {
        let mut tiles_averages = Vec::<u8>::with_capacity(TILE_COUNT_PER_SIDE as usize);
        match image::ImageReader::open(file_path) {
            Ok(opened_image) => match opened_image.decode() {
                Ok(decoded_image) => {
                    let (width, height) = decoded_image.dimensions();
                    let rect_width = width / TILE_COUNT_PER_SIDE;
                    let rect_height = height / TILE_COUNT_PER_SIDE;
                    for i in 0..TILE_COUNT_PER_SIDE {
                        for j in 0..TILE_COUNT_PER_SIDE {
                            let x = i * rect_width;
                            let y = j * rect_height;
                            let tile = decoded_image.view(x, y, rect_width, rect_height).to_image();
                            let tile_pixel_sum: u64 = tile.par_iter().map(|&v| u64::from(v)).sum();
                            let tile_average = u8::try_from(tile_pixel_sum / (4 * u64::from(rect_width) * u64::from(rect_height))).expect("Could not get average tile brightness"); // need to divide by 4, since we read RGBA

                            tiles_averages.push(tile_average);
                        }
                    }
                }
                Err(e) => error!("Could not process image {file_path}, error: {e:?}"),
            },
            Err(e) => error!("Could not process image {file_path}, error: {e:?}"),
        }
        tiles_averages
    }

    /// Reads the entries from a given folder, determining the correct name automatically.
    ///
    /// Returning a Result vec of all entries.
    pub(crate) fn read_from_folder(folder_path: &str) -> Result<Vec<Self>> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        let file_handle = File::open(target_path)?;
        Ok(io::BufReader::new(file_handle)
            .lines()
            .filter_map(|line_result| match line_result {
                Ok(line) if !line.is_empty() => Some(line),
                _ => None,
            })
            .map(|line| {
                let entry: Self = serde_json::from_str(&line).expect("Could not extract entry from string");
                entry
            })
            .collect())
    }

    /// Extracts only the file names from the given data slice
    pub(crate) fn extract_filenames(entries: &[Self]) -> Vec<String> {
        entries.iter().map(|entry| entry.filename.clone()).collect()
    }

    /// Creates the database file in the given folder path to store the data.
    pub(crate) fn create_db(folder_path: &str) -> Result<()> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }

    /// Will save the entry to the given file. Each entry will be put on a new line.
    ///
    /// New lines are automatically added.
    pub(crate) fn save(&self, mutex_file_path: &MutexGuard<File>) -> Result<()> {
        serde_json::to_writer(&**mutex_file_path, self)?;
        writeln!(&**mutex_file_path)?;
        Ok(())
    }

    /// Will check all the jpg/png files in the given folder (recursively).
    ///
    /// The data is then stored in the given folder.
    ///
    /// The progress and time remaining are automatically updated during the process. Worker count can be adjusted.
    pub(crate) fn scan(folder_path: &str, files_to_scan_list: &[String], scanning: &Arc<AtomicBool>, progress: &Arc<Mutex<f64>>, time_remaining: &Arc<Mutex<String>>, worker_count: u16) {
        let db_path = get_db_path::<Self>(folder_path);
        Self::create_db(folder_path).expect("Somehow the selected folder was deleted and is no longer available.");
        {
            // Reset from last run and drops it afterwards since it is no longer needed. The update will create a new lock.
            *progress.lock() = 0.0;
        }

        info!("Found {} potential files for scanning.", files_to_scan_list.len());

        let existing_entries = Self::read_from_folder(folder_path).expect("Could not read entries from folder.");
        let existing_entries_filenames = Self::extract_filenames(&existing_entries);
        info!("Found {} entries already in in database.", existing_entries_filenames.len());

        // Filtering out all entries, that already exist in the database.
        let files_to_scan: Vec<String> = files_to_scan_list.iter().filter(|filename| !existing_entries_filenames.contains(*filename)).cloned().collect();

        if files_to_scan.is_empty() {
            info!("Nothing to do, all files have already been scanned!");
            scanning.store(false, Ordering::Relaxed);
        } else {
            print!("Scanning {} files, skipping already scanned files. ", files_to_scan.len());
            info!("Using {worker_count} workers");

            let files_processed = Arc::new(Mutex::new(0));
            let files_to_process = files_to_scan.len();
            let work_queue = Arc::new(Mutex::new(files_to_scan));

            let db_file = OpenOptions::new().append(true).open(&db_path).expect("Cannot open file");
            let db_file = Arc::new(Mutex::new(db_file));

            let start_time = Instant::now();
            crossbeam::scope(|scope| {
                for _worker in 0..worker_count {
                    let work_queue = Arc::clone(&work_queue);
                    let files_processed = Arc::clone(&files_processed);
                    let worker_db_file = Arc::clone(&db_file);

                    scope.spawn(move |_| {
                        'file_iteration_loop: while let Some(file) = {
                            let mut queue = work_queue.lock();
                            queue.pop()
                        } {
                            // Calculate data
                            if Path::new(&file).exists() {
                                let averages = Self::calculate_image_tile_data(&file);
                                let entry = Self::new(&file, &averages);
                                {
                                    let worker_db_file = worker_db_file.lock();
                                    // TODO: do we want to stop if the saving fails? Should not fail under normal conditions.
                                    let _ = entry.save(&worker_db_file);
                                }
                            } else {
                                // We simply continue and ignore the incident.
                                error!("File {file} has been removed and can not be read. Continuing.");
                            }
                            {
                                *files_processed.lock() += 1;
                            }

                            let processed = *files_processed.lock();
                            let avg_time_per_file = start_time.elapsed() / u32::try_from(processed + 1).expect("Number of processed files exceeds 2^32");

                            let remaining_files = files_to_process - processed;
                            let estimated_remaining_time = avg_time_per_file * u32::try_from(remaining_files).expect("Number of remaining files exceeds 2^32");
                            if processed % 10 == 0 {
                                let mut time_remaining = time_remaining.lock();
                                *time_remaining = format!("Estimated: {estimated_remaining_time:06.2?} remaining.");
                            }

                            {
                                // Update progress
                                let mut progress_percent = progress.lock();
                                let processed = *files_processed.lock();
                                *progress_percent = (1.0 + processed as f64) / files_to_process as f64;
                            }

                            if !scanning.load(Ordering::Relaxed) {
                                break 'file_iteration_loop;
                            }
                        }
                    });
                }
            })
            .unwrap();

            scanning.store(false, Ordering::Relaxed);

            let elapsed = start_time.elapsed().as_millis();
            info!("Scanning took: {elapsed} ms.");
        }
    }

    pub(crate) fn scan_folders(folders: &[String], scanning: &Arc<AtomicBool>, recursive: &[bool], progress: &Arc<Mutex<f64>>, time_remaining: &Arc<Mutex<String>>, worker_count: u16) {
        let mut files_to_scan_list = Vec::<String>::new();
        for (idx, folder) in folders.iter().enumerate() {
            let mut file_list_for_folder = Self::generate_file_list(folder, *recursive.get(idx).expect("Somehow we got to the point, that our recursion and folder arrays are not matching."));
            files_to_scan_list.append(&mut file_list_for_folder);
        }
        // debug!(&files_to_scan_list);
        Self::scan(
            folders.first().expect("Somehow we got to the point, that our recursion and folder arrays are not matching."),
            &files_to_scan_list,
            scanning,
            progress,
            time_remaining,
            worker_count,
        );
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CorrelationEntry {
    pub(crate) first_path: String,
    pub(crate) second_path: String,
    pub(crate) corr: f32,
}

impl CorrelationEntry {
    pub(crate) fn new(first_path: impl Into<String>, second_path: impl Into<String>, corr: f32) -> Self {
        Self { first_path: first_path.into(), second_path: second_path.into(), corr }
    }

    pub(crate) fn get_correlation_db_path(folder_path: &str) -> String {
        format!("{folder_path}/{ANALYSING_DATA_FILE_NAME}")
    }

    pub(crate) fn create_correlation_db(folder_path: &str) -> Result<()> {
        let target_file = Self::get_correlation_db_path(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }

    pub(crate) fn read_correlation_db(folder_path: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let file_name = Self::get_correlation_db_path(folder_path);
        let file_handle = File::open(file_name)?;
        let mut result: Vec<Self> = io::BufReader::new(file_handle)
            .lines()
            .map(|line| line.and_then(|line_str| serde_json::from_str(&line_str).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))))
            .collect::<Result<Vec<Self>, _>>()?;
        result.sort_by(|a, b| b.corr.partial_cmp(&a.corr).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }

    pub(crate) fn save(&self, mutex_file_path: &MutexGuard<File>) -> Result<()> {
        serde_json::to_writer(&**mutex_file_path, self)?;
        writeln!(&**mutex_file_path)?;
        Ok(())
    }

    #[inline]
    pub(crate) fn correlation_coefficient(a: &[u8], b: &[u8]) -> f32 {
        // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        if a.len() != b.len() {
            return 0.0;
        }

        let len_a = a.len() as f32;
        let len_b = b.len() as f32;
        // data is small, therefore making a copy is fast, type conversion is expensive too
        let a_f32: Vec<f32> = a.iter().map(|v| f32::from(*v)).collect();
        let b_f32: Vec<f32> = b.iter().map(|v| f32::from(*v)).collect();
        let mean_a: f32 = a_f32.iter().sum::<f32>() / len_a;
        let mean_b: f32 = b_f32.iter().sum::<f32>() / len_b;

        let mut max_corr_coeff: f32 = 0.0;

        let var_a: f32 = a_f32.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / (len_a - 1.0);
        if var_a == 0.0 {
            return 0.0;
        }
        for variation in 0..=2 {
            let b_data = if b_f32.len() == 16 {
                match variation {
                    0 => b_f32.clone(),
                    1 => vec![b_f32[3], b_f32[2], b_f32[1], b_f32[0], b_f32[7], b_f32[6], b_f32[5], b_f32[4], b_f32[11], b_f32[10], b_f32[9], b_f32[8], b_f32[15], b_f32[14], b_f32[13], b_f32[12]], // horizontally mirrored
                    2 => vec![b_f32[12], b_f32[13], b_f32[14], b_f32[15], b_f32[8], b_f32[9], b_f32[10], b_f32[11], b_f32[4], b_f32[5], b_f32[6], b_f32[7], b_f32[0], b_f32[1], b_f32[2], b_f32[3]], // vertically mirrored
                    _ => panic!("Invalid variation"),
                }
            } else {
                b_f32.clone()
            };

            let var_b: f32 = b_data.clone().iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / (len_b - 1.0);
            if var_b == 0.0 {
                continue;
            }

            let cov = zip(a_f32.clone(), b_data.clone()).map(|(a, b)| (a - mean_a) * (b - mean_b)).sum::<f32>() / (len_a - 1.0);
            max_corr_coeff = max_corr_coeff.max((cov / (var_a * var_b).sqrt()).abs()); // we are only interested in the absolute value
        }
        max_corr_coeff
    }

    pub(crate) fn analyse(folder_path: &str, analysing: &Arc<AtomicBool>, progress: &Arc<Mutex<f64>>, time_remaining: &Arc<Mutex<String>>, worker_count: u16) {
        info!("Using {worker_count} workers.\nAnalysing database in \"{folder_path}\". Calculating correlation coefficients for entries.");
        Self::create_correlation_db(folder_path).expect("TODO: panic message");
        let db_path = get_db_path::<Self>(folder_path);

        let data_entries = DataEntry::read_from_folder(folder_path).expect("Could not read entries from folder.");
        let file_names = DataEntry::extract_filenames(&data_entries);
        info!("Found {} entries for analysis", data_entries.len());

        let total_combinations = file_names.len() * (file_names.len() - 1) / 2;
        let chunk_count = (total_combinations + 1) / ANALYSIS_CHUNK_SIZE;

        info!("Expecting {chunk_count} chunks of {ANALYSIS_CHUNK_SIZE} combinations for the total of {total_combinations} combinations.\nLoading data...");

        let mut chunk_counter = 0;
        let mut chunked_combinations = chunk::ChunkedCombinations::new(&file_names, 2, ANALYSIS_CHUNK_SIZE);

        let outer_start_time = Instant::now();
        #[allow(clippy::while_let_on_iterator)]
        while let Some(chunk) = chunked_combinations.next() {
            {
                // Reset from last run
                *progress.lock() = 0.0;
            }

            let start_time = Instant::now();

            let combinations_data = Self::generate_combination_data(&data_entries, &chunk);

            info!("\rLoading data took: {} milliseconds.", start_time.elapsed().as_millis());

            let start_time = Instant::now();
            let already_performed = Self::read_correlation_db(folder_path).unwrap_or_else(|e| {
                error!("Error: {e}");
                Vec::<Self>::new()
            });
            let combinations_data_filtered = Self::filter_combination_data(combinations_data, &already_performed);
            info!("Filtering data took: {} milliseconds.", start_time.elapsed().as_millis());
            let combinations_to_process = combinations_data_filtered.len();
            info!("Processing {combinations_to_process} combinations, using {worker_count} workers.");

            let combinations_processed = Arc::new(Mutex::new(0));
            let work_queue = Arc::new(Mutex::new(combinations_data_filtered.clone()));

            let target_db_file_handle = OpenOptions::new().append(true).open(&db_path).expect("Cannot open file");
            let target_db_file_mutex = Arc::new(Mutex::new(target_db_file_handle));

            let start_time = Instant::now();
            crossbeam::scope(|scope| {
                for _worker in 0..worker_count {
                    let work_queue = Arc::clone(&work_queue);
                    let files_processed = Arc::clone(&combinations_processed);
                    let worker_db_file = Arc::clone(&target_db_file_mutex);

                    scope.spawn(move |_| {
                        'combinations_loop: while let Some((first, first_data, second, second_data)) = {
                            let mut work_item = work_queue.lock();
                            work_item.pop()
                        } {
                            let corr: f32 = Self::correlation_coefficient(&first_data, &second_data);
                            let entry = Self::new(&first, &second, corr);
                            {
                                if entry.corr > CORRELATION_THRESHOLD {
                                    let worker_db_file = &worker_db_file.lock();
                                    let _ = entry.save(worker_db_file);
                                }
                            }
                            {
                                *files_processed.lock() += 1;
                            }

                            let processed = *files_processed.lock();
                            let avg_time_per_file = start_time.elapsed() / (processed + 1);

                            let remaining_files = combinations_to_process - processed as usize;
                            let estimated_remaining_time = avg_time_per_file * remaining_files.try_into().expect("Remaing files exceeds 2^32");
                            if processed % 10 == 0 {
                                *time_remaining.lock() = format!(
                                    "Processing chunk {} of {chunk_count}. Processed {processed} of {combinations_to_process}. Estimated: {estimated_remaining_time:06.2?} remaining.",
                                    chunk_counter + 1
                                );
                            }
                            {
                                // Update progress
                                *progress.lock() = (1.0 + f64::from(*files_processed.lock())) / combinations_to_process as f64;
                            }
                            if !analysing.load(Ordering::Relaxed) {
                                break 'combinations_loop;
                            }
                        }
                    });
                }
            })
            .unwrap();

            let elapsed = start_time.elapsed().as_millis();
            info!("\nChunk Analysing took: {elapsed} ms");

            chunk_counter += 1;
            info!("Processed chunk {chunk_counter} of {chunk_count}");
        }
        analysing.store(false, Ordering::Relaxed);
        let elapsed = outer_start_time.elapsed().as_millis();
        info!("Analysis done! Analysing took: {elapsed} ms");
    }

    fn filter_combination_data(combinations_data: Vec<(String, Vec<u8>, String, Vec<u8>)>, already_performed: &[Self]) -> Vec<(String, Vec<u8>, String, Vec<u8>)> {
        let combinations_data_filtered: Vec<(String, Vec<u8>, String, Vec<u8>)> = if already_performed.is_empty() {
            combinations_data
        } else {
            combinations_data.par_iter().filter(|(first, _, second, _)| !already_performed.iter().any(|e| e.first_path == *first && e.second_path == *second)).cloned().collect()
        };
        combinations_data_filtered
    }

    fn generate_combination_data(data_entries: &[DataEntry], chunk: &[Vec<&String>]) -> Vec<(String, Vec<u8>, String, Vec<u8>)> {
        let combinations_data: Vec<(String, Vec<u8>, String, Vec<u8>)> = chunk
            .par_iter()
            .map(|combination| {
                let first = combination.first().unwrap().to_owned();
                let second = combination.last().unwrap().to_owned();
                let first_data = data_entries.iter().find(|e| e.filename == *first).unwrap().data.clone();
                let second_data = data_entries.iter().find(|e| e.filename == *second).unwrap().data.clone();
                (first.clone(), first_data, second.clone(), second_data)
            })
            .collect();
        combinations_data
    }

    pub(crate) fn analyse_folders(folder_paths: &Vec<String>, analysing: &Arc<AtomicBool>, progress: &Arc<Mutex<f64>>, time_remaining: &Arc<Mutex<String>>, worker_count: u16) {
        for folder_path in folder_paths {
            Self::analyse(folder_path, analysing, progress, time_remaining, worker_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use {crate::data, pretty_assertions::assert_eq};

    #[test]
    fn test_corr() {
        #![allow(clippy::float_cmp)]
        let one: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let two: Vec<u8> = vec![2, 4, 7, 9, 12, 14];
        let result = data::CorrelationEntry::correlation_coefficient(&one, &two);
        assert_eq!(0.998_381_5, result);

        let one: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let two: Vec<u8> = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];
        let result = data::CorrelationEntry::correlation_coefficient(&one, &two);
        assert_eq!(1.0, result);

        let one: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let two: Vec<u8> = vec![4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13];
        let result = data::CorrelationEntry::correlation_coefficient(&one, &two);
        assert_eq!(1.0, result);

        let one: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let two: Vec<u8> = vec![13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4];
        let result = data::CorrelationEntry::correlation_coefficient(&one, &two);
        assert_eq!(1.0, result);
    }
}
