#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

use {
    crate::{
        constants::{ANALYSING_DATA_FILE_NAME, SCANNING_DATA_FILE_NAME},
        progress::LockFreeProgress,
        util,
    },
    eframe::egui::mutex::{Mutex, MutexGuard},
    image::GenericImageView,
    rayon::{
        iter::{IntoParallelRefIterator, ParallelIterator},
        prelude::*,
    },
    serde::{Deserialize, Serialize},
    std::{
        fs::{File, OpenOptions},
        io::{self, BufRead, Write},
        iter::zip,
        path::Path,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
        time::Instant,
    },
    tracing::{error, info},
};

/// Will split the image into 4x4 tiles
const TILE_COUNT_PER_SIDE: usize = 4;
const TOTAL_TILE_COUNT: usize = TILE_COUNT_PER_SIDE * TILE_COUNT_PER_SIDE;

/// Generates the lookup table for horizontal mirroring, for the set number of tiles.
const fn generate_h_mirror() -> [usize; TOTAL_TILE_COUNT] {
    let mut arr = [0; TOTAL_TILE_COUNT];
    let mut i = 0;
    while i < TOTAL_TILE_COUNT {
        let row = i / TILE_COUNT_PER_SIDE;
        let col = i % TILE_COUNT_PER_SIDE;
        arr[i] = row * TILE_COUNT_PER_SIDE + (TILE_COUNT_PER_SIDE - 1 - col);
        i += 1;
    }
    arr
}

/// Generates the lookup table for vertical mirroring, for the set number of tiles.
const fn generate_v_mirror() -> [usize; TOTAL_TILE_COUNT] {
    let mut arr = [0; TOTAL_TILE_COUNT];
    let mut i = 0;
    while i < TOTAL_TILE_COUNT {
        let row = i / TILE_COUNT_PER_SIDE;
        let col = i % TILE_COUNT_PER_SIDE;
        arr[i] = (TILE_COUNT_PER_SIDE - 1 - row) * TILE_COUNT_PER_SIDE + col;
        i += 1;
    }
    arr
}

/// Contains the horizontal mirroring indices.
const H_MIRROR: [usize; TOTAL_TILE_COUNT] = generate_h_mirror();
/// Contains the horizontal mirroring indices.
const V_MIRROR: [usize; TOTAL_TILE_COUNT] = generate_v_mirror();
/// All the currently supported image format extensions.
pub const SUPPORTED_IMAGE_FILE_EXTENSION: [&str; 4] = ["png", "jpg", "jpeg", "webp"];

/// Provides the database path for a given type (`CorrelationEntry` or `DataEntry`).
pub trait DbPathProvider {
    fn db_path_name() -> &'static str;
}

/// Returns the database path for `DataEntry`.
impl DbPathProvider for DataEntry {
    fn db_path_name() -> &'static str {
        SCANNING_DATA_FILE_NAME
    }
}

/// Returns the database path for `CorrelationEntry`.
impl DbPathProvider for CorrelationEntry {
    fn db_path_name() -> &'static str {
        ANALYSING_DATA_FILE_NAME
    }
}

/// Returns the given path for the database file.
#[must_use]
pub fn get_db_path<T>(folder_path: &str) -> String
where
    T: DbPathProvider,
{
    format!("{folder_path}/{}", T::db_path_name())
}

/// Interface for the different entry types to save the entry to disk.
trait SaveEntry: Serialize {
    fn save(&self, mutex_file_path: &MutexGuard<File>) -> Result<(), Box<dyn std::error::Error>>;
}

/// Will save the entry to the given file. Each entry will be put on a new line.
///
/// New lines are automatically added.
impl<T> SaveEntry for T
where
    T: Serialize,
{
    fn save(&self, mutex_file_path: &MutexGuard<File>) -> Result<(), Box<dyn std::error::Error>> {
        serde_json::to_writer(&**mutex_file_path, self)?;
        writeln!(&**mutex_file_path)?;
        Ok(())
    }
}

/// Holds the tile data, for a given file.
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
    /// Will only check for jpg/jpeg, png & webp files.
    pub(crate) fn generate_file_list(folder_path: &str, recursive: bool, exclude: Option<&[String]>) -> Vec<String> {
        let mut file_list = Vec::new();
        util::visit_dirs(Path::new(folder_path), recursive, exclude, &mut |entry| {
            if let Some(extension) = entry.extension() {
                if let Some(ext) = extension.to_ascii_lowercase().to_str() {
                    if SUPPORTED_IMAGE_FILE_EXTENSION.contains(&ext) {
                        if let Some(filepath) = entry.to_str() {
                            file_list.push(filepath.to_string());
                        } else {
                            error!("Could not convert file path to string for entry: {entry:?}");
                        }
                    }
                } else {
                    error!("Could not get file extension as a string for entry: {entry:?}");
                }
            }
        });
        file_list.sort();
        file_list
    }

    /// # Errors
    /// Will return an empty vector, if the image file can not be opened or read correctly.
    ///
    /// Will open the file at the given path and then return the data vector with the average tile brightness for each of the tiles.
    #[inline]
    pub(crate) fn calculate_image_tile_data(file_path: &str) -> Vec<u8> {
        #![allow(clippy::cast_possible_truncation)]
        let mut tiles_averages = Vec::<u8>::with_capacity(TOTAL_TILE_COUNT);
        match image::ImageReader::open(file_path) {
            Ok(opened_image) => match opened_image.decode() {
                Ok(decoded_image) => {
                    let (image_width, image_height) = decoded_image.dimensions();
                    let tile_width = image_width / TILE_COUNT_PER_SIDE as u32;
                    let tile_height = image_height / TILE_COUNT_PER_SIDE as u32;
                    if tile_width == 0 || tile_height == 0 {
                        error!("Image or tile too small to process. File {file_path}");
                        return tiles_averages;
                    }
                    for i in 0..TILE_COUNT_PER_SIDE as u32 {
                        for j in 0..TILE_COUNT_PER_SIDE as u32 {
                            let x = i * tile_width;
                            let y = j * tile_height;
                            let tile = decoded_image.view(x, y, tile_width, tile_height).to_image();
                            let tile_pixel_sum: u64 = tile.par_iter().map(|&v| u64::from(v)).sum();
                            let tile_average = u8::try_from(tile_pixel_sum / (4 * u64::from(tile_width) * u64::from(tile_height))).expect("Could not get average tile brightness"); // need to divide by 4, since we read RGBA

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
    pub(crate) fn read_from_folder(folder_path: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
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

    /// Creates the database file in the given folder path to store the data.
    pub(crate) fn create_db(folder_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }

    /// # Panics
    /// When the given folder can not be read.
    pub fn scan_with_progress(folder_path: &str, files_to_scan_list: &[String], scanning: &Arc<AtomicBool>, progress_tracker: &Arc<LockFreeProgress>, worker_count: u16) {
        let db_path = get_db_path::<Self>(folder_path);
        Self::create_db(folder_path).expect("Could not create database");

        info!("Found {} potential files for scanning.", files_to_scan_list.len());

        let existing_entries = Self::read_from_folder(folder_path).expect("Could not read entries from folder");
        let existing_filenames: std::collections::HashSet<String> = existing_entries.into_iter().map(|entry| entry.filename).collect();

        info!("Found {} entries already in database.", existing_filenames.len());

        // filter out files that already exist in the database
        let files_to_scan: Vec<String> = files_to_scan_list.iter().filter(|filename| !existing_filenames.contains(*filename)).cloned().collect();

        if files_to_scan.is_empty() {
            info!("Nothing to do, all files have already been scanned!");
            scanning.store(false, Ordering::Relaxed);
            return;
        }

        info!("Scanning {} files, skipping already scanned files.", files_to_scan.len());
        info!("Using {worker_count} workers");

        progress_tracker.reset(files_to_scan.len());

        let (result_tx, result_rx) = std::sync::mpsc::channel::<Self>();
        let files_processed = Arc::new(AtomicUsize::new(0));

        let write_handle = {
            let files_processed = Arc::clone(&files_processed);
            std::thread::spawn(move || {
                use std::io::BufWriter;

                let file = OpenOptions::new().create(true).append(true).open(&db_path).expect("Cannot open database file");

                let mut writer = BufWriter::with_capacity(64 * 1024, file); // 64KB buffer

                let value = files_processed.load(Ordering::Relaxed);
                while let Ok(entry) = result_rx.recv() {
                    if let Err(e) = serde_json::to_writer(&mut writer, &entry) {
                        error!("Failed to write entry: {}", e);
                    }
                    if let Err(e) = writeln!(&mut writer) {
                        error!("Failed to write newline: {}", e);
                    }

                    // Flush every 100 entries to balance performance and data safety
                    if value.is_multiple_of(100) {
                        let _ = writer.flush();
                    }
                }

                let _ = writer.flush();
            })
        };

        let start_time = Instant::now();

        let result_tx_clone = result_tx.clone();
        let files_processed_clone = Arc::clone(&files_processed);
        let progress_tracker_clone = Arc::clone(progress_tracker);
        let scanning_clone = Arc::clone(scanning);

        files_to_scan.into_par_iter().for_each(move |file_path| {
            if !scanning_clone.load(Ordering::Relaxed) {
                return;
            }

            if Path::new(&file_path).exists() {
                let averages = Self::calculate_image_tile_data(&file_path);
                if averages.len() == TOTAL_TILE_COUNT {
                    let entry = Self::new(&file_path, &averages);

                    // Send to writer thread (non-blocking)
                    if result_tx_clone.send(entry).is_err() {
                        error!("Failed to send entry to writer thread");
                    }
                }
            } else {
                error!("File {file_path} has been removed and cannot be read. Continuing.");
            }

            let processed_count = files_processed_clone.fetch_add(1, Ordering::Relaxed) + 1;
            progress_tracker_clone.increment();

            if processed_count.is_multiple_of(50) {
                let (current_processed, total, progress_pct) = progress_tracker_clone.get_progress();
                if let Some(remaining_time) = progress_tracker_clone.estimated_remaining()
                    && processed_count.is_multiple_of(1000)
                {
                    info!("Processed {}/{} files ({:.1}%), estimated remaining: {:.2?}", current_processed, total, progress_pct * 100.0, remaining_time);
                }
            }
        });

        drop(result_tx); // this will cause the writer thread to exit
        let _ = write_handle.join();

        scanning.store(false, Ordering::Relaxed);

        let elapsed = start_time.elapsed().as_millis();
        info!("Scanning took: {elapsed} ms.");
    }

    /// # Panics
    /// May panic when the given folder can not be accessed.
    pub fn scan_folders_with_progress(folders: &[String], exclude: Option<&[String]>, scanning: &Arc<AtomicBool>, recursive: &[bool], progress_tracker: &Arc<LockFreeProgress>, worker_count: u16) {
        let mut files_to_scan_list = Vec::<String>::new();
        for (idx, folder) in folders.iter().enumerate() {
            let mut file_list_for_folder = Self::generate_file_list(folder, *recursive.get(idx).expect("Folder and recursive arrays should match"), exclude);
            files_to_scan_list.append(&mut file_list_for_folder);
        }

        Self::scan_with_progress(folders.first().expect("At least one folder should be provided"), &files_to_scan_list, scanning, progress_tracker, worker_count);
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

    pub(crate) fn create_correlation_db(folder_path: &str) -> Result<(), Box<dyn std::error::Error>> {
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

    fn mirror_horizontal(input: &[f32; TOTAL_TILE_COUNT]) -> [f32; TOTAL_TILE_COUNT] {
        let mut out = [0f32; TOTAL_TILE_COUNT];
        for i in 0..TOTAL_TILE_COUNT {
            out[i] = input[H_MIRROR[i]];
        }
        out
    }

    fn mirror_vertical(input: &[f32; TOTAL_TILE_COUNT]) -> [f32; TOTAL_TILE_COUNT] {
        let mut out = [0f32; TOTAL_TILE_COUNT];
        for i in 0..TOTAL_TILE_COUNT {
            out[i] = input[V_MIRROR[i]];
        }
        out
    }

    /// Calculate correlation coefficient between two tile data arrays.
    ///
    /// This method automatically uses to either the SIMD or non-SIMD
    /// implementation based on whether the "simd" feature is enabled.
    ///
    /// # Features
    /// - With `simd` feature (default): Uses SIMD-optimized implementation for better performance
    /// - Without `simd` feature (--no-default-features): Uses standard implementation for broader compatibility
    #[must_use]
    pub fn correlation_coefficient(a: &[f32; TOTAL_TILE_COUNT], b: &[f32; TOTAL_TILE_COUNT]) -> f32 {
        #[cfg(feature = "simd")]
        {
            Self::correlation_coefficient_simd(a, b)
        }
        #[cfg(not(feature = "simd"))]
        {
            Self::correlation_coefficient_default(a, b)
        }
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn correlation_coefficient_default(a: &[f32; TOTAL_TILE_COUNT], b: &[f32; TOTAL_TILE_COUNT]) -> f32 {
        // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

        let len_a = a.len() as f32;
        let mean_a: f32 = a.iter().sum::<f32>() / len_a;
        let len_b = b.len() as f32;
        let mean_b: f32 = b.iter().sum::<f32>() / len_b;

        let var_a: f32 = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / (len_a - 1.0);
        if var_a == 0.0 {
            return 0.0;
        }

        let mut max_corr_coeff: f32 = 0.0;
        for variation in 0..=2 {
            let b_variant: [f32; TOTAL_TILE_COUNT] = match variation {
                0 => *b,
                1 => Self::mirror_horizontal(b),
                2 => Self::mirror_vertical(b),
                _ => unreachable!(),
            };

            let var_b: f32 = b_variant.iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / (len_b - 1.0);
            if var_b == 0.0 {
                continue;
            }

            let cov = zip(a, b_variant).map(|(a, b)| (a - mean_a) * (b - mean_b)).sum::<f32>() / (len_a - 1.0);
            max_corr_coeff = max_corr_coeff.max((cov / (var_a * var_b).sqrt()).abs()); // we are only interested in the absolute value
        }
        max_corr_coeff
    }

    #[cfg(feature = "simd")]
    #[must_use]
    pub fn correlation_coefficient_simd(first: &[f32; TOTAL_TILE_COUNT], second: &[f32; TOTAL_TILE_COUNT]) -> f32 {
        // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        const LANES: usize = 8;

        // Calculate mean_a with SIMD
        let mut sum_first = f32x8::splat(0.0);
        let chunks_first = first.chunks_exact(LANES);
        let remainder_first = chunks_first.remainder();

        for chunk in chunks_first {
            sum_first += f32x8::from_slice(chunk);
        }

        let mean_first = (sum_first.reduce_sum() + remainder_first.iter().sum::<f32>()) / TOTAL_TILE_COUNT as f32;
        let mean_first_simd_vec = f32x8::splat(mean_first);

        // Calculate variance of a with SIMD
        let mut var_sum_first = f32x8::splat(0.0);
        let chunks_first = first.chunks_exact(LANES);
        let remainder_first = chunks_first.remainder();

        for chunk in chunks_first {
            let vals = f32x8::from_slice(chunk);
            let diff = vals - mean_first_simd_vec;
            var_sum_first += diff * diff;
        }

        let var_first = (var_sum_first.reduce_sum() + remainder_first.iter().map(|v| (v - mean_first).powi(2)).sum::<f32>()) / (TOTAL_TILE_COUNT as f32 - 1.0);

        if var_first == 0.0 {
            return 0.0;
        }

        // Calculate mean_b once (same for all variations)
        let mut sum_second = f32x8::splat(0.0);
        let chunks_second = second.chunks_exact(LANES);
        let remainder_second = chunks_second.remainder();

        for chunk in chunks_second {
            sum_second += f32x8::from_slice(chunk);
        }

        let mean_second = (sum_second.reduce_sum() + remainder_second.iter().sum::<f32>()) / TOTAL_TILE_COUNT as f32;
        let mean_second_simd_vec = f32x8::splat(mean_second);

        let mut max_corr_coeff = 0.0f32;

        for variation in 0..=2 {
            let second_variant: [f32; TOTAL_TILE_COUNT] = match variation {
                0 => *second,
                1 => Self::mirror_horizontal(second),
                2 => Self::mirror_vertical(second),
                _ => unreachable!(),
            };

            // Calculate variance of second_variant with SIMD
            let mut var_sum_second = f32x8::splat(0.0);
            let chunks_second = second_variant.chunks_exact(LANES);
            let remainder_second = chunks_second.remainder();

            for chunk in chunks_second {
                let vals = f32x8::from_slice(chunk);
                let diff = vals - mean_second_simd_vec;
                var_sum_second += diff * diff;
            }

            let var_second = (var_sum_second.reduce_sum() + remainder_second.iter().map(|v| (v - mean_second).powi(2)).sum::<f32>()) / (TOTAL_TILE_COUNT as f32 - 1.0);

            if var_second == 0.0 {
                continue;
            }

            // Calculate covariance with SIMD
            let mut cov_sum = f32x8::splat(0.0);
            let chunks_first = first.chunks_exact(LANES);
            let chunks_second = second_variant.chunks_exact(LANES);
            let remainder_first = chunks_first.remainder();
            let remainder_second = chunks_second.remainder();

            for (chunk_first, chunk_second) in chunks_first.zip(chunks_second) {
                let vals_first = f32x8::from_slice(chunk_first);
                let vals_second = f32x8::from_slice(chunk_second);
                let diff_first = vals_first - mean_first_simd_vec;
                let diff_second = vals_second - mean_second_simd_vec;
                cov_sum += diff_first * diff_second;
            }

            let cov = (cov_sum.reduce_sum() + remainder_first.iter().zip(remainder_second).map(|(a, b)| (a - mean_first) * (b - mean_second)).sum::<f32>()) / (TOTAL_TILE_COUNT as f32 - 1.0);

            let corr_coeff = (cov / (var_first * var_second).sqrt()).abs();
            max_corr_coeff = max_corr_coeff.max(corr_coeff);
        }

        max_corr_coeff
    }

    /// Creates a slice of f32 from a Vec of u8 for processing the correlation data.
    #[inline]
    fn vec_u8_to_slice_f32(input: &[u8]) -> [f32; TOTAL_TILE_COUNT] {
        input.iter().map(|v| f32::from(*v)).collect::<Vec<f32>>().try_into().unwrap()
    }

    /// # Panics
    /// Panics if the given folder does not exist.
    /// Analysis method that processes combinations in chunks to handle large datasets
    pub fn analyse_with_progress_chunked(folder_path: &str, analysing: &Arc<AtomicBool>, progress_tracker: &Arc<LockFreeProgress>, worker_count: u16, max_combinations_per_chunk: usize) {
        info!("Using {worker_count} workers.\nAnalysing database in \"{folder_path}\" with chunked processing.");
        info!("Processing {} combinations per chunk", max_combinations_per_chunk);

        Self::create_correlation_db(folder_path).expect("Could not create correlation database");
        let db_path = get_db_path::<Self>(folder_path);

        let data_entries = DataEntry::read_from_folder(folder_path).expect("Could not read entries from folder");
        info!("Found {} entries for analysis", data_entries.len());

        if data_entries.len() < 2 {
            analysing.store(false, Ordering::Relaxed);
            return;
        }

        // Load existing correlations
        let already_performed = Self::read_correlation_db(folder_path).unwrap_or_else(|e| {
            error!("Error reading existing correlations: {e}");
            Vec::<Self>::new()
        });

        let existing_combinations: std::collections::HashSet<(String, String)> = already_performed.iter().map(|entry| (entry.first_path.clone(), entry.second_path.clone())).collect();

        info!("Found {} existing correlations", existing_combinations.len());

        // Calculate total possible combinations for progress tracking
        let total_possible = (data_entries.len() * (data_entries.len() - 1)) / 2;
        info!("Total possible combinations: {}", total_possible);

        // Process in chunks to avoid memory explosion
        let mut chunk_start_i = 0;
        let mut chunk_start_j = 1;
        let mut total_processed = 0;
        let mut chunk_number = 0;

        loop {
            if !analysing.load(Ordering::Relaxed) {
                info!("Analysis cancelled by user");
                break;
            }

            chunk_number += 1;
            let mut combinations_to_process = Vec::new();
            let mut combinations_checked = 0;

            info!("Starting chunk {} (from position {}, {})", chunk_number, chunk_start_i, chunk_start_j);

            // Generate one chunk of combinations
            'outer: for i in chunk_start_i..data_entries.len() {
                let start_j = if i == chunk_start_i { chunk_start_j } else { i + 1 };

                for j in start_j..data_entries.len() {
                    combinations_checked += 1;

                    if combinations_checked > max_combinations_per_chunk {
                        // Save where we left off for next chunk
                        chunk_start_i = i;
                        chunk_start_j = j;
                        break 'outer;
                    }

                    let first = &data_entries[i].filename;
                    let second = &data_entries[j].filename;

                    if !existing_combinations.contains(&(first.clone(), second.clone())) {
                        combinations_to_process.push((i, j));
                    }
                }

                // If we finished this i completely, reset j for next i
                if combinations_checked <= max_combinations_per_chunk {
                    chunk_start_j = 0; // Will be set to i+1 in next iteration
                }
            }

            // Check if we're completely done
            if combinations_checked == 0 {
                info!("Finished processing all combinations");
                break;
            }

            let chunk_size = combinations_to_process.len();
            info!("Chunk {}: checked {} combinations, processing {} new ones", chunk_number, combinations_checked, chunk_size);

            if chunk_size > 0 {
                // Reset progress tracker for this chunk
                progress_tracker.reset(chunk_size);

                // Process this chunk
                Self::process_combination_chunk(&data_entries, combinations_to_process, &db_path, analysing, progress_tracker, worker_count);
            }

            total_processed += combinations_checked;
            let progress_percent = (total_processed as f64 / total_possible as f64) * 100.0;
            info!("Overall progress: {}/{} ({:.2}%)", total_processed, total_possible, progress_percent);

            // Check if we've reached the end
            if chunk_start_i >= data_entries.len() - 1 {
                info!("Reached end of all combinations");
                break;
            }
        }

        analysing.store(false, Ordering::Relaxed);
        info!("Analysis complete! Total combinations processed: {} out of {} possible", total_processed, total_possible);
    }

    fn process_combination_chunk(
        data_entries: &[DataEntry],
        combinations_to_process: Vec<(usize, usize)>,
        db_path: &str,
        analysing: &Arc<AtomicBool>,
        progress_tracker: &Arc<LockFreeProgress>,
        worker_count: u16,
    ) {
        if combinations_to_process.is_empty() {
            return;
        }

        let combinations_processed = Arc::new(AtomicUsize::new(0));
        let work_queue = Arc::new(Mutex::new(combinations_to_process));

        let target_db_file_handle = OpenOptions::new().append(true).open(db_path).expect("Cannot open file");
        let target_db_file_mutex = Arc::new(Mutex::new(target_db_file_handle));

        let start_time = Instant::now();
        crossbeam::scope(|scope| {
            for _worker in 0..worker_count {
                let work_queue = Arc::clone(&work_queue);
                let files_processed = Arc::clone(&combinations_processed);
                let worker_db_file = Arc::clone(&target_db_file_mutex);
                let progress_tracker = Arc::clone(progress_tracker);

                scope.spawn(move |_| {
                    'combinations_loop: while let Some((i, j)) = {
                        let mut work_item = work_queue.lock();
                        work_item.pop()
                    } {
                        if !analysing.load(Ordering::Relaxed) {
                            break 'combinations_loop;
                        }

                        let first_entry = &data_entries[i];
                        let second_entry = &data_entries[j];

                        if first_entry.data.len() != TOTAL_TILE_COUNT || second_entry.data.len() != TOTAL_TILE_COUNT {
                            continue;
                        }

                        // Convert to f32 arrays
                        let first_data: [f32; TOTAL_TILE_COUNT] = Self::vec_u8_to_slice_f32(&first_entry.data);
                        let second_data: [f32; TOTAL_TILE_COUNT] = Self::vec_u8_to_slice_f32(&second_entry.data);
                        let corr: f32 = Self::correlation_coefficient(&first_data, &second_data);

                        let entry = Self::new(&first_entry.filename, &second_entry.filename, corr);

                        // always safe
                        if entry.corr > 0.0
                        /*CORRELATION_THRESHOLD*/
                        {
                            let worker_db_file = &worker_db_file.lock();
                            let _ = entry.save(worker_db_file);
                        }

                        // Update progress
                        let processed_count = files_processed.fetch_add(1, Ordering::Relaxed) + 1;
                        progress_tracker.increment();

                        // Log progress every 100_000 combinations
                        if processed_count.is_multiple_of(1_000_000) {
                            let (current_processed, total, progress_pct) = progress_tracker.get_progress();
                            if let Some(remaining_time) = progress_tracker.estimated_remaining() {
                                info!("Chunk progress: {}/{} ({:.1}%), estimated remaining: {:.2?}", current_processed, total, progress_pct * 100.0, remaining_time);
                            }
                        }
                    }
                });
            }
        })
        .unwrap();

        let elapsed = start_time.elapsed();
        info!("Chunk processing took: {:.2?}", elapsed);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use {crate::data, pretty_assertions::assert_eq};

    #[test]
    fn test_corr() {
        let one: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let two: [f32; 16] = [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 12.0, 11.0, 10.0, 9.0, 16.0, 15.0, 14.0, 13.0];
        let result = data::CorrelationEntry::correlation_coefficient(&one, &two);
        assert_eq!(1.0f32, result);

        let one: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let two: [f32; 16] = [13.0, 14.0, 15.0, 16.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0];
        let result = data::CorrelationEntry::correlation_coefficient(&one, &two);
        assert_eq!(1.0f32, result);
    }
    #[test]
    fn test_corr_non_simd() {
        let one: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let two: [f32; 16] = [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 12.0, 11.0, 10.0, 9.0, 16.0, 15.0, 14.0, 13.0];
        let result = data::CorrelationEntry::correlation_coefficient_default(&one, &two);
        assert_eq!(1.0f32, result);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_corr_simd() {
        let one: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let two: [f32; 16] = [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 12.0, 11.0, 10.0, 9.0, 16.0, 15.0, 14.0, 13.0];
        let result = data::CorrelationEntry::correlation_coefficient_simd(&one, &two);
        assert_eq!(1.0f32, result);
    }
}
