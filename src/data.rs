//! # Data Module
//!
//! This module contains all the core data structures, file I/O operations, and algorithms
//! for the image duplicates application.
//!
//! ## Key Components
//!
//! - **Data Structures**: `DataEntry`, `CorrelationEntry`, `ErrorEntry`, `FileMoveAction`, `AppState`
//! - **File I/O**: Database reading/writing, scanning, analysis data persistence
//! - **Algorithms**: Image scanning, correlation calculation (with SIMD optimization)
//! - **State Management**: Application state for folder processing
//!
//! ## Usage
//!
//! The module is organized around processing image files to find duplicates:
//!
//! 1. **Scanning**: `DataEntry::scan_with_progress` processes images into tile data
//! 2. **Analysis**: `CorrelationEntry::analyse_with_progress` compares images for similarities
//! 3. **Persistence**: Data is stored in JSON files alongside the image folders

#[cfg(feature = "simd")]
use std::simd::{Swizzle, f32x16, num::SimdFloat};

use {
    crate::{
        constants::{ANALYSING_DATA_FILE_NAME, ERROR_DATA_FILE_NAME, SCANNING_DATA_FILE_NAME, SUPPORTED_IMAGE_FILE_EXTENSION},
        error::Error as ImageDupeError,
        progress::LockFreeProgress,
        util,
    },
    eframe::egui::mutex::{Mutex, MutexGuard},
    image::GenericImageView,
    rayon::{iter::ParallelIterator, prelude::*},
    serde::{Deserialize, Serialize},
    std::{
        fs,
        fs::{File, OpenOptions},
        io::{self, BufRead, Write},
        iter::zip,
        path::{Path, PathBuf},
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::Instant,
    },
    tracing::{debug, error, info},
};

/// Will split the image into 4x4 tiles
const TILE_COUNT_PER_SIDE: u32 = 4;
const TOTAL_TILE_COUNT: usize = (TILE_COUNT_PER_SIDE * TILE_COUNT_PER_SIDE) as usize;

/// Generates the lookup table for horizontal mirroring, for the set number of tiles.
const fn generate_h_mirror() -> [usize; TOTAL_TILE_COUNT] {
    let tiles_per_side = TILE_COUNT_PER_SIDE as usize;
    let mut arr = [0; TOTAL_TILE_COUNT];
    let mut i = 0;
    while i < TOTAL_TILE_COUNT {
        let row = i / tiles_per_side;
        let col = i % tiles_per_side;
        arr[i] = row * tiles_per_side + (tiles_per_side - 1 - col);
        i += 1;
    }
    arr
}

/// Generates the lookup table for vertical mirroring, for the set number of tiles.
const fn generate_v_mirror() -> [usize; TOTAL_TILE_COUNT] {
    let tiles_per_side = TILE_COUNT_PER_SIDE as usize;
    let mut arr = [0; TOTAL_TILE_COUNT];
    let mut i = 0;
    while i < TOTAL_TILE_COUNT {
        let row = i / tiles_per_side;
        let col = i % tiles_per_side;
        arr[i] = (tiles_per_side - 1 - row) * tiles_per_side + col;
        i += 1;
    }
    arr
}

/// Contains the horizontal mirroring indices.
const H_MIRROR: [usize; TOTAL_TILE_COUNT] = generate_h_mirror();
/// Contains the horizontal mirroring indices.
const V_MIRROR: [usize; TOTAL_TILE_COUNT] = generate_v_mirror();
/// Contains the identity tile order, for the unmirrored correlation variant.
#[cfg(feature = "simd")]
const TILE_IDENTITY: [usize; TOTAL_TILE_COUNT] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// Number of entries the batched analysis kernel processes at once, one per
/// SIMD lane.
#[cfg(feature = "simd")]
pub const GROUP_SIZE: usize = 16;

/// Mirrors a vector of tiles horizontally.
#[cfg(feature = "simd")]
struct HorizontalMirror;

#[cfg(feature = "simd")]
impl Swizzle<16> for HorizontalMirror {
    const INDEX: [usize; 16] = H_MIRROR;
}

/// Mirrors a vector of tiles vertically.
#[cfg(feature = "simd")]
struct VerticalMirror;

#[cfg(feature = "simd")]
impl Swizzle<16> for VerticalMirror {
    const INDEX: [usize; 16] = V_MIRROR;
}
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
    Path::new(folder_path).join(T::db_path_name()).display().to_string()
}

/// Converts an image path into the form stored in a database located in
/// `db_folder`: the path relative to that folder, using `/` separators, so
/// the database stays valid when the folder is moved. Paths outside the
/// folder (e.g. comparison images or secondary scan roots) are kept absolute.
#[must_use]
fn to_db_path(db_folder: &str, file_path: &str) -> String {
    Path::new(file_path).strip_prefix(db_folder).map_or_else(|_| file_path.to_string(), |relative| relative.iter().map(|component| component.to_string_lossy()).collect::<Vec<_>>().join("/"))
}

/// Resolves a path stored in a database located in `db_folder` back into an
/// absolute path, normalizing the `/` separators used in the database to the
/// platform's native ones. Absolute entries, as written by older versions of
/// this tool, are returned unchanged.
#[must_use]
fn from_db_path(db_folder: &str, stored_path: &str) -> String {
    if Path::new(stored_path).is_absolute() { stored_path.to_string() } else { Path::new(db_folder).join(stored_path).iter().collect::<PathBuf>().display().to_string() }
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
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone)]
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
    pub(crate) fn generate_file_list(folder_path: &str, recursive: bool, exclude: Option<&[String]>) -> Result<Vec<String>, ImageDupeError> {
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
        if file_list.is_empty() {
            return Err(ImageDupeError::EmptyFileList(folder_path.to_string()));
        }
        file_list.sort();
        Ok(file_list)
    }

    /// # Errors
    /// Will return an error if the image file can not be opened or read correctly,
    /// or if the image is too small to be divided into tiles.
    ///
    /// Will open the file at the given path and then return the data vector with the average tile brightness for each of the tiles.
    /// This optimized version processes the entire image at once instead of creating separate tile views.
    #[inline]
    pub(crate) fn calculate_image_tile_data(file_path: &str) -> Result<Vec<u8>, ImageDupeError> {
        let mut tiles_averages = Vec::<u8>::with_capacity(TOTAL_TILE_COUNT);

        let opened_image = match image::ImageReader::open(file_path) {
            Ok(img) => img,
            Err(e) => {
                error!("Could not open image {file_path}, error: {e:?}");
                return Err(ImageDupeError::Io(e));
            }
        };

        let decoded_image = match opened_image.decode() {
            Ok(img) => img,
            Err(e) => {
                error!("Could not decode image {file_path}, error: {e:?}");
                return Err(ImageDupeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
            }
        };

        let (image_width, image_height) = decoded_image.dimensions();
        let tile_width = image_width / TILE_COUNT_PER_SIDE;
        let tile_height = image_height / TILE_COUNT_PER_SIDE;

        if tile_width == 0 || tile_height == 0 {
            error!("Image or tile too small to process. File {file_path}");
            return Err(ImageDupeError::TileCalculation(file_path.to_owned()));
        }

        // Convert to Luma8 (grayscale) for efficient brightness extraction
        // This uses the standard luma formula: 0.299*R + 0.587*G + 0.114*B
        // which is perceptually more accurate than (R+G+B)/3
        let luma_image = decoded_image.to_luma8();
        let image_data = luma_image.into_raw();

        // Pre-calculate tile boundaries and process all tiles in one pass
        for row in 0..TILE_COUNT_PER_SIDE {
            for col in 0..TILE_COUNT_PER_SIDE {
                let x_start = col * tile_width;
                let y_start = row * tile_height;
                let x_end = x_start + tile_width;
                let y_end = y_start + tile_height;

                let mut tile_pixel_sum: u64 = 0;

                // Process all pixels in this tile
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let pixel_idx = ((y * image_width) + x) as usize;
                        // Luma8 has 1 channel (brightness) per pixel
                        tile_pixel_sum += u64::from(image_data[pixel_idx]);
                    }
                }

                // Calculate average: sum / pixel_count (1 channel)
                let pixel_count = u64::from(tile_width) * u64::from(tile_height);
                let tile_average = u8::try_from(tile_pixel_sum / pixel_count).unwrap_or(0);
                tiles_averages.push(tile_average);
            }
        }

        Ok(tiles_averages)
    }

    /// Reads the entries from a given folder, determining the correct name automatically.
    ///
    /// Corrupt lines are logged and skipped. Stored paths are resolved into
    /// absolute paths relative to the folder.
    ///
    /// # Errors
    /// Will return an error if the database file can not be opened or read.
    ///
    /// Returning a Result vec of all entries.
    pub fn read_from_folder(folder_path: &str) -> Result<Vec<Self>, ImageDupeError> {
        let target_file = get_db_path::<Self>(folder_path);
        let file_handle = File::open(&target_file)?;
        Ok(io::BufReader::new(file_handle)
            .lines()
            .map_while(Result::ok)
            .filter(|line| !line.is_empty())
            .filter_map(|line| match serde_json::from_str::<Self>(&line) {
                Ok(entry) => Some(Self { filename: from_db_path(folder_path, &entry.filename), data: entry.data }),
                Err(e) => {
                    error!("Skipping corrupt entry in {target_file}: {e}");
                    None
                }
            })
            .collect())
    }

    /// Creates the database file in the given folder path to store the data.
    pub(crate) fn create_db(folder_path: &str) -> Result<(), ImageDupeError> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }

    /// Ensures the file has an entry in the scan database of `folder_path`:
    /// if it is missing, its tile data is computed and appended to the
    /// database. Returns whether an entry was added.
    ///
    /// # Errors
    /// Returns an error if the image cannot be opened, decoded or divided
    /// into tiles, or if the entry cannot be written to the database.
    pub fn ensure_in_db(folder_path: &str, file_path: &str) -> Result<bool, ImageDupeError> {
        let already_known = Self::read_from_folder(folder_path).is_ok_and(|entries| entries.iter().any(|entry| entry.filename == file_path));
        if already_known {
            return Ok(false);
        }
        let averages = Self::calculate_image_tile_data(file_path)?;
        let entry = Self::new(to_db_path(folder_path, file_path), &averages);
        Self::create_db(folder_path)?;
        let db_path = get_db_path::<Self>(folder_path);
        let file_handle = OpenOptions::new().create(true).append(true).open(&db_path)?;
        let file_mutex = Mutex::new(file_handle);
        let save_result = {
            let lock = file_mutex.lock();
            entry.save(&lock)
        };
        if let Err(e) = save_result {
            return Err(ImageDupeError::Io(io::Error::other(e.to_string())));
        }
        info!("Added comparison image {file_path} to the scan database");
        Ok(true)
    }

    /// # Panics
    /// When the given folder can not be read.
    #[allow(clippy::too_many_lines)]
    pub fn scan_with_progress(folder_path: &str, files_to_scan_list: &[String], scanning: &Arc<AtomicBool>, progress_tracker: &Arc<LockFreeProgress>) {
        let db_path = get_db_path::<Self>(folder_path);
        if let Err(e) = Self::create_db(folder_path) {
            error!("Could not create database for {folder_path}: {e}");
            scanning.store(false, Ordering::Relaxed);
            return;
        }

        info!("Found {} potential files for scanning.", files_to_scan_list.len());

        let existing_entries = Self::read_from_folder(folder_path).unwrap_or_else(|e| {
            error!("Error while reading folder: {e:?}");
            Vec::<_>::new()
        });
        let existing_filenames: std::collections::HashSet<String> = existing_entries.into_iter().map(|entry| entry.filename).collect();

        info!("Found {} entries already in database.", existing_filenames.len());

        // filter out files that already exist in the database
        let files_to_scan: Vec<String> =
            if existing_filenames.is_empty() { files_to_scan_list.to_vec() } else { files_to_scan_list.iter().filter(|filename| !existing_filenames.contains(*filename)).cloned().collect() };
        if files_to_scan.is_empty() {
            info!("Nothing to do, all files have already been scanned!");
            scanning.store(false, Ordering::Relaxed);
            return;
        }

        info!("Scanning {} files, skipping already scanned files.", files_to_scan.len());

        progress_tracker.reset(files_to_scan.len());

        let (result_tx, result_rx) = std::sync::mpsc::channel::<Self>();
        let files_processed = Arc::new(AtomicUsize::new(0));

        let write_handle = {
            let files_processed = Arc::clone(&files_processed);
            std::thread::spawn(move || {
                use std::io::BufWriter;

                let file = match OpenOptions::new().create(true).append(true).open(&db_path) {
                    Ok(file) => file,
                    Err(e) => {
                        error!("Cannot open database file {db_path}: {e}");
                        return;
                    }
                };

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
                match Self::calculate_image_tile_data(&file_path) {
                    Ok(averages) => {
                        let entry = Self::new(to_db_path(folder_path, &file_path), &averages);
                        // Send to writer thread (non-blocking)
                        if result_tx_clone.send(entry).is_err() {
                            error!("Failed to send entry to writer thread");
                        }
                    }
                    Err(e) => {
                        // Get the directory containing the problematic file for the error database
                        let file_dir = Path::new(&file_path).parent().and_then(|p| p.to_str()).unwrap_or(folder_path);
                        let error_db_path = get_db_path::<ErrorEntry>(file_dir);

                        // Try to create error database if it doesn't exist
                        if let Err(db_err) = ErrorEntry::create_error_db(file_dir) {
                            error!("Could not create error database for {file_dir}: {db_err}");
                        }

                        // Try to save the error - if this fails, just log it and continue
                        match OpenOptions::new().append(true).open(&error_db_path) {
                            Ok(error_db_file_handle) => {
                                let error_db_file_mutex = Arc::new(Mutex::new(error_db_file_handle));
                                let error_entry = ErrorEntry::new(&to_db_path(file_dir, &file_path), &e.to_string());
                                let lock = error_db_file_mutex.lock();
                                if let Err(save_err) = error_entry.save(&lock) {
                                    error!("Could not save error for {file_path}: {save_err}");
                                }
                                // lock is dropped here
                            }
                            Err(open_err) => {
                                error!("Could not open error database for {file_dir}: {open_err}");
                            }
                        }

                        error!("Could not process image {file_path}, error: {e:?}");
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
                    debug!("Processed {current_processed}/{total} files ({:.1}%), estimated remaining: {remaining_time:.2?}", progress_pct * 100.0);
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
    pub fn scan_folders_with_progress(folders: &[String], exclude: Option<&[String]>, scanning: &Arc<AtomicBool>, recursive: &[bool], progress_tracker: &Arc<LockFreeProgress>) {
        let mut files_to_scan_list = Vec::<String>::new();
        for (folder, recursive) in folders.iter().zip(recursive.iter()) {
            match Self::generate_file_list(folder, *recursive, exclude) {
                Ok(mut file_list_for_folder) => files_to_scan_list.append(&mut file_list_for_folder),
                Err(e) => {
                    error!("Failed to generate file list for {folder}: {e}");
                }
            }
        }

        let Some(primary_folder) = folders.first() else {
            error!("At least one folder should be provided for scanning.");
            return;
        };

        Self::scan_with_progress(primary_folder, &files_to_scan_list, scanning, progress_tracker);
    }
}

/// Centred tile data and the reciprocal standard deviation of one entry,
/// precomputed once per analysis so the pair loop only has to multiply.
///
/// Malformed entries (wrong tile count) have all-zero statistics, so their
/// correlations become zero and are never saved.
#[cfg(feature = "simd")]
pub struct EntryStats {
    /// Tile data with the mean subtracted.
    centred: [f32; TOTAL_TILE_COUNT],
    /// `1/sqrt(variance)`; zero for zero-variance or malformed entries, so
    /// their correlations become zero instead of dividing by zero.
    inv_sqrt_var: f32,
}

#[cfg(feature = "simd")]
impl EntryStats {
    /// Computes the statistics of one entry's tile data.
    #[must_use]
    pub fn new(data: &[u8]) -> Self {
        if data.len() != TOTAL_TILE_COUNT {
            return Self { centred: [0.0; TOTAL_TILE_COUNT], inv_sqrt_var: 0.0 };
        }
        let len = TOTAL_TILE_COUNT as f32;
        let tiles: [f32; TOTAL_TILE_COUNT] = data.iter().map(|v| f32::from(*v)).collect::<Vec<f32>>().try_into().unwrap_or([0.0; TOTAL_TILE_COUNT]);
        let mean = tiles.iter().sum::<f32>() / len;
        let mut centred = [0.0; TOTAL_TILE_COUNT];
        for (centred_tile, tile) in centred.iter_mut().zip(tiles.iter()) {
            *centred_tile = tile - mean;
        }
        let var = centred.iter().map(|v| v * v).sum::<f32>() / (len - 1.0);
        let inv_sqrt_var = if var == 0.0 { 0.0 } else { 1.0 / var.sqrt() };
        Self { centred, inv_sqrt_var }
    }
}

/// Stores [`GROUP_SIZE`] entries in tile-major ("structure of arrays") layout,
/// so each SIMD lane holds one entry's tile and a single vector operation
/// processes the same tile of all group entries at once.
#[cfg(feature = "simd")]
pub struct EntryGroup {
    /// `centred_tiles[t]` holds centred tile `t` of every group entry, one per lane.
    centred_tiles: [f32x16; TOTAL_TILE_COUNT],
    /// `1/sqrt(variance)` of every group entry; zero for padding lanes.
    inv_sqrt_var: f32x16,
}

#[cfg(feature = "simd")]
impl EntryGroup {
    /// Builds a group from up to [`GROUP_SIZE`] statistics, padding missing
    /// entries with zero variance so their correlations are zero.
    #[must_use]
    pub fn new(stats: &[EntryStats]) -> Self {
        let mut centred_tiles = [f32x16::splat(0.0); TOTAL_TILE_COUNT];
        for (tile_idx, column) in centred_tiles.iter_mut().enumerate() {
            let mut lane_values = [0.0; GROUP_SIZE];
            for (lane, stat) in lane_values.iter_mut().zip(stats.iter().take(GROUP_SIZE)) {
                *lane = stat.centred[tile_idx];
            }
            *column = f32x16::from_array(lane_values);
        }
        let mut inv_sqrt_var = [0.0; GROUP_SIZE];
        for (lane, stat) in inv_sqrt_var.iter_mut().zip(stats.iter().take(GROUP_SIZE)) {
            *lane = stat.inv_sqrt_var;
        }
        Self { centred_tiles, inv_sqrt_var: f32x16::from_array(inv_sqrt_var) }
    }
}

/// Everything the analysis workers need about the analysed dataset, shared
/// read-only between the workers of one analysis run.
#[cfg(feature = "simd")]
struct AnalysisSet<'a> {
    /// Precomputed statistics of every entry, indexed like `entries`.
    stats: &'a [EntryStats],
    /// Tile-major groups of the entries, see [`EntryGroup`].
    groups: &'a [EntryGroup],
    /// The database entries; they provide the file names of the saved
    /// correlations.
    entries: &'a [DataEntry],
    /// Already analysed pairs, so they are skipped and not saved twice.
    done: &'a std::collections::HashMap<usize, Vec<usize>>,
    /// Folder holding the databases; saved paths are stored relative to it.
    folder_path: &'a str,
}

/// Returns the first group that can contain partners greater than `first`.
#[cfg(feature = "simd")]
const fn first_group_of(first: usize) -> usize {
    if first % GROUP_SIZE == GROUP_SIZE - 1 { first / GROUP_SIZE + 1 } else { first / GROUP_SIZE }
}

/// Returns whether every pair of `first` with the entries of group `group`
/// is already stored in `done`.
#[cfg(feature = "simd")]
fn group_complete(done: &std::collections::HashMap<usize, Vec<usize>>, first: usize, group: usize, entry_count: usize) -> bool {
    let group_start = group * GROUP_SIZE;
    let group_end = (group_start + GROUP_SIZE).min(entry_count);
    let start = group_start.max(first + 1);
    if start >= group_end {
        // The group contains no partners greater than `first`.
        return true;
    }
    done.get(&first).is_some_and(|list| list.partition_point(|&j| j < group_end) - list.partition_point(|&j| j < start) == group_end - start)
}

/// Returns whether the pair (`first`, `second`) is already stored in `done`.
#[cfg(feature = "simd")]
fn pair_done(done: &std::collections::HashMap<usize, Vec<usize>>, first: usize, second: usize) -> bool {
    let (low, high) = if first < second { (first, second) } else { (second, first) };
    done.get(&low).is_some_and(|list| list.binary_search(&high).is_ok())
}

/// Maps already analysed pairs to sorted per-entry lookup lists, so work
/// items that are fully analysed can be skipped and already saved
/// correlations are not duplicated.
#[cfg(feature = "simd")]
fn build_done_map(already_performed: &[CorrelationEntry], filename_to_index: &std::collections::HashMap<&str, usize>) -> std::collections::HashMap<usize, Vec<usize>> {
    let mut done: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for entry in already_performed {
        if let (Some(&first), Some(&second)) = (filename_to_index.get(entry.first_path.as_str()), filename_to_index.get(entry.second_path.as_str())) {
            done.entry(first.min(second)).or_default().push(first.max(second));
        }
    }
    for list in done.values_mut() {
        list.sort_unstable();
        list.dedup();
    }
    done
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

    pub(crate) fn create_correlation_db(folder_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }

    pub(crate) fn read_correlation_db(folder_path: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let file_name = get_db_path::<Self>(folder_path);
        let file_handle = File::open(file_name)?;
        let mut result: Vec<Self> = io::BufReader::new(file_handle)
            .lines()
            .map(|line| line.and_then(|line_str| serde_json::from_str(&line_str).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))))
            .collect::<Result<Vec<Self>, _>>()?;
        for entry in &mut result {
            entry.first_path = from_db_path(folder_path, &entry.first_path);
            entry.second_path = from_db_path(folder_path, &entry.second_path);
        }
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

    #[must_use]
    pub fn correlation_coefficient_default(a: &[f32; TOTAL_TILE_COUNT], b: &[f32; TOTAL_TILE_COUNT]) -> f32 {
        // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        // Optimized version with reduced redundant calculations

        let len = TOTAL_TILE_COUNT as f32;

        // Calculate means
        let sum_a: f32 = a.iter().sum();
        let sum_b: f32 = b.iter().sum();
        let mean_a = sum_a / len;
        let mean_b = sum_b / len;

        // Pre-calculate all b variants to avoid recalculating in each loop
        let b_variants = [
            b,                           // original
            &Self::mirror_horizontal(b), // horizontal mirror
            &Self::mirror_vertical(b),   // vertical mirror
        ];

        // Calculate variance of a once (same for all b variants)
        let var_a: f32 = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / (len - 1.0);
        if var_a == 0.0 {
            return 0.0;
        }

        let mut max_corr_coeff: f32 = 0.0;

        for b_variant in b_variants {
            // Calculate variance of current b variant
            let var_b: f32 = b_variant.iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / (len - 1.0);
            if var_b == 0.0 {
                continue;
            }

            // Calculate covariance
            let cov = zip(a, b_variant).map(|(a_val, b_val)| (a_val - mean_a) * (b_val - mean_b)).sum::<f32>() / (len - 1.0);

            // Calculate correlation coefficient (absolute value)
            let corr_coeff = (cov / (var_a * var_b).sqrt()).abs();
            max_corr_coeff = max_corr_coeff.max(corr_coeff);
        }

        max_corr_coeff
    }

    /// SIMD implementation of [`CorrelationEntry::correlation_coefficient`].
    ///
    /// The whole 4x4 tile set fits into a single `f32x16` vector, so the mean,
    /// variance and covariance of all three mirroring variants are computed
    /// without any chunk or remainder handling. The mirrors become single
    /// register permutes.
    #[cfg(feature = "simd")]
    #[must_use]
    pub fn correlation_coefficient_simd(first: &[f32; TOTAL_TILE_COUNT], second: &[f32; TOTAL_TILE_COUNT]) -> f32 {
        // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        let len = TOTAL_TILE_COUNT as f32;
        let first = f32x16::from_array(*first);
        let second = f32x16::from_array(*second);

        let mean_first = first.reduce_sum() / len;
        let centred_first = first - f32x16::splat(mean_first);
        let var_first = (centred_first * centred_first).reduce_sum() / (len - 1.0);
        if var_first == 0.0 {
            return 0.0;
        }

        // Mirroring only permutes tiles, so every variant shares the mean and
        // variance of the original; only the covariance differs.
        let mean_second = second.reduce_sum() / len;
        let centred_second = second - f32x16::splat(mean_second);
        let var_second = (centred_second * centred_second).reduce_sum() / (len - 1.0);
        if var_second == 0.0 {
            return 0.0;
        }

        let original = (centred_first * centred_second).reduce_sum();
        let horizontal = (centred_first * HorizontalMirror::swizzle(centred_second)).reduce_sum();
        let vertical = (centred_first * VerticalMirror::swizzle(centred_second)).reduce_sum();
        let denominator = (var_first * var_second).sqrt() * (len - 1.0);

        original.abs().max(horizontal.abs()).max(vertical.abs()) / denominator
    }

    /// Correlation of `first` against every entry of `group`, one per lane,
    /// covering all mirroring variants.
    ///
    /// Mirroring only permutes tiles, so the variants are computed by reading
    /// the group tiles in permuted order, which needs no permute instructions.
    /// Lanes of malformed or zero-variance entries, and padding lanes, contain
    /// zero.
    #[cfg(feature = "simd")]
    #[must_use]
    pub fn correlation_group(first: &EntryStats, group: &EntryGroup) -> f32x16 {
        let mut best = f32x16::splat(0.0);
        for mirror in [TILE_IDENTITY, H_MIRROR, V_MIRROR] {
            let mut cov = f32x16::splat(0.0);
            for (tile_idx, first_tile) in first.centred.iter().enumerate() {
                cov += f32x16::splat(*first_tile) * group.centred_tiles[mirror[tile_idx]];
            }
            best = best.simd_max(cov.abs());
        }
        best * f32x16::splat(first.inv_sqrt_var / (TOTAL_TILE_COUNT as f32 - 1.0)) * group.inv_sqrt_var
    }

    /// Creates a slice of f32 from a Vec of u8 for processing the correlation data.
    #[inline]
    #[cfg(not(feature = "simd"))]
    fn vec_u8_to_slice_f32(input: &[u8]) -> [f32; TOTAL_TILE_COUNT] {
        let mut tiles = [0.0; TOTAL_TILE_COUNT];
        for (tile, value) in tiles.iter_mut().zip(input.iter()) {
            *tile = f32::from(*value);
        }
        tiles
    }

    /// # Panics
    /// Panics if the given folder does not exist.
    /// Analysis method that correlates every entry pair of the scan database,
    /// feeding the workers from a continuous work queue. Already analysed
    /// pairs are skipped.
    ///
    /// If `preloaded_data` is provided, it will be used directly instead of loading from disk.
    /// This allows the UI to pre-load data and avoid the startup delay.
    // TODO: Split into a preparation part (entry statistics, groups, done map,
    // progress totals) and a run part (work queue and workers), so neither
    // part needs the `too_many_lines` exception.
    #[allow(clippy::too_many_lines)]
    pub fn analyse_with_progress(folder_path: &str, analysing: &Arc<AtomicBool>, progress_tracker: &Arc<LockFreeProgress>, preloaded_data: Option<Vec<DataEntry>>) {
        let worker_count = u16::try_from(std::thread::available_parallelism().map_or(1, usize::from)).unwrap_or(1);
        info!("Using {worker_count} workers.\nAnalysing database in \"{folder_path}\".");

        if let Err(e) = Self::create_correlation_db(folder_path) {
            error!("Could not create correlation database for {folder_path}: {e}");
            analysing.store(false, Ordering::Relaxed);
            return;
        }

        // Use preloaded data if provided, otherwise load from disk
        let data_entries = match preloaded_data {
            Some(entries) => entries,
            None => match DataEntry::read_from_folder(folder_path) {
                Ok(entries) => entries,
                Err(e) => {
                    error!("Could not read entries from folder {folder_path}: {e}");
                    analysing.store(false, Ordering::Relaxed);
                    return;
                }
            },
        };
        let entry_count = data_entries.len();
        info!("Found {} entries for analysis", entry_count);

        if entry_count < 2 {
            analysing.store(false, Ordering::Relaxed);
            return;
        }

        // Load existing correlations to avoid reprocessing
        let already_performed = match Self::read_correlation_db(folder_path) {
            Ok(correlations) => correlations,
            Err(e) => {
                error!("Error reading existing correlations from {folder_path}: {e}. Will re-process all combinations.");
                Vec::<Self>::new()
            }
        };

        // Build a HashSet of index pairs for fast lookup without string cloning
        // First, create a mapping from filename to index
        let filename_to_index: std::collections::HashMap<&str, usize> = data_entries.iter().enumerate().map(|(idx, entry)| (entry.filename.as_str(), idx)).collect();

        // Calculate total possible combinations for logging
        let total_possible = (entry_count * (entry_count - 1)) / 2;
        info!("Total possible combinations: {}", total_possible);

        let overall_start_time = Instant::now();

        #[cfg(feature = "simd")]
        {
            // Precompute the per-entry statistics and the tile-major groups, so
            // the workers only multiply: one work item correlates one entry
            // against a whole group of entries, one SIMD lane per entry.
            let stats: Vec<EntryStats> = data_entries.iter().map(|entry| EntryStats::new(&entry.data)).collect();
            let groups: Vec<EntryGroup> = stats.chunks(GROUP_SIZE).map(EntryGroup::new).collect();
            let group_count = groups.len();
            let done = Arc::new(build_done_map(&already_performed, &filename_to_index));
            info!("Found {} existing correlations", already_performed.len());

            // Work items that are fully analysed never enter the work queue,
            // so the progress tracker only covers the remaining ones.
            let total_items: usize = (0..entry_count).map(|first| group_count.saturating_sub(first_group_of(first))).sum();
            let complete_items: usize = (0..entry_count).map(|first| (first_group_of(first)..group_count).filter(|&group| group_complete(&done, first, group, entry_count)).count()).sum();
            progress_tracker.reset(total_items.saturating_sub(complete_items));
            info!("Skipping {} fully analysed work items", complete_items);

            // Use a continuous work queue for smooth parallelism
            let (work_tx, work_rx) = crossbeam::channel::unbounded::<(usize, usize)>();
            let work_tx = Arc::new(work_tx);

            // Generate all work in a separate thread to avoid blocking workers
            let generation_handle = {
                let work_tx = Arc::clone(&work_tx);
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    let mut total_generated = 0usize;
                    for first in 0..entry_count {
                        for group in first_group_of(first)..group_count {
                            if !group_complete(&done, first, group, entry_count) {
                                if work_tx.send((first, group)).is_err() {
                                    return total_generated;
                                }
                                total_generated += 1;
                            }
                        }
                    }
                    total_generated
                })
            };

            // Drop this thread's sender handle so the workers see a channel disconnect
            // once the generation thread has finished sending all work items.
            // Without this, the workers block in recv() forever after draining the
            // queue, because this handle is only dropped after they exit - a deadlock.
            drop(work_tx);

            // Process all work from the channel
            let set = AnalysisSet { stats: &stats, groups: &groups, entries: &data_entries, done: &done, folder_path };
            Self::process_groups(&set, &work_rx, worker_count, analysing, progress_tracker);

            let total_generated = generation_handle.join().unwrap_or(0);

            // Only mark as complete if we finished naturally (not cancelled)
            if analysing.load(Ordering::Relaxed) {
                analysing.store(false, Ordering::Relaxed);
                let total_time = overall_start_time.elapsed();
                info!("Analysis complete! Work items generated: {} out of {} possible. Time taken: {:.2?}", total_generated, total_items, total_time);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            let existing_combinations: std::collections::HashSet<(usize, usize)> = already_performed
                .iter()
                .filter_map(|entry| filename_to_index.get(entry.first_path.as_str()).and_then(|&i| filename_to_index.get(entry.second_path.as_str()).map(|&j| (i, j))))
                .collect();

            info!("Found {} existing correlations", existing_combinations.len());

            // Combinations already in the database never enter the work queue,
            // so the progress tracker only covers the remaining ones. Crediting
            // the skipped combinations towards the full total instead would make
            // an incremental re-run look like a full analysis in the UI.
            let remaining_combinations = total_possible.saturating_sub(existing_combinations.len());
            progress_tracker.reset(remaining_combinations);
            info!("Skipping {} already analysed combinations", existing_combinations.len());

            // Use a continuous work queue for smooth parallelism
            let (work_tx, work_rx) = crossbeam::channel::unbounded();

            // Wrap existing_combinations in Arc<Mutex> for shared ownership
            let existing_combinations = Arc::new(Mutex::new(existing_combinations));
            let work_tx = Arc::new(work_tx);

            // Generate all work in a separate thread to avoid blocking workers
            let generation_handle = {
                let existing_combinations = Arc::clone(&existing_combinations);
                let work_tx = Arc::clone(&work_tx);
                std::thread::spawn(move || {
                    let mut total_generated = 0usize;
                    for i in 0..entry_count {
                        let start_j = i + 1;
                        for j in start_j..entry_count {
                            // Use index-based lookup to avoid string cloning
                            if !existing_combinations.lock().contains(&(i, j)) {
                                if work_tx.send((i, j)).is_err() {
                                    return total_generated;
                                }
                                total_generated += 1;
                            }
                        }
                    }
                    total_generated
                })
            };

            // Drop this thread's sender handle so the workers see a channel disconnect
            // once the generation thread has finished sending all work items.
            // Without this, the workers block in recv() forever after draining the
            // queue, because this handle is only dropped after they exit - a deadlock.
            drop(work_tx);

            // Process all work from the channel
            Self::process_pairs(&data_entries, &work_rx, folder_path, analysing, worker_count, progress_tracker);

            let total_generated = generation_handle.join().unwrap_or(0);

            // Only mark as complete if we finished naturally (not cancelled)
            if analysing.load(Ordering::Relaxed) {
                analysing.store(false, Ordering::Relaxed);
                let total_time = overall_start_time.elapsed();
                info!("Analysis complete! Total combinations generated: {}, processed: {} out of {} possible. Time taken: {:.2?}", total_generated, total_possible, total_possible, total_time);
            }
        }
    }

    /// Processes the group work items from the channel, saving every
    /// correlation above the threshold that is not already stored.
    ///
    /// # Panics
    /// May panic when the given folder can not be accessed.
    #[cfg(feature = "simd")]
    fn process_groups(set: &AnalysisSet<'_>, work_rx: &crossbeam::channel::Receiver<(usize, usize)>, worker_count: u16, analysing: &Arc<AtomicBool>, progress_tracker: &Arc<LockFreeProgress>) {
        let combinations_processed = Arc::new(AtomicUsize::new(0));
        let db_path = get_db_path::<Self>(set.folder_path);
        let target_db_file_handle = match OpenOptions::new().append(true).open(&db_path) {
            Ok(handle) => handle,
            Err(e) => {
                error!("Cannot open correlation database file {db_path}: {e}");
                return;
            }
        };

        // A single writer thread owns the file and buffers the entries, so
        // the workers never serialize on a file lock and each entry costs a
        // channel send instead of a write syscall.
        let (save_tx, save_rx) = crossbeam::channel::unbounded::<Self>();
        let writer_handle = {
            let mut writer = std::io::BufWriter::with_capacity(64 * 1024, target_db_file_handle); // 64KB buffer
            std::thread::spawn(move || {
                let mut saved = 0usize;
                while let Ok(entry) = save_rx.recv() {
                    if let Err(e) = serde_json::to_writer(&mut writer, &entry) {
                        error!("Failed to write correlation entry: {e}");
                    }
                    if let Err(e) = writeln!(&mut writer) {
                        error!("Failed to write newline: {e}");
                    }
                    saved += 1;
                }
                if let Err(e) = writer.flush() {
                    error!("Failed to flush correlation database: {e}");
                }
                saved
            })
        };

        let start_time = Instant::now();

        let scope_result = crossbeam::scope(|scope| {
            for worker_idx in 0..worker_count {
                let save_tx = save_tx.clone();
                let files_processed = Arc::clone(&combinations_processed);
                let analysing = Arc::clone(analysing);
                let progress_tracker = Arc::clone(progress_tracker);
                let work_rx = work_rx.clone();

                scope.spawn(move |_| {
                    let mut worker_local_count = 0usize;
                    while let Ok((first, group)) = work_rx.recv() {
                        if !analysing.load(Ordering::Relaxed) {
                            break;
                        }

                        // Count every dequeued work item towards progress,
                        // including the ones whose lanes are skipped below, so
                        // the tracker matches the number of queued items.
                        progress_tracker.increment();

                        let correlations = Self::correlation_group(&set.stats[first], &set.groups[group]);
                        let group_start = group * GROUP_SIZE;
                        for (lane, corr) in correlations.to_array().iter().enumerate() {
                            let second = group_start + lane;
                            if second <= first || second >= set.entries.len() || *corr <= crate::constants::CORRELATION_THRESHOLD || pair_done(set.done, first, second) {
                                continue;
                            }
                            let entry = Self::new(to_db_path(set.folder_path, &set.entries[first].filename), to_db_path(set.folder_path, &set.entries[second].filename), *corr);
                            if save_tx.send(entry).is_err() {
                                error!("Failed to send correlation entry for {} & {}: writer thread has exited", set.entries[first].filename, set.entries[second].filename);
                            }
                        }

                        worker_local_count += 1;
                        files_processed.fetch_add(1, Ordering::Relaxed);

                        // Log progress every 100_000 work items (per worker)
                        if worker_local_count.is_multiple_of(100_000) {
                            debug!("Worker {} processed {} work items", worker_idx, worker_local_count);
                        }
                    }
                });
            }
        });
        if let Err(e) = scope_result {
            error!("Processing did not finish successfully: {e:?}");
        }

        // Drop this thread's sender so the writer sees a channel disconnect
        // once the workers have finished, flushes and exits.
        drop(save_tx);
        let saved = writer_handle.join().unwrap_or(0);
        if saved > 0 {
            info!("Saved {saved} correlation entries");
        }

        let elapsed = start_time.elapsed();
        info!("Processing took: {:.2?}", elapsed);
    }

    #[cfg(not(feature = "simd"))]
    fn process_pairs(
        data_entries: &[DataEntry],
        work_rx: &crossbeam::channel::Receiver<(usize, usize)>,
        folder_path: &str,
        analysing: &Arc<AtomicBool>,
        worker_count: u16,
        progress_tracker: &Arc<LockFreeProgress>,
    ) {
        let combinations_processed = Arc::new(AtomicUsize::new(0));
        let db_path = get_db_path::<Self>(folder_path);
        let target_db_file_handle = match OpenOptions::new().append(true).open(&db_path) {
            Ok(handle) => handle,
            Err(e) => {
                error!("Cannot open correlation database file {db_path}: {e}");
                return;
            }
        };
        // A single writer thread owns the file and buffers the entries, so
        // the workers never serialize on a file lock and each entry costs a
        // channel send instead of a write syscall.
        let (save_tx, save_rx) = crossbeam::channel::unbounded::<Self>();
        let writer_handle = {
            let mut writer = std::io::BufWriter::with_capacity(64 * 1024, target_db_file_handle); // 64KB buffer
            std::thread::spawn(move || {
                let mut saved = 0usize;
                while let Ok(entry) = save_rx.recv() {
                    if let Err(e) = serde_json::to_writer(&mut writer, &entry) {
                        error!("Failed to write correlation entry: {e}");
                    }
                    if let Err(e) = writeln!(&mut writer) {
                        error!("Failed to write newline: {e}");
                    }
                    saved += 1;
                }
                if let Err(e) = writer.flush() {
                    error!("Failed to flush correlation database: {e}");
                }
                saved
            })
        };

        let start_time = Instant::now();

        let scope_result = crossbeam::scope(|scope| {
            for worker_idx in 0..worker_count {
                let save_tx = save_tx.clone();
                let files_processed = Arc::clone(&combinations_processed);
                let analysing = Arc::clone(analysing);
                let progress_tracker = Arc::clone(progress_tracker);
                let work_rx = work_rx.clone();

                scope.spawn(move |_| {
                    let mut worker_local_count = 0usize;
                    while let Ok((i, j)) = work_rx.recv() {
                        if !analysing.load(Ordering::Relaxed) {
                            break;
                        }

                        // Count every dequeued combination towards progress,
                        // including the ones skipped below, so the tracker
                        // matches the total number of possible combinations.
                        progress_tracker.increment();

                        let first_entry = &data_entries[i];
                        let second_entry = &data_entries[j];

                        if first_entry.data.len() != TOTAL_TILE_COUNT || second_entry.data.len() != TOTAL_TILE_COUNT {
                            continue;
                        }

                        // Convert to f32 arrays
                        let first_data: [f32; TOTAL_TILE_COUNT] = Self::vec_u8_to_slice_f32(&first_entry.data);
                        let second_data: [f32; TOTAL_TILE_COUNT] = Self::vec_u8_to_slice_f32(&second_entry.data);
                        let corr: f32 = Self::correlation_coefficient(&first_data, &second_data);

                        // Only save entries with significant correlation to avoid unnecessary data
                        if corr > crate::constants::CORRELATION_THRESHOLD {
                            let stored_entry = Self::new(to_db_path(folder_path, &first_entry.filename), to_db_path(folder_path, &second_entry.filename), corr);
                            if save_tx.send(stored_entry).is_err() {
                                error!("Failed to send correlation entry for {} & {}: writer thread has exited", first_entry.filename, second_entry.filename);
                            }
                        }

                        worker_local_count += 1;
                        files_processed.fetch_add(1, Ordering::Relaxed);

                        // Log progress every 100_000 combinations (per worker)
                        if worker_local_count.is_multiple_of(100_000) {
                            debug!("Worker {} processed {} combinations", worker_idx, worker_local_count);
                        }
                    }
                });
            }
        });
        if let Err(e) = scope_result {
            error!("Processing did not finish successfully: {e:?}");
        }

        // Drop this thread's sender so the writer sees a channel disconnect
        // once the workers have finished, flushes and exits.
        drop(save_tx);
        let saved = writer_handle.join().unwrap_or(0);
        if saved > 0 {
            info!("Saved {saved} correlation entries");
        }

        let elapsed = start_time.elapsed();
        info!("Processing took: {:.2?}", elapsed);
    }
}

/// Holds the tile data, for a given file.
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ErrorEntry {
    pub(crate) filename: String,
    pub(crate) error: String,
}

impl ErrorEntry {
    #[must_use]
    pub fn new(filename: &str, error: &str) -> Self {
        Self { filename: filename.to_owned(), error: error.to_owned() }
    }

    pub(crate) fn create_error_db(folder_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }
}

/// Tracks a file move/overwrite operation for undo
#[derive(Debug, Clone)]
pub struct FileMoveAction {
    pub source_path: String,
    pub target_path: String,
    pub deleted_path: String,
    pub source_data: Option<DataEntry>,
    pub target_data: Option<DataEntry>,
    pub removed_correlations: Vec<CorrelationEntry>,
}

/// Central in-memory state for a folder
#[derive(Default)]
pub struct AppState {
    pub data_entries: Vec<DataEntry>,
    pub correlation_entries: Vec<CorrelationEntry>,
    pub folder_path: String,
}

impl AppState {
    #[must_use]
    pub fn load(folder_path: &str) -> Self {
        let data_entries = DataEntry::read_from_folder(folder_path).unwrap_or_default();
        let correlation_entries = CorrelationEntry::read_correlation_db(folder_path).unwrap_or_default();
        Self { data_entries, correlation_entries, folder_path: folder_path.to_string() }
    }

    /// # Errors
    /// Will return an error if the data or correlation databases cannot be written.
    pub fn save_to_disk(&self) -> Result<(), Box<dyn std::error::Error>> {
        let scan_path = get_db_path::<DataEntry>(&self.folder_path);
        let mut scan_file = File::create(&scan_path)?;
        for entry in &self.data_entries {
            let stored_entry = DataEntry::new(to_db_path(&self.folder_path, &entry.filename), &entry.data);
            serde_json::to_writer(&mut scan_file, &stored_entry)?;
            writeln!(&mut scan_file)?;
        }

        let corr_path = get_db_path::<CorrelationEntry>(&self.folder_path);
        let mut corr_file = File::create(&corr_path)?;
        for entry in &self.correlation_entries {
            let stored_entry = CorrelationEntry::new(to_db_path(&self.folder_path, &entry.first_path), to_db_path(&self.folder_path, &entry.second_path), entry.corr);
            serde_json::to_writer(&mut corr_file, &stored_entry)?;
            writeln!(&mut corr_file)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn prepare_file_move(&self, source: &str, target: &str) -> FileMoveAction {
        let file_name = Path::new(target).file_name().and_then(|name| name.to_str()).unwrap_or(target);
        let deleted_path = Path::new(&self.folder_path).join(crate::constants::DELETED_FOLDER).join(file_name).display().to_string();

        FileMoveAction {
            source_path: source.to_string(),
            target_path: target.to_string(),
            deleted_path,
            source_data: self.find_data_entry(source).cloned(),
            target_data: self.find_data_entry(target).cloned(),
            removed_correlations: self.correlations_for_files(&[source, target]),
        }
    }

    pub fn commit_file_move(&mut self, source: &str, target: &str) {
        self.remove_data_entry(source);
        self.remove_correlations_with_file(source);
        self.remove_data_entry(target);
        self.remove_correlations_with_file(target);
        let _ = self.save_to_disk();
    }

    /// # Errors
    /// Will return an error if the files cannot be restored or if the databases cannot be saved.
    pub fn undo_file_move(&mut self, action: FileMoveAction) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(data) = action.source_data {
            self.data_entries.push(data);
        }
        if let Some(data) = action.target_data {
            self.data_entries.push(data);
        }

        self.correlation_entries.extend(action.removed_correlations);

        let deleted_folder = Path::new(&self.folder_path).join(crate::constants::DELETED_FOLDER);
        if !deleted_folder.exists() {
            fs::create_dir_all(&deleted_folder)?;
        }

        // After the replace, the source file sits at the target path and the
        // old target file in the deleted folder. Undo moves both back.
        fs::rename(&action.target_path, &action.source_path)?;
        fs::rename(&action.deleted_path, &action.target_path)?;

        self.save_to_disk()
    }

    fn find_data_entry(&self, filename: &str) -> Option<&DataEntry> {
        self.data_entries.iter().find(|e| e.filename == filename)
    }

    fn remove_data_entry(&mut self, filename: &str) {
        self.data_entries.retain(|e| e.filename != filename);
    }

    fn correlations_for_files(&self, files: &[&str]) -> Vec<CorrelationEntry> {
        let file_set: std::collections::HashSet<_> = files.iter().copied().collect();
        self.correlation_entries.iter().filter(|c| file_set.contains(&c.first_path.as_str()) || file_set.contains(&c.second_path.as_str())).cloned().collect()
    }

    fn remove_correlations_with_file(&mut self, filename: &str) {
        self.correlation_entries.retain(|c| c.first_path != filename && c.second_path != filename);
    }
}

/// Returns the database path for `ErrorEntry`.
impl DbPathProvider for ErrorEntry {
    fn db_path_name() -> &'static str {
        ERROR_DATA_FILE_NAME
    }
}

#[cfg(test)]
mod tests {
    // The correlation tests assert exact float results on integer-valued
    // tile data, where the arithmetic is exact by construction.
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

#[cfg(test)]
mod db_path_tests {
    use {super::*, pretty_assertions::assert_eq};

    fn test_folder() -> String {
        if cfg!(windows) { "C:\\pics".to_string() } else { "/pics".to_string() }
    }

    #[test]
    fn test_to_db_path_strips_folder_prefix() {
        let folder = test_folder();
        let nested = Path::new(&folder).join("sub").join("image.png").display().to_string();
        assert_eq!("sub/image.png", to_db_path(&folder, &nested));
        let direct = Path::new(&folder).join("image.png").display().to_string();
        assert_eq!("image.png", to_db_path(&folder, &direct));
    }

    #[test]
    fn test_to_db_path_keeps_outside_paths_absolute() {
        let folder = test_folder();
        let outside = if cfg!(windows) { "C:\\other\\image.png" } else { "/other/image.png" };
        assert_eq!(outside, to_db_path(&folder, outside));
    }

    #[test]
    fn test_from_db_path_resolves_relative_paths() {
        let folder = test_folder();
        let expected = Path::new(&folder).join("sub").join("image.png").display().to_string();
        #[cfg(unix)]
        assert_eq!(expected, from_db_path(&folder, "sub/image.png"));
        #[cfg(windows)]
        assert_eq!(expected, from_db_path(&folder, "sub\\image.png"));
    }

    #[test]
    fn test_from_db_path_keeps_absolute_paths() {
        let folder = test_folder();
        let absolute = if cfg!(windows) { "C:\\elsewhere\\image.png" } else { "/elsewhere/image.png" };
        assert_eq!(absolute, from_db_path(&folder, absolute));
    }
}

#[cfg(all(test, feature = "simd"))]
mod correlation_group_tests {
    // The group kernel tests assert exact float results on integer-valued
    // tile data, where the arithmetic is exact by construction.
    #![allow(clippy::float_cmp)]
    use {super::*, pretty_assertions::assert_eq};

    fn f32_tiles(entry: &[u8]) -> [f32; TOTAL_TILE_COUNT] {
        let mut tiles = [0.0; TOTAL_TILE_COUNT];
        for (tile, value) in tiles.iter_mut().zip(entry.iter()) {
            *tile = f32::from(*value);
        }
        tiles
    }

    #[test]
    fn test_group_matches_single_pair_kernel() {
        let entries: Vec<Vec<u8>> = vec![
            (0..16u8).map(|v| v * 15).collect(),
            (0..16u8).rev().map(|v| v * 15).collect(), // mirrored, correlation 1
            vec![7; 16],                               // flat, zero variance
            (0..16u8).map(|v| (v * 7) % 200).collect(),
            (0..16u8).map(|v| 100 + (v * 3) % 50).collect(),
        ];
        let stats: Vec<EntryStats> = entries.iter().map(|entry| EntryStats::new(entry)).collect();
        let group = EntryGroup::new(&stats);
        let first_tiles = f32_tiles(&entries[0]);
        let result = CorrelationEntry::correlation_group(&stats[0], &group).to_array();
        for (lane, entry) in entries.iter().enumerate() {
            let single = CorrelationEntry::correlation_coefficient_simd(&first_tiles, &f32_tiles(entry));
            assert!((result[lane] - single).abs() < 1e-5, "lane {lane}: group kernel gave {}, single-pair kernel gave {single}", result[lane]);
        }

        // Padding lanes beyond the entries are zero.
        for &corr in result.iter().skip(entries.len()) {
            assert_eq!(0.0, corr, "padding lane must be zero");
        }
    }

    #[test]
    fn test_group_handles_malformed_entries() {
        let entries: Vec<Vec<u8>> = vec![(0..16u8).map(|v| v * 15).collect(), vec![1; 15], vec![2; 17]];
        let stats: Vec<EntryStats> = entries.iter().map(|entry| EntryStats::new(entry)).collect();
        let group = EntryGroup::new(&stats);
        let result = CorrelationEntry::correlation_group(&stats[0], &group).to_array();
        assert_eq!(0.0, result[1], "malformed entry must not correlate");
        assert_eq!(0.0, result[2], "malformed entry must not correlate");
        assert_eq!(0.0, result[3], "padding must be zero");
    }

    #[test]
    fn test_first_group_of() {
        assert_eq!(0, first_group_of(0));
        assert_eq!(0, first_group_of(14));
        assert_eq!(1, first_group_of(15), "the last lane of a group has no partner in it");
        assert_eq!(1, first_group_of(16));
    }
}

#[cfg(all(test, feature = "simd"))]
mod analysis_skip_tests {
    use {
        super::*,
        pretty_assertions::assert_eq,
        std::collections::{HashMap, HashSet},
    };

    fn done_map(pairs: &[(usize, usize)]) -> HashMap<usize, Vec<usize>> {
        let mut done: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(first, second) in pairs {
            done.entry(first).or_default().push(second);
        }
        for list in done.values_mut() {
            list.sort_unstable();
            list.dedup();
        }
        done
    }

    #[test]
    fn test_group_complete() {
        let done = done_map(&[(0, 1), (0, 3)]);
        assert!(!group_complete(&done, 0, 0, 4), "pair (0, 2) is missing");
        assert!(!group_complete(&done, 1, 0, 4), "nothing is analysed for first = 1");

        let done = done_map(&[(0, 1), (0, 2), (0, 3)]);
        assert!(group_complete(&done, 0, 0, 4));
        assert!(group_complete(&done, 3, 0, 4), "no partners greater than 3");

        // A partial final group only counts existing entries.
        let done = done_map(&[(0, 1)]);
        assert!(group_complete(&done, 0, 0, 2));
        assert!(!group_complete(&done, 0, 0, 3), "pair (0, 2) is missing");
    }

    #[test]
    fn test_pair_done() {
        let done = done_map(&[(0, 2)]);
        assert!(pair_done(&done, 0, 2));
        assert!(pair_done(&done, 2, 0), "must be order-insensitive");
        assert!(!pair_done(&done, 0, 1));
        assert!(!pair_done(&done, 1, 3));
    }

    #[test]
    fn test_build_done_map() {
        let entries = [DataEntry::new("a", &[1; 16]), DataEntry::new("b", &[2; 16]), DataEntry::new("c", &[3; 16]), DataEntry::new("d", &[4; 16])];
        let filename_to_index: HashMap<&str, usize> = entries.iter().enumerate().map(|(idx, entry)| (entry.filename.as_str(), idx)).collect();
        let performed = [CorrelationEntry::new("a", "c", 0.99), CorrelationEntry::new("d", "b", 0.98), CorrelationEntry::new("missing", "a", 0.9)];

        let done = build_done_map(&performed, &filename_to_index);

        assert_eq!(Some(&vec![2]), done.get(&0));
        assert_eq!(Some(&vec![3]), done.get(&1), "swapped pair order must be normalised");
        assert!(!done.contains_key(&2));
        assert!(done.values().flatten().collect::<HashSet<_>>().len() <= performed.len());
    }
}

#[cfg(all(test, feature = "simd"))]
#[cfg(test)]
mod ensure_in_db_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use {crate::data, pretty_assertions::assert_eq};

    /// Minimal valid 8x8 PNG, so `ensure_in_db` can exercise the real image
    /// decoding path.
    const TEST_PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x08, 0x02, 0x00, 0x00, 0x00, 0x4B, 0x6D, 0x29,
        0xDC, 0x00, 0x00, 0x00, 0xD3, 0x49, 0x44, 0x41, 0x54, 0x78, 0x01, 0x01, 0xC8, 0x00, 0x37, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x10, 0x40, 0x00, 0x20, 0x60, 0x00, 0x30, 0x80, 0x00, 0x40,
        0xA0, 0x00, 0x50, 0xC0, 0x00, 0x60, 0xE0, 0x00, 0x70, 0x00, 0x00, 0x20, 0x10, 0x20, 0x20, 0x20, 0x40, 0x20, 0x30, 0x60, 0x20, 0x40, 0x80, 0x20, 0x50, 0xA0, 0x20, 0x60, 0xC0, 0x20, 0x70, 0xE0,
        0x20, 0x80, 0x00, 0x00, 0x40, 0x20, 0x20, 0x40, 0x30, 0x40, 0x40, 0x40, 0x60, 0x40, 0x50, 0x80, 0x40, 0x60, 0xA0, 0x40, 0x70, 0xC0, 0x40, 0x80, 0xE0, 0x40, 0x90, 0x00, 0x00, 0x60, 0x30, 0x20,
        0x60, 0x40, 0x40, 0x60, 0x50, 0x60, 0x60, 0x60, 0x80, 0x60, 0x70, 0xA0, 0x60, 0x80, 0xC0, 0x60, 0x90, 0xE0, 0x60, 0xA0, 0x00, 0x00, 0x80, 0x40, 0x20, 0x80, 0x50, 0x40, 0x80, 0x60, 0x60, 0x80,
        0x70, 0x80, 0x80, 0x80, 0xA0, 0x80, 0x90, 0xC0, 0x80, 0xA0, 0xE0, 0x80, 0xB0, 0x00, 0x00, 0xA0, 0x50, 0x20, 0xA0, 0x60, 0x40, 0xA0, 0x70, 0x60, 0xA0, 0x80, 0x80, 0xA0, 0x90, 0xA0, 0xA0, 0xA0,
        0xC0, 0xA0, 0xB0, 0xE0, 0xA0, 0xC0, 0x00, 0x00, 0xC0, 0x60, 0x20, 0xC0, 0x70, 0x40, 0xC0, 0x80, 0x60, 0xC0, 0x90, 0x80, 0xC0, 0xA0, 0xA0, 0xC0, 0xB0, 0xC0, 0xC0, 0xC0, 0xE0, 0xC0, 0xD0, 0x00,
        0x00, 0xE0, 0x70, 0x20, 0xE0, 0x80, 0x40, 0xE0, 0x90, 0x60, 0xE0, 0xA0, 0x80, 0xE0, 0xB0, 0xA0, 0xE0, 0xC0, 0xC0, 0xE0, 0xD0, 0xE0, 0xE0, 0xE0, 0xEE, 0x3F, 0x54, 0x01, 0x7A, 0x88, 0x11, 0xFD,
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    #[test]
    fn test_ensure_in_db() {
        let dir = std::env::temp_dir().join(format!("image_duplicates_ensure_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("Could not create test folder");
        let folder = dir.display().to_string();
        let known = dir.join("known.png").display().to_string();
        let fresh = dir.join("fresh.png").display().to_string();
        std::fs::write(&fresh, TEST_PNG).expect("Could not write test image");

        // Seed the database with one existing entry.
        let entry = data::DataEntry::new(&known, &[7; 16]);
        let db_path = data::get_db_path::<data::DataEntry>(&folder);
        std::fs::write(&db_path, format!("{}\n", serde_json::to_string(&entry).expect("Entry is serializable"))).expect("Could not write test database");

        // A file already in the database is left untouched.
        assert!(!data::DataEntry::ensure_in_db(&folder, &known).expect("Database readable"));
        assert_eq!(1, std::fs::read_to_string(&db_path).expect("Database readable").lines().count());

        // A missing file gets its tile data computed and appended.
        assert!(data::DataEntry::ensure_in_db(&folder, &fresh).expect("Comparison image is processable"));
        let entries = data::DataEntry::read_from_folder(&folder).expect("Database readable");
        assert_eq!(2, entries.len());
        assert!(entries.iter().any(|entry| entry.filename == fresh && entry.data.len() == 16), "fresh entry was not appended: {entries:?}");

        // The appended entry is stored relative to the database folder.
        let raw = std::fs::read_to_string(&db_path).expect("Database readable");
        let fresh_line = raw.lines().last().unwrap_or_default();
        assert!(fresh_line.contains("fresh.png"), "fresh entry was not appended: {raw}");
        assert!(!fresh_line.contains(&folder), "fresh entry was stored with an absolute path: {fresh_line}");

        // The second call is a no-op now.
        assert!(!data::DataEntry::ensure_in_db(&folder, &fresh).expect("Database readable"));

        // An unreadable file is an error, not a panic.
        let missing = dir.join("missing.png").display().to_string();
        assert!(data::DataEntry::ensure_in_db(&folder, &missing).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
