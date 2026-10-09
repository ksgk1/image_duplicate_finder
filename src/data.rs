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
//! 2. **Analysis**: `CorrelationEntry::analyse_with_progress_chunked` compares images for similarities
//! 3. **Persistence**: Data is stored in JSON files alongside the image folders

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

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
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::Instant,
    },
    tracing::{debug, error, info},
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
        #![allow(clippy::cast_possible_truncation)]
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
        let tile_width = image_width / TILE_COUNT_PER_SIDE as u32;
        let tile_height = image_height / TILE_COUNT_PER_SIDE as u32;

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
        for tile_idx in 0..TOTAL_TILE_COUNT {
            let row = tile_idx / TILE_COUNT_PER_SIDE;
            let col = tile_idx % TILE_COUNT_PER_SIDE;

            let x_start = col as u32 * tile_width;
            let y_start = row as u32 * tile_height;
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

        Ok(tiles_averages)
    }

    /// Reads the entries from a given folder, determining the correct name automatically.
    ///
    /// Corrupt lines are logged and skipped.
    ///
    /// Returning a Result vec of all entries.
    pub(crate) fn read_from_folder(folder_path: &str) -> Result<Vec<Self>, ImageDupeError> {
        let target_file = get_db_path::<Self>(folder_path);
        let file_handle = File::open(&target_file)?;
        Ok(io::BufReader::new(file_handle)
            .lines()
            .map_while(Result::ok)
            .filter(|line| !line.is_empty())
            .filter_map(|line| match serde_json::from_str::<Self>(&line) {
                Ok(entry) => Some(entry),
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
        let entry = Self::new(file_path, &averages);
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
    pub fn scan_with_progress(folder_path: &str, files_to_scan_list: &[String], scanning: &Arc<AtomicBool>, progress_tracker: &Arc<LockFreeProgress>, worker_count: u16) {
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
                match Self::calculate_image_tile_data(&file_path) {
                    Ok(averages) => {
                        let entry = Self::new(&file_path, &averages);
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
                                let error_entry = ErrorEntry::new(&file_path, &e.to_string());
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
    pub fn scan_folders_with_progress(folders: &[String], exclude: Option<&[String]>, scanning: &Arc<AtomicBool>, recursive: &[bool], progress_tracker: &Arc<LockFreeProgress>, worker_count: u16) {
        let mut files_to_scan_list = Vec::<String>::new();
        for (folder, recursive) in folders.iter().zip(recursive.iter()) {
            match Self::generate_file_list(folder, *recursive, exclude) {
                Ok(mut file_list_for_folder) => files_to_scan_list.append(&mut file_list_for_folder),
                Err(e) => {
                    error!("Failed to generate file list for {folder}: {e}");
                }
            }
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

    #[cfg(feature = "simd")]
    #[must_use]
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn correlation_coefficient_simd(first: &[f32; TOTAL_TILE_COUNT], second: &[f32; TOTAL_TILE_COUNT]) -> f32 {
        // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        // Optimized SIMD implementation with reduced redundant calculations
        const LANES: usize = 8;
        let len = TOTAL_TILE_COUNT as f32;

        // Pre-calculate all second variants to avoid recalculating in each loop
        let second_original = *second;
        let second_horizontal = Self::mirror_horizontal(second);
        let second_vertical = Self::mirror_vertical(second);
        let second_variants = [&second_original, &second_horizontal, &second_vertical];

        // Calculate mean of first array with SIMD
        let mut sum_first = f32x8::splat(0.0);
        for chunk in first.chunks_exact(LANES) {
            sum_first += f32x8::from_slice(chunk);
        }
        let remainder_first_sum: f32 = first[LANES * (first.len() / LANES)..].iter().sum();
        let mean_first = (sum_first.reduce_sum() + remainder_first_sum) / len;
        let mean_first_vec = f32x8::splat(mean_first);

        // Calculate variance of first array with SIMD
        let mut var_sum_first = f32x8::splat(0.0);
        for chunk in first.chunks_exact(LANES) {
            let vals = f32x8::from_slice(chunk);
            let diff = vals - mean_first_vec;
            var_sum_first += diff * diff;
        }
        let remainder_first_var: f32 = first[LANES * (first.len() / LANES)..].iter().map(|v| (v - mean_first).powi(2)).sum();
        let var_first = (var_sum_first.reduce_sum() + remainder_first_var) / (len - 1.0);

        if var_first == 0.0 {
            return 0.0;
        }

        let mut max_corr_coeff = 0.0f32;

        for second_variant in second_variants {
            // Calculate mean of current second variant (should be same as first mean for correlation)
            // But we calculate it properly for numerical stability
            let mut sum_second = f32x8::splat(0.0);
            for chunk in second_variant.chunks_exact(LANES) {
                sum_second += f32x8::from_slice(chunk);
            }
            let remainder_second_sum: f32 = second_variant[LANES * (second_variant.len() / LANES)..].iter().sum();
            let mean_second = (sum_second.reduce_sum() + remainder_second_sum) / len;
            let mean_second_vec = f32x8::splat(mean_second);

            // Calculate variance of second variant with SIMD
            let mut var_sum_second = f32x8::splat(0.0);
            for chunk in second_variant.chunks_exact(LANES) {
                let vals = f32x8::from_slice(chunk);
                let diff = vals - mean_second_vec;
                var_sum_second += diff * diff;
            }
            let remainder_second_var: f32 = second_variant[LANES * (second_variant.len() / LANES)..].iter().map(|v| (v - mean_second).powi(2)).sum();
            let var_second = (var_sum_second.reduce_sum() + remainder_second_var) / (len - 1.0);

            if var_second == 0.0 {
                continue;
            }

            // Calculate covariance with SIMD - single pass through both arrays
            let mut cov_sum = f32x8::splat(0.0);
            for (chunk_first, chunk_second) in first.chunks_exact(LANES).zip(second_variant.chunks_exact(LANES)) {
                let vals_first = f32x8::from_slice(chunk_first);
                let vals_second = f32x8::from_slice(chunk_second);
                let diff_first = vals_first - mean_first_vec;
                let diff_second = vals_second - mean_second_vec;
                cov_sum += diff_first * diff_second;
            }

            // Handle remainders
            let remainder_cov: f32 =
                first[LANES * (first.len() / LANES)..].iter().zip(second_variant[LANES * (second_variant.len() / LANES)..].iter()).map(|(a, b)| (a - mean_first) * (b - mean_second)).sum();

            let cov = (cov_sum.reduce_sum() + remainder_cov) / (len - 1.0);

            let corr_coeff = (cov / (var_first * var_second).sqrt()).abs();
            max_corr_coeff = max_corr_coeff.max(corr_coeff);
        }

        max_corr_coeff
    }

    /// Creates a slice of f32 from a Vec of u8 for processing the correlation data.
    #[inline]
    fn vec_u8_to_slice_f32(input: &[u8]) -> [f32; TOTAL_TILE_COUNT] {
        input.iter().map(|v| f32::from(*v)).collect::<Vec<f32>>().try_into().expect("Slice can be converted, with the correct number of entries")
    }

    /// # Panics
    /// Panics if the given folder does not exist.
    /// Analysis method that processes combinations in chunks to handle large datasets
    ///
    /// If `preloaded_data` is provided, it will be used directly instead of loading from disk.
    /// This allows the UI to pre-load data and avoid the startup delay.
    #[allow(clippy::mut_range_bound)]
    #[allow(clippy::too_many_lines)]
    pub fn analyse_with_progress_chunked(
        folder_path: &str,
        analysing: &Arc<AtomicBool>,
        progress_tracker: &Arc<LockFreeProgress>,
        worker_count: u16,
        max_combinations_per_chunk: usize,
        preloaded_data: Option<Vec<DataEntry>>,
    ) {
        info!("Using {worker_count} workers.\nAnalysing database in \"{folder_path}\" with chunked processing.");
        info!("Processing {} combinations per chunk", max_combinations_per_chunk);

        if let Err(e) = Self::create_correlation_db(folder_path) {
            error!("Could not create correlation database for {folder_path}: {e}");
            analysing.store(false, Ordering::Relaxed);
            return;
        }
        let db_path = get_db_path::<Self>(folder_path);

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

        let existing_combinations: std::collections::HashSet<(usize, usize)> =
            already_performed.iter().filter_map(|entry| filename_to_index.get(entry.first_path.as_str()).and_then(|&i| filename_to_index.get(entry.second_path.as_str()).map(|&j| (i, j)))).collect();

        info!("Found {} existing correlations", existing_combinations.len());

        // Calculate total possible combinations for progress tracking
        let total_possible = (entry_count * (entry_count - 1)) / 2;
        info!("Total possible combinations: {}", total_possible);

        // Combinations already in the database never enter the work queue,
        // so the progress tracker only covers the remaining ones. Crediting
        // the skipped combinations towards the full total instead would make
        // an incremental re-run look like a full analysis in the UI.
        let remaining_combinations = total_possible.saturating_sub(existing_combinations.len());
        progress_tracker.reset(remaining_combinations);
        info!("Skipping {} already analysed combinations", existing_combinations.len());

        let overall_start_time = Instant::now();

        // Use a continuous work queue instead of chunking for smoother parallelism
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
        Self::process_combination_chunk(&data_entries, work_rx, &db_path, analysing, worker_count, progress_tracker);

        let total_generated = generation_handle.join().unwrap_or(0);

        // Only mark as complete if we finished naturally (not cancelled)
        if analysing.load(Ordering::Relaxed) {
            analysing.store(false, Ordering::Relaxed);
            let total_time = overall_start_time.elapsed();
            info!("Analysis complete! Total combinations generated: {}, processed: {} out of {} possible. Time taken: {:.2?}", total_generated, total_possible, total_possible, total_time);
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn process_combination_chunk(
        data_entries: &[DataEntry],
        work_rx: crossbeam::channel::Receiver<(usize, usize)>,
        db_path: &str,
        analysing: &Arc<AtomicBool>,
        worker_count: u16,
        progress_tracker: &Arc<LockFreeProgress>,
    ) {
        let combinations_processed = Arc::new(AtomicUsize::new(0));
        let target_db_file_handle = match OpenOptions::new().append(true).open(db_path) {
            Ok(handle) => handle,
            Err(e) => {
                error!("Cannot open correlation database file {db_path}: {e}");
                return;
            }
        };
        let target_db_file_mutex = Arc::new(Mutex::new(target_db_file_handle));

        let start_time = Instant::now();

        crossbeam::scope(|scope| {
            for worker_idx in 0..worker_count {
                let worker_db_file = Arc::clone(&target_db_file_mutex);
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

                        let entry = Self::new(&first_entry.filename, &second_entry.filename, corr);

                        // Only save entries with significant correlation to avoid unnecessary data
                        if entry.corr > crate::constants::CORRELATION_THRESHOLD {
                            let worker_db_file = &worker_db_file.lock();
                            if let Err(e) = entry.save(worker_db_file) {
                                error!("Failed to save correlation entry for {} & {}: {}", first_entry.filename, second_entry.filename, e);
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
        })
        .expect("Processing finishes successfully.");

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

    #[allow(dead_code)]
    pub(crate) fn create_error_db(folder_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let target_file = get_db_path::<Self>(folder_path);
        let target_path = Path::new(&target_file);
        if !target_path.exists() {
            File::create(&target_file)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn read_error_db(folder_path: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let file_name = get_db_path::<Self>(folder_path);
        let file_handle = File::open(file_name)?;
        let result: Vec<Self> = io::BufReader::new(file_handle)
            .lines()
            .map(|line| line.and_then(|line_str| serde_json::from_str(&line_str).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))))
            .collect::<Result<Vec<Self>, _>>()?;
        Ok(result)
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
            serde_json::to_writer(&mut scan_file, entry)?;
            writeln!(&mut scan_file)?;
        }

        let corr_path = get_db_path::<CorrelationEntry>(&self.folder_path);
        let mut corr_file = File::create(&corr_path)?;
        for entry in &self.correlation_entries {
            serde_json::to_writer(&mut corr_file, entry)?;
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
mod ensure_in_db_tests {
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

        // The second call is a no-op now.
        assert!(!data::DataEntry::ensure_in_db(&folder, &fresh).expect("Database readable"));

        // An unreadable file is an error, not a panic.
        let missing = dir.join("missing.png").display().to_string();
        assert!(data::DataEntry::ensure_in_db(&folder, &missing).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
