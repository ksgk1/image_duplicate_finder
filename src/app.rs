//! # App Module
//!
//! Application state and user actions for the image duplicates application.
//!
//! This module owns the mutable state of the GUI and implements every action
//! the UI can trigger (scanning, analysis, file moves, undo). The UI module
//! only renders state and calls these methods, so no file system or thread
//! logic lives in the rendering code.
//!
//! ## Thread affinity
//!
//! - `scanning`, `analysing` and the progress trackers are shared with the
//!   worker threads and use atomics and `Arc`.
//! - Everything else is only touched on the UI thread and is plain data,
//!   so it needs no locking.

use {
    crate::{
        constants::{ANALYSIS_CHUNK_SIZE, CORRELATION_THRESHOLD, DELETED_FOLDER, POTENTIAL_DUPLICATES_FOLDER},
        data::{self, AppState, CorrelationEntry, DataEntry, FileMoveAction},
        progress::LockFreeProgress,
        util::has_valid_image_extension,
    },
    std::{
        collections::HashMap,
        fs,
        path::{Path, PathBuf},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread::available_parallelism,
        time::{Duration, Instant},
    },
    tracing::{error, info},
};

/// Configuration of a single folder selected for scanning.
#[derive(Debug, Clone)]
pub struct FolderConfig {
    /// Absolute path of the folder.
    pub path: String,
    /// Whether to scan the folder recursively.
    pub recursive: bool,
    /// Whether to exclude the folder (and its subfolders) from scanning.
    pub excluded: bool,
}

/// Direction in which to advance through the results when the current
/// entry disappears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forwards,
    Backwards,
}

/// A reversible action performed through the UI.
#[derive(Debug)]
pub enum UndoAction {
    /// An image was moved into the `potential_duplicates` folder.
    MovedToDuplicates { original: PathBuf, duplicate: PathBuf },
    /// A higher-resolution image replaced a lower-resolution one.
    Replaced(FileMoveAction),
}

/// How often the displayed files are checked for existence.
const MISSING_FILE_RECHECK_INTERVAL: Duration = Duration::from_millis(500);

pub struct ImageDuplicatesApp {
    pub folders: Vec<FolderConfig>,
    /// Set while a scan worker thread is running.
    pub scanning: Arc<AtomicBool>,
    /// Set while an analysis worker thread is running.
    pub analysing: Arc<AtomicBool>,
    pub scan_progress: Option<Arc<LockFreeProgress>>,
    pub analysis_progress: Option<Arc<LockFreeProgress>>,
    /// Whether the correlation results are currently displayed.
    pub display_results: bool,
    /// Correlation entries above the correlation threshold, filtered by the
    /// comparison image if one is selected.
    pub correlation_data: Vec<CorrelationEntry>,
    pub correlation_idx: usize,
    /// Path of the first displayed image.
    pub first_path: String,
    /// Path of the second displayed image.
    pub second_path: String,
    /// Image the results are filtered to, empty for no filter.
    pub find_image_path: String,
    /// Worker count used for scanning and analysis.
    pub workers: u16,
    /// Upper bound for `workers`.
    pub max_workers: u16,
    /// State of the primary folder, loaded lazily when a file action needs
    /// it. Reloaded automatically whenever a background operation has run
    /// since it was cached, because the databases it was loaded from have
    /// then changed on disk.
    pub app_state: Option<AppState>,
    /// The `db_epoch` at which the cached [`AppState`] was loaded.
    app_state_epoch: usize,
    /// Incremented whenever a background operation starts or the databases
    /// change, so the cached [`AppState`] is refreshed before its next use.
    db_epoch: usize,
    pub undo_history: Vec<UndoAction>,
    /// Direction to auto-advance in when the current entry disappears.
    auto_direction: Option<Direction>,
    /// Whether the result display should refresh automatically once the
    /// running analysis finishes.
    pending_result_reload: bool,
    /// Cached image dimensions, to avoid decoding image headers every frame.
    dimensions: HashMap<String, (u32, u32)>,
    /// When the displayed files were last checked for existence.
    last_fs_check: Option<Instant>,
    /// Last user-facing error message, displayed in red by the UI.
    pub status: Option<String>,
}

impl Default for ImageDuplicatesApp {
    fn default() -> Self {
        let max_workers = u16::try_from(available_parallelism().map_or(1, usize::from)).unwrap_or(1);
        Self {
            folders: Vec::new(),
            scanning: Arc::new(AtomicBool::new(false)),
            analysing: Arc::new(AtomicBool::new(false)),
            scan_progress: None,
            analysis_progress: None,
            display_results: false,
            correlation_data: Vec::new(),
            correlation_idx: 0,
            first_path: String::new(),
            second_path: String::new(),
            find_image_path: String::new(),
            workers: max_workers,
            max_workers,
            app_state: None,
            app_state_epoch: 0,
            db_epoch: 0,
            undo_history: Vec::new(),
            auto_direction: None,
            pending_result_reload: false,
            dimensions: HashMap::new(),
            last_fs_check: None,
            status: None,
        }
    }
}

impl ImageDuplicatesApp {
    /// Adds a folder to the selection.
    pub fn add_folder(&mut self, path: String) {
        self.folders.push(FolderConfig { path, recursive: false, excluded: false });
    }

    // -- background work --------------------------------------------------

    /// Starts a scan of all selected folders on a worker thread.
    ///
    /// File list generation and database access happen on the worker thread,
    /// so the UI does not stall on large folders.
    pub fn start_scan(&mut self) {
        self.scanning.store(true, Ordering::Relaxed);
        self.analysing.store(false, Ordering::Relaxed);
        self.db_epoch += 1;

        let folders: Vec<String> = self.folders.iter().map(|folder| folder.path.clone()).collect();
        let recursive: Vec<bool> = self.folders.iter().map(|folder| folder.recursive).collect();
        let excludes: Vec<String> = self.folders.iter().filter(|folder| folder.excluded).map(|folder| folder.path.clone()).collect();
        let scanning = Arc::clone(&self.scanning);
        let progress_tracker = Arc::new(LockFreeProgress::new(0));
        self.scan_progress = Some(Arc::clone(&progress_tracker));
        let worker_count = self.workers;

        std::thread::spawn(move || {
            // The progress total is set by the scan once the file list is known.
            DataEntry::scan_folders_with_progress(&folders, if excludes.is_empty() { None } else { Some(&excludes) }, &scanning, &recursive, &progress_tracker, worker_count);
        });
    }

    /// Requests the running scan to stop.
    pub fn stop_scan(&self) {
        self.scanning.store(false, Ordering::Relaxed);
    }

    /// Starts the analysis of the first selected folder on a worker thread.
    ///
    /// Reading the scan database happens on the worker thread, so the UI does
    /// not stall on large databases.
    pub fn start_analysis(&mut self) {
        let Some(folder) = self.folders.first().map(|folder| folder.path.clone()) else {
            self.status = Some("No folder selected".to_string());
            return;
        };
        self.analysing.store(true, Ordering::Relaxed);
        self.scanning.store(false, Ordering::Relaxed);
        self.db_epoch += 1;

        let analysing = Arc::clone(&self.analysing);
        let progress_tracker = Arc::new(LockFreeProgress::new(0));
        self.analysis_progress = Some(Arc::clone(&progress_tracker));
        let worker_count = self.workers;

        std::thread::spawn(move || {
            let data_entries = DataEntry::read_from_folder(&folder).unwrap_or_default();
            let total = if data_entries.len() > 1 { data_entries.len() * (data_entries.len() - 1) / 2 } else { 1 };
            progress_tracker.reset(total);
            CorrelationEntry::analyse_with_progress_chunked(&folder, &analysing, &progress_tracker, worker_count, ANALYSIS_CHUNK_SIZE, Some(data_entries));
        });
    }

    /// Requests the running analysis to stop.
    pub fn stop_analysis(&self) {
        self.analysing.store(false, Ordering::Relaxed);
    }

    /// Selects `path` as the comparison image and makes it comparable: if the
    /// image has no entry in the scan database of the first selected folder
    /// (e.g. it lives outside the folder), its tile data is computed and
    /// appended to the database. An analysis is then started so the image is
    /// correlated against all entries; pairs already in the correlation
    /// database are skipped by the analysis.
    ///
    /// # Errors
    /// Returns an error and reports it as status if the image cannot be read
    /// or processed, or if no scan data exists to compare against.
    pub fn select_comparison_image(&mut self, path: &str) -> Result<(), String> {
        self.find_image_path = path.to_string();
        if self.analysing.load(Ordering::Relaxed) || self.scanning.load(Ordering::Relaxed) {
            let message = "A scan or analysis is already running, the comparison image will be included in the next analysis".to_string();
            self.status = Some(message.clone());
            return Err(message);
        }
        let Some(folder) = self.folders.first().map(|folder| folder.path.clone()) else {
            let message = "No folder selected".to_string();
            self.status = Some(message.clone());
            return Err(message);
        };
        if !Path::new(&data::get_db_path::<DataEntry>(&folder)).exists() {
            let message = format!("No scan data found in {folder}; scan the folder before comparing images");
            self.status = Some(message.clone());
            return Err(message);
        }
        // One image to decode; cheap enough for the UI thread and it lets the
        // error surface immediately instead of only in the log.
        if let Err(e) = DataEntry::ensure_in_db(&folder, path) {
            let message = format!("Could not process comparison image {path}: {e}");
            self.status = Some(message.clone());
            return Err(message);
        }
        self.status = None;
        self.start_analysis();
        self.pending_result_reload = true;
        Ok(())
    }

    /// Called once per frame by the UI. Cleans up the progress trackers of
    /// finished background operations and refreshes the result display once
    /// an analysis started by selecting a comparison image has finished, so
    /// the filtered results appear without further clicks.
    pub fn cleanup_completed_operations(&mut self) {
        let busy = self.scanning.load(Ordering::Relaxed) || self.analysing.load(Ordering::Relaxed);
        if !busy {
            self.scan_progress = None;
            self.analysis_progress = None;
            if self.pending_result_reload {
                self.pending_result_reload = false;
                self.show_results();
            }
        }
    }

    /// Deletes the correlation database of the first selected folder.
    pub fn delete_analysis_db(&mut self) {
        self.analysing.store(false, Ordering::Relaxed);
        let Some(folder) = self.folders.first() else { return };
        let db_path = data::get_db_path::<CorrelationEntry>(&folder.path);
        if !Path::new(&db_path).exists() {
            return;
        }
        match fs::remove_file(&db_path) {
            Ok(()) => {
                info!("Successfully removed analysis data");
                self.status = None;
                // The correlation database changed on disk.
                self.db_epoch += 1;
            }
            Err(e) => {
                error!("Could not remove analysis data: {e}");
                self.status = Some(format!("Could not remove analysis data: {e}"));
            }
        }
    }

    // -- results ----------------------------------------------------------

    /// Loads the correlation results of all selected folders into memory,
    /// keeping only entries above the correlation threshold and, if a
    /// comparison image was picked, only entries involving that image.
    pub fn show_results(&mut self) {
        let mut results = Vec::new();
        let mut failed_folders = 0;
        for folder in &self.folders {
            match CorrelationEntry::read_correlation_db(&folder.path) {
                Ok(entries) => results.extend(entries.into_iter().filter(|entry| entry.corr > CORRELATION_THRESHOLD)),
                Err(e) => {
                    failed_folders += 1;
                    error!("Could not read correlation database in {}: {e}", folder.path);
                }
            }
        }
        if failed_folders == self.folders.len() && !self.folders.is_empty() {
            self.status = Some("Could not read any correlation database".to_string());
            self.display_results = false;
            return;
        }
        if !self.find_image_path.is_empty() {
            results.retain(|entry| entry.first_path == self.find_image_path || entry.second_path == self.find_image_path);
        }
        if results.is_empty() {
            self.status = Some(if self.find_image_path.is_empty() { "No correlation data found".to_string() } else { format!("No correlation data found for image: {}", self.find_image_path) });
            self.display_results = false;
            return;
        }
        self.correlation_data = results;
        self.correlation_idx = clamp_index(self.correlation_idx, self.correlation_data.len());
        self.display_results = true;
        self.status = None;
        self.set_current_image_paths();
        self.invalidate_file_caches();
    }

    /// Clears the comparison image filter. If results are currently
    /// displayed, they are reloaded without the filter — keeping the
    /// currently shown pair selected where possible — so all entries can be
    /// browsed again.
    pub fn clear_comparison_image(&mut self) {
        self.find_image_path.clear();
        // The unfiltered reload below supersedes a pending filtered refresh.
        self.pending_result_reload = false;
        if !self.display_results {
            return;
        }
        let current_first = self.first_path.clone();
        let current_second = self.second_path.clone();
        self.show_results();
        if let Some(idx) = self
            .correlation_data
            .iter()
            .position(|entry| (entry.first_path == current_first && entry.second_path == current_second) || (entry.first_path == current_second && entry.second_path == current_first))
        {
            self.correlation_idx = idx;
            self.set_current_image_paths();
        }
    }

    /// Closes the result display and discards the loaded entries. Cancels a
    /// pending automatic refresh, since the user closed the display on
    /// purpose.
    pub fn close_results(&mut self) {
        self.pending_result_reload = false;
        self.correlation_data.clear();
        self.display_results = false;
    }

    /// Moves to the previous or next result entry, wrapping around.
    pub fn go_to_adjacent_entry(&mut self, backwards: bool) {
        self.auto_direction = Some(if backwards { Direction::Backwards } else { Direction::Forwards });
        self.correlation_idx = advance_index(self.correlation_idx, self.correlation_data.len(), backwards);
        self.set_current_image_paths();
        self.invalidate_file_caches();
    }

    /// Advances to the next entry when one of the displayed files no longer
    /// exists. The check is throttled so it does not stat the file system on
    /// every frame.
    pub fn auto_advance_if_missing(&mut self) {
        let due = self.last_fs_check.is_none_or(|checked| checked.elapsed() >= MISSING_FILE_RECHECK_INTERVAL);
        if !due {
            return;
        }
        self.last_fs_check = Some(Instant::now());
        let missing = !Path::new(&self.first_path).exists() || !Path::new(&self.second_path).exists();
        if !missing || self.correlation_data.is_empty() {
            return;
        }
        let backwards = self.auto_direction == Some(Direction::Backwards);
        self.auto_direction = Some(if backwards { Direction::Backwards } else { Direction::Forwards });
        self.correlation_idx = advance_index(self.correlation_idx, self.correlation_data.len(), backwards);
        self.set_current_image_paths();
        self.invalidate_file_caches();
    }

    /// Sets the displayed image paths to the entry at the current index.
    fn set_current_image_paths(&mut self) {
        if let Some(entry) = self.correlation_data.get(self.correlation_idx) {
            self.first_path.clone_from(&entry.first_path);
            self.second_path.clone_from(&entry.second_path);
        } else {
            self.first_path.clear();
            self.second_path.clear();
        }
    }

    /// Correlation value of the currently displayed entry.
    #[must_use]
    pub fn current_correlation(&self) -> Option<f32> {
        self.correlation_data.get(self.correlation_idx).map(|entry| entry.corr)
    }

    /// Dimensions of the two displayed images, cached to avoid decoding the
    /// image headers on every frame. Returns `None` if either file cannot be
    /// opened or decoded.
    pub fn current_image_dimensions(&mut self) -> Option<((u32, u32), (u32, u32))> {
        let Self { first_path, second_path, dimensions, .. } = self;
        Some((cached_image_dimensions(dimensions, first_path)?, cached_image_dimensions(dimensions, second_path)?))
    }

    // -- file actions -----------------------------------------------------

    /// Moves `file_path` into the `potential_duplicates` subfolder of the
    /// first selected folder and records the move for undo.
    ///
    /// # Errors
    /// Returns an error if the file has no supported image extension, if the
    /// folder cannot be selected, or if the move fails.
    pub fn move_to_duplicates(&mut self, file_path: &str) -> Result<(), String> {
        if !has_valid_image_extension(Path::new(file_path)) {
            let message = format!("Cannot move {file_path}: unsupported file extension");
            self.status = Some(message.clone());
            return Err(message);
        }
        let Some(folder) = self.folders.first().map(|folder| folder.path.clone()) else {
            return Err("No folder selected".to_string());
        };
        let duplicates_folder = Path::new(&folder).join(POTENTIAL_DUPLICATES_FOLDER);
        if let Err(e) = fs::create_dir_all(&duplicates_folder) {
            let message = format!("Could not create {}: {e}", duplicates_folder.display());
            self.status = Some(message.clone());
            return Err(message);
        }
        let Some(file_name) = Path::new(file_path).file_name() else {
            return Err(format!("Invalid file path: {file_path}"));
        };
        let duplicate_path = duplicates_folder.join(file_name);
        if let Err(e) = fs::rename(file_path, &duplicate_path) {
            let message = format!("Failed to move {file_path} to duplicates folder: {e}");
            self.status = Some(message.clone());
            return Err(message);
        }
        info!("Moved {file_path} to {}", duplicate_path.display());
        self.undo_history.push(UndoAction::MovedToDuplicates { original: file_path.into(), duplicate: duplicate_path });
        self.remove_correlations_of(file_path);
        self.status = None;
        Ok(())
    }

    /// Overwrites `target` with `source`: moves `target` into the `deleted`
    /// folder, renames `source` to `target`, records the move for undo,
    /// removes the correlation entries of both files and adjusts the current
    /// index to the shrunken list. Rolls back the first move if the second
    /// one fails.
    ///
    /// # Errors
    /// Returns an error if the deleted folder cannot be created or if either
    /// rename fails.
    pub fn replace_file(&mut self, source: &str, target: &str) -> Result<(), String> {
        let Some(folder) = self.folders.first().map(|folder| folder.path.clone()) else {
            return Err("No folder selected".to_string());
        };
        let Some(state) = self.fresh_app_state() else {
            return Err("No folder selected".to_string());
        };
        let action = state.prepare_file_move(source, target);

        let deleted_folder = Path::new(&folder).join(DELETED_FOLDER);
        if let Err(e) = fs::create_dir_all(&deleted_folder) {
            let message = format!("Could not create {}: {e}", deleted_folder.display());
            self.status = Some(message.clone());
            return Err(message);
        }
        let file_name = Path::new(target).file_name().and_then(|name| name.to_str()).unwrap_or(target);
        let deleted_target = deleted_folder.join(file_name);

        if let Err(e) = fs::rename(target, &deleted_target) {
            let message = format!("Failed to move {target} to deleted folder: {e}");
            self.status = Some(message.clone());
            return Err(message);
        }
        if let Err(e) = fs::rename(source, target) {
            // Roll back the first move so the pair is not left broken.
            if let Err(rollback) = fs::rename(&deleted_target, target) {
                let message = format!("Failed to overwrite {source} → {target}: {e}. Rollback failed ({rollback}), {target} remains at {}", deleted_target.display());
                self.status = Some(message.clone());
                return Err(message);
            }
            let message = format!("Failed to overwrite {source} → {target}: {e}");
            self.status = Some(message.clone());
            return Err(message);
        }
        info!("Overwriting {source} → {target}");
        state.commit_file_move(source, target);
        self.undo_history.push(UndoAction::Replaced(action));
        for path in [source, target] {
            self.remove_correlations_of(path);
        }
        self.status = None;
        Ok(())
    }

    /// Restores the most recent action from the undo history.
    ///
    /// # Errors
    /// Returns an error if the file system or the application state prevents
    /// the restore.
    pub fn undo_last(&mut self) -> Result<(), String> {
        let Some(action) = self.undo_history.pop() else {
            return Ok(());
        };
        let result = match action {
            UndoAction::MovedToDuplicates { original, duplicate } => fs::rename(&duplicate, &original).map_err(|e| format!("Failed to restore {}: {e}", original.display())),
            UndoAction::Replaced(action) => {
                let Some(state) = self.fresh_app_state() else {
                    let message = "No folder selected".to_string();
                    self.status = Some(message.clone());
                    return Err(message);
                };
                state.undo_file_move(action).map_err(|e| format!("Failed to undo file move: {e}"))
            }
        };
        match result {
            Ok(()) => {
                info!("Restored last action");
                self.status = None;
            }
            Err(message) => {
                error!("{message}");
                self.status = Some(message.clone());
                return Err(message);
            }
        }
        self.invalidate_file_caches();
        Ok(())
    }

    /// Returns the lazily loaded application state of the first selected
    /// folder, reloading it first if a background operation has run since it
    /// was cached, because the databases on disk have then changed. Using a
    /// stale state would make the next file action truncate the databases
    /// with the old in-memory content.
    fn fresh_app_state(&mut self) -> Option<&mut AppState> {
        if self.app_state.is_some() && self.app_state_epoch == self.db_epoch {
            return self.app_state.as_mut();
        }
        let folder = self.folders.first().map(|folder| folder.path.clone())?;
        let state = AppState::load(&folder);
        self.app_state = Some(state);
        self.app_state_epoch = self.db_epoch;
        self.app_state.as_mut()
    }

    /// Removes all correlation entries involving `file_path` from the
    /// displayed results and adjusts the current index.
    fn remove_correlations_of(&mut self, file_path: &str) {
        self.correlation_data.retain(|entry| entry.first_path != file_path && entry.second_path != file_path);
        self.correlation_idx = clamp_index(self.correlation_idx, self.correlation_data.len());
        self.set_current_image_paths();
        self.invalidate_file_caches();
    }

    /// Drops cached image dimensions and existence information after a file
    /// system mutation.
    fn invalidate_file_caches(&mut self) {
        self.dimensions.clear();
        self.last_fs_check = None;
    }
}

/// Returns the image dimensions for `path`, decoding and caching them on
/// first use. Returns `None` if the file cannot be opened or decoded.
fn cached_image_dimensions(cache: &mut HashMap<String, (u32, u32)>, path: &str) -> Option<(u32, u32)> {
    if let Some(&dimensions) = cache.get(path) {
        return Some(dimensions);
    }
    if path.is_empty() {
        return None;
    }
    let dimensions = image::ImageReader::open(path).ok()?.into_dimensions().ok()?;
    cache.insert(path.to_string(), dimensions);
    Some(dimensions)
}

/// Wraps an index forwards or backwards in a list of `len` entries.
/// Returns 0 for an empty list instead of underflowing.
const fn advance_index(idx: usize, len: usize, backwards: bool) -> usize {
    if len == 0 {
        return 0;
    }
    if backwards {
        if idx > 0 { idx - 1 } else { len - 1 }
    } else if idx < len - 1 {
        idx + 1
    } else {
        0
    }
}

/// Clamps an index to the last valid position after entries were removed.
const fn clamp_index(idx: usize, len: usize) -> usize {
    let last = len.saturating_sub(1);
    if idx < last { idx } else { last }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::constants::{ANALYSING_DATA_FILE_NAME, DELETED_FOLDER, POTENTIAL_DUPLICATES_FOLDER, SCANNING_DATA_FILE_NAME},
        pretty_assertions::assert_eq,
        serde::Serialize,
    };

    /// Writes database entries as one JSON object per line.
    fn write_db(entries: &[impl Serialize], path: &std::path::Path) {
        let mut content = String::new();
        for entry in entries {
            content.push_str(&serde_json::to_string(entry).expect("Entry is serializable"));
            content.push('\n');
        }
        fs::write(path, content).expect("Could not write test database");
    }

    /// Minimal valid 8x8 PNG for exercising the real image decoding path.
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

    /// Creates a temp folder with two distinct files and an app selecting it.
    fn app_with_test_folder(name: &str) -> (PathBuf, ImageDuplicatesApp) {
        let dir = std::env::temp_dir().join(format!("image_duplicates_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("Could not create test folder");
        fs::write(dir.join("a.png"), b"content-a").expect("Could not write test file");
        fs::write(dir.join("b.png"), b"content-b").expect("Could not write test file");
        let mut app = ImageDuplicatesApp::default();
        app.add_folder(dir.display().to_string());
        (dir, app)
    }

    #[test]
    fn test_move_to_duplicates_and_undo() {
        let (dir, mut app) = app_with_test_folder("move");
        let original = dir.join("a.png").display().to_string();
        let duplicate = dir.join(POTENTIAL_DUPLICATES_FOLDER).join("a.png");

        app.move_to_duplicates(&original).expect("Move should succeed");
        assert!(!Path::new(&original).exists());
        assert!(duplicate.exists());

        app.undo_last().expect("Undo should succeed");
        assert!(Path::new(&original).exists());
        assert!(!duplicate.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_replace_and_undo() {
        let (dir, mut app) = app_with_test_folder("replace");
        let source = dir.join("a.png").display().to_string();
        let target = dir.join("b.png").display().to_string();
        let deleted = dir.join(DELETED_FOLDER).join("b.png");

        app.replace_file(&source, &target).expect("Replace should succeed");
        assert!(!Path::new(&source).exists());
        assert_eq!(b"content-a", &*fs::read(&target).expect("Target exists"));
        assert_eq!(b"content-b", &*fs::read(&deleted).expect("Deleted copy exists"));

        // Undo must restore both files to their original locations.
        app.undo_last().expect("Undo should succeed");
        assert_eq!(b"content-a", &*fs::read(&source).expect("Source exists"));
        assert_eq!(b"content-b", &*fs::read(&target).expect("Target exists"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_replace_rolls_back_on_failure() {
        let (dir, mut app) = app_with_test_folder("rollback");
        let missing = dir.join("missing.png").display().to_string();
        let target = dir.join("b.png").display().to_string();
        let deleted = dir.join(DELETED_FOLDER).join("b.png");

        assert!(app.replace_file(&missing, &target).is_err());
        // The target must be back in place, not stuck in the deleted folder.
        assert_eq!(b"content-b", &*fs::read(&target).expect("Target was rolled back"));
        assert!(!deleted.exists());
        assert!(app.status.is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    /// Reproduces the "empty .dat files after replace" bug: the UI loaded
    /// `AppState` before the databases existed, a scan and an analysis
    /// filled them on disk, and the first replace then wrote the stale
    /// (empty) in-memory state back, truncating both files.
    #[test]
    fn test_replace_keeps_unrelated_database_entries() {
        let (dir, mut app) = app_with_test_folder("stale");

        // Seed the databases as they look after a scan and an analysis.
        let scan_entries: Vec<DataEntry> = ["a", "b", "c", "d"].iter().map(|name| DataEntry::new(dir.join(format!("{name}.png")).display().to_string(), &[0; 16])).collect();
        let corr_entries = [
            CorrelationEntry::new(dir.join("a.png").display().to_string(), dir.join("b.png").display().to_string(), 0.99),
            CorrelationEntry::new(dir.join("c.png").display().to_string(), dir.join("d.png").display().to_string(), 0.98),
        ];
        write_db(&scan_entries, &dir.join(SCANNING_DATA_FILE_NAME));

        // Frame 1 of the old UI: state loaded while the scan database
        // already exists but the correlation database does not.
        let folder = dir.display().to_string();
        app.app_state = Some(AppState::load(&folder));

        // Analysis runs afterwards, filling the correlation database. It
        // bumps the database epoch, so the next file action must reload the
        // cached (stale) state from disk instead of writing it back.
        write_db(&corr_entries, &dir.join(ANALYSING_DATA_FILE_NAME));
        app.db_epoch += 1;

        let source = dir.join("a.png").display().to_string();
        let target = dir.join("b.png").display().to_string();
        app.replace_file(&source, &target).expect("Replace should succeed");

        // The untouched files' entries must survive the replace.
        let scan_after = fs::read_to_string(dir.join(SCANNING_DATA_FILE_NAME)).expect("Scan database exists");
        assert!(scan_after.contains("c.png"), "scan database lost unrelated entries: {scan_after}");
        assert!(scan_after.contains("d.png"), "scan database lost unrelated entries: {scan_after}");
        let corr_after = fs::read_to_string(dir.join(ANALYSING_DATA_FILE_NAME)).expect("Correlation database exists");
        assert!(corr_after.contains("d.png"), "correlation database lost unrelated entries: {corr_after}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// Selecting a comparison image that is not in the scan database yet
    /// must compute its tile data, append it, and run an analysis that
    /// correlates it against the existing entries.
    #[test]
    fn test_select_comparison_image_analysis() {
        let dir = std::env::temp_dir().join(format!("image_duplicates_compare_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("Could not create test folder");
        let folder = dir.display().to_string();
        let comparison = dir.join("comparison.png");
        fs::write(&comparison, TEST_PNG).expect("Could not write test image");
        let comparison = comparison.display().to_string();
        let existing = dir.join("existing.png").display().to_string();

        // Phase 1: add the comparison image once to learn its tile data.
        assert!(DataEntry::ensure_in_db(&folder, &comparison).expect("Comparison image is processable"));
        let tiles = DataEntry::read_from_folder(&folder).expect("Database readable").pop().expect("Entry exists").data;

        // Seed the database with one entry sharing the same tile data, so the
        // analysis must produce a correlation above the save threshold.
        write_db(&[DataEntry::new(&existing, &tiles)], &dir.join(SCANNING_DATA_FILE_NAME));

        // Phase 2: the actual selection flow.
        let mut app = ImageDuplicatesApp::default();
        app.add_folder(folder);
        app.select_comparison_image(&comparison).expect("Selection should start the analysis");

        // Wait for the spawned analysis to finish.
        let deadline = Instant::now() + Duration::from_secs(20);
        while app.analysing.load(Ordering::Relaxed) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!app.analysing.load(Ordering::Relaxed), "Analysis did not finish in time");

        let corr = fs::read_to_string(dir.join(ANALYSING_DATA_FILE_NAME)).expect("Correlation database exists");
        assert!(corr.contains("comparison.png"), "comparison image was not correlated: {corr}");
        assert!(corr.contains("existing.png"), "existing entry was not correlated: {corr}");

        // The per-frame cleanup refreshes the display automatically once the
        // analysis has finished, without any further clicks.
        app.cleanup_completed_operations();
        assert!(app.display_results, "Results should appear automatically after the analysis");
        assert_eq!(1, app.correlation_data.len());
        assert!(app.correlation_data[0].first_path == comparison || app.correlation_data[0].second_path == comparison);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Closing the results while a comparison analysis is still running must
    /// cancel the automatic refresh: the display stays closed afterwards.
    #[test]
    fn test_pending_reload_cancelled_by_close() {
        let dir = std::env::temp_dir().join(format!("image_duplicates_cancel_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("Could not create test folder");
        let folder = dir.display().to_string();
        let a = dir.join("a.png").display().to_string();
        let b = dir.join("b.png").display().to_string();
        write_db(&[CorrelationEntry::new(a, b, 0.99)], &dir.join(ANALYSING_DATA_FILE_NAME));

        let mut app = ImageDuplicatesApp::default();
        app.add_folder(folder);
        app.find_image_path = dir.join("x.png").display().to_string();
        app.pending_result_reload = true;

        // The user closes the display while the analysis is still running.
        app.close_results();
        app.cleanup_completed_operations();
        assert!(!app.display_results, "Cancelled refresh must not reopen the display");
        assert_eq!(0, app.correlation_data.len());

        let _ = fs::remove_dir_all(&dir);
    }

    /// Clearing the comparison image while results are displayed must
    /// restore the unfiltered entry list and keep the currently shown pair
    /// selected.
    #[test]
    fn test_clear_comparison_image_restores_all_entries() {
        let dir = std::env::temp_dir().join(format!("image_duplicates_clear_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("Could not create test folder");
        let folder = dir.display().to_string();
        let a = dir.join("a.png").display().to_string();
        let b = dir.join("b.png").display().to_string();
        let c = dir.join("c.png").display().to_string();
        let d = dir.join("d.png").display().to_string();

        let corr_entries = [CorrelationEntry::new(a.clone(), b.clone(), 0.99), CorrelationEntry::new(c, d, 0.97)];
        write_db(&corr_entries, &dir.join(ANALYSING_DATA_FILE_NAME));

        let mut app = ImageDuplicatesApp::default();
        app.add_folder(folder);

        // Results filtered by image a: only the (a, b) pair is shown.
        app.find_image_path = a.clone();
        app.show_results();
        assert!(app.display_results, "Filtered results should be shown");
        assert_eq!(1, app.correlation_data.len());
        assert_eq!(a, app.first_path);
        assert_eq!(b, app.second_path);

        // Clearing restores the full list and stays on the same pair.
        app.clear_comparison_image();
        assert_eq!(0, app.find_image_path.len());
        assert!(app.display_results, "Results should stay visible after clearing");
        assert_eq!(2, app.correlation_data.len());
        assert_eq!(a, app.first_path, "The shown pair should be preserved");
        assert_eq!(b, app.second_path, "The shown pair should be preserved");

        // Navigation moves through the full list again.
        app.go_to_adjacent_entry(false);
        assert_eq!(1, app.correlation_idx);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_advance_index_forwards() {
        assert_eq!(1, advance_index(0, 3, false));
        assert_eq!(2, advance_index(1, 3, false));
        assert_eq!(0, advance_index(2, 3, false)); // wraps to start
        assert_eq!(0, advance_index(5, 3, false)); // out-of-bounds wraps too
    }

    #[test]
    fn test_advance_index_backwards() {
        assert_eq!(0, advance_index(1, 3, true));
        assert_eq!(2, advance_index(0, 3, true)); // wraps to end
        assert_eq!(1, advance_index(2, 3, true));
        assert_eq!(4, advance_index(5, 3, true)); // out-of-bounds still steps down one
    }

    #[test]
    fn test_advance_index_degenerate() {
        // Empty list must not underflow
        assert_eq!(0, advance_index(0, 0, false));
        assert_eq!(0, advance_index(0, 0, true));
        // Single entry stays at 0 in both directions
        assert_eq!(0, advance_index(0, 1, false));
        assert_eq!(0, advance_index(0, 1, true));
    }

    #[test]
    fn test_clamp_index() {
        assert_eq!(0, clamp_index(0, 1));
        assert_eq!(0, clamp_index(0, 0));
        assert_eq!(0, clamp_index(3, 0)); // list emptied: fall back to 0
        assert_eq!(2, clamp_index(5, 3)); // out of bounds: last valid index
        assert_eq!(2, clamp_index(2, 3)); // already valid: unchanged
        assert_eq!(1, clamp_index(1, 3));
    }
}
