#![cfg_attr(feature = "simd", feature(portable_simd))]
use {
    crate::{
        constants::{ANALYSING_DATA_FILE_NAME, APPLICATION_NAME, MAX_FOLDER_SCANS, SCANNING_DATA_FILE_NAME, UI_SCALING_FACTOR, UI_WINDOW_HEIGHT, UI_WINDOW_WIDTH},
        data::{CorrelationEntry, DataEntry},
        progress::LockFreeProgress,
        ui::{create_comparison_file_element, create_scanning_controls, create_folder_selection_block, create_result_display, create_result_items},
    },
    eframe::egui::{self, ProgressBar, Ui, mutex::Mutex},
    std::{
        env,
        path::Path,
        process::exit,
        sync::{
            Arc, LazyLock,
            atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering},
        },
        thread::available_parallelism,
    },
    tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt},
};

mod constants;
mod data;
mod progress;
mod ui;
mod util;

#[derive(Debug, PartialEq)]
enum Direction {
    Forwards,
    Backwards,
    None,
}

/// Global variable to hold the undo list, if a file was accidentally moved
static UNDO_LIST: LazyLock<Arc<Mutex<Vec<String>>>> = LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));
/// Global variable for scroll direction, if entries have been removed
static AUTO_FORWARD_DIRECTION: LazyLock<Arc<Mutex<Direction>>> = LazyLock::new(|| Arc::new(Mutex::new(Direction::None)));
/// Global variable to hold how many CPUs we can use in the application
static MAX_WORKERS: LazyLock<Arc<AtomicU16>> = LazyLock::new(|| Arc::new(AtomicU16::new(0)));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // simple cli interface for testing purposes
    if args.len() > 1 && args.get(1).unwrap().contains("--cli") {
        println!("Using simplified CLI mode");
        if args.get(2).unwrap().contains("--scan") {
            let scan_path = args[3].clone();
            let scanning = Arc::new(AtomicBool::new(true));
            let recursive = true;
            let progress_tracker = Arc::new(LockFreeProgress::new(0));
            let workers = u16::try_from(available_parallelism()?.get()).expect("Number of cores < 2^16");
            DataEntry::scan_folders_with_progress(&[scan_path], &scanning, &[recursive], &progress_tracker, workers);
            println!("Scan complete");
            exit(0);
        }
        if args.get(2).unwrap().contains("--analyse") {
            let analyse_path = args[3].clone();
            let analysing = Arc::new(AtomicBool::new(true));
            let workers = u16::try_from(available_parallelism()?.get()).expect("Number of cores < 2^16");
            let data_entries = DataEntry::read_from_folder(&analyse_path).unwrap_or_default();
            let max_possible_combinations = if data_entries.len() > 1 { data_entries.len() * (data_entries.len() - 1) / 2 } else { 1 };
            let progress_tracker = Arc::new(LockFreeProgress::new(max_possible_combinations));
            CorrelationEntry::analyse_with_progress(&analyse_path, &analysing, &progress_tracker, workers);
            println!("Analyse complete");
            exit(0);
        }
    }

    let env_filter = if args.len() > 1 && args.get(1).unwrap().to_ascii_lowercase().contains("debug") {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    let custom_format = fmt::format().with_target(false).with_file(true).with_level(true).with_line_number(true).compact();
    let fmt_layer = fmt::layer().event_format(custom_format);
    tracing_subscriber::registry().with(env_filter).with(fmt_layer).init();

    {
        // Setting the max workers during the start of the application.
        MAX_WORKERS.store(u16::try_from(available_parallelism()?.get()).expect("Number of cores < 2^16"), Ordering::Relaxed);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT]).with_resizable(true).with_maximize_button(false),
        ..Default::default()
    };
    let _ = eframe::run_native(
        APPLICATION_NAME,
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::<ImageDuplicatesApp>::default())
        }),
    );
    Ok(())
}

struct ImageDuplicatesApp {
    folder_paths: Vec<String>,
    recursive_paths: Vec<bool>,
    scanning: Arc<AtomicBool>,
    analysing: Arc<AtomicBool>,
    display_images: Arc<AtomicBool>,
    correlation_entry_idx: Arc<AtomicUsize>,
    correlation_data: Arc<Mutex<Vec<CorrelationEntry>>>,
    first_path: Arc<Mutex<String>>,
    second_path: Arc<Mutex<String>>,
    find_image_path: Arc<Mutex<String>>,
    clicked_on_image: Arc<Mutex<String>>,
    workers: u16,
    scan_progress: Option<Arc<LockFreeProgress>>,
    analysis_progress: Option<Arc<LockFreeProgress>>,
}

impl Default for ImageDuplicatesApp {
    fn default() -> Self {
        Self {
            folder_paths: Vec::new(),
            recursive_paths: Vec::new(),
            scanning: Arc::new(AtomicBool::new(false)),
            analysing: Arc::new(AtomicBool::new(false)),
            display_images: Arc::new(AtomicBool::new(false)),
            correlation_entry_idx: Arc::new(AtomicUsize::new(0)),
            correlation_data: Arc::new(Mutex::new(Vec::new())),
            first_path: Arc::new(Mutex::new(String::new())),
            second_path: Arc::new(Mutex::new(String::new())),
            find_image_path: Arc::new(Mutex::new(String::new())),
            clicked_on_image: Arc::new(Mutex::new(String::new())),
            workers: 1, // will be set to actual CPU count in main()
            scan_progress: None,
            analysis_progress: None,
        }
    }
}

impl eframe::App for ImageDuplicatesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(UI_SCALING_FACTOR);
        self.cleanup_completed_operations();

        let progress_value = self.get_current_progress();
        let progress_bar = ProgressBar::new(progress_value).show_percentage();
        let image_clicked = Arc::clone(&self.clicked_on_image);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Select a folder to be scanned. You can select up to 3 folders.");

            if self.folder_paths.len() < MAX_FOLDER_SCANS
                && ui.button("Select folder…").clicked()
                && let Some(path) = rfd::FileDialog::new().pick_folder()
            {
                self.folder_paths.push(path.display().to_string());
                self.recursive_paths.push(false); // init new value in recursive vector
            }

            let mut can_analyse = false;
            let mut can_show_results = false;

            if !self.folder_paths.is_empty() {
                create_folder_selection_block(self, ui);
                let all_exists: Vec<bool> = self.folder_paths.iter().map(|path| Path::new(format!("{path}/{SCANNING_DATA_FILE_NAME}").as_str()).exists()).collect();
                if all_exists.iter().filter(|b| **b).count() == self.folder_paths.len() {
                    ui.label("All databases exist.");
                    can_analyse = true;
                }
                create_scanning_controls(self, ui, can_analyse);
                self.create_progress_display(ui, progress_bar);

                create_comparison_file_element(self, ui);

                if self.folder_paths.iter().filter(|path| Path::new(&format!("{path}/{ANALYSING_DATA_FILE_NAME}")).exists()).count() == self.folder_paths.len() {
                    can_show_results = true;
                }

                create_result_items(self, ui, can_analyse, can_show_results);
                create_result_display(self, ui, &image_clicked);
            }
        });

        // update ui continuously
        ctx.request_repaint();
    }
}

impl ImageDuplicatesApp {
    /// Get current progress as a float between 0.0 and 1.0
    fn get_current_progress(&self) -> f32 {
        if let Some(progress) = &self.scan_progress {
            let (processed, total, _) = progress.get_progress();
            if total > 0 {
                return (processed as f32) / (total as f32);
            }
        }

        if let Some(progress) = &self.analysis_progress {
            let (processed, total, _) = progress.get_progress();
            if total > 0 {
                return (processed as f32) / (total as f32);
            }
        }
        0.0
    }

    /// Get current operation status progress
    fn get_current_operation_status(&self) -> String {
        if let Some(progress) = &self.scan_progress {
            let (processed, total, _) = progress.get_progress();
            return format!("Scanning: {processed}/{total} files");
        }

        if let Some(progress) = &self.analysis_progress {
            let (processed, total, _) = progress.get_progress();
            return format!("Analyzing: {processed}/{total} combinations");
        }

        String::new()
    }

    /// Get estimated time remaining
    fn get_estimated_time_remaining(&self) -> String {
        if let Some(progress) = &self.scan_progress
            && let Some(remaining) = progress.estimated_remaining()
        {
            return format!("Estimated: {remaining:.2?} remaining");
        }

        if let Some(progress) = &self.analysis_progress
            && let Some(remaining) = progress.estimated_remaining()
        {
            return format!("Estimated: {remaining:.2?} remaining");
        }

        "Calculating time remaining...".to_string()
    }

    /// Clean up completed progress trackers
    fn cleanup_completed_operations(&mut self) {
        if !self.scanning.load(Ordering::Relaxed) {
            self.scan_progress = None;
        }

        if !self.analysing.load(Ordering::Relaxed) {
            self.analysis_progress = None;
        }
    }

    /// Creates the UI elements for the status bar
    fn create_progress_display(&self, ui: &mut Ui, progress_bar: ProgressBar) {
        let scanning = self.scanning.load(Ordering::Relaxed);
        let analysing = self.analysing.load(Ordering::Relaxed);

        if scanning || analysing {
            ui.add(progress_bar);

            let operation_status = self.get_current_operation_status();
            if !operation_status.is_empty() {
                ui.label(operation_status);
            }

            let time_remaining = self.get_estimated_time_remaining();
            ui.label(time_remaining);
        }
    }
}
