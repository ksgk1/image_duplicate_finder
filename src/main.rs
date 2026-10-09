#![cfg_attr(feature = "simd", feature(portable_simd))]

use {
    crate::{
        app::ImageDuplicatesApp,
        constants::{APPLICATION_NAME, MAX_FOLDER_SCANS, SCANNING_DATA_FILE_NAME, UI_SCALING_FACTOR, UI_WINDOW_HEIGHT, UI_WINDOW_WIDTH},
        data::{CorrelationEntry, DataEntry, get_db_path},
        progress::LockFreeProgress,
        ui::{create_bottom_bar, create_result_display, create_setup_panel, create_top_bar},
    },
    eframe::egui::{self, ProgressBar, Ui},
    std::{
        env,
        path::Path,
        process::exit,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread::available_parallelism,
        time::Duration,
    },
    tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt},
};

mod app;
mod constants;
mod data;
mod error;
mod progress;
mod ui;
mod util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // simple cli interface for testing purposes
    if args.len() > 1 && args.get(1).expect("First argument exists in args.").contains("--cli") {
        println!("Using simplified CLI mode");
        if args.get(2).expect("Second argument exists in args.").contains("--scan") {
            let scan_path = args[3].clone();
            let scanning = Arc::new(AtomicBool::new(true));
            let recursive = true;
            let progress_tracker = Arc::new(LockFreeProgress::new(0));
            let workers = u16::try_from(available_parallelism()?.get()).expect("Number of cores < 2^16");
            let mut excludes: Vec<String> = Vec::new();
            for (index, arg) in args.iter().enumerate() {
                if arg == "--exclude" {
                    excludes.push(args[index + 1].clone());
                }
            }
            DataEntry::scan_folders_with_progress(&[scan_path], if excludes.is_empty() { None } else { Some(&excludes) }, &scanning, &[recursive], &progress_tracker, workers);
            println!("Scan complete");
            exit(0);
        }
        if args.get(2).expect("Second argument exits in args.").contains("--analyse") {
            let analyse_path = args[3].clone();
            let analysing = Arc::new(AtomicBool::new(true));
            let workers = u16::try_from(available_parallelism()?.get()).expect("Number of cores < 2^16");
            let data_entries = DataEntry::read_from_folder(&analyse_path).unwrap_or_default();
            let max_possible_combinations = if data_entries.len() > 1 { data_entries.len() * (data_entries.len() - 1) / 2 } else { 1 };
            let progress_tracker = Arc::new(LockFreeProgress::new(max_possible_combinations));
            CorrelationEntry::analyse_with_progress_chunked(&analyse_path, &analysing, &progress_tracker, workers, 1_000_000, Some(data_entries));
            println!("Analyse complete");
            exit(0);
        }
    }

    let env_filter = if args.len() > 1 && args.get(1).expect("First argument exists in args.").to_ascii_lowercase().contains("debug") {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    let custom_format = fmt::format().with_target(false).with_file(true).with_level(true).with_line_number(true).compact();
    let fmt_layer = fmt::layer().event_format(custom_format);
    tracing_subscriber::registry().with(env_filter).with(fmt_layer).init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT]).with_resizable(true).with_maximize_button(false),
        ..Default::default()
    };
    // Arguments that are existing directories become the initial folder
    // selection, so a debug session does not need manual folder picking.
    let initial_folders: Vec<String> = args.iter().skip(1).filter(|arg| Path::new(arg).is_dir()).take(MAX_FOLDER_SCANS).cloned().collect();
    let _ = eframe::run_native(
        APPLICATION_NAME,
        options,
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            let mut app = ImageDuplicatesApp::default();
            for folder in &initial_folders {
                app.add_folder(folder.clone());
            }
            Ok(Box::new(app))
        }),
    );
    Ok(())
}

impl eframe::App for ImageDuplicatesApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        ui.set_pixels_per_point(UI_SCALING_FACTOR);
        self.cleanup_completed_operations();

        let progress_bar = ProgressBar::new(self.get_current_progress()).show_percentage();
        let can_analyse = self.folders.iter().all(|folder| Path::new(&folder.path).join(SCANNING_DATA_FILE_NAME).exists());
        let has_analysis_data = self.folders.first().is_some_and(|folder| Path::new(&get_db_path::<CorrelationEntry>(&folder.path)).exists());

        egui::Panel::top("top_bar").show(ui, |ui| {
            create_top_bar(self, ui);
        });

        egui::Panel::left("setup_panel").default_size(300.0).min_size(240.0).show(ui, |ui| {
            create_setup_panel(self, ui, can_analyse, has_analysis_data);
        });

        egui::Panel::bottom("bottom_bar").show(ui, |ui| {
            create_bottom_bar(self, ui, progress_bar, can_analyse, has_analysis_data);
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.display_results {
                create_result_display(self, ui);
            } else if self.folders.is_empty() {
                ui.centered_and_justified(|ui| ui.strong("Select a folder in the panel on the left to begin."));
            } else {
                ui.centered_and_justified(|ui| ui.strong("Scan and analyse your folders, then show the results from the panel on the left."));
            }
        });

        // Repaint continuously only while background work is running; input
        // events trigger repaints on their own otherwise.
        if self.scanning.load(Ordering::Relaxed) || self.analysing.load(Ordering::Relaxed) {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }
}

impl ImageDuplicatesApp {
    /// Get current progress as a float between 0.0 and 1.0
    /// Priority: Scanning > Analyzing > Idle (0.0)
    fn get_current_progress(&self) -> f32 {
        // Check scanning first (higher priority)
        if let Some(scan_progress) = &self.scan_progress {
            let (processed, total, _) = scan_progress.get_progress();
            if total > 0 {
                return (processed as f32) / (total as f32);
            }
        }

        // Then check analyzing
        if let Some(analysis_progress) = &self.analysis_progress {
            let (processed, total, _) = analysis_progress.get_progress();
            if total > 0 {
                return (processed as f32) / (total as f32);
            }
        }
        0.0
    }

    /// Get current operation status with improved formatting
    fn get_current_operation_status(&self) -> String {
        // Check scanning first (higher priority)
        if let Some(progress) = &self.scan_progress {
            let (processed, total, _) = progress.get_progress();
            return Self::format_operation_status("Scanning", processed, total, "files");
        }

        // Then check analyzing
        if let Some(progress) = &self.analysis_progress {
            let (processed, total, _) = progress.get_progress();
            return Self::format_operation_status("Analyzing", processed, total, "combinations");
        }

        String::new()
    }

    /// Format operation status with consistent formatting
    fn format_operation_status(operation: &str, processed: usize, total: usize, unit: &str) -> String {
        if total > 0 {
            let percentage = if total > 0 { (processed as f64 / total as f64) * 100.0 } else { 0.0 };
            format!("{operation}: {processed}/{total} {unit} ({percentage:.1}%)")
        } else {
            format!("{operation}: {processed} {unit}")
        }
    }

    /// Get estimated time remaining with improved formatting
    fn get_estimated_time_remaining(&self) -> String {
        // Check scanning first (higher priority)
        if let Some(progress) = &self.scan_progress {
            if let Some(remaining) = progress.estimated_remaining() {
                if remaining.is_zero() {
                    return "Complete!".to_string();
                }
                return Self::format_time_remaining(remaining);
            }
            return "Calculating time remaining...".to_string();
        }

        // Then check analyzing
        if let Some(progress) = &self.analysis_progress {
            if let Some(remaining) = progress.estimated_remaining() {
                if remaining.is_zero() {
                    return "Complete!".to_string();
                }
                return Self::format_time_remaining(remaining);
            }
            return "Calculating time remaining...".to_string();
        }

        String::new()
    }

    /// Format time remaining in a human-readable format
    fn format_time_remaining(duration: std::time::Duration) -> String {
        let total_seconds = duration.as_secs();

        if total_seconds < 60 {
            format!("Estimated: {total_seconds} seconds remaining")
        } else if total_seconds < 3600 {
            let minutes = total_seconds / 60;
            let seconds = total_seconds % 60;
            format!("Estimated: {minutes}m {seconds}s remaining")
        } else {
            let hours = total_seconds / 3600;
            let minutes = (total_seconds % 3600) / 60;
            let seconds = total_seconds % 60;
            format!("Estimated: {hours}h {minutes}m {seconds}s remaining")
        }
    }

    /// Creates the UI elements for the status bar
    fn create_progress_display(&self, ui: &mut Ui, progress_bar: ProgressBar) {
        let scanning = self.scanning.load(Ordering::Relaxed);
        let analysing = self.analysing.load(Ordering::Relaxed);

        if scanning || analysing {
            ui.horizontal(|ui| {
                ui.add(progress_bar.desired_width(240.0));

                let operation_status = self.get_current_operation_status();
                if !operation_status.is_empty() {
                    ui.label(operation_status);
                }

                let time_remaining = self.get_estimated_time_remaining();
                if !time_remaining.is_empty() {
                    ui.label(time_remaining);
                }
            });
        }
    }
}
