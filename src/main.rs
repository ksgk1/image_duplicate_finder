use {
    eframe::egui,
    image_duplicate_finder::{
        app::ImageDuplicatesApp,
        constants::{APPLICATION_NAME, MAX_FOLDER_SCANS, UI_WINDOW_HEIGHT, UI_WINDOW_WIDTH},
        data::{CorrelationEntry, DataEntry},
        progress::LockFreeProgress,
    },
    std::{
        env,
        path::Path,
        process::exit,
        sync::{Arc, atomic::AtomicBool},
    },
    tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt},
};

fn main() {
    let args: Vec<String> = env::args().collect();

    // simple cli interface for testing purposes
    if args.len() > 1 && args[1].contains("--cli") {
        println!("Using simplified CLI mode");
        let command = args.get(2).map_or("", String::as_str);
        if command.contains("--scan") {
            let Some(scan_path) = args.get(3) else {
                eprintln!("Missing folder argument for --scan");
                exit(1);
            };
            let scan_path = scan_path.clone();
            let scanning = Arc::new(AtomicBool::new(true));
            let recursive = true;
            let progress_tracker = Arc::new(LockFreeProgress::new(0));
            let mut excludes: Vec<String> = Vec::new();
            for (index, arg) in args.iter().enumerate() {
                if arg == "--exclude" {
                    excludes.push(args[index + 1].clone());
                }
            }
            DataEntry::scan_folders_with_progress(&[scan_path], if excludes.is_empty() { None } else { Some(&excludes) }, &scanning, &[recursive], &progress_tracker);
            println!("Scan complete");
            exit(0);
        }
        if command.contains("--analyse") {
            let Some(analyse_path) = args.get(3) else {
                eprintln!("Missing folder argument for --analyse");
                exit(1);
            };
            let analyse_path = analyse_path.clone();
            let analysing = Arc::new(AtomicBool::new(true));
            let data_entries = DataEntry::read_from_folder(&analyse_path).unwrap_or_default();
            let max_possible_combinations = if data_entries.len() > 1 { data_entries.len() * (data_entries.len() - 1) / 2 } else { 1 };
            let progress_tracker = Arc::new(LockFreeProgress::new(max_possible_combinations));
            CorrelationEntry::analyse_with_progress(&analyse_path, &analysing, &progress_tracker, Some(data_entries));
            println!("Analyse complete");
            exit(0);
        }
    }

    let env_filter = if args.len() > 1 && args[1].to_ascii_lowercase().contains("debug") {
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
    if let Err(e) = eframe::run_native(
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
    ) {
        eprintln!("Could not run the application: {e}");
    }
}
