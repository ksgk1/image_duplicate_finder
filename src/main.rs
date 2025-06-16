use std::sync::LazyLock;
use {
    color_eyre::eyre::Result,
    eframe::egui::{self, ProgressBar, mutex::Mutex},
    // lazy_static::lazy_static,
    std::{
        env,
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering},
        },
        thread::available_parallelism,
    },
    tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt},
    ui::{create_folder_selection_block, create_progress_bar_item, create_result_display, create_result_items, create_scanning_item},
};

mod chunk;
mod data;
mod ui;
mod util;

const APPLICATION_NAME: &str = "Image duplicate finder";
const UI_WINDOW_WIDTH: f32 = 1280.0;
const UI_WINDOW_HEIGHT: f32 = 1000.0;
const UI_SCALING_FACTOR: f32 = 1.5;
const SCANNING_DATA_FILE_NAME: &str = "image_duplicates.scan.dat";
const ANALYSING_DATA_FILE_NAME: &str = "image_duplicates.corr.dat";
/// The Folder within the scanned folder's root, to move the duplicates to
/// this folder, will be ignored when scanning for files, so the files are
/// not deleted but ignored for further scans.
const POTENTIAL_DUPLICATES_FOLDER: &str = "potential_duplicates";
/// Defines how many combinations are calculated at once. For large folders with many thousands of files, this will split the work into chunks.
///
/// If there is no chunking, a large number of files will crash the program caused to be out of memory (OOM).
const ANALYSIS_CHUNK_SIZE: usize = 10_000_000;
/// Will split the image into 4x4 tiles
const TILE_COUNT_PER_SIDE: u32 = 4;
/// Not worth saving if the resemblance is too low, it will only slow down the process and generate unnecessary data.
const CORRELATION_THRESHOLD: f32 = 0.95;
/// Defines how many different root scan folders can be used
const MAX_FOLDER_SCANS: usize = 3;

#[derive(Debug, PartialEq)]
enum Direction {
    Forwards,
    Backwards,
    None,
}

// lazy_static! {
//     // TODO: Move arc mutexes from MyApp to here.
//     static ref UNDO_LIST: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
//     static ref AUTO_FORWARD_DIRECTION: Arc<Mutex<Direction>> = Arc::new(Mutex::new(Direction::None));
//     static ref MAX_WORKERS: Arc<AtomicU16> = Arc::new(AtomicU16::new(0));
// }
static UNDO_LIST: LazyLock<Arc<Mutex<Vec<String>>>> = LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));

static AUTO_FORWARD_DIRECTION: LazyLock<Arc<Mutex<Direction>>> = LazyLock::new(|| Arc::new(Mutex::new(Direction::None)));

static MAX_WORKERS: LazyLock<Arc<AtomicU16>> = LazyLock::new(|| Arc::new(AtomicU16::new(0)));

fn main() -> Result<()> {
    color_eyre::install()?;
    let args: Vec<String> = env::args().collect();
    let env_filter = if args.len() > 1 && args.get(1).unwrap().contains("debug") {
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

#[derive(Default)]
struct ImageDuplicatesApp {
    folder_paths: Vec<String>,
    progress: Arc<Mutex<f64>>,
    recursive_paths: Vec<bool>,
    scanning: Arc<AtomicBool>,
    analysing: Arc<AtomicBool>,
    time_remaining: Arc<Mutex<String>>,
    display_images: Arc<AtomicBool>,
    correlation_entry_idx: Arc<AtomicUsize>,
    correlation_data: Arc<Mutex<Vec<data::CorrelationEntry>>>,
    first_path: Arc<Mutex<String>>,
    second_path: Arc<Mutex<String>>,
    clicked_on_image: Arc<Mutex<String>>,
    workers: u16,
}

impl eframe::App for ImageDuplicatesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(UI_SCALING_FACTOR);
        #[allow(clippy::cast_possible_truncation)]
        let progress_bar = ProgressBar::new(*self.progress.lock() as f32).show_percentage();
        let time_remain_scan = Arc::clone(&self.time_remaining);
        let time_remain_analyse = Arc::clone(&self.time_remaining);
        let image_clicked = Arc::clone(&self.clicked_on_image);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Select a folder to be scanned. You can select up to 3 folders.");

            if self.folder_paths.len() < MAX_FOLDER_SCANS {
                let btn = ui.button("Select folder…");
                if btn.clicked() && self.folder_paths.len() < MAX_FOLDER_SCANS {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.folder_paths.push(path.display().to_string());
                        self.recursive_paths.push(false); // init new value in recursive vector
                    }
                }
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
                create_scanning_item(self, ui, can_analyse, time_remain_scan, time_remain_analyse);
                create_progress_bar_item(self, ui, progress_bar);

                if self.folder_paths.iter().filter(|path| Path::new(&format!("{path}/{ANALYSING_DATA_FILE_NAME}")).exists()).count() == self.folder_paths.len() {
                    can_show_results = true;
                }

                create_result_items(self, ui, can_analyse, can_show_results);
                create_result_display(self, ui, &image_clicked);
            }
        });

        // Update ui continuously
        ctx.request_repaint();
    }
}
