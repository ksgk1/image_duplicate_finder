/// Name of the application window
pub const APPLICATION_NAME: &str = "Image duplicate finder";
/// Window width for the UI, can not me changed at the moment
pub const UI_WINDOW_WIDTH: f32 = 1280.0;
/// Window width for the UI, can not me changed at the moment
pub const UI_WINDOW_HEIGHT: f32 = 1000.0;
/// Scaling factor to make things easier to read
pub const UI_SCALING_FACTOR: f32 = 1.5;
/// Filename for the scanning data
pub const SCANNING_DATA_FILE_NAME: &str = "image_duplicates.scan.dat";
/// Filename for the analysis data
pub const ANALYSING_DATA_FILE_NAME: &str = "image_duplicates.corr.dat";
/// The Folder within the scanned folder's root, to move the duplicates to
/// this folder, will be ignored when scanning for files, so the files are
/// not deleted but ignored for further scans.
pub const POTENTIAL_DUPLICATES_FOLDER: &str = "potential_duplicates";
/// Not worth saving if the resemblance is too low, it will only slow down the process and generate unnecessary data.
pub const CORRELATION_THRESHOLD: f32 = 0.95;
/// Defines how many different root scan folders can be used
pub const MAX_FOLDER_SCANS: usize = 3;
/// Filename for the analysis data
pub const ERROR_DATA_FILE_NAME: &str = "image_duplicates.error.dat";
/// Folder to store files that were overwritten, allowing restoration on undo
pub const DELETED_FOLDER: &str = "deleted";
/// Maximum combinations to process in a single chunk during analysis
pub const ANALYSIS_CHUNK_SIZE: usize = 500_000;
/// All the currently supported image format extensions.
pub const SUPPORTED_IMAGE_FILE_EXTENSION: [&str; 4] = ["jpeg", "jpg", "png", "webp"];
