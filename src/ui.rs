//! # UI Module
//!
//! User interface rendering for the image duplicates application.
//!
//! This module only draws the current application state and forwards user
//! input. All state mutations, file system access and thread management
//! live in [`crate::app::ImageDuplicatesApp`]; the functions here call its
//! action methods and never touch the file system themselves.

use {
    crate::{
        app::ImageDuplicatesApp,
        constants::{APPLICATION_NAME, MAX_FOLDER_SCANS, SUPPORTED_IMAGE_FILE_EXTENSION},
        data::{self, CorrelationEntry},
        util::{has_valid_image_extension, have_matching_extensions, shorten_string},
    },
    eframe::egui::{self, Color32, Image, RichText, Sense, TextStyle, Ui},
    std::{
        cmp::Ordering::{Equal, Greater, Less},
        fs,
        path::Path,
        sync::atomic::Ordering,
    },
    tracing::error,
};

/// Which results button the UI should render in a frame. Rendering both at
/// once must never happen, so the choice is exclusive by state, not by click.
#[derive(Debug, PartialEq)]
enum ResultsButton {
    CloseResults,
    ShowResults,
    None,
}

impl ResultsButton {
    /// Decides which results button to render: while results are open only
    /// "Close results" is offered; "Show results" is offered when the app
    /// is idle and analysis data exists; otherwise the button is disabled.
    #[allow(clippy::fn_params_excessive_bools)] // the flags mirror the app state
    const fn select_state(display_results: bool, analysing: bool, scanning: bool, has_analysis_data: bool) -> Self {
        if display_results {
            Self::CloseResults
        } else if !analysing && !scanning && has_analysis_data {
            Self::ShowResults
        } else {
            Self::None
        }
    }
}

/// Which workflow step the user should take next, derived from the data
/// present for the selected folders.
#[derive(Debug, PartialEq)]
enum WorkflowHint {
    /// No folder is selected yet.
    NoFolders,
    /// Folders are selected but not all of them have scan data.
    NeedsScan,
    /// Scan data exists, but no analysis data yet.
    NeedsAnalysis,
    /// Analysis data exists; the results can be reviewed.
    ResultsReady,
}

impl WorkflowHint {
    /// Decides the next workflow step from the data availability. A folder
    /// selection without scan data always wins over analysis state, so the
    /// user is never told to analyse or review anything while a scan is still
    /// missing.
    const fn select_hint(has_folders: bool, can_analyse: bool, has_analysis_data: bool) -> Self {
        if !has_folders {
            Self::NoFolders
        } else if !can_analyse {
            Self::NeedsScan
        } else if !has_analysis_data {
            Self::NeedsAnalysis
        } else {
            Self::ResultsReady
        }
    }
}

/// Renders the top bar: application title
pub fn create_top_bar(app: &ImageDuplicatesApp, ui: &mut Ui) {
    ui.horizontal_centered(|ui| {
        ui.heading(APPLICATION_NAME);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            create_status_label(app, ui);
        });
    });
}

/// Renders the bottom status bar: the progress of a running scan or
/// analysis while one is running, otherwise the next workflow step. The
/// two never share the bar, so their texts cannot overlap.
pub fn create_bottom_bar(app: &ImageDuplicatesApp, ui: &mut Ui, progress_bar: egui::ProgressBar, can_analyse: bool, has_analysis_data: bool) {
    ui.horizontal_centered(|ui| {
        let scanning = app.scanning.load(Ordering::Relaxed);
        let analysing = app.analysing.load(Ordering::Relaxed);
        if scanning || analysing {
            app.create_progress_display(ui, progress_bar);
            return;
        }
        let hint = match WorkflowHint::select_hint(!app.folders.is_empty(), can_analyse, has_analysis_data) {
            WorkflowHint::NoFolders => "Select a folder in the panel on the left to begin.",
            WorkflowHint::NeedsScan => "The selected folders have no scan data yet: start a scan.",
            WorkflowHint::NeedsAnalysis => "Scan data found: analyse it to find the duplicates.",
            WorkflowHint::ResultsReady => "Analysis data found: show the results to review the duplicates.",
        };
        ui.label(hint);
    });
}

/// Renders the last status message, if any, truncated with the full text
/// available on hover.
pub fn create_status_label(app: &ImageDuplicatesApp, ui: &mut Ui) {
    if let Some(status) = &app.status {
        const STATUS_LABEL_MAX_LENGTH: usize = 60;
        ui.label(RichText::new(shorten_string(status, STATUS_LABEL_MAX_LENGTH)).color(Color32::RED)).on_hover_text(status.as_str());
    }
}

/// Renders the setup sidebar: folder selection, scanning controls,
/// comparison image selection and the results toggle, grouped into
/// sections in pipeline order.
pub fn create_setup_panel(app: &mut ImageDuplicatesApp, ui: &mut Ui, can_analyse: bool, has_analysis_data: bool) {
    section_heading(ui, "Folders");
    if app.folders.len() < MAX_FOLDER_SCANS
        && ui.button("Select folder…").clicked()
        && let Some(path) = rfd::FileDialog::new().pick_folder()
    {
        app.add_folder(path.display().to_string());
    }
    if !app.folders.is_empty() {
        ui.add_space(4.0);
        create_folder_selection_block(app, ui);
    }
    ui.separator();

    section_heading(ui, "Scanning");
    create_scanning_controls(app, ui, can_analyse);
    ui.separator();

    section_heading(ui, "Search image");
    create_search_image_element(app, ui);
    ui.separator();

    section_heading(ui, "Results");
    create_result_items(app, ui, has_analysis_data);
}

/// Renders a small section title used to structure the setup panel.
fn section_heading(ui: &mut Ui, text: &str) {
    ui.add_space(2.0);
    ui.strong(text);
}

/// Renders the list of selected folders as compact cards with their options
/// and remove button.
fn create_folder_selection_block(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    let mut remove_idx = None;
    for (idx, folder) in app.folders.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.monospace(shorten_string(&folder.path, 34)).on_hover_text(folder.path.as_str());
            ui.horizontal(|ui| {
                ui.add_enabled(!folder.excluded, egui::Checkbox::new(&mut folder.recursive, "Recursive"))
                    .on_hover_text("Also scan the subfolders of this folder.")
                    .on_disabled_hover_text("Excluded folders are never scanned.");
                if ui.checkbox(&mut folder.excluded, "Excluded").on_hover_text("Do not scan this folder.").changed() && folder.excluded {
                    folder.recursive = false; // deactivate recursive when the folder is excluded
                }
                if ui.small_button("Remove").clicked() {
                    remove_idx = Some(idx);
                }
            });
        });
        if remove_idx.is_some() {
            break; // indices no longer align after removing an element
        }
    }
    if let Some(idx) = remove_idx {
        app.folders.remove(idx);
    }
}

/// Renders the scan/analyse controls, the worker slider and the
/// "Delete analysis data" button.
fn create_scanning_controls(app: &mut ImageDuplicatesApp, ui: &mut Ui, can_analyse: bool) {
    ui.horizontal(|ui| {
        let scanning = app.scanning.load(Ordering::Relaxed);
        let analysing = app.analysing.load(Ordering::Relaxed);

        if scanning {
            if ui.button("Stop scanning").clicked() {
                app.stop_scan();
            }
        } else if ui.button("Start scanning").clicked() {
            app.start_scan();
        }

        if analysing {
            if ui.button("Stop analysing").clicked() {
                app.stop_analysis();
            }
        } else {
            let analyse_hint = if scanning { "Wait for the running scan to finish." } else { "Scan every selected folder first." };
            let analyse = ui.add_enabled(can_analyse && !scanning, egui::Button::new("Analyse data")).on_disabled_hover_text(analyse_hint);
            if analyse.clicked() {
                app.start_analysis();
            }
        }
    });

    ui.add(egui::Slider::new(&mut app.workers, 1..=app.max_workers).text("Max workers")).on_hover_ui(|ui| {
        ui.label("Select the number of maximum workers to be used for scanning/analysing.");
    });

    let correlation_db_path = app.folders.first().map(|folder| data::get_db_path::<CorrelationEntry>(&folder.path));
    let db_exists = correlation_db_path.as_deref().is_some_and(|db_path| fs::exists(db_path).unwrap_or(false));
    let delete = ui
        .add_enabled(db_exists, egui::Button::new("Delete analysis data"))
        .on_hover_text("Deletes the stored analysis data of the first selected folder.")
        .on_disabled_hover_text("No analysis data exists yet.");
    if db_exists && delete.clicked() {
        app.delete_analysis_db();
    }
}

/// Renders the search image selection: picking an image restricts the
/// displayed results to entries involving that image. Unknown images are
/// scanned and analysed first.
fn create_search_image_element(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    ui.horizontal(|ui| {
        if ui.button("Select search image…").clicked()
            && let Some(path) = rfd::FileDialog::new().add_filter("Images", &SUPPORTED_IMAGE_FILE_EXTENSION).pick_file()
            && let Err(e) = app.select_comparison_image(&path.display().to_string())
        {
            error!("{e}");
        }
        if !app.find_image_path.is_empty() && ui.button("Clear").clicked() {
            app.clear_comparison_image();
        }
    });
    if app.find_image_path.is_empty() {
        ui.label("No search image selected.").on_hover_text("Shows only pairs in the results that involve this image. Unknown images are scanned and analysed first.");
    } else {
        ui.label(format!("Search image: {}", shorten_string(&app.find_image_path, 40))).on_hover_text(app.find_image_path.as_str());
    }
}

/// Renders the show/close results button. "Show results" stays disabled
/// while a scan or analysis is running or while no analysis data exists.
fn create_result_items(app: &mut ImageDuplicatesApp, ui: &mut Ui, has_analysis_data: bool) {
    let analysing = app.analysing.load(Ordering::Relaxed);
    let scanning = app.scanning.load(Ordering::Relaxed);
    let display_results = app.display_results;

    match ResultsButton::select_state(display_results, analysing, scanning, has_analysis_data) {
        ResultsButton::CloseResults => {
            if ui.button("Close results").clicked() {
                app.close_results();
            }
        }
        ResultsButton::ShowResults => {
            if ui.button("Show results").clicked() {
                app.show_results();
            }
        }
        ResultsButton::None => {
            let hint = if analysing || scanning { "Wait for the running scan or analysis to finish." } else { "No analysis data yet: scan and analyse your folders first." };
            ui.add_enabled(false, egui::Button::new("Show results")).on_disabled_hover_text(hint);
        }
    }
}

/// Renders the result display: navigation, undo, the two images and their
/// correlation value.
pub fn create_result_display(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    if !app.display_results {
        return;
    }
    if app.first_path.is_empty() || app.second_path.is_empty() {
        app.display_results = false;
        return;
    }

    ui.horizontal(|ui| {
        if ui.button("Previous").clicked() {
            app.go_to_adjacent_entry(true);
        }
        ui.label(format!("Pair {} of {}", app.correlation_idx + 1, app.correlation_data.len()));
        if ui.button("Next").clicked() {
            app.go_to_adjacent_entry(false);
        }
        if let Some(corr) = app.current_correlation() {
            ui.label(format!("Resemblance: {:.2}%", corr * 100.0));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let undo = ui
                .add_enabled(!app.undo_history.is_empty(), egui::Button::new("Undo last action"))
                .on_hover_text("Restores the last moved or replaced image.")
                .on_disabled_hover_text("Nothing to undo yet.");
            if undo.clicked()
                && let Err(e) = app.undo_last()
            {
                error!("{e}");
            }
        });
        app.auto_advance_if_missing();
    });

    ui.label(RichText::new("Clicking an image moves it into the \"potential_duplicates\" folder. The button below the images replaces the smaller image with the larger one.").color(Color32::RED));

    display_images(app, ui);
}

/// Interaction that happened on a rendered image component in one frame.
#[derive(Debug, Default)]
struct ImageInteraction {
    /// The image itself was clicked.
    clicked: bool,
    /// The user requested opening the containing folder in the file manager.
    open_folder: bool,
}

/// Renders the two images side by side and the replace button below them.
fn display_images(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    let Some(((width1, height1), (width2, height2))) = app.current_image_dimensions() else {
        ui.label(RichText::new("Could not read one of the images.").color(Color32::RED));
        return;
    };
    let pixels1 = width1 * height1;
    let pixels2 = width2 * height2;
    let first_higher_res = pixels1 > pixels2;
    let second_higher_res = pixels1 < pixels2;

    let mut first = ImageInteraction::default();
    let mut second = ImageInteraction::default();
    ui.columns(2, |columns| {
        first = image_component(&mut columns[0], &app.first_path, (width1, height1), first_higher_res);
        second = image_component(&mut columns[1], &app.second_path, (width2, height2), second_higher_res);
    });

    let first_path = app.first_path.clone();
    let second_path = app.second_path.clone();
    if first.clicked
        && let Err(e) = app.move_to_duplicates(&first_path)
    {
        error!("{e}");
    }
    if second.clicked
        && let Err(e) = app.move_to_duplicates(&second_path)
    {
        error!("{e}");
    }
    if first.open_folder {
        app.open_containing_folder(&first_path);
    }
    if second.open_folder {
        app.open_containing_folder(&second_path);
    }

    // Replacement is only offered when both files have valid, matching
    // extensions; otherwise the button stays visible but disabled.
    let can_replace =
        has_valid_image_extension(Path::new(&first_path)) && has_valid_image_extension(Path::new(&second_path)) && have_matching_extensions(Path::new(&first_path), Path::new(&second_path));

    ui.horizontal(|ui| {
        let replacement = match pixels1.cmp(&pixels2) {
            Greater => Some((first_path, second_path, "Keep left, replace right")),
            Less => Some((second_path, first_path, "Keep right, replace left")),
            Equal => None,
        };
        match replacement {
            Some((source, target, label)) => {
                let button = ui
                    .add_enabled(can_replace, egui::Button::new(label))
                    .on_hover_text("Moves the higher-resolution image over the lower-resolution one. The replaced file is kept in the deleted folder and can be restored with \"Undo last action\".")
                    .on_disabled_hover_text("The two images have different file types, so one cannot replace the other. Click an image to move it to \"potential_duplicates\" instead.");
                if button.clicked()
                    && let Err(e) = app.replace_file(&source, &target)
                {
                    error!("{e}");
                }
            }
            None => {
                ui.add_enabled(false, egui::Button::new("Same resolution")).on_disabled_hover_text("Both images have the same resolution, so there is nothing to replace.");
            }
        }
    });
}

/// Builds the `file://` URI of `file_path` for the given platform. The
/// Windows file loader expects canonical drive URIs of the form
/// `file:///C:/dir/img.png` (forward slashes plus a leading slash after
/// the scheme); a naive `file://{path}` with backslashes is mistaken for a
/// UNC network path and the image fails to load. Only local drive paths
/// are supported: network paths are out of scope because the program
/// needs fast file access to work well.
#[must_use]
fn file_uri_for(file_path: &str, windows: bool) -> String {
    if !windows {
        return format!("file://{file_path}");
    }
    format!("file:///{}", file_path.replace('\\', "/"))
}

/// `file://` URI of `file_path` on the current platform.
#[must_use]
fn image_uri(file_path: &str) -> String {
    file_uri_for(file_path, cfg!(target_os = "windows"))
}

/// Renders one image with its file information. Returns the interactions
/// that happened on it in this frame.
fn image_component(ui: &mut Ui, file_path: &str, dimensions: (u32, u32), higher_res: bool) -> ImageInteraction {
    let file_name = Path::new(file_path).file_name().and_then(|name| name.to_str()).unwrap_or_default();
    let parent = Path::new(file_path).parent().unwrap_or_else(|| Path::new(""));
    let parent_folder = parent.file_name().and_then(|name| name.to_str()).unwrap_or_default();
    let parent_path = parent.display().to_string();

    // Space needed below each image: the three info labels plus the
    // replace-button row rendered underneath the image pair.
    let frame_margin = 4.0;
    let text_height = ui.text_style_height(&TextStyle::Body);
    let item_spacing = ui.spacing().item_spacing.y;
    let reserved_height = 4.0f32.mul_add(text_height + item_spacing, 12.0);
    let available_size = ui.available_size();
    let image = Image::new(image_uri(file_path))
        .fit_to_original_size(1.0)
        .max_width((available_size.x - frame_margin).max(0.0))
        .max_height((available_size.y - reserved_height).max(0.0))
        .sense(Sense::click());

    let mut interaction = ImageInteraction::default();

    ui.vertical(|ui| {
        let frame_fill = if higher_res { Color32::GREEN } else { Color32::TRANSPARENT };
        let frame = egui::Frame::default().inner_margin(2.0).fill(frame_fill).show(ui, |ui| ui.add(image));
        let hover_hint =
            if higher_res { "Higher resolution. Click to move this image into the \"potential_duplicates\" folder." } else { "Click to move this image into the \"potential_duplicates\" folder." };
        let image_response = frame.inner.on_hover_text(hover_hint);
        ui.label(shorten_string(file_name, 40)).on_hover_text(file_path);
        let folder_label = ui.label(format!("Parent folder: {}", shorten_string(parent_folder, 40)));
        folder_label.on_hover_ui(|ui| {
            ui.label(parent_path.as_str());
            if ui.button("Open in file manager").clicked() {
                interaction.open_folder = true;
            }
        });
        let (width, height) = dimensions;
        ui.label(format!("Image dimensions: {width}x{height}"));
        ImageInteraction { clicked: image_response.clicked(), open_folder: interaction.open_folder }
    })
    .inner
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]
    use {super::*, pretty_assertions::assert_eq};

    #[test]
    fn test_results_button_exclusive() {
        // While results are open, only "Close results" may be offered,
        // regardless of scanning/analysing state or available data.
        assert_eq!(ResultsButton::CloseResults, ResultsButton::select_state(true, false, false, false));
        assert_eq!(ResultsButton::CloseResults, ResultsButton::select_state(true, true, false, true));
        assert_eq!(ResultsButton::CloseResults, ResultsButton::select_state(true, false, true, false));
        assert_eq!(ResultsButton::CloseResults, ResultsButton::select_state(true, true, true, true));

        // While a scan or analysis is running, no button is offered.
        assert_eq!(ResultsButton::None, ResultsButton::select_state(false, true, false, true));
        assert_eq!(ResultsButton::None, ResultsButton::select_state(false, false, true, false));
        assert_eq!(ResultsButton::None, ResultsButton::select_state(false, true, true, true));

        // Idle without analysis data -> disabled, there is nothing to show.
        assert_eq!(ResultsButton::None, ResultsButton::select_state(false, false, false, false));

        // Idle with analysis data -> "Show results".
        assert_eq!(ResultsButton::ShowResults, ResultsButton::select_state(false, false, false, true));
    }

    #[test]
    fn test_workflow_hint_order() {
        // Without folders nothing else matters.
        assert_eq!(WorkflowHint::NoFolders, WorkflowHint::select_hint(false, false, false));
        assert_eq!(WorkflowHint::NoFolders, WorkflowHint::select_hint(false, true, true));

        // Folders without complete scan data always ask for a scan first.
        assert_eq!(WorkflowHint::NeedsScan, WorkflowHint::select_hint(true, false, false));
        assert_eq!(WorkflowHint::NeedsScan, WorkflowHint::select_hint(true, false, true));

        // Scan data without analysis data asks for an analysis.
        assert_eq!(WorkflowHint::NeedsAnalysis, WorkflowHint::select_hint(true, true, false));

        // With analysis data the results are ready.
        assert_eq!(WorkflowHint::ResultsReady, WorkflowHint::select_hint(true, true, true));
    }

    #[test]
    fn test_file_uri_unix() {
        assert_eq!("file:///home/user/image.png", file_uri_for("/home/user/image.png", false));
    }

    #[test]
    fn test_file_uri_windows_drive_path() {
        assert_eq!("file:///C:/Users/user/image.png", file_uri_for(r"C:\Users\user\image.png", true));
    }
}
