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
        constants::{SUPPORTED_IMAGE_FILE_EXTENSION, UI_SCALING_FACTOR},
        data::{self, CorrelationEntry},
        util::{force_string_length, has_valid_image_extension, have_matching_extensions, shorten_string},
    },
    eframe::egui::{self, Color32, Image, RichText, Sense, Ui},
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
#[derive(Debug, PartialEq, Eq)]
enum ResultsButton {
    CloseResults,
    ShowResults,
    None,
}

/// Decides which results button to render: while results are open only
/// "Close results" is offered, while a scan or analysis is running no button
/// is offered, otherwise "Show results".
const fn results_button_state(display_results: bool, analysing: bool, scanning: bool) -> ResultsButton {
    if display_results {
        ResultsButton::CloseResults
    } else if !analysing && !scanning {
        ResultsButton::ShowResults
    } else {
        ResultsButton::None
    }
}

/// Renders the list of selected folders with their options and remove button.
///
/// # Panics
/// If called without any folder selected.
pub fn create_folder_selection_block(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    let longest_folder_name = app.folders.iter().map(|folder| folder.path.len()).max().expect("Called only with at least one folder selected");

    ui.vertical(|ui| {
        let mut remove_idx = None;
        for (idx, folder) in app.folders.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label("Selected folder:");
                ui.label(RichText::new(force_string_length(&folder.path, longest_folder_name.min(50))).monospace());
                ui.add_enabled(!folder.excluded, egui::Checkbox::new(&mut folder.recursive, "Recursive scanning"));
                if ui.checkbox(&mut folder.excluded, "Exclude from scanning").changed() && folder.excluded {
                    folder.recursive = false; // deactivate recursive when the folder is excluded
                }
                if ui.button("Remove").clicked() {
                    remove_idx = Some(idx);
                }
            });
            if remove_idx.is_some() {
                break; // indices no longer align after removing an element
            }
        }
        if let Some(idx) = remove_idx {
            app.folders.remove(idx);
        }
    });
}

/// Renders the scan/analyse controls, the worker slider and the
/// "Delete analysis data" button.
pub fn create_scanning_controls(app: &mut ImageDuplicatesApp, ui: &mut Ui, can_analyse: bool) {
    ui.horizontal(|ui| {
        if app.scanning.load(Ordering::Relaxed) {
            if ui.button("Stop scanning").clicked() {
                app.stop_scan();
            }
        } else if ui.button("Start scanning").clicked() {
            app.start_scan();
        }

        if app.analysing.load(Ordering::Relaxed) {
            if ui.button("Stop analysing").clicked() {
                app.stop_analysis();
            }
        } else if can_analyse && !app.scanning.load(Ordering::Relaxed) && ui.button("Analyse data").clicked() {
            app.start_analysis();
        }

        ui.add(egui::Slider::new(&mut app.workers, 1..=app.max_workers).text("Max workers")).on_hover_ui(|ui| {
            ui.label("Select the number of maximum workers to be used for scanning/analysing.");
        });

        let correlation_db_path = app.folders.first().map(|folder| data::get_db_path::<CorrelationEntry>(&folder.path));
        if let Some(db_path) = correlation_db_path
            && fs::exists(&db_path).unwrap_or(false)
            && ui.button("Delete analysis data").clicked()
        {
            app.delete_analysis_db();
        }
    });
}

/// Renders the comparison image selection, filtering the displayed results
/// to entries involving the selected image.
pub fn create_comparison_file_element(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    ui.horizontal(|ui| {
        if ui.button("Select comparison image…").clicked()
            && let Some(path) = rfd::FileDialog::new().add_filter("Images", &SUPPORTED_IMAGE_FILE_EXTENSION).pick_file()
            && let Err(e) = app.select_comparison_image(&path.display().to_string())
        {
            error!("{e}");
        }
        let selected_image = shorten_string(&app.find_image_path, 50);
        ui.label(format!("Selected image: {selected_image}"));
        if ui.button("Clear").clicked() {
            app.clear_comparison_image();
        }
    });
}

/// Renders the show/close results button and the last status message, if any.
pub fn create_result_items(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    let analysing = app.analysing.load(Ordering::Relaxed);
    let scanning = app.scanning.load(Ordering::Relaxed);
    let display_results = app.display_results;

    match results_button_state(display_results, analysing, scanning) {
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
        ResultsButton::None => {}
    }

    if let Some(status) = &app.status {
        ui.label(RichText::new(status.clone()).color(Color32::RED));
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
        if ui.button("Next").clicked() {
            app.go_to_adjacent_entry(false);
        }
        app.auto_advance_if_missing();
    });

    ui.horizontal(|ui| {
        ui.label(RichText::new("Clicking on the images moves them into the \"potential_duplicates\" folder. Clicking the button will replace the smaller one.").color(Color32::RED));
        if !app.undo_history.is_empty()
            && ui.button("Undo last action").clicked()
            && let Err(e) = app.undo_last()
        {
            error!("{e}");
        }
    });

    display_images(app, ui);

    if let Some(corr) = app.current_correlation() {
        ui.label(format!("Resemblance: {:.2}%", corr * 100.0));
    }
}

/// Renders the two images side by side with the replace button between them.
fn display_images(app: &mut ImageDuplicatesApp, ui: &mut Ui) {
    let Some(((width1, height1), (width2, height2))) = app.current_image_dimensions() else {
        ui.label(RichText::new("Could not read one of the images.").color(Color32::RED));
        return;
    };
    let pixels1 = width1 * height1;
    let pixels2 = width2 * height2;
    let first_higher_res = pixels1 > pixels2;
    let second_higher_res = pixels1 < pixels2;

    // 4 lines of text, each around 12 px in height, are rendered below each
    // image; the width gap of 25 px hosts the replace button.
    let image_info_height = 4.0 * 12.0 * UI_SCALING_FACTOR;
    let available_size = ui.available_size();
    let max_image_size = (available_size.x / 2.0 - 25.0, available_size.y - image_info_height);

    ui.horizontal(|ui| {
        let clicked_first = image_component(ui, &app.first_path, Image::new(format!("file://{}", app.first_path)), max_image_size, (width1, height1), first_higher_res);
        if clicked_first {
            let path = app.first_path.clone();
            if let Err(e) = app.move_to_duplicates(&path) {
                error!("{e}");
            }
        }

        // Check if both files have valid extensions before allowing replacement
        let first_has_valid_ext = has_valid_image_extension(Path::new(&app.first_path));
        let second_has_valid_ext = has_valid_image_extension(Path::new(&app.second_path));
        let extensions_match = have_matching_extensions(Path::new(&app.first_path), Path::new(&app.second_path));
        let can_replace = first_has_valid_ext && second_has_valid_ext && extensions_match;

        match pixels1.cmp(&pixels2) {
            Greater => {
                if can_replace && ui.button("->").clicked() {
                    let (source, target) = (app.first_path.clone(), app.second_path.clone());
                    if let Err(e) = app.replace_file(&source, &target) {
                        error!("{e}");
                    }
                }
            }
            Less => {
                if can_replace && ui.button("<-").clicked() {
                    let (source, target) = (app.second_path.clone(), app.first_path.clone());
                    if let Err(e) = app.replace_file(&source, &target) {
                        error!("{e}");
                    }
                }
            }
            Equal => {
                ui.add_space(25.0);
            }
        }

        let clicked_second = image_component(ui, &app.second_path, Image::new(format!("file://{}", app.second_path)), max_image_size, (width2, height2), second_higher_res);
        if clicked_second {
            let path = app.second_path.clone();
            if let Err(e) = app.move_to_duplicates(&path) {
                error!("{e}");
            }
        }
    });
}

/// Renders one image with its file information. Returns whether the image
/// was clicked.
fn image_component(ui: &mut Ui, file_path: &str, image: Image, max_size: (f32, f32), dimensions: (u32, u32), higher_res: bool) -> bool {
    let file_name = Path::new(file_path).file_name().and_then(|name| name.to_str()).unwrap_or_default();
    let parent_folder = Path::new(file_path).parent().and_then(Path::file_name).and_then(|name| name.to_str()).unwrap_or_default();
    let scaled_image = image.fit_to_original_size(1.0).max_width(max_size.0).max_height(max_size.1).sense(Sense::click());

    ui.vertical(|ui| {
        let frame_fill = if higher_res { Color32::GREEN } else { Color32::TRANSPARENT };
        let response = egui::Frame::default().inner_margin(2.0).fill(frame_fill).show(ui, |ui| ui.add(scaled_image));
        ui.label(shorten_string(file_name, 37));
        ui.label(format!("Parent folder: {}", shorten_string(parent_folder, 30)));
        let (width, height) = dimensions;
        ui.label(format!("Image dimensions: {width}x{height}"));
        response.inner.clicked()
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
        // regardless of scanning/analysing state.
        assert_eq!(ResultsButton::CloseResults, results_button_state(true, false, false));
        assert_eq!(ResultsButton::CloseResults, results_button_state(true, true, false));
        assert_eq!(ResultsButton::CloseResults, results_button_state(true, false, true));
        assert_eq!(ResultsButton::CloseResults, results_button_state(true, true, true));

        // While a scan or analysis is running, no button is offered.
        assert_eq!(ResultsButton::None, results_button_state(false, true, false));
        assert_eq!(ResultsButton::None, results_button_state(false, false, true));
        assert_eq!(ResultsButton::None, results_button_state(false, true, true));

        // Idle without results open -> "Show results".
        assert_eq!(ResultsButton::ShowResults, results_button_state(false, false, false));
    }
}
