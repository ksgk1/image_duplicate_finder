use {
    crate::{
        constants::{POTENTIAL_DUPLICATES_FOLDER, UI_SCALING_FACTOR}, data::{self, CorrelationEntry, DataEntry, SUPPORTED_IMAGE_FILE_EXTENSION}, progress::LockFreeProgress, util::{force_string_length, shorten_string}, Direction,
        ImageDuplicatesApp,
        AUTO_FORWARD_DIRECTION,
        MAX_WORKERS,
        UNDO_LIST,
    },
    eframe::egui::{self, mutex::Mutex, Color32, Image, InnerResponse, RichText, Sense, Ui, Vec2},
    std::{
        cmp::Ordering::{Equal, Greater, Less},
        fs,
        path::Path,
        sync::{atomic::Ordering, Arc},
    },
    tracing::{debug, error, info},
};

pub fn create_folder_selection_block(app: &mut ImageDuplicatesApp, ui: &mut Ui) -> InnerResponse<()> {
    let longest_folder_name = app.folder_paths.iter().map(String::len).max().unwrap();

    ui.vertical(|ui| {
        let mut break_loop = false; // cannot break loop from inside ui element
        for (idx, folder) in app.folder_paths.clone().iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label("Selected folder:");
                let shortened_folder_path = force_string_length(&folder.clone(), longest_folder_name.min(50));
                ui.label(RichText::new(shortened_folder_path).monospace());
                ui.add_enabled(!*app.exclude_paths.get(idx).unwrap(), egui::Checkbox::new(app.recursive_paths.get_mut(idx).unwrap(), "Recursive scanning"));
                if ui.checkbox(app.exclude_paths.get_mut(idx).unwrap(), "Exclude from scanning").clicked() {
                    *app.recursive_paths.get_mut(idx).unwrap() = false; // deactivate recursive when exclude is selected
                }
                if ui.button("remove").clicked() {
                    let _ = &app.recursive_paths.remove(idx); // can not fail, we get the index from the iterator
                    let _ = &app.exclude_paths.remove(idx);
                    let _ = &app.folder_paths.remove(idx);
                    break_loop = true;
                }
            });
            if break_loop {
                break; // break loop because indices no longer align after removing an element
            }
        }
    })
}

pub fn create_result_items(app: &ImageDuplicatesApp, ui: &mut Ui, can_analyse: bool, can_show_results: bool) {
    let analysing = app.analysing.load(Ordering::Relaxed);
    let scanning = app.scanning.load(Ordering::Relaxed);
    if !analysing && !scanning && can_analyse {
        if app.display_images.load(Ordering::Relaxed) && ui.button("Close results").clicked() {
            app.correlation_data.lock().clear();
            app.display_images.store(false, Ordering::Relaxed);
        } else if can_show_results && ui.button("Show results").clicked() {
            for folder in app.folder_paths.clone() {
                #[allow(clippy::collection_is_never_read)]
                let mut folder_correlation_data: Vec<CorrelationEntry> = CorrelationEntry::read_correlation_db(&folder).unwrap().iter().filter(|e| e.corr > 0.9).cloned().collect();
                if !app.find_image_path.lock().is_empty() {
                    let comparison_file = app.find_image_path.lock().clone();
                    folder_correlation_data.retain(|item| item.first_path == comparison_file || item.second_path == comparison_file);
                }
                folder_correlation_data.clone_into(&mut app.correlation_data.lock());
            }

            if app.correlation_data.lock().is_empty() {
                ui.label(format!("No correlation data found for Image: {}", app.find_image_path.lock().clone()));
            }

            app.display_images.store(true, Ordering::Relaxed);
            {
                let c_data = app.correlation_data.lock();
                if let Some(entry) = c_data.get(app.correlation_entry_idx.load(Ordering::Relaxed)) {
                    app.first_path.lock().clone_from(&entry.first_path);
                    app.second_path.lock().clone_from(&entry.second_path);
                } else {
                    app.first_path.lock().clone_from(&String::new());
                    app.second_path.lock().clone_from(&String::new());
                }
                drop(c_data);
            }
        }
    }
}

pub fn create_result_display(app: &ImageDuplicatesApp, ui: &mut Ui, image_clicked: &Arc<Mutex<String>>) {
    if app.display_images.load(Ordering::Relaxed) {
        ui.horizontal(|ui| {
            previous_button(app, ui);
            next_button(app, ui);
            let first_path_exists = Path::new(&app.first_path.lock().clone()).exists();
            let second_path_exists = Path::new(&app.second_path.lock().clone()).exists();
            if !first_path_exists || !second_path_exists {
                let mut idx = app.correlation_entry_idx.load(Ordering::Relaxed);
                let mut direction = AUTO_FORWARD_DIRECTION.lock();
                match *direction {
                    Direction::Forwards => {
                        if idx < (app.correlation_data.lock().len() - 1) {
                            idx += 1;
                        } else {
                            idx = 0;
                        }
                    }
                    Direction::Backwards => {
                        if idx > 0 {
                            idx -= 1;
                        } else {
                            idx = app.correlation_data.lock().len() - 1;
                        }
                    }
                    Direction::None => {
                        if idx < (app.correlation_data.lock().len() - 1) {
                            idx += 1;
                        } else {
                            idx = 0;
                        }
                        *direction = Direction::Forwards;
                    }
                }
                drop(direction);
                app.correlation_entry_idx.store(idx, Ordering::Relaxed);
                set_image_paths(app, idx);
            }
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Clicking on the images moves them into the \"potential_duplicates\" folder. Clicking the button will replace the smaller one.").color(Color32::RED));
            {
                let mut undo_list = UNDO_LIST.lock();
                if !undo_list.is_empty() && ui.button("Restore last move").clicked() {
                    match undo_list.pop() {
                        Some(original_path) => {
                            let duplicate_file_name = Path::new(&original_path).file_name().unwrap().to_str().unwrap();
                            let selected_folder = Path::new(&original_path).parent().expect("Could not get dir of file").display().to_string();
                            let duplicate_file_path = format!("{selected_folder}/{POTENTIAL_DUPLICATES_FOLDER}/{duplicate_file_name}");
                            match fs::rename(duplicate_file_path, &original_path) {
                                Ok(()) => info!("Restored {original_path}"),
                                Err(e) => {
                                    error!("Failed to restore {original_path}. Error: {e}");
                                }
                            }
                        }
                        None => {
                            debug!("Could not get item from undo list");
                        }
                    }
                }
            }
        });

        let first_path = &app.first_path.lock().clone();
        let second_path = &app.second_path.lock().clone();

        if first_path.is_empty() || second_path.is_empty() {
            app.display_images.store(false, Ordering::Relaxed);
        }

        let image1 = Image::new(format!("file://{}", &first_path));
        let image2 = Image::new(format!("file://{}", &second_path));

        let available_size = ui.available_size();
        display_images(ui, image_clicked, first_path, second_path, image1, image2, available_size);

        {
            let c_data = app.correlation_data.lock();
            if let Some(corr) = &c_data.get(app.correlation_entry_idx.load(Ordering::Relaxed)) {
                ui.label(format!("Resemblance: {:.2}%", corr.corr * 100.0));
            }
            drop(c_data);
        }
    }
}

fn set_image_paths(app: &ImageDuplicatesApp, idx: usize) {
    {
        let c_data = app.correlation_data.lock();
        if let Some(entry) = c_data.get(idx) {
            app.first_path.lock().clone_from(&entry.first_path);
            app.second_path.lock().clone_from(&entry.second_path);
        }
        drop(c_data);
    }
}

fn display_images(ui: &mut Ui, image_clicked: &Arc<Mutex<String>>, first_path: &String, second_path: &String, image1: Image, image2: Image, available_size: Vec2) {
    ui.horizontal(|ui| {
        // cannot calculate the sizing inside, since it changes after the first image is inserted
        if first_path.is_empty() {
            info!("First path is empty, nothing to display");
            return;
        }
        let (w1, h1) = match image::ImageReader::open(first_path) {
            Ok(image) => image.into_dimensions().unwrap_or_default(),
            Err(e) => {
                error!("Error while reading file {first_path}: {e:?}");
                (0, 0)
            }
        };

        if second_path.is_empty() {
            info!("Second path is empty, nothing to display");
            return;
        }
        let (w2, h2) = match image::ImageReader::open(second_path) {
            Ok(image) => image.into_dimensions().unwrap_or_default(),
            Err(e) => {
                error!("Error while reading file {second_path}: {e:?}");
                (0, 0)
            }
        };

        let mut higher_res_1 = false;
        let mut higher_res_2 = false;
        let dimensions_comparison = (w1 * h1).cmp(&(w2 * h2));
        match dimensions_comparison {
            Less => {
                higher_res_2 = true;
            }
            Equal => {}
            Greater => {
                higher_res_1 = true;
            }
        }

        // 4 lines of text, each line is around 12 px in height.
        // Removing 25 px for the button between them
        let image_info_height = 4.0 * 12.0 * UI_SCALING_FACTOR;
        create_ui_image_component(ui, first_path, image1, (available_size.x / 2.0 - 25.0, available_size.y - image_info_height), image_clicked, /*&selected_folder,*/ higher_res_1);
        match dimensions_comparison {
            Less => {
                if ui.button("<-").clicked() {
                    info!("Overwriting {second_path} → {first_path}");
                    fs::rename(second_path, first_path).expect("Can not fail, unless files were altered in the meantime.");
                    // TODO: Remove entries from database
                }
            }
            Equal => {
                ui.label("  ");
            }
            Greater => {
                if ui.button("->").clicked() {
                    info!("Overwriting {first_path} → {second_path}");
                    fs::rename(first_path, second_path).expect("Can not fail, unless files were altered in the meantime.");
                    // TODO: Remove entries from database
                }
            }
        }
        create_ui_image_component(ui, second_path, image2, (available_size.x / 2.0 - 25.0, available_size.y - image_info_height), image_clicked, /*&selected_folder,*/ higher_res_2);
    });
}

fn previous_button(app: &ImageDuplicatesApp, ui: &mut Ui) {
    if ui.button("Previous").clicked() {
        let mut idx = app.correlation_entry_idx.load(Ordering::Relaxed);
        *AUTO_FORWARD_DIRECTION.lock() = Direction::Backwards;
        if idx > 0 {
            idx -= 1;
        } else {
            idx = app.correlation_data.lock().len() - 1;
        }
        app.correlation_entry_idx.store(idx, Ordering::Relaxed);

        set_image_paths(app, idx);
    }
}

fn next_button(app: &ImageDuplicatesApp, ui: &mut Ui) {
    if ui.button("Next").clicked() {
        let mut idx = app.correlation_entry_idx.load(Ordering::Relaxed);
        *AUTO_FORWARD_DIRECTION.lock() = Direction::Forwards;
        if idx < (app.correlation_data.lock().len() - 1) {
            idx += 1;
        } else {
            idx = 0;
        }
        app.correlation_entry_idx.store(idx, Ordering::Relaxed);

        set_image_paths(app, idx);
    }
}

pub fn create_ui_image_component(ui: &mut Ui, file_path: &str, image: Image, max_size: (f32, f32), to_delete: &Arc<Mutex<String>>, higher_res: bool) {
    if file_path.is_empty() {
        info!("File path is empty");
        return;
    }
    let filename = Path::new(&file_path).file_name().unwrap_or_default().to_str().unwrap_or_default();
    let display_filename = shorten_string(filename, 37);
    let scaled_image = image.fit_to_original_size(1.0).max_width(max_size.0).max_height(max_size.1).sense(Sense::click());
    let fallback_folder = Path::new(&file_path).parent().expect("Could not get dirname from file").display().to_string();
    let root_folder = Path::new(&to_delete.lock().to_string()).parent().unwrap_or_else(|| Path::new(&fallback_folder)).display().to_string();
    let duplicate_folder_path = format!("{root_folder}/{POTENTIAL_DUPLICATES_FOLDER}");
    ui.vertical(|ui| {
        match image::ImageReader::open(file_path) {
            Ok(image) => {
                let (width, height) = image.into_dimensions().expect("Could not get dimensions");
                let stroke_colour = if higher_res { Color32::GREEN } else { Color32::TRANSPARENT };
                egui::Frame::default().inner_margin(2.0).fill(stroke_colour).show(ui, |ui| {
                    if ui.add(scaled_image).clicked() {
                        if !Path::new(&duplicate_folder_path).exists() {
                            let _ = fs::create_dir(&duplicate_folder_path);
                        }
                        {
                            let mut u_list = UNDO_LIST.lock();
                            u_list.push(file_path.to_string());
                        }
                        let _ = fs::rename(file_path, format!("{duplicate_folder_path}/{filename}"));
                        let mut binding = to_delete.lock();
                        *binding = file_path.to_string();
                    }
                });
                ui.label(display_filename);
                ui.label(format!("Parent folder: {}", shorten_string(Path::new(file_path).parent().unwrap().file_name().unwrap().to_str().unwrap(), 30)));
                ui.label(format!("Image dimensions: {width}x{height}"));
            }
            Err(e) => {
                /*eprint!("Could not open image {e}")*/
                drop(e);
            }
        }
    });
}

pub fn create_scanning_controls(app: &mut ImageDuplicatesApp, ui: &mut Ui, can_analyse: bool) {
    let selected_folders = app.folder_paths.clone();
    let excluded_folders = app.exclude_paths.clone();

    ui.horizontal(|ui| {
        if app.scanning.load(Ordering::Relaxed) {
            if ui.button("Stop scanning").clicked() {
                app.scanning.store(false, Ordering::Relaxed);
            }
        } else if ui.button("Start scanning").clicked() {
            let scanning = Arc::clone(&app.scanning);
            scanning.store(true, Ordering::Relaxed);
            let analysing = Arc::clone(&app.analysing);
            analysing.store(false, Ordering::Relaxed);

            let folders = selected_folders.clone();
            let excludes: Vec<String> = selected_folders.iter().zip(excluded_folders.iter()).filter_map(|(folder, &is_excluded)| if is_excluded { Some(folder.clone()) } else { None }).collect();
            let recursive = app.recursive_paths.clone();
            let use_workers = app.workers;

            let mut files_to_scan_list = Vec::<String>::new();
            for (folder, &is_recursive) in folders.iter().zip(recursive.iter()) {
                let mut file_list = DataEntry::generate_file_list(folder, is_recursive, None);
                files_to_scan_list.append(&mut file_list);
            }

            let progress_tracker = Arc::new(LockFreeProgress::new(files_to_scan_list.len()));
            app.scan_progress = Some(Arc::clone(&progress_tracker));

            std::thread::spawn(move || {
                DataEntry::scan_folders_with_progress(&folders, if excludes.is_empty() { None } else { Some(&excludes) }, &scanning, &recursive, &progress_tracker, use_workers);
            });
        }

        if app.analysing.load(Ordering::Relaxed) {
            if ui.button("Stop analysing").clicked() {
                app.analysing.store(false, Ordering::Relaxed);
            }
        } else if can_analyse && !app.scanning.load(Ordering::Relaxed) && ui.button("Analyse data").clicked() {
            let analysing = Arc::clone(&app.analysing);
            analysing.store(true, Ordering::Relaxed);
            let scanning = Arc::clone(&app.scanning);
            scanning.store(false, Ordering::Relaxed);

            let folders = selected_folders.clone();
            let use_workers = app.workers;
            let data_entries = DataEntry::read_from_folder(&folders[0]).unwrap_or_default();
            let max_possible_combinations = if data_entries.len() > 1 { data_entries.len() * (data_entries.len() - 1) / 2 } else { 1 };

            let progress_tracker = Arc::new(LockFreeProgress::new(max_possible_combinations));
            app.analysis_progress = Some(Arc::clone(&progress_tracker));

            std::thread::spawn(move || {
                CorrelationEntry::analyse_with_progress_chunked(&folders[0], &analysing, &progress_tracker, use_workers, 1_000_000_000);
            });
        }

        let max_workers = MAX_WORKERS.load(Ordering::Relaxed);
        ui.add(egui::Slider::new(&mut app.workers, 1..=max_workers).text("Max workers")).on_hover_ui(|ui| {
            ui.label("Select the number of maximum workers to be used for scanning/analysing.");
        });

        if !selected_folders.is_empty() {
            let correlation_db_path = data::get_db_path::<CorrelationEntry>(&selected_folders[0]);
            if fs::exists(&correlation_db_path).unwrap_or(false) && ui.button("Delete analysis data").clicked() {
                match fs::remove_file(&correlation_db_path) {
                    Ok(()) => {
                        info!("Successfully removed analysis data");
                    }
                    Err(e) => error!("Could not remove analysis data: {}", e),
                }
                app.analysing.store(false, Ordering::Relaxed);
            }
        }
    });
}

pub fn create_comparison_file_element(app: &ImageDuplicatesApp, ui: &mut Ui) {
    ui.horizontal(|ui| {
        if ui.button("Select comparison image…").clicked()
            && let Some(path) = rfd::FileDialog::new().add_filter("Images", &SUPPORTED_IMAGE_FILE_EXTENSION).pick_file()
        {
            app.find_image_path.lock().push_str(&path.display().to_string());
        }
        let selected_image = shorten_string(app.find_image_path.lock().clone().as_str(), 50);
        ui.label(format!("Selected image: {selected_image}"));
        if ui.button("Clear").clicked() {
            app.find_image_path.lock().clear();
        }
    });
}
