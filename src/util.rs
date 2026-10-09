use {
    crate::constants::{POTENTIAL_DUPLICATES_FOLDER, SUPPORTED_IMAGE_FILE_EXTENSION},
    std::{
        fs,
        path::{Path, PathBuf},
    },
    tracing::error,
};

/// # Panics
/// This function panics, if given directory is being given for scanning, but is not existing or was removed.
/// Visits directories and will execute the callback on each of them. Can be used recursively.
pub fn visit_dirs(folder_path: &Path, recursive: bool, exclude: Option<&[String]>, callback: &mut dyn FnMut(&PathBuf)) {
    if folder_path.is_dir() {
        // Do not check our created folder
        if folder_path.display().to_string().contains(POTENTIAL_DUPLICATES_FOLDER) {
            return;
        }
        if let Some(excludes) = exclude {
            for exclude in excludes {
                if folder_path.display().to_string().starts_with(exclude) {
                    return;
                }
            }
        }

        match fs::read_dir(folder_path) {
            Ok(iter) => {
                for entry in iter {
                    let entry = entry.expect("Entry is valid.");
                    let path = entry.path();
                    if path.is_dir() && recursive {
                        visit_dirs(&path, recursive, exclude, callback);
                    } else {
                        callback(&path);
                    }
                }
            }
            Err(e) => {
                error!("error reading directory {:?}: {}", folder_path, e);
            }
        }
    }
}

/// Will shorten the string, so that the middle part is not displayed if it is too long.
#[must_use]
pub fn shorten_string(input_string: &str, max_length: usize) -> String {
    let input_length = input_string.len();
    if input_length <= max_length {
        return input_string.to_string();
    }

    let half_length = (max_length - 1) / 2; // -1 accounts for the ellipsis
    let start_slice = if max_length.is_multiple_of(2) { &input_string[..=half_length] } else { &input_string[..half_length] }; // When the desired size is even, we want to take one extra symbol from before the ellipsis.
    let end_slice = &input_string[input_length - half_length..];

    format!("{start_slice}…{end_slice}")
}

/// Will shorten the string if it is too long, otherwise pads it with spaces to force the desired length.
#[must_use]
pub fn force_string_length(input_string: &str, max_length: usize) -> String {
    let mut padded = input_string.to_string();
    let current_len = input_string.len();
    if current_len < max_length {
        let padding = max_length - current_len;
        padded.push_str(&" ".repeat(padding));
        return padded;
    }
    shorten_string(input_string, max_length)
}

/// Checks whether the file has a supported image extension (case-insensitive).
#[must_use]
pub fn has_valid_image_extension(file_path: &Path) -> bool {
    file_path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| SUPPORTED_IMAGE_FILE_EXTENSION.contains(&ext.to_ascii_lowercase().as_str()))
}

/// Checks whether two files have matching extensions (case-insensitive).
/// `jpg` and `jpeg` are the same format and therefore match each other.
#[must_use]
pub fn have_matching_extensions(file_path1: &Path, file_path2: &Path) -> bool {
    let normalize = |path: &Path| path.extension().and_then(|ext| ext.to_str()).map(|ext| if ext.eq_ignore_ascii_case("jpeg") { "jpg".to_string() } else { ext.to_ascii_lowercase() });
    match (normalize(file_path1), normalize(file_path2)) {
        (Some(ext1), Some(ext2)) => ext1 == ext2,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use {
        crate::util::{has_valid_image_extension, have_matching_extensions, shorten_string},
        pretty_assertions::assert_eq,
        std::path::Path,
    };

    #[test]
    fn test_str_short() {
        let s = "112233445566778899";
        let r = shorten_string(s, 20);
        assert_eq!(s, &r);
        let r = shorten_string(s, 5);
        assert_eq!("11…99", &r);
        let r = shorten_string(s, 6);
        assert_eq!("112…99", &r);
        let r = shorten_string(s, 7);
        assert_eq!("112…899", &r);
    }

    #[test]
    fn test_valid_image_extension() {
        assert!(has_valid_image_extension(Path::new("/some/folder/pic.png")));
        assert!(has_valid_image_extension(Path::new("pic.JPG"))); // case-insensitive
        assert!(has_valid_image_extension(Path::new("pic.jpeg")));
        assert!(has_valid_image_extension(Path::new("pic.webp")));
        assert!(!has_valid_image_extension(Path::new("pic.gif")));
        assert!(!has_valid_image_extension(Path::new("noext")));
        assert!(!has_valid_image_extension(Path::new("")));
    }

    #[test]
    fn test_matching_extensions() {
        assert!(have_matching_extensions(Path::new("a/img1.jpg"), Path::new("b/img2.jpg")));
        assert!(have_matching_extensions(Path::new("a/img1.jpg"), Path::new("b/img2.JPG"))); // case-insensitive
        assert!(have_matching_extensions(Path::new("a/img1.jpg"), Path::new("b/img2.jpeg"))); // same format
        assert!(have_matching_extensions(Path::new("a/img1.jpeg"), Path::new("b/img2.JPG"))); // same format, mixed case
        assert!(!have_matching_extensions(Path::new("a/img1.png"), Path::new("b/img2.jpg")));
        assert!(!have_matching_extensions(Path::new("a/img1"), Path::new("b/img2.jpg"))); // missing extension
        assert!(!have_matching_extensions(Path::new("a/img1.jpg"), Path::new("b/img2"))); // missing extension
        assert!(!have_matching_extensions(Path::new("a/img1"), Path::new("b/img2"))); // both missing
    }
}
