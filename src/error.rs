use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Error while trying to read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("For the folder `{0}` no valid files could be found")]
    EmptyFileList(String),
    #[error("Calculation error for file `{0}`")]
    TileCalculation(String),
}
