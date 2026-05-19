use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("point ({row}, {col}) is out of grid bounds ({height}x{width})")]
    OutOfBounds {
        row: usize,
        col: usize,
        height: usize,
        width: usize,
    },

    #[error("no path found between source and target")]
    NoPathFound,

    #[error("invalid parameter: {0}")]
    InvalidParameter(&'static str),

    #[error("dimension mismatch: expected ({eh}x{ew}), got ({gh}x{gw})")]
    DimensionMismatch {
        eh: usize,
        ew: usize,
        gh: usize,
        gw: usize,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
