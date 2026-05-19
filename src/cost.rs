use ndarray::Array2;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{Error, Result};

/// A cost field over a regular grid.
///
/// Each cell value represents cost/slowness (`1 / speed`) in the eikonal
/// solver. A value of `0.0` marks the cell as impassable (barriers/obstacles).
/// Higher values make traversal more expensive or slow the front.
///
/// In the [`crate::graph`] shortest-path APIs, the same values are used as
/// destination-cell edge multipliers:
/// `geometric_distance * cost_at_destination`.
#[derive(Clone, Debug)]
pub struct CostField {
    pub(crate) data: Array2<f64>,
}

impl CostField {
    /// Create a cost field from an existing array.
    ///
    /// All values must be non-negative and finite. Zero means impassable.
    ///
    /// # Errors
    ///
    /// Returns an error if any value is negative or non-finite.
    pub fn from_array(data: Array2<f64>) -> Result<Self> {
        if data.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(Error::InvalidParameter(
                "cost field values must be finite and non-negative",
            ));
        }
        Ok(Self { data })
    }

    /// Create a uniform cost field (all cells have cost 1.0).
    pub fn uniform(height: usize, width: usize) -> Self {
        Self {
            data: Array2::ones((height, width)),
        }
    }

    /// Create a cost field with obstacles.
    ///
    /// `obstacles` is a boolean mask where `true` = impassable.
    /// Non-obstacle cells have cost 1.0.
    pub fn with_obstacles(obstacles: &Array2<bool>) -> Self {
        let data = obstacles.mapv(|blocked| if blocked { 0.0 } else { 1.0 });
        Self { data }
    }

    /// Build a cost field from terrain slope.
    ///
    /// Converts slope (in radians) to a cost multiplier: cost increases with
    /// steepness. Non-finite slope values become impassable (cost 0).
    ///
    /// # Arguments
    /// * `slope_rad` - Slope in radians (e.g., from `terrand::slope_radians`)
    /// * `base_cost` - Cost at zero slope (typically 1.0)
    /// * `slope_factor` - How strongly slope affects cost (0 = no effect, 5 = strong)
    ///
    /// # Errors
    ///
    /// Returns an error if `base_cost` or `slope_factor` is negative or
    /// non-finite, or if the computed costs overflow to non-finite values.
    ///
    /// With the `parallel` feature enabled, uses Rayon for large grids.
    pub fn from_slope(slope_rad: &Array2<f64>, base_cost: f64, slope_factor: f64) -> Result<Self> {
        if base_cost < 0.0 || !base_cost.is_finite() {
            return Err(Error::InvalidParameter(
                "base_cost must be finite and non-negative",
            ));
        }
        if slope_factor < 0.0 || !slope_factor.is_finite() {
            return Err(Error::InvalidParameter(
                "slope_factor must be finite and non-negative",
            ));
        }

        let (h, w) = slope_rad.dim();
        let compute = |s: f64| -> f64 {
            if !s.is_finite() {
                0.0
            } else {
                base_cost + slope_factor * s.tan().abs()
            }
        };

        #[cfg(feature = "parallel")]
        let data = {
            let raw: Vec<f64> = slope_rad
                .as_slice()
                .map(|slice| slice.par_iter().map(|&s| compute(s)).collect())
                .unwrap_or_else(|| {
                    (0..h * w)
                        .into_par_iter()
                        .map(|i| compute(slope_rad[[i / w, i % w]]))
                        .collect()
                });
            Array2::from_shape_vec((h, w), raw).unwrap()
        };

        #[cfg(not(feature = "parallel"))]
        let data = {
            let _ = (h, w); // suppress unused warning
            slope_rad.mapv(compute)
        };

        Self::from_array(data)
    }

    /// Dimensions as (height, width).
    pub fn dim(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Cost value at (row, col). Returns 0.0 for out-of-bounds.
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f64 {
        let (h, w) = self.data.dim();
        if row < h && col < w {
            self.data[[row, col]]
        } else {
            0.0
        }
    }

    /// Reference to the underlying array.
    pub fn as_array(&self) -> &Array2<f64> {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_all_ones() {
        let cf = CostField::uniform(10, 10);
        assert_eq!(cf.dim(), (10, 10));
        assert_eq!(cf.at(5, 5), 1.0);
    }

    #[test]
    fn from_array_valid() {
        let data = Array2::from_elem((5, 5), 2.0);
        assert!(CostField::from_array(data).is_ok());
    }

    #[test]
    fn from_array_negative_rejected() {
        let mut data = Array2::ones((5, 5));
        data[[2, 2]] = -1.0;
        assert!(CostField::from_array(data).is_err());
    }

    #[test]
    fn from_array_nan_rejected() {
        let mut data = Array2::ones((5, 5));
        data[[2, 2]] = f64::NAN;
        assert!(CostField::from_array(data).is_err());
    }

    #[test]
    fn obstacles_blocks_cells() {
        let mut obs = Array2::from_elem((5, 5), false);
        obs[[2, 2]] = true;
        let cf = CostField::with_obstacles(&obs);
        assert_eq!(cf.at(2, 2), 0.0);
        assert_eq!(cf.at(0, 0), 1.0);
    }

    #[test]
    fn from_slope_zero_is_base() {
        let slope = Array2::zeros((5, 5));
        let cf = CostField::from_slope(&slope, 1.0, 3.0).unwrap();
        assert!((cf.at(2, 2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn from_slope_steep_is_expensive() {
        let mut slope = Array2::zeros((5, 5));
        slope[[2, 2]] = 0.7; // ~40 degrees
        let cf = CostField::from_slope(&slope, 1.0, 2.0).unwrap();
        assert!(cf.at(2, 2) > cf.at(0, 0));
    }

    #[test]
    fn from_slope_nan_is_impassable() {
        let mut slope = Array2::zeros((5, 5));
        slope[[2, 2]] = f64::NAN;
        let cf = CostField::from_slope(&slope, 1.0, 2.0).unwrap();
        assert_eq!(cf.at(2, 2), 0.0);
    }

    #[test]
    fn from_slope_infinite_slope_is_impassable() {
        let mut slope = Array2::zeros((5, 5));
        slope[[2, 2]] = f64::INFINITY;
        slope[[3, 3]] = f64::NEG_INFINITY;
        let cf = CostField::from_slope(&slope, 1.0, 2.0).unwrap();
        assert_eq!(cf.at(2, 2), 0.0);
        assert_eq!(cf.at(3, 3), 0.0);
    }

    #[test]
    fn from_slope_invalid_base_cost_rejected() {
        let slope = Array2::zeros((5, 5));
        assert!(CostField::from_slope(&slope, -1.0, 2.0).is_err());
        assert!(CostField::from_slope(&slope, f64::INFINITY, 2.0).is_err());
        assert!(CostField::from_slope(&slope, f64::NAN, 2.0).is_err());
    }

    #[test]
    fn from_slope_invalid_slope_factor_rejected() {
        let slope = Array2::zeros((5, 5));
        assert!(CostField::from_slope(&slope, 1.0, -2.0).is_err());
        assert!(CostField::from_slope(&slope, 1.0, f64::INFINITY).is_err());
        assert!(CostField::from_slope(&slope, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn from_slope_overflow_rejected() {
        let slope = Array2::from_elem((5, 5), 1.0);
        assert!(CostField::from_slope(&slope, f64::MAX, f64::MAX).is_err());
    }
}
