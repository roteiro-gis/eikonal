use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::Array2;

use crate::cost::CostField;
use crate::error::{Error, Result};

/// Configuration for terrain-aware solving.
#[derive(Clone, Copy, Debug)]
pub struct TerrainConfig {
    /// Cell size in map units (meters). Must be positive.
    pub cell_size: f64,
    /// Uphill cost multiplier. 1.0 = no penalty, >1.0 = penalize ascent.
    pub uphill_factor: f64,
    /// Downhill cost multiplier. 1.0 = no penalty, >1.0 = penalize descent.
    pub downhill_factor: f64,
}

impl TerrainConfig {
    /// Symmetric terrain config (no directional preference).
    pub fn symmetric(cell_size: f64) -> Self {
        Self {
            cell_size,
            uphill_factor: 1.0,
            downhill_factor: 1.0,
        }
    }

    /// Hiking-optimized config: uphill is harder, downhill is slightly easier.
    pub fn hiking(cell_size: f64) -> Self {
        Self {
            cell_size,
            uphill_factor: 2.0,
            downhill_factor: 0.8,
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err(Error::InvalidParameter(
                "cell_size must be finite and positive",
            ));
        }
        if !self.uphill_factor.is_finite() || self.uphill_factor <= 0.0 {
            return Err(Error::InvalidParameter(
                "uphill_factor must be finite and positive",
            ));
        }
        if !self.downhill_factor.is_finite() || self.downhill_factor <= 0.0 {
            return Err(Error::InvalidParameter(
                "downhill_factor must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Result of terrain-aware solving.
///
/// Provides the distance field and terrain-aware path extraction that includes
/// 3D surface distance, elevation profiles, and ascent/descent statistics.
pub struct TerrainResult {
    pub(crate) distance: Array2<f64>,
    pub(crate) predecessors: Vec<Option<usize>>,
    pub(crate) finalized: Vec<bool>,
    pub(crate) dem: Array2<f64>,
    pub(crate) cell_size: f64,
    pub(crate) width: usize,
}

impl TerrainResult {
    /// The distance (minimum cumulative cost) from the nearest source to each
    /// finalized cell. Unfinished early-termination cells have `f64::INFINITY`.
    pub fn distance(&self) -> &Array2<f64> {
        &self.distance
    }

    /// Extract the shortest path to `(target_row, target_col)` with full
    /// terrain statistics: surface distance, elevation profile, ascent, and
    /// descent.
    ///
    /// Returns `NoPathFound` if the target is unreachable or was not finalized
    /// by an early-terminated solve.
    pub fn path_to(&self, target_row: usize, target_col: usize) -> Result<TerrainPath> {
        let (h, _) = self.distance.dim();
        let w = self.width;
        if target_row >= h || target_col >= w {
            return Err(Error::OutOfBounds {
                row: target_row,
                col: target_col,
                height: h,
                width: w,
            });
        }

        let idx = target_row * w + target_col;
        let cost = self.distance[[target_row, target_col]];
        if cost.is_infinite() || !self.finalized[idx] {
            return Err(Error::NoPathFound);
        }

        let mut cells = Vec::new();
        let mut idx = idx;
        loop {
            let r = idx / w;
            let c = idx % w;
            cells.push((r, c));
            if let Some(pred) = self.predecessors[idx] {
                idx = pred;
            } else {
                break;
            }
        }
        cells.reverse();

        let mut surface_distance = 0.0;
        let mut projected_distance = 0.0;
        let mut total_ascent = 0.0;
        let mut total_descent = 0.0;
        let mut elevation_profile = Vec::with_capacity(cells.len());

        let (r0, c0) = cells[0];
        elevation_profile.push((0.0, self.dem[[r0, c0]]));

        for i in 1..cells.len() {
            let (r0, c0) = cells[i - 1];
            let (r1, c1) = cells[i];

            let dr = (r1 as f64 - r0 as f64) * self.cell_size;
            let dc = (c1 as f64 - c0 as f64) * self.cell_size;
            let dz = self.dem[[r1, c1]] - self.dem[[r0, c0]];

            let horiz = (dr * dr + dc * dc).sqrt();
            projected_distance += horiz;
            surface_distance += (horiz * horiz + dz * dz).sqrt();

            if dz > 0.0 {
                total_ascent += dz;
            } else {
                total_descent += dz.abs();
            }

            elevation_profile.push((surface_distance, self.dem[[r1, c1]]));
        }

        Ok(TerrainPath {
            cells,
            cost,
            surface_distance,
            projected_distance,
            elevation_profile,
            total_ascent,
            total_descent,
        })
    }
}

/// A shortest path on terrain with elevation statistics.
#[derive(Clone, Debug)]
pub struct TerrainPath {
    /// Sequence of (row, col) from source to target.
    pub cells: Vec<(usize, usize)>,
    /// Total weighted cost (includes slope penalties and cost field).
    pub cost: f64,
    /// 3D surface distance along the path.
    pub surface_distance: f64,
    /// 2D projected (horizontal) distance.
    pub projected_distance: f64,
    /// Elevation profile: `(cumulative_surface_distance, elevation)` at each cell.
    pub elevation_profile: Vec<(f64, f64)>,
    /// Total elevation gained (meters ascended).
    pub total_ascent: f64,
    /// Total elevation lost (meters descended).
    pub total_descent: f64,
}

/// Solve weighted shortest paths on terrain with 3D surface distance.
///
/// This is an 8-neighbor graph-routing solver, not the eikonal FMM solver.
/// It accounts for elevation by making the geometric distance between adjacent
/// cells the 3D surface distance
/// `sqrt(horizontal² + dz²)`, and optional slope penalties make uphill or
/// downhill traversal more expensive.
///
/// Non-finite DEM elevations (`NaN`, `+inf`, `-inf`) are treated as nodata
/// barriers. Routes never enter them, and source/early-target cells must have
/// finite elevation.
/// Optional scalar cost-field multipliers are averaged across each edge's two
/// endpoint cells. Uphill/downhill factors can still make terrain routing
/// direction-dependent when they differ.
///
/// # Arguments
/// * `dem` - Digital Elevation Model as a 2D array
/// * `config` - Terrain parameters (cell size, slope penalties)
/// * `cost_field` - Optional additional cost multiplier (e.g., land cover). `None` = uniform.
/// * `source` - Source cell as `(row, col)`
pub fn solve(
    dem: &Array2<f64>,
    config: TerrainConfig,
    cost_field: Option<&CostField>,
    source: (usize, usize),
) -> Result<TerrainResult> {
    solve_multi(dem, config, cost_field, &[source])
}

/// Solve from multiple terrain sources simultaneously.
///
/// # Errors
///
/// Returns `InvalidParameter` if `sources` is empty, if any source is on a
/// non-finite DEM cell, or if any source is blocked by `cost_field`.
pub fn solve_multi(
    dem: &Array2<f64>,
    config: TerrainConfig,
    cost_field: Option<&CostField>,
    sources: &[(usize, usize)],
) -> Result<TerrainResult> {
    solve_terrain_inner(dem, config, cost_field, sources, None)
}

/// Solve with early termination when `target` is reached.
///
/// The distance field may be incomplete; cells that were not finalized before
/// termination are reported as `f64::INFINITY`.
pub fn solve_to(
    dem: &Array2<f64>,
    config: TerrainConfig,
    cost_field: Option<&CostField>,
    source: (usize, usize),
    target: (usize, usize),
) -> Result<TerrainResult> {
    solve_terrain_inner(dem, config, cost_field, &[source], Some(target))
}

fn solve_terrain_inner(
    dem: &Array2<f64>,
    config: TerrainConfig,
    cost_field: Option<&CostField>,
    sources: &[(usize, usize)],
    target: Option<(usize, usize)>,
) -> Result<TerrainResult> {
    config.validate()?;

    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Err(Error::InvalidParameter("DEM must be at least 3x3"));
    }
    if sources.is_empty() {
        return Err(Error::InvalidParameter("at least one source is required"));
    }
    let dem_traversable: Vec<bool> = dem.iter().map(|z| z.is_finite()).collect();

    if let Some(cf) = cost_field {
        let cf_dim = cf.dim();
        if cf_dim != (h, w) {
            return Err(Error::DimensionMismatch {
                eh: h,
                ew: w,
                gh: cf_dim.0,
                gw: cf_dim.1,
            });
        }
    }

    for &(sr, sc) in sources {
        if sr >= h || sc >= w {
            return Err(Error::OutOfBounds {
                row: sr,
                col: sc,
                height: h,
                width: w,
            });
        }
        if !dem_traversable[sr * w + sc] {
            return Err(Error::InvalidParameter(
                "source DEM cell must have finite elevation",
            ));
        }
        if let Some(cf) = cost_field {
            if cf.at(sr, sc) <= 0.0 {
                return Err(Error::InvalidParameter("source cell must be traversable"));
            }
        }
    }
    if let Some((tr, tc)) = target {
        if tr >= h || tc >= w {
            return Err(Error::OutOfBounds {
                row: tr,
                col: tc,
                height: h,
                width: w,
            });
        }
        if !dem_traversable[tr * w + tc] {
            return Err(Error::InvalidParameter(
                "target DEM cell must have finite elevation",
            ));
        }
    }

    let n = grid_len(h, w)?;
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];

    let mut heap = BinaryHeap::with_capacity(n / 4);

    for &(sr, sc) in sources {
        let idx = sr * w + sc;
        dist[idx] = 0.0;
        heap.push(Node { cost: 0.0, idx });
    }

    while let Some(node) = heap.pop() {
        if visited[node.idx] {
            continue;
        }
        visited[node.idx] = true;

        if let Some((tr, tc)) = target {
            if node.idx == tr * w + tc {
                break;
            }
        }

        let row = node.idx / w;
        let col = node.idx % w;

        for &(dr, dc) in &NEIGHBORS {
            let nr = row as isize + dr;
            let nc = col as isize + dc;

            if nr < 0 || nr >= h as isize || nc < 0 || nc >= w as isize {
                continue;
            }

            let nr = nr as usize;
            let nc = nc as usize;
            let n_idx = nr * w + nc;

            if visited[n_idx] {
                continue;
            }

            if !dem_traversable[n_idx] {
                continue;
            }

            if let Some(cf) = cost_field {
                if cf.at(row, col) <= 0.0 || cf.at(nr, nc) <= 0.0 {
                    continue;
                }
            }

            let edge = edge_cost(
                dem,
                cost_field,
                config,
                (row, col),
                (nr, nc),
                dr.unsigned_abs() + dc.unsigned_abs() == 2,
            )?;

            let new_dist = dist[node.idx] + edge;
            if !new_dist.is_finite() {
                return Err(Error::InvalidParameter(
                    "terrain path costs must remain finite",
                ));
            }

            if new_dist < dist[n_idx] {
                dist[n_idx] = new_dist;
                pred[n_idx] = Some(node.idx);
                heap.push(Node {
                    cost: new_dist,
                    idx: n_idx,
                });
            }
        }
    }

    result_from_parts(h, w, dist, pred, visited, dem, config.cell_size)
}

fn result_from_parts(
    h: usize,
    w: usize,
    mut dist: Vec<f64>,
    mut pred: Vec<Option<usize>>,
    finalized: Vec<bool>,
    dem: &Array2<f64>,
    cell_size: f64,
) -> Result<TerrainResult> {
    for (idx, is_finalized) in finalized.iter().copied().enumerate() {
        if !is_finalized {
            dist[idx] = f64::INFINITY;
            pred[idx] = None;
        }
    }

    let distance = Array2::from_shape_vec((h, w), dist).unwrap();

    Ok(TerrainResult {
        distance,
        predecessors: pred,
        finalized,
        dem: dem.clone(),
        cell_size,
        width: w,
    })
}

fn edge_cost(
    dem: &Array2<f64>,
    cost_field: Option<&CostField>,
    config: TerrainConfig,
    from: (usize, usize),
    to: (usize, usize),
    diagonal: bool,
) -> Result<f64> {
    let horiz_dist = if diagonal {
        config.cell_size * std::f64::consts::SQRT_2
    } else {
        config.cell_size
    };
    if !horiz_dist.is_finite() {
        return Err(Error::InvalidParameter(
            "terrain edge distances must remain finite",
        ));
    }

    let dz = dem[[to.0, to.1]] - dem[[from.0, from.1]];
    let surface_dist = horiz_dist.hypot(dz);
    if !surface_dist.is_finite() {
        return Err(Error::InvalidParameter(
            "terrain surface distances must remain finite",
        ));
    }

    let slope_mult = if dz > 0.0 {
        config.uphill_factor
    } else if dz < 0.0 {
        config.downhill_factor
    } else {
        1.0
    };

    let terrain_mult = if let Some(cf) = cost_field {
        let mult = (cf.at(from.0, from.1) + cf.at(to.0, to.1)) * 0.5;
        if !mult.is_finite() {
            return Err(Error::InvalidParameter(
                "terrain cost multipliers must remain finite",
            ));
        }
        mult
    } else {
        1.0
    };

    let edge_cost = surface_dist * slope_mult * terrain_mult;
    if !edge_cost.is_finite() {
        return Err(Error::InvalidParameter(
            "terrain edge costs must remain finite",
        ));
    }
    Ok(edge_cost)
}

fn grid_len(height: usize, width: usize) -> Result<usize> {
    height
        .checked_mul(width)
        .ok_or(Error::InvalidParameter("grid dimensions are too large"))
}

const NEIGHBORS: [(isize, isize); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

#[derive(Clone, PartialEq)]
struct Node {
    cost: f64,
    idx: usize,
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_dem(h: usize, w: usize) -> Array2<f64> {
        Array2::zeros((h, w))
    }

    fn sloped_dem(h: usize, w: usize) -> Array2<f64> {
        Array2::from_shape_fn((h, w), |(_, c)| c as f64 * 10.0)
    }

    #[test]
    fn flat_terrain_geodesic_equals_euclidean() {
        let dem = flat_dem(20, 20);
        let config = TerrainConfig::symmetric(1.0);
        let result = solve(&dem, config, None, (0, 0)).unwrap();
        let path = result.path_to(0, 10).unwrap();
        // On flat terrain: surface distance = projected distance
        assert!((path.surface_distance - path.projected_distance).abs() < 1e-10);
        assert_eq!(path.total_ascent, 0.0);
        assert_eq!(path.total_descent, 0.0);
    }

    #[test]
    fn uphill_path_has_ascent() {
        let dem = sloped_dem(20, 20);
        let config = TerrainConfig::symmetric(1.0);
        let result = solve(&dem, config, None, (10, 0)).unwrap();
        let path = result.path_to(10, 19).unwrap();
        assert!(path.total_ascent > 0.0);
        assert!(path.surface_distance > path.projected_distance);
    }

    #[test]
    fn downhill_path_has_descent() {
        let dem = sloped_dem(20, 20);
        let config = TerrainConfig::symmetric(1.0);
        let result = solve(&dem, config, None, (10, 19)).unwrap();
        let path = result.path_to(10, 0).unwrap();
        assert!(path.total_descent > 0.0);
    }

    #[test]
    fn uphill_penalty_increases_cost() {
        let dem = sloped_dem(20, 20);
        let sym = TerrainConfig::symmetric(1.0);
        let penalized = TerrainConfig {
            cell_size: 1.0,
            uphill_factor: 3.0,
            downhill_factor: 1.0,
        };

        let r_sym = solve(&dem, sym, None, (10, 0)).unwrap();
        let r_pen = solve(&dem, penalized, None, (10, 0)).unwrap();

        let cost_sym = r_sym.path_to(10, 19).unwrap().cost;
        let cost_pen = r_pen.path_to(10, 19).unwrap().cost;
        assert!(cost_pen > cost_sym);
    }

    #[test]
    fn cost_field_obstacle_blocks() {
        let dem = flat_dem(10, 10);
        let config = TerrainConfig::symmetric(1.0);
        let mut cf_data = Array2::ones((10, 10));
        for r in 0..10 {
            cf_data[[r, 5]] = 0.0;
        }
        let cf = CostField::from_array(cf_data).unwrap();
        let result = solve(&dem, config, Some(&cf), (5, 0)).unwrap();
        // Can't cross the wall
        assert!(result.distance()[[5, 9]].is_infinite());
    }

    #[test]
    fn empty_sources_rejected() {
        let dem = flat_dem(5, 5);
        let config = TerrainConfig::symmetric(1.0);
        assert!(solve_multi(&dem, config, None, &[]).is_err());
    }

    #[test]
    fn cost_field_impassable_source_rejected() {
        let dem = flat_dem(5, 5);
        let config = TerrainConfig::symmetric(1.0);
        let mut cf_data = Array2::ones((5, 5));
        cf_data[[2, 2]] = 0.0;
        let cf = CostField::from_array(cf_data).unwrap();
        assert!(solve(&dem, config, Some(&cf), (2, 2)).is_err());
    }

    #[test]
    fn cost_field_obstacle_with_gap() {
        let dem = flat_dem(10, 10);
        let config = TerrainConfig::symmetric(1.0);
        let mut cf_data = Array2::ones((10, 10));
        for r in 0..10 {
            cf_data[[r, 5]] = 0.0;
        }
        cf_data[[5, 5]] = 1.0; // gap
        let cf = CostField::from_array(cf_data).unwrap();
        let result = solve(&dem, config, Some(&cf), (5, 0)).unwrap();
        let path = result.path_to(5, 9).unwrap();
        assert!(path.cells.iter().any(|&(r, c)| r == 5 && c == 5));
    }

    #[test]
    fn scalar_cost_field_edges_are_symmetric_on_flat_terrain() {
        let dem = flat_dem(5, 5);
        let config = TerrainConfig::symmetric(1.0);
        let mut cf_data = Array2::ones((5, 5));
        cf_data[[2, 2]] = 1.0;
        cf_data[[2, 3]] = 3.0;
        let cf = CostField::from_array(cf_data).unwrap();

        let forward = solve(&dem, config, Some(&cf), (2, 2)).unwrap();
        let reverse = solve(&dem, config, Some(&cf), (2, 3)).unwrap();

        assert!((forward.distance()[[2, 3]] - 2.0).abs() < 1e-10);
        assert!((forward.distance()[[2, 3]] - reverse.distance()[[2, 2]]).abs() < 1e-10);
    }

    #[test]
    fn elevation_profile_monotonic_distance() {
        let dem = Array2::from_shape_fn((20, 20), |(r, c)| {
            ((r as f64) * 0.3).sin() * 10.0 + c as f64
        });
        let config = TerrainConfig::symmetric(1.0);
        let result = solve(&dem, config, None, (10, 0)).unwrap();
        let path = result.path_to(10, 19).unwrap();
        for i in 1..path.elevation_profile.len() {
            assert!(path.elevation_profile[i].0 >= path.elevation_profile[i - 1].0);
        }
    }

    #[test]
    fn too_small_dem_rejected() {
        let dem = Array2::zeros((2, 5));
        let config = TerrainConfig::symmetric(1.0);
        assert!(solve(&dem, config, None, (0, 0)).is_err());
    }

    #[test]
    fn invalid_cell_size_rejected() {
        let dem = flat_dem(5, 5);
        let config = TerrainConfig::symmetric(0.0);
        assert!(solve(&dem, config, None, (0, 0)).is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let dem = flat_dem(10, 10);
        let config = TerrainConfig::symmetric(1.0);
        let cf = CostField::uniform(5, 5);
        assert!(solve(&dem, config, Some(&cf), (0, 0)).is_err());
    }

    #[test]
    fn non_finite_dem_cells_are_nodata_barriers() {
        let mut dem = flat_dem(10, 10);
        for r in 0..10 {
            dem[[r, 5]] = f64::NAN;
        }
        dem[[4, 4]] = f64::INFINITY;
        dem[[6, 4]] = f64::NEG_INFINITY;

        let config = TerrainConfig::symmetric(1.0);
        let result = solve(&dem, config, None, (5, 0)).unwrap();

        assert!(result.distance()[[5, 9]].is_infinite());
        assert!(result.distance()[[4, 4]].is_infinite());
        assert!(result.distance()[[6, 4]].is_infinite());
    }

    #[test]
    fn non_finite_dem_source_rejected() {
        let mut dem = flat_dem(5, 5);
        dem[[2, 2]] = f64::NAN;
        let config = TerrainConfig::symmetric(1.0);
        assert!(solve(&dem, config, None, (2, 2)).is_err());
    }

    #[test]
    fn non_finite_dem_target_rejected() {
        let mut dem = flat_dem(5, 5);
        dem[[4, 4]] = f64::INFINITY;
        let config = TerrainConfig::symmetric(1.0);
        assert!(solve_to(&dem, config, None, (0, 0), (4, 4)).is_err());
    }

    #[test]
    fn overflowing_terrain_edge_cost_rejected() {
        let dem = flat_dem(5, 5);
        let config = TerrainConfig::symmetric(f64::MAX);
        assert!(solve(&dem, config, None, (2, 2)).is_err());
    }

    #[test]
    fn hiking_config_prefers_contours() {
        // On a slope, hiking config should prefer traversing along contours
        // rather than straight uphill
        let dem = Array2::from_shape_fn((20, 20), |(r, _)| r as f64 * 20.0);
        let config = TerrainConfig::hiking(1.0);
        let result = solve(&dem, config, None, (0, 0)).unwrap();
        // Going to (0, 19) should be cheap (same elevation)
        // Going to (19, 0) should be expensive (straight uphill)
        let cost_flat = result.path_to(0, 19).unwrap().cost;
        let cost_up = result.path_to(19, 0).unwrap().cost;
        assert!(cost_up > cost_flat);
    }

    #[test]
    fn solve_to_early_termination() {
        let dem = flat_dem(50, 50);
        let config = TerrainConfig::symmetric(1.0);
        let result = solve_to(&dem, config, None, (0, 0), (10, 10)).unwrap();
        let path = result.path_to(10, 10).unwrap();
        assert_eq!(path.cells.first(), Some(&(0, 0)));
        assert_eq!(path.cells.last(), Some(&(10, 10)));
        assert!(result.distance()[[0, 15]].is_infinite());
        assert!(matches!(result.path_to(0, 15), Err(Error::NoPathFound)));
    }

    #[test]
    fn multi_source_terrain() {
        let dem = flat_dem(20, 20);
        let config = TerrainConfig::symmetric(1.0);
        let result = solve_multi(&dem, config, None, &[(0, 0), (19, 19)]).unwrap();
        assert_eq!(result.distance()[[0, 0]], 0.0);
        assert_eq!(result.distance()[[19, 19]], 0.0);
        // Center should be reachable from both
        assert!(result.distance()[[10, 10]].is_finite());
    }
}
