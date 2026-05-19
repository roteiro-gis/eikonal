//! Weighted shortest paths on an 8-connected grid graph.
//!
//! This module intentionally solves a graph problem, not the eikonal PDE. Use
//! the crate-level [`crate::solve`] APIs for Fast Marching Method / upwind
//! eikonal distances.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::Array2;

use crate::cost::CostField;
use crate::error::{Error, Result};

/// Result of Dijkstra shortest paths from one or more sources.
///
/// Contains the graph distance field (minimum cumulative edge cost to each
/// cell from any source) and an opaque predecessor map for path extraction.
pub struct SolveResult {
    pub(crate) distance: Array2<f64>,
    pub(crate) predecessors: Vec<u32>,
    pub(crate) width: usize,
}

/// Sentinel value indicating no predecessor (source cells, unreachable cells).
const NO_PRED: u32 = u32::MAX;

impl SolveResult {
    /// The shortest-path distance from the nearest source to each cell.
    ///
    /// Unreachable cells have `f64::INFINITY`.
    pub fn distance(&self) -> &Array2<f64> {
        &self.distance
    }

    /// Extract the shortest grid path from a source to the given target cell.
    ///
    /// Returns the path as a sequence of `(row, col)` coordinates from source
    /// to target, and the total graph cost.
    ///
    /// # Errors
    ///
    /// Returns `NoPathFound` if the target is unreachable.
    /// Returns `OutOfBounds` if the target is outside the grid.
    pub fn path_to(&self, target_row: usize, target_col: usize) -> Result<Path> {
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

        let cost = self.distance[[target_row, target_col]];
        if cost.is_infinite() {
            return Err(Error::NoPathFound);
        }

        let mut cells = Vec::new();
        let mut idx = target_row * w + target_col;
        loop {
            let r = idx / w;
            let c = idx % w;
            cells.push((r, c));
            let pred = self.predecessors[idx];
            if pred == NO_PRED {
                break;
            }
            idx = pred as usize;
        }
        cells.reverse();

        Ok(Path { cells, cost })
    }
}

/// A shortest path on the 8-connected grid graph.
#[derive(Clone, Debug)]
pub struct Path {
    /// Sequence of (row, col) from source to target.
    pub cells: Vec<(usize, usize)>,
    /// Total cumulative graph cost along the path.
    pub cost: f64,
}

/// Solve weighted shortest paths from a single source cell.
///
/// Computes the minimum-cost path from `source` to every reachable cell using
/// Dijkstra's algorithm on the 8-connected grid graph.
///
/// Edge cost from cell A to adjacent cell B is:
/// `euclidean_distance(A, B) * cost_field[B]`
///
/// Cells with cost 0 are impassable barriers.
///
/// # Arguments
/// * `cost` - Per-cell destination cost multiplier
/// * `source` - Source cell as `(row, col)`
///
/// # Errors
///
/// Returns `OutOfBounds` if the source is outside the grid.
pub fn solve(cost: &CostField, source: (usize, usize)) -> Result<SolveResult> {
    solve_multi(cost, &[source])
}

/// Solve weighted shortest paths from multiple source cells simultaneously.
///
/// All sources start at cost 0; the result gives the minimum graph cost from
/// the *nearest* source to every cell.
///
/// # Errors
///
/// Returns `OutOfBounds` if any source is outside the grid.
pub fn solve_multi(cost: &CostField, sources: &[(usize, usize)]) -> Result<SolveResult> {
    solve_inner(cost, sources, None)
}

/// Solve weighted shortest paths with early termination when `target` is reached.
///
/// The distance field may be incomplete (cells farther than the target are
/// not guaranteed to be computed), but the graph path to `target` is optimal.
pub fn solve_to(
    cost: &CostField,
    source: (usize, usize),
    target: (usize, usize),
) -> Result<SolveResult> {
    solve_inner(cost, &[source], Some(target))
}

fn solve_inner(
    cost: &CostField,
    sources: &[(usize, usize)],
    target: Option<(usize, usize)>,
) -> Result<SolveResult> {
    let (h, w) = cost.dim();
    if h == 0 || w == 0 {
        return Err(Error::InvalidParameter("cost field must be non-empty"));
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
    }

    let n = h * w;
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<u32> = vec![NO_PRED; n];
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

            let cell_cost = cost.at(nr, nc);
            if cell_cost <= 0.0 {
                continue;
            }

            let geom_dist = if dr.unsigned_abs() + dc.unsigned_abs() == 2 {
                std::f64::consts::SQRT_2
            } else {
                1.0
            };

            let edge_cost = geom_dist * cell_cost;
            let new_dist = dist[node.idx] + edge_cost;

            if new_dist < dist[n_idx] {
                dist[n_idx] = new_dist;
                pred[n_idx] = node.idx as u32;
                heap.push(Node {
                    cost: new_dist,
                    idx: n_idx,
                });
            }
        }
    }

    let distance = Array2::from_shape_vec((h, w), dist).unwrap();

    Ok(SolveResult {
        distance,
        predecessors: pred,
        width: w,
    })
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

    #[test]
    fn solve_single_source_flat() {
        let cost = CostField::uniform(10, 10);
        let result = solve(&cost, (0, 0)).unwrap();
        assert_eq!(result.distance()[[0, 0]], 0.0);
        assert!(result.distance()[[0, 1]] > 0.0);
    }

    #[test]
    fn solve_source_out_of_bounds() {
        let cost = CostField::uniform(10, 10);
        assert!(solve(&cost, (10, 0)).is_err());
    }

    #[test]
    fn solve_path_to_adjacent() {
        let cost = CostField::uniform(5, 5);
        let result = solve(&cost, (2, 2)).unwrap();
        let path = result.path_to(2, 3).unwrap();
        assert_eq!(path.cells.first(), Some(&(2, 2)));
        assert_eq!(path.cells.last(), Some(&(2, 3)));
        assert!((path.cost - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_path_to_diagonal() {
        let cost = CostField::uniform(5, 5);
        let result = solve(&cost, (0, 0)).unwrap();
        let path = result.path_to(1, 1).unwrap();
        assert!((path.cost - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn solve_unreachable_target() {
        let mut data = Array2::ones((5, 5));
        for r in 0..5 {
            data[[r, 2]] = 0.0;
        }
        let cost = CostField::from_array(data).unwrap();
        let result = solve(&cost, (2, 0)).unwrap();
        assert!(result.path_to(2, 4).is_err());
    }

    #[test]
    fn solve_routes_around_obstacle() {
        let mut data = Array2::ones((5, 5));
        for r in 0..5 {
            data[[r, 2]] = 0.0;
        }
        data[[2, 2]] = 1.0;
        let cost = CostField::from_array(data).unwrap();
        let result = solve(&cost, (0, 0)).unwrap();
        let path = result.path_to(0, 4).unwrap();
        assert!(path.cells.iter().any(|&(r, c)| r == 2 && c == 2));
    }

    #[test]
    fn solve_multi_sources() {
        let cost = CostField::uniform(10, 10);
        let result = solve_multi(&cost, &[(0, 0), (9, 9)]).unwrap();
        let d_center = result.distance()[[5, 5]];
        let d_corner = result.distance()[[0, 0]];
        assert_eq!(d_corner, 0.0);
        assert!(d_center > 0.0);
        let d_far = result.distance()[[9, 9]];
        assert_eq!(d_far, 0.0);
    }

    #[test]
    fn solve_to_early_termination() {
        let cost = CostField::uniform(100, 100);
        let result = solve_to(&cost, (0, 0), (5, 5)).unwrap();
        assert!(result.distance()[[5, 5]].is_finite());
        let path = result.path_to(5, 5).unwrap();
        assert_eq!(path.cells.first(), Some(&(0, 0)));
        assert_eq!(path.cells.last(), Some(&(5, 5)));
    }

    #[test]
    fn solve_high_cost_region_avoided() {
        let mut data = Array2::ones((5, 5));
        for c in 0..5 {
            data[[2, c]] = 100.0;
        }
        data[[2, 2]] = 1.0;
        let cost = CostField::from_array(data).unwrap();
        let result = solve(&cost, (0, 2)).unwrap();
        let path = result.path_to(4, 2).unwrap();
        assert!(path.cells.iter().any(|&(r, c)| r == 2 && c == 2));
    }

    #[test]
    fn distance_field_symmetry() {
        let cost = CostField::uniform(11, 11);
        let result = solve(&cost, (5, 5)).unwrap();
        let d = result.distance();
        assert!((d[[4, 5]] - d[[6, 5]]).abs() < 1e-10);
        assert!((d[[5, 4]] - d[[5, 6]]).abs() < 1e-10);
        assert!((d[[4, 4]] - d[[6, 6]]).abs() < 1e-10);
    }
}
