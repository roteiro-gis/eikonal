use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::Array2;

use crate::cost::CostField;
use crate::error::{Error, Result};

/// Result of solving the eikonal equation from one or more source cells.
///
/// Contains the arrival-time field produced by a first-order upwind Fast
/// Marching Method discretization of `|grad T| = C(x)`, where `C` is the
/// per-cell cost/slowness field.
pub struct SolveResult {
    pub(crate) distance: Array2<f64>,
    pub(crate) predecessors: Vec<Option<usize>>,
    pub(crate) width: usize,
}

impl SolveResult {
    /// The FMM arrival time from the nearest source to each cell.
    ///
    /// Unreachable cells and impassable barriers have `f64::INFINITY`.
    pub fn distance(&self) -> &Array2<f64> {
        &self.distance
    }

    /// Extract a monotone upwind backtrace from the target to a source.
    ///
    /// Fast Marching Method solves for an arrival-time field. It does not
    /// compute an exact discrete shortest path; this method follows the
    /// accepted neighbor that supplied the target's upwind update. For exact
    /// 8-neighbor graph paths, use [`crate::graph::solve`].
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
            if let Some(pred) = self.predecessors[idx] {
                idx = pred;
            } else {
                break;
            }
        }
        cells.reverse();

        Ok(Path { cells, cost })
    }
}

/// A monotone upwind backtrace through an FMM distance field.
#[derive(Clone, Debug)]
pub struct Path {
    /// Sequence of (row, col) from source to target.
    pub cells: Vec<(usize, usize)>,
    /// FMM arrival time at the target.
    pub cost: f64,
}

/// Solve the eikonal equation from a single source cell.
///
/// Computes the first-order upwind Fast Marching Method approximation to
/// `|grad T| = C(x)` on a unit-spaced 2D Cartesian grid, where `C(x)` is the
/// cost/slowness field (`C = 1 / speed`). Cells with cost 0 are treated as
/// impassable barriers.
///
/// # Arguments
/// * `cost` - Per-cell cost/slowness field
/// * `source` - Source cell as `(row, col)`
///
/// # Errors
///
/// Returns `OutOfBounds` if the source is outside the grid.
pub fn solve(cost: &CostField, source: (usize, usize)) -> Result<SolveResult> {
    solve_multi(cost, &[source])
}

/// Solve the eikonal equation from multiple source cells simultaneously.
///
/// All sources start at arrival time 0; the result gives the first-arrival time
/// from the nearest source under the upwind eikonal discretization.
///
/// # Errors
///
/// Returns `OutOfBounds` if any source is outside the grid.
/// Returns `InvalidParameter` if `sources` is empty.
/// Returns `InvalidParameter` if any source is on an impassable cell.
pub fn solve_multi(cost: &CostField, sources: &[(usize, usize)]) -> Result<SolveResult> {
    solve_inner(cost, sources, None)
}

/// Solve with early termination when `target` is accepted.
///
/// The distance field may be incomplete (cells farther than the target are not
/// guaranteed to be computed), but the target arrival time is final once the
/// target is accepted by the FMM front.
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
    if sources.is_empty() {
        return Err(Error::InvalidParameter("at least one source is required"));
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
        if cost.at(sr, sc) <= 0.0 {
            return Err(Error::InvalidParameter("source cell must be traversable"));
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

    let n = grid_len(h, w)?;
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    let mut state = vec![State::Far; n];
    let mut heap = BinaryHeap::with_capacity(n / 4);

    for &(sr, sc) in sources {
        let idx = sr * w + sc;
        dist[idx] = 0.0;
        state[idx] = State::Accepted;
    }

    if let Some((tr, tc)) = target {
        if dist[tr * w + tc] == 0.0 {
            return result_from_parts(h, w, dist, pred);
        }
    }

    for &(sr, sc) in sources {
        update_neighbors(cost, sr, sc, &mut dist, &mut pred, &mut state, &mut heap);
    }

    while let Some(node) = heap.pop() {
        if state[node.idx] == State::Accepted || node.cost > dist[node.idx] {
            continue;
        }

        state[node.idx] = State::Accepted;

        if let Some((tr, tc)) = target {
            if node.idx == tr * w + tc {
                break;
            }
        }

        let row = node.idx / w;
        let col = node.idx % w;
        update_neighbors(cost, row, col, &mut dist, &mut pred, &mut state, &mut heap);
    }

    result_from_parts(h, w, dist, pred)
}

fn result_from_parts(
    h: usize,
    w: usize,
    dist: Vec<f64>,
    pred: Vec<Option<usize>>,
) -> Result<SolveResult> {
    let distance = Array2::from_shape_vec((h, w), dist).unwrap();
    Ok(SolveResult {
        distance,
        predecessors: pred,
        width: w,
    })
}

fn update_neighbors(
    cost: &CostField,
    row: usize,
    col: usize,
    dist: &mut [f64],
    pred: &mut [Option<usize>],
    state: &mut [State],
    heap: &mut BinaryHeap<Node>,
) {
    let (h, w) = cost.dim();
    for &(dr, dc) in &CARDINAL_NEIGHBORS {
        let nr = row as isize + dr;
        let nc = col as isize + dc;

        if nr < 0 || nr >= h as isize || nc < 0 || nc >= w as isize {
            continue;
        }

        let nr = nr as usize;
        let nc = nc as usize;
        let idx = nr * w + nc;
        if state[idx] == State::Accepted || cost.at(nr, nc) <= 0.0 {
            continue;
        }

        if let Some((candidate, predecessor)) = upwind_update(cost, nr, nc, dist, state) {
            if candidate < dist[idx] {
                dist[idx] = candidate;
                pred[idx] = Some(predecessor);
                state[idx] = State::Trial;
                heap.push(Node {
                    cost: candidate,
                    idx,
                });
            }
        }
    }
}

fn upwind_update(
    cost: &CostField,
    row: usize,
    col: usize,
    dist: &[f64],
    state: &[State],
) -> Option<(f64, usize)> {
    let (h, w) = cost.dim();
    let slowness = cost.at(row, col);
    if slowness <= 0.0 {
        return None;
    }

    let x = best_axis_neighbor(row, col, h, w, dist, state, &[(0, -1), (0, 1)]);
    let y = best_axis_neighbor(row, col, h, w, dist, state, &[(-1, 0), (1, 0)]);

    match (x, y) {
        (None, None) => None,
        (Some((a, pred)), None) | (None, Some((a, pred))) => Some((a + slowness, pred)),
        (Some(xn), Some(yn)) => {
            let ((a, pred_a), (b, _pred_b)) = if xn.0 <= yn.0 { (xn, yn) } else { (yn, xn) };
            if b - a >= slowness {
                Some((a + slowness, pred_a))
            } else {
                let diff = b - a;
                let discriminant = 2.0 * slowness * slowness - diff * diff;
                if discriminant < 0.0 {
                    Some((a + slowness, pred_a))
                } else {
                    Some(((a + b + discriminant.sqrt()) * 0.5, pred_a))
                }
            }
        }
    }
}

fn best_axis_neighbor(
    row: usize,
    col: usize,
    height: usize,
    width: usize,
    dist: &[f64],
    state: &[State],
    offsets: &[(isize, isize); 2],
) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;

    for &(dr, dc) in offsets {
        let nr = row as isize + dr;
        let nc = col as isize + dc;
        if nr < 0 || nr >= height as isize || nc < 0 || nc >= width as isize {
            continue;
        }
        let idx = nr as usize * width + nc as usize;
        if state[idx] != State::Accepted {
            continue;
        }
        let d = dist[idx];
        match best {
            Some((best_d, _)) if best_d <= d => {}
            _ => best = Some((d, idx)),
        }
    }

    best
}

fn grid_len(height: usize, width: usize) -> Result<usize> {
    height
        .checked_mul(width)
        .ok_or(Error::InvalidParameter("grid dimensions are too large"))
}

const CARDINAL_NEIGHBORS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

#[derive(Clone, Copy, Eq, PartialEq)]
enum State {
    Far,
    Trial,
    Accepted,
}

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

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }

    #[test]
    fn solve_single_source_flat() {
        let cost = CostField::uniform(10, 10);
        let result = solve(&cost, (0, 0)).unwrap();
        assert_eq!(result.distance()[[0, 0]], 0.0);
        assert_close(result.distance()[[0, 1]], 1.0);
        assert_close(result.distance()[[1, 0]], 1.0);
    }

    #[test]
    fn solve_source_out_of_bounds() {
        let cost = CostField::uniform(10, 10);
        assert!(solve(&cost, (10, 0)).is_err());
    }

    #[test]
    fn empty_sources_rejected() {
        let cost = CostField::uniform(10, 10);
        assert!(solve_multi(&cost, &[]).is_err());
    }

    #[test]
    fn impassable_source_rejected() {
        let mut data = Array2::ones((5, 5));
        data[[2, 2]] = 0.0;
        let cost = CostField::from_array(data).unwrap();
        assert!(solve(&cost, (2, 2)).is_err());
    }

    #[test]
    fn diagonal_uses_upwind_eikonal_update() {
        let cost = CostField::uniform(5, 5);
        let result = solve(&cost, (0, 0)).unwrap();
        let expected = 1.0 + std::f64::consts::FRAC_1_SQRT_2;
        assert_close(result.distance()[[1, 1]], expected);
    }

    #[test]
    fn constant_slowness_scales_arrival_time() {
        let data = Array2::from_elem((5, 5), 2.0);
        let cost = CostField::from_array(data).unwrap();
        let result = solve(&cost, (2, 2)).unwrap();
        assert_close(result.distance()[[2, 4]], 4.0);
    }

    #[test]
    fn path_to_follows_upwind_predecessors() {
        let cost = CostField::uniform(5, 5);
        let result = solve(&cost, (0, 0)).unwrap();
        let path = result.path_to(3, 2).unwrap();
        assert_eq!(path.cells.first(), Some(&(0, 0)));
        assert_eq!(path.cells.last(), Some(&(3, 2)));
        assert_close(path.cost, result.distance()[[3, 2]]);
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
    fn solve_multi_sources() {
        let cost = CostField::uniform(10, 10);
        let result = solve_multi(&cost, &[(0, 0), (9, 9)]).unwrap();
        assert_eq!(result.distance()[[0, 0]], 0.0);
        assert_eq!(result.distance()[[9, 9]], 0.0);
        assert!(result.distance()[[5, 5]].is_finite());
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
    fn distance_field_symmetry() {
        let cost = CostField::uniform(11, 11);
        let result = solve(&cost, (5, 5)).unwrap();
        let d = result.distance();
        assert!((d[[4, 5]] - d[[6, 5]]).abs() < 1e-10);
        assert!((d[[5, 4]] - d[[5, 6]]).abs() < 1e-10);
        assert!((d[[4, 4]] - d[[6, 6]]).abs() < 1e-10);
    }
}
