use ndarray::Array2;

/// Extract an isochrone (equal-cost boundary) from a distance field.
///
/// Returns all cells reachable within `max_cost` from the source(s) that were
/// used to compute the distance field. Each entry is `(row, col, cost)`.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use eikonal::{CostField, solve, isochrone};
///
/// let cost = CostField::uniform(20, 20);
/// let result = solve(&cost, (10, 10)).unwrap();
/// let reachable = isochrone(result.distance(), 5.0);
/// assert!(reachable.iter().all(|&(_, _, c)| c <= 5.0));
/// ```
pub fn isochrone(distance: &Array2<f64>, max_cost: f64) -> Vec<(usize, usize, f64)> {
    let (h, w) = distance.dim();
    let mut cells = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let d = distance[[r, c]];
            if d <= max_cost {
                cells.push((r, c, d));
            }
        }
    }
    cells
}

/// Extract the boundary cells of an isochrone.
///
/// Returns cells that are within `max_cost` but have at least one 8-connected
/// neighbor that is *not* within `max_cost` (or is at the grid edge). These
/// form the perimeter of the reachable region.
pub fn isochrone_boundary(distance: &Array2<f64>, max_cost: f64) -> Vec<(usize, usize, f64)> {
    let (h, w) = distance.dim();
    let mut boundary = Vec::new();

    for r in 0..h {
        for c in 0..w {
            let d = distance[[r, c]];
            if d > max_cost {
                continue;
            }

            let on_edge = r == 0 || r == h - 1 || c == 0 || c == w - 1;
            if on_edge {
                boundary.push((r, c, d));
                continue;
            }

            let has_outside_neighbor = NEIGHBORS.iter().any(|&(dr, dc)| {
                let nr = (r as isize + dr) as usize;
                let nc = (c as isize + dc) as usize;
                distance[[nr, nc]] > max_cost
            });

            if has_outside_neighbor {
                boundary.push((r, c, d));
            }
        }
    }
    boundary
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{solve, CostField};

    #[test]
    fn isochrone_contains_source() {
        let cost = CostField::uniform(10, 10);
        let result = solve(&cost, (5, 5)).unwrap();
        let cells = isochrone(result.distance(), 3.0);
        assert!(cells.iter().any(|&(r, c, _)| r == 5 && c == 5));
    }

    #[test]
    fn isochrone_all_within_threshold() {
        let cost = CostField::uniform(20, 20);
        let result = solve(&cost, (10, 10)).unwrap();
        let cells = isochrone(result.distance(), 4.0);
        for &(_, _, d) in &cells {
            assert!(d <= 4.0);
        }
    }

    #[test]
    fn isochrone_grows_with_threshold() {
        let cost = CostField::uniform(20, 20);
        let result = solve(&cost, (10, 10)).unwrap();
        let small = isochrone(result.distance(), 2.0);
        let large = isochrone(result.distance(), 5.0);
        assert!(large.len() > small.len());
    }

    #[test]
    fn boundary_is_subset_of_isochrone() {
        let cost = CostField::uniform(20, 20);
        let result = solve(&cost, (10, 10)).unwrap();
        let all = isochrone(result.distance(), 4.0);
        let bnd = isochrone_boundary(result.distance(), 4.0);
        assert!(bnd.len() <= all.len());
        for &(r, c, _) in &bnd {
            assert!(all.iter().any(|&(ar, ac, _)| ar == r && ac == c));
        }
    }

    #[test]
    fn boundary_forms_perimeter() {
        let cost = CostField::uniform(20, 20);
        let result = solve(&cost, (10, 10)).unwrap();
        let bnd = isochrone_boundary(result.distance(), 3.0);
        // Source is interior, not on boundary (neighbors all within threshold)
        assert!(!bnd.iter().any(|&(r, c, _)| r == 10 && c == 10));
    }

    #[test]
    fn obstacle_shapes_isochrone() {
        let mut data = Array2::ones((20, 20));
        // Wall on the right
        for r in 0..20 {
            data[[r, 12]] = 0.0;
        }
        let cost = CostField::from_array(data).unwrap();
        let result = solve(&cost, (10, 10)).unwrap();
        let cells = isochrone(result.distance(), 5.0);
        // No cell beyond the wall should be reachable
        assert!(!cells.iter().any(|&(_, c, _)| c > 12));
    }
}
