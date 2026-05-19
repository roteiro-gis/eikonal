#![forbid(unsafe_code)]

//! **eikonal** — Fast Marching Method and grid routing utilities.
//!
//! The crate-level [`solve`] APIs solve the eikonal equation
//! `|∇T(x)| = C(x) = 1/F(x)` on 2D regular grids using a first-order upwind
//! Fast Marching Method. The [`graph`] module provides 8-neighbor Dijkstra
//! shortest paths for applications that need exact discrete graph paths, and
//! [`terrain`] builds on graph routing for DEM-aware path costs.
//!
//! # Core concepts
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`CostField`] | Per-cell cost/slowness (`1 / speed`) |
//! | [`SolveResult`] | FMM arrival-time field + upwind backtrace |
//! | [`graph::SolveResult`] | 8-neighbor Dijkstra graph distances + shortest paths |
//! | [`TerrainResult`](terrain::TerrainResult) | Terrain-routing result with elevation profiles |
//! | [`TerrainConfig`](terrain::TerrainConfig) | Slope penalty configuration |
//!
//! # Quick start — eikonal / FMM
//!
//! ```
//! use ndarray::Array2;
//! use eikonal::{CostField, solve};
//!
//! let cost = CostField::uniform(50, 50);
//! let result = solve(&cost, (0, 0)).unwrap();
//!
//! // FMM arrival time to far corner
//! let d = result.distance()[[49, 49]];
//! assert!(d > 0.0);
//!
//! // Extract an upwind backtrace through the arrival-time field
//! let path = result.path_to(49, 49).unwrap();
//! assert_eq!(path.cells[0], (0, 0));
//! ```
//!
//! # Quick start — weighted grid shortest paths
//!
//! ```
//! use eikonal::{graph, CostField};
//!
//! let cost = CostField::uniform(50, 50);
//! let result = graph::solve(&cost, (0, 0)).unwrap();
//! let path = result.path_to(49, 49).unwrap();
//! assert_eq!(path.cells[0], (0, 0));
//! ```
//!
//! # Quick start — terrain routing
//!
//! ```
//! use ndarray::Array2;
//! use eikonal::terrain::{self, TerrainConfig};
//!
//! // A DEM with a hill in the center
//! let dem = Array2::from_shape_fn((50, 50), |(r, c)| {
//!     let dr = r as f64 - 25.0;
//!     let dc = c as f64 - 25.0;
//!     100.0 * (-(dr * dr + dc * dc) / 200.0).exp()
//! });
//!
//! let config = TerrainConfig::hiking(30.0); // 30m cells
//! let result = terrain::solve(&dem, config, None, (0, 0)).unwrap();
//! let path = result.path_to(49, 49).unwrap();
//!
//! assert!(path.surface_distance > path.projected_distance);
//! assert!(path.total_ascent > 0.0);
//! ```
//!
//! # Isochrones
//!
//! ```
//! use eikonal::{CostField, solve, isochrone};
//!
//! let cost = CostField::uniform(30, 30);
//! let result = solve(&cost, (15, 15)).unwrap();
//! let reachable = isochrone(result.distance(), 10.0);
//! ```
//!
//! # Parallelism
//!
//! Enable the `parallel` feature for Rayon-accelerated cost field construction:
//!
//! ```toml
//! [dependencies]
//! eikonal = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! The FMM and graph solvers are inherently sequential priority-queue
//! algorithms. The `parallel` feature currently accelerates cost field
//! construction; isochrone extraction remains sequential.

pub mod cost;
pub mod error;
pub mod graph;
pub mod isochrone;
pub mod solver;
pub mod terrain;

pub use cost::CostField;
pub use error::{Error, Result};
pub use graph::{
    solve as solve_graph, solve_multi as solve_graph_multi, solve_to as solve_graph_to,
    Path as GraphPath, SolveResult as GraphSolveResult,
};
pub use isochrone::{isochrone, isochrone_boundary};
pub use solver::{solve, solve_multi, solve_to, Path, SolveResult};
