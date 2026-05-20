# eikonal

Fast Marching Method and grid-routing utilities for regular 2D grids.

Top-level `solve` computes first-order upwind FMM arrival times for:

```text
|grad T(x)| = C(x) = 1 / F(x)
```

Use `eikonal::graph` for exact weighted shortest paths on an 8-neighbor grid,
and `eikonal::terrain` for DEM-aware graph routing with 3D surface distances,
elevation profiles, and uphill/downhill cost factors.

Raster I/O, CRS handling, reprojection, resampling, and vertical datum handling
are outside the crate. Use `geotiff-rust` for GeoTIFF I/O and `terrand` for
DEM-derived rasters such as slope.

## Installation

```sh
cargo add eikonal
```

Or add the dependency directly:

```toml
[dependencies]
eikonal = "0.1"
```

Enable Rayon-accelerated cost-field construction with:

```toml
eikonal = { version = "0.1", features = ["parallel"] }
```

## Quick Start

```rust
use eikonal::{graph, solve, CostField};

let cost = CostField::uniform(50, 50);

let fmm = solve(&cost, (0, 0)).unwrap();
let arrival = fmm.distance()[[49, 49]];
assert!(arrival > 0.0);

let routed = graph::solve(&cost, (0, 0)).unwrap();
let path = routed.path_to(49, 49).unwrap();
assert_eq!(path.cells[0], (0, 0));
```

```rust
use ndarray::Array2;
use eikonal::terrain::{self, TerrainConfig};

let dem = Array2::from_shape_fn((50, 50), |(row, col)| {
    row as f64 * 2.0 + col as f64 * 5.0
});
let config = TerrainConfig::hiking(30.0).unwrap();

let result = terrain::solve(&dem, config, None, (0, 0)).unwrap();
let path = result.path_to(49, 49).unwrap();

assert!(path.surface_distance >= path.projected_distance);
```

## Choosing an API

Use top-level `solve` / `solve_to` when you want a continuous-style
arrival-time field, isochrones, or front propagation. FMM reduces grid
direction bias, but `path_to` is an upwind backtrace through the field, not an
exact discrete shortest path.

Use `graph::solve` / `graph::solve_to` when the route itself is primary. It is
exact for the crate's 8-neighbor graph model with endpoint-averaged edge costs.

Use `terrain::solve` / `terrain::solve_to` for graph routing over a DEM. It
uses 3D surface edge distances, optional scalar cost multipliers, and validated
`TerrainConfig` slope factors.

## Assumptions

All APIs use `(row, col)` grid coordinates. Top-level FMM and `graph` use unit
grid spacing. Terrain routing uses `TerrainConfig::cell_size()` for horizontal
map units; DEM elevations should use compatible vertical units if surface
distance and slope penalties should be physically meaningful.

`CostField` values must be finite and non-negative. Zero marks an impassable
cell. Large finite costs are accepted, but solvers reject non-finite accumulated
distances.

`solve_to` stops once the target is finalized. The returned distance field may
be incomplete; unfinished cells are reported as `f64::INFINITY`.

Graph and terrain routing allow diagonal moves. Diagonal edges only require the
two endpoint cells to be traversable, so they may pass between blocked
orthogonal neighbors; no extra corner-cutting rule is applied.

Terrain DEM nodata policy: non-finite elevations (`NaN`, `+inf`, `-inf`) are
barriers. Routes never enter them, and source/early-target cells must have
finite elevation.

## GeoTIFF Workflow

The intended geospatial pipeline is:

```text
geotiff-rust -> ndarray DEM/cost rasters -> terrand -> eikonal -> geotiff-rust
```

Read a projected, regular-grid DEM with `geotiff-reader`, preserving transform,
cell size, CRS, and nodata metadata outside `eikonal`. Convert nodata cells to
non-finite DEM elevations or zero-valued `CostField` cells. Use `terrand` to
derive slope or other terrain rasters, then build costs with
`CostField::from_slope` or pass the DEM directly to `terrain`. Write distance,
isochrone, or route outputs with `geotiff-writer` using the original raster
metadata.
