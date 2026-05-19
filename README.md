# eikonal

Fast Marching Method and grid-routing utilities for regular 2D grids.

Top-level `solve` computes first-order upwind FMM arrival times for:

```text
|grad T(x)| = C(x) = 1 / F(x)
```

Use `eikonal::graph` for exact weighted shortest paths on an 8-neighbor grid
with symmetric endpoint-averaged scalar costs, and `eikonal::terrain` for
DEM-aware routing with 3D surface distances.

```rust
use eikonal::{solve, CostField};

let cost = CostField::uniform(50, 50);
let result = solve(&cost, (0, 0)).unwrap();

assert!(result.distance()[[49, 49]] > 0.0);
```

```rust
use eikonal::{graph, CostField};

let cost = CostField::uniform(50, 50);
let result = graph::solve(&cost, (0, 0)).unwrap();
let path = result.path_to(49, 49).unwrap();

assert_eq!(path.cells[0], (0, 0));
```

Terrain DEM nodata policy: non-finite elevations are barriers; sources and
early-termination targets must have finite elevation.
