# Changelog

## 0.1.0 - 2026-05-21

Initial public release.

- First-order Fast Marching Method solver for 2D eikonal arrival-time fields.
- Multi-source solving and target early termination.
- Upwind FMM path backtracing from finalized cells.
- 8-neighbor weighted graph routing with exact shortest paths for the graph model.
- Terrain-aware graph routing with 3D surface distance, elevation profiles,
  ascent/descent statistics, and configurable uphill/downhill factors.
- Isochrone and isochrone-boundary extraction from distance fields.
- `CostField` construction from uniform values, obstacle masks, arrays, and
  slope rasters.
- Optional `parallel` feature for Rayon-accelerated cost-field construction.
