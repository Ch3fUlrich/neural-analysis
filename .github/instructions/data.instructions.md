---
name: 'Synthetic Data'
description: 'Conventions for synthetic neural data generation'
applyTo: 'src/neural_analysis/data/**'
---

# Data Generation Instructions

## Module Layout

- `synthetic_data.py` — Facade re-exporting from `generators.py`, `trajectories_gen.py`, and `datasets.py`.
- `generators.py` — Core cell-type generators (place, grid, HD, random cells).
- `trajectories_gen.py` — Trajectory generation for 1D/2D/3D environments.
- `datasets.py` — High-level dataset builders and `generate_data()` dispatcher.

Generates realistic synthetic neural datasets for testing and validation across 1D, 2D, and 3D spatial environments.

## Cell Types

| Type | Algorithm | Key Params |
|------|-----------|------------|
| Place cells | Gaussian/anisotropic place fields | `field_size`, `peak_rate`, `noise_level` |
| Grid cells | Hexagonal lattice (2D), FCC tetrahedral (3D), harmonic (1D) | `grid_spacing`, `grid_orientation` |
| Head direction | Von Mises circular tuning curves | `tuning_width`, `peak_rate` |
| Random cells | Correlated noise with temporal smoothing | `baseline_rate`, `variability`, `temporal_smoothness` |

## API Conventions

- **`generate_data(dataset_type, ...)`** — Unified dispatcher (match/case). Always use this as the entry point.
- **`generate_mixed_population_flexible(cell_config, ...)`** — Multi-type population generation via configuration dict.
- All generators return `(activity: ndarray, metadata: dict)` tuples.
- All generators accept an `rng` parameter (int or numpy Generator) for reproducible results.
- Use `reproducible(seed)` context manager for deterministic execution.
- Optional auto-plotting via PlotGrid can be triggered on generation.

## Adding a New Cell Type

1. Implement the generator function matching the `(activity, metadata)` return convention.
2. Add a `case "new_type":` branch in `generate_data`.
3. Support 1D, 2D, and 3D environments where applicable.
4. Include the type in `generate_mixed_population_flexible` support.
5. Add tests and update `docs/function_registry.md`.

## Manifold Mappings

- `map_to_ring(activity, positions)` — Map 1D activity to ring manifold (S¹)
- `map_to_torus(activity, positions)` — Map 2D activity to torus manifold (T²)
- `generate_shape_distance_datasets(...)` — SVD-based cluster embedding for shape metric validation
