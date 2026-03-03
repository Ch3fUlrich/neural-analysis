---
name: 'PlotGrid & Plotting'
description: 'Rules for the PlotGrid visualization system and all plot code'
applyTo: 'src/neural_analysis/plotting/**'
---

# Plotting Instructions

## Architecture

The plotting system has three layers:

```
Layer 3: Convenience functions  (plot_line, plot_scatter_2d, plot_violin, ...)
    ↓ builds PlotSpec objects
Layer 2: PlotGrid orchestrator  (grid_config.py + grid_dispatch.py)
    ↓ dispatches per plot_type via renderer registries
Layer 1: Renderer functions     (renderers_matplotlib.py + renderers_plotly.py)
    ↓ calls matplotlib/plotly APIs
```

`renderers.py` is a facade re-exporting from both renderer files.

All 16 plot types must work in **both** matplotlib and plotly backends.

## Rules

- Every new plot type needs:
  1. A `render_{type}_matplotlib(ax, ...)` function in `renderers_matplotlib.py`
  2. A `render_{type}_plotly(data, ...)` function in `renderers_plotly.py`
  3. Registration in `MATPLOTLIB_RENDERERS` and `PLOTLY_RENDERERS` dicts in `grid_dispatch.py`
  4. A convenience function in the appropriate module (`plots_1d`, `plots_2d`, `plots_3d`, `statistical_plots`, or `heatmaps`)
  5. Tests for both backends in `tests/`
- Never call `matplotlib.pyplot` or `plotly` directly outside the renderer layer.
- Convenience functions must build `PlotSpec` objects and return `PlotGrid` results — they are thin wrappers.
- Use `PlotConfig` for cross-plot settings (title, axis labels, figure size, DPI, save path).
- Use `GridLayoutConfig` for multi-panel layouts; prefer `auto_size_grid(n_plots)` for automatic sizing.
- Use `ColorScheme` for group-aware palette management; avoid hardcoding colors.

## PlotSpec Fields

The `PlotSpec` dataclass has ~40 fields. Key categories:
- **Data:** `data` (dict/ndarray/DataFrame), `plot_type`, `subplot_position`
- **Styling:** `color`, `marker`, `marker_size`, `line_width`, `linestyle`, `alpha`
- **Coloring:** `cmap`, `colorbar`, `colorbar_label`, `color_by`, `colors`
- **Reference:** `hlines=[{y, color, linestyle, label}]`, `vlines=[{x, ...}]`
- **Annotations:** `annotations=[{text, xy, xytext, arrowprops}]`

## Special Modules

- `synthetic_plots.py` — Facade re-exporting from `synthetic_plots_1d.py`, `synthetic_plots_2d.py`, `synthetic_plots_3d.py`. Entry: `plot_synthetic_data(activity, metadata, ...)`.
- `shape_distance.py` — MDS embeddings of shape distance matrices.
- `embeddings.py` — 2D/3D embedding scatter plots with convex hull overlays.

## Renderer Registry

Dispatch uses `MATPLOTLIB_RENDERERS: dict[str, Callable]` and `PLOTLY_RENDERERS: dict[str, Callable]` registries in `grid_dispatch.py`. To add a new plot type, add an entry to both dicts pointing to the render function. Unknown plot types raise a descriptive `ValueError`.
