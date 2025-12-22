# `docs/plotgrid.md` (restructured outline)

```markdown
# PlotGrid Overview

PlotGrid provides a metadata‑driven, backend‑agnostic plotting layer for this project. It standardises layout, styling, and backends (Matplotlib / Plotly) so analyses and examples share the same visual language.

---

## 1. Quick Start

**Most common pattern: 1×2 scatter grid**

```python
from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig

specs = [
    PlotSpec(data=data1, plot_type="scatter", label="A", color="blue"),
    PlotSpec(data=data2, plot_type="scatter", label="B", color="red"),
]
layout = GridLayoutConfig(rows=1, cols=2)

grid = PlotGrid(plot_specs=specs, layout=layout)
fig = grid.plot()
```

**High‑level helpers**

```python
from neural_analysis.plotting import plot_bar, plot_violin

fig = plot_bar(data={"A": arr1, "B": arr2})
```

Use these patterns for new code; do not call Matplotlib/Plotly directly except for minimal post‑processing.

---

## 2. Concept Map

| Concept        | Module / class                | Role                                      |
|----------------|-------------------------------|-------------------------------------------|
| Renderers      | `renderers.py`               | Backend‑specific drawing primitives       |
| Plot specs     | `PlotSpec`                   | Description of a single panel             |
| Grid layout    | `GridLayoutConfig`           | Rows/cols, sizing, shared axes            |
| Plot grid      | `PlotGrid`                   | Orchestrates specs + layout               |
| 1D plots       | `plots_1d.py`                | Histograms, bar plots, line plots         |
| 2D plots       | `plots_2d.py`                | Scatter, heatmaps, joint plots            |
| 3D plots       | `plots_3d.py`                | 3D scatter/surfaces (if enabled)          |
| Helpers        | `plot_bar`, `plot_violin`,…  | Convenience functions over PlotGrid       |

---

## 3. Core API

### 3.1 PlotSpec

Minimal usage:

```python
spec = PlotSpec(
    data=data_array_or_frame,
    plot_type="scatter",
    label="Condition A",
    color="tab:blue",
)
```

Key fields (typical):

- `data`: NumPy array, pandas DataFrame, or similar.
- `plot_type`: literal type string (`"scatter"`, `"line"`, `"hist"`, …).
- Optional styling: `label`, `color`, markers, alpha, etc.

---

### 3.2 GridLayoutConfig

Define the layout and shared axes:

```python
layout = GridLayoutConfig(
    rows=2,
    cols=2,
    share_x=True,
    share_y=True,
)
```

Typical fields:

- `rows`, `cols` – grid size.
- `share_x`, `share_y` – axis sharing.
- Optional margins and spacing, if needed.

---

### 3.3 PlotGrid

Combine specs and layout:

```python
grid = PlotGrid(plot_specs=specs, layout=layout)
fig = grid.plot()
```

Responsibilities:

- Build the backend figure.
- Deduplicate legends.
- Apply consistent styling and themes.

---

## 4. Helper Functions

High‑level helpers wrap PlotGrid for common tasks.

| Helper          | Typical input                 | Purpose                        |
|-----------------|-------------------------------|--------------------------------|
| `plot_bar`      | dict label → array            | Categorical distribution       |
| `plot_violin`   | dict label → array            | Distribution comparison        |
| `plot_line`     | x / y arrays or DataFrame     | Trajectories / time series     |

All helpers internally create `PlotSpec` objects and a `PlotGrid`. Prefer them in notebooks and simple scripts.

---

## 5. Backend Behaviour

- Backends are selected centrally (e.g. config flag or environment variable).
- PlotGrid uses renderers to construct either Matplotlib or Plotly figures.
- New code should not import Matplotlib/Plotly directly; go via PlotGrid.

When backend‑specific tuning is required (e.g. adding an annotation), perform small adjustments on the figure object returned by `grid.plot()`.

---

## 6. Patterns and Recipes

Keep this section short and task‑oriented; each recipe is 5–15 lines of code.

Example recipe types:

- “Row of violin plots for several conditions”
- “2×2 grid with shared colourbar”
- “Side‑by‑side Matplotlib and Plotly exports from the same spec”

(Insert concrete project‑specific recipes here.)

---

## 7. Advanced Usage (Optional)

For most users/agents, the sections above are sufficient. Place advanced details here:

- Custom renderers or new `plot_type` extensions.
- Backend‑specific configuration (themes, style sheets).
- Integration with interactive notebooks (e.g. Plotly widgets).

Keep this section concise and link to code where possible rather than repeating full examples.