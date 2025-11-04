# Plotting Module Architecture & Code Reuse Guide

## Overview

The neural_analysis plotting module is organized in layers to maximize code reuse and minimize duplication. This document explains the architecture and how to properly use existing functions.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  High-Level Convenience Functions                       │
│  (statistical_plots.py, plots_1d/2d/3d.py, heatmaps.py)│
│  - plot_bar(), plot_violin(), plot_box()               │
│  - plot_line(), plot_scatter_2d(), plot_scatter_3d()   │
│  - plot_heatmap(), plot_trajectory_2d/3d()             │
│  → ALL use PlotGrid internally                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  PlotGrid System (grid_config.py)                       │
│  - PlotSpec: Metadata-driven plot specification        │
│  - PlotGrid: Multi-panel grid manager                   │
│  - create_subplot_grid(), add_trace_to_subplot()       │
│  - Literal type hints for type safety                  │
│  → Uses renderer functions                              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Renderer Functions (renderers.py)                      │
│  - render_scatter_matplotlib/plotly()                   │
│  - render_line_matplotlib/plotly()                      │
│  - render_violin_matplotlib/plotly() [ENHANCED]        │
│  - render_histogram_*, render_heatmap_*, etc.          │
│  → Direct matplotlib/plotly API calls                   │
└─────────────────────────────────────────────────────────┘
```

**Key Changes from Previous Architecture:**
- ✅ **Legacy standalone functions removed** - plots_1d/2d/3d now use PlotGrid
- ✅ **Subplot utilities integrated** - create_subplot_grid() now in grid_config.py
- ✅ **Consistent architecture** - ALL plotting functions use PlotGrid system
- ✅ **No code duplication** - Single source of truth for each plot type

## When to Use Each Layer

### 1. High-Level Functions (statistical_plots.py)
**Use when:** You need common statistical visualizations

```python
from neural_analysis.plotting import plot_violin, plot_bar

# Quick violin plot comparison
data = {'Control': arr1, 'Treatment': arr2}
fig = plot_violin(data, showbox=True, showpoints=True, backend='plotly')
```

**Available functions:**
- `plot_bar()` - Bar plots with error bars
- `plot_violin()` - Enhanced violin plots (box + individual points)
- `plot_box()` - Box plots
- `plot_grouped_distributions()` - Multi-group comparisons
- `plot_comparison_distributions()` - Side-by-side grids

### 2. PlotGrid System (grid_config.py)
**Use when:** You need custom multi-panel layouts or complex visualizations

```python
from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig

# Custom grid with multiple plot types
specs = [
    PlotSpec(data=data1, plot_type='scatter', subplot_position=0, label='A'),
    PlotSpec(data=data2, plot_type='scatter', subplot_position=0, label='B'),
    PlotSpec(data=data3, plot_type='violin', subplot_position=1, label='C'),
]
grid = PlotGrid(plot_specs=specs, layout=GridLayoutConfig(rows=1, cols=2))
fig = grid.plot()
```

**Key features:**
- `subplot_position`: Overlay multiple traces in same subplot
- Automatic legend deduplication
- Type-safe with `Literal` hints
- Backend agnostic (matplotlib ↔ plotly)

### 3. Renderer Functions (renderers.py)
**Use when:** Implementing new PlotGrid features or plot types

```python
from neural_analysis.plotting import renderers

# In a PlotGrid method:
def _plot_spec_matplotlib(self, spec, ax, legend_tracker):
    if spec.plot_type == 'scatter':
        renderers.render_scatter_matplotlib(
            ax=ax,
            data=spec.data,
            color=spec.color,
            marker=spec.marker or 'o',
            alpha=spec.alpha,
            label=label_to_use
        )
```

**Available renderers:**
- Scatter: `render_scatter_matplotlib()`, `render_scatter_plotly()`, `render_scatter3d_plotly()`
- Line: `render_line_matplotlib()`, `render_line_plotly()`
- Histogram: `render_histogram_matplotlib()`, `render_histogram_plotly()`
- Heatmap: `render_heatmap_matplotlib()`, `render_heatmap_plotly()`
- Violin: `render_violin_matplotlib()`, `render_violin_plotly()` [Enhanced!]
- Box: `render_box_matplotlib()`, `render_box_plotly()`
- Bar: `render_bar_plotly()`

**Enhanced Violin Plots:**
```python
# Matplotlib backend includes box + points
renderers.render_violin_matplotlib(
    ax, data, color='blue',
    showbox=True,     # Box plot on left side
    showpoints=True,  # Individual data points
    showmeans=True    # Mean line
)

# Plotly backend
renderers.render_violin_plotly(
    data, color='blue',
    showbox=True,
    showpoints=True,
    meanline={'visible': True}
)
```

### 4. Legacy Functions (plots_*.py)
**Use when:** You need standalone plots without PlotGrid overhead

```python
from neural_analysis.plotting import plot_scatter_2d, plot_line

# Simple standalone scatter plot
fig = plot_scatter_2d(x, y, colors=labels, backend='matplotlib')

# Simple line plot
fig = plot_line(data, x=time, std=std_data, backend='plotly')
```

**Note:** These functions predate PlotGrid and use direct API calls. Keep for backward compatibility but prefer PlotGrid for new code.

## Code Reuse Guidelines

### Rule 1: Check the Registry First

Before implementing new functionality:

```bash
# Generate/update registry
python3 scripts/generate_function_registry.py

# Check for existing functions
less docs/function_registry.md
```

The registry lists all 56+ functions across 10 modules with:
- Function signatures
- Return types
- One-line summaries
- File locations with line numbers

### Rule 2: Use Renderer Functions for Plot Types

**❌ DON'T:**
```python
# Duplicating rendering logic
def my_custom_plot():
    if backend == 'matplotlib':
        ax.scatter(data[:, 0], data[:, 1], c=color, s=size, alpha=alpha)
    else:
        trace = go.Scatter(x=data[:, 0], y=data[:, 1], ...)
```

**✅ DO:**
```python
# Reuse renderer functions
from neural_analysis.plotting import renderers

def my_custom_plot():
    if backend == 'matplotlib':
        renderers.render_scatter_matplotlib(ax, data, color, size=size, alpha=alpha)
    else:
        trace = renderers.render_scatter_plotly(data, color, size=size, alpha=alpha)
```

### Rule 3: Layer New Functionality Appropriately

**Adding a new plot type:**

1. **Add renderer functions** (renderers.py):
```python
def render_newtype_matplotlib(ax, data, **kwargs):
    """Low-level rendering for matplotlib."""
    # Matplotlib API calls here
    
def render_newtype_plotly(data, **kwargs):
    """Low-level rendering for plotly."""
    # Return plotly trace
```

2. **Integrate into PlotGrid** (grid_config.py):
```python
# In PlotType Literal
PlotType = Literal['scatter', 'line', ..., 'newtype']

# In _plot_spec_matplotlib
elif spec.plot_type == 'newtype':
    renderers.render_newtype_matplotlib(ax, spec.data, ...)

# In _plot_spec_plotly
elif spec.plot_type == 'newtype':
    return renderers.render_newtype_plotly(spec.data, ...)
```

3. **Add convenience function** (statistical_plots.py if appropriate):
```python
def plot_newtype(data, labels, config=None, backend=None, **kwargs):
    """High-level convenience function."""
    specs = [PlotSpec(data=data, plot_type='newtype', ...)]
    grid = PlotGrid(plot_specs=specs, backend=backend)
    return grid.plot()
```

### Rule 4: Avoid Duplication Between Modules

**Common duplication patterns to avoid:**

1. **Color handling:** Use `core.get_default_categorical_colors()` or `DEFAULT_COLORS`
2. **Layout application:** Use `core.apply_layout_matplotlib()` / `apply_layout_plotly()`
3. **Backend detection:** Use `backend.get_backend()`
4. **Data validation:** Extract to shared utility if used multiple times

### Rule 5: Document Shared Utilities

When creating reusable functions:
- Add to appropriate module (core.py, renderers.py, etc.)
- Include comprehensive docstring
- Add type hints
- Run registry generator to update docs

## Enhanced Features

### Enhanced Violin Plots (Added 2025-11-04)

The violin plot renderers now include:
- **Box plot on the left side** for quartile information
- **Individual data points** for full transparency
- **Mean/median lines** for central tendency

**Matplotlib implementation:**
```python
render_violin_matplotlib(
    ax, data,
    showbox=True,      # Box plot shifted left
    showpoints=True,   # Scattered points with jitter
    showmeans=True,    # Mean line
    showmedians=True   # Median line
)
```

**Plotly implementation:**
```python
render_violin_plotly(
    data,
    showbox=True,          # Integrated box plot
    showpoints=True,       # Points='all', positioned right
    meanline={'visible': True}  # Mean line
)
```

## Function Registry Automation

The function registry is auto-generated by AST parsing:

```bash
# Generate registry (both .md and .json)
python3 scripts/generate_function_registry.py

# Output files:
# - docs/function_registry.md (human-readable)
# - docs/function_registry.json (machine-readable for AI agents)
```

**Registry includes:**
- All public functions (excludes private _functions)
- Function signatures with parameter names
- Return type annotations
- First line of docstring as summary
- Module name and line number
- Organized by category

## Quick Reference

### Adding New Statistical Plot

```python
# 1. Add to statistical_plots.py
def plot_mynewplot(data, labels=None, config=None, backend=None, **kwargs):
    specs = []
    for label, arr in data.items():
        specs.append(PlotSpec(
            data=arr,
            plot_type='mynewplot',  # Must be valid PlotType
            label=label,
            **kwargs
        ))
    grid = PlotGrid(plot_specs=specs, backend=backend, config=config)
    return grid.plot()

# 2. Add PlotType to grid_config.py if needed
PlotType = Literal['scatter', 'line', ..., 'mynewplot']

# 3. Add renderer if new type
render_mynewplot_matplotlib(ax, data, **kwargs)
render_mynewplot_plotly(data, **kwargs)

# 4. Update registry
python3 scripts/generate_function_registry.py
```

### Using PlotGrid for Multi-Panel

```python
from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig

# Create specifications
specs = [
    PlotSpec(data=data1, plot_type='violin', subplot_position=0, title='Group A'),
    PlotSpec(data=data2, plot_type='violin', subplot_position=1, title='Group B'),
    PlotSpec(data=data3, plot_type='box', subplot_position=2, title='Group C'),
]

# Create grid
grid = PlotGrid(
    plot_specs=specs,
    layout=GridLayoutConfig(rows=1, cols=3),
    backend='plotly'
)

# Generate
fig = grid.plot()
fig.show()
```

### Overlaying Traces

```python
# Multiple traces in same subplot using subplot_position
specs = [
    PlotSpec(data=control, plot_type='scatter', subplot_position=0, label='Control'),
    PlotSpec(data=treatment, plot_type='scatter', subplot_position=0, label='Treatment'),
    # Both in subplot 0 - will be overlaid with legend deduplication
]
```

## Summary

**Key Principles:**
1. **PlotGrid for new code** - Always prefer PlotGrid system
2. **Renderers for primitives** - Use renderer functions, don't duplicate
3. **Check registry first** - Avoid reimplementing existing functions
4. **Layer appropriately** - Keep renderer → PlotGrid → convenience hierarchy
5. **Update registry** - Run generator after adding functions
6. **Type safety** - Use Literal type hints for plot_type, backend

**This architecture ensures:**
- ✅ No code duplication
- ✅ Consistent behavior across plots
- ✅ Easy maintenance and extension
- ✅ Clear separation of concerns
- ✅ Type-safe interfaces
- ✅ Backend flexibility
