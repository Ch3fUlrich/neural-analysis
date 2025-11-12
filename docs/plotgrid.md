# PlotGrid System - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Classes](#core-classes)
4. [Usage Examples](#usage-examples)
5. [Advanced Features](#advanced-features)
6. [Supported Plot Types](#supported-plot-types)
7. [Best Practices](#best-practices)
8. [Migration Guide](#migration-guide)
9. [Architecture](#architecture)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## Overview

The PlotGrid system is a flexible, metadata-driven plotting framework for creating complex multi-panel visualizations with minimal code. It provides a single, unified API that works for everything from simple single plots to complex multi-panel layouts with multiple traces per subplot.

### What is PlotGrid?

PlotGrid is the **unified plotting interface** for the neural-analysis package, providing:

- **Unified Interface**: Single entry point for all plot types (scatter, line, bar, violin, box, heatmap, trajectory, KDE, etc.)
- **Multi-Backend Support**: Seamlessly switch between matplotlib (publication-quality) and plotly (interactive)
- **Flexible Layouts**: Easy multi-panel subplot arrangements with automatic sizing
- **Consistent Styling**: Color schemes and configurations applied uniformly
- **Metadata-Driven**: Declarative configuration instead of imperative code
- **Multiple Traces Per Subplot**: Overlay multiple data series with automatic legend generation
- **Extensible**: Easy to add new plot types

### Key Benefits

**For Users:**
- **Faster development**: 75% less code, clearer intent
- **Fewer errors**: Automatic validation and sensible defaults
- **More flexibility**: DataFrame-driven enables rich metadata
- **Better plots**: Automatic color schemes, spacing, and layouts

**For Maintainers:**
- **Modular**: Easy to extend with new plot types
- **No duplication**: Single source of truth for each plot type
- **Documented**: Comprehensive docs and examples
- **Consistent**: Single API across all plotting needs

---

## Quick Start

### Installation

PlotGrid is included in the neural-analysis package:

```bash
uv add neural-analysis  # or pip install neural-analysis
```

### Simple Example

```python
import numpy as np
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot specification
spec = PlotSpec(
    data=np.column_stack([x, y]),
    plot_type='line',
    color='blue',
    label='sin(x)'
)

# Create grid and plot
grid = PlotGrid(
    plot_specs=[spec],
    config=PlotConfig(title='Simple Line Plot'),
    backend='matplotlib'  # or 'plotly'
)

fig = grid.plot()
```

### Multi-Panel Example

```python
# Compare multiple datasets side-by-side
data_dict = {
    'Method A': result_a,
    'Method B': result_b,
    'Method C': result_c,
}

fig = plot_comparison_grid(
    data_dict,
    plot_type='scatter',
    rows=1,
    cols=3,
    config=PlotConfig(title="Method Comparison"),
    backend='plotly'
)
```

---

## Core Classes

### PlotSpec

Specification for a single plot element with all styling information.

```python
from dataclasses import dataclass

@dataclass
class PlotSpec:
    # Core parameters
    data: np.ndarray | pd.DataFrame  # Data to plot
    plot_type: str                    # 'scatter', 'line', 'histogram', etc.
    subplot_position: int | None      # Group multiple traces in same subplot
    title: str | None                 # Subplot title
    label: str | None                 # Legend label
    
    # Styling
    color: str | None                 # Color (name, hex, rgb)
    marker_size: float | None         # Marker/point size
    line_width: float | None          # Line width
    linestyle: str | None             # Line style ('-', '--', '-.', ':')
    marker: str | None                # Marker style ('o', 's', '^', etc.)
    alpha: float = 0.7                # Transparency (0-1)
    
    # Advanced features (for trajectory, KDE, grouped plots)
    color_by: Literal["time"] | None = None  # Coloring strategy
    show_points: bool = False         # Show scatter points on trajectories
    cmap: str | None = None           # Colormap name
    colorbar: bool = False            # Show colorbar
    colorbar_label: str | None = None # Colorbar label
    colors: np.ndarray | list | None = None  # Array of colors
    sizes: np.ndarray | float | None = None  # Array of sizes
    show_hulls: bool = False          # Show convex hulls (grouped scatter)
    fill: bool = True                 # Fill contours (KDE)
    n_levels: int = 10                # Contour levels (KDE)
    bandwidth: float | None = None    # KDE bandwidth
    equal_aspect: bool = False        # Equal aspect ratio
    
    # Custom parameters (axis labels, grid, value display, etc.)
    kwargs: dict = field(default_factory=dict)
```

**Key Innovation**: The `subplot_position` parameter allows multiple traces in the same subplot for easy comparisons.

### GridLayoutConfig

Configuration for grid layout and subplot arrangement.

```python
@dataclass
class GridLayoutConfig:
    rows: int | None = None           # Number of rows (auto if None)
    cols: int | None = None           # Number of columns (auto if None)
    subplot_titles: list[str] | None = None  # Titles for each subplot
    shared_xaxes: bool | str = False  # Share x-axes ('rows', 'cols', 'all', or False)
    shared_yaxes: bool | str = False  # Share y-axes
    vertical_spacing: float | None = None    # Vertical spacing (0-1)
    horizontal_spacing: float | None = None  # Horizontal spacing (0-1)
    group_by: str | None = None       # DataFrame column for grouping
```

**Auto-sizing**: If `rows` and `cols` are None, PlotGrid automatically determines optimal grid size.

### ColorScheme

Color scheme for grouped plots with automatic color assignment.

```python
@dataclass
class ColorScheme:
    palette: str | list[str] = 'tab10'  # Color palette name or list
    group_colors: dict | None = None     # Explicit color mapping
    alpha: float = 0.7                   # Default transparency
```

**Built-in palettes**: tab10, viridis, plasma, inferno, magma, cividis, Set1, Set2, Set3

### PlotGrid

Main class for creating grid-based plots with three construction methods.

```python
class PlotGrid:
    def __init__(
        self,
        plot_specs: list[PlotSpec],
        config: PlotConfig | None = None,
        layout: GridLayoutConfig | None = None,
        color_scheme: ColorScheme | None = None,
        backend: Literal['matplotlib', 'plotly'] = 'matplotlib'
    ):
        """Create PlotGrid from list of PlotSpec objects."""
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        group_by: str | None = None,
        **kwargs
    ) -> PlotGrid:
        """Create PlotGrid from structured DataFrame."""
    
    @classmethod
    def from_dict(
        cls,
        data_dict: dict[str, np.ndarray],
        plot_type: str = 'scatter',
        **kwargs
    ) -> PlotGrid:
        """Create PlotGrid from simple dictionary."""
    
    def add_plot(
        self,
        data: np.ndarray,
        plot_type: str,
        **kwargs
    ) -> None:
        """Add plot dynamically."""
    
    def plot(self) -> Figure:
        """Generate the figure with all subplots."""
```

---

## Usage Examples

### Example 1: Multi-Trace Subplot

Multiple datasets in the same subplot for direct comparison.

```python
from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig

# Multiple traces in same subplot using subplot_position
specs = [
    PlotSpec(
        data=control_data,
        plot_type='scatter',
        subplot_position=0,  # All in same subplot
        label='Control',
        color='blue',
        alpha=0.5
    ),
    PlotSpec(
        data=treatment_data,
        plot_type='scatter',
        subplot_position=0,  # Same position = same subplot
        label='Treatment',
        color='red',
        alpha=0.5
    ),
]

grid = PlotGrid(
    plot_specs=specs,
    layout=GridLayoutConfig(rows=1, cols=1)
)
fig = grid.plot()
```

### Example 2: DataFrame-Based Configuration

Create plots from structured data with metadata.

```python
import pandas as pd

# Create DataFrame with plot specifications
df = pd.DataFrame({
    'data': [data1, data2, data3, data4],
    'plot_type': ['scatter', 'scatter', 'line', 'histogram'],
    'title': ['Exp 1', 'Exp 2', 'Timeseries', 'Distribution'],
    'group': ['control', 'treatment', 'control', 'treatment']
})

# Automatic color assignment by group
grid = PlotGrid.from_dataframe(
    df,
    group_by='group',  # Colors assigned per group
    layout=GridLayoutConfig(rows=2, cols=2)
)
fig = grid.plot()
```

### Example 3: Mixed Plot Types

Different plot types in one grid for comprehensive visualization.

```python
# Different plot types in one grid
specs = [
    PlotSpec(
        data=timeseries,
        plot_type='line',
        title='Neural Activity',
        color='blue',
        line_width=2
    ),
    PlotSpec(
        data=distribution,
        plot_type='histogram',
        title='Firing Rate Distribution',
        color='green'
    ),
    PlotSpec(
        data=correlation_matrix,
        plot_type='heatmap',
        title='Correlation Matrix'
    ),
]

grid = PlotGrid(
    plot_specs=specs,
    layout=GridLayoutConfig(rows=1, cols=3)
)
fig = grid.plot()
```

### Example 4: Complex Multi-Panel with Groups

Multiple scenarios with overlaid distributions.

```python
# 4 scenarios, each with 2 distributions overlaid
plot_specs = []

for i, (name, data_pair) in enumerate(scenarios.items()):
    dist1, dist2 = data_pair
    
    # First distribution
    plot_specs.append(PlotSpec(
        data=dist1,
        plot_type='scatter',
        subplot_position=i,  # Group by scenario
        title=name,
        label='Distribution 1',
        color='blue',
        alpha=0.4
    ))
    
    # Second distribution (same subplot)
    plot_specs.append(PlotSpec(
        data=dist2,
        plot_type='scatter',
        subplot_position=i,  # Same as above
        label='Distribution 2',
        color='red',
        alpha=0.4
    ))

grid = PlotGrid(
    plot_specs=plot_specs,
    layout=GridLayoutConfig(rows=2, cols=2)
)
fig = grid.plot()
```

### Example 5: Trajectory Plot with Time Coloring

2D trajectory with time-gradient coloring.

```python
import numpy as np

# Generate trajectory data
t = np.linspace(0, 4*np.pi, 200)
x = np.sin(t)
y = np.cos(t)

# Create trajectory spec
spec = PlotSpec(
    data={'x': x, 'y': y},
    plot_type='trajectory',
    color_by="time",           # Color line by time progression
    show_points=True,          # Show scatter points along trajectory
    cmap='viridis',           # Colormap for time gradient
    line_width=2.0,
    alpha=0.8,
    colorbar=True,            # Show colorbar
    colorbar_label='Time',
    equal_aspect=True         # Equal aspect ratio
)

# Create plot
config = PlotConfig(title='2D Trajectory with Time Gradient')
grid = PlotGrid(plot_specs=[spec], config=config)
fig = grid.plot()
```

### Example 6: KDE Density Plot

2D kernel density estimation with contours.

```python
# Generate random data
x = np.random.randn(500)
y = np.random.randn(500)

# Create KDE spec
spec = PlotSpec(
    data={'x': x, 'y': y},
    plot_type='kde',
    fill=True,                # Fill contours
    n_levels=15,              # Number of contour levels
    cmap='Blues',            # Colormap
    alpha=0.7,
    show_points=True,         # Show underlying scatter points
    marker_size=3,
    bandwidth=None            # Use Scott's rule for bandwidth
)

# Create plot
config = PlotConfig(title='2D KDE Density Plot')
grid = PlotGrid(plot_specs=[spec], config=config)
fig = grid.plot()
```

### Example 7: Performance Benchmarking with Custom Styling

Colored bars with legends, value labels, and custom ticks.

```python
# Create colored bars with legends and value labels
colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'mediumpurple']
method_names = ['IQR', 'ZSCORE', 'LOF', 'ISOLATION', 'ELLIPTIC']
times = [0.123, 0.156, 0.234, 0.189, 0.201]

plot_specs = []
for idx, (name, time_val, color) in enumerate(zip(method_names, times, colors)):
    plot_specs.append(PlotSpec(
        data=np.array([time_val]),
        plot_type='bar',
        subplot_position=0,
        color=color,
        label=name,
        kwargs={
            'x': [idx],
            'show_values': True,
            'value_format': '.3f',
            'x_label': 'Method',
            'y_label': 'Execution Time (seconds)',
            'grid': {'axis': 'y', 'alpha': 0.3}
        }
    ))

# Set tick labels on the last bar spec
plot_specs[-1].kwargs['set_xticks'] = list(range(len(method_names)))
plot_specs[-1].kwargs['set_xticklabels'] = method_names

grid = PlotGrid(
    plot_specs=plot_specs,
    config=PlotConfig(
        title="Method Performance Comparison",
        figsize=(12, 6),
        legend=True
    ),
    backend='matplotlib'
)

fig = grid.plot()
```

---

## Advanced Features

### 1. Time-Gradient Coloring

Automatic time-based color gradients for trajectories using LineCollection (matplotlib) or gradient arrays (plotly).

```python
spec = PlotSpec(
    data={'x': x, 'y': y},
    plot_type='trajectory',    # or 'trajectory3d'
    color_by="time",           # Time-based coloring
    cmap='viridis',           # Colormap
    colorbar=True,            # Show colorbar
    colorbar_label='Time (s)'
)
```

**Supported for**: `trajectory` (2D), `trajectory3d` (3D)

### 2. Convex Hull Rendering

Automatic boundary computation for grouped data using scipy's ConvexHull.

```python
spec = PlotSpec(
    data={
        'Group A': (x1, y1),
        'Group B': (x2, y2)
    },
    plot_type='grouped_scatter',
    show_hulls=True,          # Show convex hulls around groups
    marker_size=30,
    alpha=0.6
)
```

**Supported for**: `grouped_scatter`, `convex_hull`

### 3. KDE Density Plots

Gaussian kernel density estimation with customizable contours.

```python
spec = PlotSpec(
    data={'x': x, 'y': y},
    plot_type='kde',
    fill=True,                # Fill contours
    n_levels=15,              # Number of contour levels
    show_points=True,         # Show underlying scatter
    bandwidth=None            # Auto-bandwidth (Scott's rule)
)
```

**Supported for**: `kde`

### 4. Custom Color Schemes

Define explicit color mappings or use built-in palettes.

```python
from neural_analysis.plotting import ColorScheme

# Define custom colors
custom_colors = ColorScheme(
    palette=['#FF0000', '#00FF00', '#0000FF', '#FFFF00'],
    alpha=0.8
)

# Or explicit mapping
custom_colors = ColorScheme(
    group_colors={
        'control': 'blue',
        'treatment': 'red',
        'baseline': 'green'
    }
)

grid = PlotGrid(
    plot_specs=specs,
    color_scheme=custom_colors
)
```

### 5. Shared Axes

Share axes across rows or columns for coordinated viewing.

```python
# Share x-axes across rows
layout = GridLayoutConfig(
    rows=3,
    cols=2,
    shared_xaxes='rows',  # 'rows', 'cols', 'all', or False
    shared_yaxes=False
)

grid = PlotGrid(plot_specs=specs, layout=layout)
```

### 6. Custom Spacing

Control gap between subplots.

```python
layout = GridLayoutConfig(
    rows=2,
    cols=2,
    vertical_spacing=0.15,     # 15% vertical gap
    horizontal_spacing=0.1,    # 10% horizontal gap
)
```

### 7. Dynamic Plot Addition

Add plots dynamically after grid creation.

```python
grid = PlotGrid(config=PlotConfig(title="Dynamic Grid"))

# Add plots dynamically
grid.add_plot(data1, plot_type='scatter', title='Plot 1')
grid.add_plot(data2, plot_type='line', title='Plot 2')
grid.add_plot(data3, plot_type='histogram', title='Plot 3')

fig = grid.plot()
```

### 8. Custom Parameters via kwargs

Pass custom parameters through PlotSpec.kwargs for advanced control.

```python
spec = PlotSpec(
    data=data,
    plot_type='bar',
    kwargs={
        'show_values': True,        # Display value labels
        'value_format': '.2f',      # Format string
        'x_label': 'Category',      # X-axis label
        'y_label': 'Value',         # Y-axis label
        'grid': {'axis': 'y', 'alpha': 0.3},  # Gridlines
        'set_xticks': [0, 1, 2],    # Custom ticks
        'set_xticklabels': ['A', 'B', 'C']  # Custom labels
    }
)
```

**Available kwargs**:
- `show_values`: Display values on bars/points
- `value_format`: Format string (e.g., `.3f`)
- `x_label`, `y_label`: Axis labels
- `grid`: Gridline configuration
- `set_xticks`, `set_xticklabels`: Custom tick control
- `set_yticks`, `set_yticklabels`: Custom y-tick control

---

## Supported Plot Types

### Basic Plot Types

| Plot Type | Description | Data Format |
|-----------|-------------|-------------|
| `scatter` | 2D scatter plot | `(n, 2)` array or `{'x': ..., 'y': ...}` |
| `scatter3d` | 3D scatter plot | `(n, 3)` array or `{'x': ..., 'y': ..., 'z': ...}` |
| `line` | Line plot | `(n,)` or `(n, 2)` array |
| `histogram` | Histogram | `(n,)` array |
| `heatmap` | 2D heatmap | `(rows, cols)` array |
| `bar` | Bar chart | `(n,)` array |
| `violin` | Violin plot | `(n,)` array or dict of arrays |
| `box` | Box plot | `(n,)` array or dict of arrays |

### Advanced 2D Plot Types

| Plot Type | Description | Special Features |
|-----------|-------------|-----------------|
| `trajectory` | 2D trajectory | Time-gradient coloring, LineCollection |
| `kde` | 2D KDE density | Contours, bandwidth control |
| `grouped_scatter` | Multi-group scatter | Convex hulls, per-group colors |
| `convex_hull` | Hull boundaries | Standalone boundary rendering |

### 3D Plot Types

| Plot Type | Description | Special Features |
|-----------|-------------|-----------------|
| `trajectory3d` | 3D trajectory | Time-gradient coloring, Line3DCollection |

**Backend Support**:
- ✅ **matplotlib**: All plot types supported
- ✅ **plotly**: All plot types supported

---

## Best Practices

### 1. Use subplot_position for Comparisons

When comparing multiple conditions/methods in the same subplot:

```python
# Good: Use subplot_position
specs = [
    PlotSpec(data=control, subplot_position=0, label='Control'),
    PlotSpec(data=treatment, subplot_position=0, label='Treatment'),
]
```

### 2. Use DataFrames for Complex Metadata

When you have rich metadata (groups, categories, labels):

```python
# Good: DataFrame with metadata
df = pd.DataFrame({
    'data': [...],
    'title': [...],
    'group': [...],
    'condition': [...]
})
grid = PlotGrid.from_dataframe(df, group_by='group')
```

### 3. Let Grid Auto-Size

Unless you have specific requirements:

```python
# Good: Let it auto-size
layout = GridLayoutConfig()  # Auto-determines rows/cols

# Manual override only when needed
layout = GridLayoutConfig(rows=2, cols=3)
```

### 4. Use Convenience Functions for Simple Cases

```python
# Simple comparison: use convenience function
fig = plot_comparison_grid(data_dict, plot_type='histogram')

# Complex multi-trace: use PlotGrid directly
grid = PlotGrid(plot_specs=complex_specs)
```

### 5. Custom Parameters via kwargs

Pass axis labels, gridlines, value display through `PlotSpec.kwargs`:

```python
# Good: Use kwargs for custom parameters
spec = PlotSpec(
    data=data,
    plot_type='bar',
    kwargs={
        'x_label': 'Method',
        'y_label': 'Time (s)',
        'show_values': True,
        'grid': {'axis': 'y'}
    }
)
```

**Why?** Custom parameters are popped in `grid_config.py` before calling renderers, preventing matplotlib errors.

### 6. Never Use Direct Matplotlib

**Rule**: Never use direct matplotlib or pyplot plotting functions (`ax.plot`, `ax.bar`, `plt.scatter`, etc.). Always use the PlotGrid system.

```python
# ❌ Bad: Direct matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y)

# ✅ Good: PlotGrid
spec = PlotSpec(data=np.column_stack([x, y]), plot_type='line')
grid = PlotGrid(plot_specs=[spec])
fig = grid.plot()
```

---

## Migration Guide

### From Manual Subplot Management

**Old Way (Manual)**:
```python
fig = create_subplot_grid(rows=2, cols=2, backend='plotly')

for i, data in enumerate(datasets):
    row = (i // 2) + 1
    col = (i % 2) + 1
    trace = go.Scatter(x=data[:, 0], y=data[:, 1])
    add_trace_to_subplot(fig, trace, row=row, col=col)
```

**New Way (PlotGrid)**:
```python
specs = [
    PlotSpec(data=data, plot_type='scatter', title=f'Dataset {i}')
    for i, data in enumerate(datasets)
]

grid = PlotGrid(plot_specs=specs, layout=GridLayoutConfig(rows=2, cols=2))
fig = grid.plot()
```

**Result**: ~75% reduction in code, clearer intent.

### From Old Convenience Functions

The convenience functions in `plots_2d.py` and `plots_3d.py` still work but can be replicated with PlotGrid:

**Old Way**:
```python
from neural_analysis.plotting import plot_trajectory_2d
fig = plot_trajectory_2d(x, y, color_by="time")
```

**PlotGrid Way**:
```python
from neural_analysis.plotting import PlotGrid, PlotSpec
spec = PlotSpec(data={'x': x, 'y': y}, plot_type='trajectory', color_by="time")
grid = PlotGrid(plot_specs=[spec])
fig = grid.plot()
```

**Recommendation**:
- **New code**: Use PlotGrid for flexibility and consistency
- **Legacy code**: Convenience functions remain available for compatibility

### From Direct Matplotlib

**Old Way**:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)
plt.show()
```

**New Way (PlotGrid)**:
```python
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

spec = PlotSpec(
    data=np.column_stack([x, y]),
    plot_type='line',
    kwargs={
        'x_label': 'X',
        'y_label': 'Y',
        'grid': True
    }
)

grid = PlotGrid(
    plot_specs=[spec],
    config=PlotConfig(title='My Plot'),
    backend='matplotlib'
)

grid.plot()
```

---

## Architecture

### Design Overview

PlotGrid uses a layered architecture with clear separation of concerns:

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
│  - render_violin_matplotlib/plotly()                    │
│  - render_histogram_*, render_heatmap_*, etc.          │
│  → Direct matplotlib/plotly API calls                   │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Custom Parameter Handling

Custom parameters (x_label, y_label, grid, show_values, etc.) are:
- Passed via `PlotSpec.kwargs`
- **Popped in `grid_config.py`** before calling renderers
- Applied to axes **after** rendering completes
- This prevents matplotlib errors from unknown parameters

#### 2. Renderer Architecture

**`renderers.py`** contains low-level rendering functions:
- One function per plot type per backend
- Accept only plot-specific parameters
- Return matplotlib artists or plotly traces
- No axis customization (labels, grids, etc.)

**`grid_config.py`** orchestrates rendering:
- Creates subplot layouts
- Calls appropriate renderers
- Applies custom settings (labels, grids, ticks)
- Manages legends and color schemes

#### 3. Backend Abstraction

Same API works for both matplotlib and plotly:
- User specifies `backend='matplotlib'` or `backend='plotly'`
- PlotGrid routes to appropriate renderer functions
- Backend-specific implementation hidden from user

#### 4. subplot_position Innovation

Allows multiple traces in same subplot:
- All specs with same `subplot_position` are grouped
- Each trace added to same axes/subplot
- Automatic legend generation
- Perfect for comparing conditions

### Files Structure

**Created/Modified**:
1. `src/neural_analysis/plotting/grid_config.py` (~750 lines)
   - PlotSpec, GridLayoutConfig, ColorScheme, PlotGrid classes
   - Convenience functions
   - Complete docstrings

2. `src/neural_analysis/plotting/renderers.py`
   - Low-level rendering functions for all plot types
   - Matplotlib and plotly implementations

3. `src/neural_analysis/plotting/plots_1d.py`, `plots_2d.py`, `plots_3d.py`
   - Refactored to use PlotGrid internally
   - ~220 lines of duplicate code removed
   - All convenience functions still work

4. `src/neural_analysis/plotting/statistical_plots.py`
   - Uses PlotGrid for all statistical visualizations
   - Enhanced with show_values, value_format, custom ticks

---

## Troubleshooting

### Issue: Colors not showing correctly

**Solution**: Ensure color format is valid (name, hex, or rgb string)

```python
color='blue'        # ✓ Named color
color='#FF0000'     # ✓ Hex
color='rgb(255,0,0)' # ✓ RGB string
```

### Issue: Multiple traces not appearing in same subplot

**Solution**: Use same `subplot_position` value

```python
PlotSpec(..., subplot_position=0)  # First subplot
PlotSpec(..., subplot_position=0)  # Same subplot
```

### Issue: Auto-sizing creates wrong grid

**Solution**: Explicitly set rows/cols

```python
layout = GridLayoutConfig(rows=2, cols=3)  # Force 2x3 grid
```

### Issue: Plot types not supported

**Solution**: Check supported types and data format

```python
# Supported: scatter, line, histogram, heatmap, scatter3d, bar, 
#            violin, box, trajectory, trajectory3d, kde, 
#            grouped_scatter, convex_hull

# Data format:
# - scatter: (n, 2) array or {'x': ..., 'y': ...}
# - scatter3d: (n, 3) array or {'x': ..., 'y': ..., 'z': ...}
# - line: (n,) or (n, 2) array
# - histogram: (n,) array
# - heatmap: (rows, cols) array
```

### Issue: Custom parameters not working

**Solution**: Pass through `PlotSpec.kwargs`, not as PlotSpec attributes

```python
# ❌ Bad: Direct attribute (will cause error)
spec = PlotSpec(data=data, plot_type='bar', x_label='X')

# ✅ Good: Via kwargs
spec = PlotSpec(
    data=data,
    plot_type='bar',
    kwargs={'x_label': 'X'}
)
```

### Issue: ValueError about data shape

**Solution**: Ensure data format matches plot type

```python
# For scatter: (n, 2) or {'x': x_arr, 'y': y_arr}
# For line: (n,) or (n, 2)
# For trajectory: {'x': x_arr, 'y': y_arr}

# If error persists, check for NaN/inf values
data = data[~np.isnan(data).any(axis=1)]  # Remove NaN rows
```

---

## API Reference

### Main Classes

- **`PlotSpec`**: Specification for single plot element
- **`GridLayoutConfig`**: Grid layout configuration
- **`ColorScheme`**: Color scheme for grouped plots
- **`PlotGrid`**: Main entry point for creating plots

### Convenience Functions

- **`plot_comparison_grid()`**: Quick multi-panel comparison
- **`plot_grouped_comparison()`**: Overlaid plots by category
- **`plot_bar()`**: Bar plots with error bars
- **`plot_violin()`**: Enhanced violin plots
- **`plot_box()`**: Box plots
- **`plot_line()`**: Line plots
- **`plot_scatter_2d()`**: 2D scatter plots
- **`plot_scatter_3d()`**: 3D scatter plots
- **`plot_trajectory_2d()`**: 2D trajectories
- **`plot_trajectory_3d()`**: 3D trajectories
- **`plot_kde_2d()`**: 2D KDE density plots
- **`plot_grouped_scatter_2d()`**: Grouped scatter with hulls
- **`plot_heatmap()`**: Heatmaps

### Complete API

See docstrings in `src/neural_analysis/plotting/grid_config.py` for complete API documentation.

---

## Performance Considerations

- **Large Grids**: For >20 subplots, use matplotlib backend (faster rendering)
- **Interactive Plots**: Use plotly for interactive exploration (<10 subplots recommended)
- **Memory**: Each PlotSpec holds data reference, not copy (efficient)
- **Batch Creation**: Use DataFrame approach for creating many similar plots

---

## Examples and Notebooks

Complete working examples are available in:
- `examples/neural_analysis_demo.ipynb` - Comprehensive demo with benchmarking
- `examples/plotting_grid_showcase.ipynb` - PlotGrid feature showcase
- `examples/statistical_plots_examples.ipynb` - Statistical plot examples
- `examples/metrics_examples.ipynb` - PlotGrid usage throughout

---

## Future Enhancements

Planned features:
- [ ] Animation support across subplots
- [ ] Automatic subplot positioning based on similarity
- [ ] Export to multi-page PDF
- [ ] Interactive subplot linking (zoom/pan synchronized)
- [ ] Custom plot type registration system
- [ ] Integration with statistical testing (add p-values to plots)
- [ ] More colormap options
- [ ] Plot templates for common use cases

---

## Contributing

To add new plot types:

1. Add type to `PlotSpec.plot_type` Literal in `grid_config.py`
2. Implement `render_<type>_matplotlib()` in `renderers.py`
3. Implement `render_<type>_plotly()` in `renderers.py`
4. Update `_plot_spec_matplotlib()` in `grid_config.py`
5. Update `_plot_spec_plotly()` in `grid_config.py`
6. Add tests and documentation
7. Update this document with usage examples

---

## Version History

- **v0.3.0** (2025-01): PlotGrid system fully implemented with all advanced features
- **v0.2.0** (2024-12): Migration to PlotGrid for all plotting functions
- **v0.1.0** (2024-11): Initial PlotGrid implementation

---

## License

Part of the neural-analysis package. See main LICENSE file.

---

## References

Design inspired by:
- Matplotlib's Object-Oriented API
- Plotly's Figure Factory
- Seaborn's FacetGrid
- pandas' plotting interface

---

## Summary

The PlotGrid system provides a **powerful, unified plotting framework** that:
- ✅ Supports all plot types in one consistent API
- ✅ Enables complex multi-panel visualizations with ease
- ✅ Works seamlessly with both matplotlib and plotly
- ✅ Reduces code by ~75% compared to manual subplot management
- ✅ Maintains backward compatibility with legacy functions
- ✅ Uses no duplicate logic across codebase
- ✅ Extensible architecture for future enhancements

**The PlotGrid system is now the recommended way to create all plots in the neural-analysis package.** 🎉
