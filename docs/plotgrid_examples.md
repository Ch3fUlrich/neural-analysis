# PlotGrid System - Advanced Examples

The PlotGrid system is now the unified plotting interface that supports all plotting capabilities previously scattered across multiple modules.

## Overview

PlotGrid provides a consistent, metadata-driven approach to creating any type of plot. All features from `plots_2d.py` and `plots_3d.py` are now available through PlotGrid's enhanced PlotSpec system.

## Supported Plot Types

### Basic Plot Types
- `scatter` - 2D scatter plots with color mapping
- `scatter3d` - 3D scatter plots
- `line` - Line plots
- `histogram` - Histograms
- `heatmap` - 2D heatmaps
- `bar` - Bar charts
- `violin` - Violin plots
- `box` - Box plots

### Advanced 2D Plot Types  
- `trajectory` - 2D trajectories with time-gradient coloring
- `kde` - 2D KDE density plots with contours
- `grouped_scatter` - Multiple groups with optional convex hulls
- `convex_hull` - Convex hull boundaries

### 3D Plot Types
- `trajectory3d` - 3D trajectories with time-gradient coloring

## Examples

### 1. Trajectory Plot with Time Coloring

```python
import numpy as np
from src.neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

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

### 2. KDE Density Plot

```python
import numpy as np
from src.neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

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

### 3. Grouped Scatter with Convex Hulls

```python
import numpy as np
from src.neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Generate grouped data
group_a_x = np.random.randn(50)
group_a_y = np.random.randn(50)
group_b_x = np.random.randn(50) + 3
group_b_y = np.random.randn(50) + 3

# Create grouped scatter spec
spec = PlotSpec(
    data={
        'Group A': (group_a_x, group_a_y),
        'Group B': (group_b_x, group_b_y)
    },
    plot_type='grouped_scatter',
    show_hulls=True,          # Show convex hulls around groups
    marker_size=30,
    alpha=0.6
)

# Create plot
config = PlotConfig(title='Grouped Scatter with Convex Hulls')
grid = PlotGrid(plot_specs=[spec], config=config)
fig = grid.plot()
```

### 4. 3D Trajectory with Time Coloring

```python
import numpy as np
from src.neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Generate 3D trajectory
t = np.linspace(0, 4*np.pi, 200)
x = np.sin(t)
y = np.cos(t)
z = t / 4

# Create 3D trajectory spec
spec = PlotSpec(
    data={'x': x, 'y': y, 'z': z},
    plot_type='trajectory3d',
    color_by="time",
    show_points=True,
    cmap='plasma',
    line_width=2.5,
    marker_size=8,
    alpha=0.8,
    colorbar=True,
    colorbar_label='Time'
)

# Create plot
config = PlotConfig(title='3D Trajectory with Time Gradient')
grid = PlotGrid(plot_specs=[spec], config=config)
fig = grid.plot()
```

### 5. Multi-Panel Plot Combining Multiple Types

```python
import numpy as np
from src.neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig, GridLayoutConfig

# Generate data
t = np.linspace(0, 2*np.pi, 100)
x1, y1 = np.sin(t), np.cos(t)
x2 = np.random.randn(200)
y2 = np.random.randn(200)

# Create specs for different subplots
specs = [
    # Subplot 0: Trajectory
    PlotSpec(
        data={'x': x1, 'y': y1},
        plot_type='trajectory',
        color_by="time",
        subplot_position=0,
        title='Trajectory',
        equal_aspect=True
    ),
    # Subplot 1: KDE
    PlotSpec(
        data={'x': x2, 'y': y2},
        plot_type='kde',
        fill=True,
        n_levels=10,
        subplot_position=1,
        title='KDE Density'
    ),
    # Subplot 2: Scatter
    PlotSpec(
        data={'x': x2, 'y': y2},
        plot_type='scatter',
        marker_size=10,
        alpha=0.5,
        subplot_position=2,
        title='Scatter Plot'
    ),
]

# Create multi-panel plot
layout = GridLayoutConfig(rows=1, cols=3)
config = PlotConfig(title='Multi-Panel Visualization')
grid = PlotGrid(plot_specs=specs, config=config, layout=layout)
fig = grid.plot()
```

## Migration from Old Functions

The convenience functions in `plots_2d.py` and `plots_3d.py` now keep their simple interfaces but can be replicated with PlotGrid:

### Old Way:
```python
from src.neural_analysis.plotting import plot_trajectory_2d
fig = plot_trajectory_2d(x, y, color_by="time")
```

### PlotGrid Way:
```python
from src.neural_analysis.plotting import PlotGrid, PlotSpec
spec = PlotSpec(data={'x': x, 'y': y}, plot_type='trajectory', color_by="time")
grid = PlotGrid(plot_specs=[spec])
fig = grid.plot()
```

## Advantages of PlotGrid

1. **Unified Interface**: Single system for all plot types
2. **Multi-Panel Support**: Easy grid layouts with `subplot_position`
3. **Backend Agnostic**: Works with both matplotlib and plotly
4. **Metadata-Driven**: Declarative configuration instead of imperative code
5. **Extensible**: Easy to add new plot types
6. **Consistent**: Same API across all plotting needs

## Advanced Features Available

- **Time-Gradient Coloring**: Automatic time-based color gradients for trajectories
- **Convex Hull Rendering**: Automatic boundary computation for grouped data
- **KDE Contours**: Gaussian kernel density estimation with customizable contours
- **Flexible Color Mapping**: Support for continuous and categorical colormaps
- **Equal Aspect Ratios**: Proper geometric proportions for trajectories
- **Colorbar Control**: Fine-grained control over colorbars
- **Bandwidth Control**: Customizable KDE bandwidth parameters

The PlotGrid system is now the recommended way to create all plots in the neural-analysis package.
