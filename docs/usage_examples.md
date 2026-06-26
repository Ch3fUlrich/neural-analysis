## Usage

### Basic Plotting Examples

```python
import numpy as np
from neural_analysis.plotting import plot_line, plot_scatter_2d, plot_heatmap

# Line plot with error bars
x = np.linspace(0, 10, 100)
y = np.sin(x)
error_y = 0.1 * np.ones_like(y)
plot_line(x, y, error_y=error_y, backend="matplotlib")

# 2D scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
plot_scatter_2d(x, y, backend="plotly")

# Heatmap with labels
data = np.random.rand(5, 5)
plot_heatmap(
    data,
    x_labels=["A", "B", "C", "D", "E"],
    y_labels=["1", "2", "3", "4", "5"],
    show_values=True,
    colorbar=True,
    backend="matplotlib"
)
```

### Backend Selection

Choose between matplotlib and plotly backends:

```python
# Static matplotlib plots (publication-ready)
plot_scatter_2d(x, y, backend="matplotlib")

# Interactive plotly plots (exploration)
plot_scatter_2d(x, y, backend="plotly")
```

### Advanced Features

```python
# Trajectory with color gradient
from neural_analysis.plotting import plot_trajectory_2d

trajectory = np.random.randn(100, 2)
plot_trajectory_2d(trajectory, color_by="time", backend="plotly")

# Grouped scatter with convex hulls
from neural_analysis.plotting import plot_grouped_scatter_2d

points = np.random.randn(100, 2)
labels = np.random.choice(["A", "B", "C"], 100)
plot_grouped_scatter_2d(points, labels, show_hulls=True, backend="matplotlib")

# Reference lines and annotations (PlotGrid)
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Create line plot with threshold lines and annotations
x = np.linspace(0, 10, 100)
y = np.exp(-0.5 * x) + np.random.normal(0, 0.1, 100)

spec = PlotSpec(
    data={'x': x, 'y': y},
    plot_type='line',
    color='steelblue',
    line_width=2,
    # Horizontal threshold line
    hlines=[{'y': 0.5, 'color': 'red', 'linestyle': '--', 'label': 'Threshold'}],
    # Vertical marker line
    vlines=[{'x': 5.0, 'color': 'orange', 'linestyle': ':', 'label': 'Key point'}],
    # Text annotation with arrow
    annotations=[{
        'text': 'Important event',
        'xy': (5.0, 0.5),
        'xytext': (6.0, 0.8),
        'arrowprops': {'color': 'darkred'}
    }]
)

grid = PlotGrid(plot_specs=[spec], config=PlotConfig())
grid.plot()  # Works with both matplotlib and plotly backends!
```