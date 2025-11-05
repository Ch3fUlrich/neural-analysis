# PlotGrid System Documentation

## Overview

The PlotGrid system is a flexible, metadata-driven plotting framework for creating complex multi-panel visualizations with minimal code. It provides a single, unified API that works for everything from simple single plots to complex multi-panel layouts with multiple traces per subplot.

## Key Features

### 1. **Metadata-Driven Configuration**
- Specify plots using structured DataFrames or PlotSpec objects
- No need to manually manage subplot positions and styling
- Automatic color schemes and grid sizing

### 2. **Multiple Traces Per Subplot**
- Use `subplot_position` parameter to overlay multiple data series
- Perfect for comparing different conditions, methods, or groups
- Automatic legend generation

### 3. **Flexible Plot Types**
- Scatter (2D and 3D)
- Line plots
- Histograms
- Heatmaps
- KDE plots
- Bar charts

### 4. **Dual Backend Support**
- Matplotlib: For publication-quality static figures
- Plotly: For interactive, web-based visualizations
- Same API works for both backends

### 5. **Automatic Layout**
- Grid size auto-calculated from number of plots
- Manual override available for custom layouts
- Flexible spacing and axis sharing

## Core Classes

### PlotSpec
Specification for a single plot element.

```python
@dataclass
class PlotSpec:
    data: np.ndarray | pd.DataFrame  # Data to plot
    plot_type: str                    # 'scatter', 'line', 'histogram', etc.
    subplot_position: int | None      # Group multiple traces in same subplot
    title: str | None                 # Subplot title
    label: str | None                 # Legend label
    color: str | None                 # Color (name, hex, rgb)
    marker_size: float | None         # Marker size
    line_width: float | None          # Line width
    alpha: float = 0.7                # Transparency
    kwargs: dict                      # Additional plot-specific args
```

### GridLayoutConfig
Configuration for grid layout.

```python
@dataclass
class GridLayoutConfig:
    rows: int | None                  # Number of rows (auto if None)
    cols: int | None                  # Number of columns (auto if None)
    subplot_titles: list[str] | None  # Titles for each subplot
    shared_xaxes: bool | str          # Share x-axes
    shared_yaxes: bool | str          # Share y-axes
    vertical_spacing: float | None    # Vertical spacing (0-1)
    horizontal_spacing: float | None  # Horizontal spacing (0-1)
    group_by: str | None              # DataFrame column for grouping
```

### ColorScheme
Color scheme for grouped plots.

```python
@dataclass
class ColorScheme:
    palette: str | list[str]          # Color palette name or list
    group_colors: dict | None         # Explicit color mapping
    alpha: float = 0.7                # Default transparency
```

### PlotGrid
Main class for creating grid-based plots.

```python
class PlotGrid:
    def __init__(
        plot_specs: list[PlotSpec],
        config: PlotConfig,
        layout: GridLayoutConfig,
        color_scheme: ColorScheme,
        backend: str
    )
    
    @classmethod
    def from_dataframe(df, ...)      # Create from DataFrame
    
    @classmethod
    def from_dict(data_dict, ...)    # Create from dict
    
    def add_plot(data, plot_type, ...) # Add plot dynamically
    
    def plot() -> Figure              # Generate the figure
```

## Usage Examples

### Example 1: Simple Comparison Grid

```python
from neural_analysis.plotting import plot_comparison_grid, PlotConfig

# Quick comparison of multiple datasets
data_dict = {
    'Method A': result_a,
    'Method B': result_b,
    'Method C': result_c,
}

fig = plot_comparison_grid(
    data_dict,
    plot_type='histogram',
    rows=1,
    cols=3,
    config=PlotConfig(title="Method Comparison"),
    backend='plotly'
)
fig.show()
```

### Example 2: Multi-Trace Subplot

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

### Example 3: DataFrame-Based Configuration

```python
import pandas as pd
from neural_analysis.plotting import PlotGrid

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

### Example 4: Mixed Plot Types

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

### Example 5: Complex Multi-Panel with Groups

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

### Example 6: 3D Scatter Grid

```python
# Multiple 3D scatter plots in a grid
methods = ['IQR', 'Z-Score', 'Isolation Forest', 'LOF']
results_3d = {...}  # 3D data for each method

specs = [
    PlotSpec(
        data=results_3d[method],
        plot_type='scatter3d',
        title=method,
        color='blue',
        marker_size=4,
        alpha=0.6
    )
    for method in methods
]

grid = PlotGrid(
    plot_specs=specs,
    layout=GridLayoutConfig(rows=2, cols=2)
)
fig = grid.plot()
```

## Advanced Features

### Dynamic Plot Addition

```python
grid = PlotGrid(config=PlotConfig(title="Dynamic Grid"))

# Add plots dynamically
grid.add_plot(data1, plot_type='scatter', title='Plot 1')
grid.add_plot(data2, plot_type='line', title='Plot 2')
grid.add_plot(data3, plot_type='histogram', title='Plot 3')

fig = grid.plot()
```

### Custom Color Schemes

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

### Shared Axes

```python
# Share x-axes across rows
layout = GridLayoutConfig(
    rows=3,
    cols=2,
    shared_xaxes='rows',  # Share within rows
    shared_yaxes=False
)

grid = PlotGrid(plot_specs=specs, layout=layout)
```

### Custom Spacing

```python
layout = GridLayoutConfig(
    rows=2,
    cols=2,
    vertical_spacing=0.15,     # 15% vertical gap
    horizontal_spacing=0.1,    # 10% horizontal gap
)
```

## Convenience Functions

### plot_comparison_grid()
Quick multi-panel comparison of datasets.

```python
fig = plot_comparison_grid(
    data_dict={'A': data_a, 'B': data_b},
    plot_type='scatter',
    rows=1,
    cols=2
)
```

### plot_grouped_comparison()
Overlaid plots grouped by category.

```python
fig = plot_grouped_comparison(
    data=df,
    x_col='time',
    y_col='activity',
    group_col='condition',  # Different colors per condition
    plot_type='line'
)
```

## Best Practices

### 1. **Use subplot_position for Comparisons**
When comparing multiple conditions/methods in the same subplot:
```python
# Good: Use subplot_position
specs = [
    PlotSpec(data=control, subplot_position=0, label='Control'),
    PlotSpec(data=treatment, subplot_position=0, label='Treatment'),
]
```

### 2. **Use DataFrames for Complex Metadata**
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

### 3. **Let Grid Auto-Size**
Unless you have specific requirements:
```python
# Good: Let it auto-size
layout = GridLayoutConfig()  # Auto-determines rows/cols

# Manual override only when needed
layout = GridLayoutConfig(rows=2, cols=3)
```

### 4. **Use Convenience Functions for Simple Cases**
```python
# Simple comparison: use convenience function
fig = plot_comparison_grid(data_dict, plot_type='histogram')

# Complex multi-trace: use PlotGrid directly
grid = PlotGrid(plot_specs=complex_specs)
```

## Common Patterns

### Pattern 1: Method Comparison
Compare multiple methods on the same data:
```python
methods = ['Method A', 'Method B', 'Method C']
results = {method: compute(method) for method in methods}

fig = plot_comparison_grid(
    results,
    plot_type='scatter',
    cols=3,
    config=PlotConfig(title="Method Comparison")
)
```

### Pattern 2: Before/After
Show data before and after processing:
```python
specs = [
    PlotSpec(data=before, subplot_position=0, label='Before'),
    PlotSpec(data=after, subplot_position=0, label='After'),
]
grid = PlotGrid(plot_specs=specs)
```

### Pattern 3: Time Series Grid
Multiple time series in a grid:
```python
specs = [
    PlotSpec(
        data=np.column_stack([times, signal]),
        plot_type='line',
        title=f'Neuron {i}',
        color='blue'
    )
    for i, signal in enumerate(signals)
]
grid = PlotGrid(plot_specs=specs, layout=GridLayoutConfig(cols=4))
```

### Pattern 4: Multi-Modal Comparison
Different modalities in same subplot:
```python
specs = [
    PlotSpec(data=eeg_data, plot_type='line', subplot_position=0, label='EEG'),
    PlotSpec(data=ecg_data, plot_type='line', subplot_position=0, label='ECG'),
    PlotSpec(data=emg_data, plot_type='line', subplot_position=0, label='EMG'),
]
```

## Migration Guide

### Old Way (Manual Subplot Management)
```python
fig = create_subplot_grid(rows=2, cols=2, backend='plotly')

for i, data in enumerate(datasets):
    row = (i // 2) + 1
    col = (i % 2) + 1
    trace = go.Scatter(x=data[:, 0], y=data[:, 1])
    add_trace_to_subplot(fig, trace, row=row, col=col)
```

### New Way (PlotGrid)
```python
specs = [
    PlotSpec(data=data, plot_type='scatter', title=f'Dataset {i}')
    for i, data in enumerate(datasets)
]

grid = PlotGrid(plot_specs=specs, layout=GridLayoutConfig(rows=2, cols=2))
fig = grid.plot()
```

## Performance Considerations

- **Large Grids**: For >20 subplots, consider using matplotlib backend (faster rendering)
- **Interactive Plots**: Use plotly for interactive exploration (<10 subplots recommended)
- **Memory**: Each PlotSpec holds data reference, not copy (efficient)
- **Batch Creation**: Use DataFrame approach for creating many similar plots

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
# Supported: scatter, line, histogram, heatmap, scatter3d, bar
# Data format:
# - scatter: (n, 2) array
# - scatter3d: (n, 3) array
# - line: (n,) or (n, 2) array
# - histogram: (n,) array
# - heatmap: (rows, cols) array
```

## API Reference

See the docstrings in `plotting/grid_config.py` for complete API documentation.

## Examples

Complete working examples are available in:
- `examples/metrics_examples.ipynb` - Demonstrates PlotGrid throughout
- Section 7 of the notebook - Comprehensive usage guide

## Future Enhancements

Planned features:
- [ ] Support for animation across subplots
- [ ] Automatic subplot positioning based on similarity
- [ ] Export to multi-page PDF
- [ ] Interactive subplot linking
- [ ] Custom plot type registration

## Contributing

To add new plot types:
1. Add type to `PlotSpec.plot_type` Literal
2. Implement in `_plot_spec_matplotlib()` 
3. Implement in `_plot_spec_plotly()`
4. Add tests and documentation

## License

Part of the neural-analysis package. See main LICENSE file.
