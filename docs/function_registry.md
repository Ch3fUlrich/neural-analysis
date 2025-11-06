# Neural Analysis Plotting Function Registry

Auto-generated function registry for all plotting functions.
**Last Updated:** Auto-generated

## Purpose

This registry helps developers and AI agents:
- Avoid code duplication by finding existing functions
- Understand the plotting module structure
- Quickly locate the right function for a task

---

## Core System

### `plotting.backend`

#### `get_backend()`

**Returns:** `BackendType`

**Purpose:** Get the current visualization backend.

**Location:** `plotting.backend.py` (line 72)

#### `set_backend(backend)`

**Returns:** `None`

**Purpose:** Set the visualization backend.

**Location:** `plotting.backend.py` (line 34)

### `plotting.core`

#### `apply_layout_matplotlib(ax, config)`

**Returns:** `None`

**Purpose:** Apply common layout (title, labels, limits, grid) for matplotlib.

**Location:** `plotting.core.py` (line 251)

#### `apply_layout_plotly(fig, config)`

**Returns:** `None`

**Purpose:** Apply common layout (title, labels, limits, grid) for Plotly.

**Location:** `plotting.core.py` (line 273)

#### `apply_layout_plotly_3d(fig, config)`

**Returns:** `None`

**Purpose:** Apply common layout for Plotly 3D plots (scene configuration).

**Location:** `plotting.core.py` (line 300)

#### `calculate_alpha(value, min_value, max_value, ...)`

**Returns:** `float | list[float]`

**Purpose:** Calculate alpha value(s) based on value's position in range.

**Location:** `plotting.core.py` (line 403)

#### `create_rgba_labels(values, alpha, cmap)`

**Returns:** `npt.NDArray`

**Purpose:** Create RGBA labels using a colormap.

**Location:** `plotting.core.py` (line 552)

#### `finalize_plot_matplotlib(config)`

**Returns:** `None`

**Purpose:** Handle save and show for matplotlib plots.

**Location:** `plotting.core.py` (line 351)

#### `finalize_plot_plotly(fig, config)`

**Returns:** `None`

**Purpose:** Handle save and show for plotly plots.

**Location:** `plotting.core.py` (line 364)

#### `generate_similar_colors(base_color, num_colors, hue_variation, ...)`

**Returns:** `list[tuple[float, float, float]]`

**Purpose:** Generate a list of similar colors based on a base color.

**Location:** `plotting.core.py` (line 492)

#### `get_default_categorical_colors(n)`

**Returns:** `list[str]`

**Purpose:** Return a list of n default categorical colors as hex strings.

**Location:** `plotting.core.py` (line 333)

#### `get_save_path()`

**Returns:** `Path | None`

**Purpose:** Get the full save path for the plot.

**Location:** `plotting.core.py` (line 150)

#### `make_list_if_not(value)`

**Returns:** `list[any]`

**Purpose:** Convert value to list if it isn't already.

**Location:** `plotting.core.py` (line 673)

#### `resolve_colormap(cmap, backend)`

**Returns:** `Any`

**Purpose:** Return a backend-appropriate colormap identifier.

**Location:** `plotting.core.py` (line 217)

#### `save_plot(save_path, format, dpi, ...)`

**Returns:** `None`

**Purpose:** Save current matplotlib figure to file.

**Location:** `plotting.core.py` (line 625)

### `plotting.grid_config`

#### `__init__(plot_specs, config, layout, ...)`

**Returns:** `Any`

**Purpose:** No description

**Location:** `plotting.grid_config.py` (line 245)

#### `add_plot(data, plot_type)`

**Returns:** `None`

**Purpose:** Add a plot to the grid.

**Location:** `plotting.grid_config.py` (line 377)

#### `auto_size_grid(n_plots)`

**Returns:** `tuple[int, int]`

**Purpose:** Automatically determine grid size from number of plots.

**Location:** `plotting.grid_config.py` (line 111)

#### `from_dataframe(df, data_col, plot_type_col, ...)`

**Returns:** `PlotGrid`

**Purpose:** Create PlotGrid from a pandas DataFrame.

**Location:** `plotting.grid_config.py` (line 260)

#### `from_dict(data_dict, plot_type)`

**Returns:** `PlotGrid`

**Purpose:** Create PlotGrid from a dictionary of {label: data}.

**Location:** `plotting.grid_config.py` (line 343)

#### `get_colors(groups)`

**Returns:** `dict[str, str]`

**Purpose:** Get color mapping for a list of groups.

**Location:** `plotting.grid_config.py` (line 162)

#### `plot()`

**Returns:** `Any`

**Purpose:** Generate the plot grid.

**Location:** `plotting.grid_config.py` (line 399)

#### `plot_comparison_grid(data_dict, plot_type, rows, ...)`

**Returns:** `Any`

**Purpose:** Create a grid comparing multiple datasets with the same plot type.

**Location:** `plotting.grid_config.py` (line 701)

#### `plot_grouped_comparison(data, x_col, y_col, ...)`

**Returns:** `Any`

**Purpose:** Create overlaid plots grouped by a category.

**Location:** `plotting.grid_config.py` (line 748)

---

## Renderers

### `plotting.renderers`

#### `render_bar_plotly(data, x, color, ...)`

**Returns:** `'go.Bar'`

**Purpose:** Render a bar plot using plotly.

**Location:** `plotting.renderers.py` (line 527)

#### `render_box_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a box plot using matplotlib.

**Location:** `plotting.renderers.py` (line 761)

#### `render_box_plotly(data, color, alpha, ...)`

**Returns:** `'go.Box'`

**Purpose:** Render a box plot using plotly.

**Location:** `plotting.renderers.py` (line 818)

#### `render_heatmap_matplotlib(ax, data, cmap, ...)`

**Returns:** `Any`

**Purpose:** Render a heatmap using matplotlib.

**Location:** `plotting.renderers.py` (line 444)

#### `render_heatmap_plotly(data, cmap, colorscale)`

**Returns:** `'go.Heatmap'`

**Purpose:** Render a heatmap using plotly.

**Location:** `plotting.renderers.py` (line 484)

#### `render_histogram_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a histogram using matplotlib.

**Location:** `plotting.renderers.py` (line 350)

#### `render_histogram_plotly(data, color, alpha, ...)`

**Returns:** `'go.Histogram'`

**Purpose:** Render a histogram using plotly.

**Location:** `plotting.renderers.py` (line 392)

#### `render_line_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a line plot using matplotlib.

**Location:** `plotting.renderers.py` (line 212)

#### `render_line_plotly(data, color, line_width, ...)`

**Returns:** `'go.Scatter'`

**Purpose:** Render a line plot using plotly.

**Location:** `plotting.renderers.py` (line 267)

#### `render_scatter3d_plotly(data, color, marker_size, ...)`

**Returns:** `'go.Scatter3d'`

**Purpose:** Render a 3D scatter plot using plotly.

**Location:** `plotting.renderers.py` (line 154)

#### `render_scatter_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a 2D scatter plot using matplotlib.

**Location:** `plotting.renderers.py` (line 47)

#### `render_scatter_plotly(data, color, marker, ...)`

**Returns:** `'go.Scatter'`

**Purpose:** Render a 2D scatter plot using plotly.

**Location:** `plotting.renderers.py` (line 96)

#### `render_violin_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a violin plot with optional box plot and points using matplotlib.

**Location:** `plotting.renderers.py` (line 579)

#### `render_violin_plotly(data, color, alpha, ...)`

**Returns:** `'go.Violin'`

**Purpose:** Render a violin plot with optional box plot and points using plotly.

**Location:** `plotting.renderers.py` (line 683)

---

## Statistical Plots

### `plotting.statistical_plots`

#### `plot_bar(data, labels, colors, ...)`

**Returns:** `Any`

**Purpose:** Create a bar plot for comparing multiple groups.

**Location:** `plotting.statistical_plots.py` (line 33)

#### `plot_box(data, labels, colors, ...)`

**Returns:** `Any`

**Purpose:** Create box plots for comparing distributions.

**Location:** `plotting.statistical_plots.py` (line 222)

#### `plot_comparison_distributions(data, plot_type, rows, ...)`

**Returns:** `Any`

**Purpose:** Create separate distribution plots for each group in a grid.

**Location:** `plotting.statistical_plots.py` (line 383)

#### `plot_grouped_distributions(data, plot_type, colors, ...)`

**Returns:** `Any`

**Purpose:** Create multiple distribution plots grouped by category.

**Location:** `plotting.statistical_plots.py` (line 299)

#### `plot_violin(data, labels, colors, ...)`

**Returns:** `Any`

**Purpose:** Create violin plots for comparing distributions.

**Location:** `plotting.statistical_plots.py` (line 133)

---

## Dimension-Specific

### `plotting.plots_1d`

#### `plot_boolean_states(states, x, config, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Visualize boolean states over time.

**Location:** `plotting.plots_1d.py` (line 375)

#### `plot_line(data, x, config, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Plot a 1D line with optional error bands.

**Location:** `plotting.plots_1d.py` (line 47)

#### `plot_multiple_lines(data_dict, x, config, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Plot multiple lines on the same axes.

**Location:** `plotting.plots_1d.py` (line 284)

### `plotting.plots_2d`

#### `plot_grouped_scatter_2d(group_data, config, show_hulls, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Plot grouped 2D scatter data with optional convex hulls.

**Location:** `plotting.plots_2d.py` (line 339)

#### `plot_kde_2d(x, y, config, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Create a 2D KDE (kernel density estimation) plot.

**Location:** `plotting.plots_2d.py` (line 495)

#### `plot_scatter_2d(x, y, config, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Create a 2D scatter plot.

**Location:** `plotting.plots_2d.py` (line 36)

#### `plot_trajectory_2d(x, y, config, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Plot a 2D trajectory with line connecting points.

**Location:** `plotting.plots_2d.py` (line 187)

### `plotting.plots_3d`

#### `plot_scatter_3d(x, y, z, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Create a 3D scatter plot.

**Location:** `plotting.plots_3d.py` (line 42)

#### `plot_trajectory_3d(x, y, z, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Plot a 3D trajectory with line connecting points.

**Location:** `plotting.plots_3d.py` (line 217)

---

## Specialized

### `plotting.heatmaps`

#### `plot_heatmap(data, config, x_labels, ...)`

**Returns:** `plt.Axes | go.Figure`

**Purpose:** Create a heatmap visualization.

**Location:** `plotting.heatmaps.py` (line 37)

### `plotting.subplots`

#### `add_trace_to_subplot(fig, trace, row, ...)`

**Returns:** `go.Figure`

**Purpose:** Add a trace to a specific subplot in a plotly figure.

**Location:** `plotting.subplots.py` (line 201)

#### `create_subplot_grid(rows, cols, config, ...)`

**Returns:** `tuple[plt.Figure, list[plt.Axes]] | go.Figure`

**Purpose:** Create a multi-panel subplot grid.

**Location:** `plotting.subplots.py` (line 30)

#### `update_subplot_axes(fig, row, col, ...)`

**Returns:** `None`

**Purpose:** Update axes properties for a specific subplot.

**Location:** `plotting.subplots.py` (line 245)
