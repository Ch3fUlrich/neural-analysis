import marimo

__generated_with = "0.18.3"

app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():
    import marimo as mo

    return mo


@app.cell
def _(mo):
    mo.md(r"""
    # 2D Plotting Functions Examples

    This notebook demonstrates the 2D plotting functions available in the `neural_analysis.plotting` module:

    - `plot_scatter_2d`: Basic 2D scatter plots with color mapping and variable sizes
    - `plot_trajectory_2d`: 2D trajectories with time-based color gradients
    - `plot_grouped_scatter_2d`: Grouped scatter plots with optional convex hulls
    - `plot_kde_2d`: 2D kernel density estimation plots

    Each function supports both matplotlib and plotly backends for flexibility.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from neural_analysis.plotting import (
        PlotConfig,
        plot_grouped_scatter_2d,
        plot_kde_2d,
        plot_scatter_2d,
        plot_trajectory_2d,
        set_backend,
    )

    # Set random seed for reproducibility
    np.random.seed(42)
    return (
        PlotConfig,
        np,
        plot_grouped_scatter_2d,
        plot_kde_2d,
        plot_scatter_2d,
        plot_trajectory_2d,
        plt,
        set_backend,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Basic 2D Scatter Plot

    Simple scatter plot showing 2D points without any color mapping.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_2d, plt, set_backend):
    # Generate random 2D points (simulating neural positions)
    n_points = 100
    _x = np.random.randn(n_points) * 2
    _y = np.random.randn(n_points) * 2
    set_backend("matplotlib")
    # Create scatter plot
    _config = PlotConfig(
        title="Basic 2D Scatter Plot", xlabel="X Position", ylabel="Y Position"
    )
    _fig = plot_scatter_2d(_x, _y, colors="steelblue", config=_config)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Scatter Plot with Color Mapping

    Scatter plot where points are colored by a continuous value (e.g., firing rate, activation level).
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_2d, plt):
    # Generate points with associated values (e.g., firing rates)
    _x = np.random.randn(100) * 3
    _y = np.random.randn(100) * 3
    firing_rates = np.sqrt(_x**2 + _y**2) + np.random.randn(100) * 0.5
    _config = PlotConfig(
        title="Neural Activity by Position",
        xlabel="X Position (mm)",
        ylabel="Y Position (mm)",
    )
    # Create scatter plot with color mapping
    _fig = plot_scatter_2d(
        _x,
        _y,
        colors=firing_rates,
        cmap="viridis",
        colorbar=True,
        colorbar_label="Firing Rate (Hz)",
        config=_config,
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Scatter Plot with Variable Marker Sizes

    Points sized according to a variable (e.g., cell body size, spike amplitude).
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_2d, plt):
    # Generate points with variable sizes
    _x = np.random.randn(80) * 2
    _y = np.random.randn(80) * 2
    sizes = np.random.uniform(20, 200, 80)  # Variable marker sizes
    _config = PlotConfig(
        title="Variable Marker Sizes", xlabel="X Position", ylabel="Y Position"
    )
    # Create scatter plot with variable sizes
    _fig = plot_scatter_2d(
        _x, _y, colors="coral", sizes=sizes, alpha=0.6, config=_config
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 2D Trajectory with Time-Based Coloring

    Visualize a trajectory path with colors indicating progression over time.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_trajectory_2d, plt):
    # Generate a spiral trajectory (simulating neural state evolution)
    _t = np.linspace(0, 4 * np.pi, 150)
    _x_traj = _t * np.cos(_t) / 4
    _y_traj = _t * np.sin(_t) / 4
    _config = PlotConfig(
        title="Neural State Trajectory Over Time", xlabel="PC1", ylabel="PC2"
    )
    # Plot trajectory with time-based colors
    _fig = plot_trajectory_2d(
        _x_traj,
        _y_traj,
        color_by="time",
        cmap="plasma",
        linewidth=2,
        show_points=True,
        config=_config,
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Trajectory with Custom Styling

    Trajectory without points, using custom line width and color.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_trajectory_2d, plt):
    # Generate a figure-8 trajectory
    _t = np.linspace(0, 2 * np.pi, 200)
    _x_traj = np.sin(_t) * 3
    _y_traj = np.sin(2 * _t) * 2
    _config = PlotConfig(
        title="Movement Trajectory (Figure-8 Pattern)",
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
    )
    # Plot trajectory without time-based colors
    _fig = plot_trajectory_2d(
        _x_traj, _y_traj, color_by=None, linewidth=3, show_points=False, config=_config
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Grouped Scatter Plot with Convex Hulls

    Visualize multiple groups with different colors and optional convex hull boundaries.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_grouped_scatter_2d, plt):
    # Generate three clusters (simulating different cell types)
    group1_x = np.random.randn(50) * 0.8 - 3
    group1_y = np.random.randn(50) * 0.8 + 2
    group2_x = np.random.randn(60) * 1.0 + 2
    group2_y = np.random.randn(60) * 1.0 + 1
    group3_x = np.random.randn(40) * 0.7 + 0
    group3_y = np.random.randn(40) * 0.7 - 3
    grouped_data = {
        "Interneurons": (group1_x, group1_y),
        "Pyramidal": (group2_x, group2_y),
        "Glial": (group3_x, group3_y),
    }
    _config = PlotConfig(
        title="Cell Types in 2D Embedding Space",
        xlabel="Component 1",
        ylabel="Component 2",
    )
    _fig = plot_grouped_scatter_2d(grouped_data, show_hulls=True, config=_config)
    # Create grouped data dictionary (dict of (x, y) tuples)
    # Plot with convex hulls
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. 2D Kernel Density Estimation

    Show the density distribution of 2D data using KDE with filled contours.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_kde_2d, plt):
    # Generate bimodal distribution (two peaks)
    n = 150
    x_kde = np.concatenate([np.random.randn(n) * 1.0 - 2, np.random.randn(n) * 0.8 + 3])
    y_kde = np.concatenate([np.random.randn(n) * 0.8 + 1, np.random.randn(n) * 1.2 - 2])
    _config = PlotConfig(
        title="Spatial Density of Spike Locations",
        xlabel="X Position (mm)",
        ylabel="Y Position (mm)",
    )
    _fig = plot_kde_2d(
        x_kde, y_kde, cmap="RdYlBu_r", fill=True, show_points=False, config=_config
    )
    # Plot KDE with filled contours
    plt.show()
    return x_kde, y_kde


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. KDE with Overlaid Scatter Points

    Combine KDE density contours with the actual data points.
    """)
    return


@app.cell
def _(PlotConfig, plot_kde_2d, plt, x_kde, y_kde):
    # Use same bimodal data
    _config = PlotConfig(
        title="KDE Contours with Data Points", xlabel="Feature 1", ylabel="Feature 2"
    )
    _fig = plot_kde_2d(
        x_kde,
        y_kde,
        cmap="viridis",
        fill=False,
        show_points=True,
        n_levels=8,
        config=_config,
    )
    plt.show()  # Use contour lines only
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Comparing Matplotlib and Plotly Backends

    The same plot rendered with both backends for comparison. Plotly provides interactive features like zooming and hovering.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_2d, plt, set_backend):
    # Generate sample data
    _x = np.random.randn(100) * 2
    _y = np.random.randn(100) * 2
    colors_data = np.sqrt(_x**2 + _y**2)
    set_backend("matplotlib")
    # Matplotlib version
    config_mpl = PlotConfig(title="Matplotlib Backend", xlabel="X", ylabel="Y")
    fig_mpl = plot_scatter_2d(
        _x,
        _y,
        colors=colors_data,
        colorbar=True,
        colorbar_label="Distance",
        config=config_mpl,
    )
    plt.show()
    set_backend("plotly")
    config_plotly = PlotConfig(
        title="Plotly Backend (Interactive)", xlabel="X", ylabel="Y"
    )
    # Plotly version (interactive)
    fig_plotly = plot_scatter_2d(
        _x,
        _y,
        colors=colors_data,
        colorbar=True,
        colorbar_label="Distance",
        config=config_plotly,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    The `neural_analysis.plotting` module provides flexible 2D plotting capabilities:

    - **`plot_scatter_2d`**: Basic scatter plots with color mapping and variable sizes
    - **`plot_trajectory_2d`**: Trajectory visualization with time-based coloring
    - **`plot_grouped_scatter_2d`**: Multi-group visualization with convex hulls
    - **`plot_kde_2d`**: Kernel density estimation for spatial distributions

    All functions support both matplotlib (static) and plotly (interactive) backends via `set_backend()` or the `backend` parameter.
    """)
    return
