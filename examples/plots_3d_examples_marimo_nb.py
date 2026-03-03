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
    # 3D Plotting Examples

    This notebook demonstrates the 3D plotting capabilities of the neural_analysis package.

    ## Features Covered:
    - 3D scatter plots with color mapping
    - 3D trajectory visualization
    - Time-based coloring
    - Both matplotlib and plotly backends
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from neural_analysis.plotting import (
        PlotConfig,
        plot_scatter_3d,
        plot_trajectory_3d,
    )

    # Set random seed for reproducibility
    np.random.seed(42)
    return PlotConfig, np, plot_scatter_3d, plot_trajectory_3d, plt


@app.cell
def _():
    pass
    return ()


@app.cell
def _(PlotConfig):
    # Print PlotConfig fields
    print("PlotConfig fields:", [f for f in PlotConfig.__dataclass_fields__.keys()])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Basic 3D Scatter Plot

    Create a simple 3D scatter plot with random data.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_3d, plt):
    # Generate random 3D data
    _n_points = 100
    x = np.random.randn(_n_points)
    y = np.random.randn(_n_points)
    z = np.random.randn(_n_points)
    _config = PlotConfig(
        title="Basic 3D Scatter Plot",
        xlabel="X axis",
        ylabel="Y axis",
        zlabel="Z axis",
        figsize=(10, 8),
    )
    # Create scatter plot
    plot_scatter_3d(x, y, z, config=_config, backend="matplotlib")
    plt.show()
    return x, y, z


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 3D Scatter with Color Mapping

    Use a fourth dimension to color the points.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_3d, plt, x, y, z):
    # Generate data with a color dimension
    colors = np.sqrt(x**2 + y**2 + z**2)  # Distance from origin
    _config = PlotConfig(
        title="3D Scatter with Distance-based Coloring",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        figsize=(10, 8),
    )
    plot_scatter_3d(
        x,
        y,
        z,
        colors=colors,
        cmap="viridis",
        sizes=50,
        alpha=0.7,
        config=_config,
        backend="matplotlib",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 3D Trajectory - Helix

    Visualize a 3D path (helix) through space.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_trajectory_3d, plt):
    # Generate helix trajectory
    _t = np.linspace(0, 4 * np.pi, 200)
    x_helix = np.sin(_t)
    y_helix = np.cos(_t)
    z_helix = _t / 4
    _config = PlotConfig(
        title="3D Helix Trajectory",
        xlabel="X",
        ylabel="Y",
        zlabel="Z (time)",
        figsize=(10, 8),
    )
    plot_trajectory_3d(
        x_helix,
        y_helix,
        z_helix,
        color_by=None,
        linewidth=2,
        config=_config,
        backend="matplotlib",
    )
    plt.show()  # No time-based coloring
    return x_helix, y_helix, z_helix


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 3D Trajectory with Time-based Coloring

    Color the trajectory based on time progression.
    """)
    return


@app.cell
def _(PlotConfig, plot_trajectory_3d, plt, x_helix, y_helix, z_helix):
    _config = PlotConfig(
        title="3D Trajectory with Time Coloring",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        figsize=(10, 8),
    )
    plot_trajectory_3d(
        x_helix,
        y_helix,
        z_helix,
        color_by="time",
        cmap="plasma",
        linewidth=3,
        show_points=True,
        config=_config,
        backend="matplotlib",
    )
    plt.show()  # Color by time progression
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Complex 3D Trajectory - Lissajous Curve

    Visualize a complex 3D parametric curve.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_trajectory_3d, plt):
    # Generate Lissajous curve
    _t = np.linspace(0, 2 * np.pi, 300)
    x_liss = np.sin(3 * _t)
    y_liss = np.cos(4 * _t)
    z_liss = np.sin(5 * _t)
    _config = PlotConfig(
        title="3D Lissajous Curve (3:4:5)",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        figsize=(10, 8),
    )
    plot_trajectory_3d(
        x_liss,
        y_liss,
        z_liss,
        color_by="time",
        cmap="coolwarm",
        linewidth=2,
        show_points=False,
        config=_config,
        backend="matplotlib",
    )
    plt.show()  # Fixed: was color_by_time
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Neural State Space Example

    Simulate neural population activity in 3D state space.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_trajectory_3d, plt):
    # Simulate neural trajectory with noise
    n_timesteps = 150
    _t = np.linspace(0, 3 * np.pi, n_timesteps)
    x_neural = np.sin(_t) * (1 + 0.3 * _t / np.max(_t))
    # Base trajectory (circular motion with drift)
    y_neural = np.cos(_t) * (1 + 0.3 * _t / np.max(_t))
    z_neural = 0.5 * _t / np.pi
    x_neural += np.random.randn(n_timesteps) * 0.1
    y_neural += np.random.randn(n_timesteps) * 0.1
    # Add noise
    z_neural += np.random.randn(n_timesteps) * 0.1
    _config = PlotConfig(
        title="Neural Population State Space Trajectory",
        xlabel="PC1",
        ylabel="PC2",
        zlabel="PC3",
        figsize=(12, 9),
    )
    plot_trajectory_3d(
        x_neural,
        y_neural,
        z_neural,
        color_by="time",
        cmap="viridis",
        linewidth=2.5,
        show_points=True,
        point_size=20,
        alpha=0.6,
        config=_config,
        backend="matplotlib",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Multiple Conditions Comparison

    Compare different experimental conditions in 3D.
    """)
    return


@app.cell
def _(PlotConfig, np, plt):
    # Generate data for two conditions
    n_trials = 50
    x1 = np.random.randn(n_trials) * 0.5
    # Condition 1: Clustered around origin
    y1 = np.random.randn(n_trials) * 0.5
    z1 = np.random.randn(n_trials) * 0.5
    x2 = np.random.randn(n_trials) * 0.5 + 2
    y2 = np.random.randn(n_trials) * 0.5 + 1
    # Condition 2: Shifted cluster
    z2 = np.random.randn(n_trials) * 0.5 + 1.5
    from neural_analysis.plotting import PlotGrid, PlotSpec

    _config = PlotConfig(
        title="Neural Activity: Two Experimental Conditions (Matplotlib)",
        xlabel="PC1",
        ylabel="PC2",
        zlabel="PC3",
        figsize=(12, 9),
    )
    specs = [
        PlotSpec(
            data=np.column_stack([x1, y1, z1]),
            plot_type="scatter3d",
            subplot_position=0,
            color="blue",
            marker_size=50,
            alpha=0.6,
            label="Condition 1",
        ),
        PlotSpec(
            data=np.column_stack([x2, y2, z2]),
            plot_type="scatter3d",
            subplot_position=0,
            color="red",
            marker_size=50,
            alpha=0.6,
            label="Condition 2",
        ),
    ]
    # Plot 1: Matplotlib backend - both conditions overlaid in same 3D plot
    grid = PlotGrid(plot_specs=specs, config=_config, backend="matplotlib")
    ax = grid.plot()
    plt.show()
    _config.figsize = (12, 9)
    specs[0].subplot_position = 0
    specs[1].subplot_position = 1
    grid = PlotGrid(plot_specs=specs, config=_config, backend="matplotlib")
    _fig, axes = grid.plot()
    plt.show()  # Same position = overlay  # Single subplot returns just axes
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Plotly Interactive 3D Scatter

    Use Plotly backend for interactive visualization.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_scatter_3d):
    # Generate clustered data
    _n_points = 200
    theta = np.random.uniform(0, 2 * np.pi, _n_points)
    phi = np.random.uniform(0, np.pi, _n_points)
    r = np.random.uniform(0.5, 2, _n_points)
    x_sphere = r * np.sin(phi) * np.cos(theta)
    y_sphere = r * np.sin(phi) * np.sin(theta)
    z_sphere = r * np.cos(phi)
    colors_sphere = r
    _config = PlotConfig(
        title="Interactive 3D Scatter (Plotly)", xlabel="X", ylabel="Y", zlabel="Z"
    )  # Color by radius
    _fig = plot_scatter_3d(
        x_sphere,
        y_sphere,
        z_sphere,
        colors=colors_sphere,
        cmap="turbo",
        sizes=5,
        alpha=0.7,
        config=_config,
        backend="plotly",
    )
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Plotly Interactive 3D Trajectory

    Interactive trajectory with time coloring.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_trajectory_3d):
    # Create a spiral trajectory
    _t = np.linspace(0, 6 * np.pi, 300)
    radius = np.linspace(0.1, 2, 300)
    x_spiral = radius * np.cos(_t)
    y_spiral = radius * np.sin(_t)
    z_spiral = _t / (2 * np.pi)
    _config = PlotConfig(
        title="Interactive 3D Spiral Trajectory (Plotly)",
        xlabel="X",
        ylabel="Y",
        zlabel="Z (time)",
    )
    _fig = plot_trajectory_3d(
        x_spiral,
        y_spiral,
        z_spiral,
        color_by="time",
        cmap="rainbow",
        linewidth=3,
        show_points=True,
        config=_config,
        backend="plotly",
    )
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    This notebook demonstrated:
    - ✅ Basic 3D scatter plots
    - ✅ Color mapping with colormaps
    - ✅ 3D trajectories with time-based coloring
    - ✅ Complex parametric curves (helix, Lissajous)
    - ✅ Neural state space visualization
    - ✅ Both matplotlib and plotly backends
    - ✅ Interactive visualization with Plotly

    All functions use the same consistent API and backend selection system!
    """)
    return
