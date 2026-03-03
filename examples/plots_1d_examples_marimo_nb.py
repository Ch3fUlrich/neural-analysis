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
    # plots_1d Module Examples

    This notebook demonstrates all functions in the `neural_analysis.plotting.plots_1d` module:
    - `plot_line()` - Basic 1D line plots with optional error bands
    - `plot_multiple_lines()` - Multiple lines on same axes
    - `plot_boolean_states()` - Visualize boolean states over time

    All functions support both **matplotlib** (static) and **plotly** (interactive) backends.
    """)
    return


@app.cell
def _():
    # Import required libraries
    import matplotlib.pyplot as plt
    import numpy as np

    from neural_analysis.plotting import (
        PlotConfig,
        PlotGrid,
        PlotSpec,
        plot_boolean_states,
        plot_line,
        plot_multiple_lines,
        set_backend,
    )

    # Set random seed for reproducibility
    np.random.seed(42)
    return (
        PlotConfig,
        PlotGrid,
        PlotSpec,
        np,
        plot_boolean_states,
        plot_line,
        plot_multiple_lines,
        plt,
        set_backend,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Basic Line Plot with `plot_line()`

    The simplest usage - plot a 1D array with default settings.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_line, plt):
    # Generate sample data
    _data = np.random.randn(100).cumsum()
    _config = PlotConfig(title="Random Walk", xlabel="Time Step", ylabel="Value")
    # Create simple line plot
    plot_line(_data, config=_config, backend="matplotlib")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Line Plot with Error Bands

    Add standard deviation bands to show uncertainty.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_line, plt):
    # Generate data with uncertainty
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x) + np.random.randn(100) * 0.2
    std = np.ones(100) * 0.3  # Constant std
    _config = PlotConfig(
        title="Noisy Sine Wave with Error Bands",
        xlabel="X",
        ylabel="Y",
        figsize=(12, 5),
    )
    # Plot with error bands
    plot_line(_y, x=_x, std=std, config=_config, color="blue", label="Data ± σ")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Multiple Lines with `plot_multiple_lines()`

    Compare multiple signals on the same plot.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_multiple_lines, plt):
    # Generate multiple signals
    _x = np.linspace(0, 2 * np.pi, 100)
    data_dict = {
        "sin(x)": np.sin(_x),
        "cos(x)": np.cos(_x),
        "sin(2x)": np.sin(2 * _x),
        "cos(2x)": np.cos(2 * _x),
    }
    _config = PlotConfig(
        title="Multiple Trigonometric Functions",
        xlabel="X (radians)",
        ylabel="Amplitude",
        figsize=(12, 6),
        grid=True,
    )
    colors = ["red", "blue", "orange", "green"]
    plot_multiple_lines(
        data_dict, x=_x, config=_config, colors=colors, backend="matplotlib"
    )
    # Plot multiple lines
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4b. Line Plot with Reference Lines

    You can add horizontal and vertical reference lines to highlight thresholds, targets, or key values using the `hlines` and `vlines` parameters in PlotSpec. This is useful for showing thresholds, baselines, or important points in your data.
    """)
    return


@app.cell
def _(PlotConfig, PlotGrid, PlotSpec, np, plt):
    _x = np.linspace(0, 10, 100)
    _y = 5 * np.exp(-0.5 * _x) + np.random.normal(0, 0.2, 100)
    _threshold = 1.0
    _cross_idx = (
        np.where(_y < _threshold)[0][0] if any(_y < _threshold) else len(_y) - 1
    )
    _cross_x = _x[_cross_idx]
    _data = {"x": _x, "y": _y}
    _spec = PlotSpec(
        data=_data,
        plot_type="line",
        title="Signal Decay with Threshold",
        color="steelblue",
        line_width=2,
        label="Signal",
        hlines=[
            {
                "y": _threshold,
                "color": "red",
                "linestyle": "--",
                "linewidth": 2,
                "alpha": 0.8,
                "label": "Threshold (1.0)",
            }
        ],
        vlines=[
            {
                "x": _cross_x,
                "color": "orange",
                "linestyle": ":",
                "linewidth": 2.5,
                "alpha": 0.7,
                "label": f"Crossing point (t={_cross_x:.2f})",
            }
        ],
        annotations=[
            {
                "text": "Signal crosses\nthreshold here",
                "xy": (_cross_x, _threshold),
                "xytext": (_cross_x - 2, _threshold + 1),
                "fontsize": 9,
                "bbox": dict(
                    boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
                ),
                "arrowprops": dict(
                    arrowstyle="->", connectionstyle="arc3,rad=0.2", color="darkred"
                ),
            }
        ],
    )
    _config = PlotConfig(
        xlabel="Time (s)", ylabel="Signal Amplitude", figsize=(12, 6), grid=True
    )
    _grid = PlotGrid(plot_specs=[_spec], config=_config)
    _grid.plot()
    plt.show()
    print(f"\n✓ Signal crosses threshold at t={_cross_x:.2f}s")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4c. Reference Lines with Plotly Backend

    The same reference line functionality works with the interactive Plotly backend. Hover over the plot to see interactive features!
    """)
    return


@app.cell
def _(PlotConfig, PlotGrid, PlotSpec, np):
    # Same data as before
    _x = np.linspace(0, 10, 100)
    _y = 5 * np.exp(-0.5 * _x) + np.random.normal(0, 0.2, 100)
    _threshold = 1.0
    _cross_idx = (
        np.where(_y < _threshold)[0][0] if any(_y < _threshold) else len(_y) - 1
    )
    _cross_x = _x[_cross_idx]
    _data = {"x": _x, "y": _y}
    _spec = PlotSpec(
        data=_data,
        plot_type="line",
        title="Signal Decay with Threshold (Interactive)",
        color="steelblue",
        line_width=2,
        label="Signal",
        hlines=[
            {
                "y": _threshold,
                "color": "red",
                "linestyle": "--",
                "linewidth": 2,
                "alpha": 0.8,
                "label": "Threshold (1.0)",
            }
        ],
        vlines=[
            {
                "x": _cross_x,
                "color": "orange",
                "linestyle": ":",
                "linewidth": 2.5,
                "alpha": 0.7,
                "label": f"Crossing point (t={_cross_x:.2f})",
            }
        ],
        annotations=[
            {
                "text": "Signal crosses<br>threshold here",
                "xy": (_cross_x, _threshold),
                "xytext": (_cross_x - 2, _threshold + 1),
                "fontsize": 9,
                "bbox": dict(
                    boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
                ),
                "arrowprops": dict(
                    arrowstyle="->", connectionstyle="arc3,rad=0.2", color="darkred"
                ),
            }
        ],
    )
    # Create line plot with Plotly backend
    _config = PlotConfig(
        xlabel="Time (s)", ylabel="Signal Amplitude", figsize=(12, 6), grid=True
    )
    _grid = PlotGrid(plot_specs=[_spec], config=_config, backend="plotly")
    _fig = _grid.plot()
    _fig.show()
    print(
        f"\n✓ Signal crosses threshold at t={_cross_x:.2f}s (Plotly interactive plot)"
    )  # Add horizontal threshold line  # Add vertical line at crossing point  # Add annotation at the crossing point
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Boolean States with `plot_boolean_states()`

    Visualize behavioral or experimental states over time.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_boolean_states, plt):
    # Simulate behavioral states (e.g., animal moving vs stationary)
    time = np.arange(0, 100)
    is_moving = np.random.rand(100) > 0.6  # 40% of time moving
    _config = PlotConfig(
        title="Animal Movement States",
        xlabel="Time (seconds)",
        ylabel="State",
        figsize=(14, 4),
    )
    # Plot boolean states
    plot_boolean_states(
        is_moving,
        x=time,
        config=_config,
        true_color="green",
        false_color="lightgray",
        true_label="Moving",
        false_label="Stationary",
        backend="matplotlib",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Custom Styling with Markers and Linestyles
    """)
    return


@app.cell
def _(PlotConfig, np, plot_line, plt):
    # Sample data with custom styling
    _x = np.linspace(0, 10, 20)
    _y = 2 * _x + 1 + np.random.randn(20) * 2
    _config = PlotConfig(
        title="Custom Line Styling", xlabel="X", ylabel="Y", figsize=(10, 5), grid=True
    )
    plot_line(
        _y,
        x=_x,
        config=_config,
        color="purple",
        linewidth=2.5,
        linestyle="--",
        marker="o",
        markersize=8,
        label="Dashed line with markers",
        backend="matplotlib",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Interactive Plotly Backend

    All functions support interactive plotly plots for exploration.
    """)
    return


@app.cell
def _(PlotConfig, np, plot_line, plt, set_backend):
    # Generate interactive plot with plotly
    _x = np.linspace(0, 4 * np.pi, 200)
    _y = np.sin(_x) * np.exp(-_x / 10)
    set_backend("matplotlib")
    _config = PlotConfig(
        title="Interactive Damped Sine Wave (Hover to explore!)",
        xlabel="Time",
        ylabel="Amplitude",
        show=False,
    )
    plot_line(
        _y, x=_x, config=_config, color="red", label="Damped sine", backend="matplotlib"
    )
    plt.show()
    set_backend("plotly")
    _config = PlotConfig(
        title="Interactive Damped Sine Wave (Hover to explore!)",
        xlabel="Time",
        ylabel="Amplitude",
        show=False,
    )
    _fig = plot_line(
        _y, x=_x, config=_config, color="red", label="Damped sine", backend="plotly"
    )  # Don't auto-show
    # Create interactive plotly figure
    _fig.show()  # Don't auto-show
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    The `plots_1d` module provides:

    ✅ **Flexible backends**: Switch between matplotlib (static) and plotly (interactive)
    ✅ **Error visualization**: Add confidence bands with `std` parameter
    ✅ **Multiple datasets**: Compare multiple signals easily
    ✅ **Custom styling**: Full control over colors, markers, linestyles
    ✅ **Domain-specific functions**: Specialized plots for loss curves and boolean states

    ### Next Steps

    - Explore the `plots_2d` module for scatter plots and 2D visualizations
    - Check `embeddings` module for dimensionality reduction plots
    - See `heatmaps` module for matrix visualizations

    For full API documentation, see the module docstrings or visit the documentation site.
    """)
    return
