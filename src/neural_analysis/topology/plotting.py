"""
Visualization functions for Structure Index analysis.

This module provides plotting functions for visualizing Structure Index
computations, including parameter sweeps, overlap matrices, and directed graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_analysis.plotting.grid_config import GridLayoutConfig, PlotGrid, PlotSpec
from neural_analysis.utils.logging import get_logger

from .structure_index import draw_overlap_graph

if TYPE_CHECKING:
    import numpy.typing as npt

logger = get_logger(__name__)


def plot_structure_index(
    data: npt.NDArray[Any] | None = None,
    labels: npt.NDArray[Any] | None = None,
    overlap_mat: npt.NDArray[Any] | None = None,
    si_value: float | None = None,
    bin_label: tuple[Any, ...] | None = None,
    sweep_results: dict[tuple[int, int], dict[str, Any]] | None = None,
    save_path: str | Path | None = None,
    title: str = "Structure Index Analysis",
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    figsize: tuple[int, int] = (18, 6),
    show: bool = True,
) -> Any:
    """
    Plot Structure Index analysis results.

    This function creates visualizations for Structure Index computations:
    - Single result: 3-panel plot (embedding, overlap matrix, directed graph)
    - Parameter sweep: Line plot of SI values across parameters

    Parameters
    ----------
    data : ndarray, optional
        Neural data for embedding visualization, shape (n_samples, n_features)
    labels : ndarray, optional
        Labels for coloring embedding, shape (n_samples, n_label_dims)
    overlap_mat : ndarray, optional
        Overlap matrix between bins, shape (n_bins, n_bins)
    si_value : float, optional
        Structure index value
    bin_label : tuple, optional
        Bin label information (assignments, coordinates)
    sweep_results : dict[str, Any], optional
        Results from parameter sweep. Keys are (n_bins, n_neighbors) tuples.
    save_path : str or Path, optional
        Path to save figure
    title : str, default='Structure Index Analysis'
        Overall figure title
    backend : str, default='matplotlib'
        Plotting backend ('matplotlib' or 'plotly')
    figsize : tuple, default=(18, 6)
        Figure size (width, height) in inches
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : Figure or plotly Figure
        The created figure object

    Examples
    --------
    >>> # Plot single result
    >>> si, bin_info, overlap_mat, shuf_si = compute_structure_index(
    ...     data, labels, n_bins=10, n_neighbors=15
    ... )
    >>> fig = plot_structure_index(
    ...     data=data,
    ...     labels=labels,
    ...     overlap_mat=overlap_mat,
    ...     si_value=si,
    ...     bin_label=bin_info,
    ...     title="Session 001"
    ... )

    >>> # Plot parameter sweep
    >>> sweep_results = compute_structure_index_sweep(
    ...     data, labels, "session_001", "results.h5",
    ...     n_neighbors_list=[10, 15, 20, 25]
    ... )
    >>> fig = plot_structure_index(
    ...     sweep_results=sweep_results,
    ...     title="Parameter Sweep"
    ... )

    Notes
    -----
    - For single results, requires data, labels, overlap_mat, si_value, bin_label
    - For parameter sweeps, requires only sweep_results
    - The overlap matrix visualization uses a directed graph representation
    """
    if sweep_results is not None:
        return _plot_parameter_sweep(
            sweep_results=sweep_results,
            save_path=save_path,
            title=title,
            backend=backend,
            show=show,
        )

    elif all(v is not None for v in [data, labels, overlap_mat, si_value, bin_label]):
        # Type narrowing: all values are not None here
        assert data is not None
        assert labels is not None
        assert overlap_mat is not None
        assert si_value is not None
        assert bin_label is not None
        return _plot_single_result(
            data=data,
            labels=labels,
            overlap_mat=overlap_mat,
            si_value=si_value,
            bin_label=bin_label,
            save_path=save_path,
            title=title,
            backend=backend,
            figsize=figsize,
            show=show,
        )

    else:
        msg = (
            "Either provide sweep_results OR "
            "(data, labels, overlap_mat, si_value, bin_label)"
        )
        raise ValueError(msg)


def _plot_single_result(
    data: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    overlap_mat: npt.NDArray[Any],
    si_value: float,
    bin_label: tuple[Any, ...],
    save_path: str | Path | None,
    title: str,
    backend: Literal["matplotlib", "plotly"],
    figsize: tuple[int, int],
    show: bool,
) -> Any:
    """Plot single Structure Index result with 3 panels."""
    # Use matplotlib for this complex multi-panel plot
    fig = plt.figure(figsize=figsize)

    # Panel 1: Embedding visualization
    ax1 = fig.add_subplot(1, 3, 1, projection="3d" if data.shape[1] >= 3 else None)

    if data.shape[1] >= 3:
        # 3D scatter plot
        scatter = ax1.scatter(  # type: ignore[misc]
            data[:, 0],
            data[:, 1],
            data[:, 2],
            c=labels.ravel() if labels.ndim > 1 else labels,
            cmap="rainbow",
            s=20,
            alpha=0.6,
        )
        ax1.set_xlabel("Dim 1")
        ax1.set_ylabel("Dim 2")
        ax1.set_zlabel("Dim 3")  # type: ignore[attr-defined]
    elif data.shape[1] == 2:
        # 2D scatter plot
        ax1.scatter(
            data[:, 0],
            data[:, 1],
            c=labels,
            cmap="tab10",
            s=30,
            alpha=0.6,
            edgecolors="k",
            linewidth=0.5,
        )
        ax1.set_xlabel("Dim 1")
        ax1.set_ylabel("Dim 2")
    else:
        # 1D plot
        scatter = ax1.scatter(
            np.arange(len(data)),
            data[:, 0],
            c=labels.ravel() if labels.ndim > 1 else labels,
            cmap="rainbow",
            s=20,
            alpha=0.6,
        )
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Value")

    ax1.set_title("Neural Embedding", fontsize=14)
    plt.colorbar(scatter, ax=ax1, label="Label Value", shrink=0.8)

    # Panel 2: Overlap matrix heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.matshow(
        overlap_mat, vmin=0, vmax=0.5, cmap=matplotlib.cm.get_cmap("viridis")
    )
    ax2.xaxis.set_ticks_position("bottom")
    ax2.set_title("Adjacency Matrix", fontsize=14)
    ax2.set_xlabel("bin-groups", fontsize=12)
    ax2.set_ylabel("bin-groups", fontsize=12)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, ticks=[0, 0.25, 0.5])
    cbar.set_label("overlap score", rotation=90, fontsize=12)

    # Panel 3: Directed graph
    ax3 = fig.add_subplot(1, 3, 3)
    node_labels = np.round(bin_label[1][:, 0, 1], 2) if len(bin_label) > 1 else None
    draw_overlap_graph(
        overlap_mat,
        ax=ax3,
        node_cmap=matplotlib.cm.get_cmap("rainbow"),
        edge_cmap=matplotlib.cm.get_cmap("Greys"),
        node_names=node_labels,
    )
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    ax3.set_xlim((xlim[0] * 1.2, xlim[1] * 1.2))
    ax3.set_ylim((ylim[0] * 1.2, ylim[1] * 1.2))
    ax3.set_title("Directed Graph", fontsize=14)

    # Add SI value as text
    ax3.text(
        0.98,
        0.05,
        f"SI: {si_value:.3f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax3.transAxes,
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_parameter_sweep(
    sweep_results: dict[tuple[int, int], dict[str, Any]],
    save_path: str | Path | None,
    title: str,
    backend: str,
    show: bool,
) -> Any:
    """Plot parameter sweep results as line plot."""
    # Extract parameter combinations and SI values
    param_combos = sorted(sweep_results.keys())

    if not param_combos:
        msg = "No results to plot"
        raise ValueError(msg)

    # Check if it's a single-variable sweep or multi-variable
    n_bins_values = sorted({n_bins for n_bins, _ in param_combos})
    n_neighbors_values = sorted({n_neighbors for _, n_neighbors in param_combos})

    if len(n_bins_values) == 1:
        # Sweep over n_neighbors only
        x_values = n_neighbors_values
        y_values = [
            sweep_results[(n_bins_values[0], n_neighbors)]["SI"]
            for n_neighbors in n_neighbors_values
        ]
        xlabel = "n_neighbors"
        plot_title = f"{title}\n(n_bins={n_bins_values[0]})"
    elif len(n_neighbors_values) == 1:
        # Sweep over n_bins only
        x_values = n_bins_values
        y_values = [
            sweep_results[(n_bins, n_neighbors_values[0])]["SI"]
            for n_bins in n_bins_values
        ]
        xlabel = "n_bins"
        plot_title = f"{title}\n(n_neighbors={n_neighbors_values[0]})"
    else:
        # Multi-variable sweep: create heatmap
        return _plot_parameter_heatmap(
            sweep_results=sweep_results,
            n_bins_values=n_bins_values,
            n_neighbors_values=n_neighbors_values,
            save_path=save_path,
            title=title,
            show=show,
        )

    # Create line plot using GridPlot
    spec = PlotSpec(
        data={"x": np.array(x_values), "y": np.array(y_values)},
        plot_type="line",
        title=plot_title,
        color="steelblue",
        line_width=2.5,
        marker="o",
        marker_size=80,
        label="Structure Index",
    )

    layout = GridLayoutConfig(rows=1, cols=1)
    config = {
        "xlabel": xlabel,
        "ylabel": "Structure Index",
        "show_legend": True,
        "grid": True,
    }

    grid = PlotGrid(
        plot_specs=[spec],
        layout=layout,
        backend=backend
        if isinstance(backend, str) and backend in ["matplotlib", "plotly"]
        else "matplotlib",  # type: ignore[arg-type]
    )

    fig = grid.plot()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if backend == "matplotlib":
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            fig.write_html(str(save_path))
        logger.info(f"Saved figure to {save_path}")

    if not show and backend == "matplotlib":
        plt.close(fig)

    return fig


def _plot_parameter_heatmap(
    sweep_results: dict[tuple[int, int], dict[str, Any]],
    n_bins_values: list[int],
    n_neighbors_values: list[int],
    save_path: str | Path | None,
    title: str,
    show: bool,
) -> Any:
    """Plot 2D parameter sweep as heatmap."""
    # Create matrix of SI values
    si_matrix = np.zeros((len(n_bins_values), len(n_neighbors_values)))

    for i, n_bins in enumerate(n_bins_values):
        for j, n_neighbors in enumerate(n_neighbors_values):
            key = (n_bins, n_neighbors)
            if key in sweep_results:
                si_matrix[i, j] = sweep_results[key]["SI"]
            else:
                si_matrix[i, j] = np.nan

    # Create heatmap using GridPlot
    spec = PlotSpec(
        data=si_matrix,
        plot_type="heatmap",
        title=title,
        cmap="viridis",
        colorbar=True,
        colorbar_label="Structure Index",
    )

    layout = GridLayoutConfig(rows=1, cols=1)
    config = {
        "xlabel": "n_neighbors",
        "ylabel": "n_bins",
        "xtick_labels": [str(x) for x in n_neighbors_values],
        "ytick_labels": [str(y) for y in n_bins_values],
    }

    grid = PlotGrid(
        plot_specs=[spec],
        layout=layout,
        backend="matplotlib",  # Heatmap works best with matplotlib
    )

    fig = grid.plot()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_structure_index_comparison(
    results_dict: dict[str, dict[tuple[int, int], dict[str, Any]]],
    parameter: str = "n_neighbors",
    fixed_params: dict[str, int] | None = None,
    save_path: str | Path | None = None,
    title: str = "Structure Index Comparison",
    backend: str = "matplotlib",
    show: bool = True,
) -> Any:
    """
    Compare Structure Index across multiple datasets or conditions.

    Parameters
    ----------
    results_dict : dict[str, Any]
        Dictionary mapping dataset names to their sweep results
    parameter : str, default='n_neighbors'
        Parameter to plot on x-axis ('n_neighbors' or 'n_bins')
    fixed_params : dict[str, Any], optional
        Fixed parameter values, e.g., {'n_bins': 10}
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    backend : str, default='matplotlib'
        Plotting backend
    show : bool, default=True
        Whether to display figure

    Returns
    -------
    fig : Figure
        The created figure

    Examples
    --------
    >>> # Load results for multiple sessions
    >>> results1 = load_structure_index_results("results.h5", "session_001")
    >>> results2 = load_structure_index_results("results.h5", "session_002")
    >>> # Compare them
    >>> fig = plot_structure_index_comparison(
    ...     {"Session 1": results1, "Session 2": results2},
    ...     parameter="n_neighbors",
    ...     fixed_params={"n_bins": 10}
    ... )
    """
    if fixed_params is None:
        fixed_params = {}

    # Create plot specs for each dataset
    plot_specs = []
    colors = matplotlib.cm.get_cmap("tab10")(np.linspace(0, 1, len(results_dict)))

    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        # Extract values for the specified parameter
        if parameter == "n_neighbors":
            param_values = sorted(
                {
                    n_neighbors
                    for (n_bins, n_neighbors) in results
                    if "n_bins" not in fixed_params or n_bins == fixed_params["n_bins"]
                }
            )
            y_values = [
                results[(fixed_params.get("n_bins", list(results.keys())[0][0]), pval)][
                    "SI"
                ]
                for pval in param_values
            ]
        else:  # parameter == 'n_bins'
            param_values = sorted(
                {
                    n_bins
                    for (n_bins, n_neighbors) in results
                    if "n_neighbors" not in fixed_params
                    or n_neighbors == fixed_params["n_neighbors"]
                }
            )
            y_values = [
                results[
                    (
                        pval,
                        fixed_params.get("n_neighbors", list(results.keys())[0][1]),
                    )
                ]["SI"]
                for pval in param_values
            ]

        spec = PlotSpec(
            data={"x": np.array(param_values), "y": np.array(y_values)},
            plot_type="line",
            color=matplotlib.colors.rgb2hex(colors[idx]),
            line_width=2,
            marker="o",
            marker_size=60,
            label=dataset_name,
        )
        plot_specs.append(spec)

    layout = GridLayoutConfig(rows=1, cols=1)
    config = {
        "xlabel": parameter,
        "ylabel": "Structure Index",
        "title": title,
        "show_legend": True,
        "grid": True,
    }

    grid = PlotGrid(
        plot_specs=plot_specs,
        layout=layout,
        backend=backend
        if isinstance(backend, str) and backend in ["matplotlib", "plotly"]
        else "matplotlib",  # type: ignore[arg-type]
    )

    fig = grid.plot()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if backend == "matplotlib":
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            fig.write_html(str(save_path))
        logger.info(f"Saved figure to {save_path}")

    if not show and backend == "matplotlib":
        plt.close(fig)

    return fig
