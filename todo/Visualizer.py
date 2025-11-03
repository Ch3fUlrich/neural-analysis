"""
MIGRATION STATUS:
=================
This file is being gradually migrated to the new modular plotting system.

MIGRATED FUNCTIONS (can be removed):
- 3D scatter plotting → src/neural_analysis/plotting/plots_3d.py::plot_scatter_3d
- 3D trajectory plotting → src/neural_analysis/plotting/plots_3d.py::plot_trajectory_3d

PENDING MIGRATION:
- Embedding visualizations (plot_embedding, plot_embedding_2d, plot_embedding_3d)
- Heatmaps and colormaps
- KDE plots
- Line plots
- Neural activity rasters
- And more...

TODO: Remove or deprecate migrated functions to reduce file size
"""

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from seaborn import heatmap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.colors as pcolors
import plotly.io as pio

from scipy.spatial import ConvexHull

from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import itertools

# Suppress tight_layout warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="This figure includes Axes that are not compatible with tight_layout",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The figure layout has changed to tight",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.ax.set_xlabel(xlabel)",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.ax.legend()",
)

import sys
from pathlib import Path
from structure_index import draw_graph

# plt.style.use("default")
plt.style.use("default")

import plotly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import colorsys

from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

import matplotlib.axes
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

# from pySankey.sankey import sankey
import cebra
import pandas as pd

from typing import Union, Optional, Tuple
from numpy.typing import ArrayLike
from Helper import *


class Vizualizer:
    def __init__(self, root_dir) -> None:
        self.save_dir = Path(root_dir).joinpath("figures")
        self.save_dir.mkdir(exist_ok=True)

    def calculate_alpha(
        value: Union[int, float, List[Union[int, float]]],
        min_value: Union[int, float, None] = None,
        max_value: Union[int, float, None] = None,
        min_alpha: float = 0.3,
        max_alpha: float = 1.0,
    ) -> Union[float, List[float]]:
        """
        Calculate alpha value(s) based on value's position in range.

        Parameters
        ----------
        value : float or list of float/int
            The value(s) for which to calculate alpha.
        min_value : float, optional
            Minimum value of the range. If None and value is a list, uses min of list.
        max_value : float, optional
            Maximum value of the range. If None and value is a list, uses max of list.
        min_alpha : float, optional
            Minimum alpha value (default 0.3).
        max_alpha : float, optional
            Maximum alpha value (default 1.0).

        Returns
        -------
        float or list of float
            Calculated alpha value(s) constrained between min_alpha and max_alpha.
        """
        # Convert single value to list for uniform processing
        values = make_list_ifnot(value)
        if len(values) == 0:
            raise ValueError("No values provided for alpha calculation.")
        elif len(values) == 1 and (min_value is None or max_value is None):
            return [1]

        # Determine range if not provided
        min_val = min_value if min_value is not None else min(values)
        max_val = max_value if max_value is not None else max(values)

        # Handle edge case where range is zero
        if max_val == min_val:
            return (
                min_alpha if not isinstance(value, list) else [min_alpha] * len(values)
            )

        # Calculate alphas
        alphas = []
        for val in values:
            normalized = (val - min_val) / (max_val - min_val)
            alpha = min_alpha + (max_alpha - min_alpha) * normalized
            # Ensure alpha stays within specified bounds
            alpha = max(min_alpha, min(max_alpha, alpha))
            alphas.append(alpha)

        # Return single value or list based on input type
        return alphas[0] if len(alphas) == 1 else alphas

    @staticmethod
    def generate_similar_colors(base_color, num_colors):
        """
        Generates a list of similar colors based on a base color.
        """
        base_hls = colorsys.rgb_to_hls(*base_color[:3])
        colors = []
        for i in range(num_colors):
            hue = (base_hls[0] + i * 0.05) % 1.0  # Small hue variations
            lightness = max(
                0, min(1, base_hls[1] + (i - num_colors / 2) * 0.1)
            )  # Slight lightness variations
            rgb = colorsys.hls_to_rgb(hue, lightness, base_hls[2])
            colors.append(rgb)
        return colors

    @staticmethod
    def add_hull(points, ax, hull_alpha=0.2, facecolor="b", edgecolor="r"):
        """
        Adds a convex hull
        """
        if len(points) < 3:
            print("Not enough points for a hull")
            return

        hull = ConvexHull(points)
        vertices = hull.points[hull.vertices]
        if points.shape[1] == 2:  # 2D case
            polygon = Polygon(
                vertices,
                closed=True,
                alpha=hull_alpha,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            ax.add_patch(polygon)

        elif points.shape[1] == 3:  # 3D case
            poly3d = []
            for s in hull.simplices:
                skip_simplices = False
                for i in s:
                    if i >= len(vertices):
                        skip_simplices = True

                if skip_simplices:
                    continue
                poly3d.append(vertices[s])

            ax.add_collection3d(
                Poly3DCollection(
                    poly3d,
                    facecolors=facecolor,
                    linewidths=0.01,
                    edgecolors=edgecolor,
                    alpha=hull_alpha,
                )
            )

    #############################  Data Plots #################################################
    @staticmethod
    def default_plot_attributes():
        return {
            "fps": None,
            "title": None,
            "ylable": None,
            "ylimits": None,
            "yticks": None,
            "xlable": None,
            "xlimits": None,
            "xticks": None,
            "num_ticks": None,
            "figsize": None,
            "save_path": None,
        }

    @staticmethod
    def define_plot_parameter(
        plot_attributes=None,
        fps=None,
        title=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        xlable=None,
        xticks=None,
        num_ticks=None,
        xlimits=None,
        figsize=None,
        save_path=None,
    ):
        if plot_attributes is None:
            plot_attributes = Vizualizer.default_plot_attributes()
        plot_attributes["fps"] = fps or plot_attributes["fps"]
        plot_attributes["title"] = title or plot_attributes["title"]
        plot_attributes["title"] = (
            plot_attributes["title"]
            if plot_attributes["title"][-4:] == "data"
            else plot_attributes["title"] + " data"
        )

        plot_attributes["ylable"] = ylable or plot_attributes["ylable"] or None
        plot_attributes["ylimits"] = ylimits or plot_attributes["ylimits"] or None
        plot_attributes["yticks"] = yticks or plot_attributes["yticks"] or None
        plot_attributes["xlable"] = xlable or plot_attributes["xlable"] or "time"
        plot_attributes["xlimits"] = xlimits or plot_attributes["xlimits"] or None
        plot_attributes["xticks"] = xticks or plot_attributes["xticks"] or None
        plot_attributes["num_ticks"] = num_ticks or plot_attributes["num_ticks"] or 100
        plot_attributes["figsize"] = figsize or plot_attributes["figsize"] or (20, 3)
        plot_attributes["save_path"] = save_path or plot_attributes["save_path"]

        # create plot dir if missing
        if plot_attributes["save_path"] is not None:
            plot_attributes["save_path"] = Path(plot_attributes["save_path"])
            if not plot_attributes["save_path"].parent.exists():
                plot_attributes["save_path"].parent.mkdir(parents=True, exist_ok=True)
        else:
            plot_attributes["save_path"] = None

        return plot_attributes

    @staticmethod
    def default_plot_start(
        plot_attributes: dict = None,
        figsize=None,
        title=None,
        xlable=None,
        xlimits=None,
        xticks=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        fps=None,
        num_ticks=50,
        save_path=None,
    ):
        plot_attributes = Vizualizer.define_plot_parameter(
            plot_attributes=plot_attributes,
            figsize=figsize,
            title=title,
            ylable=ylable,
            ylimits=ylimits,
            yticks=yticks,
            xlable=xlable,
            xlimits=xlimits,
            xticks=xticks,
            num_ticks=num_ticks,
            fps=fps,
            save_path=save_path,
        )
        plt.figure(figsize=plot_attributes["figsize"])
        plt.title(plot_attributes["title"])
        plt.ylabel(plot_attributes["ylable"])
        if plot_attributes["ylimits"]:
            plt.ylim(plot_attributes["ylimits"])
        if plot_attributes["yticks"]:
            plt.yticks(plot_attributes["yticks"][0], plot_attributes["yticks"][1])
        plt.xlabel(plot_attributes["xlable"])
        plt.tight_layout()
        plt.xlim(plot_attributes["xlimits"])
        return plot_attributes

    @staticmethod
    def plot_image(
        plot_attributes=None,
        figsize=(10, 10),
        save_path=None,
        show=False,
    ):
        if plot_attributes is None:
            plot_attributes = Vizualizer.default_plot_attributes()

        plot_attributes["figsize"] = plot_attributes["figsize"] or figsize
        plot_attributes["save_path"] = plot_attributes["save_path"] or save_path

        plt.figure(figsize=plot_attributes["figsize"])
        # Load the image
        image = plt.imread(plot_attributes["save_path"])
        plt.imshow(image)
        plt.axis("off")

        if show:
            plt.show()

    @staticmethod
    def default_plot_ending(
        plot_attributes=None,
        regenerate_plot=False,
        save_path=None,
        show=False,
        dpi=300,
        as_pdf=False,
    ):
        if plot_attributes is None:
            plot_attributes = Vizualizer.default_plot_attributes()
        plot_attributes["save_path"] = plot_attributes["save_path"] or save_path

        if regenerate_plot and plot_attributes["save_path"] is not None:
            title = Path(plot_attributes["save_path"]).stem
            save_dir = Path(plot_attributes["save_path"]).parent
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")

        if show:
            plt.show()
            plt.close()

    def data_plot_1D(
        data,
        labels=None,
        plot_attributes: dict = None,
        marker_pos=None,
        marker=None,
        seconds_interval=5,
    ):
        if data.ndim == 1 or data.ndim == 2 and data.shape[1] == 1:
            if data.ndim == 2:
                data = data.flatten()
            label = None
            if labels is not None:
                if len(labels) != 1:
                    raise ValueError(
                        f"Labels provided for 1D data, but data is already 1D. Labels: {labels}"
                    )
                else:
                    label = labels[0]

            # check if data is bool array
            if (
                isinstance(data, np.ndarray)
                and data.dtype == bool
                or isinstance(data, list)
                and all(isinstance(i, bool) for i in data)
            ):
                # color area from y 0 to 1 in gray when False and cyan when True
                for state in [False, True]:
                    # fill without overlapping and without border
                    plt.fill_between(
                        range(len(data)),
                        0,
                        1,
                        step="post",
                        interpolate=True,
                        edgecolor="none",
                        linewidth=0,
                        linestyle="solid",
                        facecolor="none",
                        hatch="",
                        where=data == state,
                        color="blue" if state else "white",
                        alpha=0.3,
                        zorder=0,
                        label="moving" if state else "stationary",
                    )
            else:
                plt.plot(data, zorder=1, label=label)
        else:
            if labels is None:
                named_dimensions = {0: "x", 1: "y", 2: "z"}
            else:
                named_dimensions = labels
                if len(named_dimensions) != data.shape[1]:
                    raise ValueError(
                        f"Labels length {len(named_dimensions)} does not match data dimensions {data.shape[1]}"
                    )
            for dim in range(data.shape[1]):
                plt.plot(data[:, dim], zorder=1, label=f"{named_dimensions[dim]}")
            plt.legend()

        if marker_pos is not None:
            marker = "." if marker is None else marker
        else:
            marker_pos = range(len(data)) if marker is not None else None

        if marker:
            plt.scatter(marker_pos, data[marker_pos], marker=marker, s=10, color="red")

        num_frames = data.shape[0]

        if plot_attributes is not None:
            if plot_attributes["xticks"]:
                if plot_attributes["xticks"] == "auto":
                    plt.xticks()
                else:
                    plt.xticks(plot_attributes["xticks"])
            else:
                xticks, xpos = Vizualizer.define_xticks(
                    plot_attributes=plot_attributes,
                    num_frames=num_frames,
                    fps=plot_attributes["fps"],
                    num_ticks=plot_attributes["num_ticks"],
                    seconds_interval=seconds_interval,
                )
            plt.xticks(xpos, xticks, rotation=45)

    @staticmethod
    def data_plot_2D(
        data,
        plot_attributes,
        position_data,
        border_limits=None,
        marker_pos=None,
        marker=None,
        fps=None,
        seconds_interval=5,
        colormap_label: str = None,
        tick_label=None,
        tick_pos=None,
        color_by="value",
        cmap="plasma",  # "viridis", "winter", "plasma"
    ):
        # data = data[:15000, :]
        # data = may_butter_lowpass_filter(
        #    data,
        #    smooth=True,
        #    cutoff=1,
        #    fps=20,
        #    order=2,
        # )
        data, position_data = force_equal_dimensions(data, position_data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # Convert coordinates to a numpy array if it's not already
        # Convert to cm for better visualization in plot
        coordinates = np.array(position_data) * 100
        border_limits = (
            np.array(border_limits) * 100 if border_limits is not None else None
        )

        # Extract x and y coordinates
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]

        # Generate a time array based on the number of coordinates
        if color_by == "time":
            num_frames = data.shape[0]
            color_map_label = f"Time"
            color_value_reference = range(num_frames)
            tick_label, tick_pos = Vizualizer.define_xticks(
                plot_attributes=plot_attributes,
                num_frames=num_frames,
                fps=plot_attributes["fps"],
                num_ticks=int(plot_attributes["num_ticks"] / 2),
                seconds_interval=seconds_interval,
            )
            scatter_alpha = 0.8
            dot_size = 1
        elif color_by == "value":
            absolute_data = np.linalg.norm(np.abs(data), axis=1)
            color_value_reference = np.array(absolute_data)
            color_map_label = colormap_label or plot_attributes["ylable"]

            if tick_label is None:
                if plot_attributes["yticks"] is not None:
                    tick_pos, tick_label = plot_attributes["yticks"]
                if tick_label is not None and tick_pos is None:
                    tick_pos = np.range(0, 1, len(tick_label))
            scatter_alpha = 0.8
            dot_size = 3

        if "cmap" in plot_attributes:
            cmap = plot_attributes["cmap"]

        # Create the plot
        scatter = plt.scatter(
            x_coords,
            y_coords,
            c=color_value_reference,
            cmap=cmap,
            s=dot_size,
            alpha=scatter_alpha,
        )

        if border_limits is not None:
            # Add border lines
            plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)
            plt.axvline(x=border_limits[0], color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=border_limits[1], color="r", linestyle="--", alpha=0.5)

        x_data_range = max(coordinates[:, 0]) - min(coordinates[:, 0])
        y_data_range = max(coordinates[:, 1]) - min(coordinates[:, 1])
        xlimits = (
            min(coordinates[:, 0] - x_data_range * 0.03),
            max(coordinates[:, 0] + x_data_range * 0.03),
        )
        ylimits = (
            min(coordinates[:, 1] - y_data_range * 0.03),
            max(coordinates[:, 1] + y_data_range * 0.03),
        )

        # define x and y ticks
        # ylimits_rounded = (round(ylimits[0]), round(ylimits[1]))
        # xlimits_rounded = (round(xlimits[0]), round(xlimits[1]))
        # y_ticks = np.arange(
        #    ylimits_rounded[0], ylimits_rounded[1], np.diff(ylimits_rounded) / 10
        # )
        # x_ticks = np.arange(
        #    xlimits_rounded[0], xlimits_rounded[1], np.diff(xlimits_rounded) / 10
        # )
        # plt.yticks(y_ticks)
        # plt.xticks(x_ticks)

        plt.xlabel("X position (cm)")
        plt.ylabel("Y position (cm)")
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        # plt.grid(True, alpha=0.2)

        # Add a colorbar to show the time mapping
        cbar = plt.colorbar(scatter, label=color_map_label)
        if tick_label is not None and tick_pos is not None:
            cbar.set_ticks(tick_pos)
            cbar.set_ticklabels(tick_label)

    @staticmethod
    def define_xticks(
        plot_attributes=None,
        num_frames=None,
        fps=None,
        num_ticks=None,
        seconds_interval=5,
    ):
        if num_frames is not None:
            xticks, xpos = range_to_times_xlables_xpos(
                end=num_frames,
                fps=fps,
                seconds_per_label=seconds_interval,
            )

            # reduce number of xticks
            if len(xpos) > num_ticks:
                steps = round(len(xpos) / num_ticks)
                xticks = xticks[::steps]
                xpos = xpos[::steps]
        else:
            xticks, xpos = None, None
        return xticks, xpos

    @staticmethod
    def plot_neural_activity_raster(
        data,
        fps,
        num_ticks=None,
        seconds_interval=5,
    ):

        binarized_data = data
        num_time_steps, num_neurons = binarized_data.shape
        # Find spike indices for each neuron
        spike_indices = np.nonzero(binarized_data)
        # Creating an empty image grid
        image = np.zeros((num_neurons, num_time_steps))
        # Marking spikes as pixels in the image grid
        image[spike_indices[1], spike_indices[0]] = 1
        # Plotting the raster plot using pixels
        plt.imshow(image, cmap="gray", aspect="auto", interpolation="none")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization of trials/neurons

        xticks, xpos = Vizualizer.define_xticks(
            num_frames=num_time_steps,
            fps=fps,
            num_ticks=num_ticks,
            seconds_interval=seconds_interval,
        )
        plt.xticks(xpos, xticks, rotation=45)

    ########################################################################################################################
    @staticmethod
    def plot_embedding(
        embedding,
        embedding_labels: dict,
        min_val=None,
        max_val=None,
        ax: Optional[matplotlib.axes.Axes] = None,
        show_hulls=False,
        title="Embedding",
        cmap="rainbow",
        plot_legend=True,
        colorbar_ticks=None,
        projection="3d",
        markersize=None,
        alpha=None,
        figsize=(10, 10),
        dpi=300,
        as_pdf=False,
        show=True,
        save_dir=None,
        additional_title="",
    ):
        embedding, labels = force_equal_dimensions(
            embedding, embedding_labels["labels"]
        )

        if projection == "3d" and embedding.shape[1] == 2:
            err_msg = f"3D projection requested, but embedding is 2D. Using 2D projection instead."
            global_logger.critical(err_msg)
            raise ValueError(err_msg)
        elif projection == "2d" and embedding.shape[1] == 3:
            global_logger.info(
                f"Converting 3D embedding to 2D using PCA for plotting embeddings"
            )
            embedding = pca_numba(embedding.astype(np.float64))
            # embedding = mds_numba(embedding.astype(np.float64))

        if embedding.shape[1] == 2:
            ax = Vizualizer.plot_embedding_2d(
                axis=ax,
                embedding=embedding,
                embedding_labels=labels,
                show_hulls=show_hulls,
                markersize=markersize,
                min_val=min_val,
                max_val=max_val,
                alpha=alpha,
                cmap=cmap,
                title=title,
                figsize=figsize,
                dpi=dpi,
                as_pdf=as_pdf,
                show=show,
                save_dir=save_dir,
                plot_legend=plot_legend,
                additional_title=additional_title,
            )
        elif embedding.shape[1] == 3:
            ax = Vizualizer.plot_embedding_3d(
                axis=ax,
                embedding=embedding,
                embedding_labels=labels,
                show_hulls=show_hulls,
                markersize=markersize,
                min_val=min_val,
                max_val=max_val,
                alpha=alpha,
                cmap=cmap,
                title=title,
                figsize=figsize,
                as_pdf=as_pdf,
                show=show,
                save_dir=save_dir,
                plot_legend=plot_legend,
                additional_title=additional_title,
            )
        else:
            raise NotImplementedError(
                "Invalid labels dimension. Choose 2D or 3D labels."
            )
        return ax

    def plot_embedding_2d(
        embedding: Union[npt.NDArray, torch.Tensor],
        embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]],
        show_hulls: bool = False,
        idx_order: Optional[Tuple[int]] = None,
        markersize: float = 0.5,
        min_val=None,
        max_val=None,
        alpha: float = 0.4,
        cmap: str = "cool",
        title: str = "2D Embedding",
        additional_title: str = "",
        axis: Optional[matplotlib.axes.Axes] = None,
        figsize: tuple = (5, 5),
        dpi: float = 100,
        plot_legend: bool = True,
        as_pdf: bool = False,
        show: bool = True,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        This function is based on the plot_embedding function from the cebra library.
        """
        markersize = markersize or 2
        alpha = alpha or 0.5
        # define the axis
        if axis is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot()
        else:
            ax = axis

        plot_labels, min_vals, max_vals = Vizualizer.create_RGBA_colors_from_2d(
            embedding_labels, min_val=min_val, max_val=max_val
        )

        # define idx order
        if idx_order is None:
            idx_order = (0, 1)

        else:
            # If the idx_order was provided by the user
            ## Check size validity
            if len(idx_order) != 2:
                raise ValueError(
                    f"idx_order must contain 2 dimension values, got {len(idx_order)}."
                )

            # Check value validity
            for dim in idx_order:
                if dim < 0 or dim > embedding.shape[1] - 1:
                    raise ValueError(
                        f"List of dimensions to plot is invalid, got {idx_order}, with {dim} invalid."
                        f"Values should be between 0 and {embedding.shape[1]}."
                    )

        # plot the embedding
        (
            idx1,
            idx2,
        ) = idx_order
        ax.scatter(
            x=embedding[:, idx1],
            y=embedding[:, idx2],
            c=plot_labels,
            cmap=cmap,
            alpha=alpha,
            s=markersize,
            **kwargs,
        )

        if show_hulls:
            print("WARNING: show hulls not implemented properly for 2D")
            color_groups = values_to_groups(
                values=plot_labels,
                points=embedding,
                filter_outliers=True,
                contamination=0.2,
            )
            for color, points in color_groups.items():
                Vizualizer.add_hull(
                    points,
                    ax,
                    hull_alpha=0.6,
                    facecolor=color,
                    edgecolor="r",
                )
        # Remove all spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.set_xlim(embedding[:, idx1].min(), embedding[:, idx1].max())
        ax.grid(False)
        title = title + additional_title
        ax.set_title(title, y=1.0, pad=-10)

        if plot_legend:
            Vizualizer.add_2d_colormap_legend(fig, discret_n_colors=plot_labels)

        if axis is None:
            Vizualizer.plot_ending(
                title=title, as_pdf=as_pdf, show=show, save_dir=save_dir
            )

        return ax

    def plot_embedding_3d(
        embedding: Union[npt.NDArray, torch.Tensor],
        embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]],
        show_hulls: bool = False,
        idx_order: Optional[Tuple[int]] = None,
        markersize: float = 0.1,
        min_val=None,
        max_val=None,
        marker: str = ".",
        alpha: float = 0.4,
        cmap: str = "cool",
        title: str = "3D Embedding",
        additional_title: str = "",
        axis: Optional[matplotlib.axes.Axes] = None,
        figsize: tuple = (5, 5),
        dpi: float = 300,
        background: str = None,
        grid: bool = False,
        plot_legend: bool = True,
        as_pdf: bool = False,
        show: bool = True,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        This function is based on the plot_embedding function from the cebra library.
        """
        markersize = markersize or 0.5
        alpha = alpha or 0.4
        # define the axis
        if axis is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(projection="3d")
        else:
            ax = axis

        plot_labels, min_vals, max_vals = Vizualizer.create_RGBA_colors_from_2d(
            embedding_labels, min_val=min_val, max_val=max_val
        )

        # define idx order
        if idx_order is None:
            idx_order = (0, 1, 2)

        # plot the embedding
        idx1, idx2, idx3 = idx_order
        ax.scatter(
            xs=embedding[:, idx1],
            ys=embedding[:, idx2],
            zs=embedding[:, idx3],
            c=plot_labels,
            cmap=cmap,
            alpha=alpha,
            marker=marker,
            s=markersize,
            **kwargs,
        )

        if show_hulls:
            color_groups = values_to_groups(values=plot_labels, points=embedding)
            for color, points in color_groups.items():
                Vizualizer.add_hull(
                    points, ax, hull_alpha=0.2, facecolor=color, edgecolor="r"
                )

        if grid:
            ax.grid(grid, alpha=0.5)
        else:
            ax.grid(False)

        if background:
            ax.xaxis.pane.set_facecolor(background)
            ax.yaxis.pane.set_facecolor(background)
            ax.zaxis.pane.set_facecolor(background)
        else:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        title = title + additional_title
        ax.set_title(title, y=1.08, pad=-10)

        if plot_legend:
            if (
                embedding_labels.ndim == 1
                or embedding_labels.ndim == 2
                and embedding_labels.shape[1] == 1
            ):
                label_size = Vizualizer.auto_fontsize(fig) * 0.8
                Vizualizer.add_1d_colormap_legend(
                    ax=ax,
                    labels=embedding_labels,
                    label_name="labels",
                    label_size=label_size,
                    ticks=None,
                    cmap=cmap,
                    shrink=0.8,
                )
            elif embedding_labels.ndim == 2:
                Vizualizer.add_2d_colormap_legend(fig=fig)
            else:
                raise ValueError("Invalid labels dimension. Choose 2D or 3D labels.")

        if axis is None:
            Vizualizer.plot_ending(
                title=title, as_pdf=as_pdf, show=show, save_dir=save_dir
            )
        return ax

    def add_1d_colormap_legend(
        labels,
        fig=None,
        ax=None,
        label_name="labels",
        label_size=10,
        shrink=1,
        ticks=None,
        cmap="rainbow",
        move_right=1,
    ):
        if fig is None and ax is None:
            raise ValueError("Plotting Issue: Either fig or ax must be provided.")
        # Create a ScalarMappable object using the specified colormap
        sm = plt.cm.ScalarMappable(cmap=cmap)
        unique_labels = np.unique(labels)
        unique_labels.sort()
        sm.set_array(unique_labels)  # Set the range of values for the colorbar

        if fig is None:
            # Manually create colorbar
            cbar = plt.colorbar(sm, ax=ax, shrink=shrink)
        else:
            # add 1D colorbar to the right of the plot
            cax = fig.add_axes([move_right, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(sm, cax=cax)

        # Adjust colorbar ticks if specified
        cbar.set_label(
            label_name, fontsize=label_size
        )  # Set the label for the colorbar
        if ticks is not None:
            cbar.ax.yaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # Adjust ticks to integers
            cbar.set_ticks(
                np.linspace(cbar.vmin, cbar.vmax, len(ticks))
            )  # Set custom ticks
            cbar.set_ticklabels(ticks, fontsize=label_size)  # Set custom tick labels

    def add_2d_colormap_legend(
        fig,
        Legend=None,
        cmap="rainbow",
        discret_n_colors=None,
        colors=None,
        xticks=None,
        yticks=None,
        additional_title="",
        legend_left=None,
    ):
        if xticks is None:
            xticks = []
        if yticks is None:
            yticks = []
        # Get the current axes (main plot)
        main_ax = fig.gca()
        # Get the position of the main plot
        main_pos = main_ax.get_position()
        # Calculate the position of the legend
        legend_left = (
            main_pos.x1 + main_pos.x1 / 10 if legend_left is None else legend_left
        )

        cax = fig.add_axes([legend_left, 0.55, 0.3, 0.3])
        cax.set_xlabel("X")
        cax.set_ylabel("Y")

        cmap = None
        norm = None
        if Legend is None:
            # make RGB image, p1 to red channel, p2 to blue channel
            cp1 = np.linspace(0, 1)
            cp2 = np.linspace(0, 1)
            Cp1, Cp2 = np.meshgrid(cp1, cp2)
            C0 = np.zeros_like(Cp1) + 0.5
            Legend = np.dstack((Cp1, C0, Cp2))

            if discret_n_colors is not None:
                # create discrete colormap by splitting colormap into n colors
                colors = cm.get_cmap(cmap)(
                    np.linspace(0, 1, discret_n_colors)
                )  # Using rainbow colormap
                cmap = mcolors.ListedColormap(colors)
                bounds = np.linspace(0, 1, discret_n_colors + 1)

                # Rescale Legend to have discrete values
                Cp1_discrete = np.zeros_like(Cp1)
                Cp2_discrete = np.zeros_like(Cp2)
                for i in np.arange(0, 1, 1 / discret_n_colors):
                    Cp1_discrete[Cp1 >= i] = i
                    Cp2_discrete[Cp2 >= i] = i

                normalized_cp1 = normalize_01(Cp1_discrete, axis=1)
                normalized_cp2 = normalize_01(Cp2_discrete, axis=0)
                Legend = np.dstack((normalized_cp1, C0, normalized_cp2))

        # parameters range between 0 and 1
        xlabels = [f"{x:.1f}" for x in xticks]
        ylabels = [f"{y:.1f}" for y in yticks]
        cax.set_xticks(np.linspace(0, 1, len(xticks)))
        cax.set_yticks(np.linspace(0, 1, len(yticks)))
        cax.imshow(
            Legend,
            # origin="lower",
            extent=[0, 1, 0, 1],
        )
        cax.set_xticklabels(xlabels, rotation=45, ha="right")
        cax.set_yticklabels(ylabels)
        cax.yaxis.tick_right()  # Enable y-ticks on the right side
        cax.yaxis.set_label_position("right")  # Set the y-label on the right side
        title = f"2D colormap - {additional_title}"
        cax.set_title(title, fontsize=10)

    def plot_multiple_embeddings(
        self,
        embeddings: dict,
        labels: dict,
        min_val=None,
        max_val=None,
        ticks=None,
        title="Embeddings",
        cmap="rainbow",
        legend_cmap=None,
        projection="3d",
        show_hulls=False,
        figsize=(5, 4),
        plot_legend=True,
        markersize=None,
        alpha=None,
        max_plot_per_row=4,
        dpi=300,
        as_pdf=False,
    ):
        figsize = figsize or (5, 4)
        preset_figsize_x, preset_figsize_y = figsize
        # Compute the number of subplots
        num_subplots = len(embeddings)
        rows = 1
        cols = num_subplots
        if num_subplots > max_plot_per_row:
            rows = int(num_subplots**0.5)
            cols = (num_subplots + rows - 1) // rows
            figsize_x = preset_figsize_x * max_plot_per_row
        else:
            figsize_x = preset_figsize_x * num_subplots
        figsize = (figsize_x, preset_figsize_y * rows)

        fig = plt.figure(figsize=figsize)
        subplot_kw_dict = {"projection": projection} if projection != "2d" else {}
        cols = max(1, cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, subplot_kw=subplot_kw_dict
        )

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        labels_list = labels["labels"]
        labels_list = (
            [labels_list] if not isinstance(labels_list, list) else labels_list
        )
        if len(labels_list) == 1 and len(labels_list) != num_subplots:
            global_logger.warning(
                f"""Only one label was found in the label list, but multiple subplots are detected ({num_subplots}). Using the same labels for all embeddings."""
            )
            labels_list = [labels_list[0]] * num_subplots

        # Plot each embedding
        for i, (subplot_title, embedding) in enumerate(embeddings.items()):
            # Get the labels for the current subplot

            session_labels = labels_list[i]
            session_labels, min_vals, max_vals = Vizualizer.create_RGBA_colors_from_2d(
                session_labels, min_val=min_val, max_val=max_val
            )

            session_labels_dict = {"name": labels["name"], "labels": session_labels}

            # plot the embedding
            ax = axes[i]
            ax = self.plot_embedding(
                ax=ax,
                min_val=min_val,
                max_val=max_val,
                embedding=embedding,
                projection=projection,
                embedding_labels=session_labels_dict,
                title=subplot_title,
                show_hulls=show_hulls,
                cmap=cmap,
                plot_legend=False,
                markersize=markersize,
                alpha=alpha,
                dpi=dpi,
            )

        if plot_legend and len(embeddings) > 0:
            if labels_list[0].shape[1] == 1:
                Vizualizer.add_1d_colormap_legend(
                    labels=labels_list[0],
                    ax=ax,
                    label_name=session_labels_dict["name"],
                    ticks=ticks,
                    cmap=cmap,
                )
            else:
                unique_rgba_colors = np.unique(session_labels_dict["labels"], axis=0)
                discrete_n_colors = (
                    int(np.ceil(np.sqrt(len(unique_rgba_colors))))
                    if legend_cmap is None
                    else None
                )

                first_embedding = list(embeddings.values())[0]
                if first_embedding.shape[1] == 2:
                    min_vals = min_val or np.min(first_embedding, axis=0)
                    max_vals = max_val or np.max(first_embedding, axis=0)
                    xticks_2d_colormap = np.linspace(min_vals[0], max_vals[0], 5)
                    yticks_2d_colormap = np.linspace(min_vals[1], max_vals[1], 5)
                    raise ValueError("This coloring is probably shit but lets see")
                else:
                    xticks_2d_colormap = None
                    yticks_2d_colormap = None

                Vizualizer.add_2d_colormap_legend(
                    fig,
                    Legend=legend_cmap,
                    discret_n_colors=discrete_n_colors,
                    xticks=xticks_2d_colormap,
                    yticks=yticks_2d_colormap,
                    additional_title=labels["name"],
                )

        for ax in axes[num_subplots:]:
            ax.remove()  # Remove any excess subplot axes
        fig.tight_layout()
        self.plot_ending(
            title,
            title_size=preset_figsize_x * cols,
            save=True,
            save_dir=self.save_dir,
            as_pdf=as_pdf,
        )

    def create_rgba_labels(values, alpha=0.8):
        """
        Create RGBA labels using a colormap.

        Parameters:
        values : array-like
            The input values to be mapped to colors. Can be 1D or 2D.
        alpha : float, optional
            The alpha level for the colors. Default is 0.8.

        Returns:
        np.ndarray
            An array of RGBA colors.

        Description:
        This function maps input values to RGBA colors using a specified colormap.
        It supports 1D and 2D input values. For 1D values, each value is mapped to
        a color. For 2D values, each pair of values is mapped to a color.
        """

        normalized_values = normalize_01(values, axis=0)
        values = np.array(values)
        if values.ndim == 1:
            cmap = plt.cm.get_cmap("rainbow")
            rgba_colors = np.array([cmap(x) for x in normalized_values])
        elif values.ndim == 2:
            cmap = lambda x, y: (x, 0.5, y, alpha)
            # Create a 2D array of RGBA values
            rgba_colors = np.array([cmap(x, y) for x, y in normalized_values])
        elif values.ndim == 3:
            raise ValueError("3D values not supported yet.")
        return rgba_colors

    @staticmethod
    def create_RGBA_colors_from_2d(
        values,
        min_val=None,
        max_val=None,
        alpha=0.8,
    ):
        # create 2D RGBA labels to overwrite 1D cmap coloring
        rgba_colors = None
        min_vals = None
        max_vals = None
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)  # Flatten to 1D if it's a single column
        if values.ndim == 1:
            min_vals = min_val if min_val is not None else np.min(values)
            max_vals = max_val if min_val is not None else np.max(values)
            rgba_colors = Vizualizer.create_rgba_labels(values, alpha=alpha)
            global_logger.debug(f"Created 1D RGBA labels for {values.shape[0]} values.")
        elif values.shape[1] == 2:
            min_vals = min_val if min_val is not None else np.min(values, axis=0)
            max_vals = max_val if min_val is not None else np.max(values, axis=0)
            rgba_colors = Vizualizer.create_rgba_labels(values, alpha=alpha)
            global_logger.debug(f"Created 2D RGBA labels for {values.shape[0]} values.")
        elif is_rgba(values):
            rgba_colors = values
        else:
            raise ValueError(f"Invalid labels shape: {values.shape}")
        return rgba_colors, min_vals, max_vals

    @staticmethod
    def auto_fontsize(fig, base_fontsize=8):
        # Use width or height as the base for scaling
        width, height = fig.get_size_inches()
        scale = (width + height) / 2
        return base_fontsize * scale / 6  # tweak denominator to your liking

    def plot_consistency_scores(self, ax1, title, embeddings, labels, dataset_ids):
        (
            time_scores,
            time_pairs,
            time_subjects,
        ) = cebra.sklearn.metrics.consistency_score(
            embeddings=embeddings,
            labels=labels,
            dataset_ids=dataset_ids,
            between="datasets",
        )
        ax1 = cebra.plot_consistency(
            time_scores,
            pairs=time_pairs,
            datasets=time_subjects,
            ax=ax1,
            title=title,
            colorbar_label="consistency score",
        )
        return ax1

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: change this to a more modular funciton, integrate into classes
        # labels to align the subjects is the position of the mouse in the arena
        # labels = {}  # {wanted_embedding 1: {animal_session_task_id: embedding}, ...}
        # for wanted_embedding in wanted_embeddings:
        #    labels[wanted_embedding] = {"embeddings": {}, "labels": {}}
        #    for wanted_stimulus_type in wanted_stimulus_types:
        #        for animal, session, task in yield_animal_session_task(animals):
        #            if task.behavior_metadata["stimulus_type"] == wanted_stimulus_type:
        #                wanted_embeddings_dict = filter_dict_by_properties(
        #                    task.embeddings,
        #                    include_properties=wanted_embedding,
        #                    exclude_properties=exclude_properties,
        #                )
        #                for embedding_key, embedding in wanted_embeddings_dict.items():
        #                    labels_id = f"{session.date[-3:]}_{task.task} {wanted_stimulus_type}"
        #                    position_lables = task.behavior.position.data
        #                    position_lables, embedding = force_equal_dimensions(
        #                        position_lables, embedding
        #                    )
        #                    labels[wanted_embedding]["embeddings"][
        #                        labels_id
        #                    ] = embedding
        #                    labels[wanted_embedding]["labels"][
        #                        labels_id
        #                    ] = position_lables
        #
        #    dataset_ids = list(labels[wanted_embedding]["embeddings"].keys())
        #    embeddings = list(labels[wanted_embedding]["embeddings"].values())
        #    labeling = list(labels[wanted_embedding]["labels"].values())
        #
        #    title = f"CEBRA-{wanted_embedding} embedding consistency"
        #    fig = plt.figure(figsize=figsize)
        #    ax1 = plt.subplot(111)
        #    ax1 = self.plot_consistency_score(
        #        ax1, title, embeddings, labeling, dataset_ids
        #    )
        #    plt.show()
        pass
        # self.plot_ending(title)

    def plot_decoding_score(
        self,
        decoded_model_lists,
        labels,
        title="Behavioral Decoding of Position",
        colors=["deepskyblue", "gray"],
        figsize=(13, 5),
    ):
        # TODO: improve this function, modularity, flexibility
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        overall_num = 0
        for color, docoded_model_list in zip(colors, decoded_model_lists):
            for num, decoded_model in enumerate(docoded_model_list):
                alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
                x_pos = overall_num + num
                width = 0.4  # Width of the bars
                ax1.bar(
                    x_pos, decoded_model.decoded[1], width=0.4, color=color, alpha=alpha
                )
                label = "".join(decoded_model.name.split("_train")).split("behavior_")[
                    -1
                ]
                ax2.scatter(
                    decoded_model.state_dict_["loss"][-1],
                    decoded_model.decoded[1],
                    s=50,
                    c=color,
                    alpha=alpha,
                    label=label,
                )
            overall_num += x_pos + 1

        x_label = "InfoNCE Loss (contrastive learning)"
        ylabel = "Median position error in cm"

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel(ylabel)
        labels = labels
        label_pos = np.arange(len(labels))
        ax1.set_xticks(label_pos)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.show()

    def plot_histogram(self, data, title, bins=100, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.hist(data, bins=bins)
        plt.show()
        plt.close()

    def plot_corr_hist_heat_salience(
        self,
        correlation: np.ndarray,
        saliences,
        title: str,
        bins: int = 100,
        sort=False,
        figsize=(17, 5),
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        self.histogam_subplot(
            correlation,
            "Correlation",
            ax1,
            bins=bins,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
        )
        Vizualizer.plot_heatmap(correlation, "Correlation Heatmap", ax2, sort=sort)
        self.histogam_subplot(
            saliences,
            "Saliences",
            ax3,
            xlim=[0, 2],
            bins=bins,
            xlabel="n",
            ylabel="Frequency",
        )
        self.plot_ending(title, save_dir=self.save_dir, save=True)

    def plot_dist_sal_dims(
        self, distances, saliences, normalized_saliences, title, bins=100
    ):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(17, 10))
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        title = title + " Histograms"

        self.histogam_subplot(
            distances,
            "Distance from Origin",
            ax1,
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            saliences,
            "Normalized Distances",
            ax2,
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 0],
            "normalized X",
            ax3,
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 1],
            "normalized Y",
            ax4,
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 2], "normalized Z", ax5, bins=bins, color=colors[4]
        )
        self.plot_ending(title, save_dir=self.save_dir, save=True)

    def plot_dist_sal_dims_2(
        self,
        distances,
        saliences,
        normalized_saliences,
        distances2,
        saliences2,
        normalized_saliences2,
        title,
        bins=100,
    ):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2, figsize=(17, 10))
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        title = title + " Histograms"

        self.histogam_subplot(
            distances,
            "Distance from Origin",
            ax1[0],
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
        )
        self.histogam_subplot(
            saliences,
            "Normalized Distances",
            ax2[0],
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 0],
            "normalized X",
            ax3[0],
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 1],
            "normalized Y",
            ax4[0],
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 2],
            "normalized Z",
            ax5[0],
            bins=bins,
            color=colors[4],
        )

        self.histogam_subplot(
            distances2,
            "Distance from Origin",
            ax1[1],
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
        )
        self.histogam_subplot(
            saliences2,
            "Normalized Distances",
            ax2[1],
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 0],
            "normalized X",
            ax3[1],
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 1],
            "normalized Y",
            ax4[1],
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 2],
            "normalized Z",
            ax5[1],
            bins=bins,
            color=colors[4],
        )
        self.plot_ending(title, save_dir=self.save_dir, save=True)

    def plot_corr_heat_corr_heat(
        self, correlation1, correlation2, title1, title2, sort=False, figsize=(17, 5)
    ):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
        title = title1 + " vs " + title2
        self.histogam_subplot(
            correlation1,
            title1 + " Correlation",
            ax1,
            bins=100,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
            color="tab:blue",
        )
        Vizualizer.plot_heatmap(correlation1, title1, ax2, sort=sort)
        self.histogam_subplot(
            correlation2,
            title2 + " Correlation",
            ax3,
            bins=100,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
            color="tab:orange",
        )
        Vizualizer.plot_heatmap(correlation2, title2, ax4, sort=sort)
        self.plot_ending(title, save_dir=self.save_dir, save=True)

    @staticmethod
    def plot_ending(title, save_dir, title_size=20, save=True, as_pdf=False, show=True):
        plt.suptitle(title, fontsize=title_size)
        plt.tight_layout()  # Ensure subplots fit within figure area

        if save:
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")

        if show:
            plt.show()
        plt.close()

    #################################################################
    ##### statistics of decoding (accuracy, precision, recall, f1-score)
    @staticmethod
    def plot_decoding_statistics(
        decoder_results: List[float],
        additional_title: str = "",
    ):
        decoded_test_datasets_reverse = None
        decoded_lists = [[decoder_results]]
        labels = [["All Cells"]]
        labels_flattened = ["All Cells"]
        for (
            scoring_type_reverse,
            decoded_test_sets,
        ) in decoded_test_datasets_reverse.items():
            decoded_lists.append([])
            labels.append([])
            for percentage, decoded_test_set in decoded_test_sets.items():
                decoded_lists[-1].append(decoded_test_set)
                label = f"{scoring_type_reverse} - {percentage}% cells"
                labels[-1].append(label)
                labels_flattened.append(label)
        print(labels)

        # viz = Vizualizer(root_dir=root_dir)
        # TODO: Is this working????????

        # viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
        title = f"{additional_title} lowest % cells fro Behavioral Decoding of stimulus"
        fig = plt.figure(figsize=(13, 5))
        fig.suptitle(title, fontsize=16)
        colors = ["green", "red", "deepskyblue"]
        ax1 = plt.subplot(111)
        # ax2 = plt.subplot(211)

        overall_num = 0
        for color, docoded_model_list, labels_list in zip(
            colors, decoded_lists, labels
        ):
            for num, (decoded, label) in enumerate(
                zip(docoded_model_list, labels_list)
            ):
                # color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
                alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
                x_pos = overall_num + num
                width = 0.4  # Width of the bars
                ax1.bar(
                    # x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
                    x_pos,
                    decoded[1],
                    width=0.4,
                    color=color,
                    alpha=alpha,
                    label=label,
                )
                # ax2.bar(
                # x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
                # x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
                # )
                ##ax2.scatter(
                #    middle_A_model.state_dict_["loss"][-1],
                #    decoded[1],
                #    s=50,
                #    c=color,
                #    alpha=alpha,
                #    label=label,
                # )
            overall_num = x_pos + 1

        ylabel = "Mean stimulus error"

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel(ylabel)
        ax1.grid(axis="y", alpha=0.2)
        print_labels = labels_flattened
        label_pos = np.arange(len(labels_flattened))
        ax1.set_xticks(label_pos)
        # ax1.set_ylim([0, 1])
        ax1.set_xticklabels(print_labels, rotation=45, ha="right")

        ylabel = "mean stimulus in cm"

        # ax2.spines["top"].set_visible(False)
        # ax2.spines["right"].set_visible(False)
        # ax2.set_ylabel(ylabel)
        # ax2.grid(axis="y", alpha=0.5)
        # print_labels = labels_flattened
        # label_pos = np.arange(len(labels_flattened))
        # ax2.set_xticks(label_pos)
        ##ax2.set_ylim([0, 130])
        # ax2.set_xticklabels(print_labels, rotation=45, ha="right")

        # plt.legend()
        plt.show()

    @staticmethod
    def plot_decoding_statistics_line(
        models,
        by="task",
        additional_title: Optional[str] = None,
        xlim: Optional[Tuple[int, int]] = None,
        labels: Optional[List[str]] = None,
        cmap: str = "tab20",
        figsize: Tuple[int, int] = None,
        show_variance: bool = True,
        save_dir: str = None,
        as_pdf: bool = False,
        markersize: float = 0.05,
        alpha: float = 0.4,
        dpi: int = 300,
    ):
        """
        decoding results by trainings iterations (left to right), 1 line for every task
        Assumtions:
            - models is a dictionary of dictionaries of dictionaries
                - models[session_date][task_name][model_name] = model
                - model has decoding_statistics attribute
            - models are sorted by time
        """
        continuouse_stats = ["mse", "rmse", "r2"]
        discrete_stats = ["accuracy", "precision", "recall", "f1-score", "roc_auc"]

        summary_decodings_by_iterations = {}
        stat_name = None
        for session_date, session_dict in models.items():
            for task_name, models_dict in session_dict.items():
                # Assuming task names are unique
                init_dict_in_dict(summary_decodings_by_iterations, task_name)
                for model_name, model in models_dict.items():
                    # Assume modles are sorted by time
                    # extract iterations from model name
                    iterations = None
                    for name_part in model_name.split("_"):
                        if "iter-" in name_part:
                            iterations = int(name_part.replace("iter-", ""))
                    if iterations is None:
                        print(
                            f"No iterations information was found in model name: {model_name}. Skipping..."
                        )
                        continue
                    else:
                        current_dict = init_dict_in_dict(
                            summary_decodings_by_iterations[task_name], iterations
                        )

                    # create a list of decodings stats
                    for stat_name, stat in model.decoding_statistics.items():
                        init_dict_in_dict(current_dict, stat_name)
                        current_dict[stat_name] = stat
        if stat_name is None:
            raise ValueError("No decoding statistics found. In models.")
        elif stat_name in discrete_stats:
            performance_measure_type = "discrete"
        elif stat_name in continuouse_stats:
            performance_measure_type = "continuouse"
        else:
            raise ValueError("Invalid performance measure type.")

        if by == "bar":
            summary_decodings_by_task_array = {}
            task_names = list(summary_decodings_by_iterations.keys())
            for task_num, (task_name, iteration_datas) in enumerate(
                summary_decodings_by_iterations.items()
            ):
                sorted_iteraion_datas = sort_dict(iteration_datas)
                for iteration, stat_datas in sorted_iteraion_datas.items():
                    current_dict = init_dict_in_dict(
                        summary_decodings_by_task_array, iteration
                    )
                    for stat_name, stat in stat_datas.items():
                        if stat_name in discrete_stats:
                            add_to_list_in_dict(current_dict, stat_name, stat)
                        elif stat_name in continuouse_stats:
                            init_dict_in_dict(current_dict, stat_name)
                            # create mean and variance list
                            for moment, value in stat.items():
                                add_to_list_in_dict(
                                    current_dict[stat_name], moment, value
                                )

            if performance_measure_type == "discrete":
                Vizualizer.plot_discrete_decoding_statistics_bar(
                    summary_decodings_by_task_array, xticks=task_names
                )
            elif performance_measure_type == "continuouse":
                Vizualizer.plot_continuous_decoding_statistics_bar(
                    summary_decodings_by_task_array,
                    xticks=task_names,
                    additional_title=additional_title,
                )

        elif by == "iterations":
            summary_decodings_by_iterations_array = {}
            task_names = list(summary_decodings_by_iterations.keys())
            iteration_values = []
            for task_num, (task_name, iteration_datas) in enumerate(
                summary_decodings_by_iterations.items()
            ):
                sorted_iteraion_datas = sort_dict(iteration_datas)
                iteration_values.append(list(sorted_iteraion_datas.keys()))
                current_dict = init_dict_in_dict(
                    summary_decodings_by_iterations_array, task_name
                )
                for iteration, stat_datas in sorted_iteraion_datas.items():
                    for stat_name, stat in stat_datas.items():
                        if stat_name in discrete_stats:
                            add_to_list_in_dict(current_dict, stat_name, stat)
                        elif stat_name in continuouse_stats:
                            init_dict_in_dict(current_dict, stat_name)
                            # create mean and variance list
                            for moment, value in stat.items():
                                add_to_list_in_dict(
                                    current_dict[stat_name], moment, value
                                )

            if performance_measure_type == "discrete":
                Vizualizer.plot_discrete_decoding_statistics_by_training_iterations(
                    summary_decodings_by_iterations_array,
                    iterations=iteration_values,
                    cmap=cmap,
                    labels=labels,
                    xlim=xlim,
                    additional_title=additional_title,
                    figsize=figsize,
                    save_dir=save_dir,
                    as_pdf=as_pdf,
                )
            elif performance_measure_type == "continuouse":
                Vizualizer.plot_continuous_decoding_statistics_by_training_iterations(
                    summary_decodings_by_iterations_array,
                    cmap=cmap,
                    iterations=iteration_values,
                    additional_title=additional_title,
                    show_variance=show_variance,
                    xlim=xlim,
                    figsize=figsize,
                    save_dir=save_dir,
                    as_pdf=as_pdf,
                )

    @staticmethod
    def plot_discrete_decoding_statistics_bar(
        decodings, xticks=None, min_max_roc_auc=(0, 1)
    ):
        raise NotImplementedError(
            f"This function can be used, but is not suiteable for the data structure used in this project. Only a single line is used for each task."
        )
        # TODO: change this funciton to plotting bars based on model??????

    @staticmethod
    def plot_continuous_decoding_statistics_bar(
        decodings,
        xticks=None,
        # min_max_r2=(-1, 1),
        # min_max_rmse=(0, 1),
        additional_title="",
    ):
        print(
            f"WARNING: This function can be used, but is not suiteable for the data structure used in this project. Only a single line is used for each task."
        )
        # TODO: change this funciton to plotting bars based on model?????? may use function plot_decoding_statistics for this?
        # Plot decodings
        c_dec = len(decodings)
        # discrete colormap (nipy_spectral) and discrete
        colormap = cm.get_cmap("tab10")
        fig, ax = plt.subplots(len(decodings), 2, figsize=(15, 3 * c_dec))
        fig.suptitle(
            f"Decoding statistics for different tasks {additional_title}", fontsize=20
        )
        for dec_num, (iter, task_data) in enumerate(decodings.items()):
            for i, (eval_name, eval_stat) in enumerate(task_data.items()):
                if "values" in eval_stat.keys():
                    values = eval_stat["values"]
                else:
                    values = eval_stat["mean"]

                var = None
                if "var" in eval_stat.keys():
                    var = eval_stat["var"]
                elif "variance" in eval_stat.keys():
                    var = eval_stat["variance"]

                ax[dec_num, i].plot(
                    xticks,
                    values,
                    # label=f"{animal_id}",
                    # color=colormap(animal_num),
                )
                ax[dec_num, i].set_title(
                    f"{eval_name} score for tasks with {iter} iterations"
                )
                ax[dec_num, i].set_ylabel(eval_name)
                min_y = -1 if eval_name == "r2" else 0
                max_y = 1 if eval_name == "r2" else None
                ax[dec_num, i].set_ylim(min_y, max_y)
                # set xticks
                xtick_pos = np.arange(len(xticks))
                ax[dec_num, i].legend()
                if dec_num == len(decodings) - 1:
                    ax[dec_num, i].set_xlabel("Task")
                    ax[dec_num, i].set_xticks(xtick_pos, xticks)
                else:
                    ax[dec_num, i].set_xticks(xtick_pos, [])

                if var is not None:
                    ax[dec_num, i].errorbar(
                        xticks,
                        values,
                        # yerr=animal_eval_stat_var,
                        # color=colormap(animal_num),
                        alpha=0.5,
                        fmt="o",
                        capsize=5,
                    )
                    # label=f"{task_name}")#, capsize=5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_discrete_decoding_statistics_by_training_iterations(
        decodings,
        iterations=None,
        additional_title="",
        labels="",
        xlim=None,
        cmap="tab10",
        figsize=(15, 8),
        to_plot=[
            "accuracy",
            "f1-score",
            "roc_auc",
        ],  # ["accuracy", "precision", "recall", "f1-score", "roc_auc"],
        save_dir=None,
        as_pdf=False,
    ):
        figsize = figsize or (15, 8)
        # plot accuracy, precision, recall, f1-score
        colormap = cm.get_cmap(cmap)
        max_iter_count = 0
        for task_iterations in iterations:
            num_iterations = len(task_iterations)
            if num_iterations > max_iter_count:
                max_iter_count = num_iterations
        num_tasks = len(decodings)

        fig, axes = plt.subplots(1, 1, figsize=figsize)
        # create list without roc_auc
        decoding_measures = [
            decoding_measure
            for decoding_measure in to_plot
            if decoding_measure != "roc_auc"
        ]
        num_measures = len(decoding_measures)
        for measure_num, decoding_measure in enumerate(decoding_measures):
            for task_num, (task_name, decoding_data) in enumerate(decodings.items()):
                if len(decoding_data) == 0:
                    continue
                colors = Vizualizer.generate_similar_colors(
                    colormap(task_num), num_measures
                )
                eval_stat = decoding_data[decoding_measure]

                axes.plot(
                    iterations[task_num],
                    eval_stat,
                    label=f"{task_name} {decoding_measure}",
                    color=colors[measure_num],
                )

                axes.set_title(f"Performance Measures {additional_title}")
                axes.set_ylabel("")
                axes.legend()
                axes.set_xlabel("Iterations")
                axes.set_xticks(
                    iterations[task_num],
                    iterations[task_num],
                    rotation=45,
                    fontsize=8,
                )
                if xlim:
                    axes.set_xlim(xlim)
        plt.tight_layout()
        title = f"Performance Measures {additional_title}"
        Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
        plt.show()

        if "roc_auc" in to_plot:
            suptitle = f"ROC curves for spatial zones for different tasks with different iterations {additional_title}"
            fig, axes = plt.subplots(
                num_tasks,
                max_iter_count,
                figsize=(5 * max_iter_count, 5 * num_tasks),
            )
            fig.suptitle(
                suptitle,
                fontsize=40,
            )
            for task_num, (task_name, decoding_data) in enumerate(decodings.items()):
                if len(decoding_data) == 0:
                    continue
                eval_stat = decoding_data["roc_auc"]
                for iter_eval_num, roc_auc_dict in enumerate(eval_stat):
                    max_iter_value = iterations[task_num][iter_eval_num]
                    max_iter_pos = max_iter_count - iter_eval_num - 1
                    ax = axes[task_num, max_iter_pos]

                    for class_num, values in roc_auc_dict.items():
                        fpr, tpr, auc = values.values()
                        ax.plot(
                            fpr,
                            tpr,
                            # label=f"{labels[class_num].capitalize()} AUC {auc:.2f}" if labels else "",
                        )
                        ax.set_title(
                            f"{task_name}: iter. {max_iter_value}", fontsize=20
                        )

                    if iter_eval_num == 0:
                        ax.set_ylabel("TPR")
                    if task_num == num_tasks - 1:
                        ax.set_xlabel("FPR")
                    ax.legend(loc="lower right")

            plt.tight_layout()
            title = f"ROC curves {additional_title}"
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
            plt.show()

    @staticmethod
    def plot_continuous_decoding_statistics_by_training_iterations(
        decodings,
        iterations=None,
        additional_title="",
        cmap="tab10",
        figsize=(15, 8),
        show_variance=True,
        xlim=None,
        save_dir=None,
        as_pdf=False,
    ):
        figsize = figsize or (15, 8)
        # Create a color map for tasks
        colormap = cm.get_cmap(cmap)
        iterations = np.array(iterations)
        if iterations.ndim < 2:
            iterations = np.array([iterations] * len(decodings))
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        for task_num, (task_name, decoding_data) in enumerate(decodings.items()):
            for i, (eval_name, eval_stat) in enumerate(decoding_data.items()):
                if "values" in eval_stat.keys():
                    values = eval_stat["values"]
                else:
                    values = eval_stat["mean"]

                var = None
                if show_variance:
                    if "var" in eval_stat.keys():
                        var = eval_stat["var"]
                    elif "variance" in eval_stat.keys():
                        var = eval_stat["variance"]

                # Vizualizer.plot_line(ax=axes[i],
                #                     values=values,
                #                     label=task_name,
                #                     xlabel="Iterations",
                #                     ylabel=eval_name,
                #                     var=var)

                axes[i].plot(
                    iterations[i],
                    values,
                    label=f"{task_name}",
                    color=colormap(task_num),
                )
                if var:
                    axes[i].errorbar(
                        iterations[i],
                        values,
                        yerr=var,
                        capsize=5,
                        fmt=".",
                        color=colormap(task_num),
                        alpha=0.4,
                    )

                axes[i].set_title(
                    f"{eval_name.upper()} score for different tasks with different iterations {additional_title}"
                )
                axes[i].set_ylabel(eval_name)
                axes[i].legend()
                axes[i].set_xlabel("Iterations")
                # axes[i].set_xticks(
                #    iterations[i], iterations[i], rotation=45, fontsize=8
                # )
                if xlim:
                    axes[i].set_xlim(xlim)
        title = f"Decoding statistics for different tasks {additional_title}"
        plt.tight_layout()
        Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
        plt.show()

    @staticmethod
    def plot_cell_activites_heatmap(
        map: np.ndarray,
        additional_title=None,
        norm_rate=False,
        sorting_indices=None,
        cmap="viridis",
        cbar_title="",
    ):
        if map.ndim == 2:
            title = "Cell Activities"
            if additional_title:
                title = title + f" {additional_title}"
            # ordered activities by peak
            if not norm_rate:
                title += " not normalized"
            if sorting_indices is not None:
                title += " order provided"

            # normalize by cell firering rate
            plot_map = map if not norm_rate else normalize_01(map, axis=1)

            sorted_map, indices = sort_arr_by(
                plot_map, axis=1, sorting_indices=sorting_indices
            )

            plt.figure()
            plt.imshow(sorted_map, cmap=cmap, aspect="auto")  # , interpolation="None")

            plt.ylabel("Cell ID")
            # remove yticks
            plt.xlabel("location (cm)")
            plt.title(title)
            plt.show(block=False)
            plt.close()
            return sorted_map
        elif map.ndim == 3:
            """
            Plots 4 random cells from the spike map in a 2x2 grid of heatmaps.
            """
            num_cells = map.shape[0]
            num_cells_to_plot = min(4, num_cells)
            #
            nrows = num_cells_to_plot // 2
            ncols = num_cells_to_plot // nrows
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
            title = f"Map for {num_cells_to_plot} Random Cells"
            title += f" {additional_title}" if additional_title else ""
            fig.suptitle(title, fontsize=16)
            # draw number without overlapping
            random_cell_idices = np.random.choice(
                num_cells, num_cells_to_plot, replace=False
            )
            for i, random_cell_idx in enumerate(random_cell_idices):
                ax = axes[i // 2, i % 2]
                heatmap(
                    map[random_cell_idx],
                    ax=ax,
                    cmap=cmap,
                    cbar=True,
                    cbar_kws={"label": cbar_title},
                )
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
                ax.set_xlabel("Position Bin X")
                ax.set_ylabel("Position Bin Y")
                ax.set_title(f"Cell {random_cell_idx}")
            plt.tight_layout()
            plt.show()
        else:
            do_critical(
                "map must be a 2D or 3D array with shape (num_cells, num_bins) or (num_cells, num_bins, num_bins)"
            )
        return map

    @staticmethod
    def create_histogram(
        data,
        title,
        xlabel,
        ylabel,
        data_labels=None,
        bins=100,
        red_line_pos=None,
        color=None,
        interactive=False,
        stacked=True,
    ):
        if interactive:
            fig = plotly.express.histogram(
                data,
                title=title,
                labels={"value": xlabel, "count": ylabel},
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False)
            fig.show()
        else:
            plt.figure(figsize=(10, 3))
            plt.hist(data, bins=bins, label=data_labels, color=color, stacked=True)
            plt.legend()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if red_line_pos is not None:
                plt.axvline(red_line_pos, color="r", linestyle="dashed", linewidth=1)
            plt.show(block=False)
            plt.close()

    @staticmethod
    def plot_zscore(
        zscore,
        additional_title=None,
        data_labels=None,
        color=None,
        interactive=False,
        zscore_threshold=2.5,
        stacked=True,
    ):
        zscore = np.array(zscore)
        title = "Zscore distribution of Spatial Information"
        xlabel = "Zscore"
        ylable = "# of cells"
        if additional_title is not None:
            title = title + f" ({additional_title})"
        zscores = split_array_by_zscore(zscore, zscore, threshold=2.5)
        if data_labels is None:
            percentage = (len(zscores[0]) / (len(zscores[0]) + len(zscores[1]))) * 100
            data_labels = [
                f"{percentage:.0f}% ({len(zscores[0])}) cells > {zscore_threshold}",
                f" {len(zscores[1])} cells < {zscore_threshold}",
            ]
        Vizualizer.create_histogram(
            zscores,
            title,
            xlabel,
            ylable,
            bins=100,
            data_labels=data_labels,
            red_line_pos=zscore_threshold,
            color=color,
            interactive=interactive,
            stacked=stacked,
        )

    @staticmethod
    def plot_si_rates(
        si_rate,
        data_labels=None,
        additional_title=None,
        zscores=None,
        zscore_threshold=2.5,
        color=None,
        interactive=False,
        stacked=True,
    ):
        si_rate = np.array(si_rate)
        title = "Spatial Information Rate"
        xlabel = "Spatial Information Rate [bits/sec]"
        ylabel = "# of cells"
        if additional_title is not None:
            title = title + f" ({additional_title})"
        data = (
            split_array_by_zscore(si_rate, zscores, threshold=zscore_threshold)
            if zscores is not None
            else si_rate
        )
        if data_labels is None and len(data) == 2:
            percentage = (len(data[0]) / (len(data[0]) + len(data[1]))) * 100
            data_labels = [
                f" {percentage:.0f}% ({len(data[0])}) <= 5% p",
                f"{len(data[1])} > 5% p",
            ]
        Vizualizer.create_histogram(
            data,
            title,
            xlabel,
            ylabel,
            bins=100,
            data_labels=data_labels,
            color=color,
            interactive=interactive,
            stacked=stacked,
        )

    @staticmethod
    def plot_sanky_example():
        #####################################################
        ################### SANKEY PLOTS ####################
        #####################################################
        #
        label_list = ["cat", "dog", "domesticated", "female", "male", "wild", "neither"]
        # cat: 0, dog: 1, domesticated: 2, female: 3, male: 4, wild: 5
        source = [0, 0, 1, 3, 4, 4, 1]
        target = [3, 4, 4, 2, 2, 5, 6]
        count = [5, 6, 22, 21, 6, 22, 5]

        fig = plotly.graph_object.Figure(
            data=[
                plotly.graph_object.Sankey(
                    node={"label": label_list},
                    link={"source": source, "target": target, "value": count},
                )
            ]
        )

        fig.show()

    @staticmethod
    def plot_multi_task_cell_activity_pos_by_time(
        task_cell_activity,
        figsize_x=20,
        norm=False,
        smooth=False,
        window_size=5,
        additional_title=None,
        savepath=None,
        lines_per_y=1,
        use_discrete_colors=False,
        cmap="inferno",
        show=True,
    ):
        """
        Plots the activity of cells across multiple tasks, with each task's activity plotted in separate subplots.
        Cell plots are normalized and smoothed. Top subplot shows the average activity across all laps, and the bottom subplot shows the activity of each lap.

        Parameters
        ----------
        task_cell_activity : dict
            Dictionary where keys are task identifiers and values are dictionaries with keys "lap_activity" and "additional_title".
            "lap_activity" is a numpy array of cell activities, and "additional_title" is a string to be added to the subplot title.
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        additional_title : str, optional
            Additional title to be added to the main title (default is None).
        savepath : str, optional
            Path to save the plot (default is None).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is False).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
        show : bool, optional
            Whether to show the plot (default is True).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        # create 2 subplots
        fig, axes = plt.subplots(
            2, len(task_cell_activity), gridspec_kw={"height_ratios": [1, 10]}
        )  # Set relative heights of subplots

        for task_num, (task, activity_and_title) in enumerate(
            task_cell_activity.items()
        ):
            traces = activity_and_title["lap_activity"]
            additional_cell_title = activity_and_title["additional_title"]

            sum_traces = np.array([np.sum(traces, axis=0)])
            if "label" in activity_and_title.keys():
                label = make_list_ifnot(activity_and_title["label"])
            else:
                label = None

            if axes.ndim == 1:
                axes = axes.reshape(1, -1)

            axes[0, task_num] = Vizualizer.traces_subplot(
                axes[0, task_num],
                sum_traces,
                labels=label,
                norm=norm,
                smooth=smooth,
                window_size=window_size,
                lines_per_y=1.1,
                xlabel="",
                yticks=None,
                additional_title=f"avg. {additional_cell_title}",
                ylabel="",
                figsize_x=figsize_x,
                use_discrete_colors=use_discrete_colors,
                cmap=cmap,
            )

            norm_traces = normalize_01(traces, axis=1) if norm else traces
            norm_traces = np.nan_to_num(norm_traces)

            axes[1, task_num] = Vizualizer.traces_subplot(
                axes[1, task_num],
                norm_traces,
                labels=None,
                norm=False,
                smooth=smooth,
                window_size=window_size,
                lines_per_y=lines_per_y,
                additional_title=additional_cell_title,
                ylabel="lap",
                figsize_x=figsize_x,
                use_discrete_colors=use_discrete_colors,
                cmap=cmap,
            )

        title = "Cell Activity"
        if additional_title:
            title += f" {additional_title}"

        fig.subplots_adjust(hspace=0.08, top=0.93)  # Decrease gap between subplots
        fig.suptitle(title, fontsize=17)

        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
            plt.close()

        return fig

    @staticmethod
    def plot_single_cell_activity(
        traces,
        figsize_x=20,
        labels=None,
        norm=False,
        smooth=False,
        window_size=5,
        additional_title=None,
        savepath=None,
        lines_per_y=1,
        use_discrete_colors=False,
        cmap="inferno",
        show=True,
    ):
        """
        Plots the activity of a single cell. Top subplot shows the average activity across all laps, and the bottom subplot shows the activity of each lap.

        Parameters
        ----------
        traces : np.ndarray
            Array of cell activity traces.
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        labels : list, optional
            Labels for the traces (default is None).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        additional_title : str, optional
            Additional title to be added to the main title (default is None).
        savepath : str, optional
            Path to save the plot (default is None).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is False).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
        show : bool, optional
            Whether to show the plot (default is True).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        # create 2 subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 10]}
        )  # Set relative heights of subplots
        fig.subplots_adjust(hspace=0.08)  # Decrease gap between subplots

        sum_traces = np.array([np.sum(traces, axis=0)])

        if labels:
            labels = make_list_ifnot(labels)

        ax1 = Vizualizer.traces_subplot(
            ax1,
            sum_traces,
            labels=labels,
            norm=norm,
            smooth=smooth,
            window_size=window_size,
            lines_per_y=1.1,
            xlabel="",
            yticks=None,
            additional_title=f"avg. {additional_title}",
            ylabel="",
            figsize_x=figsize_x,
            use_discrete_colors=use_discrete_colors,
            cmap=cmap,
        )

        norm_traces = normalize_01(traces, axis=1) if norm else traces
        norm_traces = np.nan_to_num(norm_traces)

        ax2 = Vizualizer.traces_subplot(
            ax2,
            norm_traces,
            labels=None,
            norm=False,
            smooth=smooth,
            window_size=window_size,
            lines_per_y=lines_per_y,
            additional_title=additional_title,
            ylabel="lap",
            figsize_x=figsize_x,
            use_discrete_colors=use_discrete_colors,
            cmap=cmap,
        )
        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
            plt.close()

        return fig

    @staticmethod
    def plot_traces_shifted(
        traces,
        figsize_x=20,
        labels=None,
        norm=False,
        smooth=False,
        window_size=5,
        additional_title=None,
        savepath=None,
        lines_per_y=1,
        use_discrete_colors=True,
        cmap="inferno",  # gray, magma, plasma, viridis
    ):
        """
        Plots traces shifted up by a fixed amount for each trace.

        Parameters
        ----------
        traces : np.ndarray
            Array of traces to plot.
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        labels : list, optional
            Labels for the traces (default is None).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        additional_title : str, optional
            Additional title to be added to the main title (default is None).
        savepath : str, optional
            Path to save the plot (default is None).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is True).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
            good colormaps for black background: gray, inferno, magma, plasma, viridis
            white colormaps for black background: binary, blues

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        fig, ax = plt.subplots()
        ax = Vizualizer.traces_subplot(
            ax,
            traces,
            labels=labels,
            norm=norm,
            smooth=smooth,
            window_size=window_size,
            lines_per_y=lines_per_y,
            additional_title=additional_title,
            figsize_x=figsize_x,
            use_discrete_colors=use_discrete_colors,
            cmap=cmap,
        )
        if savepath:
            plt.savefig(savepath)
        plt.show()
        plt.close()

        return ax

    # Subplots
    def traces_subplot(
        ax,
        traces,
        additional_title=None,
        color=None,
        labels=None,
        norm=False,
        smooth=False,
        window_size=5,
        lines_per_y=1,
        plot_legend=True,
        yticks="default",
        xlabel="Bins",
        ylabel="Cell",
        figsize_x=20,
        figsize_y=None,
        use_discrete_colors=True,
        cmap="inferno",
    ):
        """
        Plots traces on a given axis with options for normalization, smoothing, and color mapping.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        traces : np.ndarray
            Array of traces to plot.
        additional_title : str, optional
            Additional title to be added to the subplot title (default is None).
        color : str or None, optional
            Color for the traces (default is None).
        labels : list, optional
            Labels for the traces (default is None).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        plot_legend : bool, optional
            Whether to plot the legend (default is True).
        yticks : str or list, optional
            Y-axis tick labels (default is "default").
        xlabel : str, optional
            Label for the x-axis (default is "Bins").
        ylabel : str, optional
            Label for the y-axis (default is "Cell").
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        figsize_y : int or None, optional
            Height of the figure in inches (default is None).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is True).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
            good colormaps for black background: gray, inferno, magma, plasma, viridis
            white colormaps for black background: binary, blues

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plotted traces.
        """
        if smooth:
            traces = smooth_array(traces, window_size=window_size, axis=1)
        if norm:
            traces = normalize_01(traces, axis=1)
            traces = np.nan_to_num(traces)

        if labels is None:
            labels = [None] * len(traces)
            plot_legend = False
        else:
            labels = [
                f"{label:.3f}" if not isinstance(label, str) else label
                for label in labels
            ]

        shift_scale = 0.1 / lines_per_y
        linecolor = color or None

        min_value, max_value = np.min(traces) - np.max(traces), np.max(traces)
        for i, (trace, label) in enumerate(zip(traces, labels)):
            # min_value, max_value = np.min(trace) - np.max(trace) / 3, np.max(trace)
            upshift = i / lines_per_y + shift_scale * i
            shifted_trace = trace / lines_per_y + upshift

            if use_discrete_colors:
                ax.plot(shifted_trace, color=linecolor, label=label)
            else:
                # Create line segments
                points = np.array(
                    [np.arange(len(shifted_trace)), shifted_trace]
                ).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Calculate alpha for each segment
                alphas = Vizualizer.calculate_alpha(trace, min_value, max_value)

                # Create a LineCollection
                lc = LineCollection(segments, cmap=cmap, label=label)  # , alpha=alphas)
                lc.set_array(np.array(alphas))
                ax.add_collection(lc)

            ax.axhline(
                y=upshift, color="grey", linestyle="--", alpha=0.2
            )  # Adding grey dashed line

        if not figsize_y:
            figsize_y = int(len(traces) / lines_per_y)
            # y_size /= 2

        title = "Activity"
        if additional_title:
            title += f" {additional_title}"

        if yticks == "default":
            yticks = range(traces.shape[0])
            ytick_pos = [
                i / lines_per_y + shift_scale * i for i, tick in enumerate(yticks)
            ]
            ax.set_yticks(ytick_pos, yticks)
            ax.set_ylim(-shift_scale, np.max(ytick_pos) + 1 / lines_per_y)
        else:
            ax.set_ylim(-shift_scale, None)

        ax.set_title(title)
        ax.set_xlim(0, traces.shape[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.set_size_inches(figsize_x, figsize_y)  # Set figure size
        if plot_legend:
            # plt.legend(loc="upper right")
            ax.legend(bbox_to_anchor=(1, 1))
        return ax

    def histogam_subplot(
        self,
        data: np.ndarray,
        title: str,
        ax,
        bins=100,
        xlim=[0, 1],
        xlabel="",
        ylabel="Frequency",
        xticklabels=None,
        color=None,
    ):
        ax.set_title(title)
        ax.hist(data.flatten(), bins=bins, color=color)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticklabels == "empty":
            ax.set_xticklabels("")

    @staticmethod
    def plot_heatmap(
        data: np.ndarray,
        title: str = "",
        additional_title: str = "",
        fontsize: Optional[int] = None,
        custom_annotation: Optional[
            Union[np.ndarray, List[List[str]], pd.DataFrame]
        ] = None,
        custom_annotation_label: Optional[str] = None,
        figsize: tuple = (12, 10),
        xlabel: str = None,
        ylabel: str = None,
        no_diag: bool = False,
        sort_by: Optional[Literal["value", "similarity"]] = None,
        linkage_method: Literal[
            "single",
            "complete",
            "average",
            "ward",
            "centroid",
            "median",
            "weighted",
        ] = "ward",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ylim: Optional[Tuple[int, int]] = None,
        xlim: Optional[Tuple[int, int]] = None,
        xticks: Optional[List[str]] = None,
        yticks: Optional[List[str]] = None,
        xticks_pos: Optional[List[float]] = None,
        yticks_pos: Optional[List[float]] = None,
        annotation: bool = True,
        rotation: float = 45,
        colorbar: bool = True,
        colorbar_ticks: Optional[List[float]] = None,
        colorbar_ticks_labels: Optional[List[str]] = None,
        colorbar_label: Optional[str] = None,
        cmap: Union[str, mcolors.Colormap] = "viridis",
        save_dir: Optional[str] = None,
        as_pdf: bool = False,
        interpolation: str = "none",
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        return_figure: bool = False,
        add_line: Optional[List[Tuple[str, str]]] = None,
    ):
        """Creates and displays a heatmap with comprehensive customization options and hierarchical clustering.

        Args:
            data: The matrix data to be displayed as a heatmap.
            title: Main title for the plot.
            additional_title: Additional text to append to the title.
            fontsize: Font size for text elements.
            figsize: Figure size as (width, height) in inches.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            no_diag: If True, removes diagonal elements.
            sort_by: Sorting method ('value' or 'similarity' for hierarchical clustering).
            vmin: Minimum value for color scaling.
            vmax: Maximum value for color scaling.
            ylim: Limits for y-axis.
            xlim: Limits for x-axis.
            xticks: Custom tick labels for x-axis.
            yticks: Custom tick labels for y-axis.
            xticks_pos: Custom tick positions for x-axis.
            yticks_pos: Custom tick positions for y-axis.
            rotation: Rotation angle for x-tick labels.
            colorbar: Whether to display a colorbar.
            colorbar_label: Label for the colorbar.
            cmap: Colormap for the heatmap.
            save_dir: Directory to save the plot.
            as_pdf: Save as PDF if True, else PNG.
            interpolation: Interpolation method for imshow.
            show: Whether to display the plot.
            ax: Existing axes for the plot.
            fig: Existing figure for the plot.
            return_figure: If True, returns figure and axes along with heatmap.
            add_line: List of tuples (label1, label2) to draw thick lines between.

        Returns:
            Heatmap image or tuple of (heatmap image, figure, axes).

        Raises:
            ValueError: If data is not a valid 2D array.
        """
        data_copy = np.copy(data)
        if no_diag:
            np.fill_diagonal(data_copy, np.nan)
        if len(data_copy.shape) != 2:
            raise ValueError("Data must be a 2D array for heatmap plotting")

        # Initialize figure with space for dendrograms if similarity sorting is used
        if sort_by == "similarity" and ax is None:
            # Adjust figure size to accommodate top dendrogram only
            fig_width = figsize[0]
            fig_height = figsize[1] + 3  # Add space for top dendrogram
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Create axes for heatmap and top dendrogram
            ax_heatmap = fig.add_axes(
                [0.1, 0.1, 0.85, 0.75]
            )  # [left, bottom, width, height]
            ax_dendro_top = fig.add_axes([0.205, 0.85, 0.673, 0.15])
            # remove border from dendrogram
            ax_dendro_top.spines["top"].set_visible(False)
            ax_dendro_top.spines["right"].set_visible(False)
            ax_dendro_top.spines["left"].set_visible(False)
            ax_dendro_top.spines["bottom"].set_visible(False)
            created_fig = True
        else:
            if ax is None:
                fig, ax_heatmap = plt.subplots(1, 1, figsize=figsize)
                created_fig = True
            else:
                ax_heatmap = ax
                created_fig = False
                if fig is None and colorbar:
                    fig = ax_heatmap.figure
            ax_dendro_top = None

        sorted_indices = None
        if sort_by:
            if sort_by == "value":
                data_copy = np.sort(data_copy, axis=1)[:, ::-1]
                title += "|sorting: value"
            elif sort_by == "similarity":
                _, data_copy, sorted_indices, _, _, _, _ = hierarchical_clustering(
                    data=data_copy,
                    linkage_method=linkage_method,
                    ax=ax_dendro_top,
                )
                linkage_str = f"|sorting: {linkage_method} linkage"
                title += linkage_str if linkage_str not in title else ""
                xticks = (
                    np.array(xticks)[sorted_indices] if xticks is not None else None
                )
                yticks = (
                    np.array(yticks)[sorted_indices] if yticks is not None else None
                )

            else:
                raise ValueError(
                    f"Unknown sort_by value: {sort_by}. Only 'value' or 'similarity' are supported"
                )

        full_title = f"{title} {additional_title}" if additional_title else title
        fontsize = fontsize or (
            Vizualizer.auto_fontsize(fig)
            if fig and hasattr(Vizualizer, "auto_fontsize")
            else 12
        )

        if ax_dendro_top is not None:
            ax_dendro_top.set_title(full_title, fontsize=fontsize)
        else:
            ax_heatmap.set_title(full_title, fontsize=fontsize)
        ax_heatmap.set_xlabel(xlabel, fontsize=fontsize * 0.8)
        ax_heatmap.set_ylabel(ylabel, fontsize=fontsize * 0.8)

        if ylim is not None:
            ax_heatmap.set_ylim(ylim)
        if xlim is not None:
            ax_heatmap.set_xlim(xlim)

        if xticks is not None:
            if xticks_pos is None:
                xticks_pos = np.arange(len(xticks))
            ax_heatmap.set_xticks(xticks_pos)
            xticks = [
                str(x) if i == 0 or x != xticks[i - 1] else ""
                for i, x in enumerate(xticks)
            ]
            ax_heatmap.set_xticklabels(xticks, rotation=rotation, ha="right")

        if yticks is not None:
            if yticks_pos is None:
                yticks_pos = np.arange(len(yticks))
            ax_heatmap.set_yticks(yticks_pos)
            yticks = [
                str(y) if i == 0 or y != yticks[i - 1] else ""
                for i, y in enumerate(yticks)
            ]
            ax_heatmap.set_yticklabels(yticks)

        # decrease fontsize of ticks if amount of ticks is high
        if len(xticks) < 20:
            x_fontsize = fontsize * 0.8
        elif len(xticks) < 50:
            x_fontsize = fontsize * 0.5
        elif len(xticks) < 100:
            x_fontsize = fontsize * 0.3
        else:
            x_fontsize = fontsize * 0.1

        if len(yticks) < 20:
            y_fontsize = fontsize * 0.8
        elif len(xticks) < 50:
            y_fontsize = fontsize * 0.5
        elif len(xticks) < 100:
            y_fontsize = fontsize * 0.3
        else:
            y_fontsize = fontsize * 0.1

        ax_heatmap.tick_params(axis="x", which="major", labelsize=x_fontsize)
        ax_heatmap.tick_params(axis="y", which="major", labelsize=y_fontsize)

        # Set up the colormap
        cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cdark_gray = mcolors.to_rgba("dimgray", alpha=0.3)
        cmap.set_under(cdark_gray)
        cmap.set_over(cdark_gray)

        # Create the heatmap using imshow
        cax = ax_heatmap.imshow(
            data_copy,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )

        if annotation:
            base_fontsize = fontsize
            cell_width = ax_heatmap.get_window_extent().width / data_copy.shape[1]
            cell_height = ax_heatmap.get_window_extent().height / data_copy.shape[0]
            scaling_factor = min(cell_width, cell_height) / 50
            annot_fontsize = min(max(base_fontsize * scaling_factor, 6), 12)

            # Handle custom annotations
            if custom_annotation is None:
                annotations = data_copy
            else:
                if isinstance(custom_annotation, pd.DataFrame):
                    annotations = custom_annotation.to_numpy()
                elif isinstance(custom_annotation, list):
                    annotations = np.array(custom_annotation)
                if custom_annotation.shape != data_copy.shape:
                    raise ValueError(
                        "Custom annotation must have the same shape as data"
                    )
            nan_mask = np.isnan(data_copy)
            annotations = annotations * nan_mask.T

            # Normalize the data for colormap mapping
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            for i in range(data_copy.shape[0]):
                for j in range(data_copy.shape[1]):
                    annotation_value = annotations[i, j]
                    if isinstance(annotation_value, str):
                        annotation_str = annotation_value
                    elif is_int_like(annotation_value):
                        annotation_str = (
                            str(annotation_value) if annotation_value != 0 else ""
                        )
                    elif is_float_like(annotation_value):
                        annotation_str = (
                            f"{annotation_value:.2f}" if annotation_value != 0 else ""
                        )
                    else:
                        annotation_str = str(annotation_value)

                    if not np.isnan(annotation_value) and annotation_str:
                        # Get the cell's value from data_copy for color mapping
                        cell_value = data_copy[i, j]
                        if not np.isnan(cell_value):
                            # Map the cell value to a normalized range [0, 1]
                            normalized_value = norm(cell_value)

                            # Get the RGB color from the colormap
                            rgb_color = cmap(normalized_value)[
                                :3
                            ]  # Extract RGB, ignore alpha

                            # Calculate luminance (per ITU-R BT.601)
                            luminance = (
                                0.299 * rgb_color[0]
                                + 0.587 * rgb_color[1]
                                + 0.114 * rgb_color[2]
                            )

                            # Choose text color based on luminance
                            text_color = "black" if luminance > 0.5 else "white"
                        else:
                            # Fallback for NaN values (use default color)
                            text_color = "black"

                        if cell_width > 20 and cell_height > 20:
                            ax_heatmap.text(
                                j,
                                i,
                                annotation_str,
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=annot_fontsize,
                                rotation=45 if cell_width < cell_height else 0,
                            )

                # Custom block for legend explaining annotation
                if custom_annotation is not None:
                    if i == 0:
                        legend_elements = plt.Line2D(
                            [0],
                            [0],
                            marker="s",
                            label=custom_annotation_label,
                            markerfacecolor="gray",
                            markersize=10,
                        )
                        # Make legend outside of the heatmap
                        ax_heatmap.legend(
                            handles=[legend_elements],
                            loc="upper right",
                            bbox_to_anchor=(0, 1),
                            fontsize=fontsize * 0.8,
                        )

        if colorbar and fig:
            cbar = fig.colorbar(
                cax, ax=ax_heatmap, label=colorbar_label, fraction=0.046, pad=0.04
            )
            cbar.ax.yaxis.label.set_size(fontsize * 0.8)
            if colorbar_ticks is not None:
                cbar.set_ticks(colorbar_ticks)
            if colorbar_ticks_labels is not None:
                cbar.set_ticklabels(colorbar_ticks_labels, fontsize=fontsize * 0.8)
            cbar.ax.tick_params(labelsize=fontsize * 0.8)

        if save_dir and created_fig:
            Vizualizer.save_plot(save_dir, full_title, "pdf" if as_pdf else "png")

        # Add thick lines between labels if specified
        if add_line is not None and xticks is not None:
            for label1, label2 in add_line:
                if label1 in xticks and label2 in xticks:
                    idx1 = xticks.index(label1)
                    idx2 = xticks.index(label2)
                    pos = (idx1 + idx2) / 2
                    # Draw L-shaped line: vertical from intersection to bottom, horizontal from left to intersection
                    ax_heatmap.plot([pos, -0.5], [pos, pos], color="black", linewidth=2)
                    ax_heatmap.plot(
                        [pos, pos], [pos, pos * 2 + 0.5], color="black", linewidth=2
                    )
                else:
                    print(
                        f"Warning: One of the labels '{label1}' or '{label2}' not found in xticks. Not drawing line."
                    )

        if show and created_fig:
            plt.show()

        return (cax, fig, ax_heatmap) if return_figure else cax

    @staticmethod
    def plot_group_distr_similarities(
        similarities: dict,  # {metric: {group_name: np.array}}
        bins,
        skip=[],
        supxlabel="Bin X",
        supylabel="Bin Y",
        xticks=None,
        yticks=None,
        figsize=(4, 3),
        tick_steps=3,
        additional_title="",
        colorbar=False,
        colorbar_label="",
        cmap="viridis",
        save_dir=None,
        as_pdf=False,
        regenerate=False,
    ):
        if np.array(bins).ndim == 1:
            ticks = bins
            max_bins = len(bins)
        elif bins.ndim == 2:
            ticks = [f"{x}, {y}" for x, y in bins]
            max_bins = np.max(bins, axis=0) + 1
        tick_positions = np.arange(len(bins))

        tick_steps = make_list_ifnot(tick_steps)
        for i, tick_step in enumerate(tick_steps):
            tick_steps[i] = 1 if tick_step == 0 else tick_step
        if len(tick_steps) != len(max_bins):
            tick_steps = [tick_steps[0]] * len(max_bins)

        max_value = {}
        for name, group_similarities in similarities.items():
            if name not in max_value.keys():
                max_value[name] = 0
            for dists in group_similarities.values():
                max_value[name] = np.max([max_value[name], np.max(dists)])

        for name, group_similarities in similarities.items():
            name = name.lower()
            if name in skip:
                continue
            # plot with all groups of a distance metric into a single plot with multiple heatmaps
            title = f"{str(name).capitalize()} Distances"
            title += f" {additional_title}" if additional_title else ""
            if name == "cosine":
                title = title.replace("Distances", "Similarities")
            if name == "overlap" or name == "cross_entropy":
                title = title.replace("Distances", "")
            title = f"{title}_heatmap"

            if isinstance(max_bins, int) or max_bins.ndim == 0:
                fig, axes = plt.subplots(
                    max_bins,
                    1,
                    figsize=(figsize[0], figsize[1]),
                )
            elif max_bins.ndim == 1:
                fig, axes = plt.subplots(
                    int(max_bins[0]),
                    int(max_bins[1]),
                    figsize=(max_bins[0] * figsize[0], max_bins[1] * figsize[1]),
                )
            save_path = Vizualizer.create_save_path(
                save_dir=save_dir, title=title, format="pdf" if as_pdf else "png"
            )
            if Path(save_path).exists() and not regenerate:
                # load existing picture from save_path
                Vizualizer.plot_image(figsize=figsize, save_path=save_path, show=True)
            else:
                fontsize = Vizualizer.auto_fontsize(fig)
                fig.supxlabel(supxlabel, fontsize=fontsize, x=0.5, y=-0.03)
                fig.align_xlabels()
                fig.supylabel(supylabel, fontsize=fontsize, x=-0.02, y=0.5)
                fig.suptitle(title, fontsize=fontsize, y=1.01)
                fig.tight_layout()

                for group_i, (group_name, dists) in enumerate(
                    group_similarities.items()
                ):
                    vmin = None
                    vmax = None
                    if name in ["cosine", "overlap", "cross_entropy"]:
                        cmap = cmap.replace("_r", "")
                    else:
                        if "_r" not in cmap:
                            cmap = f"{cmap}_r"

                    if name in [
                        "euclidean",
                        "wasserstein",
                        "kolmogorov-smirnov",
                        "chi2",
                        "kullback-leibler",
                        "jensen-shannon",
                        "energy",
                        "mahalanobis",
                        "cosine",
                        "overlap",
                        "cross_entropy",
                    ]:
                        vmin = 0
                    elif name in ["cosine", "overlap", "cross_entropy"]:
                        vmin = 0
                        vmax = 1
                    elif name in ["correlation"]:
                        vmin = -1
                        vmax = 1

                    if vmax is None:
                        vmax = max_value[name]
                    # ax = axes[max_bins[1]-1-j, max_bins[0]-1-i]
                    subplot_xticks = []
                    subplot_xticks_pos = []
                    subplot_yticks = []
                    subplot_yticks_pos = []
                    if xticks is None:
                        xticks = ticks
                    xtick_positions = np.arange(len(xticks))
                    if yticks is None:
                        yticks = ticks
                    ytick_positions = np.arange(len(yticks))
                    if isinstance(group_name, str) or np.array(group_name).ndim == 0:
                        i = group_i
                        ax = axes[i]
                        maximal_bin = max_bins
                        if i == max_bins - 1:
                            subplot_xticks = xticks[:: tick_steps[0]]
                            subplot_xticks_pos = xtick_positions[:: tick_steps[0]]
                    elif np.array(group_name).ndim == 1:
                        i, j = group_name
                        ax = axes[i, j]
                        maximal_bin = max_bins[0]
                        if i == max_bins[0] - 1:
                            subplot_xticks = xticks[:: tick_steps[0]]
                            subplot_xticks_pos = xtick_positions[:: tick_steps[0]]
                        if j == 0:
                            subplot_yticks = yticks[:: tick_steps[1]]
                            subplot_yticks_pos = ytick_positions[:: tick_steps[1]]

                    subplot_title_size = maximal_bin * 1.7
                    if isinstance(group_name, tuple) or not is_integer(group_name):
                        subtitle = f"{group_name}"
                    else:
                        subtitle = f"{bins[group_name]}"

                    cax = Vizualizer.plot_heatmap(
                        dists,
                        title=subtitle,
                        title_size=subplot_title_size,
                        xlabel="",
                        ylabel="",
                        vmin=vmin,
                        vmax=vmax,
                        xticks=subplot_xticks,
                        xticks_pos=subplot_xticks_pos,
                        yticks=subplot_yticks,
                        yticks_pos=subplot_yticks_pos,
                        ax=ax,
                        cmap=cmap,
                        interpolation="none",
                    )

                if colorbar:
                    if vmin is None or vmax is None:
                        print("No colorbar label give, using default")
                        vmin = vmin or -777
                        vmax = vmax or 777
                    labels = np.linspace(vmin, vmax, 5)
                    Vizualizer.add_1d_colormap_legend(
                        fig=fig,
                        labels=labels,
                        label_name=colorbar_label,
                        label_size=figsize[0] * 2,
                        cmap=cmap,
                        move_right=1,
                    )
                Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
            plt.show()

    def plot_1d_iter_group_distr_similarities(
        similarities: dict,  # {metric: {group_name: np.array}}
        bins,
        skip=[],
        supxlabel="Bin X",
        supylabel="Bin Y",
        figsize=(3, 3),
        tick_steps=3,
        additional_title="",
        colorbar=False,
        cmap="viridis",
    ):

        # plot with all groups of a distance metric into a single plot with multiple heatmaps
        num_x_plots = len(similarities)
        num_y_plots = len(similarities[list(similarities.keys())[0]])
        figsize = (figsize[0] * num_y_plots, figsize[1] * num_x_plots)
        fig, axes = plt.subplots(
            num_y_plots,
            num_x_plots,
            figsize=(figsize[0], figsize[1]),
        )
        suptitle = (
            f"Similarity Measure comparisson using spatial zones{additional_title}"
        )
        fig.suptitle(suptitle, fontsize=figsize[0], y=0.98)
        fig.supxlabel(supxlabel, fontsize=figsize[0] / 2, x=0.5, y=-0.03)
        fig.align_xlabels()
        fig.supylabel(supylabel, fontsize=figsize[0] / 2, x=-0.02, y=0.5)

        max_value = {}
        for iter_num, (iter, metric_similarities) in enumerate(similarities.items()):
            for name, group_similarities in metric_similarities.items():
                if name not in max_value.keys():
                    max_value[name] = 0
                for dists in group_similarities.values():
                    max_value[name] = np.max([max_value[name], np.max(dists)])

        for iter_num, (iter, metric_similarities) in enumerate(similarities.items()):
            for group_num, (name, group_similarities) in enumerate(
                metric_similarities.items()
            ):
                if name in skip:
                    continue

                if np.array(bins).ndim == 1:
                    ticks = bins
                    max_bins = len(bins)
                elif bins.ndim == 2:
                    ticks = [f"{x}, {y}" for x, y in bins]
                    max_bins = np.max(bins, axis=0) + 1
                tick_positions = np.arange(len(bins))

                subplot_title = f"{str(name).capitalize()} {iter}"

                similarity_matrix = np.zeros((max_bins, max_bins))
                for group_i, (group_name, dists) in enumerate(
                    group_similarities.items()
                ):
                    similarity_matrix[group_i] = dists

                vmin = None
                vmax = None
                if name in ["cosine", "overlap"]:
                    vmax = 1
                    cmap = "viridis"
                else:
                    if "_r" not in cmap:
                        cmap = f"{cmap}_r"
                if name in [
                    "euclidean",
                    "wasserstein",
                    "kolmogorov-smirnov",
                    "chi2",
                    "kullback-leibler",
                    "jensen-shannon",
                    "energy",
                    "mahalanobis",
                    "overlap",
                ]:
                    vmin = 0
                elif name in ["correlation", "cosine"]:
                    vmin = -1
                    vmax = 1

                if vmax is None:
                    vmax = max_value[name]

                # ax = axes[max_bins[1]-1-j, max_bins[0]-1-i]
                subplot_xticks = []
                subplot_xticks_pos = []
                subplot_yticks = []
                subplot_yticks_pos = []
                xplot_num = len(similarities) - 1 - iter_num
                ax = axes[group_num, xplot_num]
                subplot_title_size = 20
                if True:  # group_num == len(metric_similarities) - 1:
                    subplot_xticks = ticks[::tick_steps]
                    subplot_xticks_pos = tick_positions[::tick_steps]
                if xplot_num == 0:
                    subplot_yticks = ticks[::tick_steps]
                    subplot_yticks_pos = tick_positions[::tick_steps]

                cax = Vizualizer.plot_heatmap(
                    similarity_matrix,
                    title=f"{subplot_title}",
                    title_size=subplot_title_size,
                    xlabel="",
                    ylabel="",
                    vmin=vmin,
                    vmax=vmax,
                    xticks=subplot_xticks,
                    xticks_pos=subplot_xticks_pos,
                    yticks=subplot_yticks,
                    yticks_pos=subplot_yticks_pos,
                    ax=ax,
                    cmap=cmap,
                    interpolation="none",
                )

                if colorbar and xplot_num == len(similarities) - 1:
                    # move position of colorbar
                    # fig.subplots_adjust(right=0.8)
                    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    # set colorbar range
                    fig.colorbar(cax, ax=ax)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def create_save_path(
        save_dir: str = None,
        title: str = "NONAME_DEFINED",
        format: str = "pdf",
    ) -> str:
        """Create a save path for the plot"""
        if save_dir:
            # clean title
            title = clean_filename(title)
            save_path = os.path.join(save_dir, f"{title}.{format}")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            return save_path
        return None

    @staticmethod
    def save_plot(save_dir=None, title="NONAME_DEFINED", format="pdf"):
        title = clean_filename(title)
        save_path = Vizualizer.create_save_path(
            save_dir=save_dir, title=title, format=format
        )
        if save_path:
            if len(save_path) > 255:
                global_logger.warning(
                    f"Save path is too long: {save_path}. Using shortened version."
                )
                save_path = save_path[:245] + f"....{format}"
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format=format)

    @staticmethod
    def _get_cmap(cmap: str = None, num_colors: int = None) -> plt.cm:
        """Get a colormap"""
        if cmap is None:
            cmap = "Set1"
            if num_colors:
                if num_colors <= 10:
                    cmap = "Set1"
                elif num_colors <= 20:
                    cmap = "tab20"
                else:
                    cmap = "rainbow"
        # check if cmap is a continuouse colormap
        if is_continuous_colormap(cmap):
            cmap = plt.cm.get_cmap(cmap, num_colors)
        else:
            cmap = plt.cm.get_cmap(cmap)
        return cmap

    @staticmethod
    def _get_base_color(
        index: Union[int, List[int]],
        num_colors: int = None,
        cmap=None,
        style: Literal["matplotlib", "plotly"] = "matplotlib",
    ) -> Union[tuple, List[tuple]]:
        """Get a base color from colormap"""
        if is_array_like(index):
            index = np.array(index)
        if is_int_like(index):
            index = np.array([index])

        if num_colors is None:
            num_colors = len(index)

        cmap = Vizualizer._get_cmap(cmap=cmap, num_colors=num_colors)
        color = []
        for i in index:
            ind = i % num_colors if i > num_colors else i
            color.append(cmap(ind))

        if style == "plotly":
            # convert to rgba string for plotly in range between 0 and 255
            color = [
                f"rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {c[3]})"
                for c in color
            ]
        elif style == "matplotlib":
            pass
        else:
            raise ValueError(f"Unknown style: {style}. Use 'matplotlib' or 'plotly'.")
        return np.array(color)

    @staticmethod
    def _get_alpha_colors(
        base_color: tuple,
        values: List[float],
        min_value: float = None,
        max_value: float = None,
        min_alpha: float = 0.5,
        max_alpha: float = 1.0,
    ) -> List[tuple]:
        """Generate colors with alpha based on value range"""
        alphas = Vizualizer.calculate_alpha(
            values, min_value, max_value, min_alpha, max_alpha
        )
        alphas = make_list_ifnot(alphas)
        colors = [mcolors.to_rgba(base_color, alpha=alpha) for alpha in alphas]
        return colors

    @staticmethod
    def _add_bars_and_labels(
        ax, positions, values, colors, width, variances=None, label=None
    ):
        """Add bars, error bars, and value labels to the plot"""
        bars = ax.bar(positions, values, width=width, color=colors, label=label)

        if variances:
            ax.errorbar(
                positions,
                values,
                yerr=variances,
                fmt="none",
                ecolor="white",
                capsize=5,
                alpha=0.3,
            )

        for bar, value in zip(bars, values):
            value = int(value) if value is not None and not np.isnan(value) else None
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        return bars

    @staticmethod
    def barplot_from_dict(
        data: Dict[str, Union[float, int]],
        title: str = "Bar Plot",
        additional_title: str = "",
        xlabel: str = "Categories",
        ylabel: str = "",
        xticks: List[str] = None,
        figsize: tuple = (5, 7),
        bar_width: float = 0.8,
        save_dir: str = None,
        as_pdf: bool = False,
    ):
        """Plot a simple bar chart from a single dictionary

        This function needs mean and variance number to function properly.

        Parameters:
        -----------
        data (Dict[str, Union[float, int]]):
            Dictionary with categories as keys and dictionaries as values with mean and variance as keys.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        categories = list(data.keys()) if xticks is None else xticks
        values = [
            v * 100 if isinstance(v, (int, float)) else v["mean"] * 100
            for v in data.values()
        ]
        variances = [
            v["variance"] * 100 if isinstance(v, dict) else None for v in data.values()
        ]
        colors = Vizualizer._get_base_color(np.arange(len(categories)))
        # colors = Vizualizer._get_alpha_colors(base_color, values)
        positions = np.arange(len(categories))

        # Plot bars
        Vizualizer._add_bars_and_labels(
            ax,
            positions,
            values,
            colors,
            bar_width,
            [v for v in variances if v is not None],
        )

        # Customize plot
        title = f"{title} {additional_title}" if additional_title else title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=45, ha="right")

        plt.tight_layout()

        if save_dir:
            ext = "pdf" if as_pdf else "png"
            plt.savefig(f"{save_dir}/{title}.{ext}", bbox_inches="tight")

        plt.show()

    @staticmethod
    def barplot_from_dict_of_dicts(
        data: Dict[str, Dict[str, Union[float, int]]],
        title: str = "Decoding of Position",
        additional_title: str = "",
        legend_title: str = "Source: Mean RMSE",
        xticks: List[str] = None,
        xlabel: str = "Model Based on Source and compared to Task",
        ylabel: str = "RMSE (cm)",
        figsize: tuple = (5, 7),
        base_bar_width: float = 1,
        distance_between_bars: float = 0,
        distance_between_sources: float = None,
        save_dir: str = None,
        as_pdf: bool = False,
        cmap: str = None,
    ):
        """
        Plot grouped bar chart from dictionary of dictionaries
        """

        num_sources = len(data)
        bar_width = base_bar_width / num_sources
        distance_between_sources = (
            bar_width * 2
            if distance_between_sources is None
            else distance_between_sources
        )

        position = 0
        all_positions = []
        all_labels = []
        figsize = figsize[0] * num_sources, figsize[1]
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = Vizualizer.auto_fontsize(fig)

        base_colors = Vizualizer._get_base_color(np.arange(num_sources), cmap=cmap)
        for source_num, (source, task_dict) in enumerate(data.items()):
            tasks = list(task_dict.keys())
            values = [
                v * 100 if isinstance(v, (int, float)) else v["mean"] * 100
                for v in task_dict.values()
            ]
            variances = [
                v["variance"] * 100 if isinstance(v, dict) else None
                for v in task_dict.values()
            ]
            base_color = base_colors[source_num]
            alpha_range = np.arange(len(task_dict))[::-1]
            colors = Vizualizer._get_alpha_colors(base_color, alpha_range)
            positions = [
                position + i * (bar_width + distance_between_bars)
                for i in range(len(tasks))
            ]

            # Plot bars
            bars = Vizualizer._add_bars_and_labels(
                ax,
                positions,
                values,
                colors,
                bar_width,
                [v for v in variances if v is not None],
                label=f"{source}: {np.mean(values):.2f}",
            )

            # Update positions and labels
            position = positions[-1] + bar_width + distance_between_sources
            all_positions.extend(positions)
            all_labels.extend([task for task in tasks])

        # Customize plot
        full_title = f"{title} {additional_title}" if additional_title else title
        ax.set_xlabel(xlabel, fontsize=fontsize * 0.9)
        ax.set_ylabel(ylabel, fontsize=fontsize * 0.9)
        ax.set_title(full_title, fontsize=fontsize)
        ax.set_xticks(all_positions)
        ax.set_xticklabels(
            all_labels if xticks is None else xticks,
            rotation=45,
            ha="right",
            fontsize=fontsize * 0.8,
        )

        plt.tight_layout()
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")

        if save_dir:
            ext = "pdf" if as_pdf else "png"
            plt.savefig(f"{save_dir}/{full_title}.{ext}", bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_3D_group_scatter(
        gropu_data: np.ndarray,
        additional_title: str = "",
        xlabel: str = "X",
        ylabel: str = "Y",
        zlabel: str = "Z",
        cmap="rainbow",
        figsize=(20, 20),
        use_alpha=True,
        filter_outlier: bool = False,
        outlier_threshold: float = 0.2,
        specific_group: Tuple[int, int] = None,
        plot_legend=True,
        save_dir=None,
        as_pdf=False,
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        title = "3D Scatter"
        title += f" {additional_title}" if additional_title else ""
        title += f" {specific_group}" if specific_group else ""
        title += f" w/o alpha" if not use_alpha else ""

        # Define unique colors for each group
        group_colors = Vizualizer._get_base_color(np.arange(len(gropu_data)), cmap=cmap)

        # Plot each group with its unique color and density-based transparency
        c_outliers = 0
        c_samples = 0
        for i, (group_name, data) in enumerate(gropu_data.items()):
            if specific_group:
                if group_name != specific_group:
                    continue

            locations_raw = data["locations"]
            values_raw = data["values"]
            c_samples += len(values_raw)
            c_outliers += np.sum(values_raw < outlier_threshold)
            usefull_idx = values_raw > outlier_threshold

            if sum(usefull_idx) == 0:
                continue
            locations = locations_raw[usefull_idx] if filter_outlier else locations_raw
            values = values_raw[usefull_idx] if filter_outlier else values_raw

            # Normalize the density values to be between 0 and 1 for alpha
            # norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            # alphas = norm(values)
            norm = mcolors.Normalize(vmin=0, vmax=max(values))
            alphas = norm(values) if use_alpha else np.ones_like(values)

            # Convert RGB color to RGBA with alpha
            rgba_colors = np.zeros((locations.shape[0], 4))
            rgba_colors[:, :3] = group_colors[i][:3]  # Assign the unique color
            rgba_colors[:, 3] = alphas  # Assign the alpha values
            # filter for every 10th point
            steps = 1
            part_locations = locations[::steps]
            rgba_colors = rgba_colors[::steps]
            ax.scatter(
                part_locations[:, 0],
                part_locations[:, 1],
                part_locations[:, 2],
                color=rgba_colors,
                label=group_name,
                edgecolor=None,
                s=10,
            )

        if filter_outlier:
            title += f" {c_outliers/c_samples:.2%} outliers"

        # Add labels and legend
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        if plot_legend:
            ax.legend()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")

        plt.show()

    def plot_2d_group_scatter(
        group_data: dict,
        filter_outlier: bool = False,
        outlier_threshold: float = 0.2,
        additional_title=None,
        supxlabel: str = "Bin X",
        supylabel: str = "Bin Y",
        figsize=(2, 2),
        plot_legend=False,
        cmap="rainbow",
        save_dir=None,
        as_pdf=False,
        use_alpha=True,
        same_subplot_range=False,
        regenerate=False,
    ):
        """
        group data is expected to have x, y coordinates as group_name
        """
        title = "Point Distributions in 2D"
        title += f" {additional_title}" if additional_title else ""
        title += f" w/o alpha" if not use_alpha else ""
        title += f" w/o outlier" if filter_outlier else ""

        # Determine the number of subplots
        unique_bins = np.unique(list(group_data.keys()), axis=0)
        max_bins = np.max(unique_bins, axis=0) + 1
        figsize = (figsize[0] * max_bins[0], figsize[1] * max_bins[1])
        fig, axes = plt.subplots(max_bins[0], max_bins[1], figsize=figsize)

        # Define unique colors for each group
        num_groups = len(unique_bins)
        group_colors = Vizualizer._get_base_color(np.arange(num_groups), cmap=cmap)

        # Plot each group in its respective subplot
        c_outliers = 0
        c_samples = 0
        min_max_subplot_x = [0, 0]
        min_max_subplot_y = [0, 0]
        part_locations_dict = {}
        part_rgba_colors = {}
        for i, (group_name, data) in enumerate(group_data.items()):
            loc_x, loc_y = group_name
            # remove axis labels
            axes[loc_x, loc_y].set_xticks([])
            axes[loc_x, loc_y].set_yticks([])

            # Optionally, add axis labels
            # axes[loc_x, loc_y].set_xlabel('X axis')
            # axes[loc_x, loc_y].set_ylabel('Y axis')

            locations_raw = data["locations"]
            values_raw = data["values"]
            c_samples += len(values_raw)
            c_outliers += np.sum(values_raw < outlier_threshold)
            usefull_idx = values_raw > outlier_threshold

            if sum(usefull_idx) == 0:
                continue
            locations = locations_raw[usefull_idx] if filter_outlier else locations_raw
            values = values_raw[usefull_idx] if filter_outlier else values_raw
            if len(locations[0]) == 3:
                # Project 3D locations to 2D
                locations = pca_numba(locations)
                # locations = mds_numba(locations)

            # Normalize the density values to be between 0 and 1 for alpha
            norm = mcolors.Normalize(vmin=0, vmax=max(values))
            alphas = norm(values) if use_alpha else np.ones_like(values)

            # Convert RGB color to RGBA with alpha
            rgba_colors = np.zeros((locations.shape[0], 4))
            rgba_colors[:, :3] = group_colors[i][:3]  # Assign the unique color
            rgba_colors[:, 3] = alphas  # Assign the alpha values

            # filter for every step point
            steps = 1
            part_locations = locations[::steps]
            rgba_colors = rgba_colors[::steps]
            part_locations_dict[group_name] = part_locations
            part_rgba_colors[group_name] = rgba_colors

        if filter_outlier:
            title += f" {c_outliers/c_samples:.2%} outliers"
        save_path = Vizualizer.create_save_path(
            save_dir=save_dir, title=title, format="pdf" if as_pdf else "png"
        )
        if Path(save_path).exists() and not regenerate:
            # load existing picture from save_path
            plt.close(fig)
            Vizualizer.plot_image(figsize=figsize, save_path=save_path, show=True)
        else:
            for i, (group_name, part_locations) in enumerate(
                part_locations_dict.items()
            ):
                loc_x, loc_y = group_name
                rgba_colors = part_rgba_colors[group_name]

                # Plot in 2D
                axes[loc_x, loc_y].scatter(
                    part_locations[:, 0],
                    part_locations[:, 1],
                    color=rgba_colors,
                    label=group_name,
                    edgecolor=None,
                    s=10,
                )
                fontsize = Vizualizer.auto_fontsize(fig)
                axes[loc_x, loc_y].set_title(
                    group_name, fontsize=fontsize * 0.3, pad=figsize[0] / max_bins[0]
                )

                # check min max for subplot
                if same_subplot_range:
                    min_max_subplot_x[0] = min(
                        min_max_subplot_x[0], np.min(part_locations[:, 0])
                    )
                    min_max_subplot_x[1] = max(
                        min_max_subplot_x[1], np.max(part_locations[:, 0])
                    )
                    min_max_subplot_y[0] = min(
                        min_max_subplot_y[0], np.min(part_locations[:, 1])
                    )
                    min_max_subplot_y[1] = max(
                        min_max_subplot_y[1], np.max(part_locations[:, 1])
                    )

            # resize subplots to have the same range
            if same_subplot_range:
                for ax in axes.flat:
                    ax.set_xlim(min_max_subplot_x)
                    ax.set_ylim(min_max_subplot_y)

            fig.suptitle(title, fontsize=fontsize, y=1.01)
            fig.supxlabel(supxlabel, fontsize=fontsize, x=0.5, y=-0.03)
            fig.align_xlabels()
            fig.supylabel(supylabel, fontsize=fontsize, x=-0.02, y=0.5)

            if plot_legend:
                plt.legend()

            fig.tight_layout()
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
            plt.show()

    @staticmethod
    def plot_structure_index(
        embedding: Optional[np.ndarray] = None,
        feature: Optional[np.ndarray] = None,
        overlapMat: Optional[np.ndarray] = None,
        save_dir: Union[str, Path] = None,
        SI: Optional[float] = None,
        binLabel: Optional[np.ndarray] = None,
        values: Optional[List[float]] = None,
        sweep_range: Optional[List] = None,
        additional_title: str = None,
        title: str = "Structural Index",
        figsize=(18, 5),
        cmap="rainbow",
        show=True,
        as_pdf=False,
    ):
        """
        Unified plotting method for structural index.

        For single SI value (sweep_range=None): Plots 3-panel figure (embedding, adjacency matrix, directed graph).
        For multiple SI values (values and sweep_range provided): Plots line plot of SI vs. sweep_range using lineplot_from_dict_of_dicts.

        Args:
            embedding, feature, overlapMat, SI, binLabel: For single-value 3-panel plot.
            values, sweep_range: For multi-value line plot (SI sweeps).
            ... (other args as before)
        """
        if values is not None and sweep_range is not None and len(sweep_range) > 1:
            # Multi-value sweep: Construct dict for lineplot_from_dict_of_dicts
            sweep_dict = {"sweep": {"SI": values}}  # Single source/task; values as list
            sweep_title = (
                f"Structure Index for {additional_title} sweep {sweep_range[0]}-{sweep_range[-1]}"
                if additional_title
                else f"Structural Index sweep {sweep_range[0]}-{sweep_range[-1]}"
            )

            Vizualizer.lineplot_from_dict_of_dicts(
                data=sweep_dict,
                to_show=["mean"],  # Single series; no std/samples needed
                title=sweep_title,
                xlabel="n_neighbors",
                ylabel="Structural Index",
                xticks=list(sweep_range),
                xtick_pos=np.arange(len(sweep_range)),
                figsize=(10, 6),  # Standard size for line plot
                save_dir=save_dir,
                as_pdf=as_pdf,
                show_plot=show,
                grid=True,
                std_alpha=0.0,  # No std shading for single series
            )
            return  # Early return after line plot

        elif (
            embedding is not None
            and feature is not None
            and overlapMat is not None
            and SI is not None
            and binLabel is not None
        ):
            # Single-value case: Original 3-panel plot
            # Create RGBA colors for sessions
            session_labels, min_vals, max_vals = Vizualizer.create_RGBA_colors_from_2d(
                feature
            )

            # Original three-panel figure
            fig, ax = plt.subplots(1, 3, figsize=figsize)
            plot_title = (
                title + f" {additional_title}" if additional_title != "" else title
            )
            fig.suptitle(plot_title, fontsize=20)

            # Plot 3D scatter (embedding)
            at = plt.subplot(1, 3, 1, projection="3d")
            embedding_title = "Embedding"

            if embedding.shape[1] != 3 and embedding.shape != (embedding.shape[0], 2):
                dummy_embedding = np.zeros((10, 3))
                dummy_labels = np.zeros(10)
                dummy_cmap = "viridis"
            else:
                dummy_embedding = None
                dummy_labels = None
                dummy_cmap = None

            session_labels_dict = {
                "name": "",
                "labels": session_labels if dummy_labels is None else dummy_labels,
            }

            Vizualizer.plot_embedding(
                ax=at,
                embedding=embedding if dummy_embedding is None else dummy_embedding,
                embedding_labels=session_labels_dict,
                title=embedding_title,
                show_hulls=False,
                cmap=cmap if dummy_cmap is None else dummy_cmap,
                plot_legend=False,
            )
            if dummy_embedding is None:
                if len(feature.shape) == 2 and feature.shape[1] == 2:
                    Vizualizer.add_2d_colormap_legend(fig=fig, legend_left=-0.1)

            # Plot adjacency matrix
            b = ax[1].matshow(
                overlapMat, vmin=0, vmax=0.5, cmap=matplotlib.cm.get_cmap("viridis")
            )
            ax[1].xaxis.set_ticks_position("bottom")
            cbar = fig.colorbar(
                b, ax=ax[1], anchor=(0, 0.2), shrink=1, ticks=[0, 0.25, 0.5]
            )
            cbar.set_label("overlap score", rotation=90, fontsize=14)
            ax[1].set_title("Adjacency matrix", size=16)
            ax[1].set_xlabel("bin-groups", size=14)
            ax[1].set_ylabel("bin-groups", size=14)

            # Plot weighted directed graph
            draw_graph(
                overlapMat,
                ax[2],
                node_cmap=matplotlib.cm.get_cmap("rainbow"),
                edge_cmap=plt.cm.Greys,
                node_names=np.round(binLabel[1][:, 0, 1], 2),
            )
            ax[2].set_xlim(1.2 * np.array(ax[2].get_xlim()))
            ax[2].set_ylim(1.2 * np.array(ax[2].get_ylim()))
            ax[2].set_title("Directed graph", size=16)
            ax[2].text(
                0.98,
                0.05,
                f"SI: {SI:.2f}",
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax[2].transAxes,
                fontsize=25,
            )

            plt.tight_layout()
            Vizualizer.plot_ending(
                title=plot_title, as_pdf=as_pdf, show=show, save_dir=save_dir
            )

        else:
            raise ValueError(
                "Invalid parameters: Provide either (embedding, feature, overlapMat, SI, binLabel) for single plot OR (values, sweep_range) for line plot."
            )

    @staticmethod
    def prep_XD_subplot_template(
        data: Dict[str, Dict[str, Union[np.ndarray, List]]] = None,
        nrows: int = None,
        ncols: int = None,
        first_direction: Literal["x", "y"] = "x",
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 6),
        plot_show: bool = True,
        sharex: bool = True,
        sharey: bool = True,
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray], bool, int, int]:
        """Prepare the template for 1D/2D subplots.

        Args:
            data: Nested dictionary of data for plotting.
            nrows: Number of rows in the subplot grid.
            ncols: Number of columns in the subplot grid.
            first_direction: Direction of subplot expansion ('x' or 'y').
            ax: Existing axes to use. If None, new axes will be created.
            figsize: Size of the figure.
            plot_show: Whether to show the plot.
            sharex: Share x-axis across subplots.
            sharey: Share y-axis across subplots.

        Returns:
            Tuple containing figure, axes, plot_show flag, number of columns, and number of rows.

        Raises:
            ValueError: If input parameters are invalid.
            TypeError: If data is not a dictionary.
        """
        if data is not None and not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary for plotting dict of dicts subplots"
            )

        if data is None and nrows is None and ncols is None:
            raise ValueError("Either data or nrows and ncols must be provided.")
        elif data is not None:
            if nrows is not None or ncols is not None:
                global_logger.debug(
                    "data and nrows/ncols provided. Using nrows/ncols as input."
                )
            len_first_level = len(data)
            first_element = data[list(data.keys())[0]]
            len_second_level = (
                len(first_element) if isinstance(first_element, dict) else 1
            )
            ncols = len_first_level
            nrows = nrows or len_second_level
            if first_direction == "y":
                nrows, ncols = ncols, nrows
        elif nrows is None or ncols is None:
            raise ValueError("nrows and ncols must be provided if data is None.")

        plot_figsize = (figsize[0] * ncols, figsize[1] * nrows)

        if ax is None:
            fig, ax = plt.subplots(
                figsize=plot_figsize,
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
            )
        else:
            plot_show = False
            if len(ax) != ncols:
                raise ValueError(
                    f"Number of axes {len(ax)} does not match number of conditions {ncols}."
                )
            fig = None
        return fig, ax, plot_show, ncols, nrows

    @staticmethod
    def end_XD_subplot(
        plot_show: bool,
        fig: plt.Figure,
        title: str,
        additional_title: str = "",
        as_pdf: bool = False,
        save_dir: str = None,
    ) -> None:
        """Finalize and display/save the subplot.

        Args:
            plot_show: Whether to display the plot.
            fig: Matplotlib figure object.
            title: Main title of the plot.
            additional_title: Additional text to append to the title.
            as_pdf: Save as PDF if True, else PNG.
            save_dir: Directory to save the plot.
        """
        if plot_show:
            title = f"{title} {additional_title}" if additional_title else title
            fontsize = Vizualizer.auto_fontsize(fig)
            # shift title up for every \n in title
            title_y_shift = 1.0 + 0.04 * title.count("\n")
            fig.suptitle(title, fontsize=fontsize, y=title_y_shift)
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
            plt.show()

    @staticmethod
    def plot_2d_line_dict_of_dicts(
        data: Dict[str, Dict[str, Dict[str, Union[float, int, List, np.ndarray]]]],
        ax: Optional[plt.Axes] = None,
        first_direction: Literal["x", "y"] = "x",
        to_show: Union[str, List[Literal["mean", "std", "samples"]]] = ["mean", "std"],
        title: str = "Decoding of Position",
        additional_title: str = "",
        legend_title: str = None,
        xlabel: str = "Iterations",
        ylabel: str = "",
        xticks: List[int] = None,
        yticks: List[int] = None,
        xtick_pos: Optional[list] = None,
        ytick_pos: Optional[list] = None,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
        line_width=3,
        figsize=(10, 6),
        save_dir: str = None,
        as_pdf: bool = False,
        cmap: str = "Set3",
        std_alpha: float = 0.2,
        sharex: bool = True,
        sharey: bool = True,
    ):
        """
        Plot line plot per 2 groups with different tasks as subgroups.

        The Plot will be created with subplots for each condition and case, which will create a grid of subplots.
        Subplots increase in X direction and Y direction.
        """
        fig, ax, plot_show, ncols, nrows = Vizualizer.prep_XD_subplot_template(
            data=data,
            first_direction=first_direction,
            ax=ax,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        for cond_num, (condition, cond_data) in enumerate(data.items()):
            additional_title = condition
            axe = ax[cond_num] if ncols > 1 else ax
            Vizualizer.plot_1d_line_dict_of_dicts(
                data=cond_data,
                to_show=to_show,
                additional_title=additional_title,
                subplot_directions=first_direction,
                ax=axe,
                title=condition,
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=xticks,
                yticks=yticks,
                xtick_pos=xtick_pos,
                ytick_pos=ytick_pos,
                xlim=xlim,
                ylim=ylim,
                line_width=line_width,
                figsize=figsize,
                save_dir=save_dir,
                as_pdf=as_pdf,
                cmap=cmap,
                std_alpha=std_alpha,
            )

        Vizualizer.end_XD_subplot(
            plot_show=plot_show,
            fig=fig,
            title=title,
            additional_title=additional_title,
            as_pdf=as_pdf,
            save_dir=save_dir,
        )

    @staticmethod
    def plot_1d_line_dict_of_dicts(
        data: Dict[str, Dict[str, Dict[str, Union[float, int, List, np.ndarray]]]],
        ax: Optional[plt.Axes] = None,
        subplot_directions: Literal["x", "y"] = "x",
        to_show: Union[str, List[Literal["mean", "std", "samples"]]] = ["mean", "std"],
        title: str = "Some Title",
        additional_title: str = "",
        xlabel: str = "Iterations",
        ylabel: str = "",
        xticks: List[int] = None,
        yticks: List[int] = None,
        xtick_pos: Optional[list] = None,
        ytick_pos: Optional[list] = None,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
        line_width=3,
        figsize=(10, 6),
        save_dir: str = None,
        as_pdf: bool = False,
        cmap: str = None,
        std_alpha: float = 0.2,
        sharex: bool = True,
        sharey: bool = True,
    ):
        """
        Plot multiple line plots one per condition with multiple source with different tasks as subgroups.

        The Plot will be created with subplots for each condition. Subplots increase in X direction.


        """
        fig, ax, plot_show, ncols, nrows = Vizualizer.prep_XD_subplot_template(
            data=data,
            nrows=1,
            first_direction=subplot_directions,
            ax=ax,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
        )

        for cond_num, (condition, cond_data) in enumerate(data.items()):
            subplot_title = f"{condition} {additional_title}"
            axe = ax[cond_num] if isinstance(ax, np.ndarray) else ax
            Vizualizer.lineplot_from_dict_of_dicts(
                data=cond_data,
                to_show=to_show,
                ax=axe,
                title=subplot_title,
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=xticks,
                yticks=yticks,
                xtick_pos=xtick_pos,
                ytick_pos=ytick_pos,
                xlim=xlim,
                ylim=ylim,
                line_width=line_width,
                figsize=figsize,
                save_dir=save_dir,
                as_pdf=as_pdf,
                cmap=cmap,
                std_alpha=std_alpha,
            )

        Vizualizer.end_XD_subplot(
            plot_show=plot_show,
            fig=fig,
            title=title,
            additional_title=additional_title,
            as_pdf=as_pdf,
            save_dir=save_dir,
        )

        return ax

    @staticmethod
    def lineplot_from_dict_of_dicts(
        data: Dict[str, Dict[str, Union[float, int]]],
        to_show: Union[str, List[Literal["mean", "std", "samples"]]] = ["mean", "std"],
        ax: Optional[plt.Axes] = None,
        title: str = "",
        additional_title: str = "",
        legend_title: str = None,
        xlabel: str = "Iterations",
        ylabel: str = "",
        xticks: List[int] = None,
        yticks: List[int] = None,
        xtick_pos: Optional[list] = None,
        ytick_pos: Optional[list] = None,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
        line_width=3,
        figsize=(10, 6),
        save_dir: str = None,
        as_pdf: bool = False,
        cmap: str = None,
        std_alpha: float = 0.2,
        grid: bool = True,
        show_plot: bool = True,
    ):
        """
        Plots a line plot per source with different tasks as subgroups.

        Args:
            data: {source: {task: values}}
                where values can be a list or numpy array of shape (n_samples, n_iterations)
                or a dict with "mean" and "std" keys. A markers key is optional and can be used to plot significance markers.
            to_show: What to show in the plot. Can be "mean", "std", or "samples".
            title: Main title of the plot.
            additional_title: Additional text to append to the title.
            legend_title: Title for the legend.
            xlabel: Label for the X-axis.
            ylabel: Label for the Y-axis.
            figsize: Size of the figure.
            save_dir: Directory to save the plot.
            as_pdf: Whether to save the plot as a PDF.
            cmap: Color map to use.
        """
        title = f"{title} {additional_title}" if additional_title else title

        to_show = make_list_ifnot(to_show)
        for show in to_show:
            if show not in ["mean", "std", "samples"]:
                raise ValueError(
                    f"to_show must be one of ['mean', 'std', 'samples'], but got {show}."
                )

        if ax is None and show_plot == True:
            fig, ax = plt.subplots(figsize=figsize)
            show_plot = True
        else:
            show_plot = False

        num_sources = len(data)
        # num_conditions = len(data[list(data.keys())[0]])

        show_samples = "samples" in to_show

        base_colors = Vizualizer._get_base_color(np.arange(num_sources), cmap=cmap)
        for source_num, (source, task_dict) in enumerate(data.items()):
            color = base_colors[source_num]
            colors = Vizualizer._get_alpha_colors(
                color, np.arange(len(task_dict))[::-1]
            )
            for task_num, (task, values) in enumerate(task_dict.items()):
                markers = None
                if isinstance(values, dict):
                    # if values is a dict, extract mean and std
                    if "mean" in to_show:
                        means = values["mean"]
                    else:
                        do_critical(
                            TypeError,
                            "values must be a list or numpy array. inside dict to plot",
                        )
                    if "std" in to_show:
                        std = values["std"]
                    else:
                        std = None

                    if "markers" in values:
                        significance_markers = values["markers"]

                    # For dict values, set values_array to None; sample plotting is determined later by checking if values_array is not None
                    values_array = None

                else:
                    values = np.array(values)
                    # make sure values are 2D
                    if len(values.shape) == 1:
                        values = np.expand_dims(values, axis=0)
                    means = values.mean(axis=0) if "mean" in to_show else values
                    std = values.std(axis=0) if "std" in to_show else None
                    values_array = values

                if (
                    "samples" in to_show
                    and values_array is not None
                    and values_array.shape[0] > 1
                ):
                    # plot gray lines for samples
                    for sample in values_array:
                        # create color based on white and add alpha without using colormap
                        color = Vizualizer._get_base_color(0, cmap="Greys")[0]
                        color = mcolors.to_rgba(color, alpha=0.5)
                        label = (
                            "samples"
                            if np.array_equal(sample, values_array[0])
                            else None
                        )
                        plot_line(
                            ax=ax,
                            values=sample,
                            label=label,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            xlim=xlim,
                            ylim=ylim,
                            xticks=xticks,
                            yticks=yticks,
                            xtick_pos=xtick_pos,
                            ytick_pos=ytick_pos,
                            std=None,
                            color=color,
                            line_width=0.8,
                        )

                # Calculate AUC if samples are not being plotted
                auc = None
                x = np.arange(len(means))
                if not show_samples:
                    auc = np.trapz(means, x=x)
                    # for accurate AUC calculation fit line and calculate AUC using
                    # get_auc(x=xtick_pos, y=means)

                # Create label with AUC if calculated
                if auc is not None:
                    label = f"{source}: {task} (AUC: {auc:.4f})"
                else:
                    label = f"{source}: {task}"

                plot_line(
                    ax=ax,
                    values=means,
                    markers=markers,
                    label=label,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    xlim=xlim,
                    ylim=ylim,
                    xticks=xticks,
                    yticks=yticks,
                    xtick_pos=xtick_pos,
                    ytick_pos=ytick_pos,
                    std=std,
                    color=colors[task_num],
                    line_width=line_width,
                    std_alpha=std_alpha,
                    grid=grid,
                )

                # Add shaded area under the curve if samples are not being plotted
                if not show_samples and num_sources == 1 and len(task_dict) == 1:
                    ax.fill_between(
                        x,
                        means,
                        0,  # baseline
                        color=colors[task_num],
                        alpha=0.15,
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Create legend and adjust layout
        if legend_title is not None:
            ax.legend(title=legend_title, bbox_to_anchor=(1.0, 1), loc="upper left")
        else:
            ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left")
        plt.tight_layout()

        if show_plot:
            plt.show()
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")

        return ax

    @staticmethod
    def plot_heatmap_dict_of_dicts(
        data: Dict[str, Dict[str, np.ndarray]],
        ax: Optional[Union[plt.Axes, np.ndarray]] = None,
        first_direction: Literal["x", "y"] = "x",
        custom_annotation: Dict[
            str, Union[np.ndarray, List[List[str]], pd.DataFrame]
        ] = None,
        custom_annotation_label: str = "Custom Annotation",
        title: str = "Wilcoxon Test p-values",
        additional_title: str = "",
        sort_by: Optional[Literal["value", "similarity"]] = None,
        vmin: float = None,
        vmax: float = None,
        colorbar_ticks: Optional[List[float]] = None,
        colorbar_ticks_labels: Optional[List[str]] = None,
        colorbar_label: str = "",
        labels: Optional[List[str]] = None,
        xlabel: str = "Tasks",
        ylabel: str = "Tasks",
        figsize: tuple = (10, 6),
        save_dir: str = None,
        as_pdf: bool = False,
        sharex: bool = True,
        sharey: bool = True,
        cmap: str = "viridis",
        add_line: Optional[List[Tuple[str, str]]] = None,
    ) -> Union[plt.Axes, np.ndarray]:
        """Plot heatmaps from a nested dictionary of data.

        Args:
            data: Nested dictionary {range_label: {condition: heatmap_data}}.
            ax: Existing axes for subplots.
            first_direction: Direction of subplot expansion ('x' or 'y').
            title: Main title of the plot.
            additional_title: Additional text to append to the title.
            labels: List of task labels for axes.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            figsize: Size of the figure.
            save_dir: Directory to save the plot.
            as_pdf: Save as PDF if True, else PNG.
            sharex: Share x-axis across subplots.
            sharey: Share y-axis across subplots.
            cmap: Colormap name for the heatmap.
            vmin: Minimum value for color scaling.
            vmax: Maximum value for color scaling.
            colorbar_ticks: Custom ticks for the colorbar.
            colorbar_ticks_labels: Custom labels for colorbar ticks.
            colorbar_label: Label for the colorbar.
            add_line: List of tuples (label1, label2) to draw thick lines between.

        Returns:
            Matplotlib axes object(s).

        Raises:
            ValueError: If data structure is invalid.
        """
        fig, ax, plot_show, ncols, nrows = Vizualizer.prep_XD_subplot_template(
            data=data,
            first_direction=first_direction,
            ax=ax,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
        )

        fontsize = Vizualizer.auto_fontsize(fig) if fig else 12

        for i, (range_label, cond_data) in enumerate(data.items()):
            if isinstance(cond_data, dict):
                for j, (condition, heatmap_data) in enumerate(cond_data.items()):
                    if custom_annotation is not None:
                        raise NotImplementedError(
                            "Custom annotation is not implemented for heatmaps based on dictionary of dictionaries in this function."
                        )
                    axe = (
                        ax[i][j]
                        if isinstance(ax, np.ndarray) and nrows > 1
                        else ax[j] if isinstance(ax, np.ndarray) else ax
                    )
                    subtitle = (
                        f"{range_label} - {condition}"
                        if len(cond_data) > 1
                        else range_label
                    )

                    # Create mask for upper triangle (excluding diagonal)
                    mask = np.triu(np.ones_like(heatmap_data, dtype=bool), k=1)
                    masked_data = np.where(mask, np.nan, heatmap_data)

                    # Only put xlabel on last row
                    plot_xlabel = xlabel if i == len(data) - 1 else None

                    # Only put ylabel on first column
                    plot_ylabel = ylabel if j == 0 else None

                    data_custom_annotation = (
                        custom_annotation[range_label] if custom_annotation else None
                    )

                    Vizualizer.plot_heatmap(
                        data=masked_data,
                        custom_annotation=data_custom_annotation,
                        custom_annotation_label=custom_annotation_label,
                        title=subtitle,
                        xlabel=plot_xlabel,
                        ylabel=plot_ylabel,
                        xticks=labels,
                        yticks=labels,
                        fontsize=fontsize * 0.7,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        colorbar=True,
                        colorbar_ticks=colorbar_ticks,
                        colorbar_ticks_labels=colorbar_ticks_labels,
                        colorbar_label=colorbar_label,
                        rotation=45,
                        save_dir=save_dir,
                        as_pdf=as_pdf,
                        ax=axe,
                        fig=fig,
                        show=False,
                        add_line=add_line,
                    )
            elif isinstance(cond_data, pd.DataFrame):
                if isinstance(ax, np.ndarray):
                    axe = ax[i]
                    subplot_fontsize = fontsize * 0.7
                else:
                    axe = ax
                    subplot_fontsize = fontsize
                condition = range_label
                subtitle = f"{condition}" if len(cond_data) > 1 else range_label

                # Create mask for upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(cond_data, dtype=bool), k=1)
                masked_data = np.where(mask, np.nan, cond_data)
                data_custom_annotation = (
                    custom_annotation[range_label] if custom_annotation else None
                )
                Vizualizer.plot_heatmap(
                    data=masked_data,
                    custom_annotation=data_custom_annotation,
                    custom_annotation_label=custom_annotation_label,
                    sort_by=sort_by,
                    title=subtitle,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    xticks=labels,
                    yticks=labels,
                    fontsize=subplot_fontsize,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=True,
                    colorbar_ticks=colorbar_ticks,
                    colorbar_ticks_labels=colorbar_ticks_labels,
                    colorbar_label=colorbar_label,
                    rotation=45,
                    save_dir=save_dir,
                    as_pdf=as_pdf,
                    ax=axe,
                    fig=fig,
                    show=False,
                    add_line=add_line,
                )

        Vizualizer.end_XD_subplot(
            plot_show=plot_show,
            fig=fig,
            title=title,
            additional_title=additional_title,
            as_pdf=as_pdf,
            save_dir=save_dir,
        )
        return ax

    def plot_mean_std_heatmap(
        data: Dict[str, np.ndarray],
        ax: Optional[plt.Axes] = None,
        first_direction: Literal["x", "y"] = "x",
        colorbar_label: str = "",
        title: str = "Heatmap",
        additional_title: str = "",
        xlabel: str = "Iterations",
        ylabel: str = "",
        vmin_mean: float = None,
        vmax_mean: float = None,
        vmin_std: float = None,
        vmax_std: float = None,
        xticks: List[int] = None,
        yticks: List[int] = None,
        xtick_pos: Optional[list] = None,
        ytick_pos: Optional[list] = None,
        figsize=(10, 6),
        save_dir: str = None,
        as_pdf: bool = False,
        cmap: str = "viridis",
        regenerate: bool = False,
    ):
        """
        Plot multiple heatmaps one per condition with multiple source with different tasks as subgroups.
        """
        fig, ax, plot_show, ncols, nrows = Vizualizer.prep_XD_subplot_template(
            data=data,
            first_direction=first_direction,
            ax=ax,
            figsize=figsize,
        )
        save_path = Vizualizer.create_save_path(
            save_dir=save_dir, title=title, format="pdf" if as_pdf else "png"
        )
        if save_path is not None and Path(save_path).exists() and not regenerate:
            # load existing picture from save_path
            Vizualizer.plot_image(figsize=figsize, save_path=save_path, show=True)
        else:
            for cond_num, (condition, cond_data) in enumerate(data.items()):
                subplot_title = f"{condition.upper()}"  # {additional_title}"
                axe = ax[cond_num] if ncols > 1 else ax
                plot_colorbar_label = f"{colorbar_label} {condition.split(' ')[0]}"
                if "mean" in condition:
                    vmin = vmin_mean
                    vmax = vmax_mean
                elif "std" in condition:
                    vmin = vmin_std
                    vmax = vmax_std
                else:
                    vmin = None
                    vmax = None
                Vizualizer.plot_heatmap(
                    ax=axe,
                    data=cond_data,
                    # additional_title=additional_title,
                    figsize=figsize,
                    title=subplot_title,
                    xticks=xticks,
                    yticks=yticks,
                    vmin=vmin,
                    vmax=vmax,
                    xticks_pos=xtick_pos,
                    yticks_pos=ytick_pos,
                    colorbar_label=plot_colorbar_label,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    save_dir=save_dir,
                    as_pdf=as_pdf,
                    cmap=cmap,
                )

            Vizualizer.end_XD_subplot(
                plot_show=plot_show,
                fig=fig,
                title=title,
                additional_title=additional_title,
                as_pdf=as_pdf,
                save_dir=save_dir,
            )

    @staticmethod
    def density(
        data: Dict[str, Dict[str, Dict[str, Union[float, int, List, np.ndarray]]]],
        space: str = "2d",
        filter_outlier: bool = False,
        outlier_threshold: float = 0.2,
        additional_title: str = "",
        save_dir: str = None,
        plot_legend: bool = False,
        use_alpha=True,
        regenerate: bool = False,
    ):
        if space == "2d":
            Vizualizer.plot_2d_group_scatter(
                data,
                additional_title=additional_title,
                plot_legend=plot_legend,
                use_alpha=use_alpha,
                filter_outlier=filter_outlier,
                outlier_threshold=outlier_threshold,
                save_dir=save_dir,
                same_subplot_range=True,
                regenerate=regenerate,
            )
        elif space == "3d":
            Vizualizer.plot_3D_group_scatter(
                data,
                additional_title=additional_title,
                plot_legend=plot_legend,
                use_alpha=use_alpha,
                filter_outlier=filter_outlier,
                outlier_threshold=outlier_threshold,
                save_dir=save_dir,
                regenerate=regenerate,
            )


def plot_line(
    values: list,
    xlabel: str = None,
    ylabel: str = None,
    label: Optional[str] = None,
    markers: Optional[list] = None,
    title: Optional[str] = None,
    std: Optional[list] = None,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_plot: bool = True,
    figsize=(8, 6),
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    xticks: Optional[list] = None,
    yticks: Optional[list] = None,
    xtick_pos: Optional[list] = None,
    ytick_pos: Optional[list] = None,
    line_width: int = 2,
    grid: bool = True,
    save_dir: Optional[str] = None,
    as_pdf: bool = False,
    std_alpha: Optional[float] = 0.1,
    legend: bool = True,
):
    """
    Plots a line with optional std. Can be used as a standalone function or with an existing axis.

    Args:
        values: Y-axis values.
        label: Label for the line.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.
        std: Standard deviation values for shading.
        color: Color of the line (optional).
        ax: Axis to plot on (optional). If not provided, a new figure and axis will be created.
        show_plot: Whether to display the plot immediately (default: True). Set to False if integrating into a larger plot.
    """
    # Create a new figure and axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = range(len(values))
    ax.plot(x, values, label=label, color=color, linewidth=line_width)
    if std is not None:
        ax.fill_between(
            x,
            np.array(values) - np.array(std),
            np.array(values) + np.array(std),
            color=color,
            alpha=std_alpha,
        )
    if markers is not None:
        if not isinstance(markers, list) or not all(
            isinstance(m, str) for m in markers
        ):
            do_critical(
                TypeError,
                "markers must be a list of strings, but got {markers} of type {type(markers)}",
            )
        if len(markers) == 1:
            markers = markers * len(values)  # repeat marker for each value
        elif len(markers) != len(values):
            do_critical(
                ValueError,
                f"markers must have the same length as values, but got {len(markers)} and {len(values)}",
            )

        unique_markers = list(set(markers))
        for marker in unique_markers:
            if marker == "":
                continue
            label = f"{marker}: {markers_meaning[marker]}"
            idx = [i for i, m in enumerate(markers) if m == marker]
            if len(idx) == 0:
                continue
            # Plot markers at the positions of the values
            x_pos = np.array(x)[idx]
            y_pos = np.array(values)[idx]
            ax.plot(
                x_pos,
                y_pos,
                marker=marker,
                markersize=10,
                color=color,
                linestyle="None",
                label=label,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()

    if xticks is not None:
        if xtick_pos is not None:
            ax.set_xticks(xtick_pos, xticks, rotation=45, ha="right")
        else:
            ax.set_xticklabels(xticks, rotation=45, ha="right")
    if yticks:
        if ytick_pos:
            ax.set_yticks(ytick_pos, yticks)
        else:
            ax.set_yticks(yticks)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(alpha=0.2)

    if save_dir:
        ext = "pdf" if as_pdf else "png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_dir}/{title}.{ext}", bbox_inches="tight")

    # Show the plot if no axis was provided and show_plot is True
    if ax is None and show_plot:
        plt.tight_layout()
        plt.show()


def plot_simple_embedd(
    coords: np.ndarray,
    ax: Optional[go.Figure] = None,
    title: str = "",
    additional_title: str = "",
    plot_df: Optional[pd.DataFrame] = None,
    labels: Optional[Union[List[str], pd.Series]] = None,
    colors: Optional[np.ndarray] = None,
    figsize: tuple = (8, 8),
    plot: str = "samples",
    dot_size: int = 100,
    alpha: float = 0.8,
    cmap: Optional[str] = None,
    save_dir: Optional[str] = None,
    add_cmap: bool = False,
    legend: bool = True,
):
    """Visualizes 2D or 3D embedded coordinates using an interactive Plotly scatter plot.

    This function creates a versatile visualization for high-dimensional data embeddings.
    It can render individual data points, group centroids with standard deviation
    ellipses/spheres, and flow lines that show transitions between different states or
    time points, either for individual samples or for group averages.

    Parameters
    ----------
    coords : np.ndarray
        A numpy array of shape (n_samples, 2) or (n_samples, 3) containing the
        embedding coordinates for each sample.
    ax : go.Figure, optional
        A Plotly Figure object to add traces to. If None, a new figure is created.
        Note: This parameter is incompatible with traditional Matplotlib axes.
    title : str, optional
        The main title for the plot.
    additional_title : str, optional
        A supplementary title string appended to the main title.
    plot_df : pd.DataFrame, optional
        A DataFrame containing metadata for each sample, indexed consistently with `coords`.
        This DataFrame is crucial for grouping, coloring, and enabling advanced plot types
        like flows. See the section below for required column names.
    labels : list or pd.Series, optional
        Labels for each data point. This is overridden by the 'plot_label' column in
        `plot_df` if provided.
    colors : np.ndarray, optional
        An array of colors for each data point. This is overridden by the 'color' column
        in `plot_df` if provided.
    figsize : tuple, optional
        The figure size in inches, which is scaled to (width*100, height*100) pixels.
    plot : str, optional
        A string specifying the plot type(s) to render. Multiple types can be combined
        using a comma separator (e.g., 'samples,center_std,center_flow').
        Available options:
        - 'samples' : Plots each individual data point.
        - 'center' : Plots the centroid (mean) of each group defined by 'plot_label'.
        - 'center_std' : Draws a shaded ellipse (2D) or sphere (3D) representing one
          standard deviation around each group centroid.
        - 'center_flow' : Draws arrows connecting the centroids of sequentially numbered tasks.
          Requires 'number' and 'previous_task_idx' columns in `plot_df`.
        - 'sample_flow' : Draws arrows connecting individual data points across tasks.
          Requires 'previous_task_idx' and 'next_task_idx' columns in `plot_df`.
        - 'self_center_flow': Draws arrows connecting group centroids in a trajectory,
          normalized to start at the origin. Requires 'group_task_number'.
        - 'self_sample_flow': Draws arrows connecting individual samples in trajectories,
          normalized to start at the origin. Requires 'group_task_number' and 'animal'.
        - 'annotate_dots' : Adds text labels next to the plotted points.
    dot_size : int, optional
        The base size of the markers in the scatter plot. Note that this is scaled
        differently for samples, centers, and flow markers.
    alpha : float, optional
        The opacity level for the sample markers, between 0 (transparent) and 1 (opaque).
    cmap : str, optional
        The name of the Matplotlib colormap to use for coloring points if `colors` or
        `plot_df['color']` are not provided.
    save_dir : str, optional
        If specified, the directory where the plot will be saved as an HTML file.
    add_cmap : bool, optional
        This parameter is not implemented for Plotly and will raise an error if True.
    legend : bool, optional
        If True, the plot legend is displayed.

    `plot_df` DataFrame Columns
    ---------------------------
    The `plot_df` DataFrame controls most of the function's advanced features. Its index
    must align with the `coords` array.

    **Required Columns:**
    - `group_key` (str or int): An identifier used to group samples. All rows with the same
      `group_key` are treated as a single experimental unit or trajectory, especially for
      'self_flow' plots.
    - `plot_label` (str): The primary label for each data point, used for coloring,
      grouping for centroids, and legend entries.

    **Optional & Conditionally Required Columns:**
    - `color` (str or tuple): A specific color for each sample (e.g., '#ff0000', 'red').
      If provided, this overrides the default color scheme.
    - `animal` (str): An identifier for individual subjects/animals. Required for
      'self_sample_flow' to pair samples across time steps. Also used to enrich labels.
    - `number` (int): A sequential number that defines the order of tasks or states within a
      group. Required for `'center_flow'`.
    - `group_task_number` (int): A sequential number for tasks within a `group_key`.
      Required for `'self_center_flow'` and `'self_sample_flow'`.
    - `previous_task_idx` (int): The DataFrame index of the sample from the preceding task.
      Required for `'center_flow'` and `'sample_flow'`.
    - `next_task_idx` (int): The DataFrame index of the sample in the subsequent task.
      Required for `'sample_flow'`.

    Returns
    -------
    tuple
        A tuple containing (`traces`, `layout`), which are lists of Plotly graph objects
        and a layout dictionary. These can be used to construct a `go.Figure` object manually.

    Raises
    ------
    ValueError
        If the length of `colors` does not match the number of samples in `coords`.
        If `plot_df` is missing a required column for a selected `plot` type.
    NotImplementedError
        If `add_cmap` is set to True.

    Example
    -------
    >>> n_samples = 50
    >>> coords = np.random.rand(n_samples, 2) * 10
    >>> plot_df = pd.DataFrame({
    ...     'group_key': ['GroupA'] * n_samples,
    ...     'plot_label': ['Task1'] * 25 + ['Task2'] * 25,
    ...     'number': [1] * 25 + [2] * 25,
    ...     'previous_task_idx': [np.nan] * 25 + list(range(25)),
    ... })
    >>>
    >>> # Visualize samples, their centers with std, and the flow between task centers
    >>> plot_simple_embedding(
    ...     coords,
    ...     plot_df=plot_df,
    ...     title="Embedding Visualization",
    ...     plot="samples,center_std,center_flow"
    ... )
    """
    annotate_dots = "annotate_dots" in plot
    # ax = None

    # Handle labels
    if plot_df is not None and "plot_label" in plot_df.columns:
        if labels is not None:
            global_logger.warning(
                "Both labels and plot_df provided. Using plot_df for plotting."
            )
        labels = plot_df["plot_label"].values
        if "animal" in plot_df.columns:
            labels = [
                f"{animal} {label}" for animal, label in zip(plot_df["animal"], labels)
            ]
    elif labels is None:
        labels = np.arange(len(coords))
    labels = np.array(labels)

    # Handle colors
    if plot_df is not None and "color" in plot_df.columns:
        if colors is not None:
            global_logger.warning(
                "Both colors and plot_df provided. Using plot_df for colors."
            )
        sample_colors = plot_df["color"].values
    else:
        unique_labels = list(dict.fromkeys(labels))
        if colors is None:
            colors = Vizualizer._get_base_color(
                np.arange(len(unique_labels)), cmap=cmap
            )
            sample_colors = [
                colors[unique_labels.index(task_name)] for task_name in labels
            ]
        else:
            sample_colors = colors
    sample_colors = np.array(sample_colors)
    if len(sample_colors) != len(coords):
        raise ValueError(
            f"colors must have same shape as coords, got {sample_colors.shape} and {coords.shape}"
        )

    D = coords.shape[1]
    traces = []
    fig = go.Figure() if not isinstance(ax, go.Figure) else None

    # Create default plot_df if None
    if plot_df is None:
        plot_df = pd.DataFrame(
            {
                "group_key": ["all"] * len(coords),
                "plot_label": labels,
            },
            index=np.arange(len(coords)),
        )
    plot_df = plot_df.copy()
    plot_df.index = range(len(plot_df))  # ensure index is 0...N
    plot_df["color"] = [mcolors.to_hex(c) for c in sample_colors]

    def define_lines(
        start_coords,  # numpy array of coordinates (N, D)
        end_coords,  # numpy array of coordinates (N, D)
        labels,  # list of labels corresponding to lines
        color: str = "#ff0000",  # hex color for the lines
        name: str = "flow",
        legendgroup: str = "flow",
        dot_size: int = 10,
        legend: bool = True,
        scale: float = 0.9,
        opacity: float = 0.8,
        annotate_dots: bool = False,
        cone_size: float = 0.3,  # size of the arrowhead for 3D
    ):
        """
        Return a list of Plotly traces (lines + arrowheads).
        - 2D: Scatter trace with line + markers.
        - 3D: Scatter3d line + Cone traces as arrowheads.
        """
        start_coords = np.atleast_2d(start_coords)
        end_coords = np.atleast_2d(end_coords)
        D = start_coords.shape[1]

        traces = []
        x_all, y_all, z_all = [], [], []
        texts_all, hover_all = [], []

        for start_coord, end_coord, lbl in zip(start_coords, end_coords, labels):
            delta = end_coord - start_coord
            length = np.linalg.norm(delta)
            if length == 0:
                continue

            delta *= scale
            start = start_coord + delta * 0.05
            end = end_coord - delta * 0.05

            # Line points
            x_all.extend([start[0], end[0], None])
            y_all.extend([start[1], end[1], None] if D > 1 else [0, 0, None])
            if D == 3:
                z_all.extend([start[2], end[2], None])

            texts_all.extend(["", "", ""])
            hover_all.extend(["", lbl, ""])

            if D == 3:
                print(
                    "3D flow line has no driangle capabilities now. uncomment lines below to add unfinished feature"
                )
                # TODO: Add cone arrowhead

                # traces.append(go.Cone(
                #     x=[end[0]], y=[end[1]], z=[end[2]],
                #     u=[delta[0]], v=[delta[1]], w=[delta[2]],
                #     sizemode="absolute",
                #     sizeref=cone_size,
                #     showscale=False,
                #     colorscale=[[0, color], [1, color]],
                #     name=name,
                #     legendgroup=legendgroup,
                #     showlegend=False,  # legend only for the line
                #     opacity=opacity,
                # ))
                pass

        if D == 2:
            # 2D line trace
            traces.insert(
                0,
                go.Scatter(
                    x=x_all,
                    y=y_all,
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=dot_size / 5, color=color),
                    text=texts_all,
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=legend,
                    hovertext=hover_all,
                    hoverinfo="text",
                    opacity=opacity,
                ),
            )
        else:
            # 3D line trace
            traces.insert(
                0,
                go.Scatter3d(
                    x=x_all,
                    y=y_all,
                    z=z_all,
                    mode="lines",
                    line=dict(color=color, width=5),
                    text=texts_all,
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=legend,
                    hovertext=hover_all,
                    hoverinfo="text",
                    opacity=opacity,
                ),
            )

        return traces

    # Group and process data
    grouped_plot_df = plot_df.groupby("group_key")
    for group_key, group_df in grouped_plot_df:
        if "number" in group_df.columns:
            group_df = group_df.sort_values(by="number")
        else:
            group_df = group_df.sort_values(by="plot_label")

        group_labels = group_df["plot_label"].unique()
        if not group_labels.size:
            continue

        # Initialize for sample_flow batching
        color_to_data = (
            defaultdict(lambda: {"starts": [], "ends": [], "labels": []})
            if "sample_flow" in plot
            else None
        )

        # Single pass through group_labels
        for label in group_labels:
            ids = group_df.index[group_df["plot_label"] == label].tolist()
            if len(ids) == 0:
                continue
            sample_c = group_df.loc[ids, "color"].values
            hex_colors = [mcolors.to_hex(c) for c in sample_c]

            # Plot samples
            if "samples" in plot:
                x = coords[ids, 0]
                y = coords[ids, 1] if D > 1 else np.zeros_like(x)
                z = coords[ids, 2] if D == 3 else None
                scatter_kwargs = {
                    "x": x,
                    "y": y,
                    "mode": "markers",
                    "marker": dict(
                        size=dot_size / 10,
                        color=hex_colors,
                        opacity=alpha,
                    ),
                    "name": str(label),
                    "text": [str(label) if annotate_dots else None] * len(x),
                    "legendgroup": str(label),
                    "showlegend": legend,
                    "hovertext": [str(label)] * len(x),
                    "hoverinfo": "text",
                }
                if D == 3:
                    scatter_kwargs["z"] = z
                    scatter = go.Scatter3d(**scatter_kwargs)
                else:
                    scatter = go.Scatter(**scatter_kwargs)
                if fig is not None:
                    fig.add_trace(scatter)
                else:
                    traces.append(scatter)

            # Center and center_std plots
            if "center" in plot or "center_std" in plot:
                centroid = np.mean(coords[ids], axis=0)
                std_dev = np.std(coords[ids], axis=0)
                x_center = centroid[0]
                y_center = centroid[1] if D > 1 else 0
                z_center = centroid[2] if D == 3 else None

                if "center" in plot:
                    scatter_kwargs = {
                        "x": [x_center],
                        "y": [y_center],
                        "mode": "markers",
                        "marker": dict(
                            size=dot_size / 5,
                            color=hex_colors[0],
                            opacity=0.8,
                        ),
                        "name": f"{label} center",
                        "text": [str(label) if annotate_dots else None],
                        "legendgroup": str(label),
                        "showlegend": legend,
                        "hovertext": str(label),
                        "hoverinfo": "text",
                    }
                    if D == 3:
                        scatter_kwargs["z"] = [z_center]
                        scatter = go.Scatter3d(**scatter_kwargs)
                    else:
                        scatter = go.Scatter(**scatter_kwargs)
                    if fig is not None:
                        fig.add_trace(scatter)
                    else:
                        traces.append(scatter)

                if "center_std" in plot and len(ids) > 1:
                    shaded_kwargs = {
                        "mode": "lines",
                        "line": dict(color=hex_colors[0], width=1),
                        "name": f"{label} std",
                        "showlegend": False,
                        "legendgroup": str(label),
                        "opacity": 0.1,
                        "hoverinfo": "skip",
                    }
                    if D != 3:
                        theta = np.linspace(0, 2 * np.pi, 100)
                        x_ellipse = x_center + std_dev[0] * np.cos(theta)
                        y_ellipse = y_center + std_dev[1] * np.sin(theta)
                        ellipse = go.Scatter(
                            x=x_ellipse,
                            y=y_ellipse,
                            fill="toself",
                            fillcolor=hex_colors[0],
                            **shaded_kwargs,
                        )
                        if fig is not None:
                            fig.add_trace(ellipse)
                        else:
                            traces.append(ellipse)
                    else:
                        u = np.linspace(0, 2 * np.pi, 20)
                        v = np.linspace(0, np.pi, 20)
                        x_sphere = x_center + std_dev[0] * np.outer(
                            np.cos(u), np.sin(v)
                        )
                        y_sphere = y_center + std_dev[1] * np.outer(
                            np.sin(u), np.sin(v)
                        )
                        z_sphere = z_center + std_dev[2] * np.outer(
                            np.ones(np.size(u)), np.cos(v)
                        )
                        sphere = go.Scatter3d(
                            x=x_sphere.flatten(),
                            y=y_sphere.flatten(),
                            z=z_sphere.flatten(),
                            **shaded_kwargs,
                        )
                        if fig is not None:
                            fig.add_trace(sphere)
                        else:
                            traces.append(sphere)

            # Center flow
            if "center_flow" in plot and "number" in group_df.columns:
                prev_ids = (
                    group_df.loc[ids, "previous_task_idx"]
                    if "previous_task_idx" in group_df.columns
                    else None
                )
                if prev_ids is None or len(prev_ids) == 0 or prev_ids.isna().all():
                    continue

                prev_labels = []
                for prev_id in prev_ids:
                    if pd.notna(prev_id) and prev_id in group_df.index:
                        prev_label = group_df.loc[prev_id, "plot_label"]
                    else:
                        prev_label = ""
                    prev_labels.append(prev_label)

                if all(pl == "" for pl in prev_labels):
                    continue

                prev_labels = np.array(prev_labels)
                nan_mask = prev_ids.isna()
                if nan_mask.all():
                    continue

                not_empty_prev_labels = [pl for pl in prev_labels if pl != ""]
                if not not_empty_prev_labels:
                    continue

                unique, counts = np.unique(not_empty_prev_labels, return_counts=True)
                unique_sorted = unique[np.argsort(-counts)]
                main_prev_label = unique_sorted[0]

                # Use full center for previous label
                prev_all_ids = group_df.index[
                    group_df["plot_label"] == main_prev_label
                ].tolist()
                if not prev_all_ids:
                    continue
                start_coords_center = np.mean(coords[prev_all_ids], axis=0)

                # Use full center for current label
                end_coords_center = np.mean(coords[ids], axis=0)

                main_label_change = f"{main_prev_label} → {label}"

                trace_args = define_lines(
                    start_coords=start_coords_center,
                    end_coords=end_coords_center,
                    labels=[main_label_change],
                    color=hex_colors[0],
                    name=main_label_change,
                    legendgroup=str(label),
                    dot_size=dot_size,
                    legend=legend,
                    annotate_dots=annotate_dots,
                )
                if trace_args is None:
                    continue

                if fig is not None:
                    for t in trace_args:  # arrows is a list of traces
                        fig.add_trace(t)
                else:
                    traces.extend(trace_args)

            # Sample flow
            if (
                "sample_flow" in plot
                and "previous_task_idx" in group_df.columns
                and "next_task_idx" in group_df.columns
            ):
                for idx in ids:
                    animal = (
                        group_df.loc[idx, "animal"]
                        if "animal" in group_df.columns
                        else ""
                    )
                    next_idx = group_df.loc[idx, "next_task_idx"]
                    color = group_df.loc[idx, "color"]
                    if pd.isnull(next_idx):
                        continue
                    next_idx = int(next_idx)
                    if next_idx not in plot_df.index:
                        continue
                    current_label = group_df.loc[idx, "plot_label"]
                    next_label = plot_df.loc[next_idx, "plot_label"]
                    arrow_label = f"{animal} {current_label} → {next_label}".strip()
                    current_coords = coords[idx]
                    next_coords = coords[next_idx]

                    delta = next_coords - current_coords
                    length = np.linalg.norm(delta)
                    if length == 0:
                        continue

                    arrow_color_hex = mcolors.to_hex(color)
                    data = color_to_data[arrow_color_hex]
                    data["starts"].append(current_coords)
                    data["ends"].append(next_coords)
                    data["labels"].append(arrow_label)
                data["main_label"] = f"{current_label} → {next_label}"

        # Add batched sample_flow arrows using define_lines
        if "sample_flow" in plot and color_to_data:
            for arrow_color_hex, data in color_to_data.items():
                if not data["starts"]:
                    continue
                start_coords = np.array(data["starts"])
                end_coords = np.array(data["ends"])
                arrow_labels = data["labels"]

                trace_args = define_lines(
                    start_coords=start_coords,
                    end_coords=end_coords,
                    labels=arrow_labels,
                    color=arrow_color_hex,
                    name=data["main_label"],
                    legendgroup=str(group_key),
                    dot_size=dot_size,
                    legend=legend,
                    opacity=0.6,
                    annotate_dots=annotate_dots,
                )

                if fig is not None:
                    for t in trace_args:  # arrows is a list of traces
                        fig.add_trace(t)
                else:
                    traces.extend(trace_args)

        if "self_sample_flow" in plot or "self_center_flow" in plot:
            if "group_task_number" not in group_df.columns:
                raise ValueError(
                    "For self_sample_flow or self_center_flow, plot_df must have 'group_task_number' column."
                )
            num_grouped = group_df.groupby("group_task_number")
            sorted_nums = sorted(num_grouped.groups.keys())

            # Handle self_sample_flow: connect individual samples between consecutive tasks, normalized to start at origin
            if "self_sample_flow" in plot:
                if "animal" not in group_df.columns:
                    global_logger.warning(
                        "For 'self_sample_flow', 'animal' column is required for pairing. Skipping this plot."
                    )
                elif not sorted_nums:
                    continue
                else:
                    # Create a map from each animal to its starting coordinate in the first task
                    first_task_df = num_grouped.get_group(sorted_nums[0])
                    ref_coords_map = {
                        row.animal: coords[idx] for idx, row in first_task_df.iterrows()
                    }

                    # Batch data per animal to create a unique legend group for each trajectory
                    animal_to_data = defaultdict(
                        lambda: {"starts": [], "ends": [], "labels": [], "color": None}
                    )

                    for i in range(len(sorted_nums) - 1):
                        current_df = num_grouped.get_group(sorted_nums[i]).reset_index()
                        next_df = num_grouped.get_group(
                            sorted_nums[i + 1]
                        ).reset_index()
                        merged_df = pd.merge(
                            current_df,
                            next_df,
                            on="animal",
                            suffixes=("_start", "_end"),
                        )

                        for _, row in merged_df.iterrows():
                            animal = row["animal"]
                            ref_coord = ref_coords_map.get(animal)
                            if ref_coord is None:
                                continue  # Skip animal if it has no starting point in the first task

                            start_coord_rel = coords[row["index_start"]] - ref_coord
                            end_coord_rel = coords[row["index_end"]] - ref_coord

                            if np.array_equal(start_coord_rel, end_coord_rel):
                                continue

                            # Populate the per-animal data dictionary
                            data = animal_to_data[animal]
                            data["starts"].append(start_coord_rel)
                            data["ends"].append(end_coord_rel)
                            data["labels"].append(
                                f"{animal} {row['plot_label_start']} → {row['plot_label_end']}"
                            )
                            if data["color"] is None:
                                data["color"] = mcolors.to_hex(row["color_start"])

                    # Iterate over each animal's data to create a separate, legend-grouped trace
                    for animal, data in animal_to_data.items():
                        if not data["starts"]:
                            continue
                        trace_args = define_lines(
                            start_coords=np.array(data["starts"]),
                            end_coords=np.array(data["ends"]),
                            labels=data["labels"],
                            color=data["color"],
                            name=f"{group_key}: {animal}",  # This text appears in the legend
                            legendgroup=f"{group_key}_{animal}",  # This groups traces for toggling
                            dot_size=dot_size,
                            legend=legend,
                            opacity=0.7,
                            annotate_dots=annotate_dots,
                        )

                        if fig is not None:
                            for t in trace_args:  # arrows is a list of traces
                                fig.add_trace(t)
                        else:
                            traces.extend(trace_args)

            # Handle self_center_flow: connect group centers between consecutive tasks
            if "self_center_flow" in plot:
                if not sorted_nums:
                    continue
                group_min_num = sorted_nums[0]
                has_proxy = group_min_num > 0
                if has_proxy:
                    prev_df = plot_df[plot_df["group_task_number"] == group_min_num - 1]
                    if prev_df.empty:
                        has_proxy = False
                        ref_coord = np.mean(
                            coords[
                                group_df[
                                    group_df["group_task_number"] == group_min_num
                                ].index
                            ],
                            axis=0,
                        )
                    else:
                        ref_coord = np.mean(coords[prev_df.index], axis=0)
                else:
                    ref_coord = np.mean(
                        coords[
                            group_df[
                                group_df["group_task_number"] == group_min_num
                            ].index
                        ],
                        axis=0,
                    )
                centers, task_names, colors_per_num = [], [], []
                for num in sorted_nums:
                    num_df = num_grouped.get_group(num)
                    centers.append(np.mean(coords[num_df.index.tolist()], axis=0))
                    task_names.append(num_df["plot_label"].iloc[0])
                    colors_per_num.append(num_df["color"].iloc[0])
                centers = np.array(centers)
                rel_centers = centers - ref_coord
                if has_proxy:
                    traj_coords = np.vstack([np.zeros(D), rel_centers])
                    line_labels = [f"avg_{group_min_num-1} → {task_names[0]}"] + [
                        f"{task_names[i]} → {task_names[i+1]}"
                        for i in range(len(task_names) - 1)
                    ]
                    segment_colors = [mcolors.to_hex(colors_per_num[0])] + [
                        mcolors.to_hex(colors_per_num[i])
                        for i in range(len(task_names) - 1)
                    ]
                else:
                    traj_coords = rel_centers
                    line_labels = [
                        f"{task_names[i]} → {task_names[i+1]}"
                        for i in range(len(task_names) - 1)
                    ]
                    segment_colors = [
                        mcolors.to_hex(colors_per_num[i])
                        for i in range(len(task_names) - 1)
                    ]
                if len(traj_coords) < 2:
                    continue
                starts, ends = traj_coords[:-1], traj_coords[1:]
                for s in range(len(starts)):
                    is_proxy = has_proxy and s == 0
                    trace_args = define_lines(
                        start_coords=starts[s],
                        end_coords=ends[s],
                        labels=[line_labels[s]],
                        color=segment_colors[s],
                        name=f"{group_key}",
                        legendgroup=group_key,
                        legend=legend and s == 0,
                        dot_size=dot_size,
                        scale=0.9,
                        opacity=0.6 if is_proxy else 0.8,
                        annotate_dots=annotate_dots,
                    )
                    if is_proxy:
                        trace_args["line"]["dash"] = "dot"
                    if trace_args is None:
                        continue

                    if fig is not None:
                        for t in trace_args:  # arrows is a list of traces
                            fig.add_trace(t)
                    else:
                        traces.extend(trace_args)

    if additional_title and additional_title != "":
        plot_title = title + f"\n{additional_title}"
    else:
        plot_title = title
    plot_title = plot_title.replace("\n", "<br>")  # for HTML line breaks
    layout = dict(
        title=plot_title,
        width=figsize[0] * 100,
        height=figsize[1] * 100,
        showlegend=legend,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2" if D > 1 else "",
        scene=(
            dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
            )
            if D == 3
            else None
        ),
    )

    if isinstance(add_cmap, bool) and add_cmap:
        # For bool True, implement as needed (e.g., default colormap); placeholder removed
        pass  # Or add default colormap logic here
    elif isinstance(add_cmap, np.ndarray):
        # Add 2D colormap as legend to Plotly based on 3D numpy array (e.g., 16, 16, 3)
        source = plot_2d_colormap(add_cmap, backend="plotly", return_source=True)
        fig.add_layout_image(
            dict(
                source=source,
                xref="paper",
                yref="paper",
                x=0.9,
                y=1,
                sizex=0.2,
                sizey=0.2,
                xanchor="left",
                yanchor="top",
            )
        )

    if fig is not None:
        fig.update_layout(**layout)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{clean_filename(title)}.html")
            fig.write_html(save_path)
        fig.show()

    return traces, layout

    # else:
    #     # Matplotlib implementation (unchanged)
    #     if ax is None:
    #         fig = plt.figure(figsize=figsize)
    #         if add_cmap is not None:
    #             if is_3d:
    #                 ax = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection="3d")
    #                 cax = fig.add_axes([0.75, 0.1, 0.1, 0.8])
    #             else:
    #                 ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    #                 cax = fig.add_axes([0.85, 0.1, 0.1, 0.8])
    #             plot_2d_colormap(add_cmap, ax=cax, fig=fig, title="Positional Bins")
    #         else:
    #             ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    #         fontsize = Vizualizer.auto_fontsize(fig) if fontsize is None else fontsize
    #         show = True
    #     else:
    #         fig = ax.figure
    #         fontsize = (
    #             Vizualizer.auto_fontsize(fig) * 0.5 if fontsize is None else fontsize
    #         )
    #         if add_cmap is not None:
    #             cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    #             plot_2d_colormap(add_cmap, ax=cax, fig=fig)
    #         show = False


def pca_component_variance_plot(
    data: Union[list, np.ndarray], labels: list, percentag=0.8
):
    from sklearn.decomposition import PCA

    datas = make_list_ifnot(data)
    labels = make_list_ifnot(labels)
    # discrete colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(datas)))
    plt.figure(figsize=(8, 5))
    for data, color, label in zip(datas, colors, labels):
        pca = PCA()
        pca.fit(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance > percentag)
        plt.plot(cumulative_variance, color=color, label=f"{label}: {n_components}")
        plt.scatter(n_components, percentag, color=color)
        # plt.axhline(percentag, color=color, linestyle="--", alpha=0.3)
        plt.axvline(n_components, color=color, linestyle="--", alpha=0.3)

    plt.title("PCA Component Variance")
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance")
    plt.grid(alpha=0.1)
    plt.legend()
    plt.show()


# plotly


def prep_subplot_template(
    data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]] = None,
    first_direction: Literal["x", "y"] = "x",
    ax: Optional[go.Figure] = None,
    figsize: Tuple[int, int] = (6, 6),
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    sharex: bool = True,
    sharey: bool = True,
) -> Tuple[
    go.Figure,
    bool,
    int,
    int,
    Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]],
]:
    """
    Prepare a Plotly subplot template for 1D or 2D subplots.

    Args:
        data: Nested dictionary of DataFrames or single-level dictionary.
        first_direction: Direction for subplot arrangement ("x" or "y").
        ax: Existing Plotly figure to use. If None, a new figure is created.
        figsize: Size of the figure (width, height) in pixels (base size per subplot).
        nrows: Number of rows in the subplot grid (optional).
        ncols: Number of columns in the subplot grid (optional).
        sharex: Share x-axis across subplots.
        sharey: Share y-axis across subplots.

    Returns:
        Tuple containing:
            - fig: Plotly figure.
            - show_plot: Whether to show the plot.
            - ncols: Number of columns.
            - nrows: Number of rows.
            - data: Processed data dictionary.

    Raises:
        TypeError: If data is not a dictionary.
        ValueError: If data or nrows/ncols are not provided or mismatched.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary for plotting subplots")

    if data is None and (nrows is None or ncols is None):
        raise ValueError("Either data or nrows and ncols must be provided.")

    # Determine dimensions based on data or provided nrows/ncols
    if data is not None:
        if nrows is not None or ncols is not None:
            global_logger.debug(
                "data and nrows/ncols are both provided. Using nrows/ncols as input."
            )
        # Check if data is nested (Dict[str, Dict[str, pd.DataFrame]]) or single-level (Dict[str, pd.DataFrame])
        is_nested = all(isinstance(v, dict) for v in data.values())
        len_first_level = len(data)
        if is_nested:
            # For nested data, use first level for ncols and second level for nrows (if x direction)
            first_element = data[list(data.keys())[0]]
            len_second_level = (
                len(first_element) if isinstance(first_element, dict) else 1
            )
            ncols = ncols or len_first_level
            nrows = nrows or len_second_level
            if first_direction == "y":
                nrows, ncols = ncols, nrows
        else:
            # For single-level data, use one row (or column) with ncols as len(data)
            ncols = ncols or len_first_level
            nrows = nrows or 1
            if first_direction == "y":
                nrows, ncols = ncols, nrows
    elif nrows is None or ncols is None:
        raise ValueError("nrows and ncols must be provided if data is None.")

    # Adjust figure size based on number of subplots (scale base figsize)
    plot_figsize = (
        figsize[0] * ncols * 100,
        figsize[1] * nrows * 100,
    )  # Convert to pixels (Plotly uses pixels)

    # Prepare data for consistent processing
    processed_data = data if is_nested else {"": data}

    if ax is None:
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_xaxes=sharex,
            shared_yaxes=sharey,
            subplot_titles=[cond for cond in processed_data.keys()],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )
        show_plot = True
        # Set figure size
        fig.update_layout(
            width=plot_figsize[0],
            height=plot_figsize[1],
        )
    else:
        fig = ax
        show_plot = False
        # Validate provided figure’s subplot structure
        fig_layout = fig.layout
        existing_rows = max(
            [
                int(k.replace("yaxis", "") or 1)
                for k in fig_layout
                if k.startswith("yaxis")
            ]
        )
        existing_cols = max(
            [
                int(k.replace("xaxis", "") or 1)
                for k in fig_layout
                if k.startswith("xaxis")
            ]
        )
        if existing_rows < nrows or existing_cols < ncols:
            raise ValueError(
                f"Provided figure has {existing_rows}x{existing_cols} subplots, but {nrows}x{ncols} are required."
            )

    return fig, show_plot, ncols, nrows, processed_data


def end_subplot(
    fig: go.Figure,
    plot_show: bool,
    title: str = "",
    additional_title: str = "",
    save_dir: Optional[str] = None,
    as_pdf: bool = False,
    theme: Literal[
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ] = "plotly_dark",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Finalize subplot with title and save/show options.
    """
    full_title = f"{title} {additional_title}".strip()
    fig.update_layout(
        title=full_title,
        showlegend=True,
        template=theme,
    )
    autoscale_axis(fig, xlim=xlim, ylim=ylim)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ext = "pdf" if as_pdf else "html"
        (
            fig.write_image(f"{save_dir}/{full_title}.{ext}")
            if as_pdf
            else fig.write_html(f"{save_dir}/{full_title}.{ext}")
        )
    if plot_show:
        fig.show()
    return fig


def calculate_axis_range(
    data_x: List[float],
    data_y: List[float],
    pad: float,
    manual_limits: Optional[Tuple[float, float]] = None,
    cross_axis_limits: Optional[Tuple[float, float]] = None,
    is_x_axis: bool = True,
) -> List[float]:
    """
    Calculate axis range with padding, optional manual limits, and cross-axis filtering.

    Args:
        data_x: List of x data points
        data_y: List of y data points
        pad: Padding percentage for the axis
        manual_limits: Optional tuple of (min, max) limits for this axis
        cross_axis_limits: Optional tuple of (min, max) limits for the other axis
        is_x_axis: True if calculating x-axis range, False for y-axis

    Returns:
        List[float]: Calculated [min, max] range for the axis
    """
    if not data_x or not data_y:
        return [0, 1]  # Default range for empty data

    # Filter data based on cross-axis limits
    filtered_x, filtered_y = data_x, data_y
    if cross_axis_limits:
        min_limit, max_limit = cross_axis_limits
        mask = [True] * len(data_x)
        if min_limit is not None:
            mask = [v >= min_limit for v in (data_y if is_x_axis else data_x)]
        if max_limit is not None:
            mask = [
                m and v <= max_limit
                for m, v in zip(mask, (data_y if is_x_axis else data_x))
            ]
        filtered_x = [x for x, m in zip(data_x, mask) if m]
        filtered_y = [y for y, m in zip(data_y, mask) if m]

    # Use filtered data for this axis
    data = filtered_x if is_x_axis else filtered_y
    if not data:
        return [0, 1]  # Default range if no data remains after filtering

    v_min, v_max = min(data), max(data)
    data_range = v_max - v_min
    padding = pad * data_range if data_range != 0 else pad

    # Initialize range with padding
    ax_range = [v_min - padding, v_max + padding]

    # Apply manual limits if provided
    if manual_limits:
        if manual_limits[0] is not None:
            ax_range[0] = manual_limits[0]
            if manual_limits[1] is None:
                filtered_data = [x for x in data if x >= manual_limits[0]]
                ax_range[1] = (
                    max(filtered_data) + padding if filtered_data else ax_range[0] + 1
                )
        if manual_limits[1] is not None:
            ax_range[1] = manual_limits[1]
            if manual_limits[0] is None:
                filtered_data = [x for x in data if x <= manual_limits[1]]
                ax_range[0] = (
                    min(filtered_data) - padding if filtered_data else ax_range[1] - 1
                )

    return ax_range


def autoscale_axis(
    fig: go.Figure,
    xpad: float = 0.01,
    ypad: float = 0.01,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """
    Automatically scales all axes (x, y) in a Plotly figure with optional padding,
    considering manual limits of the other axis.

    Args:
        fig: The Plotly figure to modify
        xpad: Padding percentage for x-axis (default: 0.01)
        ypad: Padding percentage for y-axis (default: 0.01)
        xlim: Optional tuple of (min, max) limits for x-axis
        ylim: Optional tuple of (min, max) limits for y-axis

    Returns:
        go.Figure: The modified figure with scaled axes
    """
    # Collect data from all traces
    axis_data = {"x": [], "y": []}

    for trace in fig.data:
        x_data = getattr(trace, "x", None)
        y_data = getattr(trace, "y", None)
        if x_data is not None and y_data is not None:
            # Ensure x and y data are flattened and aligned
            x_data = np.ravel(x_data).tolist()
            y_data = np.ravel(y_data).tolist()
            if len(x_data) == len(y_data):  # Ensure data points are paired
                axis_data["x"].extend(x_data)
                axis_data["y"].extend(y_data)

    # Calculate and apply ranges
    if axis_data["x"] and axis_data["y"]:
        # Calculate x-range considering y-limits
        x_range = calculate_axis_range(
            axis_data["x"],
            axis_data["y"],
            xpad,
            manual_limits=xlim,
            cross_axis_limits=ylim,
            is_x_axis=True,
        )
        fig.update_xaxes(range=x_range)

        # Calculate y-range considering x-limits
        y_range = calculate_axis_range(
            axis_data["x"],
            axis_data["y"],
            ypad,
            manual_limits=ylim,
            cross_axis_limits=xlim,
            is_x_axis=False,
        )
        fig.update_yaxes(range=y_range)

    return fig


def plot_2d_kde_dict_of_dicts(
    data: Dict[str, Dict[str, pd.DataFrame]],
    ax: Optional[go.Figure] = None,
    first_direction: Literal["x", "y"] = "x",
    title: str = "KDE Plot",
    additional_title: str = "",
    legend_title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Density",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    bins: Optional[int] = None,
    cmap: Optional[str] = None,
    theme: Literal[
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ] = "plotly_dark",
    figsize: Tuple[int, int] = (10, 6),
    save_dir: Optional[str] = None,
    as_pdf: bool = False,
    sharex: bool = True,
    sharey: bool = True,
):
    """
    Create a 2D grid of KDE subplots for nested dictionary data.
    Each condition gets a subplot, with sources plotted as KDEs.
    """
    fig, show_plot, ncols, nrows, data = prep_subplot_template(
        data=data,
        first_direction=first_direction,
        ax=ax,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )

    col = 1
    for cond_num, (condition, cond_data) in enumerate(data.items()):
        row = cond_num + 1
        plot_1d_kde_dict_of_dicts(
            data=cond_data,
            ax=fig,
            row=row,
            col=col,
            first_direction=first_direction,
            title=condition,
            additional_title=additional_title,
            legend_title=legend_title,
            xlabel=xlabel,
            ylabel=ylabel,
            bins=bins,
            cmap=cmap,
            theme=theme,
            figsize=figsize,
            save_dir=None,  # Save at the end
            as_pdf=False,  # Save at the end
            sharex=sharex,
            sharey=sharey,
            xlim=None,
            ylim=None,
        )

    return end_subplot(
        fig=fig,
        plot_show=show_plot,
        title=title,
        additional_title=additional_title,
        save_dir=save_dir,
        as_pdf=as_pdf,
        theme=theme,
        xlim=xlim,
        ylim=ylim,
    )


def plot_1d_kde_dict_of_dicts(
    data: Dict[str, pd.DataFrame],
    ax: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
    title: str = "KDE Plot",
    additional_title: str = "",
    first_direction: Literal["x", "y"] = "x",
    legend_title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Density",
    bins: Optional[int] = None,
    cmap: Optional[str] = None,
    theme: Literal[
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ] = "plotly_dark",
    figsize: Tuple[int, int] = (10, 6),
    save_dir: Optional[str] = None,
    as_pdf: bool = False,
    sharex: bool = True,
    sharey: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Create a 1D arrangement of KDE plots for a dictionary of DataFrames.
    Each source is plotted as a KDE in a single subplot.
    """
    fig, show_plot, ncols, nrows, _ = prep_subplot_template(
        data=data,
        first_direction=first_direction,
        ax=ax,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        nrows=1,
    )
    col = col or 1
    row = row or 1

    for cond_num, (condition, df) in enumerate(data.items()):
        plotting_function = kdeplot_from_dict if isinstance(df, dict) else plot_kde
        col = cond_num + 1

        plotting_function(
            data=df,
            ax=fig,
            row=row if first_direction == "x" else col,
            col=col if first_direction == "x" else row,
            legend_title=legend_title,
            xlabel=xlabel,
            ylabel=ylabel,
            bins=bins,
            cmap=cmap,
            theme=theme,
            xlim=None,
            ylim=None,
        )

        # set subplot title
        subplot_title = f"{condition} {additional_title}".strip()
        # Calculate the x-position in paper coordinates (0 to 1)
        plot_col = col if first_direction == "x" else row
        plot_row = row if first_direction == "x" else col
        x_pos = ((plot_col - 1) + 0.5) / ncols  # Center of the subplot
        y_pos = 1 - ((plot_row - 1) / nrows)  # Slightly above the subplot
        fig.add_annotation(
            text=subplot_title,
            x=x_pos,
            y=y_pos,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=20),
        )

    if ax is None:
        return end_subplot(
            fig=fig,
            plot_show=show_plot,
            title=title,
            additional_title=additional_title,
            save_dir=save_dir,
            as_pdf=as_pdf,
            theme=theme,
            xlim=xlim,
            ylim=ylim,
        )
    return fig


def kdeplot_from_dict(
    data: Dict[str, pd.DataFrame],
    ax: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
    title: str = "KDE Plot",
    legend_title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Density",
    bins: Optional[int] = None,
    cmap: Optional[str] = None,
    theme: Literal[
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ] = "plotly_dark",
):
    """
    Plot KDEs for multiple sources within a single subplot.
    """
    if ax is None:
        fig = go.Figure()
        show_plot = True
    else:
        fig = ax
        show_plot = False

    num_sources = len(data)
    colors = Vizualizer._get_base_color(
        np.arange(num_sources), cmap=cmap, style="plotly"
    )

    # Pool all data for histogram
    all_data = np.concatenate([df.values.flatten() for df in data.values()])
    all_data = all_data[~np.isnan(all_data)]

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=all_data,
            histnorm="probability density",
            name="Overall Histogram",
            nbinsx=bins,
            opacity=0.3,
            marker=dict(color="rgba(128, 128, 128, 0.8)"),
            xaxis=f"x{col}",
            yaxis=f"y{row}",
        ),
        row=row,
        col=col,
    )

    # Add KDEs for each source
    for idx, (source, df) in enumerate(data.items()):
        for column in df.columns:
            data_values = df[column].dropna().values
            if len(data_values) == 0:
                continue
            kde = gaussian_kde(data_values)
            x_range = np.linspace(min(data_values), max(data_values), 200)
            kde_values = kde(x_range)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode="lines",
                    name=f"KDE: {source} - {column}",
                    line=dict(color=colors[idx], width=2),
                    xaxis=f"x{col}",
                    yaxis=f"y{row}",
                ),
                row=row,
                col=col,
            )

    # Update layout for the subplot
    fig.update_layout(
        title=title if ax is None else None,
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        showlegend=True,
        template=theme,
    )
    if legend_title:
        fig.update_layout(legend_title_text=legend_title)

    if show_plot:
        fig.show()
    return fig


def plot_kde(
    data: pd.DataFrame,
    ax: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
    title: str = "KDE Plot",
    xlabel: str = "Value",
    ylabel: str = "Density",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    bins: Optional[int] = None,
    theme: Literal[
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ] = "plotly_dark",
    save_dir: Optional[str] = None,
    as_pdf: bool = False,
    cmap: Optional[str] = None,
    legend_title: Optional[str] = None,
):
    """
    Plot a single KDE line with optional histogram.
    """
    if ax is None:
        fig = go.Figure()
        show_plot = True
    else:
        fig = ax
        show_plot = False

    all_data = data.dropna().values.flatten()
    if len(all_data) == 0:
        return fig

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=all_data,
            histnorm="probability density",
            name=f"Histogram",
            nbinsx=bins,
            opacity=0.3,
            marker=dict(color="rgba(128, 128, 128, 0.8)"),
            xaxis=f"x{col}",
            yaxis=f"y{row}",
        ),
        col=col,
        row=row,
    )

    # Generate KDE for each dataset (column)
    num_colors = len(data.columns)
    colors = Vizualizer._get_base_color(
        np.arange(num_colors), num_colors=num_colors, cmap=cmap, style="plotly"
    )

    # Calculate axis index and set proper range based on data
    all_min = min(all_data) if len(all_data) > 0 else 0
    all_max = max(all_data) if len(all_data) > 0 else 1
    fig.update_xaxes(
        title_text=xlabel,
        row=row,
        col=col,
        title_standoff=20,
        range=[all_min, all_max],  # Set explicit range based on data
        tickmode="auto",  # Automatic tick generation
        nticks=10,  # Suggest number of ticks
    )
    fig.update_yaxes(
        title_text=ylabel,
        row=row,
        col=col,
        title_standoff=20,
    )

    # Calculate axis index: for row=1, col=1, use xaxis1; for row=1, col=2, use xaxis2, etc.
    fig.update_xaxes(title_text=xlabel, row=row, col=col, title_standoff=20, range=xlim)
    fig.update_yaxes(title_text=ylabel, row=row, col=col, title_standoff=20, range=ylim)
    for idx, column in enumerate(data.columns):
        df = data[column].dropna().values
        if len(df) == 0:
            continue

        # Compute normalized KDE
        kde = gaussian_kde(df)
        x_range = np.linspace(min(df), max(df), 200)
        kde_values = kde(x_range)

        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                mode="lines",
                name=f"{column}",
                line=dict(color=colors[idx], width=2),
                xaxis=f"x{col}",
                yaxis=f"y{row}",
            ),
            col=col,
            row=row,
        )

    if ax is None:
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=True,
            template=theme,
        )

    if legend_title:
        fig.update_layout(legend_title_text=legend_title)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ext = "pdf" if as_pdf else "html"
        (
            fig.write_image(f"{save_dir}/{title}.{ext}")
            if as_pdf
            else fig.write_html(f"{save_dir}/{title}.{ext}")
        )
    if show_plot:
        fig.show()
    return fig


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes["x"].join(target, ax)
        if sharey:
            target._shared_axes["y"].join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(which="both", labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(which="both", labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def create_2d_colormap(
    corner_colors=[
        (1, 0, 0),  # bottom-left: red
        (0, 0, 1),  # bottom-right: blue
        (0, 1, 0),  # top-left: green
        (1, 1, 0),  # top-right: yellow
    ],
    x_bins=100,
    y_bins=100,
):
    """
    Create a 2D colormap by interpolating between four corner colors.

    Parameters:
    corner_colors : list of 4 tuples
        Colors at corners in order: bottom-left, bottom-right, top-left, top-right
        Each color is a tuple of (R, G, B) with values in [0, 1]
        Example:
            colors = [
                (1, 0, 0),  # bottom-left: red
                (0, 0, 1),  # bottom-right: blue
                (0, 1, 0),  # top-left: green
                (1, 1, 0)   # top-right: yellow
            ]
    x_bins : int
        Number of bins in x direction
    y_bins : int
        Number of bins in y direction

    Returns:
    numpy.ndarray
        2D array of shape (y_bins, x_bins, 3) containing RGB colors
    """
    # Ensure corner_colors are in correct format
    corner_colors = np.array(corner_colors, dtype=float)
    if corner_colors.shape != (4, 3):
        raise ValueError("corner_colors must be a list of 4 RGB tuples")

    # Create grid for interpolation
    x = np.linspace(0, 1, x_bins)
    y = np.linspace(0, 1, y_bins)
    X, Y = np.meshgrid(x, y)

    # Initialize output array
    colormap = np.zeros((y_bins, x_bins, 3))

    # Bilinear interpolation
    for i in range(3):  # For R, G, B channels
        # Interpolate horizontally first
        bottom = (1 - X) * corner_colors[0, i] + X * corner_colors[1, i]
        top = (1 - X) * corner_colors[2, i] + X * corner_colors[3, i]
        # Then interpolate vertically
        colormap[:, :, i] = (1 - Y) * bottom + Y * top

    return colormap


import io
import base64
from PIL import Image


def plot_2d_colormap(
    colormap,
    fig=None,
    ax=None,
    title="",
    position=None,
    backend="matplotlib",
    return_source=False,
):
    """
    Plot a 2D colormap as a separate axes in the figure (Matplotlib) or return a base64 source for Plotly.

    Parameters:
    colormap : numpy.ndarray
        2D array of shape (height, width, 3) containing RGB colors (uint8 assumed for Plotly).
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure, optional
        Figure to add to. For Plotly backend, ignored unless adding image (future extension).
    ax : matplotlib.axes.Axes, optional
        For Matplotlib backend only.
    title : str
        Title (used in Matplotlib; for Plotly, add as annotation separately if needed).
    position : list, optional
        For Matplotlib backend only.
    backend : str, optional
        'matplotlib' (default) or 'plotly'.
    return_source : bool, optional
        For Plotly backend: if True, return base64 source string; else, return None (placeholder for future).

    Returns:
    matplotlib.axes.Axes or str
        Axes for Matplotlib; base64 data URI for Plotly if return_source=True.
    """
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        if ax is None:
            if fig is None:
                fig = plt.gcf()

            # Default position for colormap (right side of figure)
            if position is None:
                position = [0.85, 0.1, 0.1, 0.35]  # [left, bottom, width, height]

            # Create a new axes for the colormap
            ax_cmap = fig.add_axes(position)
        else:
            ax_cmap = ax

        # Display the colormap
        ax_cmap.imshow(colormap, origin="lower")

        # Remove axes ticks
        ax_cmap.set_xticks([])
        ax_cmap.set_yticks([])

        # Set title
        ax_cmap.set_title(title, fontsize=8)

        return ax_cmap

    elif backend == "plotly":
        # Convert numpy RGB array to base64 PNG data URI
        if colormap.dtype != np.uint8:
            # If it's float, scale to 0-255 then convert
            if np.issubdtype(colormap.dtype, np.floating):
                colormap = (colormap * 255).clip(0, 255).astype(np.uint8)
            else:
                colormap = colormap.astype(np.uint8)
        img = Image.fromarray(colormap.astype("uint8"))  # Ensure uint8 for RGB
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        source = f"data:image/png;base64,{img_str}"

        if return_source:
            return source
        # For non-source return, could add to fig layout here if fig provided, but keep simple
        return source  # Default return for consistency

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def is_continuous_colormap(name: str) -> Optional[bool]:
    """
    Check if a colormap is continuous in Matplotlib or Plotly.

    Args:
        name (str): Name of the colormap to check.

    Returns:
        Optional[bool]: True if the colormap is continuous, False if discrete,
                        None if the colormap is not found.

    Examples:
        >>> is_continuous_colormap('viridis')  # Matplotlib continuous colormap
        True
        >>> is_continuous_colormap('Set1')     # Matplotlib discrete colormap
        False
        >>> is_continuous_colormap('Plotly')   # Plotly continuous colormap
        True
        >>> is_continuous_colormap('invalid')  # Non-existent colormap
        None
    """
    # Check Matplotlib colormaps
    try:
        cmap = cm.get_cmap(name)
        if isinstance(cmap, mcolors.LinearSegmentedColormap):
            return True
        elif isinstance(cmap, mcolors.ListedColormap):
            return False
    except (ValueError, AttributeError):
        # Handle cases where the colormap is not found in Matplotlib
        pass

    # Check Plotly colormaps
    if name in pcolors.named_colorscales:
        # Plotly colormaps are continuous if they are in sequential or diverging scales
        # and discrete if in qualitative scales
        sequential = pcolors.sequential.__dict__.get(name)
        diverging = pcolors.diverging.__dict__.get(name)
        qualitative = pcolors.qualitative.__dict__.get(name)

        if sequential or diverging:
            return True
        elif qualitative:
            return False

    # Return None if colormap is not found
    return None


def get_plotly_colors(
    n_colors: int, cmap_name: str = None, reverse: bool = False
) -> List[Tuple[float, float, float]]:
    """
    Get discrete RGB values from a Plotly qualitative colormap.
    """
    if cmap_name is None:
        if n_colors <= 10:
            cmap_name = "Plotly"
        elif n_colors <= 24:
            cmap_name = "Light24"
        else:
            cmap_name = "Alphabet"

    # Get the colormap list directly from plotly
    try:
        color_list = getattr(px.colors.qualitative, cmap_name)
    except AttributeError:
        raise ValueError(f"'{cmap_name}' is not a valid Plotly qualitative colormap")

    if reverse:
        color_list = list(reversed(color_list))

    # Cycle through colors if not enough
    if n_colors > len(color_list):
        from itertools import cycle, islice

        color_list = list(islice(cycle(color_list), n_colors))
    else:
        color_list = color_list[:n_colors]

    return color_list


def hex_to_rgba_str(hex_color: str, alpha: float = 0.2) -> str:
    rgb = mcolors.to_rgb(hex_color)
    return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"


def linebar_df_group_plot(
    df: pd.DataFrame,
    value_col: str,
    compare_by: str,
    compare_by_filter: List[str] = None,
    groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
    group_by: str = None,
    type: Literal["line", "bar"] = "line",
    fit_line: Literal["linear"] = None,
    show_std: bool = True,
    title: str = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    template: Literal[
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ] = "plotly",
) -> pd.DataFrame:
    """
    Plot values from DataFrame using Plotly, optionally grouped by a specific column.

    Parameters:
        - df (pd.DataFrame): DataFrame containing the data to plot.
        - value_col (str): Column name in df to plot on the y-axis.
        - compare_by (str): Column name to compare by, plotted on the x-axis.
        - compare_by_filter (List[str], optional): List of values to filter compare_by column.
        - groups (dict, optional): Dictionary defining groups to plot.
        - group_by (str, optional): Column name to group by. If None, plots all values.
        - type (str, optional): Type of plot ('line' or 'bar'). Default is 'line'.
        - fit_line (str, optional): Type of fit line ('linear'). Default is None.
        - show_std (bool, optional): Whether to show standard deviation. Default is True.
        - title (str, optional): Title of the plot.
        - save_dir (str or Path, optional): Directory to save the plot as HTML. If None, does not save.
        - show (bool, optional): Whether to display the plot. Default is True.

    Returns:
        - grouped_df (pd.DataFrame): DataFrame with grouped values if group_by is specified.
    """
    # make plotly darkmode
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")

    # Filter the DataFrame based on compare_by_filter
    if compare_by_filter is not None:
        df = df[df[compare_by].isin(compare_by_filter)]

    # Handle grouping
    if groups is None:
        if group_by is None:
            grouped_df = df
        else:
            grouped_df = df.groupby(group_by)
    else:
        if group_by is None:
            raise ValueError(
                "If groups are provided, group_by must be specified, and group_by should have the same keys in the first layer as groups"
            )
        # Assume group_df_by_custom_groups is defined elsewhere
        grouped_df = group_df_by_custom_groups(df, groups, group_by, compare_by)

    # Set default title
    if title is None:
        title = (
            f"{value_col} by {compare_by} grouped by {group_by if group_by else 'all'}"
        )

    # Create Plotly figure
    fig = go.Figure()

    max_num_values = 0
    if group_by is None:
        # Plot all values without grouping
        means, stds = compute_statistics(group_df, value_col, compare_by)
        num_values = np.arange(len(means), dtype=int) + 1
        y_upper = means + stds
        y_lower = means - stds
        max_num_values = len(df[compare_by].unique())
        if type == "line":
            add_line_trace(
                fig,
                x=np.arange(len(means)) + 1,
                means=means,
                stds=stds,
                color="blue",
                group_name="All",
                show_std=show_std,
                fit_line=fit_line,
            )
        elif type == "bar":
            add_bar_trace(
                fig,
                x=np.arange(len(means)) + 1,
                means=means,
                stds=stds,
                group_name="All",
                show_std=show_std,
            )
    else:
        # Plot grouped values
        colors = get_plotly_colors(n_colors=len(grouped_df))
        for i, (group_name, group_df) in enumerate(grouped_df):
            # Compute means and standard deviations for each group
            means, stds = compute_statistics(group_df, value_col, compare_by)
            num_values = np.arange(len(means), dtype=int) + 1
            max_num = len(num_values)
            max_num_values = max(max_num_values, max_num)
            line_color = colors[i]
            if type == "line":
                # Add line trace with optional standard deviation fill
                add_line_trace(
                    fig,
                    x=num_values,
                    means=means,
                    stds=stds,
                    color=line_color,
                    group_name=group_name,
                    show_std=show_std,
                    fit_line=fit_line,
                )
            elif type == "bar":
                # Add bar trace with optional error bars
                add_bar_trace(
                    fig,
                    x=num_values,
                    means=means,
                    stds=stds,
                    group_name=group_name,
                    show_std=show_std,
                )

    # Update layout
    base_pixels = 600
    fig.update_layout(
        title=title,
        xaxis_title="task count",
        yaxis_title=value_col,
        xaxis=dict(
            range=[1, max_num_values],
            tickmode="array",
            tickvals=np.arange(max_num_values) + 1,
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor="left",
            yanchor="top",
            title=group_by if group_by else "All",
        ),
        showlegend=True,
        template=template,
        width=base_pixels * 2,
        height=base_pixels,
    )

    # Save the plot as HTML if save_dir is specified
    if save_dir is not None:
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        title = clean_filename(title)
        save_dir.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, file=save_dir / f"{title}.html", auto_open=False)

    # Show the plot if requested
    if show:
        fig.show()

    return grouped_df


def violin_plot(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: str,
    group_by: Optional[str] = None,
    groups: List[str] = None,
    labels: List[str] = None,
    title: str = "Violin Plot",
    additional_title: str = "",
    pvalues: Optional[Union[pd.DataFrame, Dict]] = None,
    save_dir: Optional[Union[str, Path]] = None,
):
    """
    Create box-violin plots for sample distributions in a DataFrame.
    This function generates a Plotly figure with subplots for each group (if group_by is specified),
    showing side-by-side boxplots (left) and violin plots (right) for each category in compare_by.
    Optionally performs statistical significance tests between groups.
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    pair_name_col : str
        Column name for pairing samples (used in paired statistical tests).
    compare_by : str
        Column name containing categories to compare (used for x-axis positions).
    value_col : str
        Column name containing the numerical values to plot distributions for.
    group_by : Optional[str], default=None
        Column name to group data by, creating one subplot per unique group. If None, treats the entire DataFrame as one group.
    groups : List[str], default=None
        Specific group values to include. If None and group_by is specified, uses all unique values from the group_by column.
    labels : List[str], default=None
        Labels for subplot titles. If None, uses group values as strings.
    title : str, default=""
        Main title for the plot.
    additional_title : str, default=""
        Additional text to append to the plot title.
    pvalues: Optional[Union[pd.DataFrame, Dict]] = None
        If Dict, should contain a "pvalues" key with a 2D array or DataFrame of p-values for significance annotations
        or a Dict with keys for different value_col names and values as 2D array or DataFrame of p-values.
    save_dir : Optional[Union[str, Path]], default=None
        Directory to save the plot as an HTML file. If None, does not save.
    Returns:
    -------
    None
        Displays the Plotly figure and optionally saves it to HTML.
    """
    # Get significance test results
    if pvalues is not None:
        if isinstance(pvalues, dict):
            # Case 1: Dictionary with "pvalues" key
            if "pvalues" in pvalues:
                pvalues = pvalues["pvalues"]

            # Case 2: Dictionary with keys for different value_col names
            if value_col in pvalues:
                pvalues = pvalues[value_col]
            else:
                global_logger.warning(
                    f"Warning: pvalues dict does not contain key for value_col '{value_col}'. No significance annotations will be added."
                )

            # Convert to DataFrame if it's a 2D array
            if isinstance(pvalues, (np.ndarray, list)):
                pvalues = pd.DataFrame(pvalues)

    # Handle groups
    if group_by is not None:
        if groups is None:
            groups = sorted(df[group_by].unique())
    else:
        groups = ["All"]
    num_groups = len(groups)
    unique_groups = groups  # Use groups directly since they are sorted or provided
    if labels is None:
        labels = [str(g) for g in groups]
    # Compute global min and max for y-axis range
    if group_by is not None:
        df_all = df[df[group_by].isin(groups)]
    else:
        df_all = df
    global_min = df_all[value_col].min()
    global_max = df_all[value_col].max()
    global_range = global_max - global_min

    # Handle edge case where all values are identical
    if global_range == 0:
        global_range = 1.0  # Use default range for uniform data

    # Add padding for violin plots (KDE can extend beyond data range)
    violin_padding = global_range * 0.15

    # Space for significance annotations
    bracket_delta = global_range * 0.05  # Spacing between annotation levels

    # Calculate y-axis limits with proper padding
    y_min_plot = global_min - violin_padding

    # Pre-calculate maximum number of significance pairs to determine y_max_plot
    max_sig_pairs = 0
    if pvalues is not None:
        for group_name in groups:
            if group_by is not None:
                df_group = df[df[group_by] == group_name]
            else:
                df_group = df
            df_group = df_group.sort_values(
                by=[group_by, "task_number"] if group_by else ["task_number"]
            )
            unique_tasks = df_group[compare_by].drop_duplicates().tolist()

            if isinstance(pvalues, dict):
                group_pvalues = pvalues.get(group_name, None)
            else:
                group_pvalues = pvalues

            if group_pvalues is not None and isinstance(group_pvalues, pd.DataFrame):
                sig_count = 0
                for task1, task2 in itertools.combinations(unique_tasks, 2):
                    try:
                        if (
                            task1 in group_pvalues.index
                            and task2 in group_pvalues.columns
                        ):
                            p = group_pvalues.loc[task1, task2]
                            if not pd.isna(p) and p < 0.05:
                                sig_count += 1
                    except KeyError:
                        continue
                max_sig_pairs = max(max_sig_pairs, sig_count)

    # Calculate y_max_plot based on maximum number of significance pairs
    # Each pair needs bracket_delta space, plus extra space above the violin and for text
    if max_sig_pairs > 0:
        annot_space = (
            violin_padding + (max_sig_pairs * bracket_delta) + bracket_delta * 0.5
        )
    else:
        annot_space = violin_padding
    y_max_plot = global_max + annot_space

    # Create the plot
    fig = make_subplots(
        rows=num_groups,
        cols=1,
        subplot_titles=unique_groups,
        vertical_spacing=0.1,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for i, group_name in enumerate(groups):
        if group_by is not None:
            df_group = df[df[group_by] == group_name]
        else:
            df_group = df
        # sort df_group
        df_group = df_group.sort_values(
            by=["group_key", "condition", "task_name", "task_number"]
        )
        unique_tasks = (
            df_group[compare_by].drop_duplicates().tolist()
        )  # Ordered by appearance after sort
        for task in unique_tasks:
            y = df_group[df_group[compare_by] == task][value_col].values.flatten()
            sample_labels = df_group[df_group[compare_by] == task][
                pair_name_col
            ].values.flatten()
            colors = df_group[df_group[compare_by] == task]["color"].values.flatten()
            # check if all colors are the same
            if len(set(colors)) == 1:
                color = hex_to_rgba_str(colors[0], alpha=0.5)
                violin_line_color = hex_to_rgba_str(colors[0], alpha=1.0)
            else:
                print(
                    f"Warning: Multiple colors found for task {task} in group {group_name}. Using default color."
                )
                color = None
                violin_line_color = None
            # Add violin plot with improved settings
            fig.add_trace(
                go.Violin(
                    y=y,
                    x=[task] * len(y),
                    text=sample_labels,
                    name=f"{task} violin",
                    side="positive",
                    hovertemplate=(
                        f"{pair_name_col} " + ": %{text}<br>"
                        f"{value_col}: " + "%{y}<extra></extra>"
                    ),
                    marker=dict(
                        color=color,
                        line=dict(
                            color="black",
                            width=1,  # Thin black circle outline around each point
                        ),
                        # size=6,  # If points are still hard to see, uncomment and adjust
                    ),
                    # Remove these problematic parameters
                    alignmentgroup=True,  # Not needed for single violin per category
                    # offsetgroup=1,        # Not needed, was causing spacing issues
                    width=0.8,  # Let Plotly auto-adjust width
                    # Point settings - closer to violin
                    points="all",
                    pointpos=-0.1,  # Center points on violin (0), negative = left, positive = right
                    jitter=0.1,  # Reduce jitter to keep points closer
                    # Box settings - make more visible
                    box_visible=True,
                    box=dict(
                        visible=True,
                        width=0.8,  # Make box wider (relative to violin width)
                        fillcolor="rgba(255,255,255,0.8)",  # Semi-transparent white
                        line=dict(
                            color="rgba(0,0,0,0.8)", width=2
                        ),  # Thicker, darker lines
                    ),
                    # Mean line settings
                    meanline_visible=True,
                    meanline=dict(
                        visible=True,
                        color="red",
                        width=3,  # Make mean line thicker
                    ),
                    # Violin appearance
                    fillcolor=color,
                    line_color=violin_line_color,
                    opacity=0.7,  # Make violin slightly transparent so box shows through
                ),
                row=i + 1,
                col=1,
            )
        # Add significance annotations if pvalues provided
        if pvalues is not None:
            if isinstance(pvalues, dict):
                group_pvalues = pvalues.get(group_name, None)

            # Validate that pvalues is a DataFrame
            if not isinstance(group_pvalues, pd.DataFrame):
                raise ValueError(
                    f"§pvalues must be a DataFrame or convertible 2D array, got {type(group_pvalues)}"
                )
            if group_pvalues.ndim != 2:
                do_critical(
                    ValueError,
                    f"pvalues must be a 2D array or DataFrame, got {group_pvalues.ndim}D.",
                )

            positions = {task: idx for idx, task in enumerate(unique_tasks)}
            sig_pairs = []
            for task1, task2 in itertools.combinations(unique_tasks, 2):
                if (
                    task1 not in group_pvalues.index
                    or task2 not in group_pvalues.columns
                ):
                    global_logger.warning(
                        f"Tasks {task1} or {task2} not found in pvalues index/columns."
                    )
                try:
                    p = group_pvalues.loc[task1, task2]
                except KeyError:
                    continue
                if pd.isna(p) or p >= 0.05:
                    continue
                pos1 = positions[task1]
                pos2 = positions[task2]
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                else:
                    sig = "*"
                sig_pairs.append((pos1, pos2, sig))
            if sig_pairs:
                # Sort by span descending for stacking
                sig_pairs.sort(key=lambda x: x[1] - x[0], reverse=True)
                # Start from above the data with violin padding, leaving room for annotations
                current_height = (
                    global_max + violin_padding + (len(sig_pairs) * bracket_delta)
                )
                xref = "x" if i == 0 else f"x{i+1}"
                yref = "y" if i == 0 else f"y{i+1}"
                for pos1, pos2, sig in sig_pairs:
                    height = current_height
                    # Add horizontal line
                    fig.add_shape(
                        type="line",
                        x0=pos1,
                        y0=height,
                        x1=pos2,
                        y1=height,
                        line=dict(color="black", width=1),
                        xref=xref,
                        yref=yref,
                    )
                    # Add stars annotation
                    fig.add_annotation(
                        x=(pos1 + pos2) / 2,
                        y=height + bracket_delta * 0.2,
                        text=sig,
                        showarrow=False,
                        font=dict(size=12),
                        xref=xref,
                        yref=yref,
                    )
                    current_height -= bracket_delta

    fig.update_layout(
        title_text=title + " " + additional_title,
        showlegend=False,
        height=400 * num_groups,  # Adjust height based on number of groups
        violinmode="group",  # Ensure proper grouping
    )
    # Set the same y-axis range for all subplots
    fig.update_yaxes(range=[y_min_plot, y_max_plot])
    if save_dir:
        save_path = Path(save_dir) / f"{additional_title.replace(' ', '_')}.html"
        fig.write_html(clean_filename(save_path))
    fig.show()
    print("Violin plot complete.")


def compute_statistics(
    grouped_df: pd.core.groupby.GroupBy, value_col: str, compare_by: str
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Compute means, standard deviations, and x-values for plotting."""
    compare_group_values = grouped_df.groupby(compare_by)[value_col]
    means = compare_group_values.mean()
    stds = compare_group_values.std()
    return means, stds


def add_line_trace(
    fig: go.Figure,
    x: np.ndarray,
    means: pd.Series,
    stds: pd.Series,
    color: str,
    group_name: str,
    show_std: bool,
    fit_line: Literal["linear"] = None,
) -> None:
    """Add a line trace with optional standard deviation fill."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=means,
            mode="lines+markers",
            line=dict(color=color),
            name=group_name,
            marker=dict(size=8),
            opacity=0.9,
        )
    )
    if show_std:
        y_upper = means + stds
        y_lower = means - stds
        x_band = np.concatenate([x, x[::-1]])
        y_band = np.concatenate([y_upper, y_lower[::-1]])
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=y_band,
                fill="toself",
                fillcolor=hex_to_rgba_str(color, alpha=0.2),
                line=dict(color="rgba(255,255,255,0)"),
                name=group_name,
            )
        )

    if fit_line == "linear":
        """Add a linear regression fit line."""
        slope, intercept = np.polyfit(x, means, 1)
        fit_line_y = slope * np.array(x) + intercept
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fit_line_y,
                mode="lines",
                line=dict(dash="dash", color=color),
                name=f"{group_name} fit: {slope:.2f}x + {intercept:.2f}",
                opacity=0.7,
            )
        )


def add_bar_trace(
    fig: go.Figure,
    x: np.ndarray,
    means: pd.Series,
    stds: pd.Series,
    group_name: str,
    show_std: bool,
) -> None:
    """Add a bar trace with optional error bars."""
    fig.add_trace(
        go.Bar(
            x=x,
            y=means,
            name=group_name,
            error_y=dict(type="data", array=stds, visible=show_std),
            opacity=0.5,
        )
    )


def animate_2D_positions(
    raw_pos: np.ndarray,
    interval: int = 50,
    point_size: int = 5,
    trail: bool = False,
    title="Position Animation",
    fps: float = 20,
    run_example: bool = False,
):
    """
    Animate 2D position data with rainbow coloring based on time.

    Parameters:
    - raw_pos: np.ndarray of shape (N, 2)
    - interval: Delay between frames in ms
    - point_size: Scatter point size
    - trail: If True, show path up to current point
    - title: Plot title
    - fps: Frames per second for time calculation
    """
    if run_example:
        # Generate sample spiral trajectory data
        t = np.linspace(0, 4 * np.pi, 100)
        x = t * np.cos(t)
        y = t * np.sin(t)
        raw_pos = np.column_stack((x, y))
    raw_pos = np.asarray(raw_pos)
    if raw_pos.ndim != 2 or raw_pos.shape[1] != 2:
        raise ValueError("raw_pos must be a 2D array of shape (N, 2)")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set axis limits with some padding
    x_margin = (np.max(raw_pos[:, 0]) - np.min(raw_pos[:, 0])) * 0.1
    y_margin = (np.max(raw_pos[:, 1]) - np.min(raw_pos[:, 1])) * 0.1
    ax.set_xlim(np.min(raw_pos[:, 0]) - x_margin, np.max(raw_pos[:, 0]) + x_margin)
    ax.set_ylim(np.min(raw_pos[:, 1]) - y_margin, np.max(raw_pos[:, 1]) + y_margin)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Create colormap for rainbow effect
    colormap = cm.get_cmap("hsv")  # HSV gives a nice rainbow effect

    # Initialize scatter plot
    scat = ax.scatter([], [], s=point_size, c=[], cmap=colormap, vmin=0, vmax=1)

    # Add text for displaying current time
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # For trail mode, also create a line plot
    if trail:
        (line,) = ax.plot([], [], alpha=0.3, linewidth=1)

    def update(frame):
        current_time = frame / fps  # Calculate current time in seconds

        if trail:
            # Show all points up to current frame with rainbow coloring
            data = raw_pos[: frame + 1]

            # Create color array based on normalized time
            colors = np.linspace(0, 1, len(data))

            # Update scatter plot
            scat.set_offsets(data)
            scat.set_array(colors)

            # Update trail line with rainbow coloring
            if len(data) > 1:
                # Create segments for the line with different colors
                line.set_data(data[:, 0], data[:, 1])
                # Set line color to match the progression
                line_color = colormap(colors[-1] if len(colors) > 0 else 0)
                line.set_color(line_color)
        else:
            # Show only current point
            data = raw_pos[frame : frame + 1]

            # Color based on current frame position in total sequence
            color = frame / len(raw_pos)

            scat.set_offsets(data)
            scat.set_array([color])

        # Update time display
        time_text.set_text(
            f"Time: {current_time:.2f}s\nFrame: {frame + 1}/{len(raw_pos)}"
        )

        if trail:
            return scat, line, time_text
        else:
            return scat, time_text

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(raw_pos),
        interval=interval,
        repeat=True,
        blit=False,
    )

    # Add colorbar to show time progression
    cbar = plt.colorbar(scat, ax=ax, shrink=0.8)
    cbar.set_label("Time Progression", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()
    return ani


def plot_batch_heatmap(
    df, col_name, start_idx, batch_size, cols=5, rows=4, figsize_scale=2
):
    """
    Plot a batch of up to batch_size cells in a single figure with the specified grid.
    Ensures no empty subplots are shown by deleting unused axes.
    """
    num_to_plot = min(batch_size, len(df) - start_idx)
    if num_to_plot == 0:
        return

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale)
    )
    axes = axes.flatten()

    # Use only the axes needed for this batch
    used_axes = axes[:num_to_plot]

    for i in range(num_to_plot):
        idx = start_idx + i
        cell_id = df.index[idx]
        fields_map = df.loc[cell_id, col_name][0]
        max_num = fields_map.max()
        ax = used_axes[i]
        heatmap(fields_map, cmap="viridis", ax=ax, cbar=False)
        ax.set_title(f"Cell {cell_id}\nMax {max_num}", fontsize=8)
        ax.axis("off")

    # Remove empty subplots
    for j in range(num_to_plot, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Main execution
def plot_all_cells_modular(
    df, col_name, max_cells_per_plot=20, cols=None, rows=None, figsize_scale=2
):
    """
    Main function to split plotting into multiple modular batches.
    Each batch is plotted in a separate figure with the specified grid dimensions.
    Adjust cols and rows so that cols * rows >= max_cells_per_plot.
    """
    num_cells = len(df)
    num_batches = np.ceil(num_cells / max_cells_per_plot)

    max_cells_per_plot = min(max_cells_per_plot, num_cells)
    if cols is None or rows is None:
        # Default to a square-like layout
        cols = np.ceil(np.sqrt(max_cells_per_plot))
        rows = np.ceil(max_cells_per_plot / cols)

    for batch_num in range(num_batches):
        start_idx = batch_num * max_cells_per_plot
        plot_batch_heatmap(
            df, col_name, start_idx, max_cells_per_plot, cols, rows, figsize_scale
        )

    print(f"Plotted {num_cells} cells across {num_batches} figures.")
