import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from Manimeasure import simple_embedd, _add_embedding_subplot
from Helper import *
from Visualizer import Vizualizer
from matplotlib import cm
import matplotlib.colors as mcolors

from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Literal, Optional
from pathlib import Path
from plotly.subplots import make_subplots
from matplotlib.patches import Rectangle, Polygon


def prepare_labels_and_colors(
    labels,
    animal_ids_condition_dict,
    animals_list,
    tasks_list,
    color_by,
    label_by,
    max_task_distance,
):
    """
    Prepare plot labels and colors based on specified criteria.

    Parameters:
    - labels: List of DataFrame index labels
    - animal_ids_condition_dict: Dictionary mapping conditions to animal IDs
    - animals_list: List of animal IDs
    - tasks_list: List of tasks
    - color_by: What to color by ("animal", "condition", "task")
    - label_by: What to label by ("animal", "condition", "task")
    - max_task_distance: Maximum task distance for alpha gradient

    Returns:
    - plot_labels: List of labels for plotting
    - plot_colors: List of colors for plotting
    """
    if color_by == "animal":
        num_colors = len(animals_list)
    elif color_by == "condition":
        num_colors = len(animal_ids_condition_dict.keys())
    elif color_by == "task":
        num_colors = len(tasks_list)
    elif color_by == "stimulus_type":
        num_colors = 3
    else:
        raise ValueError("color_by must be 'animal', 'condition', or 'task'")

    base_colors = Vizualizer._get_base_color(index=np.arange(num_colors))
    base_colors = make_list_ifnot(base_colors)

    plot_labels = []
    plot_colors = []

    for label in labels:
        animal_id, date, task, model_type, behavior, movement, iterations = label.split(
            "_"
        )
        cond = "cFC" if animal_id in animal_ids_condition_dict["cFC"] else "NS"

        # Determine color
        if color_by == "condition":
            index = 0 if cond == "cFC" else 1
        elif color_by == "animal":
            index_list = np.where(np.array(animals_list) == animal_id)[0]
            if len(index_list) == 0:
                continue
            index = index_list[0]
        elif color_by == "task":
            index_list = np.where(np.array(tasks_list) == task)[0]
            if len(index_list) == 0:
                continue
            index = index_list[0]
        elif color_by == "stimulus_type":
            box_type = task[:2]
            number = int(task[2:])
            if "FS" in task:
                index = 0
            elif "NS" in task:
                index = 1

            if number > 9:
                index = 2

        color = base_colors[index]

        # Determine label
        if label_by == "condition":
            plot_labels.append(cond)
        elif label_by == "animal":
            plot_labels.append(animal_id.split("-")[1])
        elif label_by == "task":
            plot_labels.append(task)
        else:
            raise ValueError("label_by must be 'animal', 'condition', or 'task'")

        if color_by == "stimulus_type":
            print(asdf)
        else:
            # Apply alpha gradient based on task number
            task_num = np.where(np.array(tasks_list) == task)[0][0]
            gradient_color = Vizualizer._get_alpha_colors(
                color,
                task_num,
                min_alpha=0.0,
                min_value=0,
                max_value=max_task_distance,
            )[0]
            plot_colors.append(gradient_color)

    return plot_labels, plot_colors


def generate_additional_title(wanted_animal_ids, wanted_tasks, color_by):
    """
    Generate additional title for plots.

    Parameters:
    - wanted_tasks: List of tasks
    - color_by: What to color by ("animal", "condition", "task")

    Returns:
    - Additional title string
    """
    additional_title = f"| color by: {color_by}"
    if len(wanted_animal_ids) < 3:
        additional_title += f" | {' '.join(wanted_animal_ids)}"
    if len(wanted_tasks) < 5:
        additional_title += f" | {' '.join(wanted_tasks)}"
    return additional_title


def plot_fluorescence_embedding(
    flu_all,
    animal_ids_condition_dict,
    animals_list,
    tasks_list,
    wanted_animal_ids,
    wanted_tasks,
    additional_title="",
    color_by="animal",
    label_by="condition",
    method="mds",
    max_task_distance=13,
):
    """
    Plot fluorescence embedding for specified conditions.

    Parameters:
    - flu_all: DataFrame containing fluorescence data
    - animal_ids_condition_dict: Dictionary mapping conditions to animal IDs
    - animals_list: List of animal IDs in desired order
    - tasks_list: List of tasks in desired order
    - wanted_animal_ids: List of animal IDs to include
    - wanted_tasks: List of tasks to include
    - additional_title: Additional title text
    - constraints: List of constraints (e.g., ["cFC", "NS"]) or None
    - color_by: What to color by ("animal", "condition", "task")
    - label_by: What to label by ("animal", "condition", "task")
    - method: Embedding method ("mds", "tsne", "umap")
    - max_task_distance: Maximum task distance for alpha gradient

    Returns:
    - Coordinates from embedding
    """
    # Filter DataFrame
    constraint_flu = filter_dataframe(flu_all, [wanted_animal_ids, wanted_tasks])

    # Prepare labels and colors
    labels = constraint_flu.index.tolist()
    plot_labels, plot_colors = prepare_labels_and_colors(
        labels,
        animal_ids_condition_dict,
        animals_list,
        tasks_list,
        color_by,
        label_by,
        max_task_distance,
    )

    # Generate additional title

    additional_title += generate_additional_title(
        wanted_animal_ids, wanted_tasks, color_by
    )

    # Generate embedding
    coords = simple_embedd(
        constraint_flu,
        additional_title=additional_title,
        labels=plot_labels,
        colors=plot_colors,
        method=method,
    )

    return coords


def plot_all_constraints_fluorescence_embedding(
    flu_all,
    animal_ids_condition_dict,
    animals_list,
    tasks_list,
    wanted_animal_ids,
    wanted_tasks,
    additional_title="",
    color_by="animal",
    label_by="condition",
    method="mds",
    max_task_distance=13,
):
    """
    Plot fluorescence embeddings for multiple constraint conditions.

    Parameters:
    - flu_all: DataFrame containing fluorescence data
    - animal_ids_condition_dict: Dictionary mapping conditions to animal IDs
    - animals_list: List of animal IDs in desired order
    - tasks_list: List of tasks in desired order
    - wanted_animal_ids: List of animal IDs to include
    - wanted_tasks: List of tasks to include
    - additional_title: Additional title text
    - color_by: What to color by ("animal", "condition", "task")
    - label_by: What to label by ("animal", "condition", "task")
    - method: Embedding method ("mds", "tsne", "umap")
    - max_task_distance: Maximum task distance for alpha gradient

    Returns:
    - Coordinates from the last embedding
    """
    num_conditions = 3
    org_wanted_animal_ids = wanted_animal_ids.copy()
    fig, axs = plt.subplots(1, num_conditions, figsize=(num_conditions * 7, 5))
    fontsize = Vizualizer.auto_fontsize(fig)
    title = f"{method.upper()} Embeddings of Procrustes Distances"
    additional_title += generate_additional_title(
        wanted_animal_ids, wanted_tasks, color_by
    )
    fig.suptitle(title + additional_title, fontsize=fontsize)
    fig.subplots_adjust(hspace=0.6, wspace=0.6)

    constraint_list = [None, "cFC", "NS"]
    coords = None

    for i, constraint in enumerate(constraint_list):
        if constraint is None:
            pass
        else:
            cleaned_wanted_animal_ids = []
            for animal_id in org_wanted_animal_ids:
                if animal_id in animal_ids_condition_dict[constraint]:
                    cleaned_wanted_animal_ids.append(animal_id)
            wanted_animal_ids = cleaned_wanted_animal_ids

        # Filter DataFrame
        constraint_flu = filter_dataframe(flu_all, [wanted_animal_ids, wanted_tasks])

        # Prepare labels and colors
        labels = constraint_flu.index.tolist()
        plot_labels, plot_colors = prepare_labels_and_colors(
            labels,
            animal_ids_condition_dict,
            animals_list,
            tasks_list,
            color_by,
            label_by,
            max_task_distance,
        )

        # Generate embedding
        coords = simple_embedd(
            constraint_flu,
            title=constraint if isinstance(constraint, str) else "All",
            ax=axs[i],
            # additional_title=additional_title,
            labels=plot_labels,
            colors=plot_colors,
            method=method,
        )

    return coords


def create_task_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create unique task labels by combining animal, date, task, and model."""
    # Ensure 'date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label"] = (
        df["animal"]
        + "_"
        + df["date"].dt.strftime("%Y%m%d")
        + "_"
        + df["task"]
        + "_"
        + df["model"]
    )
    return df


def validate_labels(
    df: pd.DataFrame, d_place: pd.DataFrame, d_notplace: pd.DataFrame
) -> pd.DataFrame:
    """Validate and filter labels present in both distance matrices."""
    missing_in_place = df["label"][~df["label"].isin(d_place.index)].tolist()
    missing_in_notplace = df["label"][~df["label"].isin(d_notplace.index)].tolist()

    if missing_in_place or missing_in_notplace:
        print(f"Missing in d_place: {len(missing_in_place)} labels")
        print(f"Missing in d_notplace: {len(missing_in_notplace)} labels")
        valid_labels = df["label"].isin(d_place.index) & df["label"].isin(
            d_notplace.index
        )
        return df[valid_labels].copy()
    return df.copy()


def plot_heatmap_matrix(
    fig_title: str,
    comparisson_df: pd.DataFrame,
    metric: str,
    task_groups: Dict[str, Dict[str, List[str]]] = None,
    cmap_label: str = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu",
    combine_colorbar: bool = False,
    show_colorbar: bool = True,
    show_number: bool = True,
    pdf: Optional[object] = None,
):
    """Plots one row of heatmaps for a given metric in comparisson_df, skipping identical matrices across conditions.

    Parameters:
    - pdf: PdfPages object to save plot to. If None, displays plot.
    """
    conditions = sorted(comparisson_df["condition"].unique())
    n_conditions = len(conditions)

    # Detect if all condition matrices are identical for this metric to plot only one
    matrices_by_condition = {}
    for cond in conditions:
        cond_df = comparisson_df[comparisson_df["condition"] == cond]
        tasks = sorted(set(cond_df["task1_id"]).union(set(cond_df["task2_id"])))
        n_tasks = len(tasks)
        plot_matrix = pd.DataFrame(
            np.full((n_tasks, n_tasks), np.nan), index=tasks, columns=tasks
        )
        # Populate matrix with values from comparisson_df
        for _, row in cond_df.iterrows():
            task1, task2 = row["task1_id"], row["task2_id"]
            value = row[metric]
            if pd.notna(value):
                plot_matrix.loc[task1, task2] = value
                plot_matrix.loc[task2, task1] = value  # Ensure symmetry
        plot_matrix[plot_matrix == 0] = np.nan
        matrices_by_condition[cond] = plot_matrix

    # Check if all matrices are equal (ignoring NaNs)
    all_matrices_equal = (
        all(
            matrices_by_condition[cond].equals(matrices_by_condition[conditions[0]])
            for cond in conditions[1:]
        )
        if n_conditions > 1
        else False
    )
    if all_matrices_equal:
        conditions_to_plot = [conditions[0]]  # Plot only the first
        title_suffix = " (Identical Across Conditions)"
        print(
            f"{fig_title}: All condition matrices identical; plotting only {conditions[0]}"
        )
    else:
        conditions_to_plot = conditions
        title_suffix = ""

    n_to_plot = len(conditions_to_plot)
    fig, axes = plt.subplots(
        1, n_to_plot, figsize=(8 * n_to_plot, 8), constrained_layout=True
    )
    if n_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Compute global vmin/vmax if combine_colorbar is True and show_colorbar is True
    if combine_colorbar and show_colorbar:
        vmin_preset = vmin is not None
        vmax_preset = vmax is not None
        if not vmin_preset or not vmax_preset:
            overall_vmin = np.inf if not vmin_preset else vmin
            overall_vmax = -np.inf if not vmax_preset else vmax
            for condition in conditions_to_plot:
                cond_df = comparisson_df[comparisson_df["condition"] == condition]
                values = cond_df[metric].dropna()
                if not values.empty:
                    current_min = values.min()
                    current_max = values.max()
                    if not vmin_preset:
                        overall_vmin = min(overall_vmin, current_min)
                    if not vmax_preset:
                        overall_vmax = max(overall_vmax, current_max)
            vmin = overall_vmin if not vmin_preset else vmin
            vmax = overall_vmax if not vmax_preset else vmax

    cmap_obj = plt.get_cmap(cmap)
    cdark_gray = mcolors.to_rgba("dimgray", alpha=0.3)
    cmap_obj.set_under(cdark_gray)
    cmap_obj.set_over(cdark_gray)

    for i, cond in enumerate(conditions_to_plot):
        cond_df = comparisson_df[comparisson_df["condition"] == cond]
        tasks = sorted(set(cond_df["task1_id"]).union(set(cond_df["task2_id"])))
        plot_matrix = matrices_by_condition[cond]  # Reuse precomputed
        labels = [lbl.replace("_", " ") for lbl in plot_matrix.index]
        try:
            is_symmetric = np.allclose(
                plot_matrix.fillna(0).values,
                plot_matrix.fillna(0).values.T,
                equal_nan=True,
            )
        except Exception:
            is_symmetric = False
        mask = (
            np.triu(np.ones(plot_matrix.shape, dtype=bool), k=1)
            if is_symmetric
            else None
        )

        # Compute vmin/vmax for this condition
        if combine_colorbar:
            cond_vmin = vmin
            cond_vmax = vmax
        else:
            values = cond_df[metric].dropna()
            cond_vmin = vmin if vmin is not None else (values.min() if not values.empty else None)
            cond_vmax = vmax if vmax is not None else (values.max() if not values.empty else None)

        heatmap_kwargs = dict(
            data=plot_matrix,
            cmap=cmap_obj,
            ax=axes[i],
            mask=mask,
            square=True,
            cbar=False,
            vmin=cond_vmin,
            vmax=cond_vmax,
        )

        sns.heatmap(**heatmap_kwargs)
        cond_title = f"{fig_title}: {cond.replace('_', ' ')}"
        if all_matrices_equal and i == 0:
            cond_title += title_suffix
        axes[i].set_title(cond_title)
        axes[i].set_xlabel("Task Name Number")
        axes[i].set_ylabel("Task Name Number")
        axes[i].set_xticklabels(labels, rotation=45, ha="right")
        axes[i].set_yticklabels(labels, rotation=0)

        if show_number:
            norm = mcolors.Normalize(vmin=cond_vmin, vmax=cond_vmax)
            for j in range(plot_matrix.shape[0]):
                for k in range(plot_matrix.shape[1]):
                    value = plot_matrix.iat[j, k]
                    if k > j:  # Skip upper triangle
                        continue
                    if not np.isnan(value):
                        normalized_value = norm(value)
                        rgb_color = cmap_obj(normalized_value)[:3]
                        luminance = (
                            0.299 * rgb_color[0]
                            + 0.587 * rgb_color[1]
                            + 0.114 * rgb_color[2]
                        )
                        text_color = "black" if luminance > 0.5 else "white"
                        axes[i].text(
                            k + 0.5,
                            j + 0.5,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                        )
        # Draw rectangles for task groups
        if task_groups is not None and cond in task_groups:
            indices = {lbl: idx for idx, lbl in enumerate(plot_matrix.index)}
            for group_name, task_names in task_groups[cond].items():
                group_indices = []
                for t in task_names:
                    # Find all matrix labels that start with the task prefix
                    matching = [indices[lbl] for lbl in indices if lbl.startswith(t)]
                    group_indices.extend(matching)
                if group_indices:
                    # Rectangle covers the min/max positions of the group
                    start, end = min(group_indices), max(group_indices) + 1
                    size = end - start
                    if is_symmetric:
                        coords = [(start, start), (start, end), (end, end)]
                        tri = Polygon(
                            coords,
                            closed=False,
                            fill=False,
                            edgecolor="black",
                            linewidth=3,
                        )
                        axes[i].add_patch(tri)
                    else:
                        rect = Rectangle(
                            (start, start),
                            size,
                            size,
                            fill=False,
                            edgecolor="black",
                            linewidth=3,
                        )
                        axes[i].add_patch(rect)

        # Add colorbar for this subplot if separate colorbars
        if show_colorbar and not combine_colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=cmap_obj, norm=plt.Normalize(vmin=cond_vmin, vmax=cond_vmax)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[i], shrink=0.5, pad=0.02)
            if cmap_label:
                cbar.set_label(cmap_label)

    if show_colorbar and combine_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
        if cmap_label:
            cbar.set_label(cmap_label)

    fig.suptitle(fig_title + title_suffix, y=1.05)

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_pointcloud_embeddings(
    task_data_df: pd.DataFrame,
    d_place: pd.DataFrame,
    d_notplace: pd.DataFrame,
    embedding_method: Union[
        bool,
        List[
            Literal[
                "tsne",
                "mds",
                "isomap",
                "lle",
                "spectral_embedding",
                "umap",
                "pca",
                "mds_pca",
            ]
        ],
    ] = ["pca", "umap"],
    n_components: int = 2,
    plot_show: Literal[
        "center", "center_std", "samples", "flow", "annotate_dots"
    ] = "samples",
    color_by: str = "condition",
    plot_save_dir: Optional[Path] = None,
    additional_title: str = "",
    pdf: Optional[object] = None,
) -> None:
    """
    Create a 2x2 subplot visualization:
    - Upper row: First condition (place cells left, not-place cells right)
    - Lower row: Second condition (place cells left, not-place cells right)

    Parameters:
    - task_data_df: DataFrame with columns ['animal', 'date', 'task', 'model', 'condition', 'task_name', 'task_number', 'label']
    - d_place: DataFrame representing place cell distances
    - d_notplace: DataFrame representing not-place cell distances
    - embedding_method: Embedding method to use (e.g., 'umap', 'pca', 'tsne', 'mds')
    - n_components: Dimensionality of the embedding (2 or 3)
    - plot_show: Plot style for simple_embedd (e.g., 'samples', 'center')
    - color_by: Column in task_data_df to use for splitting subplots (e.g., 'condition', 'animal', 'task_name')
    - plot_save_dir: Directory to save the HTML plot file (optional)
    - additional_title: Extra text to append to the plot title

    Returns:
    - None: Displays the Plotly figure and optionally saves it as HTML
    """
    # Validate inputs
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3")
    if color_by not in task_data_df.columns:
        raise ValueError(f"Column '{color_by}' not found in task_data_df")

    # Ensure task_data_df has labels
    if "label" not in task_data_df.columns:
        task_data_df = create_task_labels(task_data_df.copy())

    # Find valid labels present in both distance matrices
    valid_labels = [
        lbl
        for lbl in task_data_df["label"].unique()
        if lbl in d_place.index and lbl in d_notplace.index
    ]
    if not valid_labels:
        raise ValueError("No valid labels found in both d_place and d_notplace")

    filtered_df = task_data_df[task_data_df["label"].isin(valid_labels)].copy()

    # Get unique conditions and ensure exactly 2
    conditions = filtered_df[color_by].unique()
    if len(conditions) != 2:
        raise ValueError(
            f"Expected exactly 2 unique values in '{color_by}', found {len(conditions)}: {conditions}"
        )

    cond1, cond2 = sorted(
        conditions
    )  # Sort for consistent ordering (e.g., C1 before C2)

    # Filter dataframes for each condition
    df_cond1 = filtered_df[filtered_df[color_by] == cond1]
    df_cond2 = filtered_df[filtered_df[color_by] == cond2]

    valid_labels_cond1 = [lbl for lbl in df_cond1["label"] if lbl in valid_labels]
    valid_labels_cond2 = [lbl for lbl in df_cond2["label"] if lbl in valid_labels]

    if len(valid_labels_cond1) == 0 or len(valid_labels_cond2) == 0:
        raise ValueError(
            f"No valid labels for condition 1 ({cond1}) or condition 2 ({cond2})"
        )

    # Create 2x2 subplot layout
    subplot_titles = [
        f"{cond1} - Place Cells",
        f"{cond1} - Not-Place Cells",
        f"{cond2} - Place Cells",
        f"{cond2} - Not-Place Cells",
    ]

    if n_components == 2:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.06,
            vertical_spacing=0.1,
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=subplot_titles,
            specs=[
                [{"type": "scene"}, {"type": "scene"}],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            horizontal_spacing=0.06,
            vertical_spacing=0.1,
        )
    # Upper row: Condition 1
    plot_df_cond1 = df_cond1[df_cond1["label"].isin(valid_labels_cond1)].copy()

    # Place cells for condition 1 (row=1, col=1)
    _add_embedding_subplot(
        fig,
        d_place.loc[valid_labels_cond1, valid_labels_cond1],
        plot_df_cond1,
        plot_show=plot_show,
        n_components=n_components,
        embedding_method=embedding_method,
        row=1,
        col=1,
        subtitle=subplot_titles[0],
        legend=True,
    )

    # Not-place cells for condition 1 (row=1, col=2)
    _add_embedding_subplot(
        fig,
        d_notplace.loc[valid_labels_cond1, valid_labels_cond1],
        plot_df_cond1,
        n_components=n_components,
        embedding_method=embedding_method,
        plot_show=plot_show,
        row=1,
        col=2,
        subtitle=subplot_titles[1],
        legend=False,
    )

    # Lower row: Condition 2
    plot_df_cond2 = df_cond2[df_cond2["label"].isin(valid_labels_cond2)].copy()

    # Place cells for condition 2 (row=2, col=1)
    _add_embedding_subplot(
        fig,
        d_place.loc[valid_labels_cond2, valid_labels_cond2],
        plot_df_cond2,
        n_components=n_components,
        embedding_method=embedding_method,
        plot_show=plot_show,
        row=2,
        col=1,
        subtitle=subplot_titles[2],
        legend=False,
    )

    # Not-place cells for condition 2 (row=2, col=2)
    _add_embedding_subplot(
        fig,
        d_notplace.loc[valid_labels_cond2, valid_labels_cond2],
        plot_df_cond2,
        n_components=n_components,
        embedding_method=embedding_method,
        plot_show=plot_show,
        row=2,
        col=2,
        subtitle=subplot_titles[3],
        legend=True,
    )

    # Update overall layout
    suptitle = f"{str(embedding_method).upper()} Point Cloud | Split by {color_by} | Show: {plot_show} | {additional_title}"
    fig.update_layout(
        title_text=suptitle,
        title_x=0.5,
        title_y=0.98,
        height=800 if n_components == 2 else 1000,
        width=1200,
        showlegend=True,
        legend=dict(traceorder="normal", x=1.05, y=0.5),
    )

    # Disable axis labels and ticks for cleaner visualization
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=col)
            if n_components == 3:
                fig.update_scenes(
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    zaxis=dict(showticklabels=False, showgrid=False),
                    row=row,
                    col=col,
                )

    # Show and optionally save the plot
    if pdf is not None:
        # Convert plotly figure to static image for PDF
        import plotly.io as pio

        img_bytes = pio.to_image(
            fig, format="png", width=1200, height=800 if n_components == 2 else 1000
        )
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes))

        # Create matplotlib figure for PDF
        fig_mpl = plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(suptitle, fontsize=14, pad=20)
        pdf.savefig(fig_mpl, bbox_inches="tight")
        plt.close(fig_mpl)
    else:
        fig.show()

    if plot_save_dir is not None:
        plot_save_dir = Path(plot_save_dir)
        plot_save_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = clean_filename(suptitle)
        fig.write_html(plot_save_dir / f"{safe_filename}.html")


def plot_animal_heatmap(
    place_dist: pd.DataFrame,
    notplace_dist: pd.DataFrame,
    animal_conditions: Dict[str, str],
    pdf: Optional[object] = None,
) -> None:
    """Plot heatmap of pairwise animal distances sorted by hierarchical clustering.

    Parameters:
    - place_dist: Distance matrix for place cells
    - notplace_dist: Distance matrix for non-place cells
    - animal_conditions: Dictionary mapping animal IDs to their conditions
    - pdf: PdfPages object to save plot to. If None, displays plot.
    """

    temp_place = place_dist.copy()
    temp_place.values[np.diag_indices_from(temp_place)] = 0
    temp_notplace = notplace_dist.copy()
    temp_notplace.values[np.diag_indices_from(temp_notplace)] = 0

    place_dist_array = temp_place.fillna(0).values
    notplace_dist_array = temp_notplace.fillna(0).values

    place_condensed = squareform(place_dist_array)
    notplace_condensed = squareform(notplace_dist_array)

    place_linkage = linkage(place_condensed, method="ward")
    notplace_linkage = linkage(notplace_condensed, method="ward")

    place_order = dendrogram(place_linkage, no_plot=True)["leaves"]
    notplace_order = dendrogram(notplace_linkage, no_plot=True)["leaves"]

    sorted_animals_place = place_dist.index[place_order].tolist()
    sorted_animals_notplace = notplace_dist.index[notplace_order].tolist()

    place_labels = [
        f"{animal} {animal_conditions[animal]}".replace("_", " ")
        for animal in sorted_animals_place
    ]
    notplace_labels = [
        f"{animal} {animal_conditions[animal]}".replace("_", " ")
        for animal in sorted_animals_notplace
    ]

    plot_place_dist = place_dist.copy()
    plot_place_dist[plot_place_dist == 0] = np.nan
    plot_notplace_dist = notplace_dist.copy()
    plot_notplace_dist[plot_notplace_dist == 0] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20), constrained_layout=True)

    sns.heatmap(
        plot_place_dist.loc[sorted_animals_place, sorted_animals_place],
        cmap="RdBu",
        ax=ax1,
        cbar_kws={"label": "Place Distance", "shrink": 0.5},
        square=True,
    )
    ax1.set_xticklabels(place_labels, rotation=45, ha="right")
    ax1.set_yticklabels(place_labels, rotation=0)
    ax1.set_title("Animal Pairwise Distances (Place Cells)")

    sns.heatmap(
        plot_notplace_dist.loc[sorted_animals_notplace, sorted_animals_notplace],
        cmap="RdBu",
        ax=ax2,
        cbar_kws={"label": "Not-Place Distance", "shrink": 0.5},
        square=True,
    )
    ax2.set_xticklabels(notplace_labels, rotation=45, ha="right")
    ax2.set_yticklabels(notplace_labels, rotation=0)
    ax2.set_title("Animal Pairwise Distances (Not-Place Cells)")

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_all_tasks_heatmap(
    task_data_df: pd.DataFrame,
    d_place: pd.DataFrame,
    d_notplace: pd.DataFrame,
    combine_colormap: bool = True,
    pdf: Optional[object] = None,
) -> None:
    """Plot heatmaps of pairwise distances between all tasks.

    Parameters:
    - pdf: PdfPages object to save plot to. If None, displays plot.
    """
    # Check symmetry of d_notplace
    is_symmetric = np.allclose(d_notplace, d_notplace.T, equal_nan=True)
    if not is_symmetric:
        print("Warning: d_notplace is not symmetric. Enforcing symmetry.")
        d_notplace = (d_notplace + d_notplace.T) / 2  # Enforce symmetry

    # Sort tasks by condition, animal, task_number
    sorted_df = task_data_df.sort_values(["condition", "animal", "task_number"])
    sorted_labels = sorted_df["label"].tolist()

    # Ensure all labels are in d_notplace
    valid_labels = [
        lbl for lbl in sorted_labels if lbl in d_notplace.index and lbl in d_place.index
    ]
    if len(valid_labels) < len(sorted_labels):
        print(
            f"Reduced from {len(sorted_labels)} to {len(valid_labels)} valid labels in all tasks heatmap"
        )

    # Replace 0s with NaN
    plot_d_place = d_place.loc[valid_labels, valid_labels].copy()
    plot_d_place[plot_d_place == 0] = np.nan
    plot_d_notplace = d_notplace.loc[valid_labels, valid_labels].copy()
    plot_d_notplace[plot_d_notplace == 0] = np.nan

    # Create labels without underscores
    tick_labels = [lbl.replace("_", " ") for lbl in valid_labels]
    # If there are fewer than 2 labels, just plot as-is
    if len(valid_labels) < 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20), constrained_layout=True)
        sns.heatmap(
            plot_d_place,
            cmap="RdBu",
            ax=ax1,
            cbar_kws={"label": "Place Distance", "shrink": 0.5},
            square=True,
        )
        ax1.set_title("All Tasks Pairwise Distances (Place Cells)")
        sns.heatmap(
            plot_d_notplace,
            cmap="RdBu",
            ax=ax2,
            cbar_kws={"label": "Not-Place Distance", "shrink": 0.5},
            square=True,
        )
        ax2.set_title("All Tasks Pairwise Distances (Not-Place Cells)")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        return

    # Compute a combined distance for clustering. Use average of place and notplace where available.
    # Replace zeros with NaN so they don't bias the average (they typically represent missing).
    comb_place = plot_d_place.copy().astype(float)
    comb_notplace = plot_d_notplace.copy().astype(float)
    comb_place[comb_place == 0] = np.nan
    comb_notplace[comb_notplace == 0] = np.nan

    combined = pd.DataFrame(index=valid_labels, columns=valid_labels, dtype=float)
    for i in valid_labels:
        for j in valid_labels:
            a = comb_place.at[i, j]
            b = comb_notplace.at[i, j]
            vals = [v for v in (a, b) if not (pd.isna(v))]
            combined.at[i, j] = np.nan if len(vals) == 0 else np.mean(vals)

    # For clustering we need a condensed distance matrix. Convert NaNs to 0 for the purpose of squareform,
    # but we will set diagonal to 0 and ensure symmetry first.
    combined_values = combined.fillna(0).values
    # Enforce symmetry
    combined_values = (combined_values + combined_values.T) / 2
    # Ensure diagonal is zero
    np.fill_diagonal(combined_values, 0)

    try:
        condensed = squareform(combined_values)
        linkage_matrix = linkage(condensed, method="ward")
        order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
        ordered_labels = [valid_labels[i] for i in order]
    except Exception as e:
        # Fallback: keep original order
        print(f"Clustering failed: {e}. Falling back to original label order.")
        ordered_labels = valid_labels

    # Reorder matrices according to clustering
    plot_d_place_ordered = plot_d_place.loc[ordered_labels, ordered_labels]
    plot_d_notplace_ordered = plot_d_notplace.loc[ordered_labels, ordered_labels]
    ordered_tick_labels = [lbl.replace("_", " ") for lbl in ordered_labels]

    # If requested, compute a shared vmin/vmax across both matrices so the
    # colormap ranges are equal.
    vmin = vmax = None
    if combine_colormap:
        try:
            combined_vals = np.concatenate(
                [
                    plot_d_place_ordered.values.flatten(),
                    plot_d_notplace_ordered.values.flatten(),
                ]
            )
            combined_vals = np.where(combined_vals == 0, np.nan, combined_vals)
            vmin = float(np.nanmin(combined_vals))
            vmax = float(np.nanmax(combined_vals))
        except Exception:
            vmin = vmax = None

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 20), constrained_layout=True
    )  # Larger square format

    heatmap_kwargs_1 = dict(
        data=plot_d_place_ordered,
        cmap="RdBu",
        ax=ax1,
        square=True,
    )
    if vmin is not None:
        heatmap_kwargs_1["vmin"] = vmin
    if vmax is not None:
        heatmap_kwargs_1["vmax"] = vmax
    heatmap_kwargs_1["cbar_kws"] = {"label": "Place Distance", "shrink": 0.5}

    sns.heatmap(**heatmap_kwargs_1)
    # ax1.set_xticklabels(ordered_tick_labels, rotation=45, ha='right')
    # ax1.set_yticklabels(ordered_tick_labels, rotation=0)
    ax1.set_title("All Tasks Pairwise Distances (Place Cells)")

    heatmap_kwargs_2 = dict(
        data=plot_d_notplace_ordered,
        cmap="RdBu",
        ax=ax2,
        square=True,
    )
    if vmin is not None:
        heatmap_kwargs_2["vmin"] = vmin
    if vmax is not None:
        heatmap_kwargs_2["vmax"] = vmax
    # If combining colormaps, only show the colorbar for the first heatmap to
    # avoid duplicate colorbars. Otherwise show a separate colorbar.
    if combine_colormap:
        heatmap_kwargs_2["cbar"] = False
    else:
        heatmap_kwargs_2["cbar_kws"] = {"label": "Not-Place Distance", "shrink": 0.5}

    sns.heatmap(**heatmap_kwargs_2)
    # ax2.set_xticklabels(ordered_tick_labels, rotation=45, ha='right')
    # ax2.set_yticklabels(ordered_tick_labels, rotation=0)
    ax2.set_title("All Tasks Pairwise Distances (Not-Place Cells)")
    # remove ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _extract_metadata_from_long_df(long_format_df: pd.DataFrame) -> Dict:
    """
    Extract metadata from long-format dataframe that was previously in task_data_df.

    Parameters:
    - long_format_df: DataFrame with columns including animal_id_i/j, task_name_i/j, condition_i/j

    Returns:
    - Dictionary with extracted metadata:
        - 'animals': sorted list of unique animals
        - 'tasks': sorted list of unique tasks
        - 'conditions': sorted list of unique conditions
        - 'animal_conditions': dict mapping animal_id to condition
    """
    # Get unique animals from both _i and _j columns
    animals_i = set(long_format_df["animal_id_i"].unique())
    animals_j = set(long_format_df["animal_id_j"].unique())
    animals = sorted(animals_i.union(animals_j))

    # Get unique tasks
    tasks_i = set(long_format_df["task_name_i"].unique())
    tasks_j = set(long_format_df["task_name_j"].unique())
    tasks = sorted(tasks_i.union(tasks_j))

    # Get unique conditions
    conditions_i = set(long_format_df["condition_i"].dropna().unique())
    conditions_j = set(long_format_df["condition_j"].dropna().unique())
    conditions = sorted(conditions_i.union(conditions_j))

    # Create animal -> condition mapping
    animal_conditions = {}
    for _, row in (
        long_format_df[["animal_id_i", "condition_i"]].drop_duplicates().iterrows()
    ):
        if pd.notna(row["condition_i"]):
            animal_conditions[row["animal_id_i"]] = row["condition_i"]
    for _, row in (
        long_format_df[["animal_id_j", "condition_j"]].drop_duplicates().iterrows()
    ):
        if pd.notna(row["condition_j"]):
            animal_conditions[row["animal_id_j"]] = row["condition_j"]

    return {
        "animals": animals,
        "tasks": tasks,
        "conditions": conditions,
        "animal_conditions": animal_conditions,
    }


def analyze_and_plot_distances(
    long_format_df: pd.DataFrame,
    task_groups: Dict[str, Dict[str, List[str]]] = None,
    wanted_tasks: Optional[List[str]] = None,
    embedding_methods: List[Literal["mds", "pca", "umap", "mds_pca"]] = ["pca"],
    plot: Union[str, List[Literal["animal", "task", "all_tasks"]]] = [
        "animal",
        "task",
        "all_tasks",
    ],
    combine_colorbar: bool = True,
    save_pdf: bool = True,
    pdf_filename: Optional[str] = None,
    additional_title: str = "",
) -> None:
    """Main function to analyze distances and create plots using long-format dataframe.

    Parameters:
    - long_format_df: DataFrame with columns ['animal_id_i', 'task_name_i', 'animal_id_j', 'task_name_j',
                                                'pc_disparity', 'npc_disparity', 'condition_i', 'condition_j', ...]
    - task_groups (dict, optional): Dictionary with keys as condition names and values
        as dicts mapping group names to lists of task name prefixes to highlight with rectangles.
    - wanted_tasks: List of task names to filter
    - embedding_methods: List of embedding methods for visualization
    - plot: List of plot types to generate. Options are 'animal', 'task', 'all_tasks'.
    - combine_colorbar: If True, uses a single colorbar for all condition subplots. If False, each condition subplot has its own colorbar.
    - save_pdf: If True, saves all plots to a single PDF file
    - pdf_filename: Custom filename for the PDF. If None, generates from parameters.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime

    df = add_tasknamestrings_and_digits(long_format_df.copy())

    wanted_tasks = make_list_ifnot(wanted_tasks)
    plot = make_list_ifnot(plot)

    # Generate PDF filename if not provided
    if save_pdf and pdf_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tasks_str = f"_{len(wanted_tasks)}tasks" if wanted_tasks else "_all_tasks"
        methods_str = "_".join(embedding_methods)
        plot_types_str = "_".join(plot)
        colorbar_str = "combined_cbar" if combine_colorbar else "separate_cbar"
        pdf_filename = f"distance_analysis{tasks_str}_{methods_str}_{plot_types_str}_{colorbar_str}_{timestamp}_{additional_title}.pdf"

    # Filter long_format_df based on wanted_tasks
    if wanted_tasks:
        filtered_long_df = df[
            df["task_name_i"].isin(wanted_tasks)
            & df["task_name_j"].isin(wanted_tasks)
        ].copy()
    else:
        filtered_long_df = df.copy()

    # Extract metadata from long format dataframe
    metadata = _extract_metadata_from_long_df(filtered_long_df)

    print(f"Filtered df to {len(filtered_long_df)} comparisons")
    #print(f"Sample comparisons:\n{filtered_long_df.head()}")

    if save_pdf:
        print(f"\nSaving all plots to: {pdf_filename}")
        pdf_path = Path.cwd().joinpath("logs").joinpath(pdf_filename)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            # Create title page
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(
                "Distance Analysis Report", fontsize=24, fontweight="bold", y=0.95
            )

            # Add analysis parameters
            param_text = f"""
Analysis Parameters:
{"=" * 60}

Tasks Analyzed: {", ".join(wanted_tasks) if wanted_tasks else "All tasks"}
Number of Tasks: {len(wanted_tasks) if wanted_tasks else len(metadata["tasks"])}
Number of Comparisons: {len(filtered_long_df)}
Number of Animals: {len(metadata["animals"])}
Conditions: {", ".join(metadata["conditions"])}

Embedding Methods: {", ".join(embedding_methods)}
Plot Types: {", ".join(plot)}
Colorbar: {"Combined" if combine_colorbar else "Separate per condition"}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            plt.text(
                0.1,
                0.5,
                param_text,
                fontsize=12,
                family="monospace",
                verticalalignment="center",
                transform=fig.transFigure,
            )
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Compute pairwise distances and save plots to PDF
            if "animal" in plot:
                place_dist, notplace_dist = compute_animal_pairwise_distances_long(
                    filtered_long_df, plot=True, pdf=pdf
                )

            if "task" in plot:
                comparisson_df = compute_task_pairwise_distances_long(
                    filtered_long_df,
                    embedding_methods=embedding_methods,
                    task_groups=task_groups,
                    plot=True,
                    combine_colorbar=combine_colorbar,
                    pdf=pdf,
                )

                # run linear mixed model

            # All-tasks plot
            if "all_tasks" in plot:
                plot_all_tasks_heatmap_long(filtered_long_df, pdf=pdf)

        print(f"✓ PDF saved successfully to: {pdf_path}")
    else:
        # Original behavior without PDF
        if "animal" in plot:
            place_dist, notplace_dist = compute_animal_pairwise_distances_long(
                filtered_long_df, plot=True
            )

        if "task" in plot:
            comparisson_df = compute_task_pairwise_distances_long(
                filtered_long_df,
                embedding_methods=embedding_methods,
                task_groups=task_groups,
                plot=True,
                combine_colorbar=combine_colorbar,
            )

        # All-tasks plot
        if "all_tasks" in plot:
            plot_all_tasks_heatmap_long(filtered_long_df)


def compute_animal_pairwise_distances_long(
    long_format_df: pd.DataFrame,
    plot: bool = True,
    pdf: Optional[object] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean pairwise distances between animals using long-format dataframe.

    Parameters:
    - long_format_df: Long-format DataFrame with pairwise comparisons (must have condition_i/j columns)
    - plot: If True, will call plot_animal_heatmap
    - pdf: PdfPages object to save plot to. If None, displays plot.

    Returns:
    - place_df: Animal x Animal distance matrix for place cells
    - notplace_df: Animal x Animal distance matrix for non-place cells
    """
    # Extract metadata from long format dataframe
    metadata = _extract_metadata_from_long_df(long_format_df)
    animals = metadata["animals"]
    n_animals = len(animals)

    place_dist_matrix = np.zeros((n_animals, n_animals))
    notplace_dist_matrix = np.zeros((n_animals, n_animals))

    for i, animal1 in enumerate(animals):
        # Within-animal distances (diagonal)
        within_mask = (long_format_df["animal_id_i"] == animal1) & (
            long_format_df["animal_id_j"] == animal1
        )
        within_comparisons = long_format_df[within_mask]

        if len(within_comparisons) > 0:
            place_dist_matrix[i, i] = within_comparisons["pc_disparity"].mean()
            notplace_dist_matrix[i, i] = within_comparisons["npc_disparity"].mean()

        # Between-animal distances
        for j, animal2 in enumerate(animals):
            if i >= j:
                continue

            between_mask = (
                (long_format_df["animal_id_i"] == animal1)
                & (long_format_df["animal_id_j"] == animal2)
            ) | (
                (long_format_df["animal_id_i"] == animal2)
                & (long_format_df["animal_id_j"] == animal1)
            )
            between_comparisons = long_format_df[between_mask]

            if len(between_comparisons) > 0:
                place_dist_matrix[i, j] = between_comparisons["pc_disparity"].mean()
                place_dist_matrix[j, i] = place_dist_matrix[i, j]
                notplace_dist_matrix[i, j] = between_comparisons["npc_disparity"].mean()
                notplace_dist_matrix[j, i] = notplace_dist_matrix[i, j]

    place_df = pd.DataFrame(place_dist_matrix, index=animals, columns=animals)
    notplace_df = pd.DataFrame(notplace_dist_matrix, index=animals, columns=animals)

    if plot:
        try:
            plot_animal_heatmap(
                place_df, notplace_df, metadata["animal_conditions"], pdf=pdf
            )
        except Exception as e:
            print(f"plot_animal_heatmap failed: {e}")

    return place_df, notplace_df


def compute_task_pairwise_distances_long(
    long_format_df: pd.DataFrame,
    task_groups: Dict[str, Dict[str, List[str]]] = None,
    embedding_methods: List[Literal["mds", "pca", "umap", "mds_pca"]] = ["pca"],
    plot: bool = True,
    combine_colorbar: bool = True,
    pdf: Optional[object] = None,
) -> pd.DataFrame:
    """Compute mean pairwise distances between tasks using long-format dataframe.

    Parameters:
    - long_format_df: Long-format DataFrame with pairwise comparisons (must have condition_i/j columns)
    - task_groups: Dictionary with condition names and task groupings
    - embedding_methods: List of embedding methods for visualization
    - plot: If True, will create plots
    - combine_colorbar: If True, uses a single colorbar for all condition subplots. If False, each condition subplot has its own colorbar.
    - pdf: PdfPages object to save plots to. If None, displays plots.

    Returns:
    - comparisson_df: DataFrame with computed distances and statistics
    """
    from calculations import _compute_paired_pvalue, _compute_unpaired_pvalue

    required_columns = [
        "condition",
        "task1_id",
        "task2_id",
        "paired_place_distance",
        "paired_notplace_distance",
        "paired_place_notplace_pvalue",
        "unpaired_place_distance",
        "unpaired_notplace_distance",
        "unpaired_place_entropy",
        "unpaired_notplace_entropy",
        "group_place_comparisson_pvalues",
        "group_notplace_comparisson_pvalues",
        "n_common_animals",
        "paired_place_std",
        "paired_notplace_std",
        "unpaired_place_std",
        "unpaired_notplace_std",
        "mean_place_cell_count",
        "mean_notplace_cell_count",
    ]
    comparisson_df = pd.DataFrame(columns=required_columns)

    # Extract metadata from long format dataframe
    metadata = _extract_metadata_from_long_df(long_format_df)
    conditions = metadata["conditions"]

    # Get all unique tasks
    tasks = metadata["tasks"]
    print(f"Global tasks: {tasks}")

    n_tasks = len(tasks)
    if n_tasks < 2:
        print("Skipped due to insufficient tasks globally")
        return comparisson_df

    # Precompute condition-specific animal-task combinations
    condition_animal_tasks = {}
    for condition in conditions:
        # Get animals with their tasks for this condition
        cond_rows_i = long_format_df[long_format_df["condition_i"] == condition][
            ["animal_id_i", "task_name_i"]
        ].drop_duplicates()
        cond_rows_j = long_format_df[long_format_df["condition_j"] == condition][
            ["animal_id_j", "task_name_j"]
        ].drop_duplicates()
        cond_rows_i.columns = ["animal", "task"]
        cond_rows_j.columns = ["animal", "task"]
        cond_animal_tasks = pd.concat([cond_rows_i, cond_rows_j]).drop_duplicates()
        condition_animal_tasks[condition] = cond_animal_tasks
        print(
            f"Prepared condition {condition} with {len(cond_animal_tasks)} animal-task combinations"
        )

    # Iterate through task pairs
    for i, task1_id in enumerate(tasks):
        for j, task2_id in enumerate(tasks):
            if i >= j:
                continue

            # For each condition
            for condition in conditions:
                if condition not in condition_animal_tasks:
                    continue

                cond_animal_task_df = condition_animal_tasks[condition]

                # Get animals with both tasks in this condition
                animals_with_task1 = set(
                    cond_animal_task_df[cond_animal_task_df["task"] == task1_id][
                        "animal"
                    ].unique()
                )
                animals_with_task2 = set(
                    cond_animal_task_df[cond_animal_task_df["task"] == task2_id][
                        "animal"
                    ].unique()
                )
                common_animals = animals_with_task1.intersection(animals_with_task2)

                if len(common_animals) < 2:
                    print(
                        f"Tasks {task1_id} vs {task2_id} in {condition}: Skipped, only {len(common_animals)} common animals"
                    )
                    continue

                # Filter long_format_df for this task pair and common animals
                task_pair_mask = (
                    (
                        (long_format_df["task_name_i"] == task1_id)
                        & (long_format_df["task_name_j"] == task2_id)
                    )
                    | (
                        (long_format_df["task_name_i"] == task2_id)
                        & (long_format_df["task_name_j"] == task1_id)
                    )
                ) & (
                    long_format_df["animal_id_i"].isin(common_animals)
                    & long_format_df["animal_id_j"].isin(common_animals)
                )

                task_pair_comparisons = long_format_df[task_pair_mask]

                if len(task_pair_comparisons) == 0:
                    print(
                        f"Tasks {task1_id} vs {task2_id} in {condition}: No comparisons found"
                    )
                    continue

                # Separate paired (same animal) and unpaired (different animals) comparisons
                paired_mask = (
                    task_pair_comparisons["animal_id_i"]
                    == task_pair_comparisons["animal_id_j"]
                )
                paired_comparisons = task_pair_comparisons[paired_mask]


                # Remove duplicate unpaired comparisons
                unpaired_df = task_pair_comparisons[~paired_mask].copy()
                unpaired_df['pair_key'] = unpaired_df.apply(
                    lambda row: tuple(sorted([row["animal_id_i"], row["animal_id_j"]])),
                    axis=1
                )
                unpaired_comparisons = unpaired_df.drop_duplicates(subset='pair_key').drop(columns='pair_key')

                # Paired distances (within-animal)
                if len(paired_comparisons) > 0:
                    paired_place_vals = paired_comparisons["pc_disparity"].values
                    paired_notplace_vals = paired_comparisons["npc_disparity"].values
                    paired_place_dist = paired_place_vals.mean()
                    paired_notplace_dist = paired_notplace_vals.mean()
                    paired_place_std = paired_place_vals.std()
                    paired_notplace_std = paired_notplace_vals.std()

                    paired_pvalue, _ = _compute_paired_pvalue(
                        paired_place_vals,
                        paired_notplace_vals,
                        method="auto",
                        labels=[task1_id, task2_id],
                        group_name=f"{condition}_{task1_id}_vs_{task2_id}",
                        col="mean distances",
                    )
                else:
                    paired_place_dist = np.nan
                    paired_notplace_dist = np.nan
                    paired_place_std = np.nan
                    paired_notplace_std = np.nan
                    paired_pvalue = np.nan

                # Unpaired distances (between-animal)
                if len(unpaired_comparisons) > 0:
                    unpaired_place_vals = unpaired_comparisons["pc_disparity"].values
                    unpaired_notplace_vals = unpaired_comparisons[
                        "npc_disparity"
                    ].values
                    unpaired_place_dist = unpaired_place_vals.mean()
                    unpaired_notplace_dist = unpaired_notplace_vals.mean()
                    unpaired_place_std = unpaired_place_vals.std()
                    unpaired_notplace_std = unpaired_notplace_vals.std()

                    # Compute normalized entropy (density measure)
                    def compute_normalized_entropy(values, n_bins=10):
                        """
                        Compute normalized entropy of a distribution.
                        Normalized entropy = H / H_max, where H_max = log(n_bins)
                        
                        Returns:
                        - Normalized entropy in [0, 1], where:
                          - 0: minimum spread (all values in one bin)
                          - 1: maximum spread (uniform distribution across bins)
                        """
                        if len(values) < 2:
                            return np.nan
                        
                        # Create histogram
                        counts, _ = np.histogram(values, bins=n_bins)
                        
                        # Filter out zero counts
                        counts = counts[counts > 0]
                        
                        if len(counts) == 0:
                            return np.nan
                        
                        # Compute probabilities
                        probs = counts / counts.sum()
                        
                        # Compute entropy
                        entropy = -np.sum(probs * np.log(probs))
                        
                        # Normalize by maximum possible entropy
                        max_entropy = np.log(n_bins)
                        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        
                        return normalized_entropy
                    
                    unpaired_place_entropy = compute_normalized_entropy(unpaired_place_vals)
                    unpaired_notplace_entropy = compute_normalized_entropy(unpaired_notplace_vals)
                else:
                    unpaired_place_dist = np.nan
                    unpaired_notplace_dist = np.nan
                    unpaired_place_std = np.nan
                    unpaired_notplace_std = np.nan
                    unpaired_place_entropy = np.nan
                    unpaired_notplace_entropy = np.nan

                # Cell counts (average from both sides if available)
                if "place_cell_count_i" in task_pair_comparisons.columns:
                    mean_place_cells = (
                        task_pair_comparisons["place_cell_count_i"].mean()
                        + task_pair_comparisons["place_cell_count_j"].mean()
                    ) / 2
                    mean_notplace_cells = (
                        task_pair_comparisons["non_place_cell_count_i"].mean()
                        + task_pair_comparisons["non_place_cell_count_j"].mean()
                    ) / 2
                else:
                    mean_place_cells = np.nan
                    mean_notplace_cells = np.nan

                new_row = pd.DataFrame(
                    {
                        "condition": condition,
                        "task1_id": task1_id,
                        "task2_id": task2_id,
                        "paired_place_distance": paired_place_dist,
                        "paired_place_distances": [
                            paired_place_vals
                            if len(paired_comparisons) > 0
                            else np.array([])
                        ],
                        "paired_notplace_distance": paired_notplace_dist,
                        "paired_notplace_distances": [
                            paired_notplace_vals
                            if len(paired_comparisons) > 0
                            else np.array([])
                        ],
                        "paired_place_notplace_pvalue": paired_pvalue,
                        "unpaired_place_distance": unpaired_place_dist,
                        "unpaired_place_distances": [
                            unpaired_place_vals
                            if len(unpaired_comparisons) > 0
                            else np.array([])
                        ],
                        "unpaired_notplace_distance": unpaired_notplace_dist,
                        "unpaired_notplace_distances": [
                            unpaired_notplace_vals
                            if len(unpaired_comparisons) > 0
                            else np.array([])
                        ],
                        "unpaired_place_entropy": unpaired_place_entropy,
                        "unpaired_notplace_entropy": unpaired_notplace_entropy,
                        "paired_place_std": paired_place_std,
                        "paired_notplace_std": paired_notplace_std,
                        "unpaired_place_std": unpaired_place_std,
                        "unpaired_notplace_std": unpaired_notplace_std,
                        "mean_place_cell_count": mean_place_cells,
                        "mean_notplace_cell_count": mean_notplace_cells,
                        "group_place_comparisson_pvalues": None,
                        "group_notplace_comparisson_pvalues": None,
                        "n_common_animals": len(common_animals),
                    }
                )
                comparisson_df = pd.concat([comparisson_df, new_row], ignore_index=True)

    # Calculate group comparisons between conditions
    if len(conditions) >= 2:
        for i, task1_id in enumerate(tasks):
            for j, task2_id in enumerate(tasks):
                if i >= j:
                    continue

                task_pair_mask = (comparisson_df["task1_id"] == task1_id) & (
                    comparisson_df["task2_id"] == task2_id
                )
                comp_task_df = comparisson_df[task_pair_mask]

                if len(comp_task_df) >= 2:
                    cond_1_values = comp_task_df[
                        comp_task_df["condition"] == conditions[0]
                    ]
                    cond_2_values = comp_task_df[
                        comp_task_df["condition"] == conditions[1]
                    ]

                    if len(cond_1_values) > 0 and len(cond_2_values) > 0:
                        try:
                            group_place_pvalue, _, _ = _compute_unpaired_pvalue(
                                cond_1_values["paired_place_distances"].iloc[0],
                                cond_2_values["paired_place_distances"].iloc[0],
                                method="auto",
                                labels=[task1_id, task2_id],
                                group_name=f"{conditions[0]}_vs_{conditions[1]}_{task1_id}_vs_{task2_id}_grouped",
                                col="mean distances",
                            )
                            group_notplace_pvalue, _, _ = _compute_unpaired_pvalue(
                                cond_1_values["paired_notplace_distances"].iloc[0],
                                cond_2_values["paired_notplace_distances"].iloc[0],
                                method="auto",
                                labels=[task1_id, task2_id],
                                group_name=f"{conditions[0]}_vs_{conditions[1]}_{task1_id}_vs_{task2_id}_grouped",
                                col="mean distances",
                            )
                            comparisson_df.loc[
                                task_pair_mask, "group_place_comparisson_pvalues"
                            ] = group_place_pvalue
                            comparisson_df.loc[
                                task_pair_mask, "group_notplace_comparisson_pvalues"
                            ] = group_notplace_pvalue
                        except Exception as e:
                            print(
                                f"Error computing group comparisons for {task1_id} vs {task2_id}: {e}"
                            )

    if plot:
        # Plot heatmaps
        for metric, cmap, label, title in [
            (
                "paired_place_distance",
                "RdBu",
                "Mean Place Distance",
                "Paired mean Shape Distances (Place Cells)",
            ),
            (
                "paired_notplace_distance",
                "RdBu",
                "Mean Not-Place Distance",
                "Paired mean Shape Distances (Not-Place Cells)",
            ),
            (
                "paired_place_notplace_pvalue",
                "viridis_r",
                "P-values",
                "Paired Place Cells vs Not-Place Cells P-values",
            ),
            # (
            #     "unpaired_place_distance",
            #     "PRGn",
            #     "Mean Place Distance",
            #     "Unpaired mean Shape Distances (Place Cells)",
            # ),
            # (
            #     "unpaired_notplace_distance",
            #     "PRGn",
            #     "Mean Not-Place Distance",
            #     "Unpaired mean Shape Distances (Not-Place Cells)",
            # ),
            (
                "unpaired_place_entropy",
                "viridis",
                "Normalized Entropy",
                "Unpaired Place Cell Distribution Entropy (Spread)",
            ),
            (
                "unpaired_notplace_entropy",
                "viridis",
                "Normalized Entropy",
                "Unpaired Non-Place Cell Distribution Entropy (Spread)",
            ),
            (
                "group_place_comparisson_pvalues",
                "magma_r",
                "P-values",
                "Between Group Shape Distances P-values (Place Cells)",
            ),
            (
                "group_notplace_comparisson_pvalues",
                "magma_r",
                "P-values",
                "Between Group Shape Distances P-values (Not-Place Cells)",
            ),
        ]:
            vmin = None
            vmax = None
            if "group_" in metric:
                vmin = None
                vmax = None
            elif "pvalue" in metric:
                vmin = 0
                vmax = 0.05
            elif "entropy" in metric:
                vmin = 0
                vmax = 1

            plot_heatmap_matrix(
                fig_title=title,
                comparisson_df=comparisson_df,
                metric=metric,
                task_groups=task_groups,
                cmap_label=label,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                combine_colorbar=combine_colorbar,
                show_colorbar=True,
                pdf=pdf,
            )

        # Plot standard deviation heatmaps
        for metric, cmap, label, title in [
            (
                "paired_place_std",
                "Reds",
                "Std Dev",
                "Paired Place Cell Distance Std Dev",
            ),
            (
                "paired_notplace_std",
                "Blues",
                "Std Dev",
                "Paired Not-Place Cell Distance Std Dev",
            ),
            # (
            #     "unpaired_place_std",
            #     "Oranges",
            #     "Std Dev",
            #     "Unpaired Place Cell Distance Std Dev",
            # ),
            (
                "unpaired_notplace_std",
                "Purples",
                "Std Dev",
                "Unpaired Not-Place Cell Distance Std Dev",
            ),
        ]:
            plot_heatmap_matrix(
                fig_title=title,
                comparisson_df=comparisson_df,
                metric=metric,
                task_groups=task_groups,
                cmap_label=label,
                vmin=None,
                vmax=None,
                cmap=cmap,
                combine_colorbar=combine_colorbar,
                show_colorbar=True,
                pdf=pdf,
            )

        # Plot cell count heatmaps
        for metric, cmap, label, title in [
            (
                "mean_place_cell_count",
                "Greens",
                "Cell Count",
                "Mean Place Cell Count per Comparison",
            ),
            (
                "mean_notplace_cell_count",
                "YlOrBr",
                "Cell Count",
                "Mean Non-Place Cell Count per Comparison",
            ),
        ]:
            plot_heatmap_matrix(
                fig_title=title,
                comparisson_df=comparisson_df,
                metric=metric,
                task_groups=task_groups,
                cmap_label=label,
                vmin=None,
                vmax=None,
                cmap=cmap,
                combine_colorbar=combine_colorbar,
                show_colorbar=True,
                pdf=pdf,
            )

        # Convert long format back to wide format for embedding plots
        d_place, d_notplace = convert_long_to_wide(long_format_df)

        # Create task_data_df for embedding plots
        task_records = []
        for _, row in long_format_df.iterrows():
            task_records.append(
                {
                    "animal": row["animal_id_i"],
                    "task": row["task_name_i"],
                    "condition": row["condition_i"],
                    "color": row["color_i"],
                    "date": "2000-01-01",  # Placeholder date for embedding plots
                    "model": "cebra",  # Placeholder model for embedding plots
                }
            )
            task_records.append(
                {
                    "animal": row["animal_id_j"],
                    "task": row["task_name_j"],
                    "condition": row["condition_j"],
                    "color": row["color_j"],
                    "date": "2000-01-01",  # Placeholder date for embedding plots
                    "model": "cebra",  # Placeholder model for embedding plots
                }
            )
        embedding_task_data_df = pd.DataFrame(task_records).drop_duplicates()

        # Plot point cloud embeddings
        for embedding_method in embedding_methods:
            try:
                plot_pointcloud_embeddings(
                    task_data_df=embedding_task_data_df,
                    d_place=d_place,
                    d_notplace=d_notplace,
                    embedding_method=embedding_method,
                    plot_show=["samples"],
                    pdf=pdf,
                )
                plot_pointcloud_embeddings(
                    task_data_df=embedding_task_data_df,
                    d_place=d_place,
                    d_notplace=d_notplace,
                    embedding_method=embedding_method,
                    plot_show=["center", "flow"],
                    pdf=pdf,
                )
            except Exception as e:
                print(f"Error creating embedding plot with {embedding_method}: {e}")

    return comparisson_df


def convert_long_to_wide(
    long_format_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert long-format dataframe back to wide-format distance matrices for embedding plots.

    Parameters:
    - long_format_df: Long-format DataFrame with pairwise comparisons (must have condition_i/j and color_i/j columns)

    Returns:
    - d_place: Wide-format distance matrix for place cells
    - d_notplace: Wide-format distance matrix for non-place cells
    """
    # Create labels directly from the long format dataframe
    # Label format: Animal_Condition_Task
    label_map = {}
    all_labels = set()

    for _, row in long_format_df.iterrows():
        # Create label for i side
        label_i = f"{row['animal_id_i']}_{row['condition_i']}_{row['task_name_i']}"
        key_i = (row["animal_id_i"], row["task_name_i"])
        label_map[key_i] = label_i
        all_labels.add(label_i)

        # Create label for j side
        label_j = f"{row['animal_id_j']}_{row['condition_j']}_{row['task_name_j']}"
        key_j = (row["animal_id_j"], row["task_name_j"])
        label_map[key_j] = label_j
        all_labels.add(label_j)

    all_labels = sorted(list(all_labels))
    n = len(all_labels)

    # Initialize matrices with NaN
    place_matrix = np.full((n, n), np.nan)
    notplace_matrix = np.full((n, n), np.nan)

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    # Fill matrices
    for _, row in long_format_df.iterrows():
        label_i = label_map.get((row["animal_id_i"], row["task_name_i"]))
        label_j = label_map.get((row["animal_id_j"], row["task_name_j"]))

        if label_i and label_j and label_i in label_to_idx and label_j in label_to_idx:
            idx_i = label_to_idx[label_i]
            idx_j = label_to_idx[label_j]

            place_matrix[idx_i, idx_j] = row["pc_disparity"]
            place_matrix[idx_j, idx_i] = row["pc_disparity"]  # Ensure symmetry

            notplace_matrix[idx_i, idx_j] = row["npc_disparity"]
            notplace_matrix[idx_j, idx_i] = row["npc_disparity"]  # Ensure symmetry

    # Set diagonal to 0
    np.fill_diagonal(place_matrix, 0)
    np.fill_diagonal(notplace_matrix, 0)

    d_place = pd.DataFrame(place_matrix, index=all_labels, columns=all_labels)
    d_notplace = pd.DataFrame(notplace_matrix, index=all_labels, columns=all_labels)

    return d_place, d_notplace


def plot_all_tasks_heatmap_long(
    long_format_df: pd.DataFrame,
    pdf: Optional[object] = None,
):
    """Plot heatmaps for all tasks using long-format dataframe.

    Parameters:
    - long_format_df: Long-format DataFrame with pairwise comparisons (must have condition_i/j and color_i/j columns)
    - pdf: PdfPages object to save plot to. If None, displays plot.
    """
    # Convert to wide format for plotting
    d_place, d_notplace = convert_long_to_wide(long_format_df)

    # Create a simple task_data_df for compatibility with plot_all_tasks_heatmap
    # Extract unique (animal, task, condition) combinations
    task_records = []
    for _, row in long_format_df.iterrows():
        task_records.append(
            {
                "animal": row["animal_id_i"],
                "task": row["task_name_i"],
                "condition": row["condition_i"],
                "color": row["color_i"],
                "date": "2000-01-01",  # Placeholder date
                "model": "cebra",  # Placeholder model
            }
        )
        task_records.append(
            {
                "animal": row["animal_id_j"],
                "task": row["task_name_j"],
                "condition": row["condition_j"],
                "color": row["color_j"],
                "date": "2000-01-01",  # Placeholder date
                "model": "cebra",  # Placeholder model
            }
        )

    task_data_df = pd.DataFrame(task_records).drop_duplicates()
    task_data_df = create_task_labels(task_data_df)
    plot_all_tasks_heatmap(task_data_df, d_place, d_notplace, pdf=pdf)


def analyze_trends(df: pd.DataFrame, 
                    subject_col: str = "animal_id_i",
                    outcome_col: str = "pc_disparity_normalized",
                    additional_title: str = ""
                   ):
    from calculations import analyze_group_learning_curves
    ########################### only cFC condition #####################################
    cfc_paired_norm_better_d_df = df[
        (df["condition_i"] == "cFC") &
        (df["condition_j"] == "cFC")
    ].copy()

    learning_curve_results = analyze_group_learning_curves(
        cfc_paired_norm_better_d_df,
        subject_col=subject_col,
        outcome_col=outcome_col,
        group_col="only_task_name_i",
        time_col="digit_i",
        plot_predicted=True,
        show_diagnostics=False,
        colors=cfc_paired_norm_better_d_df["color_i"].unique().tolist(),
        additional_title=" - cFC Only Place Cells"+" "+additional_title,
    )

    ########################### only NS condition #######################################
    ns_paired_norm_better_d_df = df[
        (df["condition_i"] == "NS") &
        (df["condition_j"] == "NS")
    ].copy()
    learning_curve_results_ns = analyze_group_learning_curves(
        ns_paired_norm_better_d_df,
        subject_col=subject_col,
        outcome_col=outcome_col,
        group_col="only_task_name_i",
        time_col="digit_i",
        plot_predicted=True,
        show_diagnostics=False,
        colors=ns_paired_norm_better_d_df["color_i"].unique().tolist(),
        additional_title=" - No Schock Only Place Cells"+" "+additional_title,
    )

    ########################### FS tasks #######################################
    fs_paired_norm_better_d_df = df[
        (df["only_task_name_i"].str.startswith("FS")) &
        (df["only_task_name_j"].str.startswith("FS"))
    ].copy()
    learning_curve_results_ns = analyze_group_learning_curves(
        fs_paired_norm_better_d_df,
        subject_col=subject_col,
        outcome_col=outcome_col,
        group_col="condition_i",
        time_col="digit_i",
        plot_predicted=True,
        show_diagnostics=False,
        colors=fs_paired_norm_better_d_df["color_i"].unique().tolist(),
        additional_title=" - FS Only Place Cells"+" "+additional_title,
    )
    ########################### NS tasks #######################################
    ns_paired_norm_better_d_df = df[
        (df["only_task_name_i"].str.startswith("NS")) &
        (df["only_task_name_j"].str.startswith("NS"))
    ].copy()
    learning_curve_results_ns = analyze_group_learning_curves(
        ns_paired_norm_better_d_df,
        subject_col=subject_col,
        outcome_col=outcome_col,
        group_col="condition_i",
        time_col="digit_i",
        plot_predicted=True,
        show_diagnostics=False,
        colors=ns_paired_norm_better_d_df["color_i"].unique().tolist(),
        additional_title=" - NS Only Place Cells"+" "+additional_title,
    )

def analyze_within_vs_between_animal_disparity(
    long_format_df: pd.DataFrame,
    plot: bool = True,
    save_dir: Optional[str] = None,
    method: Literal["auto", "wilcoxon", "ttest", "mannwhitneyu", "ttest_ind"] = "auto",
    correction_method: Literal["auto", "holm", "bonferroni", "fdr_bh", "none"] = "auto",
) -> Dict[str, pd.DataFrame]:
    """
    Compare within-animal vs between-animal disparities.

    This function analyzes whether animals have lower disparities when compared to themselves
    across different tasks (e.g., animal1_FS1 vs animal1_FS2) compared to when compared to
    other animals (e.g., animal1_FS1 vs animal2_FS2).

    Parameters:
    - long_format_df: DataFrame with columns ['animal_id_i', 'task_name_i', 'animal_id_j', 'task_name_j',
                                               'pc_disparity', 'npc_disparity']
    - plot: If True, creates violin plots and statistical comparisons
    - save_dir: Directory to save plots (optional)
    - method: Statistical test method
    - correction_method: Multiple comparison correction method

    Returns:
    - Dictionary with analysis results and statistics
    """
    from calculations import statistical_comparison

    # Create a copy to avoid modifying original
    df = add_tasknamestrings_and_digits(long_format_df.copy())


    df = filter_paired_comparison_type(df)

    # Remove self-comparisons (same animal AND same task)
    df = df[
        ~(
            (df["animal_id_i"] == df["animal_id_j"])
            & (df["task_name_i"] == df["task_name_j"])
        )
    ]

    # Create pair names for tracking
    df["pair_name"] = df.apply(
        lambda row: f"{row['animal_id_i']}_{row['task_name_i']}_vs_{row['animal_id_j']}_{row['task_name_j']}",
        axis=1,
    )

    # Get statistics
    print(f"\n{'=' * 60}")
    print("Within-Animal vs Between-Animal Disparity Analysis")
    print(f"{'=' * 60}")

    within_df = df[df["comparison_type"] == "within_animal"]
    between_df = df[df["comparison_type"] == "between_animal"]

    print(f"\nWithin-animal comparisons: {len(within_df)}")
    print(f"Between-animal comparisons: {len(between_df)}")

    # Place cell statistics
    # only calculate distance to task with digit in task_name +1

    print(f"\nPlace Cell Disparities:")
    within_pc_mean = within_df["pc_disparity"].mean()
    within_pc_std = within_df["pc_disparity"].std()
    between_pc_mean = between_df["pc_disparity"].mean()
    between_pc_std = between_df["pc_disparity"].std()

    print(f"  Within-animal:  {within_pc_mean:.4f} ± {within_pc_std:.4f}")
    print(f"  Between-animal: {between_pc_mean:.4f} ± {between_pc_std:.4f}")
    print(
        f"  Difference:     {within_pc_mean - between_pc_mean:.4f} ({'lower' if within_pc_mean < between_pc_mean else 'higher'} within-animal)"
    )

    # Non-place cell statistics
    print(f"\nNon-Place Cell Disparities:")
    within_npc_mean = within_df["npc_disparity"].mean()
    within_npc_std = within_df["npc_disparity"].std()
    between_npc_mean = between_df["npc_disparity"].mean()
    between_npc_std = between_df["npc_disparity"].std()

    print(f"  Within-animal:  {within_npc_mean:.4f} ± {within_npc_std:.4f}")
    print(f"  Between-animal: {between_npc_mean:.4f} ± {between_npc_std:.4f}")
    print(
        f"  Difference:     {within_npc_mean - between_npc_mean:.4f} ({'lower' if within_npc_mean < between_npc_mean else 'higher'} within-animal)"
    )
    print(f"{'=' * 60}\n")

    # Perform statistical tests using the statistical_comparison function
    results = statistical_comparison(
        df=df,
        pair_name_col="pair_name",
        compare_by="comparison_type",
        value_col=["pc_disparity", "npc_disparity"],
        test_type="unpaired",  # comparing two independent groups
        labels=["within_animal", "between_animal"],
        title="Within vs Between Animal",
        additional_title="Disparity Comparison",
        save_dir=save_dir,
        plot_heatmaps=False,
        plot_violins=plot,
        as_pdf=False,
        correction_method=correction_method,
        method=method,
        normality=True,
    )

    # Create summary DataFrame
    summary_data = {
        "metric": ["place_cell_disparity", "non_place_cell_disparity"],
        "within_animal_mean": [within_pc_mean, within_npc_mean],
        "within_animal_std": [within_pc_std, within_npc_std],
        "between_animal_mean": [between_pc_mean, between_npc_mean],
        "between_animal_std": [between_pc_std, between_npc_std],
        "difference": [
            within_pc_mean - between_pc_mean,
            within_npc_mean - between_npc_mean,
        ],
        "n_within": [len(within_df), len(within_df)],
        "n_between": [len(between_df), len(between_df)],
    }
    summary_df = pd.DataFrame(summary_data)

    return {
        "summary": summary_df,
        "statistical_results": results,
        "data": df,
    }


def _prepare_data_for_correlation(long_format_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data by handling _i/_j columns and creating average cell counts.

    Parameters:
    - long_format_df: Input dataframe with disparity and cell count columns

    Returns:
    - valid_data: DataFrame with averaged cell counts and no missing values
    """
    # Check if we have _i/_j columns or single columns
    has_paired_columns = "place_cell_count_i" in long_format_df.columns

    if has_paired_columns:
        # Calculate average cell counts from _i and _j columns
        valid_data = long_format_df.dropna(
            subset=[
                "pc_disparity",
                "npc_disparity",
                "place_cell_count_i",
                "place_cell_count_j",
                "non_place_cell_count_i",
                "non_place_cell_count_j",
            ]
        ).copy()

        # Create average cell count columns (using minimum for conservative estimate)
        valid_data["place_cell_count"] = valid_data[
            ["place_cell_count_i", "place_cell_count_j"]
        ].values.min(axis=1)
        valid_data["non_place_cell_count"] = valid_data[
            ["non_place_cell_count_i", "non_place_cell_count_j"]
        ].values.min(axis=1)
    else:
        # Use existing columns
        valid_data = long_format_df.dropna(
            subset=[
                "pc_disparity",
                "npc_disparity",
                "place_cell_count",
                "non_place_cell_count",
            ]
        ).copy()

    return valid_data


def _calculate_correlation(
    cell_count: pd.Series, disparity: pd.Series, method: str
) -> Tuple[float, float]:
    """Calculate correlation between cell count and disparity.

    Parameters:
    - cell_count: Series of cell counts
    - disparity: Series of disparity values
    - method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
    - correlation: Correlation coefficient
    - p_value: Statistical significance
    """
    from scipy import stats

    if method == "pearson":
        corr_func = stats.pearsonr
    elif method == "spearman":
        corr_func = stats.spearmanr
    elif method == "kendall":
        corr_func = stats.kendalltau
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return corr_func(cell_count, disparity)


def _evaluate_fit(
    x_values: np.ndarray, fit_result: dict, x_min: float, x_max: float
) -> np.ndarray:
    """Evaluate fitted function at given x values using interpolation.

    Parameters:
    - x_values: X values where to evaluate the fit
    - fit_result: Dictionary with fit results from get_best_fit
    - x_min: Minimum x value for the fit range
    - x_max: Maximum x value for the fit range

    Returns:
    - y_predicted: Predicted y values at x_values
    """
    # The y_dense is computed over the range [x_min, x_max] with 100 points
    x_dense = np.linspace(x_min, x_max, 100)
    y_dense = fit_result["y_dense"]

    # Interpolate to get predicted values at our actual x locations
    return np.interp(x_values, x_dense, y_dense)


def _normalize_disparities(
    valid_data: pd.DataFrame, fit_result_pc: dict, fit_result_npc: dict
) -> pd.DataFrame:
    """Normalize disparities based on fitted curves.

    Parameters:
    - valid_data: DataFrame with disparities and cell counts
    - fit_result_pc: Fit result for place cells
    - fit_result_npc: Fit result for non-place cells

    Returns:
    - valid_data: DataFrame with added normalized columns
    """
    # Calculate expected disparities from fit
    expected_pc_disparity = _evaluate_fit(
        valid_data["place_cell_count"].values,
        fit_result_pc,
        valid_data["place_cell_count"].min(),
        valid_data["place_cell_count"].max(),
    )
    expected_npc_disparity = _evaluate_fit(
        valid_data["non_place_cell_count"].values,
        fit_result_npc,
        valid_data["non_place_cell_count"].min(),
        valid_data["non_place_cell_count"].max(),
    )

    # Normalize: divide actual by expected (ratio normalization)
    valid_data["pc_disparity_normalized"] = (
        valid_data["pc_disparity"] / expected_pc_disparity
    )
    valid_data["npc_disparity_normalized"] = (
        valid_data["npc_disparity"] / expected_npc_disparity
    )

    # Also calculate residuals (difference from expected)
    valid_data["pc_disparity_residual"] = (
        valid_data["pc_disparity"] - expected_pc_disparity
    )
    valid_data["npc_disparity_residual"] = (
        valid_data["npc_disparity"] - expected_npc_disparity
    )

    return valid_data


def _generate_color_mapping(
    valid_data: pd.DataFrame, color_by: List[str]
) -> Tuple[List, Dict[str, tuple]]:
    """Generate color mapping from column combinations.

    Parameters:
    - valid_data: DataFrame with columns to color by
    - color_by: List of column names to combine for coloring

    Returns:
    - colors: List of colors for each row
    - color_legend: Dictionary mapping sorted pair names to colors
    """
    # Create combined keys (sorted to handle symmetry: FS1-FS2 == FS2-FS1)
    combined_keys = []
    for idx, row in valid_data.iterrows():
        values = [str(row[col]) for col in color_by]
        # Sort to ensure symmetry
        sorted_values = tuple(sorted(values))
        combined_keys.append("-".join(sorted_values))

    valid_data["_color_key"] = combined_keys

    # Get unique combinations
    unique_keys = sorted(valid_data["_color_key"].unique())
    n_colors = len(unique_keys)

    # Generate colors using the Vizualizer base color function
    base_colors = Vizualizer._get_base_color(index=np.arange(n_colors))
    base_colors = make_list_ifnot(base_colors)

    # Create mapping
    color_legend = {key: base_colors[i] for i, key in enumerate(unique_keys)}

    # Assign colors to each row
    colors = [color_legend[key] for key in combined_keys]

    return colors, color_legend


def _compute_binned_statistics(
    x_data: pd.Series, y_data: pd.Series, n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute binned mean and standard deviation for line+std visualization.

    Parameters:
    - x_data: X values
    - y_data: Y values
    - n_bins: Number of bins to create

    Returns:
    - bin_centers: Center x values of each bin
    - bin_means: Mean y values in each bin
    - bin_stds: Standard deviation of y values in each bin
    """
    from scipy import stats as scipy_stats

    # Create bins
    bin_means, bin_edges, _ = scipy_stats.binned_statistic(
        x_data, y_data, statistic="mean", bins=n_bins
    )
    bin_stds, _, _ = scipy_stats.binned_statistic(
        x_data, y_data, statistic="std", bins=n_bins
    )

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove NaN values
    valid_mask = ~np.isnan(bin_means) & ~np.isnan(bin_stds)

    return bin_centers[valid_mask], bin_means[valid_mask], bin_stds[valid_mask]


def _create_scatter_subplot(
    ax,
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    colors=None,
    fit_result: dict = None,
    default_color: str = "C0",
    add_legend: bool = True,
    show: List[str] = ["scatter"],
    color_keys: Optional[pd.Series] = None,
    color_legend: Optional[Dict] = None,
):
    """Create a scatter plot with optional fitted line, line plot, and standard deviation bands.

    Parameters:
    - ax: Matplotlib axis object
    - x_data: X values
    - y_data: Y values
    - x_label: Label for x-axis
    - y_label: Label for y-axis
    - title: Plot title
    - colors: Optional list of colors for points
    - fit_result: Optional fit result dictionary to plot fitted line
    - default_color: Default color if colors not provided
    - add_legend: Whether to add legend
    - show: List of visualization types - 'scatter', 'line', 'std'
    - color_keys: Optional Series with group labels for each point (for grouped line/std)
    - color_legend: Optional dictionary mapping labels to colors (for grouped line/std)
    """
    # Scatter plot
    if "scatter" in show:
        if colors is not None:
            ax.scatter(
                x_data,
                y_data,
                alpha=0.5,
                s=30,
                c=colors,
                edgecolors="white",
                linewidths=0.5,
                label="Data points" if add_legend else None,
            )
        else:
            ax.scatter(
                x_data,
                y_data,
                alpha=0.5,
                s=30,
                color=default_color,
                edgecolors="white",
                linewidths=0.5,
                label="Data points" if add_legend else None,
            )

    # Line plot with optional std bands
    if "line" in show or "std" in show:
        if color_keys is not None and color_legend is not None:
            # Compute binned statistics separately for each group
            for label, color in sorted(color_legend.items()):
                # Get data for this group
                mask = color_keys == label
                if mask.sum() < 3:  # Need at least 3 points for meaningful binning
                    continue

                group_x = x_data[mask]
                group_y = y_data[mask]

                # Compute binned statistics for this group
                bin_centers, bin_means, bin_stds = _compute_binned_statistics(
                    group_x, group_y, n_bins=10
                )

                if len(bin_centers) == 0:
                    continue

                if "line" in show:
                    # Plot line connecting bin means with matching color
                    ax.plot(
                        bin_centers,
                        bin_means,
                        color=color,
                        linewidth=2,
                        alpha=0.8,
                        label=f"{label} (mean)" if add_legend else None,
                        zorder=5,
                    )

                if "std" in show:
                    # Plot standard deviation bands with matching color
                    ax.fill_between(
                        bin_centers,
                        bin_means - bin_stds,
                        bin_means + bin_stds,
                        color=color,
                        alpha=0.2,
                        label=f"{label} (±1 std)" if add_legend else None,
                        zorder=4,
                    )
        else:
            # Fallback to single group statistics (original behavior)
            bin_centers, bin_means, bin_stds = _compute_binned_statistics(
                x_data, y_data, n_bins=20
            )

            if "line" in show:
                # Plot line connecting bin means
                ax.plot(
                    bin_centers,
                    bin_means,
                    color="blue",
                    linewidth=2,
                    alpha=0.8,
                    label="Binned mean" if add_legend else None,
                    zorder=5,
                )

            if "std" in show:
                # Plot standard deviation bands
                ax.fill_between(
                    bin_centers,
                    bin_means - bin_stds,
                    bin_means + bin_stds,
                    color="blue",
                    alpha=0.2,
                    label="±1 std" if add_legend else None,
                    zorder=4,
                )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add fitted line if provided
    if fit_result is not None:
        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(
            x_trend,
            fit_result["y_dense"],
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"Best fit: {fit_result['function_name']} (MSE={fit_result['mse']:.3g})",
            zorder=6,
        )

    if add_legend:
        ax.legend()

    ax.grid(True, alpha=0.3)


def _plot_correlation_analysis(
    valid_data: pd.DataFrame,
    place_corr: float,
    place_pval: float,
    notplace_corr: float,
    notplace_pval: float,
    fit_result_pc: dict,
    fit_result_npc: dict,
    method: str,
    normalize: bool,
    title: str = "",
    sharex: Union[bool, Literal["row", "col"]] = False,
    sharey: Union[bool, Literal["row", "col"]] = False,
    place_corr_norm: Optional[float] = None,
    place_pval_norm: Optional[float] = None,
    notplace_corr_norm: Optional[float] = None,
    notplace_pval_norm: Optional[float] = None,
    colors: Optional[List] = None,
    color_legend: Optional[Dict] = None,
    show: List[Literal["scatter", "line", "std"]] = ["scatter"],
):
    """Create correlation analysis plots.

    Parameters:
    - valid_data: DataFrame with cell counts and disparities
    - place_corr, place_pval: Place cell correlation results
    - notplace_corr, notplace_pval: Non-place cell correlation results
    - fit_result_pc, fit_result_npc: Fit results for both cell types
    - method: Correlation method used
    - normalize: Whether normalization was applied
    - place_corr_norm, place_pval_norm: Normalized place cell results (if normalize=True)
    - notplace_corr_norm, notplace_pval_norm: Normalized non-place cell results (if normalize=True)
    - colors: List of colors for each data point
    - color_legend: Dictionary mapping labels to colors
    - show: List of visualization types to include ('scatter', 'line', 'std')
    - sharex: If True, all subplots share the x-axis (sharex=True). If 'row' or 'col', behaves like matplotlib 'row'/'col'.
    - sharey: If True, all subplots share the y-axis (sharey=True). If 'row' or 'col', behaves like matplotlib 'row'/'col'.
    """
    # Create figure
    n_rows = 2 if normalize else 1
    # Interpret sharex/sharey:
    # - If True: pass True through to matplotlib so all subplots share the axis
    # - If string ('row' or 'col'): pass through to matplotlib to share per-row or per-column
    sharex_arg = True if sharex is True else sharex if isinstance(sharex, str) else False
    sharey_arg = True if sharey is True else sharey if isinstance(sharey, str) else False
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(14, 6 * n_rows), sharex=sharex_arg, sharey=sharey_arg
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Extract color keys if available (for grouped line/std plots)
    color_keys = valid_data.get("_color_key", None)

    # Row 1: Raw disparities
    _create_scatter_subplot(
        axes[0, 0],
        valid_data["place_cell_count"],
        valid_data["pc_disparity"],
        "Place Cell Count",
        "Place Cell Disparity",
        f"Place Cells (Raw)\n{method.capitalize()} r={place_corr:.3f}, p={place_pval:.2e}",
        colors=colors,
        fit_result=fit_result_pc,
        default_color="C0",
        show=show,
        color_keys=color_keys,
        color_legend=color_legend,
    )

    _create_scatter_subplot(
        axes[0, 1],
        valid_data["non_place_cell_count"],
        valid_data["npc_disparity"],
        "Non-Place Cell Count",
        "Non-Place Cell Disparity",
        f"Non-Place Cells (Raw)\n{method.capitalize()} r={notplace_corr:.3f}, p={notplace_pval:.2e}",
        colors=colors,
        fit_result=fit_result_npc,
        default_color="orange",
        show=show,
        color_keys=color_keys,
        color_legend=color_legend,
    )

    # Row 2: Normalized disparities (if requested)
    if normalize:
        _create_scatter_subplot(
            axes[1, 0],
            valid_data["place_cell_count"],
            valid_data["pc_disparity_normalized"],
            "Place Cell Count",
            "Normalized Place Cell Disparity",
            f"Place Cells (Normalized)\n{method.capitalize()} r={place_corr_norm:.3f}, p={place_pval_norm:.2e}",
            colors=colors,
            default_color="C0",
            add_legend=False,
            show=show,
            color_keys=color_keys,
            color_legend=color_legend,
        )
        axes[1, 0].axhline(
            y=1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Expected (normalized=1)",
        )
        axes[1, 0].legend()

        _create_scatter_subplot(
            axes[1, 1],
            valid_data["non_place_cell_count"],
            valid_data["npc_disparity_normalized"],
            "Non-Place Cell Count",
            "Normalized Non-Place Cell Disparity",
            f"Non-Place Cells (Normalized)\n{method.capitalize()} r={notplace_corr_norm:.3f}, p={notplace_pval_norm:.2e}",
            colors=colors,
            default_color="orange",
            add_legend=False,
            show=show,
            color_keys=color_keys,
            color_legend=color_legend,
        )
        axes[1, 1].axhline(
            y=1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Expected (normalized=1)",
        )
        axes[1, 1].legend()

    # Add color legend if we have one (only for scatter plots without line/std)
    # If line/std is shown, legend is already included in the plots
    if color_legend is not None and "line" not in show and "std" not in show:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in sorted(color_legend.items())
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(5, len(legend_elements)),
            title="Task Combinations",
            frameon=True,
        )
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_distance_cell_count_correlation(
    long_format_df: pd.DataFrame,
    plot: bool = True,
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
    normalize: bool = True,
    color_by="mono",  # Can be "mono", "combined", or list of column names
    show: List[Literal["scatter", "line", "std"]] = ["scatter"],
    sharex: Union[bool, Literal["row", "col"]] = True,
    sharey: Union[bool, Literal["row", "col"]] = False,
    title: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Analyze correlation between distances and cell counts, with optional normalization.

    This function checks if pc_disparity and npc_disparity are correlated with
    place_cell_count and non_place_cell_count respectively in the long-format dataframe.
    It fits curves to the data and optionally normalizes disparities based on cell count.

    Parameters:
    - long_format_df: DataFrame with columns ['animal_id_i', 'task_name_i', 'animal_id_j', 'task_name_j',
                                               'pc_disparity', 'npc_disparity', 'place_cell_count', 'non_place_cell_count']
    - plot: If True, creates scatter plots showing the correlations
    - method: Correlation method - 'pearson', 'spearman', or 'kendall'
    - normalize: If True, normalizes disparities based on fitted curve
    - color_by: Color scheme for scatter plots:
                - 'mono': single color for all points
                - 'combined': use combined_color column (backward compatibility)
                - list of column names: e.g., ['task_name_i', 'task_name_j'] to color by task pairs
    - show: List of visualization types to display:
            - 'scatter': scatter plot of all data points
            - 'line': line plot connecting binned means
            - 'std': standard deviation bands around binned means
            Can combine multiple options, e.g., ['scatter', 'line', 'std']
    - title: Optional custom title for the plots. If None, uses default title.
    - sharex: If True, all subplots share the x-axis (sharex=True).
              If 'row', x-axis is shared across all columns per row. If 'col', x-axis is shared across rows per column.
    - sharey: If True, all subplots share the y-axis (sharey=True).
              If 'row', y-axis is shared across all columns per row. If 'col', y-axis is shared across rows per column.

    Returns:
    - Dictionary with correlation results and normalized dataframe:
        {
            'place_cells': {'correlation': float, 'p_value': float},
            'non_place_cells': {'correlation': float, 'p_value': float},
            'normalized_df': pd.DataFrame (with normalized columns added),
            'color_legend': dict (if list of columns provided for color_by)
        }
    """
    from calculations import get_best_fit

    df = add_tasknamestrings_and_digits(long_format_df)

    # Prepare data
    valid_data = _prepare_data_for_correlation(df)


    if len(valid_data) == 0:
        print("No valid data for correlation analysis")
        return None

    # Calculate correlations for both cell types
    place_corr, place_pval = _calculate_correlation(
        valid_data["place_cell_count"], valid_data["pc_disparity"], method
    )
    notplace_corr, notplace_pval = _calculate_correlation(
        valid_data["non_place_cell_count"], valid_data["npc_disparity"], method
    )

    # Fit curves to the data
    fit_result_pc = get_best_fit(
        valid_data["place_cell_count"].tolist(),
        valid_data["pc_disparity"].tolist(),
        plot=False,
    )
    fit_result_npc = get_best_fit(
        valid_data["non_place_cell_count"].tolist(),
        valid_data["npc_disparity"].tolist(),
        plot=False,
    )

    # Normalize disparities if requested
    place_corr_norm, place_pval_norm = None, None
    notplace_corr_norm, notplace_pval_norm = None, None

    if normalize:
        valid_data = _normalize_disparities(valid_data, fit_result_pc, fit_result_npc)

        # Calculate correlations for normalized values
        place_corr_norm, place_pval_norm = _calculate_correlation(
            valid_data["place_cell_count"],
            valid_data["pc_disparity_normalized"],
            method,
        )
        notplace_corr_norm, notplace_pval_norm = _calculate_correlation(
            valid_data["non_place_cell_count"],
            valid_data["npc_disparity_normalized"],
            method,
        )

    # Build results dictionary
    results = {
        "place_cells": {
            "correlation": place_corr,
            "p_value": place_pval,
            "n_samples": len(valid_data),
            "fit": fit_result_pc,
        },
        "non_place_cells": {
            "correlation": notplace_corr,
            "p_value": notplace_pval,
            "n_samples": len(valid_data),
            "fit": fit_result_npc,
        },
        "normalized_df": valid_data,
    }

    if normalize:
        results["place_cells"]["correlation_normalized"] = place_corr_norm
        results["place_cells"]["p_value_normalized"] = place_pval_norm
        results["non_place_cells"]["correlation_normalized"] = notplace_corr_norm
        results["non_place_cells"]["p_value_normalized"] = notplace_pval_norm

    # Print results
    print(f"\n{'=' * 60}")
    if title:
        print(f"{title}")
    else:
        print(f"Distance-Cell Count Correlation Analysis ({method.capitalize()})")
    print(f"{'=' * 60}")
    print("\nPlace Cells:")
    print(f"  Correlation (raw): {place_corr:.4f}")
    print(f"  P-value (raw): {place_pval:.4e}")
    print(f"  Significant: {'Yes' if place_pval < 0.05 else 'No'}")
    print(
        f"  Best fit: {fit_result_pc['function_name']} (MSE={fit_result_pc['mse']:.3g})"
    )
    if normalize:
        print(f"  Correlation (normalized): {place_corr_norm:.4f}")
        print(f"  P-value (normalized): {place_pval_norm:.4e}")
        print(
            f"  Significant (normalized): {'Yes' if place_pval_norm < 0.05 else 'No'}"
        )

    print("\nNon-Place Cells:")
    print(f"  Correlation (raw): {notplace_corr:.4f}")
    print(f"  P-value (raw): {notplace_pval:.4e}")
    print(f"  Significant: {'Yes' if notplace_pval < 0.05 else 'No'}")
    print(
        f"  Best fit: {fit_result_npc['function_name']} (MSE={fit_result_npc['mse']:.3g})"
    )
    if normalize:
        print(f"  Correlation (normalized): {notplace_corr_norm:.4f}")
        print(f"  P-value (normalized): {notplace_pval_norm:.4e}")
        print(
            f"  Significant (normalized): {'Yes' if notplace_pval_norm < 0.05 else 'No'}"
        )

    print(f"\nSample size: {len(valid_data)} comparisons")
    print(f"{'=' * 60}\n")

    # Plotting
    if plot:
        # Determine colors based on color_by parameter
        colors = None
        color_legend = None

        if isinstance(color_by, list):
            # New functionality: color by column combinations
            colors, color_legend = _generate_color_mapping(valid_data, color_by)
            results["color_legend"] = color_legend
        elif color_by == "combined" and "combined_color" in valid_data.columns:
            # Backward compatibility: use existing combined_color column
            colors = valid_data["combined_color"].apply(parse_color_string).tolist()

        # Use the new plotting function
        _plot_correlation_analysis(
            title=title,
            valid_data=valid_data,
            place_corr=place_corr,
            place_pval=place_pval,
            notplace_corr=notplace_corr,
            notplace_pval=notplace_pval,
            fit_result_pc=fit_result_pc,
            fit_result_npc=fit_result_npc,
            method=method,
            normalize=normalize,
            place_corr_norm=place_corr_norm,
            place_pval_norm=place_pval_norm,
            notplace_corr_norm=notplace_corr_norm,
            notplace_pval_norm=notplace_pval_norm,
            colors=colors,
            sharex=sharex,
            sharey=sharey,
            color_legend=color_legend,
            show=show,
        )

    return results


def parse_color_string(color_str):
    """Convert color string back to tuple of floats"""
    import ast

    if isinstance(color_str, str):
        return ast.literal_eval(color_str)
    return color_str


def save_df(df, filename):
    filepath = Path.cwd().joinpath("logs").joinpath(filename)
    df.to_csv(filepath)
    print(f"Saved dataframe to {filepath}")


def load_df(filename):
    filepath = Path.cwd().joinpath("logs").joinpath(filename)
    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded dataframe from {filepath}")
    return df


def split_item_name(item_name):
    # example item name: DON-007021_20211031_FS1_place_cell
    parts = item_name.split("_")
    animal_id = parts[0]
    date = parts[1]
    task_name = parts[2]
    additional = parts[3] if len(parts) > 3 else None
    return animal_id, date, task_name, additional


def reformat(pc_df1, notpc_df2=None):
    # go through line by line
    new_df = pc_df1.copy()
    for idx, row in pc_df1.iterrows():
        item_i = row["item_i"]
        item_j = row["item_j"]
        animal_id_i, date_i, task_name_i, additional_i = split_item_name(item_i)
        animal_id_j, date_j, task_name_j, additional_j = split_item_name(item_j)
        new_df.at[idx, "animal_id_i"] = animal_id_i
        new_df.at[idx, "task_name_i"] = task_name_i
        new_df.at[idx, "animal_id_j"] = animal_id_j
        new_df.at[idx, "task_name_j"] = task_name_j
        new_df.at[idx, "pc_disparity"] = row["disparity"]

        if notpc_df2 is not None:
            # find the corresponding row in notpc_df2
            matching_row = notpc_df2[
                (notpc_df2["item_i"] == item_i) & (notpc_df2["item_j"] == item_j)
            ]
            if not matching_row.empty:
                disparity = matching_row.iloc[0]["disparity"]
                if isinstance(disparity, tuple):
                    disparity = disparity[0]
                new_df.at[idx, "npc_disparity"] = disparity

    # remove original item_i and item_j and disparity columns
    new_df = new_df.drop(columns=["item_i", "item_j", "disparity"])

    return new_df


# Helper function to get task information from task_data_df
def _get_task_info(
    task_data_df: pd.DataFrame, animal_id: str, task_name: str
) -> pd.Series:
    """
    Get task information for a specific animal and task.

    Parameters:
    - task_data_df: DataFrame with task metadata
    - animal_id: Animal identifier
    - task_name: Task name

    Returns:
    - pd.Series with task information, or None if not found
    """
    task_info = task_data_df[
        (task_data_df["animal"] == animal_id) & (task_data_df["task"] == task_name)
    ]
    return task_info.iloc[0] if not task_info.empty else None


def _add_task_metadata(
    enhanced_df: pd.DataFrame, idx: int, task_info: pd.Series, suffix: str
) -> None:
    """
    Add condition and color metadata for a single task to the dataframe.

    Parameters:
    - enhanced_df: DataFrame to modify
    - idx: Row index
    - task_info: Series with task information
    - suffix: Suffix for column names ('_i' or '_j')
    """
    if task_info is not None:
        enhanced_df.at[idx, f"condition{suffix}"] = task_info["condition"]
        color = task_info["color"]
        enhanced_df.at[idx, f"color{suffix}"] = str(
            color
        )  # Store as string for CSV compatibility
        return color
    return None


def _compute_combined_color(color_i, color_j) -> str:
    """
    Compute combined color as average of two colors.

    Parameters:
    - color_i: First color (tuple/list of RGBA values)
    - color_j: Second color (tuple/list of RGBA values)

    Returns:
    - String representation of combined color
    """
    import numpy as np

    if color_i is None or color_j is None:
        return str(color_i if color_i is not None else color_j)

    # convert string representations back to tuples/lists if necessary
    if isinstance(color_i, str):
        color_i = eval(color_i)
    if isinstance(color_j, str):
        color_j = eval(color_j)

    # Convert to numpy arrays and average (works for RGBA tuples)
    if isinstance(color_i, (tuple, list)) and isinstance(color_j, (tuple, list)):
        combined = tuple(np.mean([color_i, color_j], axis=0))
        return str(combined)

    # Fallback to first color
    return str(color_i)


def _add_cell_counts(enhanced_df, idx, task_info, suffix):
    """Helper to add cell count information for a task."""
    if task_info is not None and "cell_indices" in task_info:
        cell_indices = task_info["cell_indices"]
        count = (
            len(cell_indices)
            if cell_indices is not None and len(cell_indices) > 0
            else None
        )
        return count
    return None


def add_conditions_and_colors(
    better_d_df: pd.DataFrame,
    task_data_df: pd.DataFrame,
    task_data_df_notplace: pd.DataFrame = None,
    add_cell_counts: bool = False,
) -> pd.DataFrame:
    """
    Add condition and color information to existing better_d_df.

    This is the unified function that combines the functionality of the old
    add_conditions_and_colors() and add_all_enhancements() functions.

    This function enriches the pairwise comparison dataframe with metadata from task_data_df:
    - condition_i, condition_j: Experimental conditions for each task
    - color_i, color_j: Original RGBA colors for each task
    - combined_color: Averaged color representing the task-to-task comparison
    - Optional: place_cell_count_i/j and non_place_cell_count_i/j (if add_cell_counts=True)

    Parameters:
    - better_d_df: DataFrame with columns ['animal_id_i', 'task_name_i', 'animal_id_j', 'task_name_j', ...]
    - task_data_df: DataFrame with columns ['animal', 'task', 'condition', 'color', ...]
    - task_data_df_notplace: Optional DataFrame for non-place cell information (required if add_cell_counts=True)
    - add_cell_counts: If True, adds individual cell count columns for place and non-place cells

    Returns:
    - Enhanced DataFrame with added metadata columns
    """
    # Work with a copy to avoid modifying the original
    enhanced_df = better_d_df.copy()

    # Process each row
    for idx, row in enhanced_df.iterrows():
        # Get task information for both tasks (place cells)
        task_i_info = _get_task_info(
            task_data_df, row["animal_id_i"], row["task_name_i"]
        )
        task_j_info = _get_task_info(
            task_data_df, row["animal_id_j"], row["task_name_j"]
        )

        # Add metadata for task_i and task_j (condition, color)
        color_i = _add_task_metadata(enhanced_df, idx, task_i_info, "_i")
        color_j = _add_task_metadata(enhanced_df, idx, task_j_info, "_j")

        # Add place cell counts if requested
        if add_cell_counts:
            pc_count_i = _add_cell_counts(enhanced_df, idx, task_i_info, "_i")
            pc_count_j = _add_cell_counts(enhanced_df, idx, task_j_info, "_j")
            if pc_count_i is not None:
                enhanced_df.at[idx, "place_cell_count_i"] = pc_count_i
            if pc_count_j is not None:
                enhanced_df.at[idx, "place_cell_count_j"] = pc_count_j

        # Compute and store combined color
        enhanced_df.at[idx, "combined_color"] = _compute_combined_color(
            color_i, color_j
        )

        # Add non-place cell information if requested
        if add_cell_counts and task_data_df_notplace is not None:
            task_i_npc = _get_task_info(
                task_data_df_notplace, row["animal_id_i"], row["task_name_i"]
            )
            task_j_npc = _get_task_info(
                task_data_df_notplace, row["animal_id_j"], row["task_name_j"]
            )

            npc_count_i = _add_cell_counts(enhanced_df, idx, task_i_npc, "_i")
            npc_count_j = _add_cell_counts(enhanced_df, idx, task_j_npc, "_j")
            if npc_count_i is not None:
                enhanced_df.at[idx, "non_place_cell_count_i"] = npc_count_i
            if npc_count_j is not None:
                enhanced_df.at[idx, "non_place_cell_count_j"] = npc_count_j

    return enhanced_df

def add_tasknamestrings_and_digits(long_format_df: pd.DataFrame) -> pd.DataFrame:
    df = long_format_df.copy()
    def is_digit_task(task_name: str) -> bool:
        # extract digit from task name
        digit = ''.join(filter(str.isdigit, task_name))
        return int(digit)
    
    def get_taskname(task_name: str) -> bool:
        # extract digit from task name
        digit = str(is_digit_task(task_name))
        name = task_name.replace(digit, '')
        return name

    df = long_format_df.copy()
    
    # add digit for task_name_i and task_name_j
    df['digit_i'] = df['task_name_i'].apply(is_digit_task)
    df['digit_j'] = df['task_name_j'].apply(is_digit_task)

    df['only_task_name_i'] = df['task_name_i'].apply(get_taskname)
    df['only_task_name_j'] = df['task_name_j'].apply(get_taskname)
    return df

def filter_paired_comparison_type(
    df: pd.DataFrame,
    animal_id_i_col: str = "animal_id_i",
    animal_id_j_col: str = "animal_id_j",
    digit_i_col: str = "digit_i",
    digit_j_col: str = "digit_j",
    task_name_i_col: str = "only_task_name_i",
    task_name_j_col: str = "only_task_name_j",
) -> pd.DataFrame:
    """
    Classify comparisons as within-animal or between-animal.
    
    A comparison is classified as within-animal if:
    - Same animal (animal_id_i == animal_id_j)
    - Sequential tasks (digit_i == digit_j + 1)
    - Same task type (only_task_name_i == only_task_name_j)
    
    Parameters:
    - df: Input DataFrame
    - animal_id_i_col: Column name for first animal ID
    - animal_id_j_col: Column name for second animal ID
    - digit_i_col: Column name for first task digit
    - digit_j_col: Column name for second task digit
    - task_name_i_col: Column name for first task name (without digit)
    - task_name_j_col: Column name for second task name (without digit)
    
    Returns:
    - DataFrame with added 'comparison_type' column
    """
    df = df.copy()
    
    df["comparison_type"] = df.apply(
        lambda row: "within_animal"
        if (
            row[animal_id_i_col] == row[animal_id_j_col]
            and row[digit_i_col] == row[digit_j_col] - 1
            and row[task_name_i_col] == row[task_name_j_col]
        )
        else "between_animal",
        axis=1,
    )
    
    return df