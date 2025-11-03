# type hints
from typing import List, Union, Dict, Tuple, Optional, Literal, Callable
from pathlib import Path

# show progress bar
from tqdm import tqdm, trange

# calculations
import numpy as np
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from copy import deepcopy
from sklearn.manifold import (
    TSNE,
    MDS,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.decomposition import PCA
from umap.umap_ import UMAP

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot
from plotly.subplots import make_subplots
import sys

# optional local dependency: parallel_transport_unfolding
from parallel_transport_unfolding.ptu import PTU
from Visualizer import Vizualizer, plot_line, plot_simple_embedd
from Helper import *
from structure_index import compute_structure_index


def create_multiple_embeddings(
    distances: Union[np.ndarray, pd.DataFrame],
    embeddings: Union[
        bool,
        List[
            Literal[
                "tsne",
                "mds",
                "isomap",
                "ptu",
                "lle",
                "spectral_embedding",
                "umap",
                "pca",
                "mds_pca",
            ]
        ],
    ] = ["pca", "umap"],
    plot_df: Optional[pd.DataFrame] = None,
    colors: Optional[Union[np.ndarray, List]] = None,
    n_components: int = 2,
    n_neighbors: int = 10,
    plot_show: Literal[
        "center", "center_std", "samples", "flow", "annotate_dots"
    ] = "samples",
    additional_title: str = "",
    plot_save_dir: Optional[Path] = None,
    dissimilarity: Literal["precomputed", "euclidean"] = "precomputed",
    as_pdf: bool = False,
) -> None:
    """
    General helper to create multiple embedding subplots.

    Create a grid of embedding subplots from a single distance/feature input.

    This helper centralizes the common pattern of creating multiple dimensionality
    reduction plots (UMAP, t-SNE, MDS, PCA, etc.) side-by-side so they can be
    compared visually. It calls `simple_embedd(..., return_traces=True)` for each
    requested method and inserts the returned Plotly traces into a subplot grid.

    Notes
    -----
    - The function is display-oriented and returns None. It shows the Plotly
      figure and optionally writes an HTML file to `plot_save_dir`.
    - By default the helper forwards the input to `simple_embedd` with
      `dissimilarity='precomputed'` so it is safe to pass a square distance
      matrix (e.g. a disparity matrix). Passing a feature array is also
      permitted (depending on the embedding method) but `simple_embedd` will
      interpret it according to its `dissimilarity` argument.
    - `as_pdf` is kept for API compatibility but Plotly HTML output is used for
      interactivity. If PDF export is required, convert the HTML with a browser
      or use Plotly static image export (requires additional dependencies).

    Parameters
    ----------
    distances : np.ndarray or pd.DataFrame
        Input matrix or feature array to embed. Typically a precomputed square
        distance/disparity matrix when `dissimilarity='precomputed'` is used.
    embeddings : Union[bool, List[str]]
        If bool: True defaults to ["pca", "umap"], False skips execution.
        If List[str]: Names of embedding methods to generate, e.g. ['umap', 'tsne', 'pca'].
        These strings are forwarded to `simple_embedd(..., method=...)`.
    plot_df : pd.DataFrame, optional
        Optional metadata (colors/labels) forwarded to `simple_embedd` for
        plotting. See `simple_embedd` for expected columns (e.g. 'plot_label').
    n_components : int, default 2
        Dimensionality of the embedding (2 or 3). Only 2D and 3D are supported.
    plot_show : str, default 'samples'
        Plot type/style forwarded to `simple_embedd` (e.g. 'samples', 'center').
    additional_title : str, optional
        Extra text appended to the figure title.
    plot_save_dir : pathlib.Path, optional
        If provided, an HTML representation of the figure will be saved to this
        directory. Filenames are cleaned with `clean_filename`.
    as_pdf : bool, default False
        Placeholder for backward compatibility. PDF export is not performed
        by this helper.

    Returns
    -------
    None
        The function displays the resulting Plotly figure and may save an
        interactive HTML file if `plot_save_dir` is provided.
    """
    # Handle bool case for backward compatibility
    if isinstance(embeddings, bool):
        if embeddings:
            embeddings = ["pca", "umap"]
        else:
            return

    n_plots = len(embeddings)
    if n_plots == 0:
        return

    # Determine subplot grid layout
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    figsize = (n_cols * 7, n_rows * 7)

    # Create subplot figure
    if n_components == 2:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[e.upper() for e in embeddings],
            horizontal_spacing=0.04,
            vertical_spacing=0.04,
        )
    elif n_components == 3:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[e.upper() for e in embeddings],
            specs=[[{"type": "scene"}] * n_cols] * n_rows,
            horizontal_spacing=0.04,
            vertical_spacing=0.04,
        )
    else:
        raise ValueError(
            f"Unsupported n_components: {n_components}. Only 2D and 3D embeddings are supported."
        )

    # Generate each embedding using the helper function
    for i, embedding_method in enumerate(embeddings):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0  # Only first subplot shows legend

        # Use the helper function to add subplot
        _add_embedding_subplot(
            fig=fig,
            distances=distances,
            plot_df=plot_df,
            row=row,
            col=col,
            colors=colors,
            subtitle=embedding_method.upper(),
            n_components=n_components,
            n_neighbors=n_neighbors,
            plot_show=plot_show,
            embedding_method=embedding_method,
            dissimilarity=dissimilarity,
            plot_save_dir=plot_save_dir,
            legend=show_legend,
        )

    # Create overall figure title
    suptitle = (
        f"{', '.join([e.upper() for e in embeddings])} Shape Similarity Embeddings - "
        f"{additional_title} {n_components}D {plot_show}"
    ).strip()

    # Update figure layout with title and other properties
    fig.update_layout(
        title=dict(
            text=suptitle,
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        height=figsize[1] * 100,
        showlegend=True,
        legend=dict(traceorder="normal"),
    )

    # Disable auto-scaling and tick labels for cleaner appearance
    for subplot_idx in range(1, n_rows * n_cols + 1):
        subplot_row = (subplot_idx - 1) // n_cols + 1
        subplot_col = (subplot_idx - 1) % n_cols + 1
        fig.update_xaxes(
            row=subplot_row,
            col=subplot_col,
            scaleanchor=None,
            showticklabels=False,
            showgrid=False,  # Optional: hide grid for embeddings
        )
        fig.update_yaxes(
            row=subplot_row,
            col=subplot_col,
            scaleanchor=None,
            showticklabels=False,
            showgrid=False,
        )

    # Display and save figure
    fig.show()
    if plot_save_dir is not None:
        plot_save_dir = Path(plot_save_dir)
        plot_save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{clean_filename(suptitle)}.html"
        fig.write_html(str(plot_save_dir / filename))


def _add_embedding_subplot(
    fig,
    distances,
    plot_df,
    row: int,
    col: int,
    subtitle: str,
    n_components: int,
    plot_show: str,
    embedding_method: str,
    n_neighbors: int = 10,
    colors: Optional[Union[np.ndarray, List]] = None,
    plot_save_dir: Optional[Path] = None,
    dissimilarity: Literal["precomputed", "euclidean"] = "precomputed",
    legend: bool = False,
) -> dict:
    """
    Helper: compute embedding for a distances matrix, add traces to
    subplot (row,col) and copy the subplot title from the returned layout
    when available.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The subplot figure to add traces to.
    distances : np.ndarray or pd.DataFrame
        Input matrix or feature array to embed.
    plot_df : pd.DataFrame, optional
        Metadata for plotting.
    row, col : int
        Subplot position (1-based indexing).
    subtitle : str
        Subplot title.
    n_components : int
        Dimensionality (2 or 3).
    plot_show : str
        Plot style.
    embedding_method : str
        Embedding method name.
    plot_save_dir : Path, optional
        Directory for saving.
    legend : bool
        Whether to show legend.

    Returns
    -------
    layout : dict
        The layout dict returned by simple_embedd for potential use.
    """
    coords, traces, layout = simple_embedd(
        distances,
        plot_df=plot_df,
        ax=fig,
        colors=colors,
        additional_title=subtitle,
        dissimilarity=dissimilarity,
        n_components=n_components,
        n_neighbors=n_neighbors,
        method=embedding_method,
        plot_show=plot_show,
        save_dir=plot_save_dir,
        legend=legend,
        return_traces=True,
    )

    # Add traces to the specified subplot
    for trace in traces:
        fig.add_trace(trace, row=row, col=col)

    # Update subplot annotation/title from returned layout if available
    # compute annotation index (0-based): left-to-right, top-to-bottom
    # try:
    #     ann_idx = (row - 1) * fig.layout.grid_cols + (col - 1)
    #     if "title" in layout and len(fig.layout.annotations) > ann_idx:
    #         fig.layout.annotations[ann_idx]["text"] = layout.get("title", subtitle)
    # except Exception:
    #     pass

    return layout


def decode(
    embedding_train: np.ndarray,
    embedding_test: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    labels_describe_space: bool = False,
    n_neighbors: Optional[int] = None,
    circular_values: bool = False,
    max_value: Optional[float] = None,
    metric: str = "cosine",
    n_folds: int = 5,
    detailed_metrics: bool = False,
    include_cv_stats: bool = False,
    multiply_by: float = 1.0,
    test_outlier_removal: bool = True,
    regression_outlier_removal_threshold: float = 0.004,
    min_train_class_samples: int = 200,
    min_test_class_samples: int = 30,
) -> Dict[str, Dict[str, Union[float, Dict]]]:
    """
    Decodes neural embeddings using k-Nearest Neighbors with automatic k selection.

    Before decoding, the function checks if the input data is valid and ensures that the training and test sets are compatible.
    Labels/Classes are checked to ensure that only those with sufficient training samples are used for testing.
    This function performs k-fold cross-validation to determine the optimal number of neighbors (k) for the kNN model.
    It then fits the kNN model with the optimal k and evaluates its performance on the test set.

    Parameters
    ----------
    embedding_train : np.ndarray
        Training embedding data
    embedding_test : np.ndarray
        Testing embedding data
    labels_train : np.ndarray
        Training target labels
    labels_test : np.ndarray
        Testing target labels
    labels_describe_space : bool, optional
        Whether to describe the label space (default: False).
        If True, the labels are assumed to be continuous and the prediction error is converted to a single number in euclidean space instead of multidimensional.
    n_neighbors : int, optional
        Number of neighbors for kNN (default: None, auto-determined via CV)
    metric : str, optional
        Distance metric for kNN (default: "cosine")
    n_folds : int, optional
        Number of folds for cross-validation (default: 5)
    detailed_metrics : bool, optional
        Whether to return detailed per-class metrics (default: False)
    include_cv_stats : bool, optional
        Whether to include cross-validation statistics (default: False)
    labels_describe_space : bool, optional
        If labels are describing space (default is False).
        If space is described the Frobenius norm is used to calculate the distance between decoded positional values.
    multiply_by : float, optional
        A multiplier to apply to the decoding statistics (default is 1.0). This is used to scale the decoding statistics.
    test_outlier_removal : bool, optional
        Whether to remove outliers from the test set (default: True)
    regression_outlier_removal_threshold : float, optional
        Threshold for outlier removal (default: 0.004). This is used to determine if a test sample is too far from the training samples.
        If the distance to the nearest training sample is greater than this threshold, the test sample is removed.
    min_train_class_samples : int, optional
        Minimum number of training samples per class (default: 200). This is used to determine if a class has enough training samples.
    min_test_class_samples : int, optional
        Minimum number of test samples per class (default: 30). This is used to determine if a class has enough test samples.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing decoding performance metrics
    """
    # Input validation
    if not all(
        isinstance(x, np.ndarray)
        for x in [embedding_train, embedding_test, labels_train, labels_test]
    ):
        raise ValueError("All input arrays must be numpy arrays")

    # Ensure 2D arrays
    labels_train = np.atleast_2d(labels_train)
    labels_test = np.atleast_2d(labels_test)

    # Determine if regression or classification
    is_regression = is_floating(labels_train) if not labels_describe_space else True
    knn_class = KNeighborsRegressor if is_regression else KNeighborsClassifier

    # Ensure only labels sufficiently available in training set are used for testing
    if test_outlier_removal:
        idx_remove = []
        if is_regression:
            # check if the test data is within range of the training data
            mins = np.min(labels_train, axis=0)
            maxs = np.max(labels_train, axis=0)
            ranges = maxs - mins
            if labels_describe_space:
                area = np.prod(ranges)
                min_acceptable_value = np.sqrt(
                    area * regression_outlier_removal_threshold
                )
            else:
                min_acceptable_value = ranges * regression_outlier_removal_threshold
            for k, loc in enumerate(labels_test):
                diff = loc - labels_train
                dist = (
                    np.linalg.norm(loc - labels_train, axis=1)
                    if labels_describe_space
                    else diff
                )
                cl = np.min(dist)
                if cl > min_acceptable_value:
                    global_logger.debug(
                        f"Test sample {k} will be removed from training set because too far {cl:.4f} from available training points",
                    )
                    idx_remove.append(k)
        else:
            # check if the test data class has sufficiently available training data samples
            for cl, num_test_samples in np.unique(labels_test, treturn_counts=True):
                num_train_samples = np.sum(labels_train == cl)
                if (
                    num_train_samples < min_train_class_samples
                    or num_test_samples < min_test_class_samples
                ):
                    idx_remove.extend(np.where(labels_test == cl)[0])
                    global_logger.debug(
                        f"{num_train_samples} Test samples will be removed from training set because class {cl} has too few samples. Training samples: {num_train_samples}, Test samples: {num_test_samples}. At least {min_train_class_samples} training samples and {min_test_class_samples} test samples are needed."
                    )

        if len(idx_remove) > 0:
            global_logger.debug(
                f"Removing {len(idx_remove)} test samples from training set"
            )
            embedding_test = np.delete(embedding_test, idx_remove, axis=0)
            labels_test = np.delete(labels_test, idx_remove, axis=0)

    # Define range of k values to test if n_neighbors is None
    if n_neighbors is None:
        max_k = min(embedding_train.shape[0] - 1, 50)  # Cap at 50 or n_samples-1
        k_range = np.unique(
            np.logspace(0, np.log10(max_k), num=10, base=10).astype(int)
        )
        # Internal CV to select best k
        global_logger.debug(f"Performing internal CV to select best k")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        k_scores = []

        for k in k_range:
            knn_model = knn_class(n_neighbors=k, metric=metric)
            fold_scores = []

            for train_idx, val_idx in kf.split(embedding_train):
                X_train_fold = embedding_train[train_idx]
                X_val_fold = embedding_train[val_idx]
                y_train_fold = labels_train[train_idx]
                y_val_fold = labels_train[val_idx]

                knn_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = knn_model.predict(X_val_fold)

                score = (
                    r2_score(y_val_fold, y_pred_fold)
                    if is_regression
                    else accuracy_score(y_val_fold, y_pred_fold)
                )
                fold_scores.append(score)

            k_scores.append(np.mean(fold_scores))

        # Select best k
        best_k = k_range[np.argmax(k_scores)]
        global_logger.debug(f"Best k: {best_k}")

    else:
        best_k = n_neighbors

    # Initialize final kNN model with best k
    knn_model = knn_class(n_neighbors=best_k, metric=metric)

    # Optional k-fold cross-validation for statistics (if requested)
    cv_results = []
    if include_cv_stats:
        for train_idx, val_idx in kf.split(embedding_train):
            X_train_fold = embedding_train[train_idx]
            X_val_fold = embedding_train[val_idx]
            y_train_fold = labels_train[train_idx]
            y_val_fold = labels_train[val_idx]

            knn_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = knn_model.predict(X_val_fold)
            cv_results.append({"true": y_val_fold, "pred": np.atleast_2d(y_pred_fold)})

    # Final fit on full training data and predict on test
    knn_model.fit(embedding_train, labels_train)
    test_predictions = np.atleast_2d(knn_model.predict(embedding_test))

    # Calculate metrics based on label type
    results = {}
    if is_regression:
        results.update(
            _compute_regression_metrics(
                labels_test=labels_test,
                test_predictions=test_predictions,
                cv_results=cv_results if include_cv_stats else None,
                labels_describe_space=labels_describe_space,
                circular_values=circular_values,
                max_value=max_value,
                multiply_by=multiply_by,
            )
        )
    else:
        results.update(
            _compute_classification_metrics(
                labels_test,
                test_predictions,
                cv_results if include_cv_stats else None,
                detailed_metrics,
            )
        )

    results["k_neighbors"] = best_k
    return results


def _compute_regression_metrics(
    labels_test: np.ndarray,
    test_predictions: np.ndarray,
    cv_results: Optional[list] = None,
    labels_describe_space: bool = False,
    circular_values: bool = False,
    max_value: Optional[float] = None,
    multiply_by: float = 1.0,
) -> Dict[str, Union[float, Dict[str, Union[float, Dict]]]]:
    """Compute regression metrics with optional cross-validation results.

    Parameters
    ----------
    multiply_by : float, optional
        Factor to multiply the results by to normalize if original values have been transformed before (default: 1.0)
    """
    # Test set metrics
    err = np.abs(labels_test - test_predictions)
    # Handle circular values by wrapping around
    if circular_values:
        if max_value is None:
            do_critical(
                ValueError,
                "DECODING value correction Failed: max_value must be provided if values are circular to prevent wrong decoding results.",
            )
        err = np.minimum(err, max_value - err)

    err *= multiply_by
    if labels_describe_space:
        err = np.linalg.norm(err, axis=1)
        rmse = np.mean(err)
        error_variance = np.var(err)
        r2 = r2_score(labels_test, test_predictions)
    else:
        raise NotImplementedError(
            "Label space description is not implemented for regression metrics."
        )

    results = {
        "rmse": {"mean": float(rmse), "variance": float(error_variance)},
        "r2": float(r2),
    }

    # Include CV stats if requested
    if cv_results is not None:
        cv_rmse = [
            np.mean(np.abs(r["true"] - r["pred"])) * multiply_by for r in cv_results
        ]
        cv_r2 = [r2_score(r["true"], r["pred"]) for r in cv_results]
        results["cv_metrics"] = {
            "rmse": {"mean": float(np.mean(cv_rmse)), "std": float(np.std(cv_rmse))},
            "r2": {"mean": float(np.mean(cv_r2)), "std": float(np.std(cv_r2))},
        }

    return results


def _compute_classification_metrics(
    labels_test: np.ndarray,
    test_predictions: np.ndarray,
    cv_results: Optional[list] = None,
    detailed_metrics: bool = False,
) -> Dict[str, Union[float, Dict[str, Union[float, List[float]]]]]:
    """Compute classification metrics with optional cross-validation results."""
    n_outputs = labels_test.shape[1]
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    roc_auc_data = {}

    for i in range(n_outputs):
        # Test metrics
        test_true = labels_test[:, i]
        test_pred = test_predictions[:, i]

        metric_funcs = {
            "accuracy": accuracy_score,
            "precision": lambda x, y: precision_score(
                x, y, average="macro", zero_division=0
            ),
            "recall": lambda x, y: recall_score(x, y, average="macro", zero_division=0),
            "f1": lambda x, y: f1_score(x, y, average="macro", zero_division=0),
        }

        for metric_name, func in metric_funcs.items():
            test_metrics[metric_name].append(func(test_true, test_pred))

        # ROC/AUC if requested
        if detailed_metrics:
            classes = np.unique(test_true)
            y_true_bin = label_binarize(test_true, classes=classes)
            y_pred_bin = label_binarize(test_pred, classes=classes)
            roc_auc_data[f"output_{i}"] = {}
            for j, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, j], y_pred_bin[:, j])
                auc = roc_auc_score(y_true_bin[:, j], y_pred_bin[:, j])
                roc_auc_data[f"output_{i}"][f"class_{cls}"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc),
                }

    results = {k: [float(x) for x in v] for k, v in test_metrics.items()}
    if detailed_metrics:
        results["roc_auc"] = roc_auc_data

    # Include CV stats if requested
    if cv_results is not None:
        cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for metric_name, func in metric_funcs.items():
            cv_scores = [
                func(r["true"][:, i], r["pred"][:, i])
                for r in cv_results
                for i in range(n_outputs)
            ]
            cv_metrics[metric_name] = {
                "mean": float(np.mean(cv_scores)),
                "std": float(np.std(cv_scores)),
            }
        results["cv_metrics"] = cv_metrics

    return results


def feature_similarity(
    data: np.ndarray,
    labels: np.ndarray,
    category_map: Dict[Union[Any, Tuple[int]], int],
    max_bin: List[int] = None,
    metric: Literal["cosine", "euclidean"] = "cosine",
    similarity: Literal["pairwise", "inside", "between"] = "inside",
    out_det_method: Literal["density", "contamination"] = "density",
    remove_outliers: bool = True,
    parallel: bool = True,
    additional_title: str = "",
    plot: bool = False,
    plot_save_dir: Optional[Path] = None,
    as_pdf: bool = False,
    regenerate: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show_frames: int = None,
    figsize=(6, 5),
    xticks=None,
    xticks_pos=None,
    xlabel=None,
    ylabel=None,
):
    """
    Computes the feature similarity for a Distribution.

    The feature similarity is a measure of the similarity of population vectors to each other, inside groups or between groups.

    Parameters:
    ----------
    data: np.ndarray
        The data to be compared. The data should be in the shape of (n_samples, n_features).
    metrics: str
        euclidean, wasserstein, kolmogorov-smirnov, chi2, kullback-leibler, jensen-shannon, energy, mahalanobis, cosine
        the compare_distributions is also removing outliers on default

    labels: np.ndarray
        The labels are used to group the data. The labels should fit the categories in the category_map.

    category_map: Dict[Union[Any, Tuple[int]], int]
        A dictionary mapping the labels to their corresponding bin numbers.
    """
    # Define file path for saving/loading results
    similarities = None
    if save_path is not None:
        save_path = Path(save_path)
        # Create a filename based on parameters
        param_str = (
            f"{similarity}_{metric}_remove_outliers-{remove_outliers}-{out_det_method}"
        )

        # Try to load existing results if not regenerating
        if not regenerate:
            loaded_data = npio(save_path, task="load", file_type="npz")
            if loaded_data is not None:
                global_logger.info(f"Loaded feature similarity data from {save_path}")
                similarities = loaded_data.get(param_str)

    if similarity in ["inside", "between"]:
        group_vectors, bins = group_by_binned_data(
            data=data,
            category_map=category_map,
            binned_data=labels,
            group_values="raw",
            max_bin=max_bin,
            as_array=False,
        )
        max_bin = np.max(bins, axis=0) if max_bin is None else max_bin
        max_bin = max_bin.astype(int)

        # This is a heuristic to determine the distance between two distributions, needed for density based outlier detection
        # based on the amount of space every bin has on the surface of a sphere
        neighbor_distance = np.sqrt(4 / len(group_vectors))

        if out_det_method == "density" and data.shape[1] > 10:
            global_logger.warning(
                "WARNING: Density based outlier detection is not recommended for high dimensional data. Euclidean distance is used for samples distance calculation."
            )

        if plot:
            if data.shape[1] < 4 and plot:
                densities = {}
                for group_name, locations in group_vectors.items():
                    group_densities = compute_density(locations, neighbor_distance)
                    densities[group_name] = {
                        "locations": locations,
                        "values": group_densities,
                    }

                Vizualizer.density(
                    data=densities,
                    space="2d",
                    additional_title=additional_title,
                    plot_legend=False,
                    use_alpha=False,
                    filter_outlier=True,
                    save_dir=plot_save_dir,
                    regenerate=False,
                )

        if similarities is None:
            similarities = compare_distribution_groups(
                group_vectors=group_vectors,
                max_bin=max_bin,
                metric=metric,
                compare_type=similarity,
                neighbor_distance=neighbor_distance,
                filter_outliers=remove_outliers,
                parallel=parallel,
                out_det_method=out_det_method,
            )

        if similarity == "inside":
            # Calculate similarity between vectors
            # if similarities is None:
            #     similarities = pairwise_compare(data,
            #                                     metric=metric,)

            # # Calculate similarity inside binned features
            # binned_similarities, _ = group_by_binned_data(
            #     data=similarities,
            #     category_map=category_map,
            #     binned_data=labels,
            #     group_values="mean_symmetric_matrix",
            #     max_bin=max_bin,
            #     as_array=True,
            # )
            title = f"{metric.capitalize()} Similarity Inside Binned Features"
            unique_bins = np.array(list(category_map.keys()))
            xticks = np.unique(unique_bins[:, 0])
            yticks = np.unique(unique_bins[:, 1])
            xticks_pos = xticks
            yticks_pos = yticks
            to_show_similarities = similarities
        elif similarity == "between":
            bins = np.array(bins)
            plot_bins = xticks or bins
            if not xticks:
                ticks = []
                tick_steps = []
                for dim in range(bins.ndim):
                    ticks.append(np.unique(bins[:, dim]))
                    tick_steps.append(int(len(ticks[-1]) / 3))
                xticks = ticks[0]
                yticks = ticks[1] if len(ticks) > 1 else None
            else:
                xticks = bins
            additional_title += f" from and to each Bin"
    elif similarity == "pairwise":
        # No binned features, calculate similarity directly
        if similarities is None:
            similarities = pairwise_compare(data, metric=metric)
        else:
            if not isinstance(similarities, np.ndarray):
                global_logger.critical(
                    "Similarities should be numpy array. For inside bin similarity plotting."
                )
                raise ValueError(
                    "Similarities should be numpy array. For inside bin similarity plotting."
                )
            if similarities.ndim != 2:
                global_logger.critical(
                    "Similarities should be 2D array. For plotting similarities in 2D"
                )
                raise ValueError(
                    "Similarities should be 2D array. For plotting similarities in 2D"
                )
        if show_frames is None:
            show_frames = similarities.shape[0]
        similarity = True
        to_show_similarities = (
            similarities[:show_frames, :show_frames]
            if isinstance(show_frames, int)
            else similarities
        )
        title = title or f"Neural {metric} Similarity"
        yticks = None
        yticks_pos = None
        xlabel = xlabel or "Frames"
        ylabel = ylabel or "Frames"
    else:
        do_critical(
            ValueError,
            "Invalid similarity type. Use 'inside', 'between' or 'pairwise'.",
        )

    # Save results after computation (if save_path is provided)
    if save_path is not None and similarities is not None:
        save_data = {param_str: similarities}
        npio(save_path, task="update", data=save_data, file_type="npz")

    if plot:
        if similarity in ["inside", "pairwise"]:
            Vizualizer.plot_mean_std_heatmap(
                to_show_similarities,
                additional_title=additional_title,
                figsize=figsize,
                title=title,
                xticks=xticks,
                yticks=yticks,
                xtick_pos=xticks_pos,
                ytick_pos=yticks_pos,
                colorbar_label=metric,
                xlabel="Position Bin X",
                ylabel="Position Bin Y",
                save_dir=plot_save_dir,
                as_pdf=as_pdf,
                regenerate=regenerate,
            )
        elif similarity == "between":
            Vizualizer.plot_group_distr_similarities(
                {metric: similarities},
                additional_title=additional_title,
                bins=plot_bins,
                colorbar=True,
                xticks=xticks,
                yticks=yticks,
                tick_steps=tick_steps,
                colorbar_label=metric,
                save_dir=plot_save_dir,
                as_pdf=as_pdf,
                regenerate=regenerate,
            )
    return similarities


def structure_index(
    data: np.ndarray,
    labels: np.ndarray,
    params: Dict[str, Union[int, bool, List[int]]],
    additional_title: str = "",
    plot: bool = False,
    plot_save_dir: Optional[Path] = None,
    as_pdf: bool = False,
    regenerate: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    Calculate structural indices for the task given a model.


    Raw or Embedded data as well as labels are extracted from the models that fit the naming filter.
    If n_neighbors is a list, then a parameter sweep is performed.

    This Method is based on a graph-based topological metric able to quantify the amount of structure
    present at the distribution of a given feature over a point cloud in an arbitrary D-dimensional space.
    See the publication(https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011768)
    for specific details and follow this notebook (https://colab.research.google.com/github/PridaLab/structure_index/blob/main/demos/structure_index_demo.ipynb)
    for a step by step demo.
    https://github.com/PridaLab/structure_index

    Parameters:
    -----------
    First Parameters:
        Are explained in the extract_wanted_embedding_and_labels function.
    as_pdf: bool, optional
        Whether to save the plot as a PDF (default is False).
    params: dict
        n_bins: integer (default: 10)
            number of bin-groups the label will be divided into (they will
            become nodes on the graph). For vectorial features, if one wants
            different number of bins for each entry then specify n_bins as a
            list (i.e. [10,20,5]). Note that it will be ignored if
            'discrete_label' is set to True.

        n_neighbors: integer (default: 15) or list of integers
            Number of neighbors used to compute the overlapping between
            bin-groups. This parameter controls the tradeoff between local and
            global structure.

        discrete_label: boolean (default: False)
            If the label is discrete, then one bin-group will be created for
            each discrete value it takes. Note that if set to True, 'n_bins'
            parameter will be ignored.

        num_shuffles: int (default: 100)
            Number of shuffles to be computed. Note it must fall within the
            interval [0, np.inf).

        verbose: boolean (default: False)
            Boolean controling whether or not to print internal process.

    Returns:
    --------
    structure_indices: dict with n entries where n is the number neighbors
        SI: float
            structure index

        bin_label: tuple
            Tuple containing:
                [0] Array indicating the bin-group to which each data point has
                    been assigned.
                [1] Array indicating feature limits of each bin-group. Size is
                [number_bin_groups, n_features, 3] where the last dimension
                contains [bin_st, bin_center, bin_en]

        overlap_mat: numpy 2d array of shape [n_bins, n_bins]
            Array containing the overlapping between each pair of bin-groups.

        shuf_SI: numpy 1d array of shape [num_shuffles,]
            Array containing the structure index computed for each shuffling
            iteration.
    """

    params["n_neighbors"] = make_list_ifnot(params["n_neighbors"])
    if "n_bins" not in params:
        params["n_bins"] = 10  # Default value if not provided
    sweep_range = deepcopy(params["n_neighbors"])

    # TODO: REMOVE THIS LATER ONCE ALL FILES HAVE BEEN RENAMED
    if "0.00-" in save_path.stem:
        if not save_path.exists():
            # convert old style naming to new style
            old_save_path = save_path.parent / save_path.name.replace("0.00-", "0.0-")
            if old_save_path.exists():
                # rename file
                old_save_path.rename(save_path)
    ######################################

    if save_path is None:
        global_logger.warning("Save path not provided not saving structure index.")
    else:
        # Try to load existing npz file
        existing_data = npio(save_path, task="load", file_type="npz") or {}

    struct_inds = {}
    new_data = {}

    for n_neighbors in tqdm(
        sweep_range,
        position=tqdm._get_free_pos(),
        leave=True,
        desc="Computing Structure Index",
    ):
        params["n_neighbors"] = n_neighbors

        npz_key = ""
        for key, value in params.items():
            if key == "verbose":
                continue
            npz_key += f"{key}-{value}|"

        # Check if we already have data for this configuration and don't need to regenerate
        if npz_key in existing_data and not regenerate:
            struct_inds[n_neighbors] = existing_data[npz_key]
        else:
            # Compute new data
            struct_ind_values = compute_structure_index(data, labels, **params)

            struct_ind = {
                "SI": struct_ind_values[0],
                "bin_label": struct_ind_values[1],
                "overlap_mat": struct_ind_values[2],
                "shuf_SI": struct_ind_values[3],
            }

            struct_inds[n_neighbors] = struct_ind
            new_data[npz_key] = struct_ind

    # Save all new data to the npz file
    if new_data and save_path:
        # Use update task to merge with existing data
        npio(save_path, task="update", data=new_data, file_type="npz")

    # Return single result if only one parameter was used
    if len(sweep_range) == 1:
        struct_inds = next(iter(struct_inds.values()))

    if plot:
        if len(sweep_range) == 1:
            Vizualizer.plot_structure_index(
                embedding=data,
                feature=labels,
                overlapMat=struct_inds["overlap_mat"],
                SI=struct_inds["SI"],
                binLabel=struct_inds["bin_label"],
                additional_title=additional_title,
                as_pdf=as_pdf,
                save_dir=plot_save_dir,
            )
        else:  # len(sweep_range) > 1
            values = [struct_inds[n]["SI"] for n in sweep_range]
            Vizualizer.plot_structure_index(
                values=values,
                sweep_range=sweep_range,
                additional_title=additional_title,
                as_pdf=as_pdf,
                save_dir=plot_save_dir,
            )
    return struct_inds


def load_df_sim(
    labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
    regenerate: bool = False,
) -> List[Tuple[str, str]]:
    """Extract missing item pairs from the similarity DataFrame."""
    # Generate item pairs for loading
    item_pairs = [
        (l1, l2) for i, l1 in enumerate(labels) for j, l2 in enumerate(labels) if i < j
    ]

    # Load existing data if available and not regenerating
    missing_pairs = item_pairs
    if save_path and not regenerate:
        df_sim, saved_labels = h5io(
            save_path, task="load", labels=labels, item_pairs=item_pairs
        )
        # Check for missing pairs
        computed_pairs = (
            set() if df_sim is None else set(zip(df_sim["item_i"], df_sim["item_j"]))
        )
        missing_pairs = [pair for pair in item_pairs if pair not in computed_pairs]
        if len(missing_pairs) > 0:
            global_logger.info(
                f"Computing similarities for {len(missing_pairs)} missing pairs."
            )
        else:
            global_logger.info(
                f"All {len(item_pairs)} pairs already computed. Loading existing data."
            )
            missing_pairs = None

    return missing_pairs, df_sim


def pairwise_shape_compare_hdf5io_wrapper(
    data: Dict[str, np.ndarray],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "one-to-one",
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    regenerate: bool = False,
    group_data: Optional[Dict[str, str]] = None,
    progress_desc: str = "Calculating Similarity",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Wrapper function to manage loading, computing, and saving pairwise comparisson comparisson similarity data in HDF5 format.

    Parameters:
    - data (Dict[str, np.ndarray]): Dictionary of datasets to compare (key: label, value: array).
    - compute_func (Callable[[np.ndarray, np.ndarray], float]): Function to compute similarity between two arrays.
    - labels (Optional[List[str]]): Labels for the datasets (defaults to data keys).
    - save_path (Optional[Union[str, Path]]): Path to save/load similarity data in HDF5 format.
    - regenerate (bool): If True, recompute all similarities even if data exists.
    - group_data (Optional[Dict[str, str]]): Dictionary mapping item IDs to group labels.
    - progress_desc (str): Description for progress bar.

    Returns:
    - pd.DataFrame: Long-format DataFrame with columns 'item_i', 'item_j', 'disparity', and optional 'group_i'.
    - List[str]: Labels used for the similarity matrix.
    """
    save_path = Path(save_path).with_suffix(".h5") if save_path else None
    labels = list(data.keys()) if labels is None else labels
    missing_pairs, df_sim = load_df_sim(save_path=save_path, labels=labels)

    # Compute similarities for missing or all labels
    if missing_pairs:
        sim_rows = []

        for name_i, name_j in tqdm(missing_pairs, desc=progress_desc):
            mati = data[name_i]
            matj = data[name_j]
            disparity = shape_distance(mati, matj, method=method)
            # try:
            #     disparity = compute_func(mati, matj)
            # except Exception as e:
            #     global_logger.warning(
            #         f"Error computing similarity for {name_i} vs {name_j}: {e}"
            #     )
            #     disparity = None
            sim_rows.append(
                {
                    "item_i": name_i,
                    "item_j": name_j,
                    "disparity": disparity,
                    "group_i": group_data.get(name_i) if group_data else None,
                    "group_j": group_data.get(name_j) if group_data else None,
                    "num_cells_i": mati.shape[0],
                    "num_cells_j": matj.shape[0],
                }
            )
            sim_rows.append(
                {
                    "item_i": name_j,
                    "item_j": name_i,
                    "disparity": disparity,
                    "group_i": group_data.get(name_j) if group_data else None,
                    "group_j": group_data.get(name_i) if group_data else None,
                    "num_cells_i": matj.shape[0],
                    "num_cells_j": mati.shape[0],
                }
            )

            # save every 5000 pairs to avoid memory issues and allow progress saving
            if len(sim_rows) >= 2000:
                # TODO: group_data handling
                group_data_part = group_data
                df_sim = save_update_hdf5(
                    save_path,
                    df_sim=df_sim,
                    new_df=sim_rows,
                    labels=labels,
                    group_data=group_data_part,
                )
                sim_rows = []

        df_sim = save_update_hdf5(
            save_path,
            df_sim=df_sim,
            new_df=sim_rows,
            labels=labels,
            group_data=group_data,
        )

    if df_sim is None or df_sim.empty:
        raise ValueError("No similarity data computed or loaded.")

    return df_sim, labels


def save_update_hdf5(
    save_path: Union[str, Path],
    df_sim: pd.DataFrame,
    new_df: Optional[pd.DataFrame] = None,
    labels: Optional[List[str]] = None,
    group_data: Optional[Dict[str, str]] = None,
) -> None:
    """Save or update a DataFrame in an HDF5 file."""
    # Create or append to DataFrame
    new_df = pd.DataFrame(new_df)
    if df_sim is not None:
        df_sim = pd.concat([df_sim, new_df], ignore_index=True)
    else:
        df_sim = new_df

    # Remove duplicates
    if df_sim is not None:
        df_sim = df_sim.drop_duplicates(subset=["item_i", "item_j"], keep="first")

    # Add group data if provided
    if df_sim is not None and group_data:
        raise NotImplementedError(
            "Group data handling is not implemented in this function yet."
        )
        # Convert group data to DataFrame and merge
        # Assuming group_data is a dict mapping item_i to group_i
        # This part can be customized based on the actual structure of group_data
        df_groups = (
            pd.DataFrame.from_dict(group_data, orient="index", columns=["group_i"])
            .reset_index()
            .rename(columns={"index": "item_i"})
        )
        df_sim = df_sim.merge(df_groups, on="item_i", how="left")

    # Save to HDF5
    if save_path and df_sim is not None:
        h5io(save_path, task="save", data=df_sim, labels=labels)

    return df_sim


def shape_distance(
    mtx1: np.ndarray,
    mtx2: np.ndarray,
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "soft-matching",
    max_num_cells: int = 100,
    repeats: int = 10,
    seed: Optional[int] = None,
) -> Tuple[float, dict]:
    """Compute a shape distance between two neural population activity matrices.

    Supports Procrustes, one-to-one matching, and soft-matching distances.
    Matrices may have different numbers of rows (neurons) but must have the same
    number of columns (tuning bins). Whitening and normalization are performed
    before subsampling to optimize computation. Random subsampling ensures equal
    neuron counts when necessary.

    Args:
        mtx1: Matrix of shape (cells1, tuning), neurons x tuning bins.
        mtx2: Matrix of shape (cells2, tuning), neurons x tuning bins.
        method: Distance metric ('procrustes', 'one-to-one', 'soft-matching').
        max_num_cells: Maximum number of neurons to select (default: 100) if method needs to have equal number of neurons.
        repeats: Number of random subsampling iterations (default: 10).
        seed: Random seed for reproducibility (default: None).

    Returns:
        Tuple of:
        - float: Mean distance over repeats.
        - dict: Additional info (disparities, failed_attempts).

    Raises:
        ValueError: If tuning bins differ, matrices are empty, method is unsupported,
                    or all repeats fail.
    """
    # Input validation
    cells1, tuning1 = mtx1.shape
    cells2, tuning2 = mtx2.shape
    if tuning1 != tuning2:
        raise ValueError(f"Tuning bins must match: {tuning1} != {tuning2}")
    if mtx1.size == 0 or mtx2.size == 0:
        raise ValueError("Input matrices must be non-empty")

    valid_methods = {"procrustes", "one-to-one", "soft-matching"}
    method = method.lower()
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got {method}")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Preprocess matrices: whiten and normalize
    mtx1_white = modify_mtx(mtx1, whiten=True, norm=False)
    mtx2_white = modify_mtx(mtx2, whiten=True, norm=False)

    # import procrustes from scipy.spatial
    from scipy.spatial import procrustes

    min_cells = min(cells1, cells2, max_num_cells)
    disparities = []
    pairs = None
    if method in ["procrustes", "one-to-one"]:
        while len(disparities) < repeats:
            # Subsample neurons
            neuron_indices1 = np.random.choice(cells1, min_cells, replace=False)
            neuron_indices2 = np.random.choice(cells2, min_cells, replace=False)
            shaped_mtx1 = mtx1_white[neuron_indices1]
            shaped_mtx2 = mtx2_white[neuron_indices2]

            # Re-normalize submatrices to ensure Frobenius norm = 1
            try:
                if method == "procrustes":
                    # Align and compute disparity
                    shaped_mtx2_norm_align = align_mtx(
                        shaped_mtx1_norm, shaped_mtx2_norm
                    )
                    disparity = np.sum(
                        np.square(shaped_mtx1_norm - shaped_mtx2_norm_align)
                    )

                elif method == "one-to-one":
                    # Compute cost matrix
                    shaped_mtx1_norm = modify_mtx(shaped_mtx1, whiten=False, norm=True)
                    shaped_mtx2_norm = modify_mtx(shaped_mtx2, whiten=False, norm=True)
                    cost_matrix = cdist(
                        shaped_mtx1_norm, shaped_mtx2_norm, metric="sqeuclidean"
                    )
                    # Solve linear assignment problem
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    disparity = cost_matrix[row_ind, col_ind].sum() / min_cells
                    # Note: Normalization by min_cells aligns with whitened matrix norms
            except Exception as e:
                print(f"Method {method} failed: {e}, trying again.")
                continue

            disparities.append(disparity)
        disparity = np.mean(disparities)

    elif method == "soft-matching":
        # For soft-matching, use the full matrices
        mtx1_norm = modify_mtx(mtx1_white, whiten=False, norm=True)
        mtx2_norm = modify_mtx(mtx2_white, whiten=False, norm=True)

        # Compute cost matrix
        cost_matrix = cdist(mtx1_norm, mtx2_norm, metric="sqeuclidean")
        # Define uniform distributions
        a = np.ones(cells1) / cells1
        b = np.ones(cells2) / cells2
        # Compute optimal transport plan
        transport_plan = ot.emd(a, b, cost_matrix)

        # To get "distance" (Wasserstein), take the square root
        disparity = np.sqrt(np.sum(transport_plan * cost_matrix))
        threshold = 0.01  # Adjustable threshold for significant matching probability
        pairs = {
            (i, j): matching_prob
            for i, j, matching_prob in zip(
                *np.where(transport_plan > threshold),
                transport_plan[transport_plan > threshold],
            )
        }

        raise NotImplementedError(
            "Soft-matching pairs output not fully implemented yet."
        )

    return disparity, pairs


def _plot_similarity_results(
    df_matrix: pd.DataFrame,
    labels: List[str],
    method: str,
    plot: Union[bool, List[str]],
    plot_show: Literal["center", "center_std", "samples", "flow", "annotate_dots"],
    plot_df: Optional[pd.DataFrame],
    n_components: int,
    additional_title: str,
    plot_save_dir: Optional[Path],
    as_pdf: bool,
) -> None:
    """
    Internal helper to handle all plotting for shape similarity results.

    Parameters:
    - df_matrix: Symmetric disparity matrix
    - labels: List of item labels
    - method: Similarity method used
    - plot: Which plots to generate
    - ... (other plotting params)
    """
    if not plot:
        return

    if plot is True:
        plot = ["heatmap", "mds_pca"]
    elif isinstance(plot, str):
        plot = [plot]

    # Prepare tick labels
    ticks = [""] * len(labels)
    for i, label in enumerate(labels):
        label_parts = label.split("_")
        animal_id = label_parts[0]
        task = label_parts[2] if len(label_parts) > 2 else ""
        ticks[i] = f"{animal_id}_{task}"

    # Heatmap
    if "heatmap" in plot:
        Vizualizer.plot_heatmap(
            df_matrix,
            title=f"{method.upper()} Shape Similarity",
            additional_title=additional_title,
            xticks=ticks,
            yticks=ticks,
            sort_by="similarity",
            no_diag=False,
            annotation=False,
            colorbar_label="Disparity",
            save_dir=plot_save_dir,
            as_pdf=as_pdf,
        )

    # Embedding plots (UMAP, t-SNE, MDS, etc.)
    embedding_types = [p for p in plot if p != "heatmap"]
    if embedding_types:
        create_multiple_embeddings(
            df_matrix,
            embedding_types,
            plot_df=plot_df,
            n_components=n_components,
            plot_show=plot_show,
            additional_title=additional_title,
            plot_save_dir=plot_save_dir,
            as_pdf=as_pdf,
        )


def calc_shape_similarity(
    data: Dict[str, np.ndarray],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "soft-matching",
    plot_show: Literal[
        "center", "center_std", "samples", "flow", "annotate_dots"
    ] = "samples",
    labels: Optional[List[str]] = None,
    plot_df: Optional[pd.DataFrame] = None,
    n_components: int = 2,
    additional_title: str = "",
    save_path: Optional[Union[str, Path]] = None,
    regenerate: bool = False,
    plot: Union[
        bool,
        List[
            Literal[
                "heatmap",
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
    ] = [
        "umap",
        "heatmap",
        "tsne",
        "mds",
        "isomap",
        "lle",
        "spectral_embedding",
        "pca",
        "mds_pca",
    ],
    plot_save_dir: Optional[Path] = None,
    as_pdf: bool = False,
    group_data: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute Shape similarity between datasets and optionally plot results.

    Parameters:
    - data (Dict[str, np.ndarray]): Dictionary containing the datasets to compare.
    - labels (Optional[List[str]]): Labels for the datasets (defaults to data keys).
    - plot_df (Optional[pd.DataFrame]): DataFrame with 'plot_label' and 'numbers' for advanced plotting.
    - additional_title (str): Additional title for plots.
    - save_path (Optional[Union[str, Path]]): Path to save/load similarity data in HDF5 format.
    - regenerate (bool): If True, recompute all similarities.
    - plot (Union[bool, List[str]]): Whether to plot and which plots to generate.
    - plot_save_dir (Optional[Path]): Directory to save plots.
    - as_pdf (bool): Save plots as PDF if True.
    - group_data (Optional[Dict[str, str]]): Dictionary mapping item IDs to group labels.

    Returns:
    - pd.DataFrame: Symmetric disparity matrix.
    - List[str]: Labels used for the similarity matrix.
    """
    labels = list(data.keys()) if labels is None else labels

    # Compute pairwise shape similarities
    df_sim, labels = pairwise_shape_compare_hdf5io_wrapper(
        data=data,
        method=method,
        labels=labels,
        save_path=save_path,
        regenerate=regenerate,
        group_data=group_data,
        progress_desc=f"Calculating Shape Similarity using {method.capitalize()}",
    )

    # Convert to symmetric matrix
    df_matrix = df_sim.pivot(index="item_i", columns="item_j", values="disparity")
    df_matrix = df_matrix.combine_first(df_matrix.T)

    # === PLOTTING (Extracted) ===
    try:
        _plot_similarity_results(
            df_matrix=df_matrix,
            labels=labels,
            method=method,
            plot=plot,
            plot_show=plot_show,
            plot_df=plot_df,
            n_components=n_components,
            additional_title=additional_title,
            plot_save_dir=plot_save_dir,
            as_pdf=as_pdf,
        )
    except Exception as e:
        global_logger.error(f"Plotting failed: {e}")

    return df_matrix, labels


def simple_array_to_embedd(
    values: np.ndarray,
    labels: List[str],
    n_components: int = 2,
    ax=None,
    additional_title: str = "",
):
    """
    Create a distance matrix from 1D data points and embed them using MDS.

    The functions bottom left corner and top right corner of the plot are set to the min and max values of the data points.
    """
    # create distance matrix of 1d data points
    dist_matrix = np.zeros((len(values), len(values)))
    for i in range(len(values)):
        for j in range(len(values)):
            dist_matrix[i, j] = np.linalg.norm(
                np.array(values[i]) - np.array(values[j])
            )

    # get min and max value ids
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)

    simple_embedd(
        dist_matrix,
        n_components=n_components,
        additional_title=additional_title,
        ax=ax,
        labels=labels,
        method="mds",
        min_sample_value_idx=min_idx,
        max_sample_value_idx=max_idx,
    )


def simple_embedd(
    distances: Union[np.ndarray],
    method: Literal[
        "tsne",
        "mds",
        "isomap",
        "ptu",
        "lle",
        "spectral_embedding",
        "umap",
        "pca",
        "mds_pca",
    ] = "umap",
    plot: bool = True,
    title=None,
    ax=None,
    min_sample_value_idx: int = None,
    max_sample_value_idx: int = None,
    n_components: int = 2,
    plot_show: Literal[
        "center", "center_std", "samples", "flow", "annotate_dots"
    ] = "samples",
    perplexity: int = 3,
    random_state: int = 42,
    n_neighbors: int = 10,
    additional_title: str = "",
    plot_df: Optional[pd.DataFrame] = None,
    labels: Optional[Union[List[str], pd.DataFrame]] = None,
    colors: Optional[np.ndarray] = None,
    dissimilarity: Literal["precomputed", "euclidean"] = "euclidean",
    figsize: tuple = (8, 8),
    alpha: float = 0.8,
    cmap: str = None,
    save_dir: Optional[str] = None,
    add_cmap: np.ndarray = None,
    legend=True,
    return_traces: bool = True,
):
    """Performs dimensionality reduction on a dataset and optionally visualizes the result.

    This function serves as a wrapper around various scikit-learn and UMAP embedding
    methods. It simplifies the process of reducing high-dimensional data to a lower-
    dimensional space (typically 2D) and then uses `plot_simple_embedding` to generate
    an interactive Plotly visualization.

    Parameters
    ----------
    distances : np.ndarray
        The input data. This can be either a feature array of shape
        (`n_samples`, `n_features`) or a precomputed distance matrix of shape
        (`n_samples`, `n_samples`) if `dissimilarity='precomputed'`.
    method : str, default='umap'
        The dimensionality reduction technique to use. Supported methods include:
        'tsne', 'mds', 'isomap', 'lle', 'spectral_embedding', 'umap', 'pca', and 'mds_pca'.
    plot : bool, default=True
        If `True`, generates and displays a plot of the embedded coordinates.
    title : str, optional
        Custom title for the plot. If `None`, a default title based on the method
        is generated.
    ax : go.Figure, optional
        A Plotly Figure object to add traces to. If `None`, a new figure is created.
    min_sample_value_idx : int, optional
        The index of a sample to be mapped to the origin (0,0) of the plot. Used for
        normalizing the coordinate space.
    max_sample_value_idx : int, optional
        The index of a sample to be mapped to the corner (1,1) of the plot. Used for
        normalizing the coordinate space.
    n_components : int, default=2
        The number of dimensions to reduce the data to.
    plot_show : str, default='samples'
        A string specifying the plot type(s) passed to the plotting function.
        Multiple types can be combined with commas (e.g., 'samples,center_flow').
        See the `plot_simple_embedding` function for details.
    perplexity : int, default=3
        The perplexity parameter for the 'tsne' method.
    random_state : int, default=42
        The random seed for reproducibility in methods like 'tsne', 'mds', 'umap', etc.
    n_neighbors : int, default=5
        The number of neighbors for 'umap', 'isomap', 'lle', and 'spectral_embedding' methods.
    additional_title : str, optional
        Supplementary text to append to the plot title.
    plot_df : pd.DataFrame, optional
        A DataFrame with metadata for plotting. This is the recommended way to pass
        labels and colors. See the `plot_simple_embedding` docstring for required columns.
    labels : list or pd.DataFrame, optional
        (Legacy) Labels for plotting. Overridden by `plot_df` if provided.
    colors : np.ndarray, optional
        (Legacy) Colors for plotting. Overridden by `plot_df` if provided.
    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Specifies the type of input data for 'mds'. Use 'precomputed' if `distances`
        is a distance matrix.
    figsize : tuple, default=(8, 8)
        The figure size for the plot.
    alpha : float, default=0.8
        The opacity for the scatter plot points.
    cmap : str, optional
        The colormap to use if colors are not specified.
    save_dir : str, optional
        Directory to save the plot as an HTML file.
    add_cmap : np.ndarray, optional
        Parameter passed to the plotting function.
    legend : bool, default=True
        Whether to display the plot legend.
    return_traces : bool, default=True
        If `True` and `plot=True`, returns the Plotly traces and layout along with the
        coordinates.

    Returns
    -------
    np.ndarray or tuple
        - If `plot=False`, returns only the embedded coordinates (`np.ndarray`).
        - If `plot=True` and `return_traces=False`, returns only the embedded coordinates.
        - If `plot=True` and `return_traces=True`, returns a tuple:
        (`coords`, `traces`, `layout`), where `coords` is the embedded data,
        and `traces` and `layout` are Plotly objects.

    Raises
    ------
    ValueError
        If an unsupported `method` is specified.

    Example
    -------
    >>> from sklearn.datasets import make_blobs
    >>> # Generate some sample data
    >>> features, labels = make_blobs(n_samples=100, centers=4, random_state=42)
    >>>
    >>> # Perform UMAP embedding and get the Plotly objects
    >>> coords, traces, layout = simple_embedding(
    ...     distances=features,
    ...     method='umap',
    ...     labels=[f"Cluster {l}" for l in labels],
    ...     n_neighbors=10
    ... )
    >>> print(f"Embedded coordinates shape: {coords.shape}")
    Embedded coordinates shape: (100, 2)
    """
    if distances.size == 0:
        raise ValueError("Input distances array is empty")

    if labels is None and plot_df is None and isinstance(distances, pd.DataFrame):
        labels = distances.index.tolist()
        labels = [str(label).split("_")[0] for label in labels]
    elif labels is not None and plot_df is not None:
        if "plot_label" in plot_df.columns:
            global_logger.warning(
                "Both labels and plot_df provided. Using plot_df for plotting."
            )
            labels = None

    if colors is not None and plot_df is not None:
        if "color" in plot_df.columns:
            global_logger.warning(
                "Both colors and plot_df provided. Using plot_df for colors."
            )
            colors = None

    # Determine number of samples
    if dissimilarity == "precomputed":
        n_samples = distances.shape[0]
    else:
        n_samples = distances.shape[0]

    # Adjust parameters for small datasets
    adjusted_perplexity = perplexity
    adjusted_n_neighbors = n_neighbors
    if method == "tsne" and perplexity >= n_samples:
        adjusted_perplexity = max(1, n_samples - 2)
        global_logger.warning(
            f"Adjusted perplexity from {perplexity} to {adjusted_perplexity} for {n_samples} samples"
        )
    if (
        method in ["umap", "isomap", "lle", "spectral_embedding"]
        and n_neighbors >= n_samples
    ):
        adjusted_n_neighbors = max(1, n_samples - 1)
        global_logger.warning(
            f"Adjusted n_neighbors from {n_neighbors} to {adjusted_n_neighbors} for {n_samples} samples"
        )

    if method == "tsne":
        emb_cls = TSNE(
            n_components=n_components,
            perplexity=adjusted_perplexity,
            random_state=random_state,
        )
    elif method == "mds":
        emb_cls = MDS(
            # n_init=10,
            n_components=n_components,
            dissimilarity=dissimilarity,
            random_state=random_state,
        )
        mds = emb_cls
    elif method == "isomap":
        emb_cls = Isomap(n_components=n_components, n_neighbors=adjusted_n_neighbors)
    elif method == "ptu":
        # Parallel Transport Unfolding (PTU) is an algorithm for generating a quasi-isometric, low-dimensional mapping
        # from a sparse and irregular sampling of an arbitrary manifold embedded in a high-dimensional space.
        emb_cls = PTU(
            X=distances,
            embedding_dim=n_components,
            n_neighbors=adjusted_n_neighbors,
            geod_n_neighbors=adjusted_n_neighbors,
        )
    elif method == "lle":
        emb_cls = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=adjusted_n_neighbors,
            random_state=random_state,
        )
    elif method == "spectral_embedding":
        emb_cls = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=adjusted_n_neighbors,
            random_state=random_state,
        )
    elif method == "umap":
        emb_cls = UMAP(
            n_components=n_components,
            n_neighbors=adjusted_n_neighbors,
            random_state=random_state,
        )
    elif method == "pca":
        emb_cls = PCA(n_components=n_components, random_state=random_state)
    elif method == "mds_pca":
        # First apply MDS, then PCA to reduce dimensions
        mds = MDS(
            # n_init=10,
            n_components=20,
            dissimilarity=dissimilarity,
            random_state=random_state,
        )

        distances = np.nan_to_num(distances, nan=0)
        distances = mds.fit_transform(distances)
        # get distortion
        stress = mds.stress_
        global_logger.info(f"MDS stress: {stress:.4f}")
        emb_cls = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    distances = np.nan_to_num(distances, nan=0, posinf=1e10, neginf=-1e10)

    # Try the primary embedding method
    if method != "ptu":
        coords = emb_cls.fit_transform(distances)
    else:
        coords = emb_cls.fit()
    # try:
    #     coords = emb_cls.fit_transform(distances)
    # except Exception as e:
    #     global_logger.error(
    #         f"Primary embedding method '{method}' failed: {e}. Falling back to PCA."
    #     )
    #     # Fallback to PCA
    #     fallback_emb = PCA(n_components=n_components, random_state=random_state)
    #     coords = fallback_emb.fit_transform(distances)
    #     emb_cls = fallback_emb  # Update for later attribute access
    #     raise Warning(f"Embedding with method '{method}' failed. Used PCA as fallback.")

    if "mds" in method:
        # stress
        if hasattr(mds, "stress_"):
            distortion = mds.stress_
            global_logger.info(f"MDS stress: {distortion:.4f}")
    if "pca" in method:
        # explained variance
        explained_variance = emb_cls.explained_variance_ratio_.sum()
        global_logger.info(
            f"PCA explained variance: {explained_variance:.4f} for {n_components} components"
        )

    # if min_sample_value_idx and max_sample_value_idx are provided, set the corners of the plot to the min and max values of the data points
    if min_sample_value_idx is not None and max_sample_value_idx is not None:
        min_value = coords[min_sample_value_idx]
        max_value = coords[max_sample_value_idx]
        # Prevent division by zero
        diff = max_value - min_value
        diff[diff == 0] = 1e-8  # or another small number to avoid division by zero

        coords = (coords - min_value) / diff

    if plot:
        method_str = {
            "mds": "MDS",
            "tsne": "t-SNE",
            "isomap": "Isomap",
            "lle": "Locally Linear Embedding",
            "spectral_embedding": "Spectral Embedding",
            "umap": "UMAP",
            "pca": "PCA",
            "mds_pca": "MDS + PCA",
        }.get(method)
        if "mds" in method:
            additional_title += f"\n|MDS stress {distortion:.2f}|"
        if "pca" in method:
            additional_title += f"\n|PCA exp. var. {explained_variance:.2f}|"

        title = f"{method_str} Embedding" if title is None else title
        out = plot_simple_embedd(
            coords=coords,
            title=title,
            additional_title=additional_title,
            ax=ax,
            plot=plot_show,
            plot_df=plot_df,
            labels=labels,
            colors=colors,
            figsize=figsize,
            alpha=alpha,
            cmap=cmap,
            add_cmap=add_cmap,
            save_dir=save_dir,
            legend=legend,
        )
    if return_traces:
        # If using Plotly, return the coordinates directly
        traces = out[0]
        layout = out[1]
        return coords, traces, layout
    else:
        return coords
