import sys, os
from pathlib import Path
import copy
import h5py

# type hints
from typing import List, Union, Dict, Any, Tuple, Optional, Literal
from numpy.typing import ArrayLike
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

# calculations
import numpy as np
import pandas as pd
from numba import njit, prange
import sklearn
from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    rand_score,
    adjusted_rand_score,
)
from scipy.stats import (
    wasserstein_distance,
    ks_2samp,
    entropy,
    energy_distance,
    gaussian_kde,
    mannwhitneyu,
)
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes
from sklearn.covariance import EllipticEnvelope
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

# import cupy as cp  # numpy using CUDA for faster computation
import yaml
import re

from datetime import datetime

# debugging
import logging
from time import time
from pyinstrument import Profiler
import gc

from pathlib import Path
import tempfile
import shutil

markers_meaning = {
    "": "not significant",  # no significance
    "O": "p <= 0.05",  # "*",
    "P": "p <= 0.01",  # "**",
    "*": "p <= 0.001",  # "***",
}
markers_meaning_reversed = {v: k for k, v in markers_meaning.items()}


class GlobalLogger:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, save_dir=""):
        # Only initialize once
        if not GlobalLogger._initialized:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.configure_logger(save_dir=Path(save_dir))
            GlobalLogger._initialized = True

    def configure_logger(self, save_dir=""):
        # Clear existing handlers to avoid duplicates
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        self.logger.setLevel(logging.DEBUG)  # Set the desired level here

        # Create a file handler which logs even debug messages.
        self.log_file_path = save_dir.joinpath("logs", "Global_log.log")
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.log_file_path)
        fh.setLevel(logging.DEBUG)

        # Create a console handler with a higher log level.
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create a formatter and add it to the handlers.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger.
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def set_save_dir(self, save_dir):
        # Remove all existing handlers
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        # Update the file path
        self.log_file_path = Path(save_dir).joinpath("logs", "Global_log.log")
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a new file handler
        fh = logging.FileHandler(self.log_file_path)
        fh.setLevel(logging.DEBUG)

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


# Create the singleton instance
global_logger_object = GlobalLogger()
global_logger = global_logger_object.logger


def do_critical(type: Exception, message: str):
    global_logger.critical(message)
    raise type(message)


def get_significance_marker(
    p_value: float,
    markers: Dict[str, str] = None,
) -> str:
    """
    Get significance marker based on p-value.

    Parameters
    ----------
    """
    if markers is None:
        markers = markers_meaning_reversed

    if not isinstance(p_value, (float, int)):
        raise ValueError(f"Invalid p-value: {p_value}. Must be a float or int.")
    if p_value > 0.05:
        marker = markers["not significant"]
    elif p_value <= 0.05:
        marker = markers["p <= 0.05"]
    elif p_value <= 0.01:
        marker = markers["p <= 0.01"]
    elif p_value <= 0.001:
        marker = markers["p <= 0.001"]
    elif p_value < 0:
        raise do_critical(
            ValueError,
            f"p-value {p_value} is not valid. Must be >= 0.",
        )
    return marker


def npz_loader(file_path, fname=None):
    data = None
    if os.path.exists(file_path):
        with np.load(file_path) as data:
            if fname:
                if fname in data.files:
                    data = data[fname]
                else:
                    global_logger.warning(
                        f"File {fname} not found in {file_path}, returning all data"
                    )
    else:
        global_logger.warning(f"File {file_path} not found, returning None")
    return data


def yield_animal_session_task(animals_dict):
    for animal_id, animal in animals_dict.items():
        for session_date, session in animal.sessions.items():
            for task_id, task in session.tasks.items():
                yield animal, session_date, task


def init_path_checks(path: Union[str, Path], check: str = None) -> Path:
    """
    Initialize and validate a path based on specified criteria.

    Parameters
    ----------
    path : Union[str, Path]
        The path to initialize and check. Can be a string or Path object.
    check : str, optional
        Type of check to perform on the path. Options are:
        - "dir": Validate that the path exists and is a directory
        - "file": Validate that the path exists and is a file
        - None: Only validate that the path exists

    Returns
    -------
    Path
        A validated Path object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or if check is "file" and path is not a file.
    NotADirectoryError
        If check is "dir" and path is not a directory.

    Examples
    --------
    >>> init_path_checks(".", check="dir")  # Current directory
    PosixPath('.')
    >>> init_path_checks("/non/existent/path")
    FileNotFoundError: Path /non/existent/path does not exist.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

    if check == "dir" and not path.is_dir():
        raise NotADirectoryError(f"Path {path} is not a directory.")

    if check == "file" and not path.is_file():
        raise FileNotFoundError(f"Path {path} is not a file.")

    return path


def regex_search(
    items: Union[str, Path, List[Path]],
    include_regex: Union[str, List[str]] = ".*",
    exclude_regex: Union[str, List[str]] = None,
) -> Union[bool, List[Path]]:
    """
    Filter Path object(s) based on regex patterns.

    Parameters
    ----------
    items : Union[Path, List[Path]]
        Either a single Path object or a list of Path objects to filter.
        If a single Path is provided, the function returns a boolean indicating if it matches.
    include_regex : Union[str, List[str]], optional
        Regular expression pattern(s) for items to include.
        Can be a single string pattern or a list of patterns.
        Default is ".*" (include everything).
    exclude_regex : Union[str, List[str]], optional
        Regular expression pattern(s) for items to exclude.
        Can be a single string pattern or a list of patterns.
        Default is None (exclude nothing).

    Returns
    -------
    Union[bool, List[Path]]
        If a single Path was provided:
            - Returns True if the Path matches the criteria
            - Returns False otherwise
        If a list of Paths was provided:
            - Returns a filtered list of Path objects matching the specified regex criteria

    Examples
    --------
    >>> items = list(Path(".").glob("*"))
    >>> python_files = regex_search(items, include_regex=r".*\.py$")

    >>> single_file = Path("example.py")
    >>> is_python = regex_search(single_file, include_regex=r".*\.py$")
    """
    single_item = False
    if not isinstance(items, list):
        single_item = True
        items = [Path(items)]
    include_regex = make_list_ifnot(include_regex)
    exclude_regex = make_list_ifnot(exclude_regex) or []  # Handle None case properly

    # Create a list of compiled regex patterns
    include_patterns = [
        re.compile(pattern) for pattern in include_regex if pattern is not None
    ]
    exclude_patterns = [
        re.compile(pattern) for pattern in exclude_regex if pattern is not None
    ]

    # Filter items based on include and exclude patterns
    filtered_items = []
    for item in items:
        item_name = item.name

        # Empty include_patterns should include everything
        include_match = not include_patterns or any(
            pattern.search(item_name) for pattern in include_patterns
        )
        # Empty exclude_patterns should exclude nothing
        exclude_match = any(pattern.search(item_name) for pattern in exclude_patterns)

        if include_match and not exclude_match:
            filtered_items.append(item)

    if single_item:
        match = True if len(filtered_items) > 0 else False
    else:
        match = filtered_items
    return match


def search_filedir(
    path: Union[str, Path],
    include_regex: Union[str, List[str]] = ".*",
    exclude_regex: Union[str, List[str]] = None,
    type: str = None,
) -> List[Path]:
    """
    Search for files or directories in a given path based on regex patterns.

    This function searches through the contents of a directory and filters items based on
    regular expression patterns for inclusion and exclusion. Additionally, filtering by
    item type (file or directory) is supported.

    Parameters
    ----------
    path : Union[str, Path]
        The directory path to search in. Can be a string or Path object.
    include_regex : Union[str, List[str]], optional
        Regular expression pattern(s) for files/directories to include.
        Can be a single string pattern or a list of patterns.
        Default is ".*" (include everything).
    exclude_regex : Union[str, List[str]], optional
        Regular expression pattern(s) for files/directories to exclude.
        Can be a single string pattern or a list of patterns.
        Default is None (exclude nothing).
    type : str, optional
        The type of items to include in the search results.
        Must be one of: None (both files and directories), "file" (files only),
        or "dir" (directories only). Default is None.

    Returns
    -------
    List[Path]
        A list of Path objects matching the specified criteria.

    Raises
    ------
    ValueError
        If the 'type' parameter is not one of: None, "file", or "dir".
    NotADirectoryError
        If the specified path is not a directory.
    FileNotFoundError
        If the specified path does not exist.

    Examples
    --------
    >>> # Find all Python files in current directory
    >>> python_files = search_filedir(".", include_regex=r".*\.py$", type="file")
    >>> # Find all directories except those starting with "."
    >>> visible_dirs = search_filedir(".", exclude_regex=r"^\.", type="dir")
    >>> # Find all text and markdown files
    >>> text_files = search_filedir(".", include_regex=[r".*\.txt$", r".*\.md$"], type="file")
    """
    path = init_path_checks(path, check="dir")

    # Get the list of files or directories in the path based on type
    if type is None:
        items = list(path.glob("*"))
    elif type == "file":
        items = [item for item in path.glob("*") if item.is_file()]
    elif type == "dir":
        items = [item for item in path.glob("*") if item.is_dir()]
    else:
        raise ValueError("type must be either None, 'file', or 'dir'.")

    # Apply regex filtering to the items
    filtered_items = regex_search(
        items, include_regex=include_regex, exclude_regex=exclude_regex
    )

    return filtered_items


def make_list_ifnot(var):
    """
    This function converts a variable to a list if it is not already a list or numpy array.
    """
    output = None
    if isinstance(var, np.ndarray):
        output = var.tolist()
    elif isinstance(var, list):
        output = var
    elif var is not None:
        output = [var]
    return output


def save_file_present(fpath, show_print=False):
    if fpath is None:
        global_logger.warning("File path is None, returning False")
        return False
    fpath = Path(fpath)
    file_present = False
    if fpath.exists():
        if show_print:
            global_logger.warning(f"File already present {fpath}")
        file_present = True
    else:
        if show_print:
            global_logger.info(f"Saving {fpath.name} to {fpath}")
    return file_present


# Math
def mean_diff(x, y, axis=0):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def calc_cumsum_distances(positions, length, distance_threshold=0.30):
    """
    Calculates the cumulative sum of distances between positions along a track.

    Args:
    - positions (array-like): List or array of positions along the track.
    - length (numeric): Length of the track.
    - distance_threshold (numeric, optional): Threshold for considering a frame's distance change.
                                            Defaults to 30.

    Returns:
    - cumsum_distance (numpy.ndarray): Cumulative sum of distances between positions.
    """
    cumsum_distances = []
    cumsum_distance = 0
    old_position = positions[0]
    for position in positions[1:]:
        frame_distance = position - old_position
        if abs(frame_distance) < distance_threshold:
            cumsum_distance += frame_distance
        else:
            # check if mouse moves from end to start of the track
            if frame_distance < 0:
                cumsum_distance += frame_distance + length
            # check if mouse moves from start to end of the track
            else:
                cumsum_distance += frame_distance - length
        cumsum_distances.append(cumsum_distance)
        old_position = position
    return np.array(cumsum_distances)


def moving_average(data, window_size=30):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, "valid")


def smooth_array(data, window_size=5, axis=None, method="gaussian"):
    """
    Smooth a NumPy array using either a Gaussian filter or a moving average.

    Args:
        data (np.ndarray): Input array (1D or ND).
        window_size (int): Size of the smoothing window or standard deviation.
        axis (int or tuple of ints, optional): Axis or axes along which to smooth.
            If None, smooth across all axes.
        method (str): 'gaussian' or 'moving_average'. Default is 'gaussian'.

    Returns:
        np.ndarray: Smoothed array.
    """
    data = np.asarray(data)

    if method == "gaussian":
        if axis is None:
            # Smooth across all axes
            sigma = window_size / 2
            return gaussian_filter(data, sigma=sigma)
        else:
            return gaussian_filter1d(data, sigma=window_size / 2, axis=axis)

    elif method == "moving_average":
        if axis is None:
            # Apply moving average across all axes
            for ax in range(data.ndim):
                data = _moving_average_1d(data, window_size, axis=ax)
            return data
        else:
            return _moving_average_1d(data, window_size, axis=axis)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'gaussian' or 'moving_average'."
        )


def _moving_average_1d(data, window_size, axis):
    """Apply moving average with a flat window along one axis."""
    weights = np.ones(window_size) / window_size
    return np.apply_along_axis(
        lambda x: np.convolve(x, weights, mode="same"), axis, data
    )


@njit(nopython=True)
def cosine_similarity(v1, v2):
    """
    A cosine similarity can be seen as the correlation between two vectors or point distributions.
    Returns:
        float: The cosine similarity between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / norm_product


def compute_mutual_information(
    true_labels: List[int], predicted_labels: List[int], metric="adjusted"
):
    """
    Mutual Information is a measure of the similarity between two labels of the same data.
    This metric is independent of the absolute values of the labels: a permutation of the class or
    cluster label values won’t change the score value in any way.
    Also note that the metric is not symmetric: switching true and predicted labels will return the same score value.

    parameters:
        true_labels: List[int] - true labels of the data
        predicted_labels: List[int] - predicted labels of the data
        metric: str - metric to use for the computation. Options: "mutual", "normalized", "adjusted"
            - "mutual": Mutual Information: Mutual Information (MI) is a measure of the similarity between two labels of the same data.
            - "normalized": Normalized Mutual Information: Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI)
            score to scale the results between 0 (no mutual information) and 1 (perfect correlation)
            - "adjusted": Adjusted Mutual Information: Adjusted Mutual Information is an adjustment of the Normalized Mutual Information (NMI)
            score to account for chance. -1 <= AMI <= 1.0 (1.0 is the perfect match, 0 is the random match, and -1 is the worst match)
            - "v": V-measure: The V-measure is the harmonic mean between homogeneity and completeness: v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness).
            - "fmi": Fowlkes-Mallows Index: The Fowlkes-Mallows index (FMI) is defined as the geometric mean between precision and recall: FMI = sqrt(precision * recall).
            - "rand": Rand Index: The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
            - "adjrand": Adjusted Rand Index: The Adjusted Rand Index is the corrected-for-chance version of the Rand Index.

    returns:
        mi: float - mutual information score
    """
    if metric == "mutual":
        mi = mutual_info_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "normalized":
        mi = normalized_mutual_info_score(
            true_labels.flatten(), predicted_labels.flatten()
        )
    elif metric == "ami":
        mi = adjusted_mutual_info_score(
            true_labels.flatten(), predicted_labels.flatten()
        )
    elif metric == "v":
        mi = v_measure_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "fmi":
        # particularly useful for evaluating clustering algorithms where the pairwise clustering structure is important. It provides a balanced view by taking into account both precision and recall.
        mi = fowlkes_mallows_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "rand":
        mi = rand_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "adjrand":
        mi = adjusted_rand_score(true_labels.flatten(), predicted_labels.flatten())
    else:
        raise ValueError(
            f"Metric {metric} is not supported. Use 'mutual', 'normalized', or 'adjusted'."
        )
    return mi


def same_distribution(points1, points2):
    """
    Check if two distributions are the same.

    Parameters:
    - points1: array-like, first distribution
    - points2: array-like, second distribution

    Returns:
    - bool: True if the distributions are the same, False otherwise.
    """
    if points1.shape != points2.shape:
        return False
    return np.allclose(points1, points2)


def compare_distribution_groups(
    max_bin,
    group_vectors,
    compare_type: Literal["inside", "between"] = "between",
    metric="cosine",
    neighbor_distance=None,
    filter_outliers=True,
    parallel=True,
    out_det_method="density",
):
    """
    Compare distributions between groups or inside groups based on the specified metric.

    Parameters:
        - group_vectors: dict, dictionary of group vectors
        - compare_type: str, type of comparison ('inside' or 'between')
        - other parameters are explained in compare_distributions

    Returns:
        - similarities: Union[dict, np.ndarray], dictionary of similarities or numpy array of similarities
            - if compare_type is "inside", similarities is a numpy array of shape (num_groups, num_groups)
            - if compare_type is "between", similarities is a dictionary of similarities
    """
    global_logger.info(f"Start comparing distributions {compare_type} using {metric}")
    if compare_type == "inside":
        similarities = {"mean": None, "std": None}
        similaritie_mean = np.zeros(max_bin)
        similaritie_std = np.zeros(max_bin)
        for group_i, (group_name, group1) in enumerate(group_vectors.items()):
            global_logger.info(f"Comparing {group_name} to itself")
            distances = pairwise_compare(
                group1,
                metric=metric,
                neighbor_distance=neighbor_distance,
                out_det_method=out_det_method,
                filter_outliers=filter_outliers,
                parallel=parallel,
            )

            if is_float_like(distances) or distances is None:
                # if pariwise compare is creating a single value
                mean_dist = distances
                std_dist = None
            else:
                # mean all the distances for the group of symmetric matrix without diagonal
                mean_dist = np.mean(distances[np.triu_indices(distances.shape[0], k=1)])
                std_dist = np.std(distances[np.triu_indices(distances.shape[0], k=1)])

            similaritie_std[group_name[0], group_name[1]] = std_dist
            similaritie_mean[group_name[0], group_name[1]] = mean_dist
        similarities["mean"] = similaritie_mean
        similarities["std"] = similaritie_std

    elif compare_type == "between":
        similarities = {}
        # Compare distributions between each group (bin)
        for group_i, (group_name, group1) in enumerate(group_vectors.items()):
            similarities_to_groupi = np.zeros(max_bin)
            for group_j, (group_name2, group2) in enumerate(group_vectors.items()):
                global_logger.info(f"Comparing {group_name} to {group_name2}")

                dist = compare_distributions(
                    points1=group1,
                    points2=group2,
                    metric=metric,
                    neighbor_distance=neighbor_distance,
                    filter_outliers=filter_outliers,
                    parallel=parallel,
                    out_det_method=out_det_method,
                )
                if dist is not None and np.isnan(dist):
                    raise ValueError("check what is happening")
                group_position = (
                    group_j if isinstance(group_name2, str) else group_name2
                )
                similarities_to_groupi[group_position] = dist
            similarities[group_name] = similarities_to_groupi

    return similarities


def compare_distributions(
    points1,
    points2: np.ndarray = None,
    metric="cosine",
    filter_outliers=True,
    parallel=True,
    neighbor_distance=None,
    out_det_method="density",
):
    """
    Compare two distributions using the specified metric.
    CAUTION: Not all metric calculations are optimized with numba

    Parameters:
    - points1: array-like, first distribution
    - points2: array-like, second distribution
    - neighbor_distance: float, distance threshold for outlier detection
    - out_det_method: str, method to use for outlier detection ('density', 'contamination')
    - metric: str, the metric to use for comparison ('wasserstein', 'ks', 'chi2', 'kl', 'js', 'energy', 'mahalanobis')
        - 'wasserstein': Wasserstein Distance (Earth Mover's Distance) energy needed to move one distribution to the other
        - 'kolmogorov-smirnov': Kolmogorov-Smirnov statistic for each dimension and take the maximum (typically used for 1D distributions)
        - 'chi2': Chi-Squared test (requires binned data, here we just compare histograms) - sum of squared differences between observed and expected frequencies
        - 'kl': Kullback-Leibler Divergence - measure of how one probability distribution diverges from a second, expected probability distribution
        - 'js': Jensen-Shannon Divergence - measure of similarity between two probability distributions
        - 'energy': Energy Distance - measure of the distance between two probability distributions (typically used for 1D distributions)
        - 'mahalanobis': Mahalanobis Distance - measure of the distance between two probability distributions
        - "cosine": Cosine Similarity only for mean vector of distribution- measure of the cosine of the angle between two non-zero vectors. This metric is equivalent to the Pearson correlation coefficient for normalized vectors.
        - "overlap": Overlap between two distributions using Kernel Density Estimation (KDE). This method is not working if points are lower than dimensions.

    Wasserstein Distance (smaller = more similar)
    In your case, a Wasserstein distance of 3.099 indicates that it would take an average of 3.099 units of “work” (moving mass) to transform one distribution into the other.
    Commonly used for comparing probability distributions, especially in optimal transport problems.
    Breakage: When the distributions have disjoint support (no overlap), the Wasserstein distance becomes infinite.

    Kolmogorov-Smirnov Distance (smaller = more similar)
    Measures the maximum vertical difference between the cumulative distribution functions (CDFs) of the two distributions.
    A KS distance of 0.43 suggests that the two distributions differ significantly in terms of their shape or location.
    Breakage: It assumes continuous distributions and may not work well for discrete data.


    Chi-Squared Distance (smaller = more similar)
    Compares the observed and expected frequencies in a contingency table.
    Breakage: It is sensitive to sample size and may not work well for small datasets.

    Kullback-Leibler Divergence (smaller = more similar)
    Measures the relative entropy between two probability distributions.
    Breakage: Not symmetric; it can be infinite if one distribution has zero probability where the other doesn’t.

    Jensen-Shannon Divergence (smaller = more similar)
    A smoothed version of KL divergence that is symmetric and bounded.
    Breakage: Still sensitive to zero probabilities.

    Energy Distance (smaller = more similar)
    Measures the distance between two distributions using the energy distance.
    Breakage: It is sensitive to outliers and may not work well for skewed data and small datasets.

    Mahalanobis Distance (smaller = more similar)
    Mahalanobis distance accounts for correlations between variables.
    Breakage: It assumes that the data is normally distributed and may not work well for non-normal data.

    Returns:
    - The computed distance between the two distributions.
    """
    if len(points1) == 0:
        global_logger.warning(f"First Distribution is empty returning None")
        return None

    if len(points2) == 0:
        global_logger.warning(f"Second Distribution is empty returning None")
        return None

    assert (
        points1.shape[1] == points2.shape[1]
    ), "Distributions must have the same number of dimensions"

    similarity_range = {
        "euclidean": {"lowest": np.inf, "highest": 0.0},
        "wasserstein": {"lowest": np.inf, "highest": 0.0},
        "kolmogorov-smirnov": {"lowest": np.inf, "highest": 0.0},
        "chi2": {"lowest": np.inf, "highest": 0.0},
        "kullback-leibler": {"lowest": np.inf, "highest": 0.0},
        "jensen-shannon": {"lowest": np.inf, "highest": 0.0},
        "energy": {"lowest": np.inf, "highest": 0.0},
        "mahalanobis": {"lowest": np.inf, "highest": 0.0},
        "cosine": {"lowest": 0.0, "highest": 1.0},
        "overlap": {"lowest": 0.0, "highest": 1.0},
        "cross_entropy": {"lowest": 0.0, "highest": 1.0},
    }

    if same_distribution(points1, points2):
        for key in similarity_range.keys():
            if key in metric:
                if points1.shape[0] == 0 or points2.shape[0] == 0:
                    return None
                else:
                    return similarity_range[key]["highest"]
        do_critical(NotImplementedError, f"Metric {metric} not implemented")

    if out_det_method is None:
        out_det_method = "density" if points1[0].shape[0] < 4 else "contamination"
        global_logger.debug(
            f"Outlier detection method is None, using {out_det_method} since dimensions {points1.shape[0]} < 4"
        )

    # Filter out outliers from the distributions
    if filter_outliers:
        org_points1 = points1.copy()
        org_points2 = points2.copy()
        points1 = filter_outlier(
            points1,
            method=out_det_method,
            neighbor_distance=neighbor_distance,
            parallel=parallel,
        )
        points2 = filter_outlier(
            points2,
            method=out_det_method,
            neighbor_distance=neighbor_distance,
            parallel=parallel,
        )
    if metric == "wasserstein":
        distances = [
            wasserstein_distance(points1[:, i], points2[:, i])
            for i in range(points1.shape[1])
        ]
        similarity = np.sum(distances)

    elif metric == "kolmogorov-smirnov":
        # Calculate the Kolmogorov-Smirnov statistic for each dimension and take the maximum
        ks_statistics = [
            ks_2samp(points1[:, i], points2[:, i]).statistic
            for i in range(points1.shape[1])
        ]
        similarity = np.max(ks_statistics)

    elif metric == "chi2":
        # Chi-Squared test (requires binned data, here we just compare histograms)
        hist1, hist2 = points_to_histogram(points1, points2)
        similarity = np.sum((hist1 - hist2) ** 2 / hist2)

    elif metric == "kullback-leibler":
        # Kullback-Leibler Divergence
        hist1, hist2 = points_to_histogram(points1, points2)
        similarity = entropy(hist1, hist2)

    elif metric == "jensen-shannon":
        # Jensen-Shannon Divergence
        hist1, hist2 = points_to_histogram(points1, points2)
        m = 0.5 * (hist1 + hist2)
        similarity = 0.5 * (entropy(hist1, m) + entropy(hist2, m))

    elif metric == "energy":
        # Energy Distance
        distances = [
            energy_distance(points1[:, i], points2[:, i])
            for i in range(points1.shape[1])
        ]
        similarity = np.mean(distances)

    elif metric == "euclidean":
        similarity = euclidean_distance_between_distributions(points1, points2)

    elif metric == "mahalanobis":
        similarity = mahalanobis_distance_between_distributions(points1, points2)

    elif metric == "cosine":
        if points1.ndim == 2:
            v1 = np.mean(points1, axis=0)
        if points2.ndim == 2:
            v2 = np.mean(points2, axis=0)
        similarity = cosine_similarity(v1, v2)

    elif metric == "overlap":
        kde1 = gaussian_kde(points1.T)
        kde2 = gaussian_kde(points2.T)
        overlap = normalized_kde_overlap(kde1, kde2)
        similarity = overlap

    elif metric == "cross_entropy":
        # Cross entropy: H(p, q) = -sum(p * log(q))
        if points1.ndim > 30 or points2.ndim > 30:
            raise NotImplementedError(
                "Cross entropy not implemented properly for high dimesions"
            )
        kde1 = gaussian_kde(points1.T)
        kde2 = gaussian_kde(points2.T)
        kde1_densities = kde1.evaluate(points1.T)
        kde2_densities = kde2.evaluate(points1.T)
        similarity = entropy(kde1_densities, kde2_densities)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if similarity == None:
        raise ValueError(
            f"Similarity is None. Metric: {metric}, points1: {points1}, points2: {points2}"
        )
    return similarity


def normalized_kde_overlap(kde1, kde2):
    """
    Compute the normalized overlap between two Kernel Density Estimation (KDE) distributions.
    A

    Normalization: The normalization step ensures that the final overlap measure is between 0 and 1, making it easier to interpret.
    Interpretation
    Value of 1: A normalized overlap of 1 indicates that the two KDEs are identical in terms of their distribution shapes and positions.
    Value of 0: A normalized overlap of 0 indicates no overlap between the KDEs, meaning the distributions do not share any common area.
    Intermediate Values: Values between 0 and 1 indicate partial overlap, with higher values indicating greater similarity between the distributions.
    """
    # Compute the self-overlaps
    I11 = kde1.integrate_kde(kde1)
    I22 = kde2.integrate_kde(kde2)

    # Compute the cross-overlap
    I12 = kde1.integrate_kde(kde2)

    # Normalize the overlap
    normalized_overlap = I12 / np.sqrt(I11 * I22)
    return normalized_overlap


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calc_kde_entropy(data, bandwidth=None, samples=1000, normalize=True):
    """
    Calculate entropy of a 2D distribution using Kernel Density Estimation.

    Parameters:
    data : array-like
        should have positional data in the form of (n_samples, n_dimensions)
    bandwidth : float, optional
        The bandwidth of the kernel
        It is also possible to use the
            - "scott" to use Scott's Rule of Thumb for bandwidth selection ( default )
            - "silverman" to use Silverman's Rule of Thumb for bandwidth selection
    num_samples : int, optional
        Number of samples to use for Monte Carlo integration

    Returns:
    float
        The estimated entropy
    """
    # The density array now contains the KDE values at each grid point in 3D space
    data_samples = data.shape[0]
    features = data.shape[1]
    if data_samples < features + 1:
        global_logger.warning(
            f"WARNING: Data samples {data_samples} < features {features}. Returning None"
        )
        return None

    if not is_positive_definite(np.cov(data.T)):
        global_logger.warning(
            f"WARNING: Data has no positive definite covariance matrix. Returning None"
        )
        return None

    try:
        kde = (
            gaussian_kde(data.T)
            if bandwidth is None
            else gaussian_kde(data.T, bw_method=bandwidth)
        )
    except:
        global_logger.error(
            f"WARNING: KDE failed to fit the data. Returning None. Data shape: {data.shape}. This need to be changed since perfect single point datasets are not working"
        )
        return None

    # get the probability density estimates at each data point
    # densities = kde.evaluate(data.T)

    # Draw samples from the KDE
    samples = kde.resample(size=samples)

    # Compute log density for the samples
    densities = kde.pdf(samples)
    # log_dens = kde.logpdf(samples)

    # Estimate entropy of the samples
    sample_entropy = calc_entropy(
        densities, convert_to_probabilities=True, normalize=normalize
    )

    return sample_entropy


def calc_entropy(data, convert_to_probabilities=True, normalize=True, base=None):
    """
    Calculate entropy given probabilities or a distribution of numbers that represent a value

    Parameters:
        data : array-like
            Densities to calculate entropy for.
        convert_to_probabilities : bool, optional
            If True, convert the data to probabilities before calculating entropy.
        normalize : bool, optional
            If True, normalize the entropy to be between 0 and 1. By
    """

    ## convert data to probabilities
    probabilities = data / sum(data) if convert_to_probabilities else data
    ## calculate shannon entropy
    data_entropy = entropy(probabilities + 1e-100, base=base)
    if normalize:
        data_entropy /= calc_max_entropy(len(data))
    return data_entropy


def calc_max_entropy(num_classes, base=None):
    """
    Calculate the maximum entropy for a given number of classes.

    Parameters:
        - num_classes: int, the number of classes
        - base: str, the base of the logarithm (default is 'e')

    Returns:
        - float, the maximum entropy
    """
    return np.log(num_classes) / np.log(base) if base else np.log(num_classes)


@njit(nopython=True)
def get_covariance_matrix(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    cov = np.cov(data, rowvar=False)
    return regularized_covariance(cov, epsilon=epsilon)


@njit(nopython=True)
def regularized_covariance(cov_matrix, epsilon=1e-6):
    return cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon


@njit(nopython=True, parallel=True)
def get_cov_mean_invcov(data):
    """
    Compute the mean and covariance matrix of the input data.

    Args:
    data (np.ndarray): Input data, shape (n_samples, n_features)

    Returns:
    tuple: (mean, cov)
        mean (np.ndarray): Mean vector, shape (n_features,)
        cov (np.ndarray): Covariance matrix, shape (n_features, n_features)
    """
    # Compute the covariance matrix for each distribution (can be estimated from data)
    cov = get_covariance_matrix(data)
    # Check if covariance matrix is positive definite
    if not is_positive_definite(cov):
        # If not, find the nearest positive definite matrix
        cov = nearestPD(cov)

    ## Is not possible to do try except in njit mode
    # try:
    #     cov_inv = np.linalg.inv(cov)
    # except np.linalg.LinAlgError:
    #     cov_inv = None

    # Compute the inverse of the covariance matrices
    cov_inv = np.linalg.inv(cov)
    mean = np.zeros(data.shape[1])

    # Manually compute the mean along axis=0
    for i in prange(data.shape[1]):
        mean[i] = np.sum(data[:, i]) / data.shape[0]

    return cov, mean, cov_inv


@njit(nopython=True)
def pca_numba(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Args:
    data (np.ndarray): Input data, shape (n_samples, n_features)
    n_components (int): Number of components to keep

    Returns:
        data_pca (np.ndarray): Transformed data, shape (n_samples, n_components)
    """
    # Compute the covariance matrix
    data = data.astype(np.float64)
    cov = get_covariance_matrix(data)
    # Compute the eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    # extract only the first n_components principal components
    eigvecs_sorted = np.empty((eigvecs.shape[0], n_components), dtype=np.float64)
    for i in range(n_components):
        eigvecs_sorted[:, i] = eigvecs[:, idx[i]]
    data_pca = np.dot(data, eigvecs_sorted)
    return data_pca


@njit(nopython=True)
def mds_numba(data, n_components=2, max_iter=300, tol=1e-6):
    """
    Perform Multidimensional Scaling (MDS) on the input data.

    Args:
        data (np.ndarray): Input data, shape (n_samples, n_features)
        n_components (int): Number of components to keep
        max_iter (int): Maximum number of iterations for optimization
        tol (float): Tolerance for convergence

    Returns:
        data_mds (np.ndarray): Transformed data, shape (n_samples, n_components)
    """
    # Compute the distance matrix
    data = data.astype(np.float64)
    n_samples = data.shape[0]

    # Calculate pairwise Euclidean distances
    dist_matrix = pairwise_euclidean_distance(data, data)

    # Double centering: Convert distance matrix to Gram matrix
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    B = -0.5 * H @ (dist_matrix**2) @ H

    # Compute eigenvalues and eigenvectors of B
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select the top n_components
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    # Calculate the coordinates using the eigenvectors and eigenvalues
    # We use the square root of eigenvalues to scale the eigenvectors
    data_mds = eigvecs * np.sqrt(np.maximum(eigvals, 0))

    return data_mds


@njit(nopython=True)
def sphere_to_plane(points: np.ndarray):
    """
    Convert 3D points on a sphere to 2D points on a plane.
    """
    points = points.astype(np.float64)
    # convert spherical coordinates to 2D plane
    phi = np.arctan2(points[:, 1], points[:, 0])
    theta = np.arccos(points[:, 2])
    points_2d = np.column_stack((phi, theta))
    return points_2d


@njit(nopython=True)
def all_finite(arr):
    """Check if all elements in the array are finite."""
    flat_arr = arr.flat
    all_finite = True
    for i in prange(len(flat_arr)):
        x = flat_arr[i]
        if not np.isfinite(x):
            all_finite = False
    return all_finite


@njit
def is_positive_definite(A):
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except:
        return False


@njit
def nearestPD(A):
    """
    Find the nearest positive definite matrix to input matrix A.
    Uses a more numerically stable approach with careful handling of eigenvalues.

    Parameters:
    -----------
    A : ndarray
        Input matrix to be converted to the nearest positive definite matrix

    Returns:
    --------
    A3 : ndarray
        Nearest positive definite matrix to A
    """
    n = A.shape[0]

    # Symmetrize A
    B = (A + A.T) / 2

    # Compute SVD
    U, s, Vt = np.linalg.svd(B)

    # Ensure eigenvalues are positive
    s = np.maximum(s, 0)

    # Reconstruct matrix with positive eigenvalues
    H = np.dot(U, np.dot(np.diag(s), U.T))

    # Average with original symmetrized matrix
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2  # Ensure perfect symmetry

    # If already positive definite, return result
    if is_positive_definite(A3):
        return A3

    # Otherwise, gradually add to diagonal until positive definite
    spacing = np.max(np.abs(A)) * 1e-9  # Scale spacing with matrix magnitude
    I = np.eye(n)
    k = 0
    max_attempts = 100  # Prevent infinite loops

    while not is_positive_definite(A3) and k < max_attempts:
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig + spacing)
        k += 1
        spacing *= 2  # Increase spacing geometrically if needed

    return A3


@njit(nopython=True)
def mahalanobis_distance_between_distributions(points1, points2):
    """
    Compute the Mahalanobis distance between two distributions.

    Parameters:
    ----------
    points1 : np.ndarray
        First distribution, shape (n_samples1, n_features)
    points2 : np.ndarray
        Second distribution, shape (n_samples2, n_features)

    Returns:
    -------
    float
        Mahalanobis distance between the two distributions
    """
    cov_matrix1, mean1, cov_inv1 = get_cov_mean_invcov(points1)
    cov_matrix2, mean2, cov_inv2 = get_cov_mean_invcov(points2)

    # Sum of inverses of the covariance matrices
    cov_inv_sum = cov_inv1 + cov_inv2

    return mahalanobis_distance(mean1, mean2, cov_inv_sum)


@njit(nopython=True, parallel=True)
def compute_mahalanobis_distances(points, mean, inv_cov):
    """
    Compute the Mahalanobis distance between each point and the distribution.

    Args:
    points (np.ndarray): Input points, shape (n_samples, n_features)
    mean (np.ndarray): Mean of the distribution, shape (n_features,)
    inv_cov (np.ndarray): Inverse of the covariance matrix, shape (n_features, n_features)

    Returns:
    np.ndarray: Mahalanobis distances, shape (n_samples,)
    """
    distances = np.zeros(len(points))
    for i in prange(len(points)):
        distances[i] = mahalanobis_distance(points[i], mean, inv_cov)
    return distances


@njit(nopython=True)
def mahalanobis_distance(x, mean, inv_cov):
    """
    Compute the Mahalanobis distance between a point and the distribution.

    Args:
    x (np.ndarray): Input point, shape (n_features,)
    mean (np.ndarray): Mean of the distribution, shape (n_features,)
    inv_cov (np.ndarray): Inverse of the covariance matrix, shape (n_features, n_features)

    Returns:
    float: Mahalanobis distance
    """
    x = x.astype(np.float64)
    mean = mean.astype(np.float64)
    inv_cov = inv_cov.astype(np.float64)
    diff = x - mean
    distance_squared = diff.dot(inv_cov).dot(diff)
    distance = np.sqrt(distance_squared)
    # if np.isnan(distance):
    #    raise ValueError("Mahalanobis distance is nan")
    return distance


@njit(nopython=True, parallel=True)
def pairwise_mahalanobis_distance(points1, points2, inv_cov):
    """
    Compute the pairwise Mahalanobis distance between two sets of points from the same distribution.

    Args:
    points1 (np.ndarray): First set of points, shape (n_samples1, n_features)
    points2 (np.ndarray): Second set of points, shape (n_samples2, n_features)
    inv_cov (np.ndarray): Inverse of the covariance matrix, shape (n_features, n_features)

    Returns:
    np.ndarray: Pairwise Mahalanobis distance matrix, shape (n_samples1, n_samples2)
    """
    n1, d = points1.shape
    n2 = points2.shape[0]
    distances = np.zeros((n1, n2))
    for i in prange(n1):
        point_distances = compute_mahalanobis_distances(points2, points1[i], inv_cov)
        distances[i] = point_distances
    return distances


# @njit(nopython=True, parallel=True)
def compute_mahalanobis(points, reference=None):
    """Compute Mahalanobis distances for points against a reference distribution or between sets.

    Args:
        points: Input points (n_samples, n_features) or first set
        reference: Reference points (m_samples, n_features) or None to use points' own distribution

    Returns:
        np.ndarray: Distances (n_samples,) if reference=None, else (n_samples, m_samples)
    """
    # Convert to float64 for numerical stability
    points = points.astype(np.float64)

    if reference is None:
        reference = points

    reference = reference.astype(np.float64)
    if reference.shape[0] == 1:
        # not possible to calculate covariance matrix with only one point
        distances = np.full((len(points), len(reference)), np.nan)
        return distances
    else:
        cov, mean, inv_cov = get_cov_mean_invcov(reference)
        distances = np.zeros((len(points), len(reference)))
        if np.array_equal(points, reference):
            # If points and reference are the same, compute distances in a single loop
            for i in prange(len(points)):
                for j in range(i + 1, len(reference)):
                    dist = mahalanobis_distance(points[i], reference[j], inv_cov)
                    distances[i, j] = dist
                    distances[j, i] = dist
        else:
            for i in prange(len(points)):
                for j in range(len(reference)):
                    distances[i, j] = mahalanobis_distance(
                        points[i], reference[j], inv_cov
                    )
    return distances


@njit(nopython=True)
def euclidean_distance_between_distributions(
    points1: np.ndarray, points2: np.ndarray
) -> float:
    """
    Computes a distance metric between two distributions using pairwise Euclidean distances.
    This is done by computing the pairwise Euclidean distances between all points in the two distributions,
    and then taking the mean of these distances.

    Args:
        points1 (np.ndarray): A 2D array of shape (n_samples1, n_features) representing the first distribution.
        points2 (np.ndarray): A 2D array of shape (n_samples2, n_features) representing the second distribution.
        metric (str): A string specifying the metric to use. Only "euclidean" is supported in this implementation.

    Returns:
        float: A non-negative float representing the distance between the two distributions.
    """
    points1 = np.atleast_2d(points1)
    points2 = np.atleast_2d(points2)
    d1 = pairwise_euclidean_distance(points1, points1)
    d2 = pairwise_euclidean_distance(points2, points2)
    d3 = pairwise_euclidean_distance(points1, points2)
    return np.abs(np.mean(d1) + np.mean(d2) - 2 * np.mean(d3))


@njit(nopython=True, parallel=True)
def pairwise_euclidean_distance(points1, points2=None):
    """
    Compute the pairwise Euclidean distance between two sets of points.

    Args:
    points1 (np.ndarray): First set of points, shape (n_samples1, n_features)
    points2 (np.ndarray): Second set of points, shape (n_samples2, n_features)

    Returns:
    np.ndarray: Pairwise Euclidean distance matrix, shape (n_samples1, n_samples2)
    """
    n1, d = points1.shape
    if points2 is None:
        points2 = points1
    n2 = points2.shape[0]
    distances = np.zeros((n1, n2))
    if n1 == n2:
        # If both sets of points are the same, we can compute distances in a single loop
        for i in prange(n1):
            for j in range(i + 1, n2):
                dist = euclidean_distance(points1[i], points2[j])
                distances[i, j] = dist
                distances[j, i] = dist
    else:
        for i in prange(n1):
            for j in range(n2):
                distances[i, j] = euclidean_distance(points1[i], points2[j])
    return distances


def pairwise_compare(
    vectors: np.ndarray,
    metric="pearson",
    filter_outliers=True,
    out_det_method="density",
    neighbor_distance=None,
    parallel=True,
) -> Union[float, np.ndarray]:
    """ """

    if filter_outliers:
        org_vectors = vectors.copy()
        vectors = filter_outlier(
            vectors,
            method=out_det_method,
            neighbor_distance=neighbor_distance,
            parallel=parallel,
        )

    if len(vectors) == 0:
        global_logger.warning(
            f"Vectors are empty. Pairwise comparisson is not possible. Returning None"
        )
        return None

    if metric == "pearson":
        distances = np.corrcoef(vectors)
    elif metric == "cosine":
        # normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        # distances = normalized_vectors @ normalized_vectors.T
        distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    elif metric == "manhattan":
        distances = sklearn.metrics.pairwise.manhattan_distances(vectors)
    elif metric == "euclidean":
        distances = pairwise_euclidean_distance(vectors, vectors)
    elif metric == "kde_entropy":
        # returns a single value describing the entropy of the distribution normalized to 1
        distances = calc_kde_entropy(vectors, bandwidth=None, normalize=True)
    elif metric == "mahalanobis":
        # cov, mean, cov_inv = get_cov_mean_invcov(vectors)
        # distances = pairwise_mahalanobis_distance(vectors, vectors, cov_inv)
        distances = compute_mahalanobis(vectors)
    else:
        err_msg = f"{metric} not supported in function {pairwise_compare}"
        global_logger.critical(err_msg)
        raise ValueError(err_msg)
    return distances


@njit(nopython=True)
def euclidean_distance(x, y):
    """
    Compute the Euclidean distance between two points.

    Args:
    x (np.ndarray): First point, shape (n_features,)
    y (np.ndarray): Second point, shape (n_features,)

    Returns:
    float: Euclidean distance
    """
    diff = x - y
    distances = np.sqrt(np.dot(diff, diff))
    return distances


@njit(nopython=True, parallel=True)
def compute_density(points, neighbor_distance, inv_cov=None):
    """
    Compute the density of each high-dimensional point in the distribution using a neighbor_distance and Mahalanobis distance.
    proper way to define neighbor_distance for outlier detection does not work with mahalanobis distance
    """
    n = points.shape[0]
    densities = np.zeros(n)

    for i in prange(n):
        count = 0
        for j in range(n):
            if i != j:
                """
                # proper way to define neighbor_distance for outlier detection does not work with mahalanobis distance
                if points.ndim <= 3:
                    dist = euclidean_distance(points[i], points[j])
                else:
                    # Compute Mahalanobis distance for high-dimensional data
                    dist = mahalanobis_distance(points[i], points[j], inv_cov)
                """

                ## compute euclidean distance for low dimensional data
                dist = euclidean_distance(points[i], points[j])
                if dist < neighbor_distance:
                    count += 1

        densities[i] = count

    if len(densities) > 0:
        if np.max(densities) > 0:
            densities = densities / np.max(densities)

    # remove nan
    densities = np.nan_to_num(densities, nan=0.0)

    # if densities.sum() == 0:
    # global_logger.info("No points found within neighbor_distance")
    return densities


def get_outlier_mask_numba(
    points, contamination=0.2, neighbor_distance=None, method="contamination"
):
    n, d = points.shape

    # Remove any rows with NaN or inf
    mask_valid = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        mask_valid[i] = all_finite(points[i])
    valid_points = points[mask_valid]

    if len(valid_points) < d + 1:
        # Not enough valid points to compute covariance
        return np.ones(len(valid_points), dtype=np.bool_)

    # Create mask for non-outlier points
    mask_inliers = np.zeros(len(valid_points), dtype=np.bool_)
    if method == "contamination":
        # Compute mean and covariance
        cov, mean, inv_cov = get_cov_mean_invcov(valid_points)

        if inv_cov is None or not is_positive_definite(inv_cov):
            # global_logger.warning("Inversion failed, return valid points.")
            return np.ones(len(valid_points), dtype=np.bool_)

        # distances = compute_mahalanobis_distances(valid_points, mean, inv_cov)
        distances = compute_mahalanobis
        distances = np.nan_to_num(distances, nan=np.inf)

        # Determine threshold based on contamination (percentage of outliers)
        threshold = np.percentile(
            distances[distances < np.inf], 100 * (1 - contamination)
        )
        mask_inliers = distances <= threshold

    elif method == "density":
        # Determine threshold based on density estimation
        # Compute densities for each point
        if neighbor_distance is None:
            raise ValueError(
                "neighbor_distance must be specified for density-based outlier detection"
            )
        densities = compute_density(valid_points, neighbor_distance)
        densities = np.nan_to_num(densities, nan=0.0)

        # Determine threshold based on contamination (percentage of lowest density points)
        threshold = (
            np.percentile(densities, 100 * contamination)
            if sum(densities) > 0
            else np.inf
        )
        mask_inliers = densities >= threshold

    # check if all points are outliers
    if np.sum(mask_inliers) == 0:
        global_logger.error(
            f"All points are outliers. Returning empty array. num points: {n}, dimensions: {d}"
        )
    return mask_inliers


@njit(nopython=True, parallel=True)
def points_to_histogram(points1, points2, bins=None):
    eps = 1e-10  # Small value to avoid division by zero and log(0)

    # Determine number of bins using Sturges' rule if bins are not provided
    if bins is None:
        n1, n2 = len(points1), len(points2)
        bins = int(np.ceil(np.log2(max(n1, n2)) + 1))

    # Compute range manually since we can't use np.min/max with arrays directly in Numba
    min_val = min(np.min(points1), np.min(points2))
    max_val = max(np.max(points1), np.max(points2))
    bin_range = (min_val, max_val)

    # Create bin edges manually
    bin_width = (max_val - min_val) / bins
    bin_edges = np.zeros(bins + 1)
    for i in range(bins + 1):
        bin_edges[i] = min_val + i * bin_width

    # Compute histograms manually
    hist1 = np.zeros(bins)
    hist2 = np.zeros(bins)

    # Count points in each bin for points1
    for point in points1:
        bin_idx = min(bins - 1, max(0, int((point - min_val) / bin_width)))
        hist1[bin_idx] += 1

    # Count points in each bin for points2
    for point in points2:
        bin_idx = min(bins - 1, max(0, int((point - min_val) / bin_width)))
        hist2[bin_idx] += 1

    # Normalize to get density
    hist1 = hist1 / (len(points1) * bin_width)
    hist2 = hist2 / (len(points2) * bin_width)

    # Add small value and normalize
    hist1 = hist1 + eps
    hist2 = hist2 + eps
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)

    return hist1, hist2


def filter_outlier(
    points,
    contamination=0.2,
    neighbor_distance=None,
    method="density",
    parallel=True,
    only_mask=False,
):
    """
    Filter out outliers from a set of points using a simplified Elliptic Envelope method.

    This function implements a basic form of outlier detection using Mahalanobis distances.
    It's optimized for speed using Numba's just-in-time compilation.

    Args:
    points (np.ndarray): Input points, shape (n_samples, n_features)
    contamination (float): The proportion of outliers in the data set. Default is 0.2.

    Returns:
    np.ndarray: Filtered points with outliers removed

    Note:
    - This is a simplified implementation and may not be as robust as sklearn's EllipticEnvelope
      for all datasets.
    - The first run will include compilation time; subsequent runs will be much faster.
    """
    global_logger.debug(f"Filter outliers based on {method}")
    n, d = points.shape
    if n < 10:
        global_logger.warning(
            f"Not enough points to filter outliers. Returning all points. num points: {n}, dimensions: {d}"
        )
        return points

    # Detect outlier mask
    if parallel:
        if n < d + 1:
            global_logger.debug(
                f"Not enough points to compute covariance. Returning all points. num points: {n}, dimensions: {d}"
            )
            return points
        mask_valid = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            mask_valid[i] = all_finite(points[i])
        valid_points = points[mask_valid]

        mask = get_outlier_mask_numba(
            points, contamination, neighbor_distance=neighbor_distance, method=method
        )
    else:
        if method == "density":
            raise NotImplementedError(
                f"Method 'density' not implemented for not parallel filter_outlier"
            )
        elif method == "contamination":
            # Filter out outliers from a 2D distribution using an Elliptic Envelope.
            # The algorithm fits an ellipse to the data, trying to encompass the most concentrated 80% of the points (since we set contamination to 0.2).
            # Points outside this ellipse are labeled as outliers.
            #    - It estimates the robust covariance of the dataset.
            #    - It computes the Mahalanobis distances of the points.
            #    - Points with a Mahalanobis distance above a certain threshold (determined by the contamination parameter) are considered outliers.
            global_logger.debug(
                f"Using Elliptic Envelope for outlier detection with contamination: {contamination}"
            )
            outlier_detector = EllipticEnvelope(
                contamination=contamination, random_state=42
            )
        else:
            do_critical(ValueError, f"Filtering method {method} not supported")
        num_points = points.shape[0]
        mask = (
            outlier_detector.fit_predict(points) != -1
            if num_points > 2
            else np.ones(num_points, dtype=bool)
        )

    # filter
    filtered_points = valid_points[mask]
    if only_mask:
        return mask
    else:
        return filtered_points


@njit(nopython=True)
def numba_cross(a, b):
    """
    Numba implementation of np.cross function.

    Parameters:
    a (np.ndarray): First input array.
    b (np.ndarray): Second input array.

    Returns:
    np.ndarray: Cross product of a and b.

    Note:
    - This function assumes that a and b are 2D or 3D arrays.
    """
    if a.shape[-1] == 2 and b.shape[-1] == 2:
        result = np.zeros_like(a)
        result[0] = a[0] * b[1] - a[1] * b[0]
    elif a.shape[-1] == 3 and b.shape[-1] == 3:
        result = np.zeros_like(a)
        result[0] = a[1] * b[2] - a[2] * b[1]
        result[1] = a[2] * b[0] - a[0] * b[2]
        result[2] = a[0] * b[1] - a[1] * b[0]
    else:
        raise ValueError("Input arrays must be 2D or 3D.")
    return result


def intersect_segments(seg1, seg2):
    """
    Determine if two line segments intersect in any number of dimensions (2D or 3D) and return the point of intersection.

    Parameters:
    seg1 (list of tuples): The first line segment, represented by two points [(p1), (p2)].
    seg2 (list of tuples): The second line segment, represented by two points [(q1), (q2)].

    Returns:
    tuple or None: If the segments intersect, returns the coordinates of the
                   intersection point. If they don't intersect or are parallel, returns None.

    Note:
    - The function assumes that the input segments are valid (i.e., two distinct points for each segment).
    - Parallel or collinear segments are considered as non-intersecting unless their projections overlap.
    """
    # Convert points to numpy arrays for easier manipulation
    p1, p2 = np.array(seg1[0]), np.array(seg1[1])
    q1, q2 = np.array(seg2[0]), np.array(seg2[1])

    # Direction vectors for each segment
    r = p2 - p1
    s = q2 - q1

    # Check if the segments are parallel (determinant == 0)
    r_cross_s = numba_cross(r, s)
    r_cross_s_magnitude = np.linalg.norm(r_cross_s)

    if r_cross_s_magnitude == 0:
        return None  # Segments are parallel or collinear

    # Calculate the parameters t and u for the parametric equations
    qp = q1 - p1
    t = np.sum(numba_cross(qp, s)) / r_cross_s_magnitude
    u = np.sum(numba_cross(qp, r)) / r_cross_s_magnitude

    # Check if the intersection point lies on both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate the point of intersection
        intersection_point = p1 + t * r
        return tuple(intersection_point)
    else:
        return None


def area_of_intersection(hull1, hull2):
    """
    Calculate the area of intersection between two convex hulls.

    This function finds the intersection points between the edges of two convex hulls
    and calculates the area of the resulting intersection polygon.

    Parameters:
    hull1 (scipy.spatial.ConvexHull): The first convex hull
    hull2 (scipy.spatial.ConvexHull): The second convex hull

    Returns:
    float: The area of intersection between the two convex hulls.
           Returns 0 if there's no intersection or if the intersection is a point or a line.

    Note:
    - This function assumes that the hulls are 2D convex hulls.
    - It uses the `intersect_segments` function to find edge intersections.
    - It also includes points from one hull that lie inside the other hull.
    - The function depends on an `in_hull` function (not provided) to check if a point is inside a hull.
    - It uses scipy's ConvexHull to calculate the final intersection area.

    Raises:
    Any exceptions raised by the ConvexHull constructor if the intersection points are invalid.
    """
    intersect_points = []
    for simplex1 in hull1.simplices:
        for simplex2 in hull2.simplices:
            intersect = intersect_segments(
                hull1.points[simplex1], hull2.points[simplex2]
            )
            if intersect is not None:
                intersect_points.append(intersect)

    # Also include points from hull1 that are inside hull2 and vice versa
    for point in hull1.points:
        if in_hull(point, hull2):
            intersect_points.append(point)
    for point in hull2.points:
        if in_hull(point, hull1):
            intersect_points.append(point)

    area = 0
    if len(intersect_points) > 2:
        try:
            if hull1.points.shape[1] == 2:
                intersection_hull = ConvexHull(intersect_points)
                area = intersection_hull.area
            elif hull1.points.shape[1] == 3:
                intersection_hull = ConvexHull(intersect_points)
                area = intersection_hull.volum
        except:
            area = 0
    return area


def in_hull(point, hull):
    # Check if a point is inside a convex hull
    return all((np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in hull.equations)


## normalization
def normalize_01(vector, axis):
    """
    Normalize a vector to the range [0, 1].

    Default normalization is along the columns (axis=1).

    Args:
    ------
    - vector (np.ndarray): Input vector to normalize.
    - axis (int): Axis along which to normalize the vector.

    Returns:
    ------
    - normalized_vector (np.ndarray): Normalized vector.
    """
    vector = np.array(vector)
    axis = 0 if axis == 1 and len(vector.shape) == 1 else axis
    min_val = np.min(vector, axis=axis, keepdims=True)
    max_val = np.max(vector, axis=axis, keepdims=True)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector


def create_list_of_lists(data: Union[str, List[str], List[List[str]]]):
    """
    Create a list of lists from a string or a list of strings.

    Args:
    ------
    - data (Union[str, List[str], List[List[str]]): Input data to convert to a list of lists.

    Returns:
    ------
    - data (List[List[str]]): List of lists.
    """
    if not isinstance(data, list):
        data = [data]
    elif isinstance(data, list) and all(isinstance(prop, str) for prop in data):
        data = [data]
    return data


# strings
def search_split(
    regex: Union[str, re.Pattern],
    string: str,
):
    """
    Split a string using a regex pattern.

    Args:
    - string (str): The input string to split.
    - regex (str or re.Pattern): The regex pattern to use for splitting.
        examples:
            r"[\s,;]+"  for splitting by whitespace, commas, or semicolons
            r"\d+" for splitting by digits

    Returns:
    - List[str]: A list of substrings obtained by splitting the input string.
    """
    if isinstance(regex, str):
        regex = re.compile(regex)

    search = re.search(regex, string)
    search_str = int(search.group())
    nosearch = string.replace(search.group(0), "").strip()

    return nosearch, search_str


def filter_strings_by_properties(
    strings: List[str],
    include_properties: Union[List[List[str]], List[str], str] = None,
    exclude_properties: Union[List[List[str]], List[str], str] = None,
) -> List[str]:
    """
    Filters a list of strings based on given properties.

    Args:
    - strings (list): List of strings to filter.
        - include_properties: Union[List[List[str]], List[str], str] = None,
        Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
        1. Include all models containing a specific string: "string"
        2. Include all models containing a specific combination of strings: ["string1", "string2"]
        3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
    - exclude_properties: Union[List[List[str]], List[str], str] = None,
        Same as model_naming_filter_include but for excluding models.

    Returns:
    - filtered_strings (list): List of strings filtered based on the given properties.
    """
    filtered_strings = []

    if include_properties:
        include_properties = create_list_of_lists(include_properties)

    if exclude_properties:
        exclude_properties = create_list_of_lists(exclude_properties)

    for string in strings:
        org_string = copy.deepcopy(string)
        string = string.lower()
        include_check = False
        if include_properties:
            # Check if any of the include properties lists are present in the string
            for props in include_properties:
                props = make_list_ifnot(props)
                if all(prop.lower() in string for prop in props):
                    include_check = True
                    break
        else:
            include_check = True

        exclude_check = False
        if exclude_properties:
            # Check if any of the exclude properties lists are present in the string
            for props in exclude_properties:
                props = make_list_ifnot(props)
                if all(prop.lower() in string for prop in props):
                    exclude_check = True
                    break
        else:
            exclude_check = False

        # Only include the string if it matches include properties and does not match exclude properties
        if include_check and not exclude_check:
            filtered_strings.append(org_string)

    return filtered_strings


def check_correct_metadata(string_or_list, name_parts):
    name_parts = make_list_ifnot(name_parts)
    success = True
    if isinstance(string_or_list, Path):
        if not string_or_list.exists():
            success = False
            global_logger.error(f"No matching file found")
            print(f"No matching file found")
        else:
            string_or_list = string_or_list.name

    if success:
        for name_part in name_parts:
            if not name_part in string_or_list:
                success = False
            if not success:
                global_logger.error(
                    f"Metadata naming does not match Object Metadata: {string_or_list} != {name_parts}"
                )
    return success


def clean_filename(fname):
    """
    Clean a filename by replacing special characters with underscores.
    """
    dot_chars = [" ", ":", ",", ";"]
    shift_chars = ["\n", "\r", "\t"]
    brackets = ["(", ")", "[", "]", "{", "}"]
    special_chars = [":", '"', "/", "\\", "|", "?", "*"]

    nice = Path(fname).name
    for char in dot_chars + shift_chars:
        nice = nice.replace(char, "_")
    for char in special_chars + brackets:
        nice = nice.replace(char, "")
    nice = nice.replace(">", "bigger")
    nice = nice.replace("<", "smaller")

    while "__" in nice:
        nice = nice.replace("__", "_")
    nice = Path(fname).parent / nice
    return nice


# dataframe
def filter_dataframe(
    df: pd.DataFrame,
    filter_by: Union[str, List[str], List[List[str]]],
    filter: Literal["columns", "index", "both"] = "both",
) -> pd.DataFrame:
    """
    Filter DataFrame based on animal IDs, tasks, and constraints.

    Parameters:
    - df: DataFrame containing fluorescence data
    - filter_by: String, list of strings, or list of lists of strings to filter by
    - filter: Whether to filter 'columns', 'index', or 'both'

    Returns:
    - Filtered DataFrame
    """
    # Convert filter_by to list of lists
    filter_by = filter_by if isinstance(filter_by, list) else [filter_by]
    if not isinstance(filter_by[0], list):
        filter_by = [filter_by]

    # Initialize masks
    index_mask = np.ones(len(df.index), dtype=bool)
    column_mask = np.ones(len(df.columns), dtype=bool)

    # Build patterns and update masks
    for constraints in filter_by:
        # Create pattern with word boundaries for numeric parts
        pattern = create_filter_pattern(constraints)

        if filter in ["index", "both"]:
            index_mask &= df.index.astype(str).str.contains(pattern, na=False)

        if filter in ["columns", "both"]:
            column_mask &= df.columns.astype(str).str.contains(pattern, na=False)

    # Apply filtering based on filter parameter
    if filter == "index":
        filtered_df = df.loc[index_mask, :]
    elif filter == "columns":
        filtered_df = df.loc[:, column_mask]
    else:  # both
        filtered_df = df.loc[index_mask, column_mask]

    if filtered_df.empty:
        raise ValueError(
            "Filtered DataFrame is empty. Check constraints, animal IDs, or tasks."
        )

    return filtered_df


def group_df_by_custom_groups(
    df: pd.DataFrame,
    groups: Optional[Dict[str, Dict[str, List[str]]]],
    group_by: str,
    compare_by: str,
) -> pd.core.groupby.DataFrameGroupBy:
    """
    Groups a DataFrame based on a custom grouping structure defined by a dictionary.
    Adds 'group_name' and 'group_task_number' columns to the DataFrame to represent
    the grouped structure.

    If `groups` is None, autogrouping is performed: for each unique value in `group_by`,
    a single subgroup named 'default' is created containing all unique values in `compare_by`
    for that initial group, sorted alphabetically.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data. A copy is made internally
      to avoid modifying the original DataFrame.
    - groups (Optional[Dict[str, Dict[str, List[str]]]]): Dictionary defining the grouping structure.
      Outer keys match unique values in `group_by`. Inner dictionaries have subgroup names as keys
      and lists of values (from `compare_by`) as values. Example:
        {
            'condition1': {
                'task_group1': ['task1', 'task2'],
                'task_group2': ['task3']
            },
            'condition2': {
                'task_group1': ['task1', 'task2']
            }
        }
      If None, autogrouping is applied as described above.
    - group_by (str): Column name for initial grouping (e.g., 'condition').
    - compare_by (str): Column name for subgroup matching (e.g., 'task_name').

    Returns:
    - pd.core.groupby.DataFrameGroupBy: DataFrame grouped by 'group_name', with additional columns:
      - 'group_name': Constructed as '{initial_group} - {task_group_name}'.
      - 'group_task_number': Zero-based index of the value in the subgroup list.

    Raises:
    - ValueError: If `group_by` or `compare_by` not in DataFrame columns, if grouping structure
      is invalid (e.g., non-list subgroups, non-string values, duplicates, overlaps), if initial
      groups in DataFrame are missing from `groups`, or if any rows remain unassigned after grouping.
    """
    # Validate input columns
    if group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' not found in DataFrame.")
    if compare_by not in df.columns:
        raise ValueError(f"Column '{compare_by}' not found in DataFrame.")

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Initialize new columns with NaN/None
    df["group_name"] = None
    df["task_group_name"] = None
    df["group_task_number"] = None

    # Handle autogrouping if groups is None
    if groups is None:
        groups = {}
        for init_group in df[group_by].unique():
            tasks = sorted(df[df[group_by] == init_group][compare_by].unique())
            groups[init_group] = {"default": tasks}
        raise NotImplementedError("Autogrouping is not tested yet")

    # Get unique initial groups from DataFrame
    unique_inits = df[group_by].unique()

    # Process each initial group
    for init_group_name in unique_inits:
        if init_group_name not in groups:
            raise ValueError(
                f"Initial group '{init_group_name}' not found in groups dictionary."
            )

        # Validate subgroups for this initial group: check disjoint, no duplicates, types
        all_tasks = set()
        for task_group_name, task_name_list in groups[init_group_name].items():
            if not isinstance(task_name_list, list):
                raise ValueError(
                    f"Subgroup '{task_group_name}' for '{init_group_name}' must be a list."
                )
            if not all(isinstance(item, str) for item in task_name_list):
                raise ValueError(
                    f"All items in subgroup '{task_group_name}' for '{init_group_name}' must be strings."
                )
            if len(set(task_name_list)) != len(task_name_list):
                raise ValueError(
                    f"Duplicate items in subgroup '{task_group_name}' for '{init_group_name}'."
                )

            subgroup_set = set(task_name_list)
            if subgroup_set & all_tasks:
                raise ValueError(
                    f"Overlapping items across subgroups for '{init_group_name}'."
                )
            all_tasks |= subgroup_set

        # Now assign for each subgroup
        for task_group_name, task_name_list in groups[init_group_name].items():
            group_name = f"{init_group_name} - {task_group_name}"

            # Mask for rows in this subgroup
            mask = (df[group_by] == init_group_name) & (
                df[compare_by].isin(task_name_list)
            )

            # set task_group_name
            df.loc[mask, "task_group_name"] = task_group_name

            # Assign group_name
            df.loc[mask, "group_name"] = group_name

            # Create mapping from item to its index in the list
            item_to_index = {item: idx for idx, item in enumerate(task_name_list)}

            # Assign group_task_number using map
            df.loc[mask, "group_task_number"] = df.loc[mask, compare_by].map(
                item_to_index
            )

    # Check if all rows have been assigned a group
    if df["group_name"].isna().any():
        raise ValueError(
            "Some rows were not assigned to any group. Ensure groups cover all data."
        )

    # Group by 'group_name' and return
    grouped_df = df.groupby("group_name")

    return grouped_df


def create_filter_pattern(
    items: Union[List[str], str],
) -> str:
    """
    Create a regex pattern from a list of strings, ensuring numeric parts are treated as whole words.

    This function escapes special regex characters in the input strings and adds word boundaries
    around numeric parts to prevent partial matches (e.g., "FS1" matches "FS1" but not "FS10").
    It handles edge cases such as empty inputs, non-string inputs, and constraints without numbers.

    Args:
        items: A single string or a list of strings to create a regex pattern from.

    Returns:
        str: A regex pattern string that matches any of the input items with proper boundaries.

    Raises:
        ValueError: If the input is empty, not a string or list of strings, or contains invalid elements.
    """
    # Convert single string to list for uniform processing
    items = make_list_ifnot(items)

    # Check for empty input
    if not items:
        raise ValueError(
            "Input cannot be empty. Provide at least one string constraint."
        )

    # Validate that all items are strings
    if not all(isinstance(item, str) for item in items):
        raise ValueError("All items must be strings.")

    # Initialize list to store escaped patterns
    escaped_items = []

    for item in items:
        # Skip empty strings
        if not item.strip():
            continue

        # Check if constraint contains a number (e.g., "FS1", "1FS", "bla1bla")
        match_number = re.match(r"^(.*?)(?<!\d)(\d+)(?!\d)(.*?)$", item)
        if match_number:
            # Handle case where constraint contains a number
            prefix, num, suffix = match_number.groups()
            # Escape prefix and suffix, and ensure number is matched exactly
            escaped_items.append(
                re.escape(prefix) + r"(?<!\d)" + num + r"(?!\d)" + re.escape(suffix)
            )
        else:
            # Handle non-numeric constraints (e.g., "FS")
            escaped_items.append(re.escape(item))
    # Check if any valid patterns were created
    if not escaped_items:
        raise ValueError("No valid patterns created. Check input constraints.")

    # Join all patterns with OR operator
    pattern = "|".join(escaped_items)

    return pattern


def merge_df_cols(
    df: pd.DataFrame,
    merge_by: Literal["animal", "date", "task", "stimulus"] = None,
) -> pd.DataFrame:
    # Extract unique merge_by values (e.g., animal IDs, dates, or tasks)
    df_columns = df.columns.tolist()
    column_groups = defaultdict(list)
    for task_id in df_columns:
        # Assuming task_id is a string formatted as "animalid_date_taskname"
        animal_id, date, task_name, stimulus_type = task_id.split("_")
        if merge_by == "animal":
            key = animal_id
        elif merge_by == "date":
            key = date
        elif merge_by == "task":
            key = task_name
        elif merge_by == "stimulus":
            key = stimulus_type
        else:
            raise ValueError(f"Invalid merge_by parameter: {merge_by}")
        column_groups[key].append(task_id)

    # Create a new DataFrame with merged columns
    merged_data = {}
    for key, values in column_groups.items():
        # Get all columns for this key
        # Concatenate data from these columns into a single column
        # Stack all values into a single series (longer column)
        merged_col = pd.concat([df[col] for col in values], axis=0, ignore_index=True)
        merged_data[f"{key}"] = merged_col

    # Create new DataFrame with merged columns
    merged_df = pd.DataFrame(merged_data)

    return merged_df


def filter_merge_df(
    df: pd.DataFrame,
    filter_by: Union[str, List[str], List[List[str]]] = None,
    merge_by: Literal["animal", "date", "task", "stimulus"] = None,
) -> pd.DataFrame:
    """
    Filter and merge a DataFrame based on specified criteria.
    Parameters:
    - df: DataFrame to filter and merge
    - filter_by: String, list of strings, or list of lists of strings to filter by
    - merge_by: String indicating the column to merge by ('animal', 'date', or 'task')

    Returns:
    - Merged DataFrame after filtering and merging columns
    """
    # filter the dataframe
    if filter_by is not None:
        filtered_df = filter_dataframe(df, filter_by, filter="columns")
    else:
        filtered_df = df

    # merge the dataframe
    if merge_by is not None:
        merged_df = merge_df_cols(filtered_df, merge_by=merge_by)
    else:
        merged_df = filtered_df

    return merged_df


# matrix
def modify_mtx(
    mtx: np.ndarray,
    whiten: bool = True,
    norm: bool = True,
) -> np.ndarray:
    """Modify a matrix by centering columns and/or normalizing Frobenius norm.

    Args:
        mtx: Input matrix of shape (n, m).
        whiten: If True, center columns to zero mean.
        norm: If True, normalize by Frobenius norm.

    Returns:
        Modified matrix of same shape.

    Raises:
        ValueError: If matrix is not 2D or has zero norm when norm=True.
    """
    if mtx.ndim != 2:
        raise ValueError("Input matrix must be two-dimensional")

    mtx = np.array(mtx, dtype=np.float64)
    if whiten:
        mtx = mtx - np.mean(mtx, axis=0)
    if norm:
        norm = np.linalg.norm(mtx)
        if norm == 0:
            raise ValueError("Input matrix must contain >1 unique points")
        mtx /= norm

    return mtx


def align_mtx(
    mtx1: np.ndarray,
    mtx2: np.ndarray,
    rotate: bool = True,
    scale: bool = True,
    whiten: bool = True,
    norm: bool = True,
) -> np.ndarray:
    """Align mtx2 to mtx1 using Procrustes analysis.

    Args:
        mtx1: Reference matrix of shape (n, m).
        mtx2: Matrix to align, same shape as mtx1.
        rotate: If True, apply optimal rotation.
        scale: If True, apply optimal scaling. Note: For shape similarity
            comparisons with normalized matrices, this should be False to
            preserve the normalized shape comparison. Scaling is only
            appropriate for general Procrustes analysis where scale
            differences are meaningful.
        whiten: If True, center columns to zero mean before alignment.
        norm: If True, normalize by Frobenius norm before alignment.

    Returns:
        Aligned mtx2.

    Raises:
        ValueError: If matrices have different shapes or are not 2D.
    """
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must have the same shape")
    if mtx1.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")

    mtx1 = modify_mtx(mtx1, whiten=whiten, norm=norm)
    mtx2 = modify_mtx(mtx2, whiten=whiten, norm=norm)

    # Find optimal orthogonal transformation (rotation/reflection)
    # R transforms mtx1 to mtx2: mtx1 @ R ≈ mtx2
    # So to align mtx2 to mtx1, we use: mtx2 @ R.T
    # The scale 's' is the sum of singular values (a similarity measure)
    # and also equals the optimal scaling factor for general Procrustes
    if rotate or scale:
        R, s = orthogonal_procrustes(mtx1, mtx2)
        if rotate:
            mtx2 = np.dot(mtx2, R.T)
        if scale:
            mtx2 *= s

    return mtx2


# array
class DataFilter:
    """
    A class to keep track of filtering frames of datasets connected to the same event.

    Attributes
    ----------
    filter : dict
        A dictionary containing the filter criteria for different types of data.
        Allowed keys are defined in DataFilter.allowed_keys.

    Attributes
    ----------
    use_frame : dict
        A dictionary containing the filtered frames for each key.
        Allowed keys are defined in DataFilter.allowed_keys.
    """

    allowed_keys = [
        "global",
        "photon",
        "probe",
        "position",
        "distance",
        "moving",
        "velocity",
        "acceleration",
        "stimulus",
    ]

    convert_to_euclidean = [
        "velocity",
        "acceleration",
    ]

    def __init__(
        self, filters: Dict[str, Dict[Literal["range"], Tuple[float, float]]] = {}
    ):
        self.filters = filters
        self.use_frame: Dict[str, List[bool]] = {}

    @staticmethod
    def filter_dict_to_str(
        filters: Dict[str, Dict[Literal["range"], Tuple[float, float]]],
        short: bool = False,
    ) -> str:
        """Convert the filter dictionary to a string representation.

        Parameters
        ----------
        filters : dict
            A dictionary containing the filter criteria for different types of data.
            Allowed keys are defined in DataFilter.allowed_keys.

        short : bool, optional
            If True, the string representation will be shortened.
            Default is False.

        Returns
        -------
        str
            A string representation of the filter dictionary.
        """
        desc = ""
        if filters:
            for key, value in filters.items():
                if len(value) != 0:
                    for filter_key, filter_value in value.items():
                        if filter_key == "range":
                            filter_value_txt = (
                                f"{filter_value[0]:.2f}-{filter_value[1]:.2f}"
                            )
                        elif filter_key == "parts":
                            filter_value_txt = "["
                            for part in filter_value:
                                filter_value_txt += f"({part[0]:.2f}-{part[1]:.2f}), "
                            filter_value_txt = filter_value_txt.strip(", ") + "]"

                        else:
                            filter_value_txt = str(filter_value)

                        if short:
                            desc += f"{key[:3]} {filter_value_txt}, "
                        else:
                            desc += f"{key}: {filter_value_txt}, "
        desc = desc.strip(", ")
        if len(desc) != 0 and not short:
            desc = f"\n|Filters: {desc}|"

        return desc

    def filter_description(self, short=False) -> str:
        """
        Get a string description of the current filters.

        Returns
        -------
        str
            A string representation of the current filters.
        """
        return DataFilter.filter_dict_to_str(self.filters, short=short)

    def get_use_frames(self, key: str = "global") -> List[bool]:
        """
        Get the filtered frames for the given key.

        Parameters
        ----------
        key : str
            The key to get the filtered frames for. Default is "global".
            Allowed keys are defined in DataFilter.allowed_keys.

        Returns
        -------
        List[bool]
            The filtered frames for the given key.
        """
        self.check_key(key)
        if len(self.use_frame) == 0:
            global_logger.error(
                f"No filtered frames available for {key}. Returning None."
            )
            return None
        if key not in self.use_frame:
            global_logger.error(f"Key {key} not found in use_frame. Returning None.")
            return None
        return self.use_frame[key]

    def update(
        self,
        kwargs: Dict[
            Literal[
                "position",
                "photon",
                "probe",
                "position",
                "distance",
                "moving",
                "velocity",
                "acceleration",
                "stimulus",
            ],
            Dict[Literal["range"], Tuple[float, float]],
        ],
    ):
        self.check_key(key)
        """
        Set the filter for the DataFilter.
        
        kwargs should be a dictionary with keys as the type of filter and values as the filter parameters.
        
        """
        for key, value in kwargs.items():
            if key not in self.filters:
                global_logger.warning(
                    f"Key {key} not found in filter. Adding it to the filter."
                )
            elif key in self.filters and self.filters[key] != value:
                global_logger.warning(
                    f"Key {key} already exists in filter with value {self.filters[key]}. Updating it to {value}."
                )
            self.filters[key] = value

    def define_global_filter(self):
        """
        Define the global filter for the DataFilter.

        This method should be implemented in subclasses to define the global filter.
        """

        raise NotImplementedError(
            "DataFilter.define_global_filter is not implemented yet. Use update or set_filter instead."
        )

    def check_key(self, key: str):
        """
        Check if the given key is allowed in DataFilter.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        bool
            True if the key is allowed, False otherwise.
        """
        if key not in DataFilter.allowed_keys:
            do_critical(
                ValueError,
                f"Key {key} not allowed. Allowed keys are {DataFilter.allowed_keys}.",
            )

    def in_range(self, data, range: Tuple[float, float]) -> np.ndarray:
        """
        Check if the data is within the given range.

        Parameters
        ----------
        range : tuple
            A tuple of (min, max) values to filter the data.

        Returns
        -------
            include_frames : np.ndarray
        """
        if not isinstance(range, tuple) or len(range) != 2:
            do_critical(
                ValueError,
                f"Data filter {key} requires a tuple of (min, max) values. Got {range}.",
            )

        min_val, max_val = range
        if min_val is not None:
            min_include_frames = data >= min_val
        else:
            min_include_frames = np.full_like(data, True, dtype=bool)
        if max_val is not None:
            max_include_frames = data <= max_val
        else:
            max_include_frames = np.full_like(data, True, dtype=bool)

        include_frames = min_include_frames & max_include_frames

        return include_frames

    def by_parts(
        self, data: np.ndarray, parts: List[Tuple[float, float]]
    ) -> List[bool]:
        """
        Filter the data by predefinded parts.

        A Part taken from a continuous list of same values (e.g. moving segments).

        Parameters
        ----------
        data : np.ndarray
            The data to filter.
        parts : List[Tuple[float, float]]
            A list of tuples of (min, max) values to filter the data.
            The values are given in fractions of the total length of the data.
        """
        min_frames = 10
        # make data 1d if 2d array with last dimension 1
        if len(data.shape) == 2 and data.shape[1] == 1:
            data = data[:, 0]
        if len(data.shape) != 1:
            raise ValueError("Data must be a 1D array.")

        changes = np.where(np.abs(np.diff(data)) > 0)[0] + 1
        # add end of data as change
        changes = np.append(changes, len(data))
        # check if parts overlap
        np_parts = np.array(parts)
        if np.any(np_parts[:, 0] < 0) or np.any(np_parts[:, 1] > 1):
            raise ValueError("Parts must be in the range [0, 1]")
        if np.any(np_parts[:, 0] >= np_parts[:, 1]):
            raise ValueError(
                "Parts must be in the format (start, end) with start < end"
            )
        if np.any(np_parts[:-1, 1] > np_parts[1:, 0]):
            raise ValueError("Parts must not overlap")

        start = 0
        include_frames = [False] * len(data)
        for change in changes:
            num_idx = change - start
            if num_idx >= min_frames:
                for ps, pe in parts:
                    seg_s = np.floor(ps * num_idx).astype(int) + start
                    seg_e = np.floor(pe * num_idx).astype(int) + start
                    include_frames[seg_s:seg_e] = [True] * (seg_e - seg_s)
            start = change
        return np.array(include_frames)

    def update_global_use_frames(self) -> List[bool]:
        # traveser through all saved filters and combine them
        global_include_frames = None
        for key, datasets_object_include_frames_by_filtertype in self.use_frame.items():
            # skip global key
            if key == "global":
                continue
            for (
                filter_key,
                datasets_object_include_frames,
            ) in datasets_object_include_frames_by_filtertype.items():
                if global_include_frames is None:
                    global_include_frames = datasets_object_include_frames
                else:
                    # combine the include frames with logical AND operation
                    global_include_frames = logical_and_extend(
                        global_include_frames, datasets_object_include_frames
                    )
        self.use_frame["global"] = global_include_frames
        return global_include_frames

    def filter(self, data: np.ndarray, key: str = "global") -> List[bool]:
        """
        Filter the data based on the filter criteria.

        Parameters
        ----------
        data : np.ndarray
            The data to filter.
        key : str
            The key to filter by. Default is "global".
            Allowed keys are defined in DataFilter.allowed_keys.

        Returns
        -------
            filtered_data : np.ndarray
                The filtered data.
        """
        self.check_key(key)
        if data is None:
            do_critical(
                ValueError,
                f"No data provided for filtering. Please provide data to filter.",
            )
            return None

        if key not in self.filters:
            dfilter = {}
        else:
            dfilter = self.filters[key]

        if key not in self.use_frame:
            self.use_frame[key] = {}

        # Transform data if necessary
        if key in DataFilter.convert_to_euclidean:
            transformed_data = np.linalg.norm(abs(data), axis=-1)
        else:
            transformed_data = data

        # create a mask for the full data if no filter is set
        if len(dfilter) == 0:
            include_frames = np.full(transformed_data.shape[0], True, dtype=bool)
            self.use_frame[key]["range"] = include_frames

        # apply filters
        for filter_key, value in dfilter.items():
            if filter_key == "range":
                # filter data values by range
                include_frames = self.in_range(transformed_data, value)
            elif filter_key == "parts":
                include_frames = self.by_parts(transformed_data, value)
            else:
                do_critical(
                    NotImplementedError,
                    f"Data filter {filter_key} not implemented. Please implement it in the {self.__class__} class.",
                )
            self.use_frame[key][filter_key] = include_frames

        # TODO: remove this repeated computation to reduce overhead
        global_filtered_frames = self.update_global_use_frames()

        global_filtered_frames_cut, _ = force_equal_dimensions(
            global_filtered_frames, data
        )
        filtered_data = data[global_filtered_frames_cut]
        if sum(global_filtered_frames_cut) == len(data):
            global_logger.debug(
                f"No filtering applied for {key}. Returning original data."
            )
            return data
        else:
            global_logger.debug(
                f"Filtered data for {key} with {sum(global_filtered_frames_cut)} frames. Keeping {len(filtered_data)} frames."
            )
        return filtered_data

    def plot_used_frames(self):
        """
        Plot a heatmap showing the used frames for every entry in self.use_frame.

        Each row corresponds to a (name, filtertype) pair, columns are frames.
        """
        # Collect all (name, filtertype) pairs and their boolean arrays
        entries = []
        labels = []
        # gloabal first
        entries.append(np.array(self.use_frame.get("global"), dtype=bool))
        labels.append("global")
        for name, filter_dict in self.use_frame.items():
            if name == "global":
                continue
            for filtertype, frames in filter_dict.items():
                entries.append(np.array(frames, dtype=int))
                labels.append(f"{name}:{filtertype}")

        if not entries:
            print("No entries in use_frame to plot.")
            return

        # Pad arrays to the same length
        max_len = max(len(arr) for arr in entries)
        data = np.zeros((len(entries), max_len), dtype=int)
        for i, arr in enumerate(entries):
            data[i, : len(arr)] = arr

        plt.figure(figsize=(12, len(entries) * 0.5 + 2))
        plt.imshow(data, aspect="auto", cmap="Greys", interpolation="nearest")
        plt.yticks(np.arange(len(labels)), labels)
        plt.xlabel("Frame Index")
        plt.title("Used Frames per (name, filtertype) in DataFilter")
        # ticklabels notused = 0 and used = 1
        plt.colorbar(ticks=[0, 1]).ax.set_yticklabels(["Not Used", "Used"])
        plt.tight_layout()
        plt.show()


def logical_and_extend(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Perform logical AND operation on two boolean arrays, ensuring they are of the same length.

    If the arrays are of different lengths, the shorter one is padded with False values.
    Parameters:
    - arr1: First boolean array.
    - arr2: Second boolean array.

    Returns:
    - np.ndarray: Resulting boolean array after logical AND operation.
    """
    if arr1 is None or arr2 is None:
        global_logger.debug("One of the arrays is None. Cannot perform logical AND.")
        return None
    len_diff = len(arr1) - len(arr2)
    if len_diff > 0:
        # if global include frames is longer, extend arr2
        arr2 = np.pad(
            arr2,
            (0, len_diff),
            mode="constant",
            constant_values=False,
        )
    else:
        # if global include frames is shorter, extend arr1
        arr1 = np.pad(
            arr1,
            (0, -len_diff),
            mode="constant",
            constant_values=False,
        )

    # combine include frames from all datasets
    arr1 = arr1 & arr2
    return np.logical_and(arr1, arr2)


def list_vars_in_list(list1, list2):
    """
    Check if all variables in list1 are in list2.
    """
    return all(var in list2 for var in list1)


def extend_vstack(
    matrix: Union[List, np.ndarray], vector: Union[List, np.ndarray], fill_value=np.nan
) -> np.ndarray:
    """
    Extends a 2D matrix by stacking a vector vertically.

    If the matrix and vector have different numbers of columns, it extends the shorter one with `fill_value` (default is `np.nan`).
    If the matrix is 1D, it reshapes it to 2D before stacking else it stacks the vector as a new row.

    Parameters:
    - matrix: The matrix to extend.
    - vector: The vector to stack.

    Returns:
    - The extended matrix.
    """
    matrix = np.array(matrix)
    vector = np.array(vector)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if vector.ndim < 2:
        vector = np.array(vector).reshape(1, -1)

    vector_length = vector.shape[1]
    matrix_length = matrix.shape[1]

    if matrix_length != vector_length:
        # extend matrix or vector with nan values
        diff = matrix_length - vector_length
        if diff < 0:
            matrix = np.hstack(
                (
                    matrix,
                    np.full((matrix.shape[0], -diff), fill_value),
                )
            )
        else:
            vector = np.hstack(
                (
                    vector,
                    np.full((vector.shape[0], diff), fill_value),
                )
            )

    return np.vstack((matrix, vector))


def encode_categorical(data, categories=None, category_map=None):
    """
    Encode categorical data into numerical values.

    Parameters:
    -----------
    data : array-like
        The input data to encode. If the data is 1D, it will be treated as a single category. Data should bin binned.
    categories : list, optional, default=None
        The list of categories to encode. If None, the unique values in the data will be used.

    Returns:
    --------
    encoded_data : array-like
        The encoded data.
    """
    if len(data.shape) == 1:
        # check if data is binned data by checking values to be integers
        if is_integer(data):
            global_logger.warning("Data is 1D, treating labels as categories.")
            encoded_data = data
            category_map = {category: category for category in data}
        else:
            do_critical(
                ValueError,
                "Data provided for creating categories is 1D but not binned data",
            )
    if category_map is None:
        if categories is None:
            categories = np.unique(data, axis=0)
            category_map = {tuple(category): i for i, category in enumerate(categories)}

    category_type = type(list(category_map.keys())[0])
    if category_type is tuple:
        encoded_data = np.array([category_map[tuple(category)] for category in data])

    return encoded_data, category_map


def is_rgba(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        return all(is_single_rgba(v) for v in value)
    else:
        return is_single_rgba(value)


def is_single_rgba(val):
    if isinstance(val, tuple) or isinstance(val, list) or isinstance(val, np.ndarray):
        if len(val) == 4:
            return all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in val)
    return False


def values_to_groups(
    values, points, filter_outliers=True, contamination=0.2, parallel=True
):
    """
    Group points based on corresponding values, with optional outlier filtering.

    Parameters:
    -----------
    values : array-like
        An array of values used for grouping the points. Each value corresponds to a point.

    points : array-like
        An array of points to be grouped. Each point corresponds to a value.

    filter_outliers : bool, optional, default=True
        If True, outliers in the points will be filtered out before grouping.
        Outliers are determined using the `filter_outlier` function.

    Returns:
    --------
    groups : dict
        A dictionary where keys are unique values from `values` and values are arrays of points
        corresponding to each unique value.
    """
    if len(values) != len(points):
        raise ValueError("Values and points must have the same length.")

    if filter_outliers:
        filter_mask = filter_outlier(
            points, contamination=contamination, parallel=parallel, only_mask=True
        )
    else:
        filter_mask = None
    if filter_mask is not None:
        filtered_points = points[filter_mask]
        filtered_values = values[filter_mask]
    else:
        filtered_points = points
        filtered_values = values

    groups = {}
    unique_labels = np.unique(filtered_values, axis=0)
    for label in unique_labels:
        matching_values = np.all(filtered_values == label, axis=1)
        groups[tuple(label)] = filtered_points[matching_values]

    return groups


def is_integer(array: np.ndarray) -> bool:
    """Check if a NumPy array has an integer data type.
        array (np.ndarray): Input array.

    Returns:
        bool: True if the array's data type is an integer type, False otherwise.
    """
    return np.issubdtype(array.dtype, np.integer)


def is_floating(array: np.ndarray) -> bool:
    """Check if a NumPy array has a floating-point data type.

    Args:
        array (np.ndarray): Input array.

    Returns:
        bool: True if the array's data type is a floating-point type, False otherwise.
    """
    return np.issubdtype(array.dtype, np.floating)


def is_int_like(variable) -> bool:
    """Check if a variable is an integer-like type.

    Args:
        variable: The input variable to check.

    Returns:
        bool: True if the variable is an instance of int or a NumPy integer type, False otherwise.
    """
    return isinstance(variable, (int, np.integer))


def is_float_like(variable) -> bool:
    """Check if a variable is a floating-point-like type.

    Args:
        variable: The input variable to check.

    Returns:
        bool: True if the variable is an instance of float or a NumPy floating-point type, False otherwise.
    """
    return isinstance(variable, (float, np.floating))


def is_array_like(variable) -> bool:
    """Check if a variable is an array-like structure.

    Args:
        variable: The input variable to check.

    Returns:
        bool: True if the variable is a list, tuple, or NumPy ndarray, False otherwise.
    """
    return isinstance(variable, (list, tuple, np.ndarray))


def is_list_of_ndarrays(variable) -> bool:
    """Check if a variable is a list (or tuple) of NumPy ndarrays.

    Args:
        variable: The input variable to check.

    Returns:
        bool: True if the variable is a list or tuple where every element is an instance of np.ndarray, False otherwise.
    """
    return isinstance(variable, (list, tuple)) and all(
        isinstance(element, np.ndarray) for element in variable
    )


def force_1_dim_larger(data: np.ndarray):
    if data is None:
        return None
    if len(data.shape) == 1 or data.shape[0] < data.shape[1]:
        # global_logger.warning(
        #    f"Data is probably transposed. Needed Shape [Time, cells] Transposing..."
        # )
        # print("Data is probably transposed. Needed Shape [Time, cells] Transposing...")
        return data.T  # Transpose the array if the condition is met
    else:
        return data  # Return the original array if the condition is not met


def force_equal_dimensions(array1: np.ndarray, array2: np.ndarray):
    """
    Force two arrays to have the same dimensions.
    By cropping the larger array to the size of the smaller array.
    """
    if is_list_of_ndarrays(array1) and is_list_of_ndarrays(array2):
        for i in range(len(array1)):
            array1[i], array2[i] = force_equal_dimensions(array1[i], array2[i])
        return array1, array2
    elif isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        shape_0_diff = array1.shape[0] - array2.shape[0]
        if shape_0_diff != 0:
            global_logger.warning(
                f"Array 1 and 2 have different number of rows. Cropping larger array to smaller one."
            )
            if shape_0_diff > 0:
                array1 = array1[:-shape_0_diff]
            elif shape_0_diff < 0:
                array2 = array2[:shape_0_diff]
    return array1, array2


def sort_arr_by(arr, axis=1, sorting_indices=None):
    """
    if no sorting indices are given array is sorted by maximum value of 2d array
    """
    if sorting_indices is not None:
        indices = sorting_indices
    else:
        maxes = np.argmax(arr, axis=axis)
        indices = np.argsort(maxes)
    sorted_arr = arr[indices]
    return sorted_arr, indices


def split_array_by_zscore(array, zscore, threshold=2.5):
    above_threshold = np.where(zscore >= threshold)[0]
    below_threshold = np.where(zscore < threshold)[0]
    return array[above_threshold], array[below_threshold]


def find_min_max_values(
    data: Union[List, np.ndarray, List[np.ndarray]], axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the minimum and maximum values along a specified axis for a list of arrays or a single array.

    Parameters:
        data (Union[List[np.ndarray], np.ndarray]): Input data, which can be a single NumPy array or a list of NumPy arrays.
        axis (int): The axis along which to compute the min and max values. Default is 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - min_val_labels: Array of minimum values along the specified axis.
            - max_val_labels: Array of maximum values along the specified axis.

    Raises:
        ValueError: If the input data is empty or if the specified axis is invalid for the input data.
    """
    if not data:
        raise ValueError("Input data cannot be empty.")

    data = make_list_ifnot(data)
    # Check if the axis is valid for the first array in the data
    if axis >= data[0].ndim:
        global_logger.error(
            f"Axis {axis} is invalid for input data with {data[0].ndim} dimensions."
        )
        raise ValueError(
            f"Axis {axis} is invalid for input data with {data[0].ndim} dimensions."
        )
    # Determine the shape of the output arrays based on the specified axis
    output_shape = list(data[0].shape)
    del output_shape[axis]  # Remove the axis dimension

    # Initialize min and max values with infinity and negative infinity
    min_val_labels = np.full(output_shape, np.inf)
    max_val_labels = np.full(output_shape, -np.inf)

    # Iterate over each model's labels
    for model_labels in data:
        # Calculate the min and max along the columns (axis=0)
        current_min = np.min(model_labels, axis=axis)
        current_max = np.max(model_labels, axis=axis)

        # Update min_val_labels and max_val_labels element-wise
        min_val_labels = np.minimum(min_val_labels, current_min)
        max_val_labels = np.maximum(max_val_labels, current_max)

    return min_val_labels, max_val_labels


def npio(
    path: Union[str, Path],
    task: str,
    data: Union[np.ndarray, Dict[str, Any]] = None,
    file_type: str = "auto",
    allow_overwrite: bool = True,
):
    """
    Manages numpy file operations for both .npy and .npz files.

    Parameters
    ----------
    path : str or Path
        The full path for the numpy file (including extension if needed).
        If the directories in the path don't exist, they will be created.
    task : str
        Specifies the operation to perform:
        - "save": Saves the provided data
        - "exists": Checks if a file exists
        - "load": Loads the file from the specified path
        - "update": Updates an existing .npz file with new data (merges with existing data)
    data : np.ndarray or Dict[str, Any], optional
        For .npy: The numpy array to be saved.
        For .npz: Dictionary of arrays to be saved, with keys used as variable names.
        Required when task is "save" or "update", otherwise ignored.
    file_type : str, default="auto"
        Specifies the file type to use:
        - "auto": Determine from path extension or data type
        - "npy": Use .npy format (single array)
        - "npz": Use .npz format (multiple arrays)
    allow_overwrite : bool, default=True
        When task is "save", determines if existing file should be overwritten.

    Returns
    -------
    bool, np.ndarray, Dict[str, np.ndarray], or None
        - When task is "exists": Returns True if the file exists, False otherwise
        - When task is "save" or "update": Returns None after saving the file
        - When task is "load": Returns loaded data, or None if file doesn't exist
          (.npy returns array or dict, .npz returns dict of arrays)

    Raises
    ------
    ValueError
        - If task is "save" or "update" and no data is provided
        - If task is not one of the recognized operations
        - If file_type is invalid
        - If task is "update" but file_type is "npy"
        - If task is "save", allow_overwrite is False, and file already exists

    Examples
    --------
    # Working with .npy files:
    array_data = np.random.rand(10, 10)
    numpy_io("/path/to/data.npy", task="save", data=array_data)
    loaded_array = numpy_io("/path/to/data.npy", task="load")

    # Working with .npz files:
    dict_data = {'arr1': np.random.rand(10), 'arr2': np.zeros(5)}
    numpy_io("/path/to/data.npz", task="save", data=dict_data)
    # Update existing .npz file
    numpy_io("/path/to/data.npz", task="update", data={'arr3': np.ones(3)})
    # Load all arrays
    loaded_dict = numpy_io("/path/to/data.npz", task="load")
    """
    path = Path(path)

    # Determine file type
    if file_type == "auto":
        if path.suffix == ".npy":
            file_type = "npy"
        elif path.suffix == ".npz":
            file_type = "npz"
        elif isinstance(data, dict):
            file_type = "npz"
            path = path.with_suffix(".npz")
        else:
            file_type = "npy"
            path = path.with_suffix(".npy")
    elif file_type == "npy":
        path = path.with_suffix(".npy")
    elif file_type == "npz":
        path = path.with_suffix(".npz")
    else:
        raise ValueError(
            f"Invalid file_type: {file_type}. Must be 'auto', 'npy', or 'npz'."
        )

    # Ensure directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle "exists" task (same for both file types)
    if task == "exists":
        return path.exists()

    # Handle update task (npz only)
    elif task == "update":
        if file_type == "npy":
            raise ValueError("Update operation is only supported for .npz files.")

        if data is None:
            do_critical(ValueError, f"Data must be provided to save in {path}.")

        # Load existing data if available
        existing_data = {}
        if path.exists():
            try:
                with np.load(path, allow_pickle=True) as loaded:
                    existing_data = {key: loaded[key] for key in loaded.files}
            except Exception as e:
                global_logger.warning(f"Failed to load existing file for update: {e}")

        # Merge with new data (new data takes precedence)
        combined_data = {**existing_data, **data}

        # Save combined data
        np.savez(path, **combined_data)
        global_logger.info(f"Updated {path}")

    # Handle save task
    elif task == "save":
        if data is None:
            do_critical(ValueError, f"Data must be provided to save in {path}.")

        if path.exists() and not allow_overwrite:
            global_logger.warning(
                f"File {path} already exists and allow_overwrite is False."
            )
            raise ValueError(
                f"File {path} already exists and allow_overwrite is False."
            )

        if file_type == "npy":
            np_data = np.array(data)
            np.save(path, np_data)
        else:  # npz
            if not isinstance(data, dict):
                # If data is not a dict, wrap it in a dict with a default key
                data = {"data": data}
            np.savez(path, **data)

        global_logger.info(f"Saved {path}")

    # Handle load task
    elif task == "load":
        if path.exists():
            try:
                if file_type == "npy":
                    try:
                        data = np.load(path, allow_pickle=True).item()
                    except:
                        data = np.load(path, allow_pickle=True)
                else:  # npz
                    with np.load(path, allow_pickle=True) as loaded:
                        data = {
                            key: (
                                loaded[key].item()
                                if loaded[key].ndim == 0
                                else loaded[key]
                            )
                            for key in loaded.files
                        }

                global_logger.info(f"Loaded {path}")
                return data
            except Exception as e:
                global_logger.error(f"Error loading {path}: {e}")
                return None
        else:
            global_logger.warning(f"LOADING failed: No file found at {path}")
            return None

    # Handle unrecognized task
    else:
        global_logger.critical(f"Task {task} not recognized.")
        raise ValueError(f"Task {task} not recognized.")


def h5io(
    path: Union[str, Path],
    task: Literal["save", "load"],
    data: Union[pd.DataFrame, np.ndarray, None] = None,
    labels: Optional[List[str]] = None,
    item_pairs: Optional[List[Tuple[str, str]]] = None,
    data_key: str = "comparisons",
    labels_key: str = "labels",
    metadata_key: str = "metadata",
    retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[Optional[Union[pd.DataFrame, np.ndarray]], Optional[List[str]]]:
    """
    Load or save data to/from an HDF5 file, with robust handling of DataFrames, ndarrays, and labels.

    Parameters:
    - path (Union[str, Path]): Path to the HDF5 file.
    - task (Literal["save", "load"]): Task to perform ("save" or "load").
    - data (Union[pd.DataFrame, np.ndarray, None]): Data to save (for "save" task).
    - labels (Optional[List[str]]): Labels for validation or metadata (not used for DataFrame filtering).
    - item_pairs (Optional[List[Tuple[str, str]]]): List of (item_i, item_j) pairs to filter DataFrame in load task.
    - data_key (str): HDF5 key for the main data (default: "comparisons").
    - labels_key (str): HDF5 key for labels (default: "labels").
    - metadata_key (str): HDF5 key for metadata (default: "metadata").
    - retries (int): Number of retries for file operations (default: 3).
    - retry_delay (float): Delay between retries in seconds (default: 1.0).

    Returns:
    - Optional[Union[pd.DataFrame, np.ndarray]]: Loaded data (for "load") or None (for "save").
    - Optional[List[str]]: Loaded labels or None.
    """
    path = Path(path).with_suffix(".h5")

    if task == "save":
        if data is None:
            raise ValueError("Data must be provided for save task.")

        # Validate directory permissions
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True)
            except OSError as e:
                raise OSError(f"Cannot create directory {parent_dir}: {e}")
        if not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")

        # Use temporary file to avoid partial writes
        temp_fd, temp_path = tempfile.mkstemp(suffix=".h5", dir=parent_dir)
        os.close(temp_fd)
        temp_path = Path(temp_path)

        for attempt in range(retries):
            try:
                if isinstance(data, np.ndarray):
                    # Validate labels
                    if labels is not None:
                        if len(labels) != data.shape[0]:
                            raise ValueError(
                                f"Number of labels ({len(labels)}) does not match data rows ({data.shape[0]})."
                            )
                        if len(data.shape) == 2 and len(labels) != data.shape[1]:
                            raise ValueError(
                                f"Number of labels ({len(labels)}) does not match data columns ({data.shape[1]})."
                            )
                    else:
                        labels = [str(i) for i in range(data.shape[0])]
                        global_logger.warning(
                            "No labels provided; using default numeric labels."
                        )

                    # Save array and metadata with h5py
                    with h5py.File(temp_path, "w") as f:
                        f.create_dataset(
                            data_key, data=data, compression="gzip", compression_opts=9
                        )
                        f.attrs["data_type"] = "ndarray"
                        f.attrs["shape"] = data.shape
                        f.attrs["metadata"] = str(
                            {"data_type": "ndarray", "shape": data.shape}
                        )

                    # Save labels with pandas
                    if labels is not None:
                        pd.DataFrame({"label": labels}).to_hdf(
                            temp_path, key=labels_key, mode="a"
                        )
                else:
                    # Save DataFrame with pandas
                    if labels is not None:
                        if set(labels) - set(data.get("item_i", data.index)):
                            global_logger.warning(
                                "Some provided labels not found in DataFrame."
                            )
                    data.to_hdf(
                        temp_path, key=data_key, mode="w", complevel=9, complib="zlib"
                    )
                    # Save labels and metadata
                    if labels is not None:
                        pd.DataFrame({"label": labels}).to_hdf(
                            temp_path, key=labels_key, mode="a"
                        )
                    # Add metadata using h5py
                    with h5py.File(temp_path, "a") as f:
                        f.attrs["data_type"] = "DataFrame"
                        f.attrs["metadata"] = str(
                            {"data_type": "DataFrame", "shape": data.shape[0]}
                        )

                # Move temporary file to final path
                shutil.move(temp_path, path)
                global_logger.info(f"Saved data to {path}")
                return None, None

            except (h5py.H5ExtError, OSError, PermissionError) as e:
                global_logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed to save {path}: {e}"
                )
                if attempt == retries - 1:
                    temp_path.unlink(missing_ok=True)  # Clean up temporary file
                    raise RuntimeError(
                        f"Failed to save HDF5 file {path} after {retries} attempts: {e}. "
                        "Ensure no other process is locking the file and check write permissions."
                    )
                time.sleep(retry_delay)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # Ensure cleanup

    elif task == "load":
        try:
            if not path.exists():
                global_logger.info(f"HDF5 file {path} does not exist.")
                return None, []

            with h5py.File(path, "r") as f:
                # Check metadata
                data_type = f.attrs.get("data_type", "DataFrame")

                if data_type == "ndarray":
                    # Load array
                    if data_key not in f:
                        global_logger.error(f"Key '{data_key}' not found in {path}.")
                        return None, []
                    data = np.array(f[data_key])
                    # Load labels
                    saved_labels = (
                        pd.read_hdf(path, key=labels_key)["label"].tolist()
                        if labels_key in f
                        else []
                    )
                    # Validate labels if provided
                    if labels is not None and saved_labels:
                        missing_labels = set(labels) - set(saved_labels)
                        if missing_labels:
                            global_logger.warning(
                                f"Labels not found in saved data: {missing_labels}"
                            )
                    return data, saved_labels
                else:
                    # Load DataFrame
                    df = pd.read_hdf(path, key=data_key)
                    # Remove duplicates for long-format DataFrames
                    if "item_i" in df.columns and "item_j" in df.columns:
                        df = df.drop_duplicates(
                            subset=["item_i", "item_j"], keep="first"
                        )
                    # Load labels
                    saved_labels = (
                        pd.read_hdf(path, key=labels_key)["label"].tolist()
                        if labels_key in f
                        else []
                    )
                    # Validate labels if provided
                    if labels is not None and saved_labels:
                        missing_labels = set(labels) - set(saved_labels)
                        if missing_labels:
                            global_logger.warning(
                                f"Labels not found in saved data: {missing_labels}"
                            )
                    # Filter by item_pairs if provided
                    if item_pairs is not None:
                        if "item_i" not in df.columns or "item_j" not in df.columns:
                            global_logger.error(
                                "DataFrame missing 'item_i' or 'item_j' columns for filtering."
                            )
                            return None, saved_labels
                        pair_df = pd.DataFrame(item_pairs, columns=["item_i", "item_j"])
                        df = df.merge(pair_df, on=["item_i", "item_j"], how="inner")
                        if df.empty:
                            global_logger.info(
                                f"No matching item pairs found in {path}."
                            )
                            return None, saved_labels
                    elif item_pairs is None and "item_i" in df.columns:
                        global_logger.warning(
                            "No item_pairs provided; returning all data."
                        )
                    return df, saved_labels

        except (KeyError, FileNotFoundError, ValueError) as e:
            global_logger.error(f"Error loading HDF5 file {path}: {e}")
            return None, []

    else:
        raise ValueError("Task must be 'save' or 'load'.")


def bin_array_1d(
    arr: List[float],
    bin_size: float,
    min_bin: float = None,
    max_bin: float = None,
    restrict_to_range=True,
):
    """
    Bin a 1D array of floats based on a given bin size.

    Parameters:
    arr (numpy.ndarray): Input array of floats.
    bin_size (float): Size of each bin.
    min_bin (float, optional): Minimum bin value. If None, use min value of the array.
    max_bin (float, optional): Maximum bin value. If None, use max value of the array.

    Returns:
    numpy.ndarray: Binned array, starting at bin 0 to n-1.
    """
    min_bin = min_bin or np.min(arr)
    max_bin = max_bin or np.max(arr)

    # Calculate the number of bins
    num_bins = int(np.ceil((max_bin - min_bin) / bin_size))

    # Calculate the edges of the bins
    bin_edges = np.linspace(min_bin, min_bin + num_bins * bin_size, num_bins + 1)

    # Bin the array
    binned_array = np.digitize(arr, bin_edges) - 1

    if restrict_to_range:
        binned_array = np.clip(binned_array, 0, num_bins - 1)

    return binned_array


def bin_array(
    arr: List[float],
    bin_size: List[float],
    min_bin: List[float] = None,
    max_bin: List[float] = None,
):
    """
    Bin an array of floats based on a given bin size.

    Parameters:
    arr (numpy.ndarray): Input array of floats. Can be 1D or 2D.
    bin_size (float): Size of each bin.
    min_bin (float, optional): Minimum bin value. If None, use min value of the array.
    max_bin (float, optional): Maximum bin value. If None, use max value of the array.

    Returns:
    numpy.ndarray: Binned array, starting at bin 0 to n-1.
    """
    min_bin = make_list_ifnot(min_bin if min_bin is not None else np.min(arr, axis=0))
    max_bin = make_list_ifnot(max_bin if max_bin is not None else np.max(arr, axis=0))
    bin_size = make_list_ifnot(bin_size)
    np_arr = np.array(arr)
    if np_arr.ndim == 1:
        np_arr = np_arr.reshape(-1, 1)
    else:
        num_dims = min(arr.shape)
        if len(min_bin) == 1:
            min_bin = [min_bin[0]] * num_dims
        if len(max_bin) == 1:
            max_bin = [max_bin[0]] * num_dims
        if len(bin_size) == 1:
            bin_size = [bin_size[0]] * num_dims

    # Transform the array if 1st dimension is smaller than 2nd dimension
    if np_arr.shape[0] < np_arr.shape[1]:
        np_arr = np_arr.T

    # Bin each dimension of the array
    binned_array = np.zeros_like(np_arr, dtype=int)
    for i in range(np_arr.shape[1]):
        binned_array[:, i] = bin_array_1d(
            arr=np_arr[:, i],
            bin_size=bin_size[i],
            min_bin=min_bin[i],
            max_bin=max_bin[i],
        )

    return binned_array


def get_frame_count(duration: Union[int, float], fps: Union[int, float]):
    """
    Calculate the number of frames in a given duration at a specific frame rate.

    Parameters:
    duration (float): Duration in seconds.
    fps (int): Frame rate in frames per second.

    Returns:
    int: Number of frames in the given duration at the specified frame rate.
    """
    frame_count = duration * fps
    if frame_count % 1 != 0:
        global_logger.warning(
            f"Frame count is not an integer. Rounding to {int(frame_count)}"
        )
    return int(frame_count)


def fill_continuous_array(
    data_array: np.ndarray, fps: Union[int, float], time_gap: Union[int, float]
):
    """
    Fills gaps in a continuous array where values remain the same for a specified time gap.

    Parameters:
    data_array (numpy.ndarray): The input array containing continuous data.
    fps (int): Frames per second, used to calculate the frame gap.
    time_gap (float): The time gap in seconds to consider for filling gaps.

    Returns:
    numpy.ndarray: The array with gaps filled where values remain the same for the specified time gap.

    Description:
    This function identifies indices in the `data_array` where the values change. It then fills the gaps
    between these indices if the gap is less than or equal to the specified `time_gap` (converted to frames
    using `fps`) and the values before and after the gap are the same. This is useful for ensuring continuity
    in data where small gaps may exist due to missing or noisy data points.

    Example:
    >>> data_array = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1])
    >>> fps = 30
    >>> time_gap = 2
    >>> fill_continuous_array(data_array, fps, time_gap)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    """
    frame_gap = get_frame_count(time_gap, fps)
    # Find indices where values change
    value_changes = np.where(np.abs(np.diff(data_array)) > 0)[0]
    filled_array = data_array.copy()

    # Fill gaps after continuous time_gap seconds of the same value
    before_frame = 0
    for current_frame in value_changes:
        diff = current_frame - before_frame
        if diff <= frame_gap:
            fill_value = filled_array[before_frame - 1]
            filled_array[before_frame:current_frame] = fill_value
        before_frame = current_frame
    return filled_array


def add_stream_lag(
    array: np.ndarray,
    min_stream_duration: Union[int, float],
    fps: Union[int, float],
    lag: Union[int, float],
):
    """
    Add a lag to the end of continuous streams in an array.

    Parameters:
    ----------
        array (numpy.ndarray): The input array containing continuous data.
        min_stream_duration (float): The minimum duration of a continuous stream in seconds.
        fps (int): Frames per second, used to calculate the frame count.
        lag (float): The lag duration in seconds to add to the end of each stream.

    Returns:
    -------
        numpy.ndarray: The array with lag added to the end of continuous streams.
    """
    min_stream_frames = get_frame_count(min_stream_duration, fps)
    lag_frames = get_frame_count(lag, fps)

    value_changes = np.where(np.abs(np.diff(array)) > np.finfo(float).eps)[0] + 1
    manipulated_array = array.copy()
    for i in reversed(range(len(value_changes) - 1)):
        start = value_changes[i]
        end = value_changes[i + 1]
        if end - start > min_stream_frames:
            fill_value = array[start]
            manipulated_array[start : end + lag_frames] = fill_value

    if value_changes[0] > min_stream_frames:
        fill_value = array[0]
        manipulated_array[: value_changes[0] + lag_frames] = fill_value

    return manipulated_array


def convert_values_to_binary(vec: np.ndarray, threshold=2.5):
    smaller_idx = np.where(vec < threshold)
    bigger_idx = np.where(vec > threshold)
    vec[smaller_idx] = 0
    vec[bigger_idx] = 1
    vec = vec.astype(int)
    return vec


def per_frame_to_per_second(data, fps=None):
    if not fps:
        data = data
        global_logger.debug("No fps provided. Output is value per frame")
    else:
        data *= fps
    return data


def fill_vector(vector, indices, fill_value=np.nan):
    filled_vector = vector.copy()
    filled_vector[indices] = fill_value
    return filled_vector


def fill_matrix(matrix, indices, fill_value=np.nan, axis=0):
    filled_matrix = matrix.copy()
    if axis == 0:
        filled_matrix[indices, :] = fill_value
    elif axis == 1:
        filled_matrix[:, indices] = fill_value
    else:
        global_logger.critical("Axis must be 0 or 1. 3D not supported.")
        raise ValueError("Axis must be 0 or 1. 3D not supported.")
    return filled_matrix


def fill(matrix, indices, fill_value=np.nan):
    if len(matrix.shape) == 1:
        return fill_vector(matrix, indices, fill_value)
    else:
        return fill_matrix(matrix, indices, fill_value)


def fill_inputs(inputs: dict, indices: np.ndarray, fill_value=np.nan):
    filtered_inputs = inputs.copy()
    for key, value in filtered_inputs.items():
        if len(value.shape) == 1:
            filtered_inputs[key] = fill_vector(value, indices, fill_value=fill_value)
        else:
            filtered_inputs[key] = fill_matrix(value, indices, fill_value=fill_value)
    return filtered_inputs


def get_top_percentile_indices(vector, percentile=5, indices_smaller=True):
    cutoff = np.percentile(vector, 100 - percentile)
    if indices_smaller:
        indices = np.where(vector < cutoff)[0]
    else:
        indices = np.where(vector > cutoff)[0]
    return indices


# dict
def init_dict_in_dict(dict, key):
    if key not in dict:
        dict[key] = {}
    return dict[key]


def add_to_list_in_dict(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


def dict_value_keylist(dict, keylist):
    for key in keylist:
        dict = dict[key]
    return dict


def check_extract_from_parameter_sweep(
    analysis_values: Dict[str, Any], value_key: str
) -> Union[np.ndarray, List[np.ndarray]]:
    if value_key not in analysis_values and isinstance(analysis_values, dict):
        # check if parameter_sweep is used
        parameter_sweep = True
        for key in analysis_values.keys():
            if key == value_key:
                parameter_sweep = False
                break

        # extract the values from the model statistics
        if parameter_sweep:
            values = [
                one_out_of_sweep[value_key]
                for one_out_of_sweep in analysis_values.values()
            ]
            sweep_values = list(analysis_values.keys())
        elif value_key in analysis_values:
            values = analysis_values[value_key]
            sweep_values = None
        else:
            do_critical(
                ValueError,
                f"Key {value_key} not found in model analysis statistics.",
            )
        return sweep_values, values
    else:
        raise NotImplementedError(
            f"Analysis values {analysis_values} do not contain the key {value_key}."
        )


def create_params_dict(exclude=[], **kwargs):
    return {k: v for k, v in kwargs.items() if k not in exclude}


def create_unique_dict_key(dictionary, key):
    """
    Create a unique key for a dictionary by appending a number to the key.

    Parameters:
    dictionary (dict): The dictionary to check for existing keys.
    key (str): The key to check for uniqueness.

    Returns:
    str: A unique key based on the input key.
    """
    unique_key = key
    count = 1
    while unique_key in dictionary:
        unique_key = f"{key}_{count}"
        count += 1
    return unique_key


def is_dict_of_dicts(dict):
    return all(type(value) == type(dict) for value in dict.values())


def equal_number_entries(
    dict1: Union[Dict[str, np.ndarray], np.ndarray],
    dict2: Dict[str, np.ndarray],
) -> bool:
    """
    Compare the number of entries in two two dictionaries or arrays.
    If dict1 is a numpy array, it will be treated as a dictionary with a single key.
    If dict2 is a dictionary, it will be treated as a dictionary with multiple keys.

    Parameters:
    ----------
        dict1 : Union[Dict[str, np.ndarray], np.ndarray]
            The first dictionary or numpy array to compare.
        dict2 : Dict[str, np.ndarray]
            The second dictionary to compare.
    Returns:
    -------
        bool
            True if the number of entries in both dictionaries or arrays is equal, False otherwise.
    """

    for key, labels in dict2.items():
        if isinstance(dict1, np.ndarray):
            dict1 = {"dummy_name": dict1}
        for key2, emb in dict1.items():
            if labels.shape[0] != emb.shape[0]:
                return False
    return True


def add_descriptive_metadata(text, metadata=None, keys=None, comment=None):
    if isinstance(metadata, dict) and isinstance(keys, list):
        text += get_str_from_dict(
            dictionary=metadata,
            keys=keys,
        )
    text += f"{' '+str(comment) if comment else ''}"
    return text


def equal_dicts(dict1, dict2, unimportant_keys=None):
    """
    Check if two dictionaries are equal.
    """
    equal = True
    for mkey, mvalue in dict1.items():
        if unimportant_keys and mkey in unimportant_keys:
            continue
        for fmkey, fmvalue in dict2.items():
            if mkey == fmkey and mvalue != fmvalue:
                global_logger.error(
                    f"Parameter {mkey} with value {mvalue} does not match to {fmvalue}"
                )
                equal = False
    return equal


def group_by_binned_data(
    binned_data: np.ndarray,
    category_map: dict = None,
    data: np.ndarray = None,
    as_array: bool = False,
    group_values: str = "raw",
    max_bin=None,
):
    """
    Group data based on binned data.
    If category_map is provided, the binned data will be used as keys to group the data. Mostly used to map 1D discrete labels to multi dimensional position data.

    Parameters:
    ----------
        - binned_data: The binned data used for grouping.
        - category_map: A dictionary mapping binned data to categories.
        - group_by: The method used for grouping the data.
            - 'symmetric_matrix' is in the group by options. It is filtering in 2D dimensions
            - 'raw': Return the raw data.
            - 'mean': Return the mean of the grouped data.
            - 'count': Return the count of the grouped data.
    """
    if data is None and group_values != "count":
        global_logger.critical("Data needed for grouping.")
        raise ValueError("Data needed for grouping.")
    if binned_data is None or len(binned_data) == 0:
        global_logger.critical("No binned data provided.")
        raise ValueError("No binned data provided.")

    if group_values != "count":
        data, binned_data = force_equal_dimensions(data, binned_data)
    # Define bins and counts
    if category_map is None:
        bins = unique_locations = np.unique(binned_data, axis=0, return_counts=False)
        uncoded_binned_data = binned_data
    else:
        unique_locations = list(category_map.keys())
        bins = list(category_map.keys())
        uncoded_binned_data = uncode_categories(binned_data, category_map)

    max_bin = np.max(bins, axis=0) if max_bin is None else max_bin

    # create groups
    groups = {}
    for unique_loc in unique_locations:
        idx = np.all(
            force_1_dim_larger(np.atleast_2d(uncoded_binned_data == unique_loc)), axis=1
        )
        # calculation is for cells x cells matrix if symmetric_matrix
        if group_values == "count":
            values_at_bin = np.sum(idx)
        elif group_values == "count_norm":
            values_at_bin = np.sum(idx) / len(uncoded_binned_data)
        else:
            filtered_data = (
                data[idx][:, idx] if "symmetric_matrix" in group_values else data[idx]
            )
            if "raw" in group_values:
                values_at_bin = filtered_data
            elif "mean" in group_values:
                values_at_bin = np.mean(filtered_data)
                # np.mean(distances[np.triu_indices(distances.shape[0], k=1)])
                raise NotImplementedError(
                    f"Check if mean calculation is correct. Maybe the new version commented is needed since only a triangle should be used for mean calculation."
                )

        groups[unique_loc] = values_at_bin

    if as_array:
        groups_array = np.zeros(max_bin.astype(int))
        for coordinates, values in groups.items():
            groups_array[coordinates] = values
        return groups_array, bins
    return groups, bins


def uncode_categories(data: Union[dict, np.ndarray, list], category_map: dict):
    """
    Converts a dictionary with categorical keys to a dictionary with tuple keys.
    """
    rev_category_map = {v: k for k, v in category_map.items()}
    if isinstance(data, dict):
        uncoded = {rev_category_map[key]: value for key, value in data.items()}
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        uncoded = np.array(
            [rev_category_map[encoded_category] for encoded_category in data.flatten()]
        )
    return uncoded


def filter_dict_by_properties(
    dictionary: Dict[str, Any],
    include_properties: List[List[str]] = None,  # or [str] or str
    exclude_properties: List[List[str]] = None,  # or [str] or str):
) -> Dict[str, Any]:
    """
    Filter a dictionary with descriptive keys based on given properties.

    Parameters:
    ----------
    - dictionary (dict): The dictionary to filter.
    - include_properties: Union[List[List[str]], List[str], str] = None,
        Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
        1. Include all models containing a specific string: "string"
        2. Include all models containing a specific combination of strings: ["string1", "string2"]
        3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
    - exclude_properties: Union[List[List[str]], List[str], str] = None,
        Same as model_naming_filter_include but for excluding models.

    Returns:
    -------
    - filtered_dict (dict): The filtered dictionary containing only keys that match the include/exclude criteria.
    """
    dict_keys = dictionary.keys()
    if include_properties or exclude_properties:
        dict_keys = filter_strings_by_properties(
            dict_keys,
            include_properties=include_properties,
            exclude_properties=exclude_properties,
        )
    filtered_dict = {key: dictionary[key] for key in dict_keys}
    return filtered_dict


def wanted_object(obj, wanted_keys_values):
    """
    Check if an object has the wanted keys and values.

    Parameters:
    ----------
    obj : object
        The object to check.
    wanted_keys_values : dict
        A dictionary of keys and values to check for in the object.

    Returns:
    -------
    bool : True if the object has the wanted keys and values, False otherwise.
    """
    if isinstance(obj, dict):
        dictionary = obj
    else:
        if hasattr(obj, "__dict__"):
            dictionary = obj.__dict__
        else:
            raise ValueError("Object has no dictionary")

    for key, value in wanted_keys_values.items():
        if key not in dictionary:
            return False
        else:
            obj_val = dictionary[key]
            if isinstance(value, list):
                if obj_val not in value:
                    return False
            elif obj_val != value:
                return False
    return True


def sort_dict(dictionary, reverse=False):
    keys = list(dictionary.keys())
    # create lexicographical order of the keys
    sorted_keys = sorted(
        keys,
        key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r"(\d+)", x)],
    )

    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict[key] = dictionary[key]
    return sorted_dict


def load_yaml(path, mode="r"):
    if not Path(path).exists():
        # check for .yml file
        if Path(path).with_suffix(".yml").exists():
            path = Path(path).with_suffix(".yml")
        else:
            raise FileNotFoundError(f"No yaml file found: {path}")
    with open(path, mode) as file:
        dictionary = yaml.safe_load(file)
    return dictionary


def load_yaml_data_into_class(
    cls,
    yaml_path: Union[str, Path] = None,
    name_parts: Union[str, List] = None,
    needed_attributes: List[str] = None,
):
    yaml_path = yaml_path or cls.yaml_path
    name_parts = name_parts or cls.id.split("_")[-1]
    string_to_check = yaml_path.stem if isinstance(yaml_path, Path) else yaml_path
    success = check_correct_metadata(
        string_or_list=string_to_check, name_parts=name_parts
    )

    if success:
        metadata_dict = load_yaml(yaml_path)
        # Load any additional metadata into session object
        set_attributes_check_presents(
            propertie_name_list=metadata_dict.keys(),
            set_object=cls,
            propertie_values=metadata_dict.values(),
            needed_attributes=needed_attributes,
        )
    else:
        raise ValueError(
            f"Animal {cls.id} does not correspond to metadata in {yaml_path}"
        )
    return metadata_dict


def get_str_from_dict(dictionary, keys):
    present_keys = dictionary.keys()
    string = ""
    for variable in keys:
        if variable in present_keys:
            string += f" {dictionary[variable]}"
    return string


def keys_missing(dictionary, keys):
    present_keys = dictionary.keys()
    missing = []
    keys = make_list_ifnot(keys)
    for key in keys:
        if key not in present_keys:
            missing.append(key)
    if len(missing) > 0:
        return missing
    return False


def check_needed_keys(metadata, needed_attributes):
    missing = keys_missing(metadata, needed_attributes)
    if missing:
        raise NameError(f"Missing metadata for: {missing} not defined")


def add_missing_keys(metadata, needed_attributes, fill_value=None):
    missing = keys_missing(metadata, needed_attributes)
    if missing:
        for key in missing:
            metadata[key] = fill_value
    return metadata


def traverse_dicts(dicts, keys=None, wanted_key=None):
    """
    Traverse a nested dictionary and yield each key-value pair with a key list.
    """
    for key, value in dicts.items():
        keys_list = keys + [key] if keys else [key]
        if wanted_key and key == wanted_key:
            yield keys_list, value
        else:
            if isinstance(value, dict):
                yield from traverse_dicts(value, keys=keys_list, wanted_key=wanted_key)
            else:
                yield keys_list, value


def delete_nested_key(d, keys):
    """
    Deletes a key from a nested dictionary.

    Parameters:
    d (dict): The dictionary to modify.
    keys (list): A list of keys to traverse the dictionary.

    Example:
    delete_nested_key(my_dict, ['a', 'b', 'c']) will delete my_dict['a']['b']['c']
    """
    for key in keys[:-1]:
        d = d[key]
    del d[keys[-1]]


# class
def check_obj_equivalence(obj1, obj2):
    """
    Check if two objects are equivalent.
    This function checks if two objects are equivalent by comparing their memory locations.
    If obj1 is None and obj2 is not None, it assigns obj1 to obj2.

    Parameters:
        - obj1: The first object to check.
        - obj2: The second object to check.

    Raises:
        - ValueError: If the objects are not equivalent.
    """
    if obj1 is None and obj2 is not None:
        obj1 = obj2
    elif obj1 is not obj2:
        do_critical(
            ValueError,
            f"Objects {obj1} and {obj2} are not equivalent. They are not pointing to the same memory location.",
        )


def safe_isinstance(obj, cls) -> bool:
    """
    Check if an object is an instance of a class, including subclasses even if reloaded or re-imported differently.
    """
    obj_cls = obj.__class__
    return obj_cls.__name__ == cls.__name__ and obj_cls.__module__ == cls.__module__


def define_cls_attributes(cls_object, attributes_dict, override=False):
    """
    Defines attributes for a class object based on a dictionary.

    Args:
        cls_object (object): The class object to define attributes for.
        attributes_dict (dict): A dictionary containing attribute names as keys and their corresponding values.
        override (bool, optional): If True, existing attributes will be overridden. Defaults to False.

    Returns:
        object: The modified class object with the defined attributes.
    """
    for key, value in attributes_dict.items():
        if key not in cls_object.__dict__.keys() or override:
            setattr(cls_object, key, value)
    return cls_object


def copy_attributes_to_object(
    propertie_name_list,
    set_object,
    get_object=None,
    propertie_values=None,
    override=True,
):
    """
    Set attributes of a target object based on a list of property names and values.

    This function allows you to set attributes on a target object (the 'set_object') based on a list of
    property names provided in 'propertie_name_list' and corresponding values. You can specify these
    values directly through 'propertie_values' or retrieve them from another object ('get_object').
    If 'propertie_values' is not provided, this function will attempt to fetch the values from the
    'get_object' using the specified property names.

    Args:
        propertie_name_list (list): A list of property names to set on the 'set_object.'
        set_object (object): The target object for attribute assignment.
        get_object (object, optional): The source object to retrieve property values from. Default is None.
        propertie_values (list, optional): A list of values corresponding to the property names.
            Default is None.

    Returns:
        None

    Raises:
        ValueError: If the number of properties in 'propertie_name_list' does not match the number of values
            provided in 'propertie_values' (if 'propertie_values' is specified).

    Example Usage:
        # Example 1: Set attributes directly with values
        copy_attributes_to_object(["attr1", "attr2"], my_object, propertie_values=[value1, value2])

        # Example 2: Retrieve attribute values from another object
        copy_attributes_to_object(["attr1", "attr2"], my_object, get_object=source_object)
    """
    propertie_name_list = list(propertie_name_list)
    if propertie_values:
        propertie_values = list(propertie_values)
        if len(propertie_values) != len(propertie_name_list):
            raise ValueError(
                f"Number of properties does not match given propertie values: {len(propertie_name_list)} != {len(propertie_values)}"
            )
    # If propertie_values is not provided, get values from get_object
    elif get_object:
        propertie_values = []
        for propertie in propertie_name_list:
            if propertie in get_object.__dict__.keys():
                propertie_values.append(getattr(get_object, propertie))
            else:
                propertie_values.append(None)

    # If propertie_values is still None, create a list of None values
    propertie_values = (
        propertie_values if propertie_values else [None] * len(propertie_name_list)
    )

    # Set attributes on the set_object
    for propertie, value in zip(propertie_name_list, propertie_values):
        if propertie in set_object.__dict__.keys() and not override:
            continue
        setattr(set_object, propertie, value)


def attributes_present(attribute_names, object):
    for attribute_name in attribute_names:
        defined_variable = getattr(object, attribute_name)
        if defined_variable == None:
            return attribute_name
    return True


def set_attributes_check_presents(
    propertie_name_list,
    set_object,
    get_object=None,
    propertie_values=None,
    needed_attributes=None,
):
    copy_attributes_to_object(
        propertie_name_list=propertie_name_list,
        set_object=set_object,
        get_object=get_object,
        propertie_values=propertie_values,
    )

    if needed_attributes:
        present = attributes_present(needed_attributes, set_object)
        if present != True:
            raise NameError(
                f"Variable {present} is not defined in yaml file for {set_object.id}"
            )
    return set_object


# matlab
def load_matlab_metadata(mat_fpath):
    import scipy.io as sio

    mat = sio.loadmat(mat_fpath)
    # get the metadata
    dtypes = mat["metadata"][0][0].dtype.fields
    values = mat["metadata"][0][0]
    for attribute, value in zip(dtypes, values):
        value = value[0] if len(value) > 0 else " "
        print(f"{attribute:>12}: {str(value)}")


# date time
def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


def extract_date_from_filename(filename: str) -> str:
    """Extract the date from the filename.

    Parameters
    ----------
        filename (str): The filename to extract the date from.

    Returns
    -------
        str: The extracted date in YYYYMMDD format.
    """
    match = re.search(r"\d{8}", filename)
    if match:
        date = num_to_date(match.group(0))
        if date:
            return match.group(0)
        else:
            raise ValueError(f"Invalid date format in filename {filename}.")


def range_to_seconds(end: int, fps: float, start: int = 0):
    return np.arange(start, end) / fps


def second_to_time(second: float, striptime_format: str = "%M:%S"):
    return datetime.fromtimestamp(second).strftime("%M:%S")


def seconds_to_times(seconds: List[float]):
    if seconds[-1] > 60 * 60:  # if more than 1 hour
        times = np.full(len(seconds), "00:00:00")
        striptime_format = "%H:%M:%S"
    else:
        times = np.full(len(seconds), "00:00")
        striptime_format = "%M:%S"

    for i, sec in enumerate(seconds):
        times[i] = second_to_time(sec, striptime_format)
    return times


def range_to_times(end: int, fps: float, start: int = 0):
    seconds = range_to_seconds(end, fps, start)
    times = seconds_to_times(seconds)
    return times


def range_to_times_xlables_xpos(
    end: int, fps: float, start: int = 0, seconds_per_label: float = 30
):
    seconds = range_to_seconds(end, fps, start)
    if seconds[-1] > 60 * 60:  # if more than 1 hour
        striptime_format = "%H:%M:%S"
    else:
        striptime_format = "%M:%S"
    minutes = seconds_to_times(seconds)
    xticks = [minutes[0]]
    xpos = [0]
    for i, minute in enumerate(minutes[1:]):
        time_before = datetime.strptime(xticks[-1], striptime_format)
        time_current = datetime.strptime(minute, striptime_format)
        if (time_current - time_before).seconds > seconds_per_label:
            xticks.append(minute)
            xpos.append(i + 1)
    return xticks, xpos


# generators
def generate_linspace_samples(bounds, num):
    # Generate a list of linspace arrays for each dimension
    linspaces = [np.linspace(start, stop, num) for start, stop in bounds]

    # Create the meshgrid for all combinations
    meshgrids = np.meshgrid(*linspaces, indexing="ij")

    # Combine the grids into a single (N, dimensions) array of points
    samples = np.vstack([grid.ravel() for grid in meshgrids]).T

    return samples


# decorator functions timer
def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def profile_function(file_name="profile_output"):
    """
    Run a function and profile it using the Profiler class.

    Installation via pip install pyinstrument.

    Example:
        @profile_function(file_name="profile_output_parallel")
        def do_parallel(output_fname):
            print("Running parallel function")

        do_parallel()
    """

    def decorator(func):
        def wrap_func(*args, **kwargs):
            profiler = Profiler()
            profiler.start()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print("Error in function")
                result = e
            profiler.stop()
            output_text = profiler.output_text(
                unicode=True,
                color=True,
            )

            # extract duration and cpu time
            print(
                "\033[92mDuration:\033[0m {}, \033[94mCPU time:\033[0m {}".format(
                    *re.search(
                        r"Duration:\s*([\d.]+)\s+CPU time:\s*([\d.]+)", output_text
                    ).groups()
                )
            )

            # Save the output to an HTML file
            with open(f"./logs/{file_name}.html", "w") as f:
                f.write(profiler.output_html())
            # delete the profiler instance
            del profiler
            return result

        return wrap_func

    return decorator


def cleanup_profiler():
    """
    Cleanup function to stop all active Profiler and delete instances.
    """
    # Find all active Profiler instances
    profilers = [obj for obj in gc.get_objects() if isinstance(obj, Profiler)]

    # Stop them all if still running
    for profiler in profilers:
        try:
            profiler.stop()
        except Exception as e:
            print(f"Could not stop profiler: {e}")
    # Delete all instances
    for profiler in profilers:
        try:
            del profiler
        except Exception as e:
            print(f"Could not delete profiler: {e}")


def iter_dict_with_progress(input_dict, desc: str = "Processing"):
    """
    Iterate through a dictionary with a tqdm progress bar.

    Args:
        input_dict (dict): Dictionary to iterate through
    Yields:
        tuple: (key, value) pairs from the dictionary
    """
    # Create a tqdm progress bar for the dictionary items
    for key, value in tqdm(
        input_dict.items(), desc=desc, position=tqdm._get_free_pos(), leave=True
    ):
        yield key, value


def hierarchical_clustering(
    data: np.ndarray,
    linkage_method: Literal[
        "single",
        "complete",
        "average",
        "ward",
        "centroid",
        "median",
        "weighted",
    ] = "ward",
    xticks: Optional[List[str]] = None,
    yticks: Optional[List[str]] = None,
    xticks_pos: Optional[List[float]] = None,
    yticks_pos: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    plot: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[int],
    Optional[List[str]],
    Optional[List[str]],
    Optional[List[float]],
    Optional[List[float]],
]:
    """Performs hierarchical clustering on a distance matrix and optionally plots a dendrogram.

    Args:
        data: 2D NumPy array representing a distance matrix (symmetric) or data matrix.
        linkage_method: Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward', etc.).
        xticks: Optional custom tick labels for x-axis (reordered by clustering).
        yticks: Optional custom tick labels for y-axis (reordered by clustering).
        xticks_pos: Optional custom tick positions for x-axis (reordered by clustering).
        yticks_pos: Optional custom tick positions for y-axis (reordered by clustering).
        ax: Optional matplotlib axes to plot the dendrogram. If None and plot=True, creates a new figure.
        plot: If True and ax is None, creates a new figure to plot the dendrogram.

    Returns:
        Tuple containing:
        - Z: Linkage matrix from hierarchical clustering.
        - data_reordered: Reordered data matrix based on clustering.
        - sorted_indices: Indices of the reordered data points.
        - xticks: Reordered x-axis tick labels (if provided, else None).
        - yticks: Reordered y-axis tick labels (if provided, else None).
        - xticks_pos: Reordered x-axis tick positions (if provided, else None).
        - yticks_pos: Reordered y-axis tick positions (if provided, else None).

    Raises:
        ValueError: If data is not a 2D array, distance matrix is not symmetric, or linkage method is invalid.
    """
    # Validate input data
    if len(data.shape) != 2:
        raise ValueError("Data must be a 2D array for hierarchical clustering")

    # Validate linkage method
    valid_methods = [
        "single",
        "complete",
        "average",
        "ward",
        "centroid",
        "median",
        "weighted",
    ]
    if linkage_method not in valid_methods:
        raise ValueError(
            f"Invalid linkage method: {linkage_method}. Choose from {valid_methods}"
        )

    # Create a copy of the data to avoid modifying the input
    data_copy = np.copy(data)

    # Handle NaN values and check if data is a distance matrix
    nan_mask = np.isnan(data_copy)
    data_copy[nan_mask] = 0  # Temporarily replace NaNs for symmetry check
    is_distance_matrix = np.allclose(data_copy, data_copy.T, rtol=1e-5, atol=1e-8)

    # Prepare distance matrix
    if is_distance_matrix:
        # Use the input data directly as the distance matrix
        dist_matrix = np.copy(data_copy)
    else:
        # Handle NaN values for correlation calculation
        data_clean = np.nan_to_num(data_copy, nan=np.nanmean(data_copy))
        # Calculate distance matrix (1 - correlation)
        dist_matrix = squareform(1 - np.corrcoef(data_clean))

    # Handle any remaining NaN or inf values in the distance matrix
    dist_matrix = np.nan_to_num(dist_matrix, nan=0)

    # Verify symmetry of distance matrix
    if not np.allclose(dist_matrix, dist_matrix.T, rtol=1e-5, atol=1e-8):
        raise ValueError("Distance matrix must be symmetric")

    # Perform hierarchical clustering
    Z = linkage(squareform(dist_matrix), method=linkage_method, optimal_ordering=True)
    sorted_indices = leaves_list(Z)

    # Restore NaNs in data_copy and reorder
    data_copy[nan_mask] = np.nan
    data_reordered = data_copy[sorted_indices, :][:, sorted_indices]

    # Reorder ticks if provided
    if xticks is not None:
        xticks = [xticks[i] for i in sorted_indices]
    if yticks is not None:
        yticks = [yticks[i] for i in sorted_indices]
    if xticks_pos is not None:
        xticks_pos = [xticks_pos[i] for i in sorted_indices]
    if yticks_pos is not None:
        yticks_pos = [yticks_pos[i] for i in sorted_indices]

    # Plot dendrogram if requested
    if plot or ax is not None:
        if ax is None:
            # Create a new figure if no axes provided and plot=True
            fig, ax = plt.subplots(figsize=(8, 3))
        dendrogram(
            Z,
            ax=ax,
            orientation="top",
            labels=xticks,
            leaf_rotation=90,
            leaf_font_size=8,
        )
        ax.set_xticks([])  # Hide x-ticks as in original code
        ax.set_yticks([])  # Hide y-ticks as in original code
        if ax is None:
            plt.show()

    return Z, data_reordered, sorted_indices, xticks, yticks, xticks_pos, yticks_pos
