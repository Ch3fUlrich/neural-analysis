"""Structure Index: Quantifying Neural Manifold Organization.

This module implements the Structure Index (SI), a dimensionless metric that
quantifies how well neural population activity organizes according to external
behavioral or stimulus variables. The SI captures the degree to which neural
representations form a structured manifold aligned with task-relevant features.

The Structure Index is computed by:
1. Binning neural data according to label variables (e.g., position, velocity)
2. Creating a weighted directed graph where nodes are bins and edges represent
   overlap between neural activity patterns
3. Computing a summary statistic that quantifies manifold coherence
4. Comparing against shuffled null distributions to assess significance

Key Features:
- Multi-dimensional label support (vectorial features)
- Flexible distance metrics (Euclidean, geodesic via Isomap)
- Neighborhood-based (k-NN) or radius-based overlap computation
- Batch processing with HDF5 saving for large-scale analyses
- Fast nearest-neighbor search using FAISS when available
- Automatic outlier filtering to handle noisy data
- Statistical significance testing via label shuffling

The Structure Index was developed to characterize how neural population dynamics
relate to behavioral state spaces, with applications in motor control, navigation,
decision-making, and other domains where neural activity must track external
variables in a structured manner.

References
----------
.. [1] Bernardi et al. (2020). "The Geometry of Abstraction in the Hippocampus
       and Prefrontal Cortex." Cell, 183(4), 954-967.

Examples
--------
Basic usage with neural data and position labels:

>>> import numpy as np
>>> from neural_analysis.topology import compute_structure_index
>>> # Neural activity: 1000 timepoints × 50 neurons
>>> data = np.random.randn(1000, 50)
>>> # 2D position labels: 1000 timepoints × 2 coordinates
>>> labels = np.random.randn(1000, 2)
>>> # Compute structure index
>>> SI, bin_info, overlap_mat, shuf_SI = compute_structure_index(
...     data, labels, n_bins=10, n_neighbors=15
... )
>>> print(f"Structure Index: {SI:.3f}")
>>> print(f"Chance level (mean): {np.mean(shuf_SI):.3f}")

Batch processing with HDF5 saving:

>>> from pathlib import Path
>>> results_path = Path("results/structure_indices.h5")
>>> for session_id, (neural_data, position) in data_dict.items():
...     SI, _, overlap_mat, _ = compute_structure_index(neural_data, position)
...     save_structure_index_batch(
...         results_path,
...         item_id=session_id,
...         si=SI,
...         overlap_mat=overlap_mat,
...         n_bins=10,
...         n_neighbors=15
...     )

Visualizing the overlap graph:

>>> import matplotlib.pyplot as plt
>>> from neural_analysis.topology import draw_overlap_graph
>>> fig, ax = plt.subplots(figsize=(10, 10))
>>> draw_overlap_graph(
...     overlap_mat,
...     ax=ax,
...     node_names=[f"Bin {i}" for i in range(10)],
...     scale_edges=5
... )
>>> plt.show()

Notes
-----
- The Structure Index is sensitive to the number of bins and neighbors chosen
- For high-dimensional label spaces, consider using fewer bins per dimension
- Geodesic distances (via Isomap) capture manifold structure better than
  Euclidean distances for nonlinear embeddings
- Shuffled null distributions account for chance-level structure
- Outlier filtering helps remove noise but may discard valid boundary points

See Also
--------
draw_overlap_graph : Visualize the structure index graph
save_structure_index_batch : Save results for batch processing
neural_analysis.utils.io.save_hdf5 : General HDF5 saving utilities
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from decorator import decorator  # type: ignore[import-untyped]
from scipy.spatial import distance_matrix
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

try:
    import faiss

    USE_FAST = True
except ImportError:
    USE_FAST = False

logger = logging.getLogger(__name__)

# Supported distance metrics
DISTANCE_OPTIONS = ["euclidean", "geodesic"]


def validate_args_types(**decls: Any) -> Any:
    """Decorator to check argument types.

    Usage:

    @check_args(name=str, text=(int,str))
    def parse_rule(name, text): ...
    """

    @decorator  # type: ignore[misc]
    def wrapper(func: Any, *args: Any, **kwargs: Any) -> Any:
        code = func.__code__
        fname = func.__name__
        names = code.co_varnames[: code.co_argcount]
        for argname, argtype in decls.items():
            arg_provided = True
            if argname in names:
                argval = args[names.index(argname)]
            elif argname in kwargs:
                argval = kwargs.get(argname)
            else:
                arg_provided = False
            if arg_provided and not isinstance(argval, argtype):
                raise TypeError(
                    f"{fname}(...): arg '{argname}': type is {type(argval)}, "
                    f"must be {argtype}"
                )
        return func(*args, **kwargs)

    return wrapper


def _outlier_detection(data: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Filter outliers from data based on local density.
    
    Identifies points that have fewer neighbors than expected based on
    distance thresholds, marking them as potential outliers.
    
    Parameters:
    ----------
        data: numpy 2d array of shape [n_samples, n_features]
            Array containing data points to filter
    
    Returns:
    -------
        noiseIdx: numpy 1d array
            Indices of identified outlier points
    """
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(np.nanpercentile(D, 1) > D, axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    return noiseIdx.astype(np.float64)


def _meshgrid2(arrs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Create a meshgrid from a tuple of 1D arrays.
    
    Similar to numpy.meshgrid but with a different implementation that
    handles arbitrary dimensions more efficiently.
    
    Parameters:
    ----------
        arrs: tuple of numpy arrays
            Tuple containing 1D arrays for each dimension
    
    Returns:
    -------
        ans: tuple of numpy arrays
            Meshgrid arrays for each dimension
    """
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)


def _create_ndim_grid(
    label: npt.NDArray[np.floating[Any]],
    n_bins: list[int],
    min_label: list[float],
    max_label: list[float],
    discrete_label: list[bool],
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Create an N-dimensional grid for binning data.
    
    Divides the label space into bins and assigns data points to bins.
    Handles both continuous and discrete label dimensions.
    
    Parameters:
    ----------
        label: numpy 2d array of shape [n_samples, n_features]
            Array containing label values to bin
        
        n_bins: list[Any] of integers
            Number of bins for each dimension
        
        min_label: list[Any] of scalars
            Minimum value for each dimension
        
        max_label: list[Any] of scalars
            Maximum value for each dimension
        
        discrete_label: list[Any] of booleans
            Whether each dimension should be treated as discrete
    
    Returns:
    -------
        grid: numpy 1d array of objects
            Array where each element contains indices of points in that bin
        
        coords: numpy 3d array
            Coordinates of bin edges and centers [n_bins, n_dims, 3]
            where the last dimension contains [min_edge, center, max_edge]
    """
    ndims = label.shape[1]
    grid_edges = []

    for nd in range(ndims):
        if discrete_label[nd]:
            grid_edges.append(np.tile(np.unique(label[:, nd]).reshape(-1, 1), (1, 2)))
        else:
            edges = np.linspace(min_label[nd], max_label[nd], n_bins[nd] + 1).reshape(
                -1, 1
            )
            grid_edges.append(np.concatenate((edges[:-1], edges[1:]), axis=1))

    grid = np.empty([e.shape[0] for e in grid_edges], object)
    mesh = _meshgrid2(tuple([np.arange(s) for s in grid.shape]))
    meshIdx = np.vstack([col.ravel() for col in mesh]).T
    coords = np.zeros(meshIdx.shape + (3,))
    grid = grid.ravel()

    for elem, idx in enumerate(meshIdx):
        logic = np.zeros(label.shape[0])
        for dim in range(len(idx)):
            min_edge = grid_edges[dim][idx[dim], 0]
            max_edge = grid_edges[dim][idx[dim], 1]
            logic = logic + 1 * np.logical_and(
                label[:, dim] >= min_edge, label[:, dim] <= max_edge
            )
            coords[elem, dim, 0] = min_edge
            coords[elem, dim, 1] = 0.5 * (min_edge + max_edge)
            coords[elem, dim, 2] = max_edge
        grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])

    return grid, coords


def _cloud_overlap_neighbors(
    cloud1: npt.NDArray[np.floating[Any]],
    cloud2: npt.NDArray[np.floating[Any]],
    k: int,
    distance_metric: str,
) -> tuple[float, float]:
    """Compute overlapping between two clouds of points using k-nearest neighbors.

    Parameters:
    ----------
        cloud1: numpy 2d array of shape [n_samples_1, n_features]
            Array containing the first cloud of points

        cloud2: numpy 2d array of shape [n_samples_2, n_features]
            Array containing the second cloud of points

        k: int
            Number of neighbors used to compute the overlapping between
            bin-groups. This parameter controls the tradeoff between local
            and global structure. Larger k captures more global structure.

        distance_metric: str
            Type of distance used to compute the closest n_neighbors. Options:
            - 'euclidean': Standard Euclidean distance
            - 'geodesic': Geodesic distance via Isomap

    Returns:
    -------
        overlap_1_2: float
            Degree of overlap of cloud1 over cloud2 (fraction of cloud1's 
            neighbors that belong to cloud2)

        overlap_2_1: float
            Degree of overlap of cloud2 over cloud1 (fraction of cloud2's
            neighbors that belong to cloud1)

    Notes:
    -----
        - Uses FAISS library for fast nearest neighbor search if available
        - Falls back to sklearn's NearestNeighbors if FAISS not available
        - Handles edge case where k >= total number of points
    """
    cloud_all = np.vstack((cloud1, cloud2)).astype("float32")
    idx_sep = cloud1.shape[0]
    n_points = cloud_all.shape[0]

    if k > n_points - 1:
        overlap_1_2 = (
            cloud1.shape[0] * cloud2.shape[0] / (cloud1.shape[0] * (n_points - 1))
        )
        overlap_2_1 = (
            cloud1.shape[0] * cloud2.shape[0] / (cloud2.shape[0] * (n_points - 1))
        )
        return overlap_1_2, overlap_2_1

    if distance_metric == "euclidean":
        if USE_FAST:
            index = faiss.IndexFlatL2(cloud_all.shape[1])
            index.add(cloud_all)
            _, I = index.search(cloud_all, k + 1)
            I = I[:, 1:]
        else:
            knn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(
                cloud_all
            )
            I = knn.kneighbors(return_distance=False)
    elif distance_metric == "geodesic":
        model_iso = Isomap(n_components=1)
        model_iso.fit_transform(cloud_all)
        dist_mat = model_iso.dist_matrix_
        knn = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(dist_mat)
        I = knn.kneighbors(return_distance=False)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")


    # Compute overlapping: fraction of neighbors belonging to other cloud
    # For cloud1, count how many neighbors belong to cloud2 (indices >= idx_sep)
    overlap_1_2 = np.sum(I[:idx_sep, :] >= idx_sep) / (cloud1.shape[0] * k)
    # For cloud2, count how many neighbors belong to cloud1 (indices < idx_sep)
    overlap_2_1 = np.sum(I[idx_sep:, :] < idx_sep) / (cloud2.shape[0] * k)

    return overlap_1_2, overlap_2_1


def _cloud_overlap_radius(
    cloud1: npt.NDArray[np.floating[Any]],
    cloud2: npt.NDArray[np.floating[Any]],
    r: float,
    distance_metric: str,
) -> tuple[float, float]:
    """Compute overlapping between two clouds of points using radius-based neighborhoods.

    Parameters:
    ----------
        cloud1: numpy 2d array of shape [n_samples_1, n_features]
            Array containing the first cloud of points

        cloud2: numpy 2d array of shape [n_samples_2, n_features]
            Array containing the second cloud of points

        r: float
            Radius used to define neighborhoods. Points within this radius
            are considered neighbors.

        distance_metric: str
            Type of distance used to compute neighborhoods. Options:
            - 'euclidean': Standard Euclidean distance
            - 'geodesic': Geodesic distance via Isomap

    Returns:
    -------
        overlap_1_2: float
            Degree of overlap of cloud1 over cloud2 (fraction of cloud1's 
            neighbors that belong to cloud2)

        overlap_2_1: float
            Degree of overlap of cloud2 over cloud1 (fraction of cloud2's
            neighbors that belong to cloud1)

    Notes:
    -----
        Overlap is directional - overlap_1_2 measures how much cloud1 extends
        into cloud2's region, while overlap_2_1 measures the reverse.
    """
    cloud_all = np.vstack((cloud1, cloud2)).astype("float32")
    idx_sep = cloud1.shape[0]

    if distance_metric == "euclidean":
        D = distance_matrix(cloud_all, cloud_all, p=2)
    elif distance_metric == "geodesic":
        model_iso = Isomap(n_components=1)
        model_iso.fit_transform(cloud_all)
        D = model_iso.dist_matrix_
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    I = np.argsort(D, axis=1)
    for row in range(I.shape[0]):
        D[row, :] = D[row, I[row, :]]
    I = I[:, 1:].astype("float32")
    D = D[:, 1:]

    I[r < D] = np.nan
    num_neigh = I.shape[0] - np.sum(np.isnan(I), axis=1).astype("float32") - 1

    overlap_1_2 = np.sum(I[:idx_sep, :] >= idx_sep) / np.sum(num_neigh[:idx_sep])
    overlap_2_1 = np.sum(I[idx_sep:, :] < idx_sep) / np.sum(num_neigh[idx_sep:])

    return overlap_1_2, overlap_2_1


@validate_args_types(  # type: ignore[misc]
    data=np.ndarray,
    label=np.ndarray,
    n_bins=(int, np.integer, list),
    dims=(type(None), list),
    distance_metric=str,
    n_neighbors=(int, np.integer),
    num_shuffles=(int, np.integer),
    discrete_label=(list, bool),
    verbose=bool,
)
def compute_structure_index(
    data: npt.NDArray[Any],
    label: npt.NDArray[Any],
    n_bins: int | list[int] = 10,
    dims: list[int] | None = None,
    **kwargs: Any,
) -> tuple[float, tuple[Any, ...], npt.NDArray[Any], npt.NDArray[Any]]:
    """Compute structure index main function.

    Parameters
    ----------
    data : numpy 2d array of shape [n_samples, n_dimensions]
        Array containing the neural activity signal.

    label : numpy 2d array of shape [n_samples, n_features]
        Array containing the labels of the data. It can either be a
        column vector (scalar feature) or a 2D array (vectorial feature).

    n_bins : int or list of int, default=10
        Number of bin-groups the label will be divided into (they will
        become nodes on the graph). For vectorial features, if one wants
        different number of bins for each entry then specify n_bins as a
        list (i.e. [10,20,5]). Note that it will be ignored if
        'discrete_label' is set to True.

    dims : list[Any] of int or None, default=None
        List of integers containing the dimensions of data along which the
        structure index will be computed. Provide None to compute it along
        all dimensions of data.

    **kwargs : dict[str, Any]
        Additional keyword arguments:

        distance_metric: str (default: 'euclidean')
            Type of distance used to compute the closest n_neighbors. See
            'distance_options' for currently supported distances.

        n_neighbors: int (default: 15)
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
    -------
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
    # __________________________________________________________________________
    # |                                                                        |#
    # |                        0. CHECK INPUT VALIDITY                         |#
    # |________________________________________________________________________|#
    # Note input type validity is handled by the decorator. Here the values
    # themselves are being checked.
    # i) data input
    assert data.ndim == 2, "data must be 2D array"
    # ii) label input
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    assert label.ndim == 2, "label must be 1D or 2D array"

    # iii) n_bins input
    # Process n_bins
    if isinstance(n_bins, (int, np.integer)):
        assert n_bins > 1, "n_bins must be > 1"
        n_bins = [n_bins for _ in range(label.shape[1])]
    elif isinstance(n_bins, list):
        assert all(nb > 1 for nb in n_bins), "All n_bins must be > 1"

    # iv) dims input
    if dims is None:
        dims = list(range(data.shape[1]))

    # v) distance_metric
    distance_metric = kwargs.get("distance_metric", "euclidean")
    assert (
        distance_metric in DISTANCE_OPTIONS
    ), f"Invalid distance_metric. Choose from {DISTANCE_OPTIONS}"

    # ix) n_neighbors input
    if "n_neighbors" in kwargs and "radius" in kwargs:
        raise ValueError("Specify either n_neighbors or radius, not both")

    if "radius" in kwargs:
        neighborhood_size = kwargs["radius"]
        assert neighborhood_size > 0, "radius must be > 0"
        cloud_overlap = _cloud_overlap_radius
    else:
        neighborhood_size = float(kwargs.get("n_neighbors", 15))
        assert neighborhood_size > 2, "n_neighbors must be > 2"
        cloud_overlap = _cloud_overlap_neighbors  # type: ignore[assignment]

    # x) discrete_label input
    discrete_label = kwargs.get("discrete_label", False)
    if isinstance(discrete_label, bool):
        discrete_label = [discrete_label for _ in range(label.shape[1])]
    else:
        assert all(
            isinstance(d, bool) for d in discrete_label
        ), "discrete_label must be bool or list of bool"

    # xi) num_shuffles input
    num_shuffles = kwargs.get("num_shuffles", 100)
    assert num_shuffles >= 0, "num_shuffles must be >= 0"

    # xii) verbose input
    verbose = kwargs.get("verbose", False)

    # __________________________________________________________________________
    # |                                                                        |#
    # |                           1. PREPROCESS DATA                           |#
    # |________________________________________________________________________|#
    # i).Keep only desired dims
    data = data[:, dims]
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # ii).Delete nan values from label and data
    data_nans = np.any(np.isnan(data), axis=1)
    label_nans = np.any(np.isnan(label), axis=1)
    delete_nans = np.where(data_nans + label_nans)[0]
    data = np.delete(data, delete_nans, axis=0)
    label = np.delete(label, delete_nans, axis=0)

    # iii).Binarize label
    if verbose:
        print("Computing bin-groups...", end="")

    for dim in range(label.shape[1]):
        num_unique = len(np.unique(label[:, dim]))
        if discrete_label[dim]:
            n_bins[dim] = num_unique
        elif n_bins[dim] >= num_unique:
            warnings.warn(
                f"Column {dim}: fewer unique values ({num_unique}) than n_bins "
                f"({n_bins[dim]}). Setting n_bins={num_unique} and discrete=True",
                stacklevel=2,
            )
            n_bins[dim] = num_unique
            discrete_label[dim] = True

    # b) Create bin edges of bin-groups
    min_label = kwargs.get("min_label", np.percentile(label, 5, axis=0))
    if not isinstance(min_label, list):
        min_label = min_label.tolist()
    max_label = kwargs.get("max_label", np.percentile(label, 95, axis=0))
    if not isinstance(max_label, list):
        max_label = max_label.tolist()

    # Clip labels to bounds
    for ld in range(label.shape[1]):
        label[np.where(label[:, ld] < min_label[ld])[0], ld] = min_label[ld] + 0.00001
        label[np.where(label[:, ld] > max_label[ld])[0], ld] = max_label[ld] - 0.00001

    grid, coords = _create_ndim_grid(label, n_bins, min_label, max_label, discrete_label)
    bin_label = np.zeros(label.shape[0], dtype=int) * np.nan

    for b in range(len(grid)):
        bin_label[grid[b]] = b

    # iv). Clean outliers from each bin-groups if specified in kwargs
    if kwargs.get("filter_noise", False):
        for bin_idx in range(len(grid)):
            noise_idx = _outlier_detection(data[bin_label == bin_idx, :])
            noise_idx = np.where(bin_label == bin_idx)[0][noise_idx.astype(int)]
            bin_label[noise_idx] = np.nan

    # v). Discard outlier bin-groups (n_points < n_neighbors)
    # a) Compute number of points in each bin-group
    unique_bin_label = np.unique(bin_label[~np.isnan(bin_label)])
    n_points = np.array([np.sum(bin_label == val) for val in unique_bin_label])
    
    # b) Get the bin-groups that do not meet criteria and delete them
    min_points_per_bin = 0.1 * data.shape[0] / np.prod(n_bins)
    del_labels = np.where(n_points < min_points_per_bin)[0]

    # c) delete outlier bin-groups
    for del_idx in del_labels:
        bin_label[bin_label == unique_bin_label[del_idx]] = np.nan

    # d) re-computed valid bins
    unique_bin_label = np.unique(bin_label[~np.isnan(bin_label)])
    if len(unique_bin_label) <= 1:
        return np.nan, (np.nan, np.nan), np.nan, np.nan  # type: ignore[return-value]

    if verbose:
        print(" Done")

    # __________________________________________________________________________
    # |                                                                        |#
    # |                       2. COMPUTE STRUCTURE INDEX                       |#
    # |________________________________________________________________________|#
    # i). compute overlap between bin-groups pairwise
    num_bins = len(unique_bin_label)
    overlap_mat = np.zeros((num_bins, num_bins)) * np.nan

    if verbose:
        bar = tqdm(total=int((num_bins**2 - num_bins) / 2), desc="Computing overlap")

    for a in range(num_bins):
        A = data[bin_label == unique_bin_label[a]]
        for b in range(a + 1, num_bins):
            B = data[bin_label == unique_bin_label[b]]
            overlap_a_b, overlap_b_a = cloud_overlap(
                A, B, neighborhood_size, distance_metric
            )
            overlap_mat[a, b] = overlap_a_b
            overlap_mat[b, a] = overlap_b_a
            if verbose:
                bar.update(1)

    if verbose:
        bar.close()

    # ii). compute structure_index (SI)
    if verbose:
        print("Computing structure index...", end="")

    degree_nodes = np.nansum(overlap_mat, axis=1)
    SI = 1 - np.mean(degree_nodes) / (num_bins - 1)
    SI = 2 * (SI - 0.5)
    SI = float(max(SI, 0.0))

    if verbose:
        print(f" {SI:.2f}")

    # iii). Shuffling
    if num_shuffles == 0:
        # Return immediately if no shuffles requested
        return SI, (bin_label, coords), overlap_mat, np.array([])
    
    shuf_SI = np.zeros(num_shuffles) * np.nan
    shuf_overlap_mat = np.zeros(overlap_mat.shape)

    if verbose:
        bar = tqdm(total=num_shuffles, desc="Computing shuffles")

    for s_idx in range(num_shuffles):
        shuf_bin_label = copy.deepcopy(bin_label)
        np.random.shuffle(shuf_bin_label)
        shuf_overlap_mat *= np.nan

        for a in range(shuf_overlap_mat.shape[0]):
            A = data[shuf_bin_label == unique_bin_label[a]]
            for b in range(a + 1, shuf_overlap_mat.shape[1]):
                B = data[shuf_bin_label == unique_bin_label[b]]
                overlap_a_b, overlap_b_a = cloud_overlap(
                    A, B, neighborhood_size, distance_metric
                )
                shuf_overlap_mat[a, b] = overlap_a_b
                shuf_overlap_mat[b, a] = overlap_b_a

        # iii) compute structure_index (SI)
        degree_nodes = np.nansum(shuf_overlap_mat, axis=1)
        shuf_SI[s_idx] = 1 - np.mean(degree_nodes) / (num_bins - 1)
        shuf_SI[s_idx] = 2 * (shuf_SI[s_idx] - 0.5)
        shuf_SI[s_idx] = float(max(shuf_SI[s_idx], 0.0))

        if verbose:
            bar.update(1)

    if verbose:
        bar.close()
        print(f"Shuffling 99th percentile: {np.percentile(shuf_SI, 99):.2f}")

    return SI, (bin_label, coords), overlap_mat, shuf_SI


def draw_overlap_graph(
    overlap_mat: npt.NDArray[np.floating[Any]],
    ax: Any = None,
    node_cmap: Any = None,
    edge_cmap: Any = None,
    **kwargs: Any,
) -> Any:
    """Draw weighted directed graph from overlap matrix.

    Parameters
    ----------
    overlap_mat : ndarray, shape (n_bins, n_bins)
        Pairwise overlap matrix
    ax : matplotlib Axes, optional
        Axes to draw on
    node_cmap : colormap, default=plt.cm.tab10
        Colormap for nodes
    edge_cmap : colormap, default=plt.cm.Greys
        Colormap for edges
    **kwargs : dict[str, Any]
        Additional parameters:
        - node_size: scalar or array (default: 800)
        - scale_edges: float (default: 5)
        - edge_vmin: float (default: 0)
        - edge_vmax: float (default: 0.5)
        - node_names: list[Any] of str
        - node_color: list[Any] of colors
        - arrow_size: int (default: 20)
        - layout_type: networkx layout function

    Returns
    -------
    graph : networkx graph drawing

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from neural_analysis.topology import compute_structure_index, draw_overlap_graph
    >>> # After computing structure index...
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> draw_overlap_graph(overlap_mat, ax=ax, node_names=[f"Bin {i}" for i in range(10)])
    >>> plt.show()
    """
    if int(nx.__version__[0]) < 3:
        g = nx.from_numpy_matrix(overlap_mat, create_using=nx.DiGraph)
    else:
        g = nx.from_numpy_array(overlap_mat, create_using=nx.DiGraph)

    number_nodes = g.number_of_nodes()
    node_size = kwargs.get("node_size", 800)
    scale_edges = kwargs.get("scale_edges", 5)
    edge_vmin = kwargs.get("edge_vmin", 0)
    edge_vmax = kwargs.get("edge_vmax", 0.5)
    arrow_size = kwargs.get("arrow_size", 20)
    node_color = kwargs.get("node_color", False)
    layout_type = kwargs.get("layout_type", nx.circular_layout)

    if "node_names" in kwargs:
        node_names = kwargs["node_names"]
        nodes_info = list(g.nodes(data=True))
        names_dict = {val[0]: node_names[i] for i, val in enumerate(nodes_info)}
        with_labels = True
        node_val = node_names if not isinstance(node_names[0], str) else range(number_nodes)
    else:
        names_dict = {}
        node_val = range(number_nodes)
        with_labels = False

    if not node_color:
        norm_cmap = matplotlib.colors.Normalize(
            vmin=np.min(node_val), vmax=np.max(node_val)
        )
        node_color = [
            np.array(node_cmap(norm_cmap(node_val[ii]), bytes=True)) / 255
            for ii in range(number_nodes)
        ]

    widths = nx.get_edge_attributes(g, "weight")

    wdg = nx.draw_networkx(
        g,
        pos=layout_type(g),
        node_size=node_size,
        node_color=node_color,
        width=np.array(list(widths.values())) * scale_edges,
        edge_color=np.array(list(widths.values())),
        edge_cmap=edge_cmap,
        arrowsize=arrow_size,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        labels=names_dict,
        arrows=True,
        connectionstyle="arc3,rad=0.15",
        with_labels=with_labels,
        ax=ax,
    )

    return wdg


def compute_structure_index_sweep(
    data: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    dataset_name: str,
    save_path: str | Path | None = None,
    n_neighbors_list: list[int] | None = None,
    n_bins_list: list[int] | None = None,
    data_indices: npt.NDArray[np.int_] | None = None,
    discrete_label: bool = False,
    num_shuffles: int = 100,
    distance_metric: str = "euclidean",
    regenerate: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run Structure Index computation with parameter sweeps and automatic caching.
    
    This function orchestrates multiple Structure Index computations across
    different parameter combinations, automatically saving results to an HDF5
    file and loading cached results when available.
    
    Parameters
    ----------
    data : ndarray
        Neural activity data, shape (n_samples, n_features)
    labels : ndarray
        Behavioral/stimulus labels, shape (n_samples, n_label_dims)
    dataset_name : str
        Unique identifier for this dataset (e.g., "session_001")
    save_path : str or Path, optional
        Path to HDF5 file for saving results. If None, defaults to 
        './output/structure_indices.h5'
    n_neighbors_list : list[Any] of int, optional
        List of n_neighbors values to sweep. Default: [10, 15, 20]
    n_bins_list : list[Any] of int, optional
        List of n_bins values to sweep. Default: [10]
    data_indices : ndarray of int, optional
        Indices of data points to use. If None, uses all data.
    discrete_label : bool, default=False
        Whether labels are discrete categories
    num_shuffles : int, default=100
        Number of shuffles for statistical testing
    distance_metric : str, default='euclidean'
        Distance metric ('euclidean' or 'geodesic')
    regenerate : bool, default=False
        If True, recompute even if cached results exist
    verbose : bool, default=False
        Print detailed progress information
    **kwargs
        Additional parameters passed to compute_structure_index
        
    Returns
    -------
    results : dict[str, Any]
        Dictionary with keys (n_bins, n_neighbors) and values containing
        SI results: {'SI': float, 'bin_label': tuple, 'overlap_mat': ndarray,
        'shuf_SI': ndarray, 'metadata': dict[str, Any]}
        
    Examples
    --------
    >>> results = compute_structure_index_sweep(
    ...     data=neural_data,
    ...     labels=position_labels,
    ...     dataset_name="session_001",
    ...     save_path="results/structure_indices.h5",
    ...     n_neighbors_list=[10, 15, 20],
    ...     n_bins_list=[8, 10, 12]
    ... )
    >>> si_value = results[(10, 15)]['SI']
    """
    from neural_analysis.utils.io import (
        load_results_from_hdf5_dataset,
        save_result_to_hdf5_dataset,
    )
    
    # Set default save path if not provided
    if save_path is None:
        save_path = Path("./output/structure_indices.h5")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set defaults
    if n_neighbors_list is None:
        n_neighbors_list = [10, 15, 20]
    if n_bins_list is None:
        n_bins_list = [10]
    
    # Subset data if indices provided
    if data_indices is not None:
        data = data[data_indices]
        labels = labels[data_indices]
        indices_key = _array_to_key(data_indices)  # type: ignore[name-defined]
    else:
        indices_key = "all"
    
    # Load existing results
    existing_data = load_results_from_hdf5_dataset(
        save_path=save_path,
        dataset_name=dataset_name,
    )
    existing_results = _parse_loaded_results(existing_data, dataset_name)  # type: ignore[name-defined]
    
    results = {}
    total_iterations = len(n_bins_list) * len(n_neighbors_list)
    
    with tqdm(
        total=total_iterations,
        desc=f"Computing SI for {dataset_name}",
        disable=not verbose,
    ) as pbar:
        for n_bins in n_bins_list:
            for n_neighbors in n_neighbors_list:
                param_key = (n_bins, n_neighbors)
                
                # Check if already computed
                if param_key in existing_results and not regenerate:
                    indices_key_present = existing_results[param_key]["metadata"].get(
                        "indices_key"
                        )
                    if indices_key_present == indices_key:
                        results[param_key] = existing_results[param_key]
                        pbar.update(1)
                        continue
                
                # Compute structure index
                si, bin_label, overlap_mat, shuf_si = compute_structure_index(
                    data=data,
                    label=labels,
                    n_bins=n_bins,
                    n_neighbors=n_neighbors,
                    discrete_label=discrete_label,
                    num_shuffles=num_shuffles,
                    distance_metric=distance_metric,
                    verbose=False, **kwargs)
                
                # Store result
                metadata = {
                    "n_bins": n_bins,
                    "n_neighbors": n_neighbors,
                    "distance_metric": distance_metric,
                    "num_shuffles": num_shuffles,
                    "discrete_label": discrete_label,
                    "indices_key": indices_key,
                    "SI": float(si),
                    "shuf_SI_mean": (
                        float(np.mean(shuf_si)) if len(shuf_si) > 0 else np.nan
                    ),
                    "shuf_SI_std": (
                        float(np.std(shuf_si)) if len(shuf_si) > 0 else np.nan
                    ),
                }
                
                result = {
                    "SI": si,
                    "bin_label": bin_label,
                    "overlap_mat": overlap_mat,
                    "shuf_SI": shuf_si,
                    "metadata": metadata,
                }
                
                results[param_key] = result
                
                # Save incrementally
                result_key = f"bins{n_bins}_neighbors{n_neighbors}_{indices_key}"
                save_result_to_hdf5_dataset(
                    save_path=save_path,
                    dataset_name=dataset_name,
                    result_key=result_key,
                    scalar_data=metadata,
                    array_data={
                        "overlap_mat": overlap_mat,
                        "shuf_SI": shuf_si,
                        "bin_label_assignments": bin_label[0],
                        "bin_label_coords": bin_label[1],
                    },
                )
                
                pbar.update(1)
    
    logger.info(
        f"Completed Structure Index sweep for {dataset_name}: "
        f"{len(results)} parameter combinations"
    )
    
    return results


def load_structure_index_results(
    save_path: str | Path,
    dataset_name: str | None = None,
    n_bins: int | None = None,
    n_neighbors: int | None = None,
    indices_key: str | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load Structure Index results from HDF5 file.
    
    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    dataset_name : str, optional
        Filter by specific dataset. If None, loads all datasets.
    n_bins : int, optional
        Filter by specific n_bins value
    n_neighbors : int, optional
        Filter by specific n_neighbors value
    indices_key : str, optional
        Filter by specific data indices key
        
    Returns
    -------
    results : dict[str, Any]
        Dictionary with keys (n_bins, n_neighbors) and values containing
        SI results and metadata
        
    Examples
    --------
    >>> # Load all results for a dataset
    >>> results = load_structure_index_results(
    ...     "results/structure_indices.h5",
    ...     dataset_name="session_001"
    ... )
    >>> 
    >>> # Load specific parameter combination
    >>> results = load_structure_index_results(
    ...     "results/structure_indices.h5",
    ...     dataset_name="session_001",
    ...     n_bins=10,
    ...     n_neighbors=15
    ... )
    """
    from neural_analysis.utils.io import load_results_from_hdf5_dataset
    
    save_path = Path(save_path)
    
    if not save_path.exists():
        logger.debug(f"No saved results found at {save_path}")
        return {}
    
    # Build filter
    filter_attrs = {}
    if n_bins is not None:
        filter_attrs["n_bins"] = n_bins
    if n_neighbors is not None:
        filter_attrs["n_neighbors"] = n_neighbors
    if indices_key is not None:
        filter_attrs["indices_key"] = indices_key  # type: ignore[assignment]
    
    # Load data
    loaded_data = load_results_from_hdf5_dataset(
        save_path=save_path,
        dataset_name=dataset_name,
        filter_attrs=filter_attrs if filter_attrs else None,
    )
    
    # Parse results
    results = {}
    for ds_name, _ in loaded_data.items():
        ds_parsed = _parse_loaded_results(loaded_data, ds_name)  # type: ignore[name-defined]
        results.update(ds_parsed)
    
    return results


def _parse_loaded_results(
    loaded_data: dict[str, dict[str, Any]],
    dataset_name: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Parse loaded HDF5 data into structure index results format."""
    results = {}
    
    if dataset_name not in loaded_data:
        return results
    
    for _, result_data in loaded_data[dataset_name].items():
        attrs = result_data["attributes"]
        arrays = result_data["arrays"]
        
        # Extract parameter key
        n_bins = attrs.get("n_bins")
        n_neighbors = attrs.get("n_neighbors")
        
        if n_bins is None or n_neighbors is None:
            continue
        
        param_key = (int(n_bins), int(n_neighbors))
        
        # Reconstruct result
        results[param_key] = {
            "SI": attrs.get("SI", np.nan),
            "overlap_mat": arrays.get("overlap_mat", np.array([])),
            "shuf_SI": arrays.get("shuf_SI", np.array([])),
            "bin_label": (
                arrays.get("bin_label_assignments", np.array([])),
                arrays.get("bin_label_coords", np.array([])),
            ),
            "metadata": attrs,
        }
    
    return results


def _array_to_key(arr: npt.NDArray[np.int_]) -> str:
    """Convert array of indices to compact string key."""
    if len(arr) > 10:
        # Use hash for large arrays
        return f"hash_{hash(arr.tobytes())}"
    else:
        # Use actual indices for small arrays
        return "_".join(map(str, arr))



