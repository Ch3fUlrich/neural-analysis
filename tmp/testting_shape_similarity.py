from numpy._typing._array_like import NDArray
from numpy._typing._array_like import NDArray
from numpy._typing._array_like import NDArray
from numpy import int_
import time
import numpy as np
from numpy.typing import NDArray
import numpy.typing as npt
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypedDict, Union, Literal

from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import ot

OT_AVAILABLE = True

def modify_matrix(
    mtx: npt.NDArray[np.floating[Any]],
    whiten: bool = True,
    normalize: bool = True,
) -> npt.NDArray[np.floating[Any]]:
    """Preprocess matrix for shape comparison.

    Optionally whitens and/or normalizes to unit Frobenius norm.

    Parameters
    ----------
    mtx : ndarray of shape (n_samples, n_features)
        The matrix to preprocess.
    whiten : bool, default=True
        If True, center and scale each feature (column) to unit variance.
    normalize : bool, default=True
        If True, scale the entire matrix to have Frobenius norm = 1.

    Returns
    -------
    ndarray
        The preprocessed matrix.

    Notes
    -----
    - Whitening standardizes features to zero mean and unit variance
    - Normalization scales the overall matrix magnitude
    - Both operations preserve shape structure while removing scale effects
    """
    out = mtx.copy().astype(np.float64)

    if whiten:
        # Center and scale each column
        means = out.mean(axis=0, keepdims=True)
        stds = out.std(axis=0, keepdims=True, ddof=1)
        stds[stds == 0] = 1.0  # Avoid division by zero
        out = (out - means) / stds

    if normalize:
        # Scale to unit Frobenius norm
        norm = np.linalg.norm(out, ord="fro")
        if norm > 0:
            out = out / norm

    return out


def align_mtx(
    mtx1: npt.NDArray[np.floating],
    mtx2: npt.NDArray[np.floating],
    rotate: bool = True,
    scale: bool = True,
    whiten: bool = True,
    norm: bool = True,
) -> npt.NDArray[np.floating]:
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

    mtx1 = modify_matrix(mtx1, whiten=whiten, normalize=norm)
    mtx2 = modify_matrix(mtx2, whiten=whiten, normalize=norm)

    # Find optimal orthogonal transformation (rotation/reflection)
    # R transforms mtx1 to mtx2: mtx1 @ R ≈ mtx2
    # So to align mtx2 to mtx1, we use: mtx2 @ R.T
    # The scale 's' is the sum of singular values (a similarity measure)
    # and also equals the optimal scaling factor for general Procrustes
    if rotate or scale:
        r, s = orthogonal_procrustes(mtx1, mtx2)
        if rotate:
            mtx2 = np.dot(mtx2, r.T)
        if scale:
            mtx2 *= s

    return mtx2


def shape_distance_procrustes(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using Procrustes alignment.

    This method aligns the two matrices using Procrustes analysis and returns
    the residual disparity after optimal rotation/reflection, along with
    point-to-point correspondence information.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1. Must have same shape as mtx1.

    Returns
    -------
    distance : float
        Procrustes disparity (sum of squared Euclidean distances after
        optimal alignment). Lower values indicate more similar shapes.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping point index pairs (i, i) to their post-alignment
        distances. Since Procrustes preserves point correspondence, each
        point i in mtx1 is aligned to point i in mtx2.

    Notes
    -----
    The Procrustes method finds the optimal orthogonal transformation
    (rotation + reflection) that minimizes the distance between matrices.
    The scipy.spatial.procrustes function automatically standardizes both
    matrices (zero mean, unit variance per column, unit Frobenius norm).

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_procrustes(mtx1, mtx2)
    >>> print(f"Procrustes distance: {dist:.3f}")
    >>> print(f"Number of aligned pairs: {len(pairs)}")
    """
    m1, m2, disparity = procrustes(mtx1, mtx2)

    # Compute distance between pairs
    distances = np.linalg.norm(m1 - m2, axis=1)
    # Create pairs dictionary with point indices and their post-alignment squared distances
    pairs = {(i, i): float(distance) for i, distance in enumerate(distances)}

    return float(disparity), pairs


def shape_distance_one_to_one(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "sqeuclidean",
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using optimal one-to-one point matching via optimal transport.

    Uses optimal transport with hard assignment constraint (one-to-one matching) to find
    the optimal bijective matching between points. Similar to soft-matching but enforces
    that each point is matched to exactly one other point (hard assignment).

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1. Must have same number of samples.
    metric : str, default='sqeuclidean'
        Distance metric for computing transport costs. Passed to cdist.
        Common options: 'sqeuclidean', 'euclidean', 'cosine', 'correlation'.

    Returns
    -------
    distance : float
        Optimal transport cost with hard assignment (sum of transport plan * cost matrix).
        Uses squared distances when metric='sqeuclidean' for comparability with Procrustes.
        Lower values indicate more similar shapes.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping optimal point assignments (i, j) -> distance,
        where point i from mtx1 is matched to point j from mtx2.
        Each i and j appears exactly once (bijective matching).
        Values represent the distance between matched points.

    Notes
    -----
    This method uses optimal transport (Earth Mover's Distance) with uniform distributions
    and equal sizes, which naturally enforces one-to-one matching (hard assignment).
    Unlike soft-matching, each point can only be matched to one other point.

    Matrices are normalized to unit Frobenius norm before comparison (consistent with
    procrustes and soft-matching methods).

    Requires the `pot` package: pip install pot

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_one_to_one(mtx1, mtx2)
    >>> print(f"One-to-one distance: {dist:.3f}")
    >>> print(f"Number of matched pairs: {len(pairs)}")
    """
    if not OT_AVAILABLE:
        msg = "one-to-one requires the 'pot' library. Install with: pip install pot"
        raise ImportError(msg)

    # Normalize matrices to unit Frobenius norm (consistent with procrustes and soft-matching)
    m1 = modify_matrix(mtx1, whiten=False, normalize=True)
    m2 = modify_matrix(mtx2, whiten=False, normalize=True)

    # Compute cost matrix
    cost_matrix = cdist(m1, m2, metric=metric)

    # Uniform distributions (equal mass at each point)
    a = np.ones(m1.shape[0]) / m1.shape[0]
    b = np.ones(m2.shape[0]) / m2.shape[0]

    # Compute optimal transport plan with hard assignment (EMD gives one-to-one matching)
    transport_plan = ot.emd(a, b, cost_matrix)

    # Compute distance: sum of transport plan * cost matrix
    # For sqeuclidean metric, this gives sum of squared distances (comparable to Procrustes)
    distance = np.sum(transport_plan * cost_matrix)

    # Extract pairs with non-zero transport (hard assignment: exactly one match per point)
    i_indices, j_indices = np.where(transport_plan > 0)
    pairs = {
        (int(i), int(j)): float(cost_matrix[i, j])
        for i, j in zip(i_indices, j_indices, strict=False)
    }
    return float(distance), pairs


def shape_distance_soft_matching(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "sqeuclidean",
    approx: bool = False,
    reg: float = 0.1,
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using soft optimal transport matching.

    Uses optimal transport to compute a soft matching between point distributions,
    allowing fractional assignment of mass. Can use exact (Earth Mover's Distance)
    or approximate (Sinkhorn) algorithms.

    Reference:
    https://proceedings.mlr.press/v243/khosla24a/khosla24a.pdf

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples1, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples2, n_features)
        Second matrix to compare with mtx1. Can have different number of samples.
    metric : str, default='sqeuclidean'
        Distance metric for computing transport costs. Passed to cdist.
        Common options: 'sqeuclidean', 'euclidean', 'cosine', 'correlation'.
        Use 'sqeuclidean' for comparability with one-to-one and Procrustes.
    approx : bool, default=False
        If True, use Sinkhorn algorithm (faster, approximate).
        If False, use exact EMD algorithm (slower, exact).
        Note: With exact EMD and equal-sized distributions, soft-matching may equal
        one-to-one (both give optimal hard assignment). Sinkhorn approximation may
        violate the theoretical ordering property soft-matching ≤ one-to-one ≤ procrustes.
    reg : float, default=0.1
        Entropic regularization parameter for Sinkhorn algorithm.
        Higher values lead to more uniform (diffuse) transport plans.
        Only used when approx=True.

    Returns
    -------
    distance : float
        Square root of the optimal transport cost (Wasserstein-like distance).
        Lower values indicate more similar point distributions.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping point pairs (i, j) to transport probabilities.
        Unlike hard matching, multiple pairs can involve the
        same point i or j (soft assignment).
        Values represent the fraction of mass transported from point i to point j.

    Raises
    ------
    ImportError
        If the POT (Python Optimal Transport) library is not installed.

    Notes
    -----
    Unlike hard one-to-one matching, optimal transport allows fractional
    assignment of mass between points, providing a smoother distance metric.
    This is particularly useful when point clouds have different sizes or when
    you want a continuous, differentiable distance measure.

    Matrices are normalized to unit Frobenius norm before comparison.
    Points are treated as uniform distributions (equal mass at each point).

    Requires the `pot` package: pip install pot

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(60, 10)  # Different size OK
    >>> dist, pairs = shape_distance_soft_matching(mtx1, mtx2, approx=True)
    >>> print(f"Wasserstein distance: {dist:.3f}")
    >>> print(f"Number of significant transport pairs: {len(pairs)}")
    >>> # Check transport probabilities sum to ~1
    >>> print(f"Total transport mass: {sum(pairs.values()):.3f}")
    """
    if not OT_AVAILABLE:
        msg = "soft-matching requires the 'pot' library. Install with: pip install pot"
        raise ImportError(msg)

    m1 = modify_matrix(mtx1, whiten=False, normalize=True)
    m2 = modify_matrix(mtx2, whiten=False, normalize=True)

    # Compute cost matrix
    cost_matrix = cdist(m1, m2, metric=metric)

    # Uniform distributions
    a = np.ones(m1.shape[0]) / m1.shape[0]
    b = np.ones(m2.shape[0]) / m2.shape[0]

    # Compute optimal transport plan
    if approx:
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg)
    else:
        transport_plan = ot.emd(a, b, cost_matrix)
    
    # Compute distance: sum of transport plan * cost matrix
    # For sqeuclidean metric: cost_matrix contains squared distances
    #   - Do NOT take sqrt to maintain consistency with one-to-one and Procrustes
    #   - All three methods should use squared distances for comparability
    #   - This ensures: soft-matching ≤ one-to-one ≤ procrustes
    # For euclidean metric: cost_matrix contains regular distances
    #   - No sqrt needed (Wasserstein-1 distance)
    # Note: The theoretical property soft-matching ≤ one-to-one ≤ procrustes
    # holds because soft assignment is more flexible than hard assignment,
    # which is more flexible than fixed correspondence (Procrustes).
    distance = np.sum(transport_plan * cost_matrix)

    threshold = 1e-9
    i_idx, j_idx = np.where(transport_plan > threshold)
    pairs = {
        (int(i), int(j)): float(transport_plan[i, j])
        for i, j in zip(i_idx, j_idx, strict=False)
    }
    return float(distance), pairs

RunResult = TypedDict('RunResult', {
    "output": Tuple[float, Dict[Tuple[int, int], float]],
    "metadata": Dict[str, Dict[int, npt.NDArray[np.int_]]]
})

def run_with_subsampling(
    func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                   Tuple[float, Dict[Tuple[int, int], float]]],
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    subsamples: Sequence[int],
    subsample_axes: Sequence[int],
    repeats: int = 10,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Run `func` multiple times with random subsampling along given axes."""
    rng = np.random.default_rng(seed)

    if len(subsamples) != len(subsample_axes):
        raise ValueError("`subsamples` and `subsample_axes` must have same length")

    runs: List[Dict[str, Any]] = []

    for _ in range(repeats):
        indexers1: Dict[int, npt.NDArray[np.int_]] = {}
        indexers2: Dict[int, npt.NDArray[np.int_]] = {}

        for s, axis in zip(subsamples, subsample_axes, strict=False):
            n1, n2 = mtx1.shape[axis], mtx2.shape[axis]
            size = min(s, n1, n2)
            idx1 = rng.choice(n1, size=size, replace=False)
            idx2 = rng.choice(n2, size=size, replace=False)
            indexers1[axis] = idx1
            indexers2[axis] = idx2

        idx1_list = [indexers1.get(ax, slice(None)) for ax in range(mtx1.ndim)]
        idx2_list = [indexers2.get(ax, slice(None)) for ax in range(mtx2.ndim)]
        sub_mtx1 = mtx1[tuple(idx1_list)]
        sub_mtx2 = mtx2[tuple(idx2_list)]

        output = func(sub_mtx1, sub_mtx2)

        metadata = {"indices": (idx1_list, idx2_list)}
        runs.append({"output": output, "metadata": metadata})

    return runs

def shape_distance(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "procrustes",
    metric: str = "sqeuclidean",
    subsamples: Sequence[int] | None = None,
    subsample_axes: Sequence[int] | None = None,
    repeats: int = 10,
    seed: int | None = None,
    **method_kwargs: Any,  # Accept Any for now, validated at runtime
) -> Any:
    """Compute shape distance between two matrices.

    Unified interface for multiple shape comparison methods. Delegates to
    specific method functions.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1.
    method : {'procrustes', 'one-to-one', 'soft-matching'}, default='procrustes'
        Shape comparison method:
        - 'procrustes': Optimal orthogonal alignment (rotation/reflection).
            Preserves point correspondence, best for aligned data.
        - 'one-to-one': Optimal hard assignment (Hungarian algorithm).
            Permutation-invariant, finds best bijective matching.
        - 'soft-matching': Optimal transport with soft assignment.
            Allows fractional matching, handles different point cloud sizes.
    metric : str, default='sqeuclidean'
        Distance metric for 'one-to-one' and 'soft-matching' methods.
        Ignored for 'procrustes' method.
        Use 'sqeuclidean' for comparability with Procrustes (which uses squared distances).
    **method_kwargs
        Additional keyword arguments passed to the specific method:
        - For 'soft-matching': approx (bool), reg (float)

    return_pairs : bool, default=False
        If True, return both distance and pair information.
        If False, return only the distance value.

    Returns
    -------
    distance : float
        Shape distance between the two matrices. Lower values indicate
        more similar shapes. Scale depends on the method used.
        Only returned if return_pairs=False.
    (distance, pairs) : tuple[float, dict]
        If return_pairs=True, returns both distance and point correspondence:
        - For 'procrustes': {(i, i): distance} - aligned point distances
        - For 'one-to-one': {(i, j): distance} - optimal matching pairs
        - For 'soft-matching': {(i, j): probability} - transport probabilities

    Raises
    ------
    ValueError
        If an unknown method is specified.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> mtx1 = rng.randn(50, 10)
    >>> mtx2 = rng.randn(50, 10)

    >>> # Procrustes alignment
    >>> dist, pairs = shape_distance(mtx1, mtx2, method='procrustes')
    >>> print(f"Procrustes distance: {dist:.3f}")

    >>> # Optimal matching
    >>> dist, pairs = shape_distance(mtx1, mtx2, method='one-to-one',
    ...                               metric='euclidean')

    >>> # Soft optimal transport (can handle different sizes)
    >>> mtx3 = rng.randn(60, 10)
    >>> dist, pairs = shape_distance(mtx1, mtx3, method='soft-matching',
    ...                               approx=True, reg=0.05)
    """
    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")

    def core_compute(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> Tuple[float, Dict[Tuple[int, int], float]]:
        match method:
            case "procrustes":
                return shape_distance_procrustes(a, b)
            case "one-to-one":
                return shape_distance_one_to_one(a, b, metric=metric)
            case "soft-matching":
                return shape_distance_soft_matching(a, b, metric=metric, **method_kwargs)
            case _:
                raise ValueError(f"Unknown method '{method}'.")

    metadata: Dict[str, Any] = {
        "method": method,
        "metric": metric,
        "mtx1_shape": mtx1.shape,
        "mtx2_shape": mtx2.shape,
    }
    
    # Exact shape match = single run
    if mtx1.shape == mtx2.shape:
        dist, pairs = core_compute(mtx1, mtx2)
        metadata["runs"] = 1
        return dist, pairs, metadata
    
    # Auto-configure subsampling when shapes differ
    if subsamples is None:
        subsamples = [min(mtx1.shape[i], mtx2.shape[i]) for i in range(mtx1.ndim)]
    if subsample_axes is None:
        subsample_axes = list(range(mtx1.ndim))
    
    if len(subsamples) != len(subsample_axes):
        raise ValueError("subsamples and subsample_axes must have same length")
    
    runs = run_with_subsampling(
        func=core_compute,
        mtx1=mtx1,
        mtx2=mtx2,
        subsamples=subsamples,
        subsample_axes=subsample_axes,
        repeats=repeats,
        seed=seed,
    )
    
    distances = [run["output"][0] for run in runs]
    pairs_list = [run["output"][1] for run in runs]
    metadata["runs"] = len(runs)
    metadata["indices"] = [run["metadata"]["indices"] for run in runs]
    
    return distances, pairs_list, metadata


def main() -> None:
    np.random.seed(42)

    bins = 30 #256
    n_samples = bins * bins          # fixed number of samples (columns)
    neuron_range = (10, 100)     # variable number of neurons (rows)
    n_pairs = 20

    methods: list[tuple[str, dict[str, Any]]] = [
        ("procrustes",               {}),
        ("one-to-one",               {}),
        ("soft-matching-subsampling", {"approx": False}),
        ("soft-matching-exact",      {"approx": False}),
        ("soft-matching-approx",     {"approx": True, "reg": 0.1}),
    ]

    test_cases: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for _ in range(n_pairs):
        n_neurons_1 = np.random.randint(neuron_range[0], neuron_range[1])
        while True:
            n_neurons_2 = np.random.randint(neuron_range[0], neuron_range[1])
            if n_neurons_2 != n_neurons_1:
                break
        mtx1 = np.random.randn(n_neurons_1, n_samples)
        mtx2 = np.random.randn(n_neurons_2, n_samples)
        test_cases.append((mtx1, mtx2))

    all_results: list[dict[str, Any]] = []

    print(
        f"Testing over {n_pairs} pairs: n_samples={n_samples}, "
        f"neurons in [{neuron_range[0]}, {neuron_range[1]})"
    )

    start_total = time.perf_counter()

    for i, (mtx1, mtx2) in enumerate(test_cases, start=1):
        print(f"\nPair {i}/{n_pairs}: mtx1.shape={mtx1.shape}, mtx2.shape={mtx2.shape}")
        pair_results: dict[str, Any] = {}

        for label, extra_kwargs in methods:
            if label == "procrustes":
                core_method = "procrustes"
                use_subsampling = mtx1.shape != mtx2.shape
            elif label == "one-to-one":
                core_method = "one-to-one"
                use_subsampling = mtx1.shape != mtx2.shape
            elif label == "soft-matching-subsampling":
                core_method = "soft-matching"
                use_subsampling = mtx1.shape != mtx2.shape
            elif label in {"soft-matching-exact", "soft-matching-approx"}:
                core_method = "soft-matching"
                use_subsampling = False
            else:
                raise ValueError(f"Unknown label {label}")

            t0 = time.perf_counter()

            if use_subsampling:
                # Only subsample neurons (rows, axis 0)
                subsample_axes = [0]
                subsamples = [min(mtx1.shape[0], mtx2.shape[0])]
                distances, pairs_list, meta = shape_distance(
                    mtx1,
                    mtx2,
                    method=core_method,
                    metric="sqeuclidean",
                    subsamples=subsamples,
                    subsample_axes=subsample_axes,
                    repeats=10,
                    seed=42 + i,
                    **extra_kwargs,
                )
            else:
                distances, pairs_list, meta = shape_distance(
                    mtx1,
                    mtx2,
                    method=core_method,
                    metric="sqeuclidean",
                    subsamples=None,
                    subsample_axes=None,
                    repeats=1,
                    seed=42 + i,
                    **extra_kwargs,
                )

            dt = time.perf_counter() - t0

            if isinstance(distances, float):
                dist_summary = distances
                n_runs = 1
            else:
                dist_summary = float(np.mean(distances))
                n_runs = len(distances)

            print(
                f"  {label:24s}: dist={dist_summary:.6f}  "
                f"(runs={n_runs})  time={dt:.4f}s"
            )

            pair_results[label] = {
                "distances": distances,
                "pairs_list": pairs_list,
                "meta": meta,
                "time": dt,
            }

        all_results.append(pair_results)

    total_time = time.perf_counter() - start_total

    print("\n" + "=" * 60)
    print(f"SUMMARY (Total time: {total_time:.2f}s)")
    print("=" * 60)

    method_labels = [label for label, _ in methods]
    mean_dists = np.full((n_pairs, len(methods)), np.nan, dtype=float)

    for i, results in enumerate(all_results):
        for j, label in enumerate(method_labels):
            if label in results:
                d = results[label]["distances"]
                if isinstance(d, float):
                    mean_dists[i, j] = d
                else:
                    mean_dists[i, j] = float(np.mean(d))

    print("\nAverage distances across pairs:")
    for j, label in enumerate(method_labels):
        col = mean_dists[:, j]
        col = col[~np.isnan(col)]
        if col.size > 0:
            print(
                f"  {label:24s}: {np.mean(col):.6f} ± {np.std(col):.6f}  (n={col.size})"
            )

    print("\n`all_results` contains full per-pair, per-method statistics.")


if __name__ == "__main__":
    main()