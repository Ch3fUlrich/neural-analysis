"""Subsampling utilities for neural data analysis.

This module provides functions for repeated random subsampling of arrays,
useful for robust comparisons when datasets differ in size.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import numpy.typing as npt


def run_with_subsampling(
    func: Callable[..., float],
    arrays: Sequence[npt.NDArray[np.float64]],
    subsamples: Sequence[int],
    subsample_axes: Sequence[int],
    repeats: int = 10,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], Dict[str, Any]]:
    """Run `func` multiple times with random subsampling along given axes.

    This function repeatedly draws random subsets from input arrays along specified
    axes, applies the function to each subset, and returns all results together
    with detailed indexing metadata. Useful for robust comparisons when datasets
    differ in size.

    Parameters
    ----------
    func : callable
        Callable that accepts len(arrays) ndarrays (all already subsampled)
        and returns a single float (e.g. distance).
    arrays : sequence of ndarray
        Sequence of input arrays to be subsampled jointly. All arrays must
        have the same ndim and be indexable along the subsample_axes.
    subsamples : sequence of int
        Sequence of subsample sizes, one per axis in `subsample_axes`.
    subsample_axes : sequence of int
        Axes along which to subsample. Same length as `subsamples`.
    repeats : int, default=10
        Number of independent subsampling runs.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    values : ndarray of shape (repeats,)
        1D array of length `repeats` containing the per-run outputs.
    metadata : dict
        Dict containing per-run indices used for each array:
        {
            "indices": [
                [ {axis: idx_array}, {axis: idx_array}, ... ],  # run 0
                [ {axis: idx_array}, {axis: idx_array}, ... ],  # run 1
                ...
            ]
        }
        where the inner list is one entry per array.

    Examples
    --------
    >>> import numpy as np
    >>> def euclidean_dist(a, b):
    ...     return np.linalg.norm(a - b)
    >>> arr1 = np.random.randn(100, 50)
    >>> arr2 = np.random.randn(80, 50)
    >>> values, meta = run_with_subsampling(
    ...     func=euclidean_dist,
    ...     arrays=(arr1, arr2),
    ...     subsamples=[70],  # Subsample to 70 rows
    ...     subsample_axes=[0],  # Along axis 0 (rows)
    ...     repeats=5,
    ...     seed=42
    ... )
    >>> len(values)
    5
    """
    if not arrays:
        raise ValueError("`arrays` must contain at least one array")

    ndims = {arr.ndim for arr in arrays}
    if len(ndims) != 1:
        raise ValueError("All arrays must have the same number of dimensions")
    ndim = arrays[0].ndim

    if len(subsamples) != len(subsample_axes):
        raise ValueError("`subsamples` and `subsample_axes` must have same length")

    rng = np.random.default_rng(seed)

    values: List[float] = []
    all_indices: List[List[Dict[int, npt.NDArray[np.int_]]]] = []

    for _ in range(repeats):
        # For each array, keep a dict axis -> indices
        per_array_indexers: List[Dict[int, npt.NDArray[np.int_]]] = []

        for arr in arrays:
            indexers: Dict[int, npt.NDArray[np.int_]] = {}
            for s, axis in zip(subsamples, subsample_axes, strict=False):
                n = arr.shape[axis]
                size = min(s, n)
                idx = rng.choice(n, size=size, replace=False)
                indexers[axis] = idx
            per_array_indexers.append(indexers)

        # Build actual subsampled arrays
        sub_arrays: List[npt.NDArray[np.float64]] = []
        for arr, indexers in zip(arrays, per_array_indexers, strict=False):
            idx_list = [indexers.get(ax, slice(None)) for ax in range(ndim)]
            sub_arr = arr[tuple(idx_list)]
            sub_arrays.append(sub_arr)

        # Compute output
        value = func(*sub_arrays)
        values.append(float(value))
        all_indices.append(per_array_indexers)

    values_array = np.asarray(values, dtype=float)
    metadata: Dict[str, Any] = {"indices": all_indices}
    return values_array, metadata

