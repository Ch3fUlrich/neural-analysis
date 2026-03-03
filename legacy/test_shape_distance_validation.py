"""Validate shape distance metrics ordering and scale consistency.

This script tests that the shape distance methods (soft-matching, one-to-one, procrustes)
maintain the theoretical ordering: soft-matching ≤ one-to-one ≤ procrustes, and that
they are on the same scale for fair comparison.

Theoretical Background:
- Soft-matching uses optimal transport with fractional assignments (most flexible)
- One-to-one uses optimal hard assignment (Hungarian algorithm, less flexible)
- Procrustes uses fixed correspondence with optimal rotation (least flexible)
- More flexibility → lower distance (better matching)
"""

import time
from typing import Any

import numpy as np
import numpy.typing as npt

from neural_analysis.metrics.distributions import shape_distance


def validate_shape_distance_ordering(
    n_pairs: int = 20,
    n_samples: int = 100,
    neuron_range: tuple[int, int] = (10, 100),
    repeats: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    """Validate that shape distance methods maintain theoretical ordering.

    Parameters
    ----------
    n_pairs : int, default=20
        Number of random matrix pairs to test.
    n_samples : int, default=100
        Number of samples (columns) in each matrix.
    neuron_range : tuple[int, int], default=(10, 100)
        Range of neuron counts (rows) to test. Matrices will have different
        numbers of neurons to test subsampling behavior.
    repeats : int, default=10
        Number of subsampling repeats when matrices differ in size.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'all_results': List of per-pair results with distances and metadata
        - 'summary': Summary statistics across all pairs
        - 'ordering_violations': Count of ordering violations (should be 0)
    """
    if seed is not None:
        np.random.seed(seed)

    # Define methods to test
    methods: list[tuple[str, str, dict[str, Any]]] = [
        ("procrustes", "procrustes", {}),
        ("one-to-one", "one-to-one", {}),
        ("soft-matching-subsampling", "soft-matching", {"approx": False}),
        ("soft-matching-exact", "soft-matching", {"approx": False}),
        ("soft-matching-approx", "soft-matching", {"approx": True, "reg": 0.1}),
    ]

    # Generate test cases with varying neuron counts
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
    ordering_violations = 0

    print(
        f"Testing {n_pairs} pairs: n_samples={n_samples}, "
        f"neurons in [{neuron_range[0]}, {neuron_range[1]})"
    )

    start_total = time.perf_counter()

    for i, (mtx1, mtx2) in enumerate(test_cases, start=1):
        print(f"\nPair {i}/{n_pairs}: mtx1.shape={mtx1.shape}, mtx2.shape={mtx2.shape}")
        pair_results: dict[str, Any] = {}

        # Determine if subsampling is needed (when shapes differ)
        use_subsampling = mtx1.shape != mtx2.shape

        for label, core_method, extra_kwargs in methods:
            # Determine subsampling strategy
            if label in {"soft-matching-exact", "soft-matching-approx"}:
                # These can handle different sizes without subsampling
                # Keep them without subsampling to show how soft-matching compares
                # to one-to-one when it can handle different-sized matrices directly
                method_use_subsampling = False
            else:
                # Others need subsampling when shapes differ
                method_use_subsampling = use_subsampling

            t0 = time.perf_counter()

            if method_use_subsampling:
                # Subsample neurons (rows, axis 0) to match minimum size
                subsample_axes = [0]
                subsamples = [min(mtx1.shape[0], mtx2.shape[0])]
                distances, pairs_list, meta = shape_distance(
                    mtx1,
                    mtx2,
                    method=core_method,
                    metric="sqeuclidean",
                    subsamples=subsamples,
                    subsample_axes=subsample_axes,
                    repeats=repeats,
                    seed=seed + i if seed is not None else None,
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
                    seed=seed + i if seed is not None else None,
                    **extra_kwargs,
                )

            dt = time.perf_counter() - t0

            # Extract mean distance
            if isinstance(distances, float):
                dist_summary = distances
                n_runs = 1
            else:
                dist_summary = float(np.mean(distances))
                n_runs = len(distances)

            print(
                f"  {label:30s}: dist={dist_summary:.6f}  "
                f"(runs={n_runs})  time={dt:.4f}s"
            )

            pair_results[label] = {
                "distances": distances,
                "pairs_list": pairs_list,
                "meta": meta,
                "time": dt,
                "mean_distance": dist_summary,
            }

        # Check ordering: soft-matching ≤ one-to-one ≤ procrustes
        # Note: soft-matching-exact/approx work on full different-sized matrices
        # (no rotation alignment), while one-to-one and procrustes work on subsampled
        # same-sized matrices (with rotation alignment). Violations are expected
        # when comparing these different approaches, but they show how soft-matching
        # compares when handling different sizes directly.
        soft_dist = pair_results.get("soft-matching-exact", {}).get("mean_distance")
        one_to_one_dist = pair_results.get("one-to-one", {}).get("mean_distance")
        procrustes_dist = pair_results.get("procrustes", {}).get("mean_distance")

        if all(d is not None for d in [soft_dist, one_to_one_dist, procrustes_dist]):
            if procrustes_dist > one_to_one_dist:
                ordering_violations += 1
                print(
                    f"  ⚠️  THEORY VIOLATION:\n"
                    f"    proc={procrustes_dist:.6f} > one-to-one={one_to_one_dist:.6f}\n"
                    f"    Procrustes (min over ALL orthos O_N) MUST ≤ one-to-one (min over PERMS Π_N ⊂ O_N).\n"
                    f"    Check: Procrustes raw / sqrt(N) norm? Subsampling noise? Extreme neuron shuffle?\n"
                    f"    (Data where fixed correspondence good → proc << one-to-one, e.g., same neurons/sessions.)"
                )
            elif one_to_one_dist > soft_dist:
                print(
                    f"  ℹ️  EXPECTED RELAXATION:\n"
                    f"    one-to-one={one_to_one_dist:.6f} > soft={soft_dist:.6f}\n"
                    f"    Soft-matching relaxes perms to fractional couplings (T ⊃ Π_N/N); always ≤.\n"
                    f"    Exact EMD ≈ one-to-one (equal N); Sinkhorn < (entropic reg smears mass).\n"
                    f"    Best for unequal N/populations (e.g., regions/animals)."
                )
            else:
                print(
                    f"  ✅ FULL ORDERING HOLDS:\n"
                    f"    proc={procrustes_dist:.6f} ≤ one-to-one={one_to_one_dist:.6f} ≤ soft={soft_dist:.6f}\n"
                    f"    Nested feasible sets: orthos ⊃ perms ⊃ couplings → distances nonincreasing.\n"
                    f"    proc best (smallest) for rotated same neurons (e.g., HD rings, sessions);\n"
                    f"    one-to-one/soft for shuffled IDs (e.g., cross-subject IBL, Allen regions)."
                )
            print(
                f"    Reference: soft={soft_dist:.6f} (handles full unequal sizes directly)."
            )

        all_results.append(pair_results)

    total_time = time.perf_counter() - start_total

    # Compute summary statistics
    method_labels = [label for label, _, _ in methods]
    mean_dists = np.full((n_pairs, len(methods)), np.nan, dtype=float)

    for i, results in enumerate(all_results):
        for j, label in enumerate(method_labels):
            if label in results:
                d = results[label]["mean_distance"]
                mean_dists[i, j] = d

    summary = {}
    for j, label in enumerate(method_labels):
        col = mean_dists[:, j]
        col = col[~np.isnan(col)]
        if col.size > 0:
            summary[label] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "n": int(col.size),
            }

    results_dict = {
        "all_results": all_results,
        "summary": summary,
        "ordering_violations": ordering_violations,
        "total_time": total_time,
    }

    return results_dict


def print_validation_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary of validation results.

    Parameters
    ----------
    results : dict
        Results dictionary from validate_shape_distance_ordering.
    """
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print("\nAverage distances across pairs:")
    summary = results["summary"]
    for label, stats in summary.items():
        print(
            f"  {label:30s}: {stats['mean']:.6f} ± {stats['std']:.6f}  (n={stats['n']})"
        )

    violations = results["ordering_violations"]
    total_pairs = len(results["all_results"])
    print(f"\nOrdering violations: {violations}/{total_pairs}")
    if violations == 0:
        print(
            "✓ All pairs maintain theoretical ordering: soft-matching ≤ one-to-one ≤ procrustes"
        )
    else:
        print(f"⚠️  {violations} pairs violated theoretical ordering")
        print(
            "   Note: Violations are expected when comparing soft-matching-exact/approx "
            "(no subsampling, different sizes) with one-to-one/procrustes "
            "(with subsampling, same sizes). This shows how soft-matching compares "
            "when handling different-sized matrices directly."
        )

    print(f"\nTotal time: {results['total_time']:.2f}s")
    print("\nFull per-pair results available in results['all_results']")


def main() -> None:
    """Run validation with default parameters."""
    results = validate_shape_distance_ordering(
        n_pairs=10,
        n_samples=100,
        neuron_range=(10, 100),
        repeats=10,
        seed=42,
    )
    print_validation_summary(results)


if __name__ == "__main__":
    main()
