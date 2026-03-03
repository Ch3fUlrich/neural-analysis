from numpy._typing._array_like import NDArray
from numpy import int_
import time
import numpy as np
from numpy.typing import NDArray
import numpy.typing as npt
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypedDict, Union, Literal

from neural_analysis.metrics.distributions import shape_distance
from typing import Any, Callable, Dict, List, Sequence, Tuple
import numpy as np
import numpy.typing as npt


def main() -> None:
    np.random.seed(42)

    bins = 10  # 256
    n_samples = bins * bins  # fixed number of samples (columns)
    neuron_range = (10, 100)  # variable number of neurons (rows)
    n_pairs = 20
    repeats = 10

    methods: list[tuple[str, dict[str, Any]]] = [
        ("procrustes", {}),
        ("one-to-one", {}),
        ("soft-matching-subsampling", {"approx": False}),
        ("soft-matching-exact", {"approx": False}),
        ("soft-matching-approx", {"approx": True, "reg": 0.1}),
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
                    repeats=repeats,
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
