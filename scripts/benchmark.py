#!/usr/bin/env python3
"""Benchmark suite for neural_analysis.

Measures wall-clock performance of key operations and writes results to JSON.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --output benchmarks/latest.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from neural_analysis.data import generate_data
from neural_analysis.embeddings import compute_embedding
from neural_analysis.metrics import compute_pairwise_matrix, shape_distance
from neural_analysis.topology import compute_structure_index


def benchmark(
    name: str,
    func: Any,
    *args: Any,
    repeats: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *func* multiple times and report timing statistics."""
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        "name": name,
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural_analysis benchmarks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/latest.json"),
        help="Path for JSON output (default: benchmarks/latest.json)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per benchmark (default: 3)",
    )
    args = parser.parse_args()
    repeats: int = args.repeats
    results: list[dict[str, Any]] = []

    print("Generating test data...")
    activity, meta = generate_data(
        "place_cells", n_cells=100, n_timesteps=5000, seed=42
    )

    # --- Pairwise euclidean ---
    print("  Benchmarking: pairwise euclidean...")
    results.append(
        benchmark(
            "pairwise_euclidean_100x5000",
            compute_pairwise_matrix,
            activity,
            activity,
            "euclidean",
            repeats=repeats,
        )
    )

    # --- Shape distance (procrustes) ---
    half = activity.shape[0] // 2
    print("  Benchmarking: procrustes shape distance...")
    results.append(
        benchmark(
            "procrustes_shape_distance",
            shape_distance,
            activity[:half],
            activity[half : 2 * half],
            "procrustes",
            repeats=repeats,
        )
    )

    # --- PCA embedding ---
    print("  Benchmarking: PCA embedding...")
    results.append(
        benchmark(
            "pca_100x5000_to_3d",
            compute_embedding,
            activity,
            method="pca",
            n_components=3,
            repeats=repeats,
        )
    )

    # --- Structure Index ---
    positions = meta.get("positions") if isinstance(meta, dict) else None
    if positions is not None:
        print("  Benchmarking: structure index...")
        results.append(
            benchmark(
                "structure_index_100x5000",
                compute_structure_index,
                activity,
                positions,
                repeats=repeats,
            )
        )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}")

    # Summary table
    print(f"\n{'Benchmark':<35} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10}")
    print("-" * 67)
    for r in results:
        print(
            f"{r['name']:<35} {r['mean_s']:>10.4f} {r['std_s']:>10.4f} {r['min_s']:>10.4f}"
        )


if __name__ == "__main__":
    main()
