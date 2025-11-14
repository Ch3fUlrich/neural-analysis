"""Utility subpackage for neural_analysis.

This package provides IO, validation, array, preprocessing utilities, and
computational functions (trajectories, geometry) used across manifold analysis,
metrics, and plotting modules.
"""

from typing import Any

__all__ = [
    "save_array",
    "load_array",
    "update_array",
    "h5io",
    "save_hdf5",
    "load_hdf5",
    "do_critical",
    "configure_logging",
    "get_logger",
    "log_section",
    "log_kv",
    "log_calls",
    # Phase 3: HDF5 comparison storage
    "save_comparison",
    "load_comparison",
    "query_comparisons",
    "rebuild_index",
    # Trajectory utilities
    "prepare_trajectory_segments",
    "compute_colors",
    # Geometry utilities
    "compute_convex_hull",
    "compute_kde_2d",
]

import importlib

from .io import (
    h5io,
    load_array,
    load_hdf5,
    save_array,
    save_hdf5,
    update_array,
)


def __getattr__(name: str) -> Any:
    if name == "do_critical":
        mod = importlib.import_module("neural_analysis.utils.validation")
        return getattr(mod, name)
    if name in {
        "configure_logging",
        "get_logger",
        "log_section",
        "log_kv",
        "log_calls",
    }:
        mod = importlib.import_module("neural_analysis.utils.logging")
        return getattr(mod, name)
    # Trajectory utilities
    if name in {"prepare_trajectory_segments", "compute_colors"}:
        mod = importlib.import_module("neural_analysis.utils.trajectories")
        return getattr(mod, name)
    # Geometry utilities
    if name in {"compute_convex_hull", "compute_kde_2d"}:
        mod = importlib.import_module("neural_analysis.utils.geometry")
        return getattr(mod, name)
    # Phase 3: HDF5 comparison storage
    if name in {
        "save_comparison",
        "load_comparison",
        "query_comparisons",
        "rebuild_index",
    }:
        mod = importlib.import_module("neural_analysis.utils.comparison_store")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
