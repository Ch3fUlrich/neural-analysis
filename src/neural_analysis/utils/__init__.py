"""Utility subpackage for neural_analysis.

This package provides IO, validation, array, preprocessing utilities, and
computational functions (trajectories, geometry) used across manifold analysis,
metrics, and plotting modules.
"""

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
    # Trajectory utilities
    "prepare_trajectory_segments",
    "compute_colors",
    # Geometry utilities
    "compute_convex_hull",
    "compute_kde_2d",
]

from .io import (
    save_array,
    load_array,
    update_array,
    h5io,
    save_hdf5,
    load_hdf5,
)
import importlib


def __getattr__(name: str):
    if name == "do_critical":
        mod = importlib.import_module("neural_analysis.utils.validation")
        return getattr(mod, name)
    if name in {"configure_logging", "get_logger", "log_section", "log_kv", "log_calls"}:
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
