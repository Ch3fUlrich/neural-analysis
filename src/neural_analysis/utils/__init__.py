"""Utility subpackage for neural_analysis.

This package provides IO, validation, array, and preprocessing utilities used
across manifold analysis, metrics, and plotting modules.
"""

__all__ = [
    "save_array",
    "load_array",
    "update_array",
    "h5io",
    "save_hdf5",
    "load_hdf5",
    "do_critical",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
