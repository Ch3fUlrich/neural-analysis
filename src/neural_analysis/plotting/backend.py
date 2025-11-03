"""
Backend selection and management for the plotting package.

This module provides a flexible backend system that allows switching between
matplotlib and plotly for different visualization needs. Matplotlib is better
for static, publication-quality figures, while Plotly excels at interactive
visualizations.

Example:
    >>> from neural_analysis.plotting import set_backend, get_backend
    >>> set_backend('matplotlib')
    >>> current = get_backend()
    >>> print(current)
    BackendType.MATPLOTLIB
"""

from enum import Enum
from typing import Literal

__all__ = ["BackendType", "set_backend", "get_backend"]


class BackendType(Enum):
    """Enumeration of supported visualization backends."""
    
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"


# Global backend state
_current_backend: BackendType = BackendType.MATPLOTLIB


def set_backend(backend: Literal["matplotlib", "plotly"] | BackendType) -> None:
    """
    Set the visualization backend.
    
    Parameters
    ----------
    backend : {'matplotlib', 'plotly'} or BackendType
        The backend to use for subsequent visualizations.
        
    Raises
    ------
    ValueError
        If an invalid backend name is provided.
        
    Examples
    --------
    >>> set_backend('matplotlib')
    >>> set_backend(BackendType.PLOTLY)
    >>> set_backend('plotly')
    """
    global _current_backend
    
    if isinstance(backend, BackendType):
        _current_backend = backend
    elif isinstance(backend, str):
        try:
            _current_backend = BackendType(backend.lower())
        except ValueError:
            valid_backends = [b.value for b in BackendType]
            raise ValueError(
                f"Invalid backend '{backend}'. Must be one of {valid_backends}"
            ) from None
    else:
        raise TypeError(
            f"backend must be str or BackendType, got {type(backend).__name__}"
        )


def get_backend() -> BackendType:
    """
    Get the current visualization backend.
    
    Returns
    -------
    BackendType
        The currently active backend.
        
    Examples
    --------
    >>> backend = get_backend()
    >>> print(backend)
    BackendType.MATPLOTLIB
    """
    return _current_backend
