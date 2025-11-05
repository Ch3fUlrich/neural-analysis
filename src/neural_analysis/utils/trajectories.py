"""
Trajectory computation utilities for neural data analysis.

This module provides functions for processing and analyzing trajectory data,
including time-based segmentation and color mapping for visualization.
"""

import numpy as np
from typing import Optional, Literal


def prepare_trajectory_segments(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Prepare 2D or 3D trajectory data as line segments for visualization.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates (1D array, length n)
    y : np.ndarray
        Y coordinates (1D array, length n)
    z : np.ndarray, optional
        Z coordinates (1D array, length n). If provided, creates 3D segments.
        
    Returns
    -------
    segments : np.ndarray
        Line segments array with shape (n-1, 2, 2) for 2D or (n-1, 2, 3) for 3D
        Each segment connects consecutive points
        
    Examples
    --------
    >>> # 2D trajectory
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([0, 1, 0, 1])
    >>> segments = prepare_trajectory_segments(x, y)
    >>> # segments.shape = (3, 2, 2) - three line segments
    
    >>> # 3D trajectory
    >>> z = np.array([0, 0.5, 1, 1.5])
    >>> segments = prepare_trajectory_segments(x, y, z)
    >>> # segments.shape = (3, 2, 3) - three line segments
    """
    if z is None:
        # 2D trajectory
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
        
        if len(x) < 2:
            raise ValueError("Need at least 2 points for trajectory")
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    else:
        # 3D trajectory
        if not (len(x) == len(y) == len(z)):
            raise ValueError(
                f"x, y, z must have same length, got {len(x)}, {len(y)}, {len(z)}"
            )
        
        if len(x) < 2:
            raise ValueError("Need at least 2 points for trajectory")
        
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


def compute_colors(
    n_points: int,
    color_by: Literal["time"] = "time"
) -> np.ndarray:
    """
    Compute color values based on specified method.
    
    Parameters
    ----------
    n_points : int
        Number of points in trajectory
    color_by : Literal["time"], default="time"
        Method for color computation. Currently only "time" is supported.
        Future options may include "speed", "direction", etc.
        
    Returns
    -------
    colors : np.ndarray
        Color values array (1D array of floats from 0 to n_points-1)
        
    Examples
    --------
    >>> colors = compute_colors(100, color_by="time")
    >>> # colors = array([0, 1, 2, ..., 99])
    """
    if n_points < 1:
        raise ValueError("Need at least 1 point")
    
    if color_by == "time":
        return np.arange(n_points)
    else:
        raise ValueError(f"Unsupported color_by method: {color_by}")
