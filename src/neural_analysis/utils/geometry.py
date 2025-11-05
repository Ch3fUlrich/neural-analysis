"""
Geometric computation utilities for neural data analysis.

This module provides functions for computing geometric properties and
transformations, such as convex hulls, density estimates, and spatial statistics.
"""

import numpy as np
from typing import Tuple


def compute_convex_hull(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Compute the convex hull boundary points for 2D data.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates (1D array)
    y : np.ndarray
        Y coordinates (1D array)
        
    Returns
    -------
    hull_x, hull_y : tuple of np.ndarray or None
        Convex hull boundary coordinates in order.
        Returns None if hull cannot be computed (e.g., < 3 points or collinear).
        
    Example
    -------
    >>> x = np.array([0, 1, 0.5, 0.25])
    >>> y = np.array([0, 0, 1, 0.5])
    >>> hull_x, hull_y = compute_convex_hull(x, y)
    >>> # hull_x, hull_y contain the boundary points
    """
    if len(x) < 3 or len(y) < 3:
        return None
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    try:
        from scipy.spatial import ConvexHull
        
        points = np.column_stack([x, y])
        hull = ConvexHull(points)
        
        # Get hull vertices in order
        hull_points = points[hull.vertices]
        hull_x = hull_points[:, 0]
        hull_y = hull_points[:, 1]
        
        # Close the hull by adding first point at end
        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])
        
        return hull_x, hull_y
        
    except Exception:
        # Hull computation can fail for collinear points or other edge cases
        return None


def compute_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
    grid_size: int = 100,
    expand_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D kernel density estimation on a grid.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates (1D array)
    y : np.ndarray
        Y coordinates (1D array)
    bandwidth : float, optional
        KDE bandwidth. If None, uses Scott's rule.
    grid_size : int, default=100
        Number of points in each dimension of evaluation grid
    expand_fraction : float, default=0.1
        Fraction to expand grid beyond data range
        
    Returns
    -------
    xi : np.ndarray
        X coordinates of grid (1D array)
    yi : np.ndarray
        Y coordinates of grid (1D array)
    zi : np.ndarray
        Density values on grid (2D array, shape: (grid_size, grid_size))
        
    Example
    -------
    >>> x = np.random.randn(500)
    >>> y = np.random.randn(500)
    >>> xi, yi, zi = compute_kde_2d(x, y)
    >>> # Use xi, yi, zi for contour plotting
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 points for KDE")
    
    from scipy.stats import gaussian_kde
    
    # Calculate KDE
    values = np.vstack([x, y])
    kernel = gaussian_kde(values, bw_method=bandwidth)
    
    # Create grid for evaluation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    xi = np.linspace(
        x_min - expand_fraction * x_range,
        x_max + expand_fraction * x_range,
        grid_size
    )
    yi = np.linspace(
        y_min - expand_fraction * y_range,
        y_max + expand_fraction * y_range,
        grid_size
    )
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Evaluate KDE on grid
    zi = kernel(np.vstack([xi_grid.flatten(), yi_grid.flatten()]))
    zi = zi.reshape(xi_grid.shape)
    
    return xi, yi, zi
