"""Shared test fixtures for neural_analysis test suite."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.data import generate_data


@pytest.fixture(scope="session")
def place_cells_2d() -> tuple[np.ndarray, dict]:
    """Session-scoped place cell dataset (50 cells, 2000 timesteps)."""
    activity, metadata = generate_data(
        "place_cells", n_features=50, n_samples=2000, seed=42, plot=False
    )
    return activity, metadata


@pytest.fixture(scope="session")
def random_activity() -> np.ndarray:
    """Session-scoped random activity matrix (1000x50)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, size=(1000, 50))


@pytest.fixture(scope="session")
def random_labels() -> np.ndarray:
    """Session-scoped random 2D labels (1000x2)."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=(1000, 2))


@pytest.fixture(scope="function")
def small_matrix() -> np.ndarray:
    """Small matrix for quick unit tests (10x5)."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, size=(10, 5))
