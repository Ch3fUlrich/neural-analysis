"""Coverage tests for neural_analysis.utils.reproducibility."""

from __future__ import annotations

import numpy as np

from neural_analysis.utils.reproducibility import (
    _get_version,
    get_provenance,
    reproducible,
)


def test_reproducible_is_deterministic() -> None:
    with reproducible(seed=123) as rng:
        a = rng.normal(size=5)
    with reproducible(seed=123) as rng:
        b = rng.normal(size=5)
    assert np.allclose(a, b)


def test_reproducible_restores_global_numpy_state() -> None:
    np.random.seed(7)
    before = np.random.get_state()[1][:5].copy()
    with reproducible(seed=999):
        np.random.random()
    after = np.random.get_state()[1][:5]
    assert np.array_equal(before, after)


def test_get_version_known_package() -> None:
    assert _get_version("numpy") != "unknown"


def test_get_version_unknown_package_returns_unknown() -> None:
    # Exercises the except branch (lines 75-76).
    assert _get_version("definitely-not-a-real-package-xyz-123") == "unknown"


def test_get_provenance_has_expected_string_keys() -> None:
    prov = get_provenance()
    for key in (
        "library_version",
        "python_version",
        "platform",
        "timestamp_utc",
        "numpy_version",
        "scipy_version",
    ):
        assert key in prov
        assert isinstance(prov[key], str)
