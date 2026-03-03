"""Tests for reproducibility utilities."""

from __future__ import annotations

import numpy as np

from neural_analysis.utils.reproducibility import get_provenance, reproducible


class TestReproducible:
    def test_yields_generator(self) -> None:
        with reproducible(seed=42) as rng:
            assert isinstance(rng, np.random.Generator)

    def test_deterministic_output(self) -> None:
        with reproducible(seed=123) as rng:
            a = rng.normal(0, 1, size=100)
        with reproducible(seed=123) as rng:
            b = rng.normal(0, 1, size=100)
        np.testing.assert_array_equal(a, b)

    def test_restores_numpy_state(self) -> None:
        np.random.seed(999)
        before = np.random.random()

        np.random.seed(999)
        with reproducible(seed=0) as _rng:
            _ = np.random.random()
        after = np.random.random()
        assert before == after

    def test_restores_python_state(self) -> None:
        import random

        random.seed(999)
        before = random.random()

        random.seed(999)
        with reproducible(seed=0) as _rng:
            _ = random.random()
        after = random.random()
        assert before == after


class TestProvenance:
    def test_keys_present(self) -> None:
        prov = get_provenance()
        expected_keys = {
            "library_version",
            "python_version",
            "platform",
            "timestamp_utc",
            "numpy_version",
            "scipy_version",
        }
        assert expected_keys <= set(prov.keys())

    def test_library_version(self) -> None:
        prov = get_provenance()
        assert prov["library_version"] == "0.1.0"

    def test_values_are_strings(self) -> None:
        prov = get_provenance()
        for v in prov.values():
            assert isinstance(v, str)
