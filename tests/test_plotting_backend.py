"""Tests for plotting backend management."""

import pytest

from neural_analysis.plotting.backend import (
    BackendType,
    get_backend,
    set_backend,
)


class TestBackendType:
    """Tests for BackendType enum."""

    def test_backend_type_values(self):
        """Test BackendType enum values."""
        assert BackendType.MATPLOTLIB.value == "matplotlib"
        assert BackendType.PLOTLY.value == "plotly"


class TestSetBackend:
    """Tests for set_backend function."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("matplotlib", BackendType.MATPLOTLIB),
            ("plotly", BackendType.PLOTLY),
            ("MATPLOTLIB", BackendType.MATPLOTLIB),
            ("PLOTLY", BackendType.PLOTLY),
            (BackendType.MATPLOTLIB, BackendType.MATPLOTLIB),
            (BackendType.PLOTLY, BackendType.PLOTLY),
        ],
        ids=[
            "str-matplotlib",
            "str-plotly",
            "str-MATPLOTLIB",
            "str-PLOTLY",
            "enum-matplotlib",
            "enum-plotly",
        ],
    )
    def test_set_backend_valid(self, input_val, expected):
        """Test setting backend with valid values."""
        set_backend(input_val)
        assert get_backend() == expected

    def test_set_backend_invalid_string(self):
        """Test setting backend with invalid string (covers lines 61-67)."""
        with pytest.raises(ValueError, match="Invalid backend"):
            set_backend("invalid_backend")

    def test_set_backend_invalid_type(self):
        """Test setting backend with invalid type (covers line 67)."""
        with pytest.raises(TypeError, match="backend must be str or BackendType"):
            set_backend(123)  # type: ignore
        with pytest.raises(TypeError, match="backend must be str or BackendType"):
            set_backend(None)  # type: ignore


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_backend_default(self):
        """Test getting default backend."""
        # Reset to default
        set_backend("matplotlib")
        backend = get_backend()
        assert backend == BackendType.MATPLOTLIB

    def test_get_backend_after_set(self):
        """Test getting backend after setting it."""
        set_backend("plotly")
        backend = get_backend()
        assert backend == BackendType.PLOTLY
        set_backend("matplotlib")
        backend = get_backend()
        assert backend == BackendType.MATPLOTLIB
