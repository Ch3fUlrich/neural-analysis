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

    def test_set_backend_with_string_matplotlib(self):
        """Test setting backend with string 'matplotlib'."""
        set_backend("matplotlib")
        assert get_backend() == BackendType.MATPLOTLIB

    def test_set_backend_with_string_plotly(self):
        """Test setting backend with string 'plotly'."""
        set_backend("plotly")
        assert get_backend() == BackendType.PLOTLY

    def test_set_backend_with_string_case_insensitive(self):
        """Test setting backend with string is case-insensitive."""
        set_backend("MATPLOTLIB")
        assert get_backend() == BackendType.MATPLOTLIB
        set_backend("PLOTLY")
        assert get_backend() == BackendType.PLOTLY

    def test_set_backend_with_enum(self):
        """Test setting backend with BackendType enum (covers line 57)."""
        set_backend(BackendType.MATPLOTLIB)
        assert get_backend() == BackendType.MATPLOTLIB
        set_backend(BackendType.PLOTLY)
        assert get_backend() == BackendType.PLOTLY

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



