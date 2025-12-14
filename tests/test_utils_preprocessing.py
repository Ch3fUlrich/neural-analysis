"""Tests for preprocessing utilities (deprecated shim)."""

import pytest

from neural_analysis.utils import preprocessing


class TestPreprocessing:
    """Test preprocessing module (deprecated shim)."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        assert preprocessing is not None
        # Module is deprecated, just verify it exists
        assert hasattr(preprocessing, "__doc__")



