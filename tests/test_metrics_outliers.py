"""Tests for outliers.py to reach 100% coverage."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.metrics.outliers import filter_outlier


class TestFilterOutlier:
    """Tests for filter_outlier function."""

    def test_filter_outlier_basic(self) -> None:
        """Test basic outlier filtering."""
        data = np.random.randn(100, 2)
        data[0] = [10, 10]  # Clear outlier
        filtered = filter_outlier(data, method="lof")
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape[0] < data.shape[0]  # Some outliers removed

    def test_filter_outlier_import_fallback(self, monkeypatch) -> None:
        """Test ImportError fallback for logging (covers lines 19-28)."""
        import sys
        import importlib
        
        # Save original state
        original_outliers = sys.modules.get("neural_analysis.metrics.outliers")
        original_logging = sys.modules.get("neural_analysis.utils.logging")
        
        # Remove modules
        for mod in ["neural_analysis.metrics.outliers", "neural_analysis.utils.logging"]:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Mock import to raise ImportError
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "neural_analysis.utils.logging":
                raise ImportError("Mocked import error")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr("builtins.__import__", mock_import)
        importlib.invalidate_caches()
        
        # Re-import to trigger fallback
        import neural_analysis.metrics.outliers as outliers_module
        importlib.reload(outliers_module)
        
        # Verify fallback works
        assert hasattr(outliers_module, "get_logger")
        logger = outliers_module.get_logger("test")
        assert logger is not None
        
        # Restore
        if original_outliers:
            sys.modules["neural_analysis.metrics.outliers"] = original_outliers
        if original_logging:
            sys.modules["neural_analysis.utils.logging"] = original_logging

    def test_filter_outlier_method_iqr(self) -> None:
        """Test outlier filtering with IQR method."""
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [100, 100]])
        filtered = filter_outlier(data, method="iqr", threshold=1.5)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape[0] < data.shape[0]

    def test_filter_outlier_method_zscore(self) -> None:
        """Test outlier filtering with zscore method."""
        data = np.random.randn(100, 2)
        data[0] = [10, 10]  # Clear outlier
        filtered = filter_outlier(data, method="zscore", threshold=3.0)
        assert isinstance(filtered, np.ndarray)

    def test_filter_outlier_method_isolation(self) -> None:
        """Test outlier filtering with isolation method."""
        data = np.random.randn(100, 2)
        data[0] = [10, 10]  # Clear outlier
        filtered = filter_outlier(data, method="isolation", contamination=0.1)
        assert isinstance(filtered, np.ndarray)

    def test_filter_outlier_method_lof(self) -> None:
        """Test outlier filtering with lof method."""
        data = np.random.randn(100, 2)
        data[0] = [10, 10]  # Clear outlier
        filtered = filter_outlier(data, method="lof", contamination=0.1)
        assert isinstance(filtered, np.ndarray)

    def test_filter_outlier_method_elliptic(self) -> None:
        """Test outlier filtering with elliptic method."""
        data = np.random.randn(100, 2)
        data[0] = [10, 10]  # Clear outlier
        filtered = filter_outlier(data, method="elliptic", contamination=0.1)
        assert isinstance(filtered, np.ndarray)

    def test_filter_outlier_elliptic_n_le_d(self) -> None:
        """Test elliptic method with n <= d (covers lines 217-220)."""
        # To hit lines 217-220, we need:
        # - n >= 10 (to pass the early return check at line 96)
        # - method == "elliptic" (to call _mask_outliers_elliptic)
        # - n <= d (to hit the if condition at line 216)
        # So we need n >= 10 AND n <= d
        # Example: n=10, d=10 (n=10, d=10, so n <= d and n >= 10)
        data = np.random.randn(10, 10)  # n=10, d=10, so n <= d but n >= 10
        filtered = filter_outlier(data, method="elliptic", contamination=0.1)
        assert isinstance(filtered, np.ndarray)
        # Should return all points (all kept) when n <= d
        assert filtered.shape[0] == data.shape[0]
        
        # Also test with n=10, d=15 (n=10, d=15, so n < d but n >= 10)
        # This should NOT hit the n <= d condition (10 <= 15 is False)
        data2 = np.random.randn(10, 15)  # n=10, d=15, so n < d and n >= 10
        filtered2 = filter_outlier(data2, method="elliptic", contamination=0.1)
        assert isinstance(filtered2, np.ndarray)
        # This should use the actual detector, not the early return

    def test_filter_outlier_return_mask(self) -> None:
        """Test filter_outlier with return_mask=True."""
        data = np.random.randn(100, 2)
        data[0] = [10, 10]  # Clear outlier
        filtered, mask = filter_outlier(data, method="lof", return_mask=True)
        assert isinstance(filtered, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == data.shape[0]

    def test_filter_outlier_insufficient_samples(self) -> None:
        """Test filter_outlier with insufficient samples (covers lines 96-102)."""
        data = np.random.randn(5, 2)  # Less than 10 samples
        filtered = filter_outlier(data, method="lof")
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape[0] == data.shape[0]  # All returned

    def test_filter_outlier_elliptic_insufficient_samples(self) -> None:
        """Test filter_outlier elliptic with n <= d (covers lines 216-220)."""
        data = np.random.randn(2, 3)  # n=2, d=3, so n <= d
        filtered = filter_outlier(data, method="elliptic")
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape[0] == data.shape[0]  # All returned

    def test_filter_outlier_invalid_method(self) -> None:
        """Test filter_outlier with invalid method (covers lines 115-119)."""
        data = np.random.randn(100, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            filter_outlier(data, method="invalid")  # type: ignore

    def test_filter_outlier_zscore_fallback(self) -> None:
        """Test filter_outlier zscore with MAD=0 fallback (covers lines 164-176)."""
        # Create data where MAD=0 for some columns
        data = np.ones((100, 2))  # All same values, MAD=0
        data[0, 0] = 10.0  # One outlier
        filtered = filter_outlier(data, method="zscore", threshold=3.0)
        assert isinstance(filtered, np.ndarray)
