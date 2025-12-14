"""Complete tests for dimensionality_reduction.py to reach 100% coverage."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.embeddings.dimensionality_reduction import compute_embedding


class TestComputeEmbeddingEdgeCases:
    """Tests for compute_embedding edge cases (covers lines 41-43, 228, 401)."""

    def test_compute_embedding_umap_unavailable(self, monkeypatch) -> None:
        """Test compute_embedding when UMAP is unavailable (covers lines 41-43)."""
        import sys
        import importlib
        
        # Mock umap import to fail
        original_umap = sys.modules.get("umap")
        if "umap" in sys.modules:
            del sys.modules["umap"]
        
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "umap" or name.startswith("umap"):
                raise ImportError("Mocked umap import error")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr("builtins.__import__", mock_import)
        importlib.invalidate_caches()
        
        # Re-import module
        import neural_analysis.embeddings.dimensionality_reduction as dimred_module
        importlib.reload(dimred_module)
        
        # Try to use UMAP - should raise ImportError
        data = np.random.randn(100, 10)
        with pytest.raises(ImportError, match="UMAP"):
            dimred_module.compute_embedding(data, method="umap")
        
        # Restore
        if original_umap:
            sys.modules["umap"] = original_umap

    def test_compute_embedding_invalid_method(self) -> None:
        """Test compute_embedding with invalid method (covers line 228)."""
        data = np.random.randn(100, 10)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_embedding(data, method="invalid")  # type: ignore

    def test_compute_multiple_embeddings_import_error(self, monkeypatch) -> None:
        """Test compute_multiple_embeddings with ImportError (covers line 401)."""
        from neural_analysis.embeddings.dimensionality_reduction import compute_multiple_embeddings, compute_embedding
        from unittest.mock import patch, MagicMock
        
        data = np.random.randn(100, 10)
        
        # Mock compute_embedding to raise ImportError for a specific method
        original_compute_embedding = compute_embedding
        def mock_compute_embedding(*args, method=None, **kwargs):
            if method == "tsne":
                raise ImportError("Mocked tsne import error")
            return original_compute_embedding(*args, method=method, **kwargs)
        
        # Test with a method that will trigger ImportError
        with patch("neural_analysis.embeddings.dimensionality_reduction.compute_embedding", side_effect=mock_compute_embedding):
            with patch("neural_analysis.embeddings.dimensionality_reduction.logger") as mock_logger:
                result = compute_multiple_embeddings(data, methods=["pca", "tsne"])
                # Should have PCA but not t-SNE
                assert "pca" in result
                assert "tsne" not in result
                # Verify warning was logged (line 401)
                mock_logger.warning.assert_called()
                call_args = str(mock_logger.warning.call_args)
                assert "Skipping" in call_args and "tsne" in call_args

