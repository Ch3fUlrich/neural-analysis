"""Tests for decoding functions."""

import numpy as np
import pytest

from neural_analysis.decoding import (
    compare_highd_lowd_decoding,
    cross_validated_knn_decoder,
    evaluate_decoder,
    knn_decoder,
    population_vector_decoder,
)
from neural_analysis.synthetic_data import (
    generate_mixed_population_flexible,
    generate_place_cells,
)


class TestPopulationVectorDecoder:
    """Tests for population vector decoder."""

    def test_weighted_average_2d(self):
        """Test weighted average decoding in 2D."""
        # Generate place cells
        activity, meta = generate_place_cells(
            n_cells=30, n_samples=500, n_dims=2, field_size=0.2, seed=42
        )

        # Decode positions
        decoded = population_vector_decoder(
            activity, meta["field_centers"], method="weighted_average"
        )

        # Check shape
        assert decoded.shape == (500, 2)

        # Check decoding error is reasonable (should be < 0.3 for good place cells)
        errors = np.linalg.norm(decoded - meta["positions"], axis=1)
        mean_error = errors.mean()
        assert mean_error < 0.3, f"High decoding error: {mean_error}"

    def test_peak_method_1d(self):
        """Test peak decoding in 1D."""
        activity, meta = generate_place_cells(
            n_cells=20, n_samples=300, n_dims=1, field_size=0.15, seed=123
        )

        decoded = population_vector_decoder(
            activity, meta["field_centers"], method="peak"
        )

        assert decoded.shape == (300, 1)

        # Peak method should have higher error than weighted average
        errors = np.abs(decoded - meta["positions"])
        assert errors.mean() < 0.4

    def test_weighted_average_3d(self):
        """Test weighted average decoding in 3D."""
        activity, meta = generate_place_cells(
            n_cells=50, n_samples=400, n_dims=3, field_size=0.25, seed=789
        )

        decoded = population_vector_decoder(
            activity, meta["field_centers"], method="weighted_average"
        )

        assert decoded.shape == (400, 3)

        # 3D decoding should still work reasonably well
        errors = np.linalg.norm(decoded - meta["positions"], axis=1)
        assert errors.mean() < 0.4

    def test_zero_activity_handling(self):
        """Test decoder handles zero activity correctly."""
        activity = np.zeros((10, 20))  # All zeros
        field_centers = np.random.rand(20, 2)

        decoded = population_vector_decoder(
            activity, field_centers, method="weighted_average"
        )

        # Should default to mean of field centers
        expected = field_centers.mean(axis=0)
        np.testing.assert_allclose(decoded, expected, rtol=1e-5)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        activity = np.random.rand(100, 20)
        field_centers = np.random.rand(20, 2)

        with pytest.raises(ValueError, match="Unknown method"):
            population_vector_decoder(activity, field_centers, method="invalid")


class TestKNNDecoder:
    """Tests for k-NN decoder."""

    def test_knn_highd_2d(self):
        """Test k-NN decoding on high-D activity in 2D."""
        activity, meta = generate_place_cells(
            n_cells=40, n_samples=800, n_dims=2, field_size=0.2, seed=42
        )

        # Split train/test
        train_act, test_act = activity[:600], activity[600:]
        train_pos, test_pos = meta["positions"][:600], meta["positions"][600:]

        # Decode
        decoded = knn_decoder(train_act, train_pos, test_act, k=5, weights="distance")

        # Check shape
        assert decoded.shape == test_pos.shape

        # Check decoding error
        errors = np.linalg.norm(decoded - test_pos, axis=1)
        mean_error = errors.mean()
        assert mean_error < 0.25, f"High k-NN error: {mean_error}"

    def test_knn_lowd_2d(self):
        """Test k-NN decoding on low-D embedding in 2D."""
        from sklearn.decomposition import PCA

        activity, meta = generate_place_cells(
            n_cells=60, n_samples=1000, n_dims=2, field_size=0.18, seed=123
        )

        # Reduce to 10 dimensions
        pca = PCA(n_components=10)
        embedding = pca.fit_transform(activity)

        # Split
        train_emb, test_emb = embedding[:700], embedding[700:]
        train_pos, test_pos = meta["positions"][:700], meta["positions"][700:]

        # Decode from embedding
        decoded = knn_decoder(train_emb, train_pos, test_emb, k=8, weights="distance")

        assert decoded.shape == test_pos.shape

        # Decoding from embedding should still work well
        errors = np.linalg.norm(decoded - test_pos, axis=1)
        assert errors.mean() < 0.3

    def test_knn_1d_labels(self):
        """Test k-NN with 1D labels."""
        activity, meta = generate_place_cells(
            n_cells=30, n_samples=600, n_dims=1, field_size=0.15, seed=456
        )

        train_act, test_act = activity[:400], activity[400:]
        train_pos, test_pos = meta["positions"][:400], meta["positions"][400:]

        decoded = knn_decoder(train_act, train_pos, test_act, k=5)

        # Should work with 1D labels
        assert decoded.shape == test_pos.shape
        errors = np.abs(decoded - test_pos)
        assert errors.mean() < 0.25

    def test_knn_uniform_weights(self):
        """Test k-NN with uniform weights."""
        activity, meta = generate_place_cells(
            n_cells=35, n_samples=500, n_dims=2, seed=789
        )

        train_act, test_act = activity[:350], activity[350:]
        train_pos, test_pos = meta["positions"][:350], meta["positions"][350:]

        decoded = knn_decoder(train_act, train_pos, test_act, k=5, weights="uniform")

        assert decoded.shape == test_pos.shape
        # Uniform weights should still give reasonable results
        errors = np.linalg.norm(decoded - test_pos, axis=1)
        assert errors.mean() < 0.35

    def test_knn_different_k(self):
        """Test k-NN with different k values."""
        activity, meta = generate_place_cells(
            n_cells=40, n_samples=600, n_dims=2, seed=111
        )

        train_act, test_act = activity[:400], activity[400:]
        train_pos, test_pos = meta["positions"][:400], meta["positions"][400:]

        # Test multiple k values
        for k in [3, 5, 10, 20]:
            decoded = knn_decoder(train_act, train_pos, test_act, k=k)
            assert decoded.shape == test_pos.shape
            errors = np.linalg.norm(decoded - test_pos, axis=1)
            assert errors.mean() < 0.4, f"k={k} failed"


class TestCrossValidatedKNN:
    """Tests for cross-validated k-NN decoder."""

    def test_cv_knn_2d(self):
        """Test cross-validated k-NN in 2D."""
        activity, meta = generate_place_cells(
            n_cells=45, n_samples=1000, n_dims=2, field_size=0.2, seed=42
        )

        metrics = cross_validated_knn_decoder(
            activity, meta["positions"], k=5, n_folds=5
        )

        # Check metrics structure
        assert "mean_r2" in metrics
        assert "std_r2" in metrics
        assert "mean_mse" in metrics
        assert "mean_error" in metrics
        assert len(metrics["r2_scores"]) == 5

        # Check performance is good
        assert metrics["mean_r2"] > 0.7, f"Low R²: {metrics['mean_r2']}"
        assert metrics["mean_error"] < 0.3, f"High error: {metrics['mean_error']}"

    def test_cv_knn_with_predictions(self):
        """Test cross-validated k-NN returns predictions."""
        activity, meta = generate_place_cells(
            n_cells=30, n_samples=600, n_dims=2, seed=123
        )

        metrics = cross_validated_knn_decoder(
            activity, meta["positions"], k=5, n_folds=3, return_predictions=True
        )

        assert "predictions" in metrics
        assert len(metrics["predictions"]) == 3  # 3 folds

        # Check each fold has required info
        for fold in metrics["predictions"]:
            assert "test_idx" in fold
            assert "predictions" in fold
            assert "true_labels" in fold


class TestCompareHighDLowD:
    """Tests for comparing high-D vs low-D decoding."""

    def test_compare_pca(self):
        """Test comparison with PCA embedding."""
        from sklearn.decomposition import PCA

        activity, meta = generate_place_cells(
            n_cells=80, n_samples=1200, n_dims=2, field_size=0.2, seed=42
        )

        # Create PCA embedding
        pca = PCA(n_components=10)
        embedding = pca.fit_transform(activity)

        comparison = compare_highd_lowd_decoding(
            activity, embedding, meta["positions"], k=5, n_folds=5
        )

        # Check structure
        assert "high_d" in comparison
        assert "low_d" in comparison
        assert "performance_ratio" in comparison
        assert "error_increase" in comparison
        assert "dimensionality_reduction" in comparison

        # High-D should perform well
        assert comparison["high_d"]["mean_r2"] > 0.7

        # Low-D should preserve most performance (PCA is linear, should be good)
        assert comparison["performance_ratio"] > 0.7

        # Check dimensionality info
        assert comparison["n_cells"] == 80
        assert comparison["n_components"] == 10

    def test_compare_noisy_data(self):
        """Test comparison with noisy data."""
        from sklearn.decomposition import PCA

        activity, meta = generate_place_cells(
            n_cells=60,
            n_samples=1000,
            n_dims=2,
            field_size=0.2,
            noise_level=0.15,
            seed=789,  # High noise
        )

        pca = PCA(n_components=8)
        embedding = pca.fit_transform(activity)

        comparison = compare_highd_lowd_decoding(
            activity, embedding, meta["positions"], k=5
        )

        # Even with noise, should get reasonable results
        assert comparison["high_d"]["mean_r2"] > 0.4
        # Performance ratio might be lower with noise
        assert comparison["performance_ratio"] > 0.5


class TestEvaluateDecoder:
    """Tests for evaluate_decoder function."""

    def test_evaluate_knn(self):
        """Test evaluate_decoder with k-NN."""
        activity, meta = generate_place_cells(
            n_cells=40, n_samples=800, n_dims=2, seed=42
        )

        train_act, test_act = activity[:600], activity[600:]
        train_pos, test_pos = meta["positions"][:600], meta["positions"][600:]

        metrics = evaluate_decoder(
            train_act, train_pos, test_act, test_pos, decoder="knn", k=5
        )

        assert "r2_score" in metrics
        assert "mse" in metrics
        assert "mean_error" in metrics
        assert metrics["decoder"] == "knn"

        # Check reasonable performance
        assert metrics["r2_score"] > 0.7
        assert metrics["mean_error"] < 0.3

    def test_evaluate_population_vector(self):
        """Test evaluate_decoder with population vector."""
        activity, meta = generate_place_cells(
            n_cells=35, n_samples=600, n_dims=2, seed=123
        )

        train_act, test_act = activity[:400], activity[400:]
        train_pos, test_pos = meta["positions"][:400], meta["positions"][400:]

        metrics = evaluate_decoder(
            train_act,
            train_pos,
            test_act,
            test_pos,
            decoder="population_vector",
            field_centers=meta["field_centers"],
            method="weighted_average",
        )

        assert "r2_score" in metrics
        assert "mean_error" in metrics
        assert metrics["decoder"] == "population_vector"

        # Population vector should give reasonable results
        assert metrics["r2_score"] > 0.5

    def test_evaluate_invalid_decoder(self):
        """Test evaluate_decoder with invalid decoder type."""
        activity = np.random.rand(100, 20)
        labels = np.random.rand(100, 2)

        with pytest.raises(ValueError, match="Unknown decoder"):
            evaluate_decoder(activity, labels, activity, labels, decoder="invalid")

    def test_evaluate_population_vector_missing_centers(self):
        """Test population vector without field_centers raises error."""
        activity = np.random.rand(100, 20)
        labels = np.random.rand(100, 2)

        with pytest.raises(ValueError, match="requires 'field_centers'"):
            evaluate_decoder(
                activity, labels, activity, labels, decoder="population_vector"
            )


class TestIntegrationDecoding:
    """Integration tests with mixed populations."""

    def test_decode_mixed_population(self):
        """Test decoding from mixed cell population."""
        activity, meta = generate_mixed_population_flexible(n_samples=1000, seed=42)

        # Split
        train_act, test_act = activity[:700], activity[700:]
        train_pos, test_pos = meta["positions"][:700], meta["positions"][700:]

        # Decode with k-NN
        decoded = knn_decoder(train_act, train_pos, test_act, k=5)

        # Should still decode reasonably (mixed pop has noise)
        errors = np.linalg.norm(decoded - test_pos, axis=1)
        assert errors.mean() < 0.4

    def test_cv_mixed_population(self):
        """Test cross-validated decoding on mixed population."""
        activity, meta = generate_mixed_population_flexible(n_samples=1200, seed=789)

        metrics = cross_validated_knn_decoder(
            activity, meta["positions"], k=5, n_folds=4
        )

        # Mixed population should still allow decoding
        assert metrics["mean_r2"] > 0.5
        assert metrics["mean_error"] < 0.5
