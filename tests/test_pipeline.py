"""Tests for the analysis pipeline module."""

from __future__ import annotations

import numpy as np

from neural_analysis.pipeline import (
    PipelineConfig,
    PipelineResult,
    _extract_label_array,
    run_analysis,
)


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.dataset_type == "place_cells"
        assert cfg.n_samples == 5000
        assert cfg.embedding_methods == ["pca", "umap"]

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(
            dataset_type="swiss_roll",
            n_samples=200,
            embedding_methods=["pca"],
        )
        assert cfg.dataset_type == "swiss_roll"
        assert cfg.n_samples == 200


class TestExtractLabelArray:
    def test_numpy_array(self) -> None:
        arr = np.arange(10, dtype=np.float64)
        result = _extract_label_array(arr)
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_dict_with_positions_2d(self) -> None:
        positions = np.random.default_rng(0).random((50, 2))
        result = _extract_label_array({"positions": positions})
        assert result is not None
        assert result.shape == (50,)

    def test_dict_with_labels(self) -> None:
        labels = np.array([0, 1, 2, 0, 1], dtype=np.float64)
        result = _extract_label_array({"labels": labels})
        assert result is not None
        np.testing.assert_array_equal(result, labels)

    def test_empty_dict_returns_none(self) -> None:
        assert _extract_label_array({}) is None


class TestRunAnalysis:
    def test_swiss_roll_basic(self) -> None:
        result = run_analysis(
            dataset_type="swiss_roll",
            n_samples=200,
            embedding_methods=["pca"],
            compute_si=False,
            compute_decoding=False,
        )
        assert isinstance(result, PipelineResult)
        assert result.activity.shape[0] == 200
        assert "pca" in result.embeddings
        assert result.decoding is None
        assert result.structure_index is None

    def test_blobs_with_decoding(self) -> None:
        result = run_analysis(
            dataset_type="blobs",
            n_samples=200,
            n_features=10,
            embedding_methods=["pca"],
            n_components=2,
            compute_si=True,
            compute_decoding=True,
            si_n_bins=5,
            decoding_k=3,
            decoding_n_folds=2,
        )
        assert isinstance(result, PipelineResult)
        assert result.activity.shape == (200, 10)
        assert result.decoding is not None
        assert result.structure_index is not None
        assert "structure_index" in result.structure_index

    def test_with_config_object(self) -> None:
        cfg = PipelineConfig(
            dataset_type="moons",
            n_samples=100,
            embedding_methods=["pca"],
            n_components=2,
            compute_si=False,
            compute_decoding=False,
        )
        result = run_analysis(config=cfg)
        assert result.activity.shape[0] == 100
        assert "pca" in result.embeddings

    def test_kwargs_override_config(self) -> None:
        cfg = PipelineConfig(dataset_type="moons", n_samples=500)
        result = run_analysis(
            config=cfg,
            n_samples=100,
            n_components=2,
            embedding_methods=["pca"],
            compute_si=False,
            compute_decoding=False,
        )
        assert result.activity.shape[0] == 100
