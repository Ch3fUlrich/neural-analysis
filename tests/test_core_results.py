"""Tests for core result dataclasses."""

from __future__ import annotations

import numpy as np

from neural_analysis.core.results import (
    AnalysisResult,
    DecodingResult,
    EmbeddingResult,
    MetricResult,
)


class TestAnalysisResult:
    def test_creation(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        result = AnalysisResult(data=data, metadata={"seed": 42})
        assert np.array_equal(result.data, data)
        assert result.metadata == {"seed": 42}

    def test_default_metadata(self) -> None:
        result = AnalysisResult(data=np.zeros(3))
        assert result.metadata == {}

    def test_frozen(self) -> None:
        result = AnalysisResult(data=np.zeros(3))
        try:
            result.data = np.ones(3)  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestMetricResult:
    def test_creation(self) -> None:
        data = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = MetricResult(data=data, metric_name="euclidean", value=1.0)
        assert result.metric_name == "euclidean"
        assert result.value == 1.0
        assert result.pairs is None

    def test_with_pairs(self) -> None:
        result = MetricResult(
            data=np.zeros(2),
            metric_name="ot",
            value=0.5,
            pairs={(0, 1): 0.8, (1, 0): 0.2},
        )
        assert len(result.pairs) == 2


class TestEmbeddingResult:
    def test_creation(self) -> None:
        data = np.random.randn(100, 2)
        result = EmbeddingResult(data=data, method="pca", n_components=2)
        assert result.method == "pca"
        assert result.n_components == 2
        assert result.explained_variance is None

    def test_with_variance(self) -> None:
        var = np.array([0.8, 0.15])
        result = EmbeddingResult(
            data=np.zeros((10, 2)),
            method="pca",
            n_components=2,
            explained_variance=var,
        )
        assert np.array_equal(result.explained_variance, var)


class TestDecodingResult:
    def test_creation(self) -> None:
        result = DecodingResult(r_squared=0.95, mse=0.01)
        assert result.r_squared == 0.95
        assert result.mse == 0.01
        assert result.predictions is None
        assert result.metadata == {}

    def test_with_predictions(self) -> None:
        preds = np.array([1.0, 2.0, 3.0])
        result = DecodingResult(
            r_squared=0.9, mse=0.05, predictions=preds, metadata={"model": "knn"}
        )
        assert np.array_equal(result.predictions, preds)
        assert result.metadata["model"] == "knn"
