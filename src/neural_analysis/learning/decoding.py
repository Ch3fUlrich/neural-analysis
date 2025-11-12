"""Decoding methods for neural data.

This module provides functions for decoding behavioral variables (position,
head direction, etc.) from neural activity or low-dimensional embeddings.

Supported methods:
- Population vector decoder (weighted average, peak)
- k-Nearest Neighbors decoder (for high-D and low-D comparison)
- Bayesian decoder (future)

These decoders can be applied to:
- Raw neural activity (high-dimensional)
- Low-dimensional embeddings (UMAP, Isomap, etc.)

This enables comparison of decoding performance across dimensions and
evaluation of embedding quality.

Examples:
    Decode position from neural activity:
        >>> from neural_analysis.learning.decoding import knn_decoder
        >>> activity, meta = generate_place_cells(50, 1000)
        >>> decoded_pos = knn_decoder(
        ...     activity, meta['positions'], activity,
        ...     k=5, method='weighted'
        ... )

    Decode from embedding:
        >>> embedding = umap.fit_transform(activity)
        >>> decoded_from_embedding = knn_decoder(
        ...     activity, meta['positions'], embedding,
        ...     k=10, method='distance'
        ... )

    Compare high-D vs low-D decoding:
        >>> from neural_analysis.learning.decoding import evaluate_decoder
        >>> # Decode from raw activity
        >>> metrics_highd = evaluate_decoder(
        ...     activity, meta['positions'], activity,
        ...     decoder='knn', k=5
        ... )
        >>> # Decode from embedding
        >>> metrics_lowd = evaluate_decoder(
        ...     activity, meta['positions'], embedding,
        ...     decoder='knn', k=5
        ... )
        >>> print(f"High-D error: {metrics_highd['mean_error']:.3f}")
        >>> print(f"Low-D error: {metrics_lowd['mean_error']:.3f}")
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


def population_vector_decoder(
    activity: npt.NDArray[np.float64],
    field_centers: npt.NDArray[np.float64],
    method: Literal["weighted_average", "peak"] = "weighted_average",
) -> npt.NDArray[np.float64]:
    """Decode position from population activity using population vector.

    Classic neuroscience method that uses the known tuning properties
    (field centers) of cells to decode behavioral variables.

    Args:
        activity: Neural activity matrix, shape (n_samples, n_cells).
        field_centers: Preferred locations/angles of cells, shape (n_cells, n_dims).
            These are the tuning curve centers (e.g., place field centers).
        method: Decoding method:
            - 'weighted_average': Weighted average of field centers by activity
            - 'peak': Use position of most active cell

    Returns:
        decoded_positions: Decoded positions, shape (n_samples, n_dims).

    Examples:
        >>> from neural_analysis.data.synthetic_data import generate_place_cells
        >>> activity, meta = generate_place_cells(50, 1000)
        >>> decoded = population_vector_decoder(
        ...     activity, meta['field_centers'], method='weighted_average'
        ... )
        >>> # Compare with true positions
        >>> error = np.linalg.norm(decoded - meta['positions'], axis=1).mean()
        >>> print(f"Mean decoding error: {error:.3f} m")
    """
    n_samples, n_cells = activity.shape
    n_dims = field_centers.shape[1]

    decoded_positions = np.zeros((n_samples, n_dims))

    if method == "weighted_average":
        # Weighted average of field centers by activity
        for t in range(n_samples):
            weights = activity[t, :]
            if weights.sum() > 0:
                decoded_positions[t] = np.average(
                    field_centers, axis=0, weights=weights
                )
            else:
                decoded_positions[t] = field_centers.mean(axis=0)

    elif method == "peak":
        # Use position of most active cell
        for t in range(n_samples):
            peak_cell = np.argmax(activity[t, :])
            decoded_positions[t] = field_centers[peak_cell]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'weighted_average' or 'peak'")

    return decoded_positions


def knn_decoder(
    train_activity: npt.NDArray[np.float64],
    train_labels: npt.NDArray[np.float64],
    test_activity: npt.NDArray[np.float64],
    k: int = 5,
    weights: Literal["uniform", "distance"] = "distance",
    metric: str = "euclidean",
) -> npt.NDArray[np.float64]:
    """Decode behavioral variables using k-Nearest Neighbors.

    This decoder works on both:
    - High-dimensional neural activity (n_cells dimensions)
    - Low-dimensional embeddings (2-3 dimensions)

    This enables direct comparison of decoding performance across dimensions
    and evaluation of embedding quality.

    Args:
        train_activity: Training data, shape (n_train, n_features).
            Can be raw activity or embedding.
        train_labels: Training labels (positions, angles, etc.),
            shape (n_train, n_dims).
        test_activity: Test data, shape (n_test, n_features).
            Must have same n_features as train_activity.
        k: Number of nearest neighbors to use.
        weights: Weight function for neighbors:
            - 'uniform': All neighbors weighted equally
            - 'distance': Weight by inverse distance (closer = higher weight)
        metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.).

    Returns:
        decoded_labels: Decoded labels, shape (n_test, n_dims).

    Examples:
        >>> # Decode from raw activity (high-D)
        >>> activity, meta = generate_place_cells(100, 1000)
        >>> train_act, test_act = activity[:800], activity[800:]
        >>> train_pos, test_pos = meta['positions'][:800], meta['positions'][800:]
        >>> decoded_highd = knn_decoder(train_act, train_pos, test_act, k=5)

        >>> # Decode from embedding (low-D)
        >>> embedding = umap.fit_transform(activity)
        >>> train_emb, test_emb = embedding[:800], embedding[800:]
        >>> decoded_lowd = knn_decoder(train_emb, train_pos, test_emb, k=10)

        >>> # Compare errors
        >>> error_highd = np.linalg.norm(decoded_highd - test_pos, axis=1).mean()
        >>> error_lowd = np.linalg.norm(decoded_lowd - test_pos, axis=1).mean()
        >>> print(f"High-D: {error_highd:.3f}, Low-D: {error_lowd:.3f}")
    """
    # Ensure labels are 2D
    if train_labels.ndim == 1:
        train_labels = train_labels.reshape(-1, 1)

    # Initialize k-NN regressor
    knn = KNeighborsRegressor(
        n_neighbors=k,
        weights=weights,
        metric=metric,
        n_jobs=-1,  # Use all CPU cores
    )

    # Fit on training data
    knn.fit(train_activity, train_labels)

    # Predict on test data
    decoded_labels = knn.predict(test_activity)

    return decoded_labels


def cross_validated_knn_decoder(
    activity: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    k: int = 5,
    n_folds: int = 5,
    weights: Literal["uniform", "distance"] = "distance",
    metric: str = "euclidean",
    return_predictions: bool = False,
) -> dict[str, Any]:
    """k-NN decoder with cross-validation.

    Evaluates decoding performance using k-fold cross-validation.
    Returns comprehensive metrics including R², MSE, and mean error.

    Args:
        activity: Neural activity or embedding, shape (n_samples, n_features).
        labels: True labels to decode, shape (n_samples, n_dims).
        k: Number of nearest neighbors.
        n_folds: Number of cross-validation folds.
        weights: 'uniform' or 'distance'.
        metric: Distance metric.
        return_predictions: If True, return predictions for each fold.

    Returns:
        metrics: Dictionary with:
            - 'r2_scores': R² score for each fold
            - 'mse_scores': MSE for each fold
            - 'mean_r2': Mean R² across folds
            - 'std_r2': Std of R² across folds
            - 'mean_mse': Mean MSE across folds
            - 'std_mse': Std of MSE across folds
            - 'mean_error': Mean Euclidean error across folds
            - 'predictions': (optional) Predictions for each fold

    Examples:
        >>> # Evaluate decoding on raw activity
        >>> activity, meta = generate_place_cells(80, 1200)
        >>> metrics_highd = cross_validated_knn_decoder(
        ...     activity, meta['positions'], k=5, n_folds=5
        ... )
        >>> print(f"R²: {metrics_highd['mean_r2']:.3f} ± {metrics_highd['std_r2']:.3f}")

        >>> # Evaluate on embedding
        >>> embedding = umap.fit_transform(activity)
        >>> metrics_lowd = cross_validated_knn_decoder(
        ...     embedding, meta['positions'], k=10, n_folds=5
        ... )
        >>> print(f"Embedding R²: {metrics_lowd['mean_r2']:.3f}")
    """
    from sklearn.model_selection import KFold

    # Ensure labels are 2D
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    r2_scores = []
    mse_scores = []
    euclidean_errors = []
    all_predictions: list[dict[str, Any]] = [] if return_predictions else None  # type: ignore[assignment]

    for train_idx, test_idx in kfold.split(activity):
        # Split data
        train_act, test_act = activity[train_idx], activity[test_idx]
        train_lab, test_lab = labels[train_idx], labels[test_idx]

        # Decode
        predictions = knn_decoder(
            train_act, train_lab, test_act, k=k, weights=weights, metric=metric
        )

        # Compute metrics
        r2 = r2_score(test_lab, predictions, multioutput="uniform_average")
        mse = mean_squared_error(test_lab, predictions, multioutput="uniform_average")

        # Euclidean error (for spatial decoding)
        if labels.shape[1] > 1:
            euclidean_error = np.linalg.norm(predictions - test_lab, axis=1).mean()
        else:
            euclidean_error = np.abs(predictions - test_lab).mean()

        r2_scores.append(r2)
        mse_scores.append(mse)
        euclidean_errors.append(euclidean_error)

        if return_predictions and all_predictions is not None:
            all_predictions.append(
                {
                    "test_idx": test_idx,
                    "predictions": predictions,
                    "true_labels": test_lab,
                }
            )

    metrics = {
        "r2_scores": np.array(r2_scores),
        "mse_scores": np.array(mse_scores),
        "euclidean_errors": np.array(euclidean_errors),
        "mean_r2": np.mean(r2_scores),
        "std_r2": np.std(r2_scores),
        "mean_mse": np.mean(mse_scores),
        "std_mse": np.std(mse_scores),
        "mean_error": np.mean(euclidean_errors),
        "std_error": np.std(euclidean_errors),
        "k": k,
        "n_folds": n_folds,
    }

    if return_predictions:
        metrics["predictions"] = all_predictions

    return metrics


def compare_highd_lowd_decoding(
    activity: npt.NDArray[np.float64],
    embedding: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    k: int = 5,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Compare decoding performance on high-D activity vs low-D embedding.

    This is a key analysis for evaluating embedding quality: good embeddings
    should preserve decodable information about behavioral variables.

    Args:
        activity: High-dimensional neural activity, shape (n_samples, n_cells).
        embedding: Low-dimensional embedding, shape (n_samples, n_components).
        labels: True behavioral labels, shape (n_samples, n_dims).
        k: Number of nearest neighbors.
        n_folds: Number of cross-validation folds.

    Returns:
        comparison: Dictionary with:
            - 'high_d': Metrics from high-D decoding
            - 'low_d': Metrics from low-D decoding
            - 'dimensionality_reduction': n_cells → n_components
            - 'performance_ratio': low-D R² / high-D R²
            - 'error_increase': (low-D error - high-D error) / high-D error

    Examples:
        >>> activity, meta = generate_place_cells(100, 1500)
        >>> embedding = umap.fit_transform(activity)
        >>> comparison = compare_highd_lowd_decoding(
        ...     activity, embedding, meta['positions'], k=5
        ... )
        >>> print(f"High-D R²: {comparison['high_d']['mean_r2']:.3f}")
        >>> print(f"Low-D R²: {comparison['low_d']['mean_r2']:.3f}")
        >>> print(f"Performance ratio: {comparison['performance_ratio']:.2%}")
    """
    # Decode from high-D activity
    metrics_highd = cross_validated_knn_decoder(activity, labels, k=k, n_folds=n_folds)

    # Decode from low-D embedding
    metrics_lowd = cross_validated_knn_decoder(embedding, labels, k=k, n_folds=n_folds)

    # Compute comparison metrics
    performance_ratio = metrics_lowd["mean_r2"] / (metrics_highd["mean_r2"] + 1e-10)
    error_increase = (metrics_lowd["mean_error"] - metrics_highd["mean_error"]) / (
        metrics_highd["mean_error"] + 1e-10
    )

    comparison = {
        "high_d": metrics_highd,
        "low_d": metrics_lowd,
        "dimensionality_reduction": f"{activity.shape[1]} → {embedding.shape[1]}",
        "n_cells": activity.shape[1],
        "n_components": embedding.shape[1],
        "performance_ratio": performance_ratio,
        "error_increase": error_increase,
        "information_preserved": performance_ratio > 0.8,  # Heuristic threshold
    }

    return comparison


def evaluate_decoder(
    train_activity: npt.NDArray[np.float64],
    train_labels: npt.NDArray[np.float64],
    test_activity: npt.NDArray[np.float64],
    test_labels: npt.NDArray[np.float64],
    decoder: Literal["knn", "population_vector"] = "knn",
    **decoder_params: Any,
) -> dict[str, float]:
    """Evaluate decoder on train/test split.

    Unified interface for evaluating different decoder types.

    Args:
        train_activity: Training neural data, shape (n_train, n_features).
        train_labels: Training labels, shape (n_train, n_dims).
        test_activity: Test neural data, shape (n_test, n_features).
        test_labels: Test labels, shape (n_test, n_dims).
        decoder: Decoder type ('knn' or 'population_vector').
        **decoder_params: Decoder-specific parameters.
            For 'knn': k, weights, metric
            For 'population_vector': method, field_centers

    Returns:
        metrics: Dictionary with R², MSE, and mean error.

    Examples:
        >>> activity, meta = generate_place_cells(80, 1000)
        >>> train_act, test_act = activity[:800], activity[800:]
        >>> train_pos, test_pos = meta['positions'][:800], meta['positions'][800:]

        >>> # Evaluate k-NN
        >>> metrics = evaluate_decoder(
        ...     train_act, train_pos, test_act, test_pos,
        ...     decoder='knn', k=5
        ... )

        >>> # Evaluate population vector
        >>> metrics = evaluate_decoder(
        ...     train_act, train_pos, test_act, test_pos,
        ...     decoder='population_vector',
        ...     field_centers=meta['field_centers'],
        ...     method='weighted_average'
        ... )
    """
    # Decode
    if decoder == "knn":
        k = decoder_params.get("k", 5)
        weights = decoder_params.get("weights", "distance")
        metric = decoder_params.get("metric", "euclidean")
        predictions = knn_decoder(
            train_activity,
            train_labels,
            test_activity,
            k=k,
            weights=weights,
            metric=metric,
        )

    elif decoder == "population_vector":
        field_centers = decoder_params.get("field_centers")
        method = decoder_params.get("method", "weighted_average")
        if field_centers is None:
            raise ValueError(
                "population_vector decoder requires 'field_centers' parameter"
            )
        predictions = population_vector_decoder(
            test_activity, field_centers, method=method
        )

    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    # Ensure 2D
    if test_labels.ndim == 1:
        test_labels = test_labels.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # Compute metrics
    r2 = r2_score(test_labels, predictions, multioutput="uniform_average")
    mse = mean_squared_error(test_labels, predictions)

    # Euclidean error
    if test_labels.shape[1] > 1:
        mean_error = np.linalg.norm(predictions - test_labels, axis=1).mean()
    else:
        mean_error = np.abs(predictions - test_labels).mean()

    metrics = {
        "r2_score": r2,
        "mse": mse,
        "mean_error": mean_error,
        "decoder": decoder,
    }

    return metrics
