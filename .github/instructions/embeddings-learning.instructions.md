---
name: 'Embeddings & Learning'
description: 'Dimensionality reduction, decoding, and classification'
applyTo: 'src/neural_analysis/embeddings/**,src/neural_analysis/learning/**'
---

# Embeddings & Learning Instructions

## Embeddings Module

### 7 Embedding Methods (via `compute_embedding()`)

| Method | Backend | Notes |
|--------|---------|-------|
| PCA | sklearn | Default; supports explained variance analysis |
| UMAP | umap-learn | Optional; graceful fallback if unavailable |
| t-SNE | sklearn | Perplexity auto-adjusted for small datasets |
| MDS | sklearn | Supports precomputed distance matrices |
| Isomap | sklearn | Supports precomputed distances |
| LLE | sklearn | Locally linear embedding |
| Spectral | sklearn | Spectral embedding via graph Laplacian |

- `compute_multiple_embeddings(data, methods)` — Compare several methods at once.
- `pca_explained_variance(data, n_components)` — Variance analysis with 90/95/99% thresholds.
- Visualization via `plot_multiple_embeddings` and `plot_pca_variance` (PlotGrid-based).
- `EmbeddingConfig` dataclass groups embedding parameters. `EmbeddingResult` dataclass provides structured returns.
- All methods accept a `random_state` parameter for reproducibility.

## Learning Module

### Decoding (`decoding.py`)

- `population_vector_decoder` — Weighted-average or peak decoding
- `knn_decoder` / `cross_validated_knn_decoder` — k-NN regression with CV
- `compare_highd_lowd_decoding` — Compare raw vs. embedding decoding quality (key analysis tool)
- `evaluate_decoder` — Unified train/test evaluation
- `DecodingResult` dataclass provides structured decoding returns (r_squared, mse, predictions, metadata).

### Classification (`classification.py`)

- 9 supervised: random_forest, svc, svc_rbf, logistic_regression, knn, naive_bayes, mlp, gradient_boosting, adaboost
- 7 unsupervised: kmeans, dbscan, agglomerative, gaussian_mixture, spectral, birch, mean_shift
- `extract_cell_features(activity, metadata, positions)` — Automated feature extraction
- `compare_classifiers()` / `compare_clusterers()` — Benchmark all methods with timing
- Factory pattern via `match/case` for classifier/clusterer instantiation

## Adding New Methods

1. Add the method name to the relevant constant/enum.
2. Implement using existing patterns (factory via `match/case`).
3. Ensure optional deps are guarded with `try/except ImportError`.
4. Add to comparison benchmarks.
5. Test, update `docs/function_registry.md`.
