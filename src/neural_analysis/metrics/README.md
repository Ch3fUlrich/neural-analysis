# Metrics

This module provides computational distances, distribution comparisons, and outlier detection logic.

## Purpose

Quantifying relationships within the data is essential to compare how different cell populations, or entire datasets, behave relative to each other or relative to themselves over time.

## What can be analyzed

- **Pairwise Metrics**: Compute standard distance matrices (e.g., Euclidean, Cosine, Correlation) across neural observations, frequently accelerated via Numba.
- **Distribution Comparison**: Calculate statistical distances (like Jensen-Shannon divergence or Wasserstein distance) between sets of points or distributions.
- **Shape Distance**: Analyze higher-order shape characteristics of datasets or embeddings.
- **Outliers**: Cleanse data by identifying and filtering out anomalous data points before running heavier analysis pipelines.
