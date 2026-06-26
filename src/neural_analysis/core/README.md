# Core

This module defines the central structured datatypes and base results used throughout the `neural_analysis` package.

## Purpose

Provides foundational data structures, such as `AnalysisResult`, `MetricResult`, `EmbeddingResult`, and `DecodingResult`, which ensure a unified return type and API consistency across various modules like `metrics`, `embeddings`, and `learning`.

## What can be analyzed

This module itself does not perform analysis, but rather defines the structures that *contain* analysis results, ensuring downstream consumers (like the plotting module or external scripts) have a reliable, strongly-typed interface to work with.
