# Data

This module provides tools for generating, formatting, and handling neural data and relevant trajectories.

## Purpose

The primary goal is to supply raw inputs for analysis pipelines. It contains logic to synthesize artificial neural recordings (e.g., place cells, grid cells, head direction cells) and the underlying continuous paths (trajectories).

## What can be analyzed

- **Synthetic Data Generation**: Create configurable populations of place cells, grid cells, and head direction cells along 1D, 2D, or 3D trajectories.
- **Trajectory Generation**: Generate random walks or structured spatial traversals.
- **Testing Grounds**: Use the generated datasets to validate metric accuracy, dimensionality reduction performance, and decoding models under controlled conditions with known ground truths.
