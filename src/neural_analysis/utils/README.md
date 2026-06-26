# Utils

This module houses general-purpose utilities, helper functions, and foundational sub-systems that support the core computational modules.

## Purpose

To centralize common functionality, ensuring DRY (Don't Repeat Yourself) principles and keeping the higher-level analysis modules focused on their specific domains.

## What can be analyzed

This module provides the *infrastructure* for analysis:
- **Storage**: The multi-layered `StorageManager` coordinates caching and persistence across Redis, DuckDB, and HDF5, allowing rapid retrieval of previous analytical comparisons.
- **File Management & I/O**: Safely read, write, and restructure nested HDF5 files.
- **Logging & Progress**: Standardized output tracking for long-running scripts.
- **Validation**: Enforce expected types and shapes on input data matrices.
- **Geometry & Subsampling**: Perform common mathematical transformations and safely downsample large arrays for memory efficiency.
- **Statistics & Signal Processing**: Foundational statistical tests and filtering models.
