# Research Report: `neural-analysis` Repository Deep Dive

> **Date:** 2026-03-03  
> **Repository:** Ch3fUlrich/neural-analysis  
> **Branch:** migration (default: main)  
> **Version:** 0.0.0 (pre-release)

---

## Table of Contents

- [Research Report: `neural-analysis` Repository Deep Dive](#research-report-neural-analysis-repository-deep-dive)
  - [Table of Contents](#table-of-contents)
  - [1. Executive Summary](#1-executive-summary)
  - [2. Project Identity and Goals](#2-project-identity-and-goals)
  - [3. Technology Stack](#3-technology-stack)
  - [4. Repository Structure](#4-repository-structure)
  - [5. Module Architecture](#5-module-architecture)
    - [5.1 Dependency Flow](#51-dependency-flow)
    - [5.2 data — Synthetic Neural Data Generation](#52-data--synthetic-neural-data-generation)
      - [Cell Types](#cell-types)
      - [Key Functions](#key-functions)
    - [5.3 metrics — Distance, Distribution, and Shape Metrics](#53-metrics--distance-distribution-and-shape-metrics)
      - [5.3.1 pairwise\_metrics.py (~2,033 lines) — Unified Pairwise Metrics](#531-pairwise_metricspy-2033-lines--unified-pairwise-metrics)
      - [5.3.2 distributions.py (~1,907 lines) — Statistical and Shape Distances](#532-distributionspy-1907-lines--statistical-and-shape-distances)
      - [5.3.3 outliers.py (~210 lines) — Outlier Detection](#533-outlierspy-210-lines--outlier-detection)
    - [5.4 embeddings — Dimensionality Reduction](#54-embeddings--dimensionality-reduction)
    - [5.5 learning — Decoding and Classification](#55-learning--decoding-and-classification)
      - [5.5.1 decoding.py (~340 lines)](#551-decodingpy-340-lines)
      - [5.5.2 classification.py (~816 lines)](#552-classificationpy-816-lines)
    - [5.6 topology — Structure Index](#56-topology--structure-index)
      - [Algorithm](#algorithm)
      - [Key Functions](#key-functions-1)
    - [5.7 plotting — PlotGrid Visualization System](#57-plotting--plotgrid-visualization-system)
      - [Architecture Layers](#architecture-layers)
      - [16 Plot Types](#16-plot-types)
      - [Core Components](#core-components)
      - [Special Visualization Modules](#special-visualization-modules)
    - [5.8 utils — Shared Utilities](#58-utils--shared-utilities)
      - [I/O (`utils/io.py`, ~1,008 lines)](#io-utilsiopy-1008-lines)
      - [Comparison Store (`utils/comparison_store.py`, ~638 lines)](#comparison-store-utilscomparison_storepy-638-lines)
      - [Logging (`utils/logging.py`, ~216 lines)](#logging-utilsloggingpy-216-lines)
      - [Other Utilities](#other-utilities)
  - [6. Storage Architecture](#6-storage-architecture)
    - [6.1 Three-Layer Stack](#61-three-layer-stack)
    - [6.2 HDF5 — Persistent Ground Truth](#62-hdf5--persistent-ground-truth)
    - [6.3 DuckDB — Metadata Index](#63-duckdb--metadata-index)
    - [6.4 Redis — In-Memory Cache](#64-redis--in-memory-cache)
    - [6.5 StorageManager Orchestrator](#65-storagemanager-orchestrator)
  - [7. PlotGrid System Deep Dive](#7-plotgrid-system-deep-dive)
    - [Pipeline Flow](#pipeline-flow)
    - [PlotSpec Highlights](#plotspec-highlights)
    - [Multi-Panel Layouts](#multi-panel-layouts)
    - [Factory Constructors](#factory-constructors)
  - [8. Testing and Quality](#8-testing-and-quality)
    - [Test Suite](#test-suite)
    - [Test Naming Convention](#test-naming-convention)
    - [Coverage Configuration](#coverage-configuration)
    - [Quality Checks (in order)](#quality-checks-in-order)
  - [9. CI/CD Pipeline](#9-cicd-pipeline)
    - [GitHub Actions (`ci.yml`)](#github-actions-ciyml)
    - [Local CI](#local-ci)
  - [10. Development Workflow and Conventions](#10-development-workflow-and-conventions)
    - [Branch Strategy](#branch-strategy)
    - [UV-Only Execution](#uv-only-execution)
    - [Code Standards](#code-standards)
    - [File Lifecycle](#file-lifecycle)
  - [11. Legacy Code and Migration Status](#11-legacy-code-and-migration-status)
    - [Legacy Code Location](#legacy-code-location)
    - [Migration Roadmap (from TODO.md)](#migration-roadmap-from-todomd)
    - [Planned Future Features](#planned-future-features)
  - [12. Docker Environment](#12-docker-environment)
    - [Dockerfile](#dockerfile)
    - [docker-compose.yml](#docker-composeyml)
  - [13. Key Design Patterns](#13-key-design-patterns)
  - [14. Strengths](#14-strengths)
  - [15. Weaknesses and Technical Debt](#15-weaknesses-and-technical-debt)
  - [16. Summary Statistics](#16-summary-statistics)

---

## 1. Executive Summary

`neural-analysis` is a Python library for **analyzing neural population data**, with a focus on synthetic data generation, distance/shape metrics, dimensionality reduction, neural decoding, topological structure analysis, and publication-quality visualization. It is built around a modular architecture with a **three-layer storage stack** (HDF5 + DuckDB + Redis), a **metadata-driven plotting system** (PlotGrid), and strict quality gates enforced via UV, ruff, mypy, and pytest.

The project is currently in an active **migration phase**, porting ~16,000+ lines of legacy research code (in `todo/`) into the clean modular architecture under `src/neural_analysis/`. The core library is functional with 181+ passing tests and comprehensive documentation.

---

## 2. Project Identity and Goals

- **Domain:** Computational neuroscience — analysis of neural population recordings (place cells, grid cells, head direction cells)
- **Primary use cases:**
  1. Generate synthetic neural datasets for method validation
  2. Compute pairwise distances, distribution comparisons, and shape similarities between neural representations
  3. Perform dimensionality reduction and evaluate embedding quality via decoding
  4. Quantify topological structure preservation (Structure Index)
  5. Visualize all of the above with publication-ready or interactive plots
- **Design philosophy:** Incremental progress, test-driven, explicit > clever, composition over inheritance, fail-fast error handling, no global state

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python ≥3.10, <3.14 | Core development |
| **Package manager** | UV (Astral) | Reproducible environments, dependency management, execution |
| **Build system** | Hatchling | PEP 517 build backend |
| **Core numerics** | NumPy ≥1.24, SciPy ≥1.11, Pandas ≥2.0 | Array operations, statistics, data frames |
| **ML** | scikit-learn ≥1.7.2 | Classifiers, clusterers, cross-validation, metrics |
| **Acceleration** | Numba ≥0.59 (optional) | JIT-compiled pairwise distance kernels |
| **Dimensionality reduction** | UMAP-learn ≥0.5.3 (optional), sklearn | Embedding computation |
| **Optimal transport** | POT ≥0.9 (optional) | Earth Mover's Distance, Sinkhorn soft matching |
| **Graph analysis** | NetworkX ≥3.4.2 | Structure index overlap graphs |
| **Similarity search** | FAISS-cpu ≥1.8.0 (optional) | Fast k-NN for structure index |
| **Visualization** | Matplotlib ≥3.7, Plotly ≥5.18 | Dual-backend plotting via PlotGrid |
| **Storage** | h5py ≥3.10, DuckDB ≥0.10 (optional), Redis ≥5.0 (optional) | Three-layer persistence stack |
| **Notebooks** | Marimo ≥0.18.3 | Interactive examples (replacing Jupyter) |
| **Testing** | pytest ≥8.0, pytest-xdist, pytest-cov | Parallel test execution, coverage |
| **Linting/Formatting** | ruff ≥0.14 | Linting (15 rule sets) + formatting |
| **Type checking** | mypy ≥1.18 (strict mode) | Static type analysis |
| **CI** | GitHub Actions + act (local) | Automated quality gates |
| **Containerization** | Docker + docker-compose | Redis service, reproducible dev environment |

---

## 4. Repository Structure

```
neural-analysis/
├── src/neural_analysis/         # Core library (7 subpackages)
│   ├── data/                    #   Synthetic data generators
│   ├── metrics/                 #   Distance, distribution, shape, outlier metrics
│   ├── embeddings/              #   Dimensionality reduction
│   ├── learning/                #   Decoding and classification
│   ├── topology/                #   Structure index analysis
│   ├── plotting/                #   PlotGrid visualization system (12 modules)
│   └── utils/                   #   I/O, logging, validation, storage
│       └── storage/             #     StorageManager, Redis, DuckDB backends
├── tests/                       # 65+ test files
├── docs/                        # 20+ documentation files
├── examples/                    # 17 marimo notebooks + HTML outputs
├── legacy/                      # Old Jupyter notebooks
├── scripts/                     # CI, setup, conversion scripts
├── todo/                        # Legacy code being migrated (~16K lines)
├── pyproject.toml               # Project config (deps, ruff, mypy, pytest)
├── Makefile                     # 14 build targets
├── Dockerfile + docker-compose.yml  # Container environment
└── .github/workflows/ci.yml    # GitHub Actions pipeline
```

---

## 5. Module Architecture

### 5.1 Dependency Flow

```
utils (io, logging, validation, storage, geometry, trajectories, subsampling)
  ↓
data (synthetic_data)
  ↓
metrics (pairwise_metrics, distributions, outliers)
  ↓
embeddings (dimensionality_reduction, visualization)
  ↓
learning (decoding, classification)
  ↓
topology (structure_index, plotting)
  ↓
plotting (PlotGrid, renderers, 1d/2d/3d/statistical/heatmaps/shape_distance/synthetic)
```

Lower layers never import from higher layers. The `plotting` module is used at every level for optional visualization but is not a hard dependency for computation.

---

### 5.2 data — Synthetic Neural Data Generation

**File:** `data/synthetic_data.py` (~2,485 lines)

Generates realistic synthetic neural datasets for testing and validation. Supports **1D, 2D, and 3D spatial environments**.

#### Cell Types

| Cell Type | Algorithm | Key Parameters |
|-----------|-----------|----------------|
| **Place cells** | Gaussian/anisotropic place fields | `field_size`, `peak_rate`, `noise_level` |
| **Grid cells** | Hexagonal lattice (2D), FCC tetrahedral (3D), harmonic (1D) | `grid_spacing`, `grid_orientation` |
| **Head direction cells** | Von Mises circular tuning curves | `tuning_width`, `peak_rate` |
| **Random cells** | Correlated noise with temporal smoothing | `baseline_rate`, `variability`, `temporal_smoothness` |

#### Key Functions

- **`generate_data(dataset_type, ...)`** — Unified `match/case` dispatcher to all generators
- **`generate_position_trajectory(...)`** — Ornstein-Uhlenbeck speed process with wall-bouncing in 1D/2D/3D arenas
- **`generate_mixed_population_flexible(cell_config, ...)`** — Configurable multi-type population generation
- **`map_to_ring(activity, positions)`** — Map 1D activity to ring manifold (S¹)
- **`map_to_torus(activity, positions)`** — Map 2D activity to torus manifold (T²)
- **`generate_shape_distance_datasets(...)`** — SVD-based cluster embedding for shape metric validation

All neural generators return `(activity: ndarray, metadata: dict)` tuples. Optional auto-plotting via PlotGrid on generation.

---

### 5.3 metrics — Distance, Distribution, and Shape Metrics

Three submodules totaling ~4,150 lines.

#### 5.3.1 pairwise_metrics.py (~2,033 lines) — Unified Pairwise Metrics

**Metric Categories (typed constants):**

| Category | Metrics |
|----------|---------|
| `POINT_TO_POINT_METRICS` | euclidean, manhattan, cosine, mahalanobis |
| `DISTRIBUTION_METRICS` | wasserstein, ks, jsd |
| `SHAPE_METRICS` | procrustes, one_to_one, soft_matching |
| `SCALAR_METRICS` | correlation, cosine_similarity |

**Core API:**

- **`compute_pairwise_matrix(x, y, metric, parallel)`** — Unified dispatcher for all metric types
- **`compare_datasets(data, data2, mode, metric)`** — Orchestration API: auto-routes `within`/`between`/`all-pairs` comparisons with HDF5 caching
- **`spatial_autocorrelation(activity, positions, arena_size)`** — FFT-based 1D/2D/3D spatial autocorrelation
- **`similarity_matrix(data, method)`** — Correlation/cosine/angular similarity with optional heatmap plot
- Numba-accelerated parallel implementations for Euclidean, cosine, Manhattan, Spearman, Kendall

#### 5.3.2 distributions.py (~1,907 lines) — Statistical and Shape Distances

**Shape Distance Methods (hierarchical complexity):**

| Method | Algorithm | Complexity |
|--------|-----------|------------|
| **Procrustes** | SVD of cross-correlation matrix: $R = UV^T$, distance = $\|X - RY\|_F / \sqrt{N}$ | $O(N \cdot D^2)$ |
| **One-to-one** | Hungarian assignment on pairwise cost matrix | $O(N^3)$ |
| **Soft matching** | Earth Mover's Distance (exact) or Sinkhorn (approximate with entropic regularization) | $O(N^2)$ to $O(N^3)$ |

**Distribution Metrics:**

- **Wasserstein** — Sum of per-feature 1D Wasserstein distances
- **Kolmogorov-Smirnov** — Max KS statistic across features
- **Jensen-Shannon Divergence** — Adaptive binning for high-D histograms

**Key Functions:**

- **`shape_distance(mtx1, mtx2, method, metric)`** — Main shape distance API with automatic subsampling
- **`modify_matrix(mtx, whiten, normalize, scale_variance)`** — Matrix preprocessing (centering, scaling, normalization)
- **`align_mtx(mtx1, mtx2, rotate, scale)`** — Procrustes alignment
- **`pairwise_distribution_comparison_batch(data, metrics, ...)`** — Batch all-pairs with HDF5 persistence and incremental resume

#### 5.3.3 outliers.py (~210 lines) — Outlier Detection

Five methods via `match/case` dispatch:

| Method | Algorithm |
|--------|-----------|
| IQR | Per-feature IQR with 1.5× multiplier |
| Z-score | Robust z-score using MAD |
| Isolation Forest | sklearn `IsolationForest` |
| LOF | sklearn `LocalOutlierFactor` (default) |
| Elliptic Envelope | sklearn `EllipticEnvelope` |

---

### 5.4 embeddings — Dimensionality Reduction

**File:** `embeddings/dimensionality_reduction.py` (~380 lines)

**7 embedding methods** via unified `compute_embedding()` API:

| Method | Backend | Notes |
|--------|---------|-------|
| PCA | sklearn | Default; supports `explained_variance` analysis |
| UMAP | umap-learn | Optional; graceful fallback if unavailable |
| t-SNE | sklearn | Perplexity auto-adjusted for small datasets |
| MDS | sklearn | Supports precomputed distance matrices |
| Isomap | sklearn | Supports precomputed distances |
| LLE | sklearn | Locally linear embedding |
| Spectral | sklearn | Spectral embedding via graph Laplacian |

**Additional:**

- **`compute_multiple_embeddings(data, methods)`** — Compare several methods at once
- **`pca_explained_variance(data, n_components)`** — Variance analysis with 90%/95%/99% thresholds

**Visualization** (`embeddings/visualization.py`, ~340 lines):

- **`plot_multiple_embeddings(embeddings, labels)`** — Auto-layout grid comparing embedding methods via PlotGrid
- **`plot_pca_variance(variance_info)`** — Scree plot with cumulative variance curves

---

### 5.5 learning — Decoding and Classification

#### 5.5.1 decoding.py (~340 lines)

| Function | Description |
|----------|-------------|
| `population_vector_decoder` | Weighted-average or peak decoding using known tuning properties |
| `knn_decoder` | k-NN regression wrapper |
| `cross_validated_knn_decoder` | k-fold CV with R², MSE, Euclidean error |
| `compare_highd_lowd_decoding` | Compare raw neural activity vs. embedding decoding quality |
| `evaluate_decoder` | Unified train/test evaluation interface |

The **`compare_highd_lowd_decoding`** function is the key analysis tool — it evaluates whether dimensionality reduction preserves decodable information by comparing decoding accuracy on original high-D activity vs. low-D embeddings, returning `performance_ratio` and `information_preserved` metrics.

#### 5.5.2 classification.py (~816 lines)

**Supervised classification** — 9 methods:
`random_forest`, `svc`, `svc_rbf`, `logistic_regression`, `knn`, `naive_bayes`, `mlp`, `gradient_boosting`, `adaboost`

**Unsupervised clustering** — 7 methods:
`kmeans`, `dbscan`, `agglomerative`, `gaussian_mixture`, `spectral`, `birch`, `mean_shift`

**Key capabilities:**

- **`extract_cell_features(activity, metadata, positions)`** — Automated feature extraction: mean rate, CV, sparsity, spatial information, periodicity (autocorrelation-based), head direction tuning strength
- **`compare_classifiers(...)`** / **`compare_clusterers(...)`** — Benchmark all methods with timing, returning ranked performance tables
- Factory-pattern via `match/case` for classifier/clusterer instantiation

---

### 5.6 topology — Structure Index

**File:** `topology/structure_index.py` (~1,306 lines)

The **Structure Index (SI)** quantifies how well behavioral structure (e.g., spatial position) is preserved in neural population activity space.

#### Algorithm

1. **Bin** neural data by behavioral label → N-dimensional grid
2. **Compute** pairwise overlap between bins via k-NN (or radius) neighborhood queries
3. **Build** weighted directed graph of overlap
4. **Calculate:** $SI = 2 \times \left(1 - \frac{\bar{d}}{n_{bins} - 1}\right) - 1$, clamped $\geq 0$
5. **Shuffle** labels `num_shuffles` times for null distribution

#### Key Functions

- **`compute_structure_index(data, label, n_bins, n_neighbors, ...)`** — Core computation
- **`compute_structure_index_sweep(data, labels, n_neighbors_list, n_bins_list, ...)`** — Parameter sweep with automatic HDF5 caching per parameter combination
- **`draw_overlap_graph(overlap_mat)`** — NetworkX directed graph visualization
- **`load_structure_index_results(save_path, dataset_name, ...)`** — Load cached results

**Performance features:** FAISS-accelerated k-NN when available, Redis caching via StorageManager, progress bars for long computations, incremental HDF5 persistence.

**Visualization** (`topology/plotting.py`, ~533 lines):

- 3-panel view: scatter + overlap heatmap + directed graph
- Parameter sweep line plots
- Multi-dataset comparison plots

---

### 5.7 plotting — PlotGrid Visualization System

The largest subsystem at **~10,000+ lines across 12 modules**. Fully dual-backend (matplotlib + plotly).

#### Architecture Layers

```
Layer 3: Convenience functions  (plot_line, plot_scatter_2d, plot_violin, ...)
    ↓ builds PlotSpec objects
Layer 2: PlotGrid orchestrator  (grid_config.py — 2,479 lines)
    ↓ dispatches per plot_type
Layer 1: Renderer functions     (renderers.py — 2,785 lines)
    ↓ calls matplotlib/plotly APIs
```

#### 16 Plot Types

| Plot Type | Backend Support | Module |
|-----------|----------------|--------|
| `scatter` | mpl + plotly | plots_2d |
| `line` | mpl + plotly | plots_1d |
| `histogram` | mpl + plotly | statistical_plots |
| `heatmap` | mpl + plotly | heatmaps |
| `scatter3d` | mpl + plotly | plots_3d |
| `violin` | mpl + plotly | statistical_plots |
| `box` | mpl + plotly | statistical_plots |
| `bar` | mpl + plotly | statistical_plots |
| `trajectory` | mpl + plotly | plots_2d |
| `trajectory3d` | mpl + plotly | plots_3d |
| `kde` | mpl + plotly | plots_2d |
| `grouped_scatter` | mpl + plotly | plots_2d |
| `convex_hull` | mpl + plotly | plots_2d |
| `boolean_states` | mpl + plotly | plots_1d |
| `ellipse` | mpl + plotly | renderers |
| `heatmap_walls` | mpl + plotly | renderers |

#### Core Components

- **`PlotSpec`** (dataclass, ~40 fields) — Declarative plot description: data, type, styling, annotations, reference lines
- **`PlotConfig`** (dataclass, ~20 fields) — Cross-plot configuration: titles, axis labels, limits, figure size, DPI, save options
- **`GridLayoutConfig`** (dataclass) — Grid dimensions, spacing, shared axes, width/height ratios; includes `auto_size_grid(n_plots)`
- **`ColorScheme`** (dataclass) — Palette management with group-aware color assignment
- **`PlotGrid`** (class) — Central orchestrator:
  - `__init__(plot_specs, config, layout, color_scheme, backend)`
  - `from_dataframe(df, ...)` / `from_dict(data_dict, ...)` — Factory constructors
  - `plot()` — Groups specs by subplot position, creates figure grid, dispatches to matplotlib/plotly renderers
  - Colorbar deduplication via `colormap_tracker`
  - Overlap prevention between subplots

#### Special Visualization Modules

- **`synthetic_plots.py`** (~2,452 lines) — Comprehensive multi-panel visualization for synthetic neural data: population rasters, spatial rate maps (1D/2D/3D), FFT periodicity analysis, coverage heatmaps, behavior trajectories, embedding plots. Entry: `plot_synthetic_data(activity, metadata, ...)`
- **`shape_distance.py`** (~265 lines) — MDS embeddings of shape distance matrices
- **`embeddings.py`** (~305 lines) — 2D/3D embedding scatter plots with convex hull overlays

---

### 5.8 utils — Shared Utilities

#### I/O (`utils/io.py`, ~1,008 lines)

The **single source of truth** for all file I/O operations:

- **NumPy:** `save_array`, `load_array`, `update_array` (.npy/.npz)
- **HDF5:** `save_hdf5`, `load_hdf5`, `h5io` (legacy compat)
- **Hierarchical results:** `save_result_to_hdf5_dataset`, `load_results_from_hdf5_dataset` — creates `/{dataset_name}/{result_key}/` structure with scalar attributes + array datasets, optionally syncing to Redis/DuckDB
- **Batch operations:** `save_comparison_batch`, `get_missing_comparisons` (incremental resume)
- **Inspection:** `get_hdf5_dataset_names`, `get_hdf5_result_summary`

#### Comparison Store (`utils/comparison_store.py`, ~638 lines)

High-level comparison storage API wrapping `io.py`:

- `save_comparison(filepath, metric, dataset_i, dataset_j, mode, value)` — Organizes as `/{metric}/{di}___{dj}/`
- `load_comparison(filepath, metric, dataset_i, dataset_j)` — Reconstructs scalar/matrix/dict values
- `query_comparisons(filepath, metric, ...)` — Returns metadata DataFrame
- `try_load_cached_comparison` / `save_comparison_result` — Transparent caching for expensive computations

Key scheme: `/{metric}/{dataset_i}___{dataset_j}/` within HDF5. Every comparison records an ISO-format UTC timestamp.

#### Logging (`utils/logging.py`, ~216 lines)

- **`configure_logging(level, fmt, stream, file_path)`** — One-shot configuration; reads `NEURAL_ANALYSIS_LOG_LEVEL` env var
- **`get_logger(name)`** — Returns `"neural_analysis.{name}"` logger
- **`log_section(title)`** — Visual `====` separator
- **`log_kv(prefix, mapping)`** — Structured key-value logging
- **`@log_calls(level, timeit)`** — Decorator for automatic entry/exit/timing logs

All modules use `get_logger(__name__)`. No `print()` in library code.

#### Other Utilities

| Module | Size | Purpose |
|--------|------|---------|
| `validation.py` | ~37 lines | `do_critical(exc, message)` — fail-fast with CRITICAL log |
| `geometry.py` | ~129 lines | `compute_convex_hull(x, y)`, `compute_kde_2d(x, y, bandwidth)` |
| `trajectories.py` | ~109 lines | `prepare_trajectory_segments(x, y, z)` for LineCollection, `compute_colors(n_points)` |
| `subsampling.py` | ~140 lines | `run_with_subsampling(func, arrays, subsamples, repeats)` — repeated random subsampling with full provenance |
| `preprocessing.py` | ~10 lines | Deprecated shim; directs to sklearn |

---

## 6. Storage Architecture

### 6.1 Three-Layer Stack

```
┌─────────────────────────────────────┐
│         StorageManager              │  ← Unified API
│  save_data / load_data / query_data │
├──────────┬──────────┬───────────────┤
│  Redis   │  DuckDB  │    HDF5       │
│  (cache) │  (index) │  (persistent) │
│ Optional │ Optional │   Required    │
└──────────┴──────────┴───────────────┘
```

**Cascade order:** Redis (fastest) → DuckDB (fast queries) → HDF5 (authoritative store)  
**Graceful degradation:** Missing Redis or DuckDB never crashes the system; each layer checks `is_available()`.

### 6.2 HDF5 — Persistent Ground Truth

**Always available.** Hierarchical structure:

```
file.h5
├── {dataset_name}/
│   ├── {result_key}/
│   │   ├── [attributes]: scalar metadata (n_bins, metric, value, etc.)
│   │   └── [datasets]: array data (overlap_matrix, shuffled_si, etc.)
│   └── ...
└── ...
```

Two concrete schemas:

| Schema | File | Key Format | Attributes | Arrays |
|--------|------|------------|------------|--------|
| Distribution comparisons | `distribution_comparisons.h5` | `{di}_vs_{dj}_{metric}` | dataset_i/j, metric, value, n_samples, n_features | pair_indices, pair_values |
| Structure index | `structure_indices.h5` | `nbins{n}_nneigh{k}` | structure_index, n_bins, n_neighbors, num_shuffles | overlap_matrix, shuffled_si |

Compression: gzip level 6, shuffle filter enabled, chunk threshold 10,000.

### 6.3 DuckDB — Metadata Index

**Optional disk-backed SQL.** Three tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `datasets` | Track HDF5 groups | id (PK), file_path, group_path, metadata_json |
| `comparisons` | Index pairwise results | dataset_i/j, metric, mode, value_type, metadata_json |
| `chunks` | Track data chunks | dataset_id (FK), chunk_key, offset_start/end, size_bytes |

File: `.neural_analysis_meta.db`. Uses `INSERT OR REPLACE` for idempotent re-indexing. JSON metadata column for flexible schema-less attributes.

### 6.4 Redis — In-Memory Cache

**Optional.** Namespace-prefixed keys: `cache:{namespace}:{key}`

Features:
- Pickle protocol (highest) for serialization
- Size guard: rejects items exceeding `cache_max_size_mb` (default 100 MB)
- Configurable TTL (default 3600s)
- `SCAN_ITER` + batch `DELETE` for pattern-based invalidation
- Auth support via `redis_password`

### 6.5 StorageManager Orchestrator

Context-manager compatible (`with StorageManager() as sm:`):

```python
# Save: indexes in Redis + DuckDB, actual data in HDF5 via io.py
sm.save_data(key, data, metadata, file_path, group_path)

# Load: checks Redis first, falls back to HDF5 loader
sm.load_data(key, hdf5_loader=lambda: load_from_hdf5(...))

# Query: SQL on DuckDB metadata
sm.query_data(filters={"metric": "procrustes"})
```

Configuration via environment variables:

| Variable | Default |
|----------|---------|
| `NEURAL_ANALYSIS_USE_REDIS` | true |
| `NEURAL_ANALYSIS_REDIS_HOST` | localhost |
| `NEURAL_ANALYSIS_REDIS_PORT` | 6379 |
| `NEURAL_ANALYSIS_USE_SQL` | true |
| `NEURAL_ANALYSIS_SQL_PATH` | .neural_analysis_meta.db |
| `NEURAL_ANALYSIS_CACHE_TTL` | 3600 |
| `NEURAL_ANALYSIS_CACHE_NAMESPACE` | neural_analysis |

---

## 7. PlotGrid System Deep Dive

The PlotGrid system is the **mandatory** plotting pathway. All new visualization code must use it.

### Pipeline Flow

```
User code
  ↓
plot_line(data, x, color="blue", backend="matplotlib")
  ↓ builds
PlotSpec(data={'x': x, 'y': data}, plot_type='line', color='blue')
  ↓ passed to
PlotGrid(plot_specs=[spec], config=PlotConfig(title=...))
  ↓ calls
PlotGrid.plot()
  ↓ dispatches via plot_type to
render_line_matplotlib(ax, data, color, ...)  OR  render_line_plotly(data, color, ...)
  ↓ returns
matplotlib Figure  OR  plotly Figure
```

### PlotSpec Highlights

The `PlotSpec` dataclass supports ~40 fields covering:

- **Data:** dict, ndarray, or DataFrame (auto-normalized)
- **Styling:** color, marker, marker_size, line_width, linestyle, alpha
- **Coloring:** cmap, colorbar, colorbar_label, color_by, colors array
- **Reference lines:** `hlines=[{y, color, linestyle, label}]`, `vlines=[{x, ...}]`
- **Annotations:** `annotations=[{text, xy, xytext, arrowprops}]`
- **Statistical:** show_points, showmeans, showmedians, notch
- **Boolean states:** true_color, false_color, true_label, false_label
- **Ellipses:** centers, widths, heights, angles (1D/2D/3D)
- **KDE:** n_levels, bandwidth, fill

### Multi-Panel Layouts

```python
# Auto-sized grid
layout = GridLayoutConfig()
layout.auto_size_grid(n_plots=6)  # → rows=2, cols=3

# Custom ratios
layout = GridLayoutConfig(rows=2, cols=2, width_ratios=[2, 1], height_ratios=[1, 1])
```

### Factory Constructors

```python
# From DataFrame
grid = PlotGrid.from_dataframe(df, x_col='time', y_col='value', group_col='condition')

# From dict
grid = PlotGrid.from_dict({'A': data_a, 'B': data_b}, plot_type='violin')
```

---

## 8. Testing and Quality

### Test Suite

- **65+ test files** in `tests/`
- **181+ tests passing** (as of last update)
- Test categories:
  - Unit tests per module (plotting, metrics, utils, storage, etc.)
  - Integration tests (phase3 comparison functions)
  - Both backends tested (matplotlib + plotly where applicable)
  - Edge cases: single-point trajectories, empty inputs, invalid parameters

### Test Naming Convention

```
test_<what>_<condition>_<expected>
```

### Coverage Configuration

- Source: `src/`
- Branch coverage enabled
- Reports: terminal (show_missing) + HTML + XML
- Excludes: `__init__.py`, `pragma: no cover`, `TYPE_CHECKING`, `__repr__`, `abstractmethod`

### Quality Checks (in order)

```bash
uv run ruff check src tests --fix    # Lint (15 rule sets)
uv run ruff format .                  # Format
uv run mypy src tests                 # Type check (strict mode)
uv run pytest -v -n auto --cov       # Tests + coverage
./scripts/run_ci_locally.sh           # Full CI via act + Docker
```

---

## 9. CI/CD Pipeline

### GitHub Actions (`ci.yml`)

Currently set to **manual dispatch only** (automatic triggers commented out).

**Job 1: `test`** (ubuntu-latest, Python 3.12)
1. Checkout → setup uv (with cache) → install Python
2. Verify lockfile integrity
3. Install dependencies
4. `ruff check` (GitHub output format)
5. `ruff format --check` (continue-on-error)
6. `mypy` type check (continue-on-error)
7. `pytest` with coverage (maxfail=5, XML + HTML reports)
8. Upload coverage to Codecov

**Job 2: `docs`** (depends on test)
1. Build Sphinx documentation (`cd docs && uv run make html`)
2. Upload HTML artifact (7-day retention)

**Note:** mypy and ruff format are non-blocking (`continue-on-error: true`).

### Local CI

```bash
make check            # Quick: lint + format + type + test
make ci               # Full: runs GitHub Actions locally via act + Docker
./scripts/run_ci_locally.sh  # Direct script
```

---

## 10. Development Workflow and Conventions

### Branch Strategy

- **Never push to `main`**
- Feature branches: `feat/...`, `fix/...`, `chore/...`
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `refactor:`

### UV-Only Execution

```bash
# NEVER use python or pip directly
uv run pytest -v           # ✓
uv run ruff check .        # ✓
uv add numpy               # ✓
python -m pytest            # ✗ FORBIDDEN
pip install numpy           # ✗ FORBIDDEN
```

### Code Standards

- Type hints on all function signatures
- Google-style docstrings on all public items
- `@log_calls` decorator on public functions
- No `print()` in library code — use structured logging
- No direct matplotlib/plotly calls — use PlotGrid
- No raw Redis key construction — use StorageManager

### File Lifecycle

When adding/modifying functions:
1. Update `docs/function_registry.md`
2. Add/extend tests in `tests/`
3. Update all call sites if renaming/moving
4. Add follow-ups to `TODO.md`

---

## 11. Legacy Code and Migration Status

### Legacy Code Location

The `todo/` directory contains **~16,134 lines** of legacy research code awaiting migration:

| File | Lines | Status | Target |
|------|-------|--------|--------|
| `Manimeasure.py` | ~3,500 | **TOP PRIORITY** | embeddings/, learning/, metrics/ |
| `Visualizer.py` | ~7,586 | Partially migrated | plotting/ |
| `Helper.py` | ~4,731 | Needs categorization | utils/ |
| `structure_index.py` | (migrated) | **Done** | topology/ |
| `yaml_creator.py` | ~200 | Low priority | scripts/ |
| `restructure.py` | ~100 | Utility | — |

Additional legacy files: `calculations.py`, `Classes.py`, `Datasets.py`, `ModelCls.py`, `Setups.py`, `SignalProcessing.py`, `temporary.py`.

### Migration Roadmap (from TODO.md)

| Phase | Focus | Hours Est. |
|-------|-------|------------|
| Phase 1: Critical | Core analysis (Manimeasure) | 50-73 |
| Phase 2: Visualization | Remaining Visualizer.py | 15-25 |
| Phase 3: Helpers | Helper.py utilities | 11-21 |
| Phase 4: Cleanup | Legacy removal | 6-9 |
| **Total** | | **82-128 hours** |

### Planned Future Features

- Grid cell torus mapping
- Tangling metric
- Power-law PCA analysis
- Graph analysis module (20-30 hours estimated)
- High-dimensional property integration

---

## 12. Docker Environment

### Dockerfile

- Base: `python:3.12-slim`
- Includes: build-essential, git, curl, redis-server
- Installs: `pip install -e ".[dev,storage]"`

### docker-compose.yml

Two services:

| Service | Image | Purpose | Ports |
|---------|-------|---------|-------|
| `app` | Built from Dockerfile | Dev environment | 8888 (Jupyter), 6379 (Redis) |
| `redis` | redis:7-alpine | Dedicated Redis | 6379 |

Environment variables auto-configured for the compose stack:
```
NEURAL_ANALYSIS_USE_REDIS=true
NEURAL_ANALYSIS_REDIS_HOST=redis
NEURAL_ANALYSIS_USE_SQL=true
NEURAL_ANALYSIS_SQL_PATH=/app/.neural_analysis_meta.db
```

---

## 13. Key Design Patterns

| Pattern | Where Used | Description |
|---------|-----------|-------------|
| **match/case dispatch** | synthetic_data, pairwise_metrics, distributions, outliers, classification, dimensionality_reduction | Python 3.10+ structural pattern matching for method/metric selection |
| **`@log_calls` decorator** | All public functions | Automatic entry/exit/timing instrumentation |
| **Declarative specs** | PlotGrid system | `PlotSpec` dataclass describes what to plot; rendering is separate |
| **Three-layer storage** | StorageManager | Redis → DuckDB → HDF5 with graceful degradation |
| **Graceful optional deps** | numba, umap, POT, faiss, redis, duckdb | `try/except` at import; fallback to pure Python/scipy |
| **Unified API entry points** | `generate_data`, `compute_pairwise_matrix`, `shape_distance`, `compute_embedding`, `filter_outlier` | Single function dispatches to multiple implementations |
| **Factory pattern** | classification.py | `_get_supervised_classifier` / `_get_unsupervised_clusterer` via match/case |
| **`(data, metadata)` returns** | All neural generators | Consistent tuple format for data + provenance |
| **Context manager** | `StorageManager` | Resource cleanup via `with` blocks |
| **Lazy imports** | utils `__init__`, metrics `__init__`, io.py | `importlib.import_module` or in-function imports to avoid circular deps |
| **Auto-save/load caching** | comparison_store, structure_index | `try_load_cached → compute → save_comparison_result` pattern |
| **Incremental resume** | `get_missing_comparisons`, sweep functions | Compare requested vs. existing results; compute only what's missing |
| **Env-var configuration** | StorageConfig, logging | All settings overridable via `NEURAL_ANALYSIS_*` variables |
| **Dual backend** | Entire plotting module | Every renderer has `_matplotlib` and `_plotly` variants |

---

## 14. Strengths

1. **Clean modular architecture** with enforced dependency flow (lower layers don't import upper)
2. **Comprehensive plotting system** — 16 plot types, dual backend, declarative API, auto-layout
3. **Sophisticated storage stack** — Three layers with graceful degradation; no data loss if Redis/DuckDB unavailable
4. **Broad metric coverage** — Point-to-point, distribution, shape metrics with multiple algorithms per category
5. **Realistic synthetic data** — 1D/2D/3D environments, multiple cell types, configurable noise
6. **Extensive documentation** — 20+ docs files covering every subsystem, architecture decisions, and usage patterns
7. **Strong tooling** — UV for reproducible environments, ruff for linting, mypy (strict), pytest with coverage
8. **Incremental resume** — Long computations can be interrupted and resumed without losing progress
9. **Optional acceleration** — Numba JIT, FAISS k-NN used when available; pure Python fallbacks otherwise
10. **Consistent logging** — Structured, searchable logs throughout; no `print()` in library code

---

## 15. Weaknesses and Technical Debt

1. **Large legacy codebase** — ~16K+ lines in `todo/` still awaiting migration (82-128 hours estimated)
2. **CI triggers disabled** — GitHub Actions set to manual dispatch only; no automated PR checks
3. **mypy and ruff format non-blocking** — CI allows type errors and formatting violations to pass
4. **Massive files** — Several files exceed 2,000 lines (`grid_config.py`: 2,479, `renderers.py`: 2,785, `synthetic_plots.py`: 2,452, `synthetic_data.py`: 2,485, `pairwise_metrics.py`: 2,033) — these would benefit from further decomposition
5. **PlotGrid dispatch** — 16-branch `if/elif` chains in `_plot_spec_matplotlib`/`_plotly` could be refactored to a registry pattern
6. **Version 0.0.0** — No versioning scheme or release process established
7. **Test file proliferation** — 65+ test files, many with `_additional`, `_comprehensive`, `_final`, `_more` suffixes suggesting incremental accumulation rather than structured organization
8. **preprocessing.py is empty** — Deprecated shim taking up module space
9. **Python 3.14 compatibility** — Blocked by Numba (noted in TODO.md)
10. **Docker setup installs via pip** — Dockerfile uses `pip install` instead of `uv`, inconsistent with project standards
11. **Marimo notebooks** — 17 example notebooks but no automated execution/validation in CI
12. **No release/publish workflow** — No PyPI publishing or versioned releases

---

## 16. Summary Statistics

| Metric | Value |
|--------|-------|
| **Total source files** | ~30 `.py` files in `src/neural_analysis/` |
| **Total source lines (est.)** | ~18,000-20,000 lines |
| **Test files** | 65+ |
| **Tests passing** | 181+ |
| **Plot types supported** | 16 |
| **Embedding methods** | 7 |
| **Distance/similarity metrics** | 10+ |
| **Shape distance methods** | 3 |
| **Supervised classifiers** | 9 |
| **Unsupervised clusterers** | 7 |
| **Cell types (synthetic)** | 4 (place, grid, HD, random) |
| **Spatial dimensions** | 1D, 2D, 3D |
| **Documentation files** | 20+ |
| **Example notebooks** | 17 (marimo) |
| **Legacy code awaiting migration** | ~16,134 lines |
| **Dependencies (core)** | 15 |
| **Dependencies (dev)** | 7 |
| **Optional dependency groups** | 2 (viz, storage) |

---

*This report was generated by deep-reading every source file, documentation page, configuration file, script, and legacy artifact in the repository.*
