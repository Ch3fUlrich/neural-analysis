# Neural Analysis Function Registry

Auto-generated function registry for all public functions in neural_analysis package.
**Last Updated:** Auto-generated

## Purpose

This registry helps developers and AI agents:
- Avoid code duplication by finding existing functions
- Understand the package structure and available functionality
- Quickly locate the right function for a task

---

## Synthetic Data

### `data.synthetic_data`

#### `add_noise(data, noise_type, noise_level, ...)`

**Returns:** `npt.NDArray[np.floating[Any]]`

**Purpose:** Add noise to data.

**Location:** `data.synthetic_data.py` (line 1604)

#### `generate_cluster_templates(n_clusters, n_features, cluster_separation, ...)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Generate cluster templates with fundamentally different structures.

**Location:** `data.synthetic_data.py` (line 2174)

#### `generate_data(dataset_type, n_samples, n_features, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | dict[str, Any]]`

**Purpose:** Generate synthetic datasets with unified interface.

**Location:** `data.synthetic_data.py` (line 75)

#### `generate_dataset_from_cluster_template(template, n_neurons, noise_scale, ...)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Generate a neural dataset from a cluster template.

**Location:** `data.synthetic_data.py` (line 2252)

#### `generate_grid_cells(n_cells, n_samples, positions, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], dict[str, Any]]`

**Purpose:** Generate grid cell firing data in 1D, 2D, or 3D.

**Location:** `data.synthetic_data.py` (line 1070)

#### `generate_head_direction(n_samples, turning_rate, seed)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Generate head direction trajectory.

**Location:** `data.synthetic_data.py` (line 788)

#### `generate_head_direction_cells(n_cells, n_samples, head_direction, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], dict[str, Any]]`

**Purpose:** Generate head direction cell firing data.

**Location:** `data.synthetic_data.py` (line 1324)

#### `generate_mixed_neural_population(n_place, n_grid, n_hd, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], dict[str, Any]]`

**Purpose:** Generate mixed population of place, grid, and head direction cells.

**Location:** `data.synthetic_data.py` (line 1529)

#### `generate_mixed_population_flexible(cell_config, n_samples, arena_size, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], dict[str, Any]]`

**Purpose:** Generate flexible mixed neural population with custom configuration.

**Location:** `data.synthetic_data.py` (line 1977)

#### `generate_place_cells(n_cells, n_samples, positions, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], dict[str, Any]]`

**Purpose:** Generate place cell firing data in 1D, 2D, or 3D.

**Location:** `data.synthetic_data.py` (line 826)

#### `generate_position_trajectory(n_samples, arena_size, speed, ...)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Generate realistic position trajectory for a freely moving animal.

**Location:** `data.synthetic_data.py` (line 720)

#### `generate_random_cells(n_cells, n_samples, baseline_rate, ...)`

**Returns:** `tuple[npt.NDArray[np.float64], dict[str, Any]]`

**Purpose:** Generate random cells with no specific tuning properties.

**Location:** `data.synthetic_data.py` (line 1416)

#### `generate_s_curve(n_samples, noise, seed)`

**Returns:** `tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]`

**Purpose:** Generate S-curve manifold dataset using scikit-learn.

**Location:** `data.synthetic_data.py` (line 1687)

#### `generate_shape_distance_datasets(n_datasets, n_clusters, min_neurons, ...)`

**Returns:** `tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.int_]]`

**Purpose:** Generate multiple neural datasets with distinct cluster structure.

**Location:** `data.synthetic_data.py` (line 2362)

#### `generate_swiss_roll(n_samples, noise, seed)`

**Returns:** `tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]`

**Purpose:** Generate Swiss roll manifold dataset using scikit-learn.

**Location:** `data.synthetic_data.py` (line 1655)

#### `map_to_ring(activity, positions, plot)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Map population activity to ring manifold (1D circular).

**Location:** `data.synthetic_data.py` (line 1719)

#### `map_to_torus(activity, positions, major_radius, ...)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Map population activity to torus manifold (2D periodic).

**Location:** `data.synthetic_data.py` (line 1838)

---

## Embeddings

### `embeddings.dimensionality_reduction`

#### `compute_embedding(data, method, n_components, ...)`

**Returns:** `npt.NDArray[np.floating]`

**Purpose:** Compute dimensionality reduction embedding using specified method.

**Location:** `embeddings.dimensionality_reduction.py` (line 65)

#### `compute_multiple_embeddings(data, methods, n_components, ...)`

**Returns:** `dict[str, npt.NDArray[np.floating]]`

**Purpose:** Compute multiple embeddings for comparison.

**Location:** `embeddings.dimensionality_reduction.py` (line 315)

#### `pca_explained_variance(data, n_components, cumulative)`

**Returns:** `dict[str, npt.NDArray[np.floating]]`

**Purpose:** Compute explained variance for PCA components.

**Location:** `embeddings.dimensionality_reduction.py` (line 408)

### `embeddings.visualization`

#### `plot_multiple_embeddings(embeddings, labels, colors, ...)`

**Returns:** `Figure`

**Purpose:** Plot multiple embeddings side-by-side for comparison.

**Location:** `embeddings.visualization.py` (line 35)

#### `plot_pca_variance(variance_info, cumulative, n_components_to_show, ...)`

**Returns:** `Figure`

**Purpose:** Plot PCA explained variance (scree plot).

**Location:** `embeddings.visualization.py` (line 261)

### `plotting.embeddings`

#### `compute_convex_hull(points)`

**Returns:** `ConvexHull | None`

**Purpose:** Compute convex hull for a set of points.

**Location:** `plotting.embeddings.py` (line 307)

#### `group_points_by_labels(points, labels)`

**Returns:** `dict[Any, npt.NDArray[np.floating[Any]]]`

**Purpose:** Group points by their labels.

**Location:** `plotting.embeddings.py` (line 339)

#### `plot_embedding(embedding, labels)`

**Returns:** `Any`

**Purpose:** Plot embeddings with automatic 2D/3D detection.

**Location:** `plotting.embeddings.py` (line 41)

#### `plot_embedding_2d(embedding, labels)`

**Returns:** `Any`

**Purpose:** Plot 2D embeddings with optional convex hulls.

**Location:** `plotting.embeddings.py` (line 121)

#### `plot_embedding_3d(embedding, labels)`

**Returns:** `Any`

**Purpose:** Plot 3D embeddings with optional convex hulls.

**Location:** `plotting.embeddings.py` (line 212)

---

## Decoding

### `learning.decoding`

#### `compare_highd_lowd_decoding(activity, embedding, labels, ...)`

**Returns:** `dict[str, Any]`

**Purpose:** Compare decoding performance on high-D activity vs low-D embedding.

**Location:** `learning.decoding.py` (line 305)

#### `cross_validated_knn_decoder(activity, labels, k, ...)`

**Returns:** `dict[str, Any]`

**Purpose:** k-NN decoder with cross-validation.

**Location:** `learning.decoding.py` (line 190)

#### `evaluate_decoder(train_activity, train_labels, test_activity, ...)`

**Returns:** `dict[str, float]`

**Purpose:** Evaluate decoder on train/test split.

**Location:** `learning.decoding.py` (line 368)

#### `knn_decoder(train_activity, train_labels, test_activity, ...)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Decode behavioral variables using k-Nearest Neighbors.

**Location:** `learning.decoding.py` (line 119)

#### `population_vector_decoder(activity, field_centers, method)`

**Returns:** `npt.NDArray[np.float64]`

**Purpose:** Decode position from population activity using population vector.

**Location:** `learning.decoding.py` (line 60)

---

## Metrics - Distance

---

## Metrics - Distributions

### `metrics.distributions`

#### `align_mtx(mtx1, mtx2, rotate, ...)`

**Returns:** `npt.NDArray[np.floating]`

**Purpose:** Align mtx2 to mtx1 using Procrustes analysis.

**Location:** `metrics.distributions.py` (line 1200)

#### `batch_comparison(datasets, comparison_fn)`

**Returns:** `pd.DataFrame`

**Purpose:** Generic batch-comparison utility for arbitrary comparison functions.

**Location:** `metrics.distributions.py` (line 1023)

#### `core_compute(a, b)`

**Returns:** `tuple[float, Dict[Tuple[int, int], float]]`

**Purpose:** No description

**Location:** `metrics.distributions.py` (line 1769)

#### `decorator(func)`

**Returns:** `Callable[..., Any]`

**Purpose:** No description

**Location:** `metrics.distributions.py` (line 74)

#### `distance_only(a, b)`

**Returns:** `float`

**Purpose:** No description

**Location:** `metrics.distributions.py` (line 1833)

#### `distribution_distance(points1, points2, mode, ...)`

**Returns:** `float | dict[str, float] | tuple[float, dict[tuple[int, int], float]]`

**Purpose:** Compute pairwise distances within or between distributions.

**Location:** `metrics.distributions.py` (line 588)

#### `get_logger(name)`

**Returns:** `logging.Logger`

**Purpose:** No description

**Location:** `metrics.distributions.py` (line 79)

#### `jensen_shannon_divergence(points1, points2, bins)`

**Returns:** `float`

**Purpose:** Compute Jensen-Shannon divergence between point distributions.

**Location:** `metrics.distributions.py` (line 464)

#### `kolmogorov_smirnov_distance(points1, points2)`

**Returns:** `float`

**Purpose:** Compute maximum Kolmogorov-Smirnov statistic over all features.

**Location:** `metrics.distributions.py` (line 402)

#### `log_calls()`

**Returns:** `Callable[[Callable[..., Any]], Callable[..., Any]]`

**Purpose:** No description

**Location:** `metrics.distributions.py` (line 71)

#### `modify_matrix(mtx, whiten, normalize, ...)`

**Returns:** `npt.NDArray[np.floating[Any]]`

**Purpose:** Preprocess matrix for shape comparison.

**Location:** `metrics.distributions.py` (line 1099)

#### `pairwise_distribution_comparison_batch(data, metrics)`

**Returns:** `pd.DataFrame`

**Purpose:** Compute all-pairs distribution comparisons with caching and persistence.

**Location:** `metrics.distributions.py` (line 831)

#### `shape_distance(mtx1, mtx2, method, ...)`

**Returns:** `tuple[Union[float, npt.NDArray[np.float64]], Union[Dict[Tuple[int, int], float], List[Dict[Tuple[int, int], float]]], Dict[str, Any]]`

**Purpose:** Compute a shape distance between two neural population activity matrices,

**Location:** `metrics.distributions.py` (line 1596)

#### `shape_distance_one_to_one(mtx1, mtx2, metric)`

**Returns:** `tuple[float, dict[tuple[int, int], float]]`

**Purpose:** One-to-one (permutation) shape distance: d_P(X, Y) = min_{Π in Π_N} ||X - Π Y||_F.

**Location:** `metrics.distributions.py` (line 1351)

#### `shape_distance_procrustes(mtx1, mtx2, return_pairs)`

**Returns:** `tuple[float, dict[tuple[int, int], float] | None]`

**Purpose:** Orthogonal Procrustes shape distance: d_O(X, Y) = min_{Q in O_N} ||X - Q Y||_F.

**Location:** `metrics.distributions.py` (line 1259)

#### `shape_distance_soft_matching(mtx1, mtx2, metric, ...)`

**Returns:** `tuple[float, dict[tuple[int, int], float]]`

**Purpose:** Soft-matching (OT/Wasserstein) distance: d_T(X,Y) = min_T sum T_ij C_ij, T in transport polytope.

**Location:** `metrics.distributions.py` (line 1460)

#### `wasserstein_distance_multi(points1, points2)`

**Returns:** `float`

**Purpose:** Compute sum of Wasserstein distances over all features.

**Location:** `metrics.distributions.py` (line 331)

---

## Metrics - Similarity

---

## Metrics - Outliers

### `metrics.outliers`

#### `decorator(func)`

**Returns:** `Any`

**Purpose:** No description

**Location:** `metrics.outliers.py` (line 22)

#### `filter_outlier(points, method, contamination, ...)`

**Returns:** `npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]`

**Purpose:** Filter outliers from a point distribution.

**Location:** `metrics.outliers.py` (line 38)

#### `get_logger(name)`

**Returns:** `Any`

**Purpose:** No description

**Location:** `metrics.outliers.py` (line 27)

#### `log_calls()`

**Returns:** `Any`

**Purpose:** No description

**Location:** `metrics.outliers.py` (line 21)

---

## Utils - IO

### `utils.io`

#### `decorator(func)`

**Returns:** `Callable[..., Any]`

**Purpose:** No description

**Location:** `utils.io.py` (line 43)

#### `get_hdf5_dataset_names(save_path)`

**Returns:** `list[str]`

**Purpose:** Get list of all top-level dataset names in HDF5 file.

**Location:** `utils.io.py` (line 837)

#### `get_hdf5_result_summary(save_path, dataset_name)`

**Returns:** `pd.DataFrame`

**Purpose:** Get summary DataFrame of all results in HDF5 file.

**Location:** `utils.io.py` (line 870)

#### `get_logger(name)`

**Returns:** `logging.Logger`

**Purpose:** No description

**Location:** `utils.io.py` (line 47)

#### `get_missing_comparisons(item_pairs, metrics_dict, df_results)`

**Returns:** `list[tuple[str, str, str]]`

**Purpose:** Determine which comparisons need to be computed.

**Location:** `utils.io.py` (line 529)

#### `h5io(path)`

**Returns:** `tuple[Any, Any] | tuple[tuple[Any, Any], dict[str, Jsonable]] | None`

**Purpose:** Compatibility wrapper replicating legacy `h5io` API.

**Location:** `utils.io.py` (line 580)

#### `load_array(path)`

**Returns:** `npt.NDArray[Any] | dict[str, npt.NDArray[Any]] | None`

**Purpose:** Load an array or dict of arrays from .npy/.npz. Returns None if missing.

**Location:** `utils.io.py` (line 259)

#### `load_distribution_comparisons(save_path, comparison_name, dataset_i, ...)`

**Returns:** `dict[str, dict[str, Any]]`

**Purpose:** Load distribution comparison results from HDF5.

**Location:** `utils.io.py` (line 931)

#### `load_hdf5(path)`

**Returns:** `tuple[Any, Any] | tuple[tuple[Any, Any], dict[str, Jsonable]]`

**Purpose:** Load previously saved HDF5 content.

**Location:** `utils.io.py` (line 369)

#### `load_results_from_hdf5_dataset(save_path, dataset_name, result_key, ...)`

**Returns:** `dict[str, dict[str, Any]]`

**Purpose:** Load analysis results from HDF5 file.

**Location:** `utils.io.py` (line 731)

#### `log_calls()`

**Returns:** `Callable[[Callable[..., Any]], Callable[..., Any]]`

**Purpose:** No description

**Location:** `utils.io.py` (line 40)

#### `save_array(path, data)`

**Returns:** `Path`

**Purpose:** Save a single array (.npy) or a dict of arrays (.npz).

**Location:** `utils.io.py` (line 222)

#### `save_comparison_batch(result_rows, df_results, save_path)`

**Returns:** `pd.DataFrame`

**Purpose:** Save batch of comparison results to HDF5.

**Location:** `utils.io.py` (line 481)

#### `save_hdf5(path, data)`

**Returns:** `None`

**Purpose:** Save a DataFrame or array with optional labels into an HDF5 file.

**Location:** `utils.io.py` (line 311)

#### `save_result_to_hdf5_dataset(save_path, dataset_name, result_key, ...)`

**Returns:** `None`

**Purpose:** Save analysis results to HDF5 file with hierarchical structure.

**Location:** `utils.io.py` (line 611)

#### `update_array(path, new_data)`

**Returns:** `Path`

**Purpose:** Update or create an .npz file by merging in new arrays.

**Location:** `utils.io.py` (line 282)

---

## Utils - Preprocessing

---

## Utils - Validation

### `utils.validation`

#### `do_critical(exc, message)`

**Returns:** `None`

**Purpose:** Log a critical error and raise the provided exception type.

**Location:** `utils.validation.py` (line 25)

#### `get_logger(name)`

**Returns:** `Any`

**Purpose:** No description

**Location:** `utils.validation.py` (line 15)

---

## Utils - Trajectories

### `utils.trajectories`

#### `compute_colors(n_points, color_by)`

**Returns:** `NDArray[np.floating]`

**Purpose:** Compute color values based on specified method.

**Location:** `utils.trajectories.py` (line 77)

#### `prepare_trajectory_segments(x, y, z)`

**Returns:** `NDArray[np.floating]`

**Purpose:** Prepare 2D or 3D trajectory data as line segments for visualization.

**Location:** `utils.trajectories.py` (line 13)

---

## Plotting - Core System

### `plotting.backend`

#### `get_backend()`

**Returns:** `BackendType`

**Purpose:** Get the current visualization backend.

**Location:** `plotting.backend.py` (line 72)

#### `set_backend(backend)`

**Returns:** `None`

**Purpose:** Set the visualization backend.

**Location:** `plotting.backend.py` (line 34)

### `plotting.core`

#### `apply_layout_matplotlib(ax, config)`

**Returns:** `None`

**Purpose:** Apply common layout (title, labels, limits, grid) for matplotlib.

**Location:** `plotting.core.py` (line 260)

#### `apply_layout_plotly(fig, config)`

**Returns:** `None`

**Purpose:** Apply common layout (title, labels, limits, grid) for Plotly.

**Location:** `plotting.core.py` (line 282)

#### `apply_layout_plotly_3d(fig, config)`

**Returns:** `None`

**Purpose:** Apply common layout for Plotly 3D plots (scene configuration).

**Location:** `plotting.core.py` (line 310)

#### `calculate_alpha(value, min_value, max_value, ...)`

**Returns:** `float | list[float]`

**Purpose:** Calculate alpha value(s) based on value's position in range.

**Location:** `plotting.core.py` (line 417)

#### `create_rgba_labels(values, alpha, cmap)`

**Returns:** `npt.NDArray[np.floating[Any]]`

**Purpose:** Create RGBA labels using a colormap.

**Location:** `plotting.core.py` (line 562)

#### `finalize_plot_matplotlib(config)`

**Returns:** `None`

**Purpose:** Handle save and show for matplotlib plots.

**Location:** `plotting.core.py` (line 364)

#### `finalize_plot_plotly(fig, config)`

**Returns:** `None`

**Purpose:** Handle save and show for plotly plots.

**Location:** `plotting.core.py` (line 377)

#### `generate_similar_colors(base_color, num_colors, hue_variation, ...)`

**Returns:** `list[tuple[float, float, float]]`

**Purpose:** Generate a list of similar colors based on a base color.

**Location:** `plotting.core.py` (line 504)

#### `get_default_categorical_colors(n)`

**Returns:** `list[str]`

**Purpose:** Return a list of n default categorical colors as hex strings.

**Location:** `plotting.core.py` (line 346)

#### `get_save_path()`

**Returns:** `Path | None`

**Purpose:** Get the full save path for the plot.

**Location:** `plotting.core.py` (line 151)

#### `make_list_if_not(value)`

**Returns:** `list[Any]`

**Purpose:** Convert value to list if it isn't already.

**Location:** `plotting.core.py` (line 683)

#### `resolve_colormap(cmap, backend)`

**Returns:** `str`

**Purpose:** Return a backend-appropriate colormap identifier.

**Location:** `plotting.core.py` (line 223)

#### `save_plot(save_path, format, dpi, ...)`

**Returns:** `None`

**Purpose:** Save current matplotlib figure to file.

**Location:** `plotting.core.py` (line 635)

### `plotting.grid_config`

#### `__init__(plot_specs, config, layout, ...)`

**Returns:** `Any`

**Purpose:** No description

**Location:** `plotting.grid_config.py` (line 497)

#### `add_plot(data, plot_type)`

**Returns:** `None`

**Purpose:** Add a plot to the grid.

**Location:** `plotting.grid_config.py` (line 629)

#### `add_trace_to_subplot(fig, trace, row, ...)`

**Returns:** `Any`

**Purpose:** Add a trace to a specific subplot in a plotly figure.

**Location:** `plotting.grid_config.py` (line 2438)

#### `auto_size_grid(n_plots)`

**Returns:** `tuple[int, int]`

**Purpose:** Automatically determine grid size from number of plots.

**Location:** `plotting.grid_config.py` (line 343)

#### `create_subplot_grid(rows, cols, config, ...)`

**Returns:** `Any`

**Purpose:** Create a multi-panel subplot grid.

**Location:** `plotting.grid_config.py` (line 2134)

#### `from_dataframe(df, data_col, plot_type_col, ...)`

**Returns:** `PlotGrid`

**Purpose:** Create PlotGrid from a pandas DataFrame.

**Location:** `plotting.grid_config.py` (line 512)

#### `from_dict(data_dict, plot_type)`

**Returns:** `PlotGrid`

**Purpose:** Create PlotGrid from a dictionary of {label: data}.

**Location:** `plotting.grid_config.py` (line 595)

#### `get_colors(groups)`

**Returns:** `dict[str, str]`

**Purpose:** Get color mapping for a list of groups.

**Location:** `plotting.grid_config.py` (line 395)

#### `plot()`

**Returns:** `Any`

**Purpose:** Generate the plot grid.

**Location:** `plotting.grid_config.py` (line 665)

#### `plot_comparison_grid(data_dict, plot_type, rows, ...)`

**Returns:** `Any`

**Purpose:** Create a grid comparing multiple datasets with the same plot type.

**Location:** `plotting.grid_config.py` (line 2012)

#### `plot_grouped_comparison(data, x_col, y_col, ...)`

**Returns:** `Any`

**Purpose:** Create overlaid plots grouped by a category.

**Location:** `plotting.grid_config.py` (line 2054)

---

## Plotting - Renderers

### `plotting.renderers`

#### `extract_xy_from_data(data)`

**Returns:** `tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]`

**Purpose:** Extract x and y coordinates from various data formats.

**Location:** `plotting.renderers.py` (line 44)

#### `extract_xyz_from_data(data)`

**Returns:** `tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]`

**Purpose:** Extract x, y, and z coordinates from various data formats.

**Location:** `plotting.renderers.py` (line 79)

#### `render_bar_matplotlib(ax, data, x, ...)`

**Returns:** `Any`

**Purpose:** Render a bar plot using matplotlib.

**Location:** `plotting.renderers.py` (line 1094)

#### `render_bar_plotly(data, x, color, ...)`

**Returns:** `go.Bar`

**Purpose:** Render a bar plot using plotly.

**Location:** `plotting.renderers.py` (line 1219)

#### `render_boolean_states_matplotlib(ax, x, states, ...)`

**Returns:** `Any`

**Purpose:** Render boolean states as filled regions using matplotlib.

**Location:** `plotting.renderers.py` (line 2376)

#### `render_boolean_states_plotly(x, states, true_color, ...)`

**Returns:** `Any`

**Purpose:** Render boolean states as filled regions using plotly.

**Location:** `plotting.renderers.py` (line 2469)

#### `render_box_matplotlib(ax, data, position, ...)`

**Returns:** `Any`

**Purpose:** Render a box plot with sample points using matplotlib.

**Location:** `plotting.renderers.py` (line 1560)

#### `render_box_plotly(data, color, alpha, ...)`

**Returns:** `go.Box`

**Purpose:** Render a box plot with sample points using plotly.

**Location:** `plotting.renderers.py` (line 1643)

#### `render_convex_hull_matplotlib(ax, hull_x, hull_y, ...)`

**Returns:** `Any`

**Purpose:** Render a convex hull boundary using matplotlib.

**Location:** `plotting.renderers.py` (line 2248)

#### `render_convex_hull_plotly(hull_x, hull_y, color, ...)`

**Returns:** `Any`

**Purpose:** Render a convex hull boundary using plotly.

**Location:** `plotting.renderers.py` (line 2313)

#### `render_ellipse_matplotlib(ax, centers, widths, ...)`

**Returns:** `Any`

**Purpose:** Render ellipses using matplotlib patches.

**Location:** `plotting.renderers.py` (line 2570)

#### `render_ellipse_plotly(centers, widths, heights, ...)`

**Returns:** `list[Any]`

**Purpose:** Render ellipses using plotly shapes.

**Location:** `plotting.renderers.py` (line 2689)

#### `render_heatmap_matplotlib(ax, data, cmap, ...)`

**Returns:** `Any`

**Purpose:** Render a heatmap using matplotlib.

**Location:** `plotting.renderers.py` (line 790)

#### `render_heatmap_plotly(data, cmap, colorscale)`

**Returns:** `go.Heatmap`

**Purpose:** Render a heatmap using plotly.

**Location:** `plotting.renderers.py` (line 1039)

#### `render_heatmap_walls_matplotlib(ax, data, cmap, ...)`

**Returns:** `Any`

**Purpose:** Render three orthogonal heatmap projections on the walls of a 3D axes.

**Location:** `plotting.renderers.py` (line 889)

#### `render_histogram_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a histogram using matplotlib.

**Location:** `plotting.renderers.py` (line 700)

#### `render_histogram_plotly(data, color, alpha, ...)`

**Returns:** `go.Histogram`

**Purpose:** Render a histogram using plotly.

**Location:** `plotting.renderers.py` (line 737)

#### `render_kde_matplotlib(ax, xi, yi, ...)`

**Returns:** `Any`

**Purpose:** Render a 2D KDE plot using matplotlib contour/contourf.

**Location:** `plotting.renderers.py` (line 2102)

#### `render_kde_plotly(xi, yi, zi, ...)`

**Returns:** `Any`

**Purpose:** Render a 2D KDE plot using plotly contour.

**Location:** `plotting.renderers.py` (line 2170)

#### `render_line_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a line plot using matplotlib with optional error bands.

**Location:** `plotting.renderers.py` (line 414)

#### `render_line_plotly(data, color, line_width, ...)`

**Returns:** `go.Scatter`

**Purpose:** Render a line plot using plotly with optional error bands.

**Location:** `plotting.renderers.py` (line 566)

#### `render_scatter3d_plotly(data, color, colors, ...)`

**Returns:** `go.Scatter3d`

**Purpose:** Render a 3D scatter plot using plotly.

**Location:** `plotting.renderers.py` (line 325)

#### `render_scatter_matplotlib(ax, data, color, ...)`

**Returns:** `Any`

**Purpose:** Render a 2D scatter plot using matplotlib.

**Location:** `plotting.renderers.py` (line 151)

#### `render_scatter_plotly(data, color, colors, ...)`

**Returns:** `go.Scatter`

**Purpose:** Render a 2D scatter plot using plotly.

**Location:** `plotting.renderers.py` (line 238)

#### `render_trajectory3d_matplotlib(ax, x, y, ...)`

**Returns:** `Any`

**Purpose:** Render a 3D trajectory using matplotlib Line3DCollection.

**Location:** `plotting.renderers.py` (line 1998)

#### `render_trajectory3d_plotly(x, y, z, ...)`

**Returns:** `Any`

**Purpose:** Render a 3D trajectory using plotly.

**Location:** `plotting.renderers.py` (line 1911)

#### `render_trajectory_matplotlib(ax, x, y, ...)`

**Returns:** `Any`

**Purpose:** Render a 2D trajectory using matplotlib LineCollection.

**Location:** `plotting.renderers.py` (line 1713)

#### `render_trajectory_plotly(x, y, colors, ...)`

**Returns:** `Any`

**Purpose:** Render a 2D trajectory using plotly.

**Location:** `plotting.renderers.py` (line 1816)

#### `render_violin_matplotlib(ax, data, position, ...)`

**Returns:** `Any`

**Purpose:** Render a half violin plot (right side) with points on the left using matplotlib.

**Location:** `plotting.renderers.py` (line 1296)

#### `render_violin_plotly(data, color, alpha, ...)`

**Returns:** `go.Violin`

**Purpose:** Render a half violin plot (right side) with points on the left using plotly.

**Location:** `plotting.renderers.py` (line 1463)

---

## Plotting - Statistical

### `plotting.statistical_plots`

#### `plot_bar(data, labels, colors, ...)`

**Returns:** `Any`

**Purpose:** Create a bar plot for comparing multiple groups.

**Location:** `plotting.statistical_plots.py` (line 46)

#### `plot_box(data, labels, colors, ...)`

**Returns:** `Any`

**Purpose:** Create box plots for comparing distributions.

**Location:** `plotting.statistical_plots.py` (line 283)

#### `plot_comparison_distributions(data, plot_type, rows, ...)`

**Returns:** `Any`

**Purpose:** Create separate distribution plots for each group in a grid.

**Location:** `plotting.statistical_plots.py` (line 442)

#### `plot_grouped_distributions(data, plot_type, colors, ...)`

**Returns:** `Any`

**Purpose:** Create multiple distribution plots grouped by category.

**Location:** `plotting.statistical_plots.py` (line 358)

#### `plot_violin(data, labels, colors, ...)`

**Returns:** `Any`

**Purpose:** Create violin plots for comparing distributions.

**Location:** `plotting.statistical_plots.py` (line 195)

---

## Plotting - 1D/2D/3D

### `plotting.plots_1d`

#### `plot_boolean_states(states, x, config, ...)`

**Returns:** `Any`

**Purpose:** Visualize boolean states over time using PlotGrid.

**Location:** `plotting.plots_1d.py` (line 176)

#### `plot_line(data, x, config, ...)`

**Returns:** `Any`

**Purpose:** Plot a 1D line with optional error bands using PlotGrid.

**Location:** `plotting.plots_1d.py` (line 23)

#### `plot_multiple_lines(data_dict, x, config, ...)`

**Returns:** `Any`

**Purpose:** Plot multiple lines on the same axes using PlotGrid.

**Location:** `plotting.plots_1d.py` (line 99)

### `plotting.plots_2d`

#### `plot_grouped_scatter_2d(group_data, config, show_hulls, ...)`

**Returns:** `Any`

**Purpose:** Plot grouped scatter data with optional convex hulls.

**Location:** `plotting.plots_2d.py` (line 190)

#### `plot_kde_2d(x, y, config, ...)`

**Returns:** `Any`

**Purpose:** Create a 2D KDE (kernel density estimation) plot.

**Location:** `plotting.plots_2d.py` (line 256)

#### `plot_scatter_2d(x, y, config, ...)`

**Returns:** `Any`

**Purpose:** Create a 2D scatter plot.

**Location:** `plotting.plots_2d.py` (line 43)

#### `plot_trajectory_2d(x, y, config, ...)`

**Returns:** `Any`

**Purpose:** Plot a 2D trajectory with line connecting points.

**Location:** `plotting.plots_2d.py` (line 124)

### `plotting.plots_3d`

#### `plot_scatter_3d(x, y, z, ...)`

**Returns:** `Any`

**Purpose:** Create a 3D scatter plot.

**Location:** `plotting.plots_3d.py` (line 46)

#### `plot_trajectory_3d(x, y, z, ...)`

**Returns:** `Any`

**Purpose:** Plot a 3D trajectory with line connecting points.

**Location:** `plotting.plots_3d.py` (line 124)

---

## Plotting - Specialized

### `plotting.embeddings`

#### `compute_convex_hull(points)`

**Returns:** `ConvexHull | None`

**Purpose:** Compute convex hull for a set of points.

**Location:** `plotting.embeddings.py` (line 307)

#### `group_points_by_labels(points, labels)`

**Returns:** `dict[Any, npt.NDArray[np.floating[Any]]]`

**Purpose:** Group points by their labels.

**Location:** `plotting.embeddings.py` (line 339)

#### `plot_embedding(embedding, labels)`

**Returns:** `Any`

**Purpose:** Plot embeddings with automatic 2D/3D detection.

**Location:** `plotting.embeddings.py` (line 41)

#### `plot_embedding_2d(embedding, labels)`

**Returns:** `Any`

**Purpose:** Plot 2D embeddings with optional convex hulls.

**Location:** `plotting.embeddings.py` (line 121)

#### `plot_embedding_3d(embedding, labels)`

**Returns:** `Any`

**Purpose:** Plot 3D embeddings with optional convex hulls.

**Location:** `plotting.embeddings.py` (line 212)

### `plotting.heatmaps`

#### `plot_heatmap(data, config, cmap, ...)`

**Returns:** `Any`

**Purpose:** Create a heatmap visualization using PlotGrid.

**Location:** `plotting.heatmaps.py` (line 23)

### `plotting.synthetic_plots`

#### `calc_grid(n_total)`

**Returns:** `tuple[int, int]`

**Purpose:** Calculate grid dimensions for given number of plots.

**Location:** `plotting.synthetic_plots.py` (line 300)

#### `plot_synthetic_data(activity, metadata, show_raster, ...)`

**Returns:** `Figure`

**Purpose:** Plot comprehensive visualization of synthetic neural data.

**Location:** `plotting.synthetic_plots.py` (line 776)

---
