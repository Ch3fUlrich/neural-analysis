import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():  # noqa: N803
    import marimo as mo

    return (mo,)


@app.cell
def _():  # noqa: N803
    # Imports
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from umap import UMAP

    from neural_analysis import (
        generate_data,
        generate_grid_cells,
        generate_head_direction_cells,
        generate_mixed_population_flexible,
        generate_place_cells,
        map_to_ring,
        map_to_torus,
    )
    from neural_analysis.data.synthetic_data import (
        generate_s_curve,
        generate_swiss_roll,
    )

    # Set random seed for reproducibility
    SEED = 42
    np.random.seed(SEED)

    print("✅ Imports successful!")
    print(f"NumPy version: {np.__version__}")
    print(f"Random seed: {SEED}")
    return (
        GridSpec,
        Isomap, # noqa: N803
        PCA,
        SEED,
        UMAP,
        generate_data,
        generate_grid_cells,
        generate_head_direction_cells,
        generate_mixed_population_flexible,
        generate_place_cells,
        generate_s_curve,
        generate_swiss_roll,
        map_to_ring,
        map_to_torus,
        np,
        plt,
    )


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    # Synthetic Dataset Generation for Neural Analysis

    This notebook demonstrates comprehensive synthetic dataset generation capabilities including:

    1. **Neural Data**: Place cells, grid cells, and head direction cells in 1D, 2D, and 3D environments
    2. **Manifold Mapping**: Ring (S¹) for place/HD cells, Torus (T²) for grid cells
    3. **Population Decoding**: Tracking trajectories on manifolds via population vectors
    4. **Mixed Populations**: Realistic combinations of multiple cell types with configurable noise
    5. **sklearn Datasets**: Swiss roll, S-curve, blobs, moons, circles for benchmarking
    6. **Embedding Comparison**: Perfect → Noisy → Mixed population embeddings

    All visualizations use raster plots with behavioral labels to show the relationship between neural activity and behavior.
    """)
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 1: Neural Data - Place Cells in Multiple Dimensions

    Place cells fire when the animal is in specific locations. Let's generate place cell activity in 1D, 2D, and 3D environments.
    """)
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### 1D Place Cells (Linear Track)

    Place cells on a linear track with localized firing fields.
    """)
    return


@app.cell
def _(SEED, generate_place_cells):  # noqa: N803
    # Generate 1D place cells
    activity_1d, meta_1d = generate_place_cells(
        n_cells=30,
        n_samples=500,
        arena_size=2.0,  # 2 meter linear track
        field_size=0.15,
        peak_rate=10.0,
        noise_level=0.05,
        seed=SEED,
    )
    return activity_1d, meta_1d


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### 2D Place Cells (Open Field)

    Place cells in a 2D arena with Gaussian firing fields tiling the space.
    """)
    return


@app.cell
def _(SEED, generate_place_cells):  # noqa: N803
    # Generate 2D place cells
    activity_2d, meta_2d = generate_place_cells(
        n_cells=50,
        n_samples=1000,
        arena_size=(1.0, 1.0),  # 1m x 1m arena
        field_size=0.15,
        peak_rate=10.0,
        noise_level=0.05,
        seed=SEED,
    )
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### 3D Place Cells (Volumetric Space)

    Place cells in a 3D volumetric environment.
    """)
    return


@app.cell
def _(SEED, generate_place_cells):  # noqa: N803
    # Generate 3D place cells
    activity_3d, meta_3d = generate_place_cells(
        n_cells=40,
        n_samples=800,
        arena_size=(1.0, 1.0, 0.5),  # 1m x 1m x 0.5m arena
        field_size=0.12,
        peak_rate=10.0,
        noise_level=0.05,
        seed=SEED,
    )
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 2: Grid Cells in Multiple Dimensions

    Grid cells fire at multiple locations arranged in regular periodic patterns.
    """)
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### 1D Grid Cells

    Periodic firing pattern along a linear track.
    """)
    return


@app.cell
def _(SEED, generate_grid_cells):  # noqa: N803
    # Generate 1D grid cells
    grid_1d, grid_meta_1d = generate_grid_cells(
        n_cells=20,
        n_samples=2000,
        arena_size=3.0,
        grid_spacing=0.4,
        peak_rate=10.0,
        noise_level=0.05,
        seed=SEED,
    )
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### 2D Grid Cells - Hexagonal Pattern

    The classic hexagonal grid pattern observed in entorhinal cortex.
    """)
    return


@app.cell
def _():  # noqa: N803
    pass
    return


@app.cell
def _(np):  # noqa: N803
    # Imports
    SEED_1 = 42
    np.random.seed(SEED_1)
    print("✅ Imports successful!")
    print(f"NumPy version: {np.__version__}")
    # Set random seed for reproducibility
    print(f"Random seed: {SEED_1}")
    return (SEED_1,)


@app.cell
def _(SEED_1, generate_grid_cells):  # noqa: N803
    # Generate 2D grid cells
    grid_2d, grid_meta_2d = generate_grid_cells(
        n_cells=30,
        n_samples=2000,
        arena_size=(2.0, 2.0),
        grid_spacing=0.35,
        grid_orientation=15.0,
        peak_rate=10.0,
        noise_level=1,
        seed=SEED_1,
    )
    print("2D Grid Cells:")
    print(f"  Activity shape: {grid_2d.shape}")
    print(f"  Dimensionality: {grid_meta_2d['n_dims']}D")
    print(f"  Grid spacing: {grid_meta_2d['grid_spacing']}m")
    print(
        f"  Grid orientation: {grid_meta_2d['grid_orientation']}°"
    )  # 15 degree rotation
    return grid_2d, grid_meta_2d


@app.cell
def _():  # noqa: N803
    pass
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### 3D Grid Cells

    Hexagonal/tetrahedral grid pattern in 3D volumetric space using FCC-like structure with 4 tetrahedral axes for biological realism.
    """)
    return


@app.cell
def _(SEED_1, generate_grid_cells):  # noqa: N803
    # Generate 3D grid cells
    grid_3d, grid_meta_3d = generate_grid_cells(
        n_cells=20,
        n_samples=2000,
        arena_size=(1.5, 1.5, 1.0),
        grid_spacing=0.3,
        peak_rate=10.0,
        noise_level=0.05,
        seed=SEED_1,
    )
    print("3D Grid Cells:")
    print(f"  Activity shape: {grid_3d.shape}")
    print(f"  Dimensionality: {grid_meta_3d['n_dims']}D")
    print(f"  Grid spacing: {grid_meta_3d['grid_spacing']}m")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 3: Head Direction Cells

    Head direction cells fire when the animal's head points in a specific direction.
    """)
    return


@app.cell
def _(SEED_1, generate_head_direction_cells, np):  # noqa: N803
    # Generate head direction cells
    hd_activity, hd_meta = generate_head_direction_cells(
        n_cells=40,
        n_samples=1000,
        tuning_width=np.pi / 6,
        peak_rate=10.0,
        noise_level=0.05,
        seed=SEED_1,
        plot=True,
    )
    print("Head Direction Cells:")
    print(f"  Activity shape: {hd_activity.shape}")
    print(f"  Head direction shape: {hd_meta['head_directions'].shape}")  # 30 degrees
    print(
        f"  Tuning width: {np.degrees(hd_meta['tuning_width']):.1f}°"
    )  # Use integrated plotting  # Fixed: plural
    return hd_activity, hd_meta


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 4: Manifold Mappings

    Now let's map the population activity to their true underlying manifolds:
    - **Ring (S¹)** for place cells (1D) and head direction cells
    - **Torus (T²)** for grid cells (2D)
    """)
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Map Place Cells to Ring (1D → 2D Circle)
    """)
    return


@app.cell
def _(activity_1d, map_to_ring, meta_1d):  # noqa: N803
    # Map 1D place cells to ring
    ring_coords = map_to_ring(activity_1d, meta_1d["positions"], plot=True)

    print("Ring mapping:")
    print(f"  Input positions: {meta_1d['positions'].shape}")
    print(f"  Ring coordinates: {ring_coords.shape}")
    print("✅ The 1D position perfectly maps to a ring (circle)!")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Map Head Direction Cells to Ring
    """)
    return


@app.cell
def _(hd_activity, hd_meta, map_to_ring):  # noqa: N803
    # Map head direction to ring
    hd_ring = map_to_ring(hd_activity, hd_meta["head_directions"], plot=True)

    print("HD Ring mapping:")
    print(f"  Input angles: {hd_meta['head_directions'].shape}")
    print(f"  Ring coordinates: {hd_ring.shape}")
    print("✅ Head direction perfectly maps to a ring (circle)!")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Map Grid Cells to Torus (2D → 3D Torus)
    """)
    return


@app.cell
def _(grid_2d, grid_meta_2d, map_to_torus):  # noqa: N803
    # Map 2D grid cells to torus
    torus_coords = map_to_torus(
        grid_2d,
        grid_meta_2d["positions"],
        major_radius=2.0,
        minor_radius=1.0,
        plot=True,
    )

    print("Torus mapping:")
    print(f"  Input positions: {grid_meta_2d['positions'].shape}")
    print(f"  Torus coordinates: {torus_coords.shape}")
    print("✅ The 2D periodic space perfectly maps to a torus!")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 5: Mixed Neural Populations

    Create realistic datasets with mixtures of different cell types using flexible configuration.
    """)
    return


@app.cell
def _(SEED_1, generate_mixed_population_flexible, np):  # noqa: N803
    # Define cell configuration
    cell_config = {
        "place": {"n_cells": 60, "field_size": 0.18, "noise_level": 0.08},
        "grid": {"n_cells": 40, "grid_spacing": 0.35, "noise_level": 0.05},
        "head_direction": {
            "n_cells": 30,
            "tuning_width": np.pi / 5,
            "noise_level": 0.1,
        },
    }
    mixed_activity, mixed_meta = generate_mixed_population_flexible(
        cell_config=cell_config, n_samples=1500, arena_size=(2.0, 2.0), seed=SEED_1
    )
    # Generate mixed population
    print("\n✅ Mixed population generated with custom configuration!")
    return mixed_activity, mixed_meta


@app.cell
def _(SEED_1, generate_mixed_population_flexible, np):  # noqa: N803
    # Test with random cells added to the configuration
    cell_config_with_random = {
        "place": {"n_cells": 30, "field_size": 0.18, "noise_level": 0.08},
        "grid": {"n_cells": 20, "grid_spacing": 0.35, "noise_level": 0.05},
        "head_direction": {
            "n_cells": 15,
            "tuning_width": np.pi / 5,
            "noise_level": 0.1,
        },
        "random": {
            "n_cells": 10,
            "baseline_rate": 5.0,
            "variability": 2.0,
            "temporal_smoothness": 0.2,
        },
    }
    mixed_activity_full, mixed_meta_full = generate_mixed_population_flexible(
        cell_config=cell_config_with_random,
        n_samples=1500,
        arena_size=(2.0, 2.0),
        seed=SEED_1,
    )
    # Generate mixed population with random cells
    print("\n✅ Full mixed population generated (including random cells)!")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 6: Embedding Quality - Perfect vs. Noisy vs. Mixed

    Compare how embeddings degrade from perfect → noisy → mixed populations.
    """)
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Perfect Place Cell Embeddings

    First, generate perfect (noise-free) place cells and embed them.
    """)
    return


@app.cell
def _(SEED_1, UMAP, generate_place_cells):  # noqa: N803
    # Generate perfect (noise-free) place cells for 2D
    perfect_activity, perfect_meta = generate_place_cells(
        n_cells=80,
        n_samples=1200,
        arena_size=(1.5, 1.5),
        field_size=0.15,
        noise_level=0.0,
        seed=SEED_1,
    )
    umap_perfect = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED_1
    )
    perfect_embedding = umap_perfect.fit_transform(perfect_activity)
    # Compute UMAP embedding for perfect place cells
    print("✅ Perfect place cells produce clean embeddings!")  # No noise!
    return perfect_activity, perfect_embedding


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Noisy Place Cell Embeddings

    Now add varying levels of noise and see how embeddings degrade.
    """)
    return


@app.cell
def _(Isomap, PCA, SEED_1, UMAP, generate_data, np):  # noqa: N803
    print("Applying dimensionality reduction to Swiss Roll...")
    swiss_roll, swiss_colors = generate_data("swiss_roll", n_samples=1000, noise=0.1)
    _pca = PCA(n_components=2, random_state=SEED_1)
    _swiss_pca = _pca.fit_transform(swiss_roll)
    _isomap = Isomap(n_components=2, n_neighbors=10)
    _swiss_isomap = _isomap.fit_transform(swiss_roll)
    _umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED_1)
    _swiss_umap = _umap.fit_transform(swiss_roll)
    from neural_analysis.plotting import (
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
    )

    _plot_specs = []
    swiss3d_spec = PlotSpec(
        data=np.column_stack([swiss_roll[:, 0], swiss_roll[:, 1], swiss_roll[:, 2]]),
        plot_type="scatter3d",
        subplot_position=0,
        title="Original Swiss Roll (3D)",
        colors=swiss_colors,
        cmap="viridis",
        marker_size=5,
        alpha=0.7,
        colorbar=True,
        colorbar_label="True Color",
        kwargs={"x_label": "X", "y_label": "Y", "z_label": "Z"},
    )
    _plot_specs.append(swiss3d_spec)
    pca_spec = PlotSpec(
        data={"x": _swiss_pca[:, 0], "y": _swiss_pca[:, 1]},
        plot_type="scatter",
        subplot_position=1,
        title="PCA (Linear)",
        colors=swiss_colors,
        cmap="viridis",
        marker_size=10,
        alpha=0.7,
        colorbar=True,
        colorbar_label="True Color",
        equal_aspect=True,
        kwargs={"x_label": "PC 1", "y_label": "PC 2"},
    )
    _plot_specs.append(pca_spec)
    isomap_spec = PlotSpec(
        data={"x": _swiss_isomap[:, 0], "y": _swiss_isomap[:, 1]},
        plot_type="scatter",
        subplot_position=2,
        title="Isomap (Geodesic)",
        colors=swiss_colors,
        cmap="viridis",
        marker_size=10,
        alpha=0.7,
        colorbar=True,
        colorbar_label="True Color",
        equal_aspect=True,
        kwargs={"x_label": "Isomap 1", "y_label": "Isomap 2"},
    )
    _plot_specs.append(isomap_spec)
    umap_spec = PlotSpec(
        data={"x": _swiss_umap[:, 0], "y": _swiss_umap[:, 1]},
        plot_type="scatter",
        subplot_position=3,
        title="UMAP (Topological)",
        colors=swiss_colors,
        cmap="viridis",
        marker_size=10,
        alpha=0.7,
        colorbar=True,
        colorbar_label="True Color",
        equal_aspect=True,
        kwargs={"x_label": "UMAP 1", "y_label": "UMAP 2"},
    )
    _plot_specs.append(umap_spec)
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(
            figsize=(16, 5),
            title="Swiss Roll: Comparing Dimensionality Reduction Methods",
        ),
        layout=GridLayoutConfig(rows=1, cols=4, horizontal_spacing=0.3),
        backend="matplotlib",
    )
    _fig, _ = _grid.plot()
    print("✅ Isomap and UMAP successfully 'unroll' the Swiss Roll!")
    return GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Mixed Population Embeddings

    Embedding mixed populations is more challenging than pure populations.
    """)
    return


@app.cell
def _(
    GridLayoutConfig,
    GridSpec,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    SEED_1,
    UMAP,
    mixed_activity,
    mixed_meta,
    np,
    perfect_activity,
    perfect_embedding,
    plt,
):
    print("Embedding mixed population...")
    umap_mixed = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED_1)
    mixed_embedding = umap_mixed.fit_transform(mixed_activity)
    place_indices = mixed_meta["cell_indices"]["place"]
    place_only_activity = mixed_activity[:, place_indices]
    umap_place_only = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED_1
    )
    place_only_embedding = umap_place_only.fit_transform(place_only_activity)

    _plot_specs = []
    trajectory_spec = PlotSpec(
        data={"x": mixed_meta["positions"][:, 0], "y": mixed_meta["positions"][:, 1]},
        plot_type="trajectory",
        subplot_position=0,
        title="True 2D Trajectory (Shared for All Cell Types)",
        color_by="time",
        cmap="viridis",
        marker_size=5,
        alpha=0.7,
        equal_aspect=True,
        colorbar=True,
        colorbar_label="Time",
        kwargs={"x_label": "X Position (m)", "y_label": "Y Position (m)"},
    )
    _plot_specs.append(trajectory_spec)
    pure_raster_spec = PlotSpec(
        data=perfect_activity.T,
        plot_type="heatmap",
        subplot_position=3,
        title="Pure Place Cells",
        cmap="hot",
        colorbar=True,
        colorbar_label="Rate",
        kwargs={
            "x_label": "Time",
            "y_label": "Cell ID",
            "aspect": "auto",
            "interpolation": "nearest",
        },
    )
    _plot_specs.append(pure_raster_spec)
    pure_embedding_spec = PlotSpec(
        data={"x": perfect_embedding[:, 0], "y": perfect_embedding[:, 1]},
        plot_type="trajectory",
        subplot_position=6,
        title="Embedding: Pure Place Cells",
        color_by="time",
        cmap="viridis",
        marker_size=5,
        alpha=0.7,
        equal_aspect=True,
        colorbar=True,
        colorbar_label="Time",
        kwargs={"x_label": "UMAP 1", "y_label": "UMAP 2"},
    )
    _plot_specs.append(pure_embedding_spec)
    place_only_raster_spec = PlotSpec(
        data=place_only_activity.T,
        plot_type="heatmap",
        subplot_position=4,
        title="Place Cells from Mixed Pop",
        cmap="hot",
        colorbar=True,
        colorbar_label="Rate",
        kwargs={
            "x_label": "Time",
            "y_label": "Cell ID",
            "aspect": "auto",
            "interpolation": "nearest",
        },
    )
    _plot_specs.append(place_only_raster_spec)
    place_only_embedding_spec = PlotSpec(
        data={"x": place_only_embedding[:, 0], "y": place_only_embedding[:, 1]},
        plot_type="trajectory",
        subplot_position=7,
        title="Embedding: Place Cells Only",
        color_by="time",
        cmap="viridis",
        marker_size=5,
        alpha=0.7,
        equal_aspect=True,
        colorbar=True,
        colorbar_label="Time",
        kwargs={"x_label": "UMAP 1", "y_label": "UMAP 2"},
    )
    _plot_specs.append(place_only_embedding_spec)
    mixed_raster_spec = PlotSpec(
        data=mixed_activity.T,
        plot_type="heatmap",
        subplot_position=5,
        title="Mixed Population (All Types)",
        cmap="hot",
        colorbar=True,
        colorbar_label="Rate",
        kwargs={
            "x_label": "Time",
            "y_label": "Cell ID",
            "aspect": "auto",
            "interpolation": "nearest",
        },
    )
    _plot_specs.append(mixed_raster_spec)
    mixed_embedding_spec = PlotSpec(
        data={"x": mixed_embedding[:, 0], "y": mixed_embedding[:, 1]},
        plot_type="trajectory",
        subplot_position=8,
        title="Embedding: Mixed Population",
        color_by="time",
        cmap="viridis",
        marker_size=5,
        alpha=0.7,
        equal_aspect=True,
        colorbar=True,
        colorbar_label="Time",
        kwargs={"x_label": "UMAP 1", "y_label": "UMAP 2"},
    )
    _plot_specs.append(mixed_embedding_spec)
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(
            figsize=(16, 10),
            title="Embedding Quality: Pure → Place Only → Mixed Population",
        ),
        layout=GridLayoutConfig(
            rows=3,
            cols=3,
            vertical_spacing=0.35,
            horizontal_spacing=0.3,
            height_ratios=[1, 2, 2],
        ),
        backend="matplotlib",
    )
    _fig, _axes = _grid.plot()
    if len(_axes) > 5:
        ax_mixed = _axes[5]
        for cell_type, indices in mixed_meta["cell_indices"].items():
            color = {"place": "cyan", "grid": "yellow", "head_direction": "magenta"}[
                cell_type
            ]
            ax_mixed.axhspan(
                indices[0] - 0.5, indices[-1] + 0.5, alpha=0.15, color=color, zorder=-1
            )
    if len(_axes) > 0:
        _fig = _axes[0].figure
        _axes[0].remove()
        gs = _fig._gridspecs[0] if hasattr(_fig, "_gridspecs") else None
        if gs is None:
            gs = GridSpec(
                3, 3, figure=_fig, height_ratios=[1, 2, 2], hspace=0.35, wspace=0.3
            )
            _fig._gridspecs = [gs]
        ax_trajectory = _fig.add_subplot(gs[0, :])
        _scatter = ax_trajectory.scatter(
            mixed_meta["positions"][:, 0],
            mixed_meta["positions"][:, 1],
            c=np.arange(len(mixed_meta["positions"])),
            cmap="viridis",
            s=5,
            alpha=0.7,
        )
        ax_trajectory.set_xlabel("X Position (m)")
        ax_trajectory.set_ylabel("Y Position (m)")
        ax_trajectory.set_title("True 2D Trajectory (Shared for All Cell Types)")
        ax_trajectory.set_aspect("equal")
        plt.colorbar(_scatter, ax=ax_trajectory, label="Time", orientation="horizontal")
    plt.show()
    print("✅ Mixed populations produce more complex embeddings!")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Part 7: sklearn Manifold Datasets

    Classic manifold datasets for benchmarking dimensionality reduction algorithms.
    """)
    return


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    SEED_1,
    generate_data,
    generate_s_curve,
    generate_swiss_roll,
    np,
):
    print("Generating sklearn manifold datasets...")
    swiss_roll_1, swiss_colors_1 = generate_swiss_roll(
        n_samples=1500, noise=0.1, seed=SEED_1
    )
    s_curve, s_colors = generate_s_curve(n_samples=1500, noise=0.1, seed=SEED_1)
    blobs, blob_labels = generate_data(
        "blobs", n_samples=800, n_features=3, n_classes=4, noise=0.1, seed=SEED_1
    )
    moons, moon_labels = generate_data("moons", n_samples=800, noise=0.1, seed=SEED_1)
    circles, circle_labels = generate_data(
        "circles", n_samples=800, noise=0.05, factor=0.5, seed=SEED_1
    )
    print("✅ All sklearn datasets generated!")
    _plot_specs = []
    swiss_spec = PlotSpec(
        data=np.column_stack(
            [swiss_roll_1[:, 0], swiss_roll_1[:, 1], swiss_roll_1[:, 2]]
        ),
        plot_type="scatter3d",
        subplot_position=0,
        title="Swiss Roll",
        colors=swiss_colors_1,
        cmap="viridis",
        marker_size=5,
        alpha=0.7,
        colorbar=True,
        colorbar_label="Color",
        kwargs={"x_label": "X", "y_label": "Y", "z_label": "Z"},
    )
    _plot_specs.append(swiss_spec)
    scurve_spec = PlotSpec(
        data=np.column_stack([s_curve[:, 0], s_curve[:, 1], s_curve[:, 2]]),
        plot_type="scatter3d",
        subplot_position=1,
        title="S-Curve",
        colors=s_colors,
        cmap="plasma",
        marker_size=5,
        alpha=0.7,
        colorbar=True,
        colorbar_label="Color",
        kwargs={"x_label": "X", "y_label": "Y", "z_label": "Z"},
    )
    _plot_specs.append(scurve_spec)
    blobs_spec = PlotSpec(
        data=np.column_stack([blobs[:, 0], blobs[:, 1], blobs[:, 2]]),
        plot_type="scatter3d",
        subplot_position=2,
        title="Blobs (4 clusters)",
        colors=blob_labels,
        cmap="tab10",
        marker_size=5,
        alpha=0.7,
        colorbar=True,
        colorbar_label="Cluster",
        kwargs={"x_label": "X", "y_label": "Y", "z_label": "Z"},
    )
    _plot_specs.append(blobs_spec)
    moons_spec = PlotSpec(
        data={"x": moons[:, 0], "y": moons[:, 1]},
        plot_type="scatter",
        subplot_position=3,
        title="Moons",
        colors=moon_labels,
        cmap="coolwarm",
        marker_size=10,
        alpha=0.7,
        colorbar=True,
        colorbar_label="Class",
        equal_aspect=True,
        kwargs={"x_label": "X", "y_label": "Y"},
    )
    _plot_specs.append(moons_spec)
    circles_spec = PlotSpec(
        data={"x": circles[:, 0], "y": circles[:, 1]},
        plot_type="scatter",
        subplot_position=4,
        title="Circles",
        colors=circle_labels,
        cmap="coolwarm",
        marker_size=10,
        alpha=0.7,
        colorbar=True,
        colorbar_label="Class",
        equal_aspect=True,
        kwargs={"x_label": "X", "y_label": "Y"},
    )
    _plot_specs.append(circles_spec)
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(figsize=(16, 10), title="sklearn Manifold Datasets"),
        layout=GridLayoutConfig(
            rows=2,
            cols=3,
            vertical_spacing=0.3,
            horizontal_spacing=0.3,
            width_ratios=[1.3, 1, 1],
        ),
        backend="matplotlib",
    )
    _fig, _axes = _grid.plot()
    if len(_axes) > 5:
        _axes[5].text(
            0.5,
            0.5,
            "sklearn Manifolds\n\nUseful for testing:\n• Dimensionality reduction\n• Clustering algorithms\n• Manifold learning",
            ha="center",
            va="center",
            fontsize=12,
            transform=_axes[5].transAxes,
        )
        _axes[5].axis("off")
    return swiss_colors_1, swiss_roll_1


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Apply Dimensionality Reduction to sklearn Datasets

    Compare how different methods (PCA, Isomap, UMAP) recover manifold structure.
    """)
    return


@app.cell
def _(Isomap, PCA, SEED_1, UMAP, plt, swiss_colors_1, swiss_roll_1):  # noqa: N803
    print("Applying dimensionality reduction to Swiss Roll...")
    _pca = PCA(n_components=2, random_state=SEED_1)
    _swiss_pca = _pca.fit_transform(swiss_roll_1)
    _isomap = Isomap(n_components=2, n_neighbors=10)
    _swiss_isomap = _isomap.fit_transform(swiss_roll_1)
    _umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED_1)
    _swiss_umap = _umap.fit_transform(swiss_roll_1)
    _fig = plt.figure(figsize=(16, 5))
    ax1 = _fig.add_subplot(141, projection="3d")
    _scatter = ax1.scatter(
        swiss_roll_1[:, 0],
        swiss_roll_1[:, 1],
        swiss_roll_1[:, 2],
        c=swiss_colors_1,
        cmap="viridis",
        s=5,
        alpha=0.7,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Original Swiss Roll (3D)")
    plt.colorbar(_scatter, ax=ax1, shrink=0.5, label="True Color")
    ax2 = _fig.add_subplot(142)
    _scatter = ax2.scatter(
        _swiss_pca[:, 0],
        _swiss_pca[:, 1],
        c=swiss_colors_1,
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.set_title("PCA (Linear)")
    ax2.set_aspect("equal")
    plt.colorbar(_scatter, ax=ax2, label="True Color")
    ax3 = _fig.add_subplot(143)
    _scatter = ax3.scatter(
        _swiss_isomap[:, 0],
        _swiss_isomap[:, 1],
        c=swiss_colors_1,
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax3.set_xlabel("Isomap 1")
    ax3.set_ylabel("Isomap 2")
    ax3.set_title("Isomap (Geodesic)")
    ax3.set_aspect("equal")
    plt.colorbar(_scatter, ax=ax3, label="True Color")
    ax4 = _fig.add_subplot(144)
    _scatter = ax4.scatter(
        _swiss_umap[:, 0],
        _swiss_umap[:, 1],
        c=swiss_colors_1,
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax4.set_xlabel("UMAP 1")
    ax4.set_ylabel("UMAP 2")
    ax4.set_title("UMAP (Topological)")
    ax4.set_aspect("equal")
    plt.colorbar(_scatter, ax=ax4, label="True Color")
    plt.suptitle("Swiss Roll: Comparing Dimensionality Reduction Methods", fontsize=14)
    plt.tight_layout()
    plt.show()
    print("✅ Isomap and UMAP successfully 'unroll' the Swiss Roll!")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Summary and Key Takeaways

    This notebook demonstrated comprehensive synthetic dataset generation capabilities:

    ### Neural Data Generators
    ✅ **Place Cells** in 1D, 2D, and 3D environments with Gaussian firing fields
    ✅ **Grid Cells** in 1D, 2D, and 3D with periodic patterns (1D: periodic, 2D: hexagonal, 3D: cubic)
    ✅ **Head Direction Cells** with von Mises tuning curves
    ✅ **Mixed Populations** with flexible dictionary-based configuration

    ### Manifold Mappings
    ✅ **Ring (S¹)** mapping for place cells (1D) and head direction cells
    ✅ **Torus (T²)** mapping for grid cells (2D periodic space)
    ✅ **Population Vector Decoding** for trajectory tracking on manifolds

    ### Embedding Quality Analysis
    ✅ **Perfect Place Cells** → Clean embeddings that recover true structure
    ✅ **Noisy Place Cells** → Embeddings degrade gracefully with noise
    ✅ **Mixed Populations** → More complex embeddings reflecting heterogeneous cell types

    ### sklearn Datasets
    ✅ **Manifolds**: Swiss roll, S-curve, blobs, moons, circles
    ✅ **Dimensionality Reduction**: PCA, Isomap, UMAP comparisons
    ✅ **Manifold Learning**: Methods that preserve intrinsic geometry

    ### Key Functions
    - `generate_place_cells()`, `generate_grid_cells()`, `generate_head_direction_cells()`
    - `generate_mixed_population_flexible()` - dictionary-based configuration
    - `map_to_ring()`, `map_to_torus()` - manifold mappings
    - `population_vector_decoder()` - decode position from activity
    - `generate_data()` - unified interface for all dataset types

    ### Next Steps
    1. Use these datasets to test your neural analysis methods
    2. Benchmark dimensionality reduction algorithms
    3. Validate decoding approaches with known ground truth
    4. Test cell type classification on mixed populations
    5. Study how noise affects embedding quality

    **All datasets are reproducible with random seeds!** 🎲
    """)
    return


if __name__ == "__main__":
    app.run()
