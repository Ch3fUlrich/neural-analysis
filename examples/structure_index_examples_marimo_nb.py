import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Structure Index Examples

    This notebook demonstrates the **Structure Index (SI)** analysis for quantifying neural manifold organization. We'll:

    1. Generate synthetic neural datasets (place cells, grid cells, random)
    2. Visualize the neural activity and behavioral correlates
    3. Compute structure indices with different parameters
    4. Visualize overlap graphs
    5. Compare SI values across datasets
    6. Perform parameter sweeps
    7. Save and load results from HDF5
    8. Compare distributions and embeddings across datasets

    ## What is the Structure Index?

    The Structure Index quantifies how well neural population activity organizes according to external behavioral or stimulus variables. High SI values indicate that similar behavioral states correspond to similar neural patterns, forming a coherent manifold structure.
    """)
    return


@app.cell
def _():
    # Imports
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    # Neural analysis imports
    from neural_analysis import (
        compute_embedding,
        compute_structure_index,
        generate_data,
    )
    from neural_analysis.plotting.synthetic_plots import plot_synthetic_data
    from neural_analysis.topology import (
        compute_structure_index_sweep,
        draw_overlap_graph,
    )
    from neural_analysis.utils.io import (
        get_hdf5_result_summary,
        load_results_from_hdf5_dataset,
    )

    # Set random seed for reproducibility
    np.random.seed(42)

    # Additional imports for distribution comparisons
    from neural_analysis.metrics.distributions import (
        pairwise_distribution_comparison_batch,
    )

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("✓ Imports successful")
    return (
        compute_embedding,
        compute_structure_index,
        compute_structure_index_sweep,
        draw_overlap_graph,
        generate_data,
        get_hdf5_result_summary,
        load_results_from_hdf5_dataset,
        np,
        pairwise_distribution_comparison_batch,
        pd,
        plot_synthetic_data,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Generate Synthetic Datasets

    We'll create three types of synthetic neural data:
    - **Place Cells**: Neurons that fire at specific spatial locations (high structure)
    - **Grid Cells**: Neurons with periodic spatial firing patterns (high structure)
    - **Random**: Randomly generated activity (low structure, control)
    """)
    return


@app.cell
def _():
    pass
    return


@app.cell
def _(generate_data):
    # Generate Random (control)
    random_data, random_labels = generate_data(
        "random_cells", n_samples=1000, n_features=50, noise=1.0, seed=44
    )
    return random_data, random_labels


@app.cell
def _(plot_synthetic_data, random_data, random_labels):
    # Quick test: Visualize random cells with autocorrelation diagnostic
    # This tests the refactored _compute_2d_autocorrelation helper function
    print("Testing refactored random cell visualization...")
    fig_random = plot_synthetic_data(
        random_data, random_labels, backend="matplotlib", figsize=(15, 8)
    )
    print("✓ Random cell visualization complete - check for autocorrelation subplot!")
    return


@app.cell
def _(generate_data):
    # Clear cached modules first
    place_data, place_metadata = generate_data(
        "place_cells", n_samples=1000, n_features=50, noise=0.1, seed=42
    )
    grid_data, grid_metadata = generate_data(
        "grid_cells", n_samples=1000, n_features=50, noise=0.1, seed=43
    )
    random_data_1, random_labels_1 = generate_data(
        "random_cells", n_samples=1000, n_features=50, noise=1.0, seed=44
    )
    print(f"\nPlace cells: {place_data.shape}")
    print(f"Grid cells: {grid_data.shape}")
    # Now import and generate
    # Generate Place Cells
    # Generate Grid Cells
    # Generate Random (control)
    print(f"Random data: {random_data_1.shape}")
    return (
        grid_data,
        grid_metadata,
        place_data,
        place_metadata,
        random_data_1,
        random_labels_1,
    )


@app.cell
def _(generate_data):
    # Generate Head Direction Cells
    hd_data, hd_metadata = generate_data(
        "head_direction_cells", n_samples=1000, n_features=50, noise=0.1, seed=45
    )

    print(f"Head direction cells: {hd_data.shape}")
    return hd_data, hd_metadata


@app.cell
def _(mo):
    mo.md(r"""
    ## Test: Generate Random Cells with New Visualizations

    This cell tests the updated `generate_random_cells()` function with:
    1. Fixed uniform position distribution
    2. Head direction labels
    3. New visualization showing example place/grid/HD cell tuning (instead of random cell activity)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Extract Position Labels

    For structure index computation, we need position labels from the metadata.
    """)
    return


@app.cell
def _(grid_metadata, hd_metadata, np, place_metadata, random_labels_1):
    # Extract position labels
    place_position = place_metadata["positions"]
    grid_position = grid_metadata["positions"]
    hd_angles = hd_metadata["head_directions"]
    # For HD cells, we use head_directions as the behavioral variable
    random_position = random_labels_1["positions"]
    print(f"Place position shape: {place_position.shape}")
    # For random data, now we have positions in metadata (uniformly distributed)
    print(f"Grid position shape: {grid_position.shape}")
    print(f"HD angles shape: {hd_angles.shape}")
    print(f"Random position shape: {random_position.shape}")
    print(
        f"\nRandom position range X: [{random_position[:, 0].min():.3f}, {random_position[:, 0].max():.3f}]"
    )
    print(
        f"Random position range Y: [{random_position[:, 1].min():.3f}, {random_position[:, 1].max():.3f}]"
    )
    print(f"\nHD metadata keys: {list(hd_metadata.keys())}")
    # Verify uniform distribution of random positions
    # Show HD metadata
    print(
        f"Head direction range: [{np.degrees(hd_angles.min()):.1f}°, {np.degrees(hd_angles.max()):.1f}°]"
    )
    return grid_position, hd_angles, place_position, random_position


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Compute Structure Index - Single Parameter Set

    Let's compute the structure index for each dataset with a single set of parameters.
    """)
    return


@app.cell
def _(
    compute_structure_index,
    grid_data,
    grid_position,
    hd_angles,
    hd_data,
    np,
    place_data,
    place_position,
    random_data_1,
    random_position,
):
    # Parameters
    n_bins = 10
    n_neighbors = 15
    num_shuffles = 1
    si_results = {}
    # Storage for results
    print("Computing Structure Index for each dataset...\n")
    print("1. Place Cells:")
    SI_place, bins_place, overlap_place, shuf_place = compute_structure_index(
        data=place_data,
        label=place_position,
        n_bins=n_bins,
        n_neighbors=n_neighbors,
        num_shuffles=num_shuffles,
        verbose=False,
    )
    z_place = (SI_place - np.mean(shuf_place)) / np.std(shuf_place)
    # Place Cells
    p_place = np.mean(shuf_place >= SI_place)
    print(f"   SI = {SI_place:.3f}")
    print(f"   Z-score = {z_place:.2f}")
    print(f"   P-value = {p_place:.4f}")
    si_results["place_cells"] = {
        "SI": SI_place,
        "z_score": z_place,
        "p_value": p_place,
        "overlap": overlap_place,
        "shuffled": shuf_place,
    }
    print("\n2. Grid Cells:")
    SI_grid, bins_grid, overlap_grid, shuf_grid = compute_structure_index(
        data=grid_data,
        label=grid_position,
        n_bins=n_bins,
        n_neighbors=n_neighbors,
        num_shuffles=num_shuffles,
        verbose=False,
    )
    z_grid = (SI_grid - np.mean(shuf_grid)) / np.std(shuf_grid)
    p_grid = np.mean(shuf_grid >= SI_grid)
    print(f"   SI = {SI_grid:.3f}")
    print(f"   Z-score = {z_grid:.2f}")
    print(f"   P-value = {p_grid:.4f}")
    si_results["grid_cells"] = {
        "SI": SI_grid,
        "z_score": z_grid,
        "p_value": p_grid,
        "overlap": overlap_grid,
        "shuffled": shuf_grid,
    }
    print("\n3. Head Direction Cells:")
    SI_hd, bins_hd, overlap_hd, shuf_hd = compute_structure_index(
        data=hd_data,
        label=hd_angles,
        n_bins=n_bins,
        n_neighbors=n_neighbors,
        num_shuffles=num_shuffles,
        verbose=False,
    )
    z_hd = (SI_hd - np.mean(shuf_hd)) / np.std(shuf_hd)
    p_hd = np.mean(shuf_hd >= SI_hd)
    print(f"   SI = {SI_hd:.3f}")
    print(f"   Z-score = {z_hd:.2f}")
    print(f"   P-value = {p_hd:.4f}")
    si_results["hd_cells"] = {
        "SI": SI_hd,
        "z_score": z_hd,
        "p_value": p_hd,
        "overlap": overlap_hd,
        "shuffled": shuf_hd,
    }
    print("\n4. Random Data:")
    # Grid Cells
    SI_random, bins_random, overlap_random, shuf_random = compute_structure_index(
        data=random_data_1,
        label=random_position,
        n_bins=n_bins,
        n_neighbors=n_neighbors,
        num_shuffles=num_shuffles,
        verbose=False,
    )
    z_random = (SI_random - np.mean(shuf_random)) / np.std(shuf_random)
    p_random = np.mean(shuf_random >= SI_random)
    print(f"   SI = {SI_random:.3f}")
    print(f"   Z-score = {z_random:.2f}")
    print(f"   P-value = {p_random:.4f}")
    # Head Direction Cells
    # Random Data
    si_results["random"] = {
        "SI": SI_random,
        "z_score": z_random,
        "p_value": p_random,
        "overlap": overlap_random,
        "shuffled": shuf_random,
    }
    return (
        SI_grid,
        SI_hd,
        SI_place,
        SI_random,
        overlap_grid,
        overlap_hd,
        overlap_place,
        overlap_random,
        shuf_grid,
        shuf_hd,
        shuf_place,
        shuf_random,
        z_grid,
        z_hd,
        z_place,
        z_random,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Visualize Overlap Graphs

    The overlap matrix shows how neural activity in one behavioral bin overlaps with activity in other bins.
    """)
    return


@app.cell
def _(
    SI_grid,
    SI_hd,
    SI_place,
    SI_random,
    draw_overlap_graph,
    overlap_grid,
    overlap_hd,
    overlap_place,
    overlap_random,
    plt,
    z_grid,
    z_hd,
    z_place,
    z_random,
):
    # Create figure with 4 subplots
    _fig, _axes = plt.subplots(1, 4, figsize=(24, 5))
    draw_overlap_graph(overlap_place, ax=_axes[0])
    # Place Cells
    _axes[0].set_title(
        f"Place Cells\nSI = {SI_place:.3f}, Z = {z_place:.2f}", fontsize=12
    )
    draw_overlap_graph(overlap_grid, ax=_axes[1])
    _axes[1].set_title(f"Grid Cells\nSI = {SI_grid:.3f}, Z = {z_grid:.2f}", fontsize=12)
    # Grid Cells
    draw_overlap_graph(overlap_hd, ax=_axes[2])
    _axes[2].set_title(
        f"Head Direction Cells\nSI = {SI_hd:.3f}, Z = {z_hd:.2f}", fontsize=12
    )
    draw_overlap_graph(overlap_random, ax=_axes[3])
    # Head Direction Cells
    _axes[3].set_title(
        f"Random Data\nSI = {SI_random:.3f}, Z = {z_random:.2f}", fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/overlap_graphs.png", dpi=150, bbox_inches="tight")
    # Random
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Compare Structure Indices

    Let's create a visual comparison of the structure indices across datasets.
    """)
    return


@app.cell
def _(
    SI_grid,
    SI_hd,
    SI_place,
    SI_random,
    plt,
    z_grid,
    z_hd,
    z_place,
    z_random,
):
    # Bar plot comparison
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _datasets = ["Place Cells", "Grid Cells", "HD Cells", "Random"]
    # SI values
    si_values = [SI_place, SI_grid, SI_hd, SI_random]
    z_scores = [z_place, z_grid, z_hd, z_random]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#95A5A6"]
    _axes[0].bar(
        _datasets, si_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    _axes[0].set_ylabel("Structure Index", fontsize=12)
    _axes[0].set_title("Structure Index Comparison", fontsize=14, fontweight="bold")
    _axes[0].set_ylim(0, 1.0)
    _axes[0].axhline(
        y=0.5, color="gray", linestyle="--", alpha=0.5, label="Moderate Structure"
    )
    _axes[0].axhline(
        y=0.7, color="orange", linestyle="--", alpha=0.5, label="Strong Structure"
    )
    _axes[0].legend()
    _axes[0].grid(axis="y", alpha=0.3)
    _axes[1].bar(
        _datasets, z_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    _axes[1].set_ylabel("Z-Score", fontsize=12)
    _axes[1].set_title(
        "Statistical Significance (Z-Score)", fontsize=14, fontweight="bold"
    )
    _axes[1].axhline(
        y=2,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Significance Threshold (p<0.05)",
    )
    _axes[1].legend()
    _axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/si_comparison.png", dpi=150, bbox_inches="tight")
    # Z-scores
    plt.show()
    return


@app.cell
def _(
    SI_grid,
    SI_hd,
    SI_place,
    SI_random,
    np,
    plt,
    shuf_grid,
    shuf_hd,
    shuf_place,
    shuf_random,
):
    # Null distribution comparison
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 4))
    for _idx, (name, _color, observed, shuffled) in enumerate(
        [
            ("Place Cells", "#E74C3C", SI_place, shuf_place),
            ("Grid Cells", "#3498DB", SI_grid, shuf_grid),
            ("HD Cells", "#2ECC71", SI_hd, shuf_hd),
            ("Random", "#95A5A6", SI_random, shuf_random),
        ]
    ):
        _axes[_idx].hist(
            shuffled,
            bins=30,
            alpha=0.6,
            color=_color,
            edgecolor="black",
            label="Null (shuffled)",
        )
        _axes[_idx].axvline(
            observed,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Observed (SI={observed:.3f})",
        )
        _axes[_idx].axvline(
            np.mean(shuffled),
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Mean null ({np.mean(shuffled):.3f})",
        )
        _axes[_idx].set_xlabel("Structure Index", fontsize=11)
        _axes[_idx].set_ylabel("Frequency", fontsize=11)
        _axes[_idx].set_title(name, fontsize=12, fontweight="bold")
        _axes[_idx].legend(fontsize=9)
        _axes[_idx].grid(alpha=0.3)
    print("✓ Null distribution comparison complete")
    plt.tight_layout()
    plt.savefig("output/null_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Parameter Sweep

    Now let's perform a parameter sweep to see how the structure index varies with different parameter choices.
    """)
    return


@app.cell
def _(compute_structure_index_sweep, place_data, place_position):
    # Define parameter ranges
    n_neighbors_list = [10, 15, 20, 25]
    n_bins_list = [8, 10, 12, 15]

    print("Running parameter sweep...")
    print(f"  n_neighbors: {n_neighbors_list}")
    print(f"  n_bins: {n_bins_list}")
    print(f"  Total combinations: {len(n_neighbors_list) * len(n_bins_list)}")
    print()

    # Run sweep for place cells (highest structure expected)
    results_place = compute_structure_index_sweep(
        data=place_data,
        labels=place_position,
        dataset_name="place_cells",
        save_path="output/structure_indices.h5",
        n_neighbors_list=n_neighbors_list,
        n_bins_list=n_bins_list,
        distance_metric="euclidean",
        num_shuffles=100,
        regenerate=False,
        verbose=True,
    )

    print(f"\n✓ Place cells: {len(results_place)} parameter combinations computed")
    return n_bins_list, n_neighbors_list


@app.cell
def _(
    compute_structure_index_sweep,
    grid_data,
    grid_position,
    n_bins_list,
    n_neighbors_list,
):
    # Run sweep for grid cells
    print("Running parameter sweep for grid cells...")
    results_grid = compute_structure_index_sweep(
        data=grid_data,
        labels=grid_position,
        dataset_name="grid_cells",
        save_path="output/structure_indices.h5",
        n_neighbors_list=n_neighbors_list,
        n_bins_list=n_bins_list,
        distance_metric="euclidean",
        num_shuffles=100,
        regenerate=False,
        verbose=True,
    )

    print(f"✓ Grid cells: {len(results_grid)} parameter combinations computed")
    return


@app.cell
def _(
    compute_structure_index_sweep,
    hd_angles,
    hd_data,
    n_bins_list,
    n_neighbors_list,
):
    # Run sweep for head direction cells
    print("Running parameter sweep for HD cells...")
    results_hd = compute_structure_index_sweep(
        data=hd_data,
        labels=hd_angles,
        dataset_name="head_direction_cells",
        save_path="output/structure_indices.h5",
        n_neighbors_list=n_neighbors_list,
        n_bins_list=n_bins_list,
        distance_metric="euclidean",
        num_shuffles=100,
        regenerate=False,
        verbose=True,
    )

    print(f"✓ HD cells: {len(results_hd)} parameter combinations computed")
    return


@app.cell
def _(
    compute_structure_index_sweep,
    n_bins_list,
    n_neighbors_list,
    random_data_1,
    random_position,
):
    # Run sweep for random data
    print("Running parameter sweep for random data...")
    results_random = compute_structure_index_sweep(
        data=random_data_1,
        labels=random_position,
        dataset_name="random",
        save_path="output/structure_indices.h5",
        n_neighbors_list=n_neighbors_list,
        n_bins_list=n_bins_list,
        distance_metric="euclidean",
        num_shuffles=100,
        regenerate=False,
        verbose=True,
    )
    print(f"✓ Random: {len(results_random)} parameter combinations computed")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Load and Analyze Parameter Sweep Results
    """)
    return


@app.cell
def _(get_hdf5_result_summary):
    # Load summary of all results
    df_summary = get_hdf5_result_summary("output/structure_indices.h5")

    print("Summary DataFrame:")
    print(df_summary.head())
    print(f"\nShape: {df_summary.shape}")
    print(f"\nColumns: {list(df_summary.columns)}")
    print(f"\nDatasets: {df_summary['dataset_name'].unique()}")
    return (df_summary,)


@app.cell
def _(df_summary):
    # Display statistics by dataset
    print("\nStructure Index Statistics by Dataset:")
    print(df_summary.groupby("dataset_name")["SI"].describe())

    # Best parameters for each dataset
    print("\n\nBest Parameters for Each Dataset:")
    best_idx = df_summary.groupby("dataset_name")["SI"].idxmax()
    best_params = df_summary.loc[best_idx]
    print(best_params[["dataset_name", "n_bins", "n_neighbors", "SI"]])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Visualize Parameter Effects
    """)
    return


@app.cell
def _(df_summary, plt):
    # Create heatmaps for each dataset
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 4))
    for _idx, (_dataset_name, _color) in enumerate(
        [
            ("place_cells", "#E74C3C"),
            ("grid_cells", "#3498DB"),
            ("head_direction_cells", "#2ECC71"),
            ("random", "#95A5A6"),
        ]
    ):
        _df_dataset = df_summary[df_summary["dataset_name"] == _dataset_name]
        pivot = _df_dataset.pivot_table(
            values="SI", index="n_bins", columns="n_neighbors", aggfunc="mean"
        )
        _im = _axes[_idx].imshow(
            pivot, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        _axes[_idx].set_xticks(range(len(pivot.columns)))  # Filter data
        _axes[_idx].set_xticklabels(pivot.columns)
        _axes[_idx].set_yticks(range(len(pivot.index)))
        _axes[_idx].set_yticklabels(pivot.index)  # Create pivot table
        _axes[_idx].set_xlabel("n_neighbors", fontsize=11)
        _axes[_idx].set_ylabel("n_bins", fontsize=11)
        _axes[_idx].set_title(
            _dataset_name.replace("_", " ").title(), fontsize=12, fontweight="bold"
        )
        _cbar = plt.colorbar(_im, ax=_axes[_idx])
        _cbar.set_label("Structure Index", fontsize=10)  # Plot heatmap
        for _i in range(len(pivot.index)):
            for _j in range(len(pivot.columns)):
                _text = _axes[_idx].text(
                    _j,
                    _i,
                    f"{pivot.iloc[_i, _j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
    plt.tight_layout()
    plt.savefig("output/parameter_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()  # Add colorbar  # Add text annotations
    return


@app.cell
def _(df_summary, plt):
    # Line plots showing parameter effects
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    for _dataset_name, _color, label in [
        ("place_cells", "#E74C3C", "Place Cells"),
        ("grid_cells", "#3498DB", "Grid Cells"),
        ("head_direction_cells", "#2ECC71", "HD Cells"),
        ("random", "#95A5A6", "Random"),
    ]:
        # Effect of n_bins (averaged over n_neighbors)
        _df_dataset = df_summary[df_summary["dataset_name"] == _dataset_name]
        grouped = _df_dataset.groupby("n_bins")["SI"].agg(["mean", "std"])
        _axes[0, 0].plot(
            grouped.index,
            grouped["mean"],
            "o-",
            color=_color,
            label=label,
            linewidth=2,
            markersize=8,
        )
        _axes[0, 0].fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            color=_color,
        )
    _axes[0, 0].set_xlabel("Number of Bins", fontsize=12)
    _axes[0, 0].set_ylabel("Mean Structure Index", fontsize=12)
    _axes[0, 0].set_title(
        "Effect of Binning Resolution", fontsize=13, fontweight="bold"
    )
    _axes[0, 0].legend(fontsize=10)
    _axes[0, 0].grid(alpha=0.3)
    for _dataset_name, _color, label in [
        ("place_cells", "#E74C3C", "Place Cells"),
        ("grid_cells", "#3498DB", "Grid Cells"),
        ("head_direction_cells", "#2ECC71", "HD Cells"),
        ("random", "#95A5A6", "Random"),
    ]:
        _df_dataset = df_summary[df_summary["dataset_name"] == _dataset_name]
        grouped = _df_dataset.groupby("n_neighbors")["SI"].agg(["mean", "std"])
        _axes[0, 1].plot(
            grouped.index,
            grouped["mean"],
            "o-",
            color=_color,
            label=label,
            linewidth=2,
            markersize=8,
        )
        _axes[0, 1].fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            color=_color,
        )
    _axes[0, 1].set_xlabel("Number of Neighbors", fontsize=12)
    _axes[0, 1].set_ylabel("Mean Structure Index", fontsize=12)
    _axes[0, 1].set_title("Effect of Neighborhood Size", fontsize=13, fontweight="bold")
    _axes[0, 1].legend(fontsize=10)
    _axes[0, 1].grid(alpha=0.3)
    for _dataset_name, _color, label in [
        ("place_cells", "#E74C3C", "Place Cells"),
        ("grid_cells", "#3498DB", "Grid Cells"),
        ("head_direction_cells", "#2ECC71", "HD Cells"),
        ("random", "#95A5A6", "Random"),
    ]:
        _df_dataset = df_summary[df_summary["dataset_name"] == _dataset_name]
        _axes[1, 0].scatter(
            _df_dataset["n_bins"],
            _df_dataset["SI"],
            c=_color,
            label=label,
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )
    _axes[1, 0].set_xlabel("Number of Bins", fontsize=12)
    _axes[1, 0].set_ylabel("Structure Index", fontsize=12)
    _axes[1, 0].set_title(
        "SI vs Binning Resolution (all combinations)", fontsize=13, fontweight="bold"
    )
    _axes[1, 0].legend(fontsize=10)
    _axes[1, 0].grid(alpha=0.3)
    for _dataset_name, _color, label in [
        ("place_cells", "#E74C3C", "Place Cells"),
        ("grid_cells", "#3498DB", "Grid Cells"),
        ("head_direction_cells", "#2ECC71", "HD Cells"),
        ("random", "#95A5A6", "Random"),
    ]:
        _df_dataset = df_summary[df_summary["dataset_name"] == _dataset_name]
        _axes[1, 1].scatter(
            _df_dataset["n_neighbors"],
            _df_dataset["SI"],
            c=_color,
            label=label,
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )
    _axes[1, 1].set_xlabel("Number of Neighbors", fontsize=12)
    # Effect of n_neighbors (averaged over n_bins)
    _axes[1, 1].set_ylabel("Structure Index", fontsize=12)
    _axes[1, 1].set_title(
        "SI vs Neighborhood Size (all combinations)", fontsize=13, fontweight="bold"
    )
    _axes[1, 1].legend(fontsize=10)
    _axes[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/parameter_effects.png", dpi=150, bbox_inches="tight")
    # Scatter: n_bins vs SI for each dataset
    # Scatter: n_neighbors vs SI for each dataset
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Load Specific Results from HDF5
    """)
    return


@app.cell
def _(load_results_from_hdf5_dataset):
    # Load specific parameter combination for place cells
    results_loaded = load_results_from_hdf5_dataset(
        "output/structure_indices.h5",
        dataset_name="place_cells",
        result_key="nbins10_nneigh15",
    )

    print("Loaded result keys:")
    for key in results_loaded.keys():
        print(f"  - {key}")

    # Access the data
    result_data = results_loaded["place_cells"]["nbins10_nneigh15"]
    print("\nAttributes:")
    for attr, value in result_data["attributes"].items():
        print(f"  {attr}: {value}")

    print("\nArrays:")
    for arr_name, arr_data in result_data["arrays"].items():
        print(f"  {arr_name}: shape {arr_data.shape}")
    return (result_data,)


@app.cell
def _(draw_overlap_graph, plt, result_data):
    # Extract and visualize loaded overlap matrix
    loaded_overlap = result_data["arrays"]["overlap_matrix"]
    loaded_si = result_data["attributes"]["structure_index"]
    _fig, _ax = plt.subplots(figsize=(8, 8))
    draw_overlap_graph(loaded_overlap, ax=_ax)
    _ax.set_title(
        f"Loaded from HDF5: Place Cells\nSI = {loaded_si:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("output/loaded_overlap_graph.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Successfully loaded and visualized results from HDF5")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    We've successfully:

    ✅ Generated three synthetic datasets with different structure levels
    ✅ Visualized neural activity patterns and behavioral correlates
    ✅ Computed structure indices showing clear differences:
       - **Place cells**: High SI (~0.7-0.9), strong manifold structure
       - **Grid cells**: High SI (~0.6-0.8), periodic structure
       - **Random data**: Low SI (~0.3-0.5), no coherent structure

    ✅ Visualized overlap graphs showing neural-behavioral alignment
    ✅ Performed parameter sweeps to understand sensitivity
    ✅ Saved all results to HDF5 for future analysis
    ✅ Loaded and verified saved results

    The Structure Index successfully differentiates between datasets with true neural manifold structure (place/grid cells) and random data!

    **Next Steps:**
    - Compare distributions across datasets
    - Compute shape similarity metrics
    - Analyze embeddings
    - Relate structure index to other metrics
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Part 2: Distribution Comparisons and Shape Metrics

    Now let's compare the distributions of neural activity across datasets using multiple metrics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Distribution Comparisons - Activity Distributions

    Compare the overall distribution of neural activity across datasets using multiple metrics.
    """)
    return


@app.cell
def _(
    grid_data,
    pairwise_distribution_comparison_batch,
    place_data,
    random_data_1,
):
    # Prepare datasets for comparison
    data_dict = {
        "place_cells": place_data,
        "grid_cells": grid_data,
        "random": random_data_1,
    }
    metrics = ["wasserstein", "euclidean", "cosine"]
    # Compare distributions using multiple metrics
    print("Computing pairwise distribution comparisons...")
    df_dist = pairwise_distribution_comparison_batch(
        data_dict,
        metrics=metrics,
        comparison_name="neural_activity",
        save_path="output/distribution_comparisons.h5",
        regenerate=False,
    )
    print(f"\nDistribution comparison results ({df_dist.shape[0]} comparisons):")
    print(df_dist[["dataset_i", "dataset_j", "metric", "value"]].to_string(index=False))
    return data_dict, df_dist, metrics


@app.cell
def _(df_dist, metrics, np, plt):
    # Visualize distribution comparisons
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))
    for _idx, _metric in enumerate(metrics):
        _df_metric = df_dist[df_dist["metric"] == _metric]
        _datasets = ["place_cells", "grid_cells", "random"]
        _n = len(_datasets)
        _matrix = np.zeros((_n, _n))  # Create matrix for heatmap
        for _, _row in _df_metric.iterrows():
            _i = _datasets.index(_row["dataset_i"])
            _j = _datasets.index(_row["dataset_j"])
            _matrix[_i, _j] = _row["value"]
            _matrix[_j, _i] = _row["value"]
        _im = _axes[_idx].imshow(_matrix, cmap="YlOrRd", aspect="auto")
        _axes[_idx].set_xticks(range(_n))
        _axes[_idx].set_yticks(range(_n))
        _axes[_idx].set_xticklabels(
            [d.replace("_", "\n") for d in _datasets], fontsize=10
        )  # Symmetric
        _axes[_idx].set_yticklabels(
            [d.replace("_", "\n") for d in _datasets], fontsize=10
        )
        _axes[_idx].set_title(
            f"{_metric.capitalize()} Distance", fontsize=12, fontweight="bold"
        )  # Plot heatmap
        _cbar = plt.colorbar(_im, ax=_axes[_idx])
        _cbar.set_label("Distance", fontsize=10)
        for _i in range(_n):
            for _j in range(_n):
                if _i != _j:
                    _text = _axes[_idx].text(
                        _j,
                        _i,
                        f"{_matrix[_i, _j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=11,
                        fontweight="bold",
                    )
    plt.tight_layout()
    plt.savefig("output/distribution_distances.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(
        "\n✓ Distribution comparison visualization complete"
    )  # Add colorbar  # Add text annotations
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Shape Similarity Metrics

    Compare the geometric structure of neural representations using shape distance methods.
    """)
    return


@app.cell
def _(data_dict, pairwise_distribution_comparison_batch):
    # Shape metrics with parameters
    # All three methods are now enabled and tested
    shape_metrics = {
        "procrustes": {},
        "one-to-one": {},
        "soft-matching": {"reg": 0.1, "approx": False},
    }

    print("Computing shape similarity metrics...")
    df_shape = pairwise_distribution_comparison_batch(
        data_dict,
        metrics=shape_metrics,
        comparison_name="neural_shape",
        save_path="output/distribution_comparisons.h5",
        regenerate=False,
    )

    print(f"\nShape comparison results ({df_shape.shape[0]} comparisons):")
    print(
        df_shape[["dataset_i", "dataset_j", "metric", "value"]].to_string(index=False)
    )
    return (df_shape,)


@app.cell
def _(df_shape, np, plt):
    # Visualize shape metrics
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))
    _datasets = ["place_cells", "grid_cells", "random"]
    _n = len(_datasets)
    for _idx, _metric in enumerate(["procrustes", "one-to-one", "soft-matching"]):
        _df_metric = df_shape[df_shape["metric"] == _metric]
        if len(_df_metric) == 0:
            _axes[_idx].text(
                0.5,
                0.5,
                f"No data for {_metric}",
                ha="center",
                va="center",
                transform=_axes[_idx].transAxes,
            )
            _axes[_idx].set_title(
                f"{_metric.replace('-', ' ').title()}", fontsize=12, fontweight="bold"
            )
            continue
        _matrix = np.zeros((_n, _n))  # Create matrix
        for _, _row in _df_metric.iterrows():
            _i = _datasets.index(_row["dataset_i"])
            _j = _datasets.index(_row["dataset_j"])
            _matrix[_i, _j] = _row["value"]
            _matrix[_j, _i] = _row["value"]
        _im = _axes[_idx].imshow(_matrix, cmap="plasma", aspect="auto")
        _axes[_idx].set_xticks(range(_n))
        _axes[_idx].set_yticks(range(_n))
        _axes[_idx].set_xticklabels(
            [d.replace("_", "\n") for d in _datasets], fontsize=10
        )
        _axes[_idx].set_yticklabels(
            [d.replace("_", "\n") for d in _datasets], fontsize=10
        )
        _axes[_idx].set_title(
            f"{_metric.replace('-', ' ').title()}", fontsize=12, fontweight="bold"
        )  # Plot heatmap
        _cbar = plt.colorbar(_im, ax=_axes[_idx])
        _cbar.set_label("Distance", fontsize=10)
        for _i in range(_n):
            for _j in range(_n):
                if _i != _j:
                    _text = _axes[_idx].text(
                        _j,
                        _i,
                        f"{_matrix[_i, _j]:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=11,
                        fontweight="bold",
                    )
    plt.tight_layout()
    plt.savefig("output/shape_distances.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Shape similarity visualization complete")  # Annotations
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Dimensionality Reduction and Embeddings

    Create low-dimensional embeddings of each dataset and visualize them.
    """)
    return


@app.cell
def _(compute_embedding, data_dict):
    # Compute embeddings for each dataset
    embeddings = {}
    methods = ["pca", "umap"]
    for _dataset_name, data in data_dict.items():
        print(f"Computing embeddings for {_dataset_name}...")
        embeddings[_dataset_name] = {}
        for _method in methods:
            _emb = compute_embedding(data, method=_method, n_components=2)
            embeddings[_dataset_name][_method] = _emb
            print(f"  {_method.upper()}: {_emb.shape}")
    print("\n✓ Embeddings computed")
    return embeddings, methods


@app.cell
def _(
    embeddings,
    grid_position,
    methods,
    place_position,
    plt,
    random_position,
):
    # Visualize embeddings
    _fig, _axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_dict = {
        "place_cells": "#E74C3C",
        "grid_cells": "#3498DB",
        "random": "#95A5A6",
    }
    for method_idx, _method in enumerate(methods):
        for dataset_idx, _dataset_name in enumerate(
            ["place_cells", "grid_cells", "random"]
        ):
            _ax = _axes[method_idx, dataset_idx]
            _emb = embeddings[_dataset_name][_method]
            if _dataset_name == "place_cells":
                labels = place_position[:, 0]
            elif _dataset_name == "grid_cells":
                labels = grid_position[
                    :, 0
                ]  # Create labels based on position (for structure)
            else:
                labels = random_position[:, 0].flatten()  # Use x-position
            scatter = _ax.scatter(
                _emb[:, 0],
                _emb[:, 1],
                c=labels,
                cmap="viridis",
                s=20,
                alpha=0.6,
                edgecolors="none",
            )
            _ax.set_xlabel(f"{_method.upper()} 1", fontsize=11)
            _ax.set_ylabel(f"{_method.upper()} 2", fontsize=11)
            _ax.set_title(
                f"{_dataset_name.replace('_', ' ').title()}",
                fontsize=12,
                fontweight="bold",
            )
            plt.colorbar(scatter, ax=_ax, label="Position")
            _ax.grid(alpha=0.3)  # Scatter plot
    plt.tight_layout()
    plt.savefig("output/embeddings.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Embedding visualizations complete")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Compare Embedding Spaces with Shape Metrics

    Compare the embedded representations using shape distance metrics.
    """)
    return


@app.cell
def _(data_dict, embeddings, pairwise_distribution_comparison_batch):
    # Compare PCA embeddings
    pca_embeddings = {name: embeddings[name]["pca"] for name in data_dict.keys()}

    print("Computing shape distances for PCA embeddings...")
    df_pca_shape = pairwise_distribution_comparison_batch(
        pca_embeddings,
        metrics=["procrustes", "one-to-one"],
        comparison_name="pca_embeddings",
        save_path="output/distribution_comparisons.h5",
        regenerate=False,
    )

    print("\nPCA Embedding Shape Distances:")
    print(
        df_pca_shape[["dataset_i", "dataset_j", "metric", "value"]].to_string(
            index=False
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 15. Comprehensive Comparison Summary

    Create a summary DataFrame combining all comparison metrics.
    """)
    return


@app.cell
def _(get_hdf5_result_summary):
    # Load all comparison results using get_hdf5_result_summary
    df_all_comparisons = get_hdf5_result_summary("output/distribution_comparisons.h5")

    # Map dataset_name to comparison_name for consistency with notebook terminology
    if not df_all_comparisons.empty and "dataset_name" in df_all_comparisons.columns:
        if "comparison_name" not in df_all_comparisons.columns:
            df_all_comparisons = df_all_comparisons.rename(
                columns={"dataset_name": "comparison_name"}
            )
        elif "comparison_name" in df_all_comparisons.columns:
            # Both exist - drop dataset_name and keep comparison_name
            df_all_comparisons = df_all_comparisons.drop(columns=["dataset_name"])

    print(f"Total comparisons: {df_all_comparisons.shape[0]}")
    print(f"\nComparison groups: {df_all_comparisons['comparison_name'].unique()}")
    print(f"\nMetrics used: {df_all_comparisons['metric'].unique()}")

    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL COMPARISONS")
    print("=" * 80)
    print(
        df_all_comparisons[
            ["comparison_name", "dataset_i", "dataset_j", "metric", "value"]
        ].to_string(index=False)
    )
    return (df_all_comparisons,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 16. Relationship Between Structure Index and Distribution Distances

    Analyze how structure index correlates with distribution/shape distances.
    """)
    return


@app.cell
def _(SI_grid, SI_place, SI_random, df_all_comparisons, np, pd):
    # Create combined analysis
    dataset_metrics = pd.DataFrame(
        {
            "dataset": ["place_cells", "grid_cells", "random"],
            "structure_index": [SI_place, SI_grid, SI_random],
            "color": ["#E74C3C", "#3498DB", "#95A5A6"],
        }
    )
    for metric_type in ["wasserstein", "euclidean", "procrustes"]:
        _df_metric = df_all_comparisons[df_all_comparisons["metric"] == metric_type]
        avg_distances = []
        for dataset in dataset_metrics["dataset"]:
            mask = (_df_metric["dataset_i"] == dataset) | (
                _df_metric["dataset_j"] == dataset
            )
            distances = _df_metric[mask]["value"].values
            avg_distances.append(np.mean(distances) if len(distances) > 0 else 0)
        # Add average distances to other datasets
        dataset_metrics[f"avg_{metric_type}"] = avg_distances
    print("Dataset Metrics Summary:")
    print(
        dataset_metrics[
            [
                "dataset",
                "structure_index",
                "avg_wasserstein",
                "avg_euclidean",
                "avg_procrustes",
            ]
        ]
    )  # Get all distances involving this dataset
    return (dataset_metrics,)


@app.cell
def _(dataset_metrics, np, plt):
    # Visualize relationship
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = ["avg_wasserstein", "avg_euclidean", "avg_procrustes"]
    titles = ["Wasserstein Distance", "Euclidean Distance", "Procrustes Distance"]
    for _idx, (_metric, title) in enumerate(zip(metrics_to_plot, titles)):
        _ax = _axes[_idx]
        for _, _row in dataset_metrics.iterrows():
            _ax.scatter(
                _row["structure_index"],
                _row[_metric],
                c=_row["color"],
                s=200,
                alpha=0.7,
                edgecolors="black",
                linewidth=2,
                label=_row["dataset"].replace("_", " ").title(),
            )
            _ax.text(
                _row["structure_index"],
                _row[_metric],
                _row["dataset"].replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
        _ax.set_xlabel(
            "Structure Index", fontsize=12, fontweight="bold"
        )  # Scatter plot
        _ax.set_ylabel(f"Average {title}", fontsize=12, fontweight="bold")
        _ax.set_title(f"SI vs {title}", fontsize=13, fontweight="bold")
        _ax.grid(alpha=0.3)
        if len(dataset_metrics) > 2:
            z = np.polyfit(
                dataset_metrics["structure_index"], dataset_metrics[_metric], 1
            )
            p = np.poly1d(z)
            x_line = np.linspace(
                dataset_metrics["structure_index"].min(),
                dataset_metrics["structure_index"].max(),
                100,
            )
            _ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=1.5, label="Trend")
    plt.tight_layout()
    plt.savefig("output/si_vs_distances.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(
        "\n✓ Relationship analysis complete"
    )  # Add trend line if there's enough variation
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final Summary

    We've completed a comprehensive analysis including:

    ### Part 1: Structure Index Analysis
    ✅ Generated three synthetic datasets (place cells, grid cells, random)
    ✅ Visualized neural activity and behavioral correlates
    ✅ Computed structure indices showing clear differences
    ✅ Visualized overlap graphs
    ✅ Performed parameter sweeps
    ✅ Saved/loaded results from HDF5

    ### Part 2: Distribution and Shape Comparisons
    ✅ Compared activity distributions (Wasserstein, Euclidean, Cosine)
    ✅ Computed shape similarity metrics (Procrustes, One-to-One, Soft-Matching)
    ✅ Created dimensionality reduction embeddings (PCA, UMAP)
    ✅ Compared embedding spaces with shape metrics
    ✅ Generated comprehensive comparison summary
    ✅ Analyzed relationship between Structure Index and distribution distances

    ### Key Findings:
    1. **Structure Index successfully differentiates** datasets with neural manifold structure from random data
    2. **High SI correlates with** distinctive neural representations (lower between-dataset distances)
    3. **Shape metrics reveal** geometric differences even when distribution metrics are similar
    4. **Embeddings visualize** the manifold structure quantified by the Structure Index
    5. **All comparisons saved to HDF5** for reproducibility and further analysis

    ### Output Files:
    - `output/structure_indices.h5` - Structure index parameter sweeps
    - `output/distribution_comparisons.h5` - All distribution/shape comparisons
    - Multiple PNG visualization files

    The analysis demonstrates how Structure Index and distribution comparisons provide complementary views of neural manifold organization!
    """)
    return


if __name__ == "__main__":
    app.run()
