"""Example: Random cell diagnostic plotting demonstration.

This example shows how the diagnostic plots verify that random cells
truly lack the spatial and directional structure of place, grid, and
head direction cells.
"""

from neural_analysis.data.synthetic_data import generate_random_cells

# Generate random cells with comprehensive diagnostic plotting
print("Generating random cells with diagnostic visualization...")
print("=" * 70)

activity, metadata = generate_random_cells(
    n_cells=25,
    n_samples=600,
    baseline_rate=2.5,
    variability=1.8,
    temporal_smoothness=0.15,
    seed=42,
    plot=True,  # Enable comprehensive plotting
)

print(f"\n✓ Generated {activity.shape[1]} random cells")
print(f"✓ {activity.shape[0]} time points")
print(f"✓ Mean firing rate: {activity.mean():.2f} Hz")
print(f"✓ Baseline: {metadata['baseline_rate']} Hz")
print(f"✓ Variability: {metadata['variability']} Hz")
print(f"✓ Temporal smoothness: {metadata['temporal_smoothness']}")

print("\n" + "=" * 70)
print("DIAGNOSTIC PLOTS EXPLAINED:")
print("=" * 70)
print("""
The comprehensive visualization includes:

1. ACTIVITY RASTER
   - Shows firing rates over time for all cells
   - Should show random, unstructured activity

2. SINGLE CELL 'SPATIAL MAP'
   - Tests if the cell has place field-like structure
   - Should be noisy/uniform (no hotspots)
   - Indicates lack of spatial tuning

3. POPULATION 'COVERAGE'
   - Average firing across all cells in synthetic space
   - Should be uniform (no spatial bias)
   - Verifies no population-level spatial code

4. AUTOCORRELATION
   - Tests for grid-like periodic patterns
   - Should show no hexagonal/regular structure
   - Only central peak (self-correlation)

5. 'DIRECTIONAL TUNING'
   - Tests for head direction-like tuning
   - Should be flat across all directions
   - Indicates no directional preference

6-9. EXAMPLE CELLS (2-4)
   - Time series showing individual firing patterns
   - Should show temporally smooth but spatially random activity

10-11. EMBEDDINGS (PCA & UMAP)
   - Low-dimensional representations
   - Should show diffuse structure (no clear clusters/manifolds)
   - Unlike place/grid cells which form rings/tori

These diagnostics confirm random cells are suitable as negative controls
when testing decoding algorithms or analyzing neural representations.
""")

print("=" * 70)
print("✓ Example complete!")
print("Close the plot window to exit.")
