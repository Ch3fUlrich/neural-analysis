"""Test colorbar with PlotGrid."""
import numpy as np
import matplotlib.pyplot as plt
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Generate test data
np.random.seed(42)
data = np.random.randn(100, 2)
colors = np.linspace(0, 1, 100)

# Create plot spec
spec = PlotSpec(
    data=data,
    plot_type='scatter',
    title='Test Colorbar',
    colors=colors,
    cmap='viridis',
    colorbar=True,
    colorbar_label='Test Values',
    alpha=0.8,
    marker_size=50,
)

config = PlotConfig(
    xlabel='X',
    ylabel='Y',
    figsize=(8, 6),
)

grid = PlotGrid(plot_specs=[spec], config=config)
result = grid.plot()
print(f"Result type: {type(result)}")
print(f"Result: {result}")

plt.savefig('/tmp/test_colorbar.png')
print("✓ Saved to /tmp/test_colorbar.png")
