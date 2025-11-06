"""Debug colorbar issue."""
import numpy as np
import matplotlib.pyplot as plt
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Generate test data
np.random.seed(42)
data = np.random.randn(100, 2)
colors = np.linspace(0, 1, 100)

print(f"Data shape: {data.shape}")
print(f"Colors shape: {colors.shape}")
print(f"Colors is not None: {colors is not None}")

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

print(f"Spec.colorbar: {spec.colorbar}")
print(f"Spec.colors is not None: {spec.colors is not None}")
print(f"Spec.cmap: {spec.cmap}")

config = PlotConfig(
    xlabel='X',
    ylabel='Y',
    figsize=(8, 6),
)

grid = PlotGrid(plot_specs=[spec], config=config)
result = grid.plot()

# Check if colorbar is present
fig = result.get_figure() if hasattr(result, 'get_figure') else None
print(f"Figure: {fig}")
if fig:
    print(f"Number of axes: {len(fig.axes)}")
    for i, ax in enumerate(fig.axes):
        print(f"  Axis {i}: {ax}")

plt.savefig('/tmp/test_colorbar_debug.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Saved to /tmp/test_colorbar_debug.png")
