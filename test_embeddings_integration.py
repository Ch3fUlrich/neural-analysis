"""Quick test to verify embeddings integration works."""
import numpy as np
import matplotlib.pyplot as plt

from neural_analysis.embeddings import (
    compute_embedding,
    compute_multiple_embeddings,
    pca_explained_variance,
)
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig, GridLayoutConfig

# Generate test data
np.random.seed(42)
n_samples = 500
n_features = 50

# Create structured data
data = np.random.randn(n_samples, n_features) * 0.5
for i in range(5):
    signal = np.sin(np.linspace(0, 2*np.pi, n_samples) + i * np.pi/3) * (5 - i)
    data[:, i] += signal

# Time colors
time_colors = np.linspace(0, 1, n_samples)

print(f"Test data shape: {data.shape}")

# Test variance analysis
variance_info = pca_explained_variance(data, cumulative=True)
print(f"Components for 95% variance: {variance_info['n_components_95']}")

# Test embeddings
embeddings = compute_multiple_embeddings(
    data,
    methods=['pca', 'tsne'],
    n_components=2,
    random_state=42
)
print(f"Computed embeddings: {list(embeddings.keys())}")

# Test PlotGrid with colors
pca_embedding = embeddings['pca']
spec = PlotSpec(
    data=pca_embedding,
    plot_type='scatter',
    title='PCA Test',
    colors=time_colors,
    cmap='viridis',
    alpha=0.6,
    marker_size=20,
)

grid = PlotGrid(plot_specs=[spec], config=PlotConfig(figsize=(8, 6)))
fig = grid.plot()
plt.savefig('/tmp/test_embeddings.png')
print("✓ Test successful! Plot saved to /tmp/test_embeddings.png")
