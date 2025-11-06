Examples
========

This section contains comprehensive examples demonstrating various features of Neural Analysis.

Example Notebooks
-----------------

The following Jupyter notebooks are available in the ``examples/`` directory:

Plotting Examples
^^^^^^^^^^^^^^^^^

- **plots_1d_examples.ipynb**: Line plots, histograms, boolean states
- **plots_2d_examples.ipynb**: Scatter plots, trajectories, KDE plots, grouped scatter
- **plots_3d_examples.ipynb**: 3D scatter and trajectory visualization
- **statistical_plots_examples.ipynb**: Violin, box, and bar plots
- **plotting_grid_showcase.ipynb**: Advanced multi-panel layouts

Analysis Examples
^^^^^^^^^^^^^^^^^

- **embeddings_demo.ipynb**: PCA, t-SNE, UMAP dimensionality reduction
- **metrics_examples.ipynb**: Distance metrics and statistical measures
- **neural_analysis_demo.ipynb**: Complete workflow demonstration

Utility Examples
^^^^^^^^^^^^^^^^

- **io_h5io_examples.ipynb**: HDF5 data I/O operations
- **logging_examples.ipynb**: Logging and debugging utilities

Advanced Plotting Examples
--------------------------

Trajectory Visualization with Time Coloring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from neural_analysis.plotting import plot_trajectory_2d, PlotConfig
   import numpy as np

   # Generate trajectory data
   t = np.linspace(0, 4*np.pi, 200)
   x = np.cos(t) * (1 + 0.5*np.sin(5*t))
   y = np.sin(t) * (1 + 0.5*np.sin(5*t))
   
   config = PlotConfig(
       title="Spiral Trajectory",
       xlabel="X Position",
       ylabel="Y Position"
   )
   
   fig = plot_trajectory_2d(x, y, color_by='time', config=config)

Grouped Scatter with Convex Hulls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from neural_analysis.plotting import plot_grouped_scatter_2d, PlotConfig
   import numpy as np

   # Generate grouped data
   np.random.seed(42)
   group1 = np.random.randn(50, 2)
   group2 = np.random.randn(50, 2) + [3, 3]
   group3 = np.random.randn(50, 2) + [0, 4]
   
   data = {
       'Group A': group1,
       'Group B': group2,
       'Group C': group3
   }
   
   config = PlotConfig(
       title="Grouped Data with Hulls",
       xlabel="Feature 1",
       ylabel="Feature 2"
   )
   
   fig = plot_grouped_scatter_2d(
       data,
       config=config,
       show_hull=True,
       backend='matplotlib'
   )

Heatmap with Custom Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from neural_analysis.plotting import plot_heatmap, PlotConfig
   import numpy as np

   # Create correlation matrix
   data = np.random.randn(50, 10)
   corr_matrix = np.corrcoef(data.T)
   
   # Define labels
   x_labels = [f'Neuron {i+1}' for i in range(10)]
   y_labels = x_labels
   
   config = PlotConfig(
       title="Neural Correlation Matrix",
       xlabel="Neurons",
       ylabel="Neurons"
   )
   
   fig = plot_heatmap(
       corr_matrix,
       x_labels=x_labels,
       y_labels=y_labels,
       show_values=True,
       value_format='.2f',
       config=config
   )

Multi-Panel Figure with Different Plot Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig
   import numpy as np

   # Generate data
   x = np.linspace(0, 10, 100)
   
   # Create specs for different plot types
   specs = [
       PlotSpec(
           data=np.column_stack([x, np.sin(x)]),
           plot_type='line',
           title='Sine Wave',
           subplot=0
       ),
       PlotSpec(
           data=np.column_stack([x, np.cos(x)]),
           plot_type='line',
           title='Cosine Wave',
           subplot=1
       ),
       PlotSpec(
           data=np.random.randn(100),
           plot_type='histogram',
           title='Random Distribution',
           subplot=2
       ),
       PlotSpec(
           data=np.column_stack([
               np.random.randn(100),
               np.random.randn(100)
           ]),
           plot_type='scatter',
           title='Random Scatter',
           subplot=3
       )
   ]
   
   layout = GridLayoutConfig(rows=2, cols=2, figsize=(12, 10))
   grid = PlotGrid(specs=specs, layout=layout)
   fig = grid.plot()

Complete Analysis Workflow
---------------------------

Here's a complete example combining multiple features:

.. code-block:: python

   from neural_analysis.embeddings import run_pca, run_tsne
   from neural_analysis.metrics import euclidean_distance
   from neural_analysis.plotting import (
       plot_grouped_scatter_2d,
       PlotGrid,
       PlotSpec,
       GridLayoutConfig,
       PlotConfig
   )
   import numpy as np

   # Generate high-dimensional data with groups
   np.random.seed(42)
   group1 = np.random.randn(100, 50)
   group2 = np.random.randn(100, 50) + 2
   data = np.vstack([group1, group2])
   labels = np.array(['A']*100 + ['B']*100)
   
   # Dimensionality reduction
   pca_2d = run_pca(data, n_components=2)
   tsne_2d = run_tsne(data, n_components=2, perplexity=30)
   
   # Calculate distances
   dist = euclidean_distance(
       group1.mean(axis=0),
       group2.mean(axis=0)
   )
   
   # Create visualization
   data_dict = {
       'Group A': pca_2d[:100],
       'Group B': pca_2d[100:]
   }
   
   config = PlotConfig(
       title=f'PCA Projection (Distance: {dist:.2f})',
       xlabel='PC1',
       ylabel='PC2'
   )
   
   fig = plot_grouped_scatter_2d(
       data_dict,
       config=config,
       show_hull=True,
       backend='plotly'
   )

For more examples, see the Jupyter notebooks in the ``examples/`` directory.
