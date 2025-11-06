Quick Start Guide
=================

This guide will help you get started with Neural Analysis quickly.

Basic Plotting
--------------

Line Plots
^^^^^^^^^^

.. code-block:: python

   from neural_analysis.plotting import plot_line, PlotConfig
   import numpy as np

   # Generate data
   x = np.linspace(0, 10, 100)
   y = np.sin(x)
   
   # Create configuration
   config = PlotConfig(
       title="Sine Wave",
       xlabel="Time (s)",
       ylabel="Amplitude",
       figsize=(10, 6)
   )
   
   # Plot
   fig = plot_line(x, y, config=config)

Scatter Plots
^^^^^^^^^^^^^

.. code-block:: python

   from neural_analysis.plotting import plot_scatter_2d, PlotConfig
   import numpy as np

   # Generate data
   x = np.random.randn(100)
   y = np.random.randn(100)
   colors = np.random.rand(100)
   
   # Create plot
   config = PlotConfig(title="Random Scatter", xlabel="X", ylabel="Y")
   fig = plot_scatter_2d(x, y, colors=colors, config=config)

Advanced Features
-----------------

Reference Lines
^^^^^^^^^^^^^^^

Add horizontal/vertical reference lines with annotations:

.. code-block:: python

   from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig
   import numpy as np

   x = np.linspace(0, 10, 100)
   y = np.sin(x)
   
   spec = PlotSpec(
       data=np.column_stack([x, y]),
       plot_type='line',
       hlines=[
           {'y': 0.5, 'color': 'red', 'linestyle': '--', 'linewidth': 2},
           {'y': -0.5, 'color': 'blue', 'linestyle': '--', 'linewidth': 2}
       ],
       vlines=[
           {'x': 5, 'color': 'green', 'linestyle': ':', 'linewidth': 2}
       ],
       annotations=[
           {
               'text': 'Peak',
               'xy': (np.pi/2, 1.0),
               'xytext': (np.pi/2 + 1, 1.2),
               'arrowprops': {'arrowstyle': '->', 'color': 'red'}
           }
       ]
   )
   
   config = PlotConfig(title="Sine with References")
   grid = PlotGrid(specs=[spec], config=config)
   fig = grid.plot()

Multiple Subplots
^^^^^^^^^^^^^^^^^

Create multi-panel figures:

.. code-block:: python

   from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig
   import numpy as np

   # Create multiple specs
   specs = []
   for i in range(4):
       x = np.linspace(0, 10, 100)
       y = np.sin(x + i * np.pi/4)
       spec = PlotSpec(
           data=np.column_stack([x, y]),
           plot_type='line',
           title=f"Phase {i*45}°",
           subplot=i
       )
       specs.append(spec)
   
   # Create grid with 2x2 layout
   layout = GridLayoutConfig(rows=2, cols=2)
   grid = PlotGrid(specs=specs, layout=layout)
   fig = grid.plot()

Backend Selection
-----------------

Neural Analysis supports both matplotlib and plotly backends:

.. code-block:: python

   # Use matplotlib (default)
   fig = plot_line(x, y, backend='matplotlib')
   
   # Use plotly for interactive plots
   fig = plot_line(x, y, backend='plotly')

Dimensionality Reduction
-------------------------

.. code-block:: python

   from neural_analysis.embeddings import run_pca, run_tsne
   import numpy as np

   # High-dimensional data
   data = np.random.randn(100, 50)
   
   # PCA
   pca_result = run_pca(data, n_components=2)
   
   # t-SNE
   tsne_result = run_tsne(data, n_components=2, perplexity=30)

Distance Metrics
----------------

.. code-block:: python

   from neural_analysis.metrics import euclidean_distance, mahalanobis_distance
   import numpy as np

   # Two datasets
   A = np.random.randn(100, 10)
   B = np.random.randn(100, 10)
   
   # Calculate distances
   euclidean = euclidean_distance(A, B)
   mahalanobis = mahalanobis_distance(A, B)

Next Steps
----------

- Explore the :doc:`api/index` for complete API documentation
- Check out :doc:`examples` for more detailed examples
- Read :doc:`contributing` to contribute to the project
