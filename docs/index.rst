Neural Analysis Documentation
==============================

Welcome to Neural Analysis, a comprehensive Python package for neural data analysis and visualization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples
   contributing

Features
--------

**Unified Plotting System**
   - Consistent API across all plot types (1D, 2D, 3D, statistical, heatmaps)
   - Dual backend support: matplotlib and plotly
   - Metadata-driven plotting with PlotGrid architecture

**Comprehensive Plot Types**
   - Line plots with error bars
   - Scatter plots (2D/3D) with color mapping
   - Trajectory visualization with time-based coloring
   - Heatmaps with customizable labels
   - Statistical plots (violin, box, bar)
   - Boolean state visualization
   - KDE plots and grouped scatter

**Advanced Features**
   - Reference lines (horizontal/vertical) with annotations
   - Convex hull visualization for grouped data
   - Flexible configuration with PlotConfig
   - Grid layouts for multi-plot figures
   - Interactive plotly plots with hover information

**Data Analysis Tools**
   - Dimensionality reduction (PCA, t-SNE, UMAP)
   - Distance metrics (Euclidean, Mahalanobis, etc.)
   - Statistical utilities
   - Data I/O with HDF5 support

Installation
------------

.. code-block:: bash

   pip install neural-analysis

Or install from source:

.. code-block:: bash

   git clone https://github.com/Ch3fUlrich/neural-analysis.git
   cd neural-analysis
   pip install -e .

Quick Start
-----------

.. code-block:: python

   from neural_analysis.plotting import plot_line, PlotConfig
   import numpy as np

   # Generate sample data
   x = np.linspace(0, 10, 100)
   y = np.sin(x)

   # Create a simple line plot
   config = PlotConfig(
       title="Sine Wave",
       xlabel="Time (s)",
       ylabel="Amplitude"
   )
   
   fig = plot_line(x, y, config=config)

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details.

License
-------

This project is licensed under the MIT License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
