Installation
============

Requirements
------------

- Python >= 3.10
- NumPy >= 1.20
- Matplotlib >= 3.5 (optional, for matplotlib backend)
- Plotly >= 5.0 (optional, for plotly backend)
- scikit-learn (optional, for dimensionality reduction)

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install neural-analysis

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/Ch3fUlrich/neural-analysis.git
   cd neural-analysis
   pip install -e ".[dev]"

This will install the package in editable mode along with development dependencies.

Optional Dependencies
---------------------

Install with all optional dependencies:

.. code-block:: bash

   pip install "neural-analysis[all]"

Or install specific optional features:

.. code-block:: bash

   # For plotly support
   pip install "neural-analysis[plotly]"
   
   # For UMAP support
   pip install "neural-analysis[umap]"
   
   # For development tools
   pip install "neural-analysis[dev]"

Verification
------------

Verify your installation:

.. code-block:: python

   import neural_analysis
   print(neural_analysis.__version__)
   
   # Run a quick test
   from neural_analysis.plotting import plot_line
   import numpy as np
   
   x = np.linspace(0, 10, 100)
   y = np.sin(x)
   fig = plot_line(x, y)
   print("Installation successful!")
