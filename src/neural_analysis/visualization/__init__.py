"""Deprecated package namespace.

The package ``neural_analysis.visualization`` has been renamed to
``neural_analysis.plotting``. Please update your imports accordingly.

For example:

    from neural_analysis.plotting import PlotConfig, plot_line, set_backend

Attempting to import from this module will raise an ImportError to surface the
change early during development.
"""

raise ImportError(
    "neural_analysis.visualization is deprecated. Use neural_analysis.plotting instead."
)
