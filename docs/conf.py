# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# -- Project information -----------------------------------------------------
project = 'Neural Analysis'
copyright = '2025, Neural Analysis Contributors'
author = 'Neural Analysis Contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.napoleon',          # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other project documentation
    'sphinx_autodoc_typehints',     # Better type hint display
    'myst_parser',                  # Markdown support
]

# Autosummary settings
autosummary_generate = True  # Turn on autosummary for stub files
autosummary_imported_members = False  # Don't include imported members

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_mock_imports = ['matplotlib', 'plotly', 'umap', 'sklearn']
autodoc_type_aliases = {
    'plt.Axes': 'matplotlib.axes.Axes',
    'go.Figure': 'plotly.graph_objects.Figure'
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MyST settings
myst_enable_extensions = [
    "deflist",
    "colon_fence",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# -- Source suffix -----------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
