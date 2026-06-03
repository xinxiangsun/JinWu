"""
Sphinx configuration for JinWu documentation.

JinWu: Joint Inference for high-energy transient light-curve & spectral analysis
        with Unifying physical modeling.
"""
import os
import sys
from pathlib import Path

# -- Path setup ----------------------------------------------------------------
# Add the src/ directory so Sphinx can import jinwu
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -------------------------------------------------------
project = "JinWu"
copyright = "2025-2026, Xinxiang Sun (孙新翔)"
author = "Xinxiang Sun"
release = "0.0.26"
version = "0.0.26"

# -- General configuration -----------------------------------------------------
extensions = [
    # Core Sphinx
    "sphinx.ext.autodoc",           # Pull docstrings from Python code
    "sphinx.ext.napoleon",          # NumPy/Google-style docstrings
    "sphinx.ext.intersphinx",       # Cross-reference other projects
    "sphinx.ext.viewcode",          # Link to source code
    "sphinx.ext.mathjax",           # LaTeX math rendering
    # Auto API generation (simpler than autosummary for large packages)
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    # Markdown support (so we can include README.md)
    "myst_parser",
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST settings
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist"]

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Automodapi settings
automodapi_toctreedirnm = "api"
automodapi_writereprocessed = True
automodsumm_inherited_members = True

# Add __init__ docstrings to package pages
automodapi_inheritance_diagram = False

# -- Options for HTML output ---------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "JinWu Documentation"
html_short_title = "JinWu"

html_theme_options = {
    "github_url": "https://github.com/xinxiangsun/jinwu",
    "show_toc_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xinxiangsun/jinwu",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# These paths are relative to conf.py
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- LaTeX & manual page output ------------------------------------------------
latex_engine = "xelatex"
