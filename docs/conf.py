"""JinWu Sphinx config."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# -- Mock environment-specific modules BEFORE any jinwu import ----------------
# sphinx_automodapi imports the package during builder-inited, before autodoc
# runs, so autodoc_mock_imports is too late.  We must mock at sys.modules level.
if "xspec" not in sys.modules:
    sys.modules["xspec"] = MagicMock()

# GDT submodules (astro-gdt may be installed but sub-path varies by version;
# our mock must sit in sys.modules before the real gdt.missions parent prevents
# traversal into non-existent .fermi child.)
_gdt_mocks = [
    "gdt.missions.fermi",
    "gdt.missions.fermi.gbm",
    "gdt.missions.fermi.gbm.detectors",
]
for _mod in _gdt_mocks:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# If gdt itself isn't installed at all, also mock it (belt and suspenders)
if "gdt" not in sys.modules:
    sys.modules["gdt"] = MagicMock()

# -- Path setup ----------------------------------------------------------------
# Add the src/ directory so Sphinx can import jinwu
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -------------------------------------------------------
project = "JinWu"
copyright = "2025-2026, Xinxiang Sun (孙新翔)"
author = "Xinxiang Sun"

# Read version from jinwu itself (works for both editable and installed)
try:
    import jinwu
    release = jinwu.__version__
except Exception:
    release = "0.0.27"
version = release.rsplit(".", 1)[0] if "." in release else release

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

# -- Internationalization ------------------------------------------------------
# Supported languages.  RTD sets 'language' config at build time via
# the project's Admin → Languages setting.  Do NOT hardcode 'language' here.
locale_dirs = ["locale/"]
gettext_compact = False          # one .po per doc, not per directory

# Mock troublesome imports that automodapi can't handle
autodoc_mock_imports = [
    "xspec",                # HEASoft XSPEC — not available on RTD, top-level import in core/plot.py
    "jinwu.cluster",        # ImportError in cluster.cluster
]

# Suppress unresolvable cross-reference warnings
nitpick_ignore = [
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "Path"),
    ("py:class", "pathlib.Path"),
    ("py:class", "astropy.time.Time"),
    ("py:class", "astropy.units.Quantity"),
    ("py:class", "matplotlib.figure.Figure"),
    ("py:class", "matplotlib.axes.Axes"),
]

# -- LaTeX & manual page output ------------------------------------------------
latex_engine = "xelatex"
