"""JinWu Sphinx config."""
import importlib
import sys
from unittest.mock import MagicMock

# -- Mock environment-specific modules BEFORE any jinwu import ----------------
# sphinx_automodapi imports the package during builder-inited, before autodoc
# runs, so autodoc_mock_imports is too late.  We must mock at sys.modules level.
if "xspec" not in sys.modules:
    sys.modules["xspec"] = MagicMock()


def _use_or_mock(name: str) -> None:
    """Import *name* for real when possible; mock it when unavailable.

    Unconditionally mocking a package whose real parents import cleanly
    breaks submodule imports: a MagicMock has no usable ``__path__``, so
    ``import parent.child`` fails with "is not a package".  Try the real
    import first and only fall back to a sys.modules mock.  Local builds
    with a working GDT keep the real modules (better docstrings); Read the
    Docs, which has no GDT, ends up with everything mocked as before.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except Exception:
        sys.modules[name] = MagicMock()


# Every dotted GDT path imported (directly or lazily) by jinwu.fermi.gbm;
# parents are listed before children so the mock chain stays importable.
_gdt_modules = [
    "gdt",
    "gdt.core",
    "gdt.core.data_primitives",
    "gdt.core.plot",
    "gdt.core.plot.sky",
    "gdt.core.plot.plot",
    "gdt.missions",
    "gdt.missions.fermi",
    "gdt.missions.fermi.plot",
    "gdt.missions.fermi.gbm",
    "gdt.missions.fermi.gbm.detectors",
    "gdt.missions.fermi.gbm.finders",
    "gdt.missions.fermi.gbm.poshist",
    "gdt.missions.fermi.gbm.localization",
    "gdt.missions.fermi.gbm.saa",
]
for _mod in _gdt_modules:
    _use_or_mock(_mod)

# -- Path setup ----------------------------------------------------------------
# No sys.path hack: the docs build requires the monorepo distributions to be
# installed (see docs/index.rst "For development" and .readthedocs.yaml).
# The repo root has no src/ and the root pyproject is not installable.

# -- Project information -------------------------------------------------------
project = "JinWu"
copyright = "2025-2026, Xinxiang Sun (孙新翔)"
author = "Xinxiang Sun"

# Read version from the installed distribution metadata (single source of truth:
# packages/jinwu/pyproject.toml).  Fail loudly instead of silently drifting.
try:
    from importlib.metadata import version as _dist_version

    release = _dist_version("jinwu")
except Exception as exc:
    raise RuntimeError(
        "Cannot determine JinWu version: the 'jinwu' distribution is not "
        "installed in this environment. Install the monorepo packages first "
        "(see docs/index.rst → Installation → For development)."
    ) from exc
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
# 版本来自发行元数据（见上方 release），与发布的 jinwu 包保持一致且可见
html_title = f"JinWu Documentation (v{release})"
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
