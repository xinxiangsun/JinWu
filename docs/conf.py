"""Build current JinWu source documentation locally and on Read the Docs."""
from pathlib import Path
import os
import sys
import tomllib

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_ext"))  # Documentation extension only.
project = "JinWu"
author = "Xinxiang Sun（孙新翔）"
copyright = "2025–2026, Xinxiang Sun"
# Editable distribution metadata can be stale; use the release tool's source.
release = tomllib.loads((ROOT / "packages/jinwu/pyproject.toml").read_text())["project"]["version"]
version = release
language = "zh_CN"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.mathjax", "myst_parser", "jinwu_docs"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
viewcode_follow_imported_members = False  # Source backlinks use canonical module pages.
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
autodoc_mock_imports = ["xspec", "bxa", "ultranest", "naima", "gbm_drm_gen", "responsum", "gbmgeometry", "pandas", "seaborn", "sklearn", "astroquery", "ipyaladin", "plotly", "regions", "swifttools", "batanalysis", "rich", "gdt"]
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
exclude_patterns = ["_build", "api/jinwu.*", "Thumbs.db", ".DS_Store", "locale"]
html_theme = "pydata_sphinx_theme"
html_title = f"JinWu {release} 使用手册"
html_short_title = "JinWu"
html_theme_options = {
    "navbar_align": "left", "show_toc_level": 2, "navigation_depth": 3,
    "show_nav_level": 1, "header_links_before_dropdown": 5,
    "icon_links": [{"name": "GitHub", "url": "https://github.com/xinxiangsun/jinwu", "icon": "fa-brands fa-github"}],
    "footer_start": ["copyright"], "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_context = {"default_mode": "light"}
latex_engine = "xelatex"
os.environ.setdefault("MPLBACKEND", "Agg")
