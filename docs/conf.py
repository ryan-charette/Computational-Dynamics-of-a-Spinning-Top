"""Sphinx configuration."""

project = "Computational Dynamics of a Spinning Top"
author = "Ryan Charette"
copyright = "2026, Ryan Charette"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]
html_theme = "pydata_sphinx_theme"
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
