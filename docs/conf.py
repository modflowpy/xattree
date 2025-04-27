# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Xattree'
copyright = '2025, Xattree Developers'
author = 'Xattree Developers'
release = '0.1.0.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.mermaid",
    "myst_parser",
    "nbsphinx",
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    "github_user": "modflowpy",
    "github_repo": "xattree",
    "github_version": "develop",
    "doc_path": "docs",
}
html_static_path = ['_static']
html_theme = "pydata_sphinx_theme"
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#references
html_theme_options = {
    "announcement": "This project is highly experimental.",
    # "logo": {"image_dark": "???"},
    "github_url":"https://github.com/modflowpy/xattree",
    "navbar_align": "left",
}
html_show_sourcelink = True
html_logo = "_static/xattree.svg"


# -- nbsphinx configuration -------------------------------------------------
nbsphinx_custom_formats = {
    '.py': ['jupytext.reads', {'fmt': 'py:light'}],
}
