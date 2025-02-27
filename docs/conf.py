# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- convert tutorial scripts and run example notebooks ----------------------
on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    cmd = ("python3", "-m", "ipython", "kernel", "install", "--user", "--name", "xattree")
    print(" ".join(cmd))
    os.system(" ".join(cmd))
nbs_py = list(Path("examples").glob("*.py"))
for py in nbs_py:
    ipynb = py.with_suffix(".ipynb")
    if ipynb.exists():
        print(f"{ipynb} already exists, skipping")
        continue
    cmd = ("jupytext", "--to", "ipynb", "--execute", str(py))
    print(" ".join(cmd))
    os.system(" ".join(cmd))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xattree'
copyright = '2025, Xattree Developers'
author = 'Wes Bonelli, Michael Reno, Marnix Kraus'
release = '0.1.0.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "nbsphinx",
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
