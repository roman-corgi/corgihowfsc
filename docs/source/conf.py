# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Make the repository root importable so `import corgihowfsc` works.
# This file lives in docs/source/conf.py, so repo root is two levels up.
sys.path.insert(0, os.path.abspath("../.."))

project = 'corgihowfsc'
copyright = '2025, Roman Corongraph CPP Team'
author = 'Roman Corongraph CPP Team'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    "sphinx.ext.autosummary",
    'myst_parser',  # Add this for Markdown support
]

# Generate autosummary pages automatically (useful for API docs)
autosummary_generate = True

# Sensible defaults for autodoc blocks
autodoc_default_options = {
    "members": True,
    "show_inheritance": True,
    "member_order": "bysource",
}

# Configure MyST
myst_enable_extensions = [
    "colon_fence",      # ::: fences for directives
    "deflist",          # Definition lists
    "html_image",       # HTML images
    "linkify",          # Auto-link URLs
    "replacements",     # Text replacements
    "smartquotes",      # Smart quotes
    "substitution",     # Variable substitution
    "tasklist",         # Task lists
]

# File formats
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# GitHub Pages
html_baseurl = 'https://roman-corgi.github.io/corgihowfsc/'
