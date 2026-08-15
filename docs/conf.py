# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import inspect
import importlib

# -- Project information -----------------------------------------------------

project = "crystallite"
copyright = ""
author = "Zachary Morgan"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "numpydoc",
]

templates_path = ["_templates"]

exclude_patterns = []

root_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = True

html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zjmorgan/crystallite",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "show_nav_level": 2,
}

# -- Extension configuration -------------------------------------------------

add_module_names = False


def linkcode_resolve(domain, info):
    baseurl = "https://github.com/zjmorgan/crystallite/blob/main/src/{}.py"
    if "py" not in domain:
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    url = baseurl.format(filename)
    mod = importlib.import_module(info["module"])
    objname, *attrname = info["fullname"].split(".")
    obj = getattr(mod, objname)
    if attrname:
        for attr in attrname:
            obj = getattr(obj, attr)
    try:
        lines = inspect.getsourcelines(obj)
        start, stop = lines[1], lines[1] + len(lines[0]) - 1
        return "{}#L{}-L{}".format(url, start, stop)
    except (TypeError, OSError):
        return url
