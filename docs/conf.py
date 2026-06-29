project = "DataForgeML"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "numpydoc",
]
html_theme = "pydata_sphinx_theme"
napoleon_numpy_docstring = True
autodoc_typehints = "description"
numpydoc_show_class_members = False
exclude_patterns = ["_build", "adr/**"]
