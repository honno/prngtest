project = "prngtest"
copyright = "2022, Matthew Barber"
author = "Matthew Barber"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_theme_options = {
    "nosidebar": True,
}
collapse_navigation = True

autodoc_member_order = "bysource"
autodoc_typehints = "none"
