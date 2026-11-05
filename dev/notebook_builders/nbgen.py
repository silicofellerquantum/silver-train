"""Tiny helper to assemble Jupyter notebooks from markdown / code strings."""
import os
import textwrap

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))


def _clean(text):
    return textwrap.dedent(text).strip("\n")


def md(text):
    return nbformat.v4.new_markdown_cell(_clean(text))


def code(text):
    return nbformat.v4.new_code_cell(_clean(text))


def common_imports(extra_scipy_imports="pass"):
    src = open(os.path.join(HERE, "common_imports.py")).read()
    return code(src.replace("__EXTRA_SCIPY_IMPORTS__", extra_scipy_imports))


def common_utils():
    return code(open(os.path.join(HERE, "common_utils.py")).read())


def write_notebook(path, cells, title):
    nb = nbformat.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": title,
    }
    nbformat.validate(nb)
    with open(path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    return path
