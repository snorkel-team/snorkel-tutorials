import difflib
import glob
import os
from pprint import pprint

import click
import jupytext


class Notebook:
    def __init__(self, notebook_path: str) -> None:
        self.basename = os.path.splitext(notebook_path)[0]

    @property
    def py(self) -> str:
        return f"{self.basename}.py"

    @property
    def ipynb(self) -> str:
        return f"{self.basename}.ipynb"


def check_notebook(notebook: Notebook) -> None:
    assert os.path.exists(notebook.py)
    py_file = jupytext.readf(notebook.py, fmt=dict(extension=".py"))
    ipynb_expected = jupytext.writes(py_file, fmt="ipynb")
    with open(notebook.ipynb, "r") as f:
        ipynb_actual = f.read()
    ipynb_actual_lines = ipynb_actual.splitlines(keepends=False)
    ipynb_expected_lines = ipynb_expected.splitlines(keepends=False)
    if ipynb_actual_lines != ipynb_expected_lines:
        d = difflib.Differ()
        pprint(list(d.compare(ipynb_actual_lines, ipynb_expected_lines)))
        raise ValueError("Notebook version has unexpected differences.")


@click.command()
@click.argument("tutorial_dir")
def check_notebooks(tutorial_dir: str) -> None:
    path = os.path.abspath(tutorial_dir)
    notebook_paths = glob.glob(os.path.join(path, "*.ipynb"), recursive=True)
    for notebook in map(Notebook, notebook_paths):
        check_notebook(notebook)


if __name__ == "__main__":
    check_notebooks()
