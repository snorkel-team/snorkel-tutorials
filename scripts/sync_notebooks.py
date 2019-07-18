import glob
import os
import subprocess
import tempfile

import click
import jupytext
from jupytext.compare import compare_notebooks


class Notebook:
    def __init__(self, notebook_path: str) -> None:
        self.basename = os.path.splitext(notebook_path)[0]

    @property
    def py(self) -> str:
        return f"{self.basename}.py"

    @property
    def ipynb(self) -> str:
        return f"{self.basename}.ipynb"


def call_jupytext(notebook: Notebook, out_fname: str) -> None:
    args = [
        "jupytext",
        "--to",
        "ipynb",
        "--from",
        "py:percent",
        "--opt",
        "notebook_metadata_filter=-all",
        "--execute",
        notebook.py,
        "-o",
        out_fname,
    ]
    subprocess.run(args)


def check_notebook(notebook: Notebook) -> None:
    assert os.path.exists(notebook.py)
    notebook_actual = jupytext.read(notebook.ipynb, fmt=dict(extension="ipynb"))
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
        call_jupytext(notebook, f.name)
        notebook_expected = jupytext.read(f.name, fmt=dict(extension="ipynb"))
        compare_notebooks(notebook_actual, notebook_expected, compare_outputs=True)


def sync_notebook(notebook: Notebook) -> None:
    call_jupytext(notebook, notebook.ipynb)


@click.command()
@click.argument("tutorial_dir")
@click.option("--test", is_flag=True)
def sync_notebooks(tutorial_dir: str, test: bool) -> None:
    path = os.path.abspath(tutorial_dir)
    pattern = "*.ipynb" if test else "*.py"
    notebook_paths = glob.glob(os.path.join(path, pattern), recursive=True)
    for notebook in map(Notebook, notebook_paths):
        if test:
            check_notebook(notebook)
        else:
            sync_notebook(notebook)


if __name__ == "__main__":
    sync_notebooks()
