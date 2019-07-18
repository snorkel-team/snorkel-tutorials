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


def check_notebook(notebook: Notebook) -> None:
    assert os.path.exists(notebook.py)
    notebook_actual = jupytext.read(notebook.ipynb, fmt=dict(extension="ipynb"))
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
        args = ["jupytext", "--to", "notebook", "--execute", notebook.py, "-o", f.name]
        subprocess.run(args)
        notebook_expected = jupytext.read(f.name, fmt=dict(extension="ipynb"))
        compare_notebooks(notebook_actual, notebook_expected, compare_outputs=True)


@click.command()
@click.argument("tutorial_dir")
def check_notebooks(tutorial_dir: str) -> None:
    path = os.path.abspath(tutorial_dir)
    notebook_paths = glob.glob(os.path.join(path, "*.ipynb"), recursive=True)
    for notebook in map(Notebook, notebook_paths):
        check_notebook(notebook)


if __name__ == "__main__":
    check_notebooks()
