import logging
import os
import re
import subprocess
import tempfile
import urllib
from typing import List

import click
import jupytext
from jupytext.compare import compare_notebooks

logging.basicConfig(level=logging.INFO)


NOTEBOOKS_CONFIG_FNAME = ".notebooks"
SCRIPTS_CONFIG_FNAME = ".scripts"
EXCLUDE_TAG = "md-exclude"

# Credit to: https://gist.github.com/pchc2005/b5f13e136a9c9bb2984e5b92802fc7c9
# Original source: https://gist.github.com/dperini/729294
MARKDOWN_URL_REGEX = re.compile(
    "\("
    # protocol identifier
    "(?:(?:(?:https?|ftp):)?//)"
    # user:pass authentication
    "(?:\S+(?::\S*)?@)?" "(?:"
    # IP address exclusion
    # private & local networks
    "(?!(?:10|127)(?:\.\d{1,3}){3})"
    "(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    "(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    "(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    "(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    "(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    "|"
    # host & domain names, may end with dot
    # can be replaced by a shortest alternative
    # u"(?![-_])(?:[-\w\u00a1-\uffff]{0,63}[^-_]\.)+"
    # u"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # # domain name
    # u"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    "(?:"
    "(?:"
    "[a-z0-9\u00a1-\uffff]"
    "[a-z0-9\u00a1-\uffff_-]{0,62}"
    ")?"
    "[a-z0-9\u00a1-\uffff]\."
    ")+"
    # TLD identifier name, may end with dot
    "(?:[a-z\u00a1-\uffff]{2,}\.?)" ")"
    # port number (optional)
    "(?::\d{2,5})?"
    # resource path (optional)
    "(?:[/?#]\S*)?" "\)",
    re.UNICODE | re.I,
)


class Notebook:
    def __init__(self, notebook_path: str) -> None:
        self.basename = os.path.splitext(notebook_path)[0]

    @property
    def py(self) -> str:
        return f"{self.basename}.py"

    @property
    def ipynb(self) -> str:
        return f"{self.basename}.ipynb"


def check_links(script_path: str) -> None:
    with open(script_path, "r") as f:
        contents = f.read()
    link_matches = list(MARKDOWN_URL_REGEX.finditer(contents))
    for link_match in link_matches:
        url = link_match.group(0).rstrip(")").lstrip("(")
        req = urllib.request.Request(url, headers={"User-Agent": "Magic Browser"})
        logging.info(f"Checking link [{url}]")
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            raise ValueError(f"Bad link [{url}] found in {script_path}: {e}")
        except Exception as e:
            logging.warn(
                f"SKIPPING: Could not access [{url}] found in {script_path}: {e}"
            )


def call_jupytext(notebook: Notebook, out_fname: str, to_ipynb: bool) -> None:
    to_fmt = "ipynb" if to_ipynb else "py:percent"
    from_fmt = "py:percent" if to_ipynb else "ipynb"
    args = [
        "jupytext",
        "--to",
        to_fmt,
        "--from",
        from_fmt,
        "--opt",
        "notebook_metadata_filter=-all",
        "--opt",
        "cell_metadata_filter=tags",
        notebook.py if to_ipynb else notebook.ipynb,
        "-o",
        out_fname,
    ]
    if to_ipynb:
        args.append("--execute")
    subprocess.run(args, check=True)


def get_notebooks(tutorial_dir: str) -> List[Notebook]:
    path = os.path.abspath(tutorial_dir)
    config_path = os.path.join(path, NOTEBOOKS_CONFIG_FNAME)
    if not os.path.isfile(config_path):
        logging.info(f"No {NOTEBOOKS_CONFIG_FNAME} config file in {path}")
        return []
    with open(config_path, "r") as f:
        notebooks = f.read().splitlines()
    return [Notebook(os.path.join(path, nb)) for nb in notebooks if nb]


def get_scripts(tutorial_dir: str) -> List[Notebook]:
    path = os.path.abspath(tutorial_dir)
    config_path = os.path.join(path, SCRIPTS_CONFIG_FNAME)
    if not os.path.isfile(config_path):
        logging.info(f"No {SCRIPTS_CONFIG_FNAME} config file in {path}")
        return []
    with open(config_path, "r") as f:
        scripts = [os.path.join(path, s) for s in f.read().splitlines() if s]
    return scripts


def check_notebook(notebook: Notebook) -> None:
    assert os.path.exists(notebook.py), f"No file {notebook.py}"
    logging.info(f"Checking links in [{notebook.py}]")
    check_links(notebook.py)
    notebook_actual = jupytext.read(notebook.ipynb, fmt=dict(extension="ipynb"))
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
        logging.info(f"Executing notebook [{notebook.py}]")
        call_jupytext(notebook, f.name, to_ipynb=True)
        notebook_expected = jupytext.read(f.name, fmt=dict(extension="ipynb"))
        compare_notebooks(notebook_actual, notebook_expected)


def check_script(script_path: str) -> None:
    assert os.path.exists(script_path), f"No file {script_path}"
    logging.info(f"Checking links in [{script_path}]")
    check_links(script_path)
    logging.info(f"Executing script [{script_path}]")
    check_run = subprocess.run(["python", script_path])
    if check_run.returncode:
        raise ValueError(f"Error running {script_path}")


def build_markdown_notebook(
    notebook: Notebook, build_dir: str, exclude_output: bool
) -> None:
    assert os.path.exists(notebook.ipynb), f"No file {notebook.ipynb}"
    os.makedirs(build_dir, exist_ok=True)
    args = [
        "jupyter",
        "nbconvert",
        notebook.ipynb,
        "--to",
        "markdown",
        f"--TagRemovePreprocessor.remove_cell_tags={{'{EXCLUDE_TAG}'}}",
        "--output-dir",
        build_dir,
    ]
    if exclude_output:
        args.append("--TemplateExporter.exclude_output=True")
    subprocess.run(args, check=True)


def sync_notebook(notebook: Notebook) -> None:
    assert os.path.exists(notebook.py), f"No file {notebook.py}"
    call_jupytext(notebook, notebook.ipynb, to_ipynb=True)


def sync_py(notebook: Notebook) -> None:
    assert os.path.exists(notebook.ipynb), f"No file {notebook.ipynb}"
    call_jupytext(notebook, notebook.py, to_ipynb=False)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("tutorial_dir")
def test(tutorial_dir: str) -> None:
    for notebook in get_notebooks(tutorial_dir):
        check_notebook(notebook)
    for script in get_scripts(tutorial_dir):
        check_script(script)


@cli.command()
@click.argument("tutorial_dir")
@click.option("--exclude-output", is_flag=True)
def markdown(tutorial_dir: str, exclude_output: bool) -> None:
    build_dir = os.path.abspath(os.path.join(tutorial_dir, "..", "build"))
    for notebook in get_notebooks(tutorial_dir):
        build_markdown_notebook(notebook, build_dir, exclude_output)


@cli.command()
@click.argument("tutorial_dir")
@click.option("--py", is_flag=True)
def sync(tutorial_dir: str, py: bool) -> None:
    for notebook in get_notebooks(tutorial_dir):
        if py:
            sync_py(notebook)
        else:
            sync_notebook(notebook)


if __name__ == "__main__":
    cli()
