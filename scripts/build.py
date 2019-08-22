import logging
import os
import re
import subprocess
import tempfile
import urllib
import yaml
from typing import List, Optional

import click
import jupytext
from jupytext.compare import compare_notebooks

logging.basicConfig(level=logging.INFO)


NOTEBOOKS_CONFIG_FNAME = ".notebooks"
SCRIPTS_CONFIG_FNAME = ".scripts"
EXCLUDE_CELL_TAG = "md-exclude"
EXCLUDE_OUTPUT_TAG = "md-exclude-output"
BUILD_DIR = "build"
WEB_YML = ".web.yml"


HEADER_TEMPLATE = """---
layout: default
title: {title}
description: {description}
excerpt: {description}
order: {order}
github_link: {github_link}
---

"""


GITHUB_LINK_TEMPLATE = (
    "https://github.com/snorkel-team/snorkel-tutorials/blob/master/{notebook_path}"
)


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


class MarkdownHeader:
    def __init__(
        self, title: str, description: str, order: int, github_link: str
    ) -> None:
        self.title = title
        self.description = description
        self.order = order
        self.github_link = github_link

    def render(self):
        return HEADER_TEMPLATE.format(
            title=self.title,
            description=self.description,
            order=self.order,
            github_link=self.github_link,
        )


class TutorialWebpage:
    def __init__(
        self,
        ipynb_path: str,
        header: Optional[MarkdownHeader],
        exclude_all_output: bool,
    ) -> None:
        self.ipynb = ipynb_path
        self.header = header
        self.exclude_all_output = exclude_all_output

    def markdown_path(self) -> str:
        return os.path.join(
            BUILD_DIR, f"{os.path.splitext(os.path.basename(self.ipynb))[0]}.md"
        )


def parse_web_yml(tutorial_dir: Optional[str]) -> List[TutorialWebpage]:
    # Read .web.yml
    with open(WEB_YML, "r") as f:
        web_config = yaml.safe_load(f)
    tutorial_webpages = []
    # Process webpage configs in order
    i = 1
    for cfg in web_config["tutorials"]:
        # If tutorial directory specified, skip if not in specified directory
        notebook_path = cfg["notebook"]
        notebook_dir = notebook_path.split("/")[0]
        if tutorial_dir is not None and notebook_dir != tutorial_dir:
            continue
        # If full notebook path supplied, just use that
        if notebook_path.endswith(".ipynb"):
            notebook = Notebook(os.path.abspath(notebook_path))
        # If only directory supply, ensure that there's only one notebook
        else:
            notebooks = get_notebooks(notebook_path)
            if len(notebooks) > 1:
                raise ValueError(f"Multiple notebooks found in {notebook_path}")
            notebook = notebooks[0]
        # If no title or description, don't generate order for header
        title = cfg.get("title")
        description = cfg.get("description")
        if title is not None and description is not None:
            full_notebook_path = notebook.ipynb.split("/snorkel-tutorials/")[-1]
            github_link = GITHUB_LINK_TEMPLATE.format(notebook_path=full_notebook_path)
            header = MarkdownHeader(title, description, i, github_link)
            i += 1
        else:
            header = None
        # Create TutorialWebpage object
        tutorial_webpages.append(
            TutorialWebpage(
                ipynb_path=notebook.ipynb,
                header=header,
                exclude_all_output=cfg.get("exclude_all_output", False),
            )
        )
    return tutorial_webpages


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
            logging.warning(
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
    os.environ["IS_TEST"] = "true"
    logging.info(f"Checking links in [{notebook.py}]")
    check_links(notebook.py)
    notebook_actual = jupytext.read(notebook.ipynb, fmt=dict(extension="ipynb"))
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
        logging.info(f"Executing notebook [{notebook.py}]")
        call_jupytext(notebook, f.name, to_ipynb=True)
        notebook_expected = jupytext.read(f.name, fmt=dict(extension="ipynb"))
        # notebook_metadata_filter gets flipped during execution. Remove it to ensure
        # all metadata is tested.
        notebook_actual.metadata.get("jupytext", {}).pop(
            "notebook_metadata_filter", None
        )
        notebook_expected.metadata.get("jupytext", {}).pop(
            "notebook_metadata_filter", None
        )
        compare_notebooks(notebook_actual, notebook_expected)


def check_script(script_path: str) -> None:
    assert os.path.exists(script_path), f"No file {script_path}"
    logging.info(f"Checking links in [{script_path}]")
    check_links(script_path)
    logging.info(f"Executing script [{script_path}]")
    check_run = subprocess.run(["python", script_path])
    if check_run.returncode:
        raise ValueError(f"Error running {script_path}")


def build_markdown_notebook(tutorial: TutorialWebpage) -> None:
    assert os.path.exists(tutorial.ipynb), f"No file {tutorial.ipynb}"
    os.makedirs(BUILD_DIR, exist_ok=True)
    # Call nbconvert
    args = [
        "jupyter",
        "nbconvert",
        tutorial.ipynb,
        "--to",
        "markdown",
        f"--TagRemovePreprocessor.remove_cell_tags={{'{EXCLUDE_CELL_TAG}'}}",
        f"--TagRemovePreprocessor.remove_all_outputs_tags={{'{EXCLUDE_OUTPUT_TAG}'}}",
        "--output-dir",
        BUILD_DIR,
    ]
    if tutorial.exclude_all_output:
        args.append("--TemplateExporter.exclude_output=True")
    subprocess.run(args, check=True)
    # Prepend header by reading generated file then writing back
    if tutorial.header is not None:
        with open(tutorial.markdown_path(), "r") as f:
            content = f.read()
        with open(tutorial.markdown_path(), "w") as f:
            f.write(tutorial.header.render() + content)


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
@click.option("--tutorial-dir")
def markdown(tutorial_dir: Optional[str]) -> None:
    for tutorial_webpage in parse_web_yml(tutorial_dir):
        build_markdown_notebook(tutorial_webpage)


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
