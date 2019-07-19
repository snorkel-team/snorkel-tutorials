# Snorkel Tutorials
A collection of tutorials for Snorkel

## Adding a new tutorial

See the `example` directory for an example.

1. Create a new top-level directory (e.g. `my_tutorial_dir`)
1. Add a file called `.notebooks` to your tutorial directory and add a line to it with for each tutorial script/notebook pair with the base name of the files (e.g. `my_tutorial`).
1. Add a `requirements.txt` to your directory if additional ones are needed
1. Add a command to `[testenv]` in `tox.ini` by copying `example` and add the `requirements.txt` file if necessary. Also add the command name to `envlist`.
1. Write your tutorial either as a Python script (e.g. `my_tutorial_dir/my_tutorial.py`) in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc) or a Jupyter notebook
    * Run `tox -e my_tutorial_dir -- sync` to generate a notebook version from the Python script version. Run this command to update when changes are made to the tutorial script.
    * Run `tox -e my_tutorial_dir -- sync --py` to generate a Python script version from the notebook version. Run this command to update when changes are made to the tutorial notebook.
1. Run `tox -e my_tutorial_dir` to test out your tutorial
