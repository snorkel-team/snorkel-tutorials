# Snorkel Tutorials
A collection of tutorials for Snorkel

## Adding a new tutorial

1. Create a new top-level directory (e.g. `my_tutorial_dir`)
1. Add a `.notebooks` file to your tutorial directory with the base name of your tutorial file (e.g. `my_tutorial`)
1. Add a `requirements.txt` to your directory if additional ones are needed
1. Write your tutorial as a Python script (e.g. `my_tutorial_dir/my_tutorial.py`) in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc)
1. Run `tox -e my_tutorial_dir -- sync` to generate a notebook version. Run this command to update when changes are made to the tutorial script.
1. Add a command to `[testenv]` in `tox.ini` by copying `example` and add the `requirements.txt` file if necessary. Also add the command name to `envlist`.
1. Run `tox -e my_tutorial_dir` to test out your tutorial
