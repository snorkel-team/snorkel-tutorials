# Snorkel Tutorials
A collection of tutorials for Snorkel

## Adding a new tutorial

1. Create a new top-level directory (e.g. `my_tutorial_dir`)
2. Add a `requirements.txt` to your directory if additional ones are needed
3. Write your tutorial as a Python script (e.g. `my_tutorial_dir/my_tutorial.py`) in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc)
4. Run `tox -e sync -- my_tutorial_dir` to generate a notebook version. Run this command to update when changes are made to the tutorial script.
5. Add an entry `[testenv:my_tutorial]` in `tox.ini` by copying `[testenv:example]` and changing the paths
6. Run `tox -e my_tutorial` to test out your tutorial
