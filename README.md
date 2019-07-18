# Snorkel Tutorials
A collection of tutorials for Snorkel

## Adding a new tutorial

0. Install your dev environment with `tox -e dev` then `source .env/bin/activate`
1. Create a new top-level directory
2. Add a `requirements.txt` to your directory if additional ones are needed
3. Write your tutorial in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc)
4. Run `jupytext --set-formats ipynb,py my_tutorial.py` on your tutorial script, then `jupytext --sync --execute my_tutorial.ipynb` to create a notebook version
5. If you make updates to the tutorial script, run `jupytext --sync --execute my_tutorial.ipynb` to update the notebook
6. Add an entry `[testenv:my_tutorial]` in `tox.ini` by copying `[testenv:example]` and changing the paths
7. Run `tox -e my_tutorial` to test out your tutorial
