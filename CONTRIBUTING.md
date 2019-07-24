## Adding a new tutorial

See the `example` directory for an example.

1. Create a new top-level directory (e.g. `my_tutorial_dir`)
2. Add a file called `.notebooks` to your tutorial directory and add the base name of each tutorial script/notebook pair (e.g. `my_tutorial`) as a separate line in `.notebooks`
3. Add a `requirements.txt` to your directory if additional ones are needed
4. Add a command to `[testenv]` in `tox.ini` by copying `example` and add the `requirements.txt` file if necessary. Also add the command name to `envlist`.
5. Write your tutorial either as a Python script (e.g. `my_tutorial_dir/my_tutorial.py`) in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc) or a Jupyter notebook
    * Run `tox -e my_tutorial_dir -- sync` to generate a notebook version from the Python script version. Run this command to update when changes are made to the tutorial script.
    * Run `tox -e my_tutorial_dir -- sync --py` to generate a Python script version from the notebook version. Run this command to update when changes are made to the tutorial notebook.
6. Run `tox -e my_tutorial_dir` to test out your tutorial
