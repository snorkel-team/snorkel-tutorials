## Adding a new tutorial

1. Create a new top-level directory (e.g. `my_tutorial_dir`)
2. Add a file called `.notebooks` to your tutorial directory and add the base name of each tutorial script/notebook pair (e.g. `my_tutorial`) as a separate line in `.notebooks`
3. Add a `requirements.txt` to your directory if additional ones are needed
4. Add a command to `[testenv]` in `tox.ini` by copying `spam` and add the `requirements.txt` file if necessary. Also add the command name to `envlist`.
5. Write your tutorial either as a Python script (e.g. `my_tutorial_dir/my_tutorial.py`) in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc) or a Jupyter notebook
    * Run `tox -e my_tutorial_dir -- sync` to generate a notebook version from the Python script version. Run this command to update when changes are made to the tutorial script.
    * Run `tox -e my_tutorial_dir -- sync --py` to generate a Python script version from the notebook version. Run this command to update when changes are made to the tutorial notebook.
6. Run `tox -e my_tutorial_dir` to test out your tutorial. This is also the test that Travis CI will use to check compliance before merges.

## Rendering tutorials as Markdown

In order to display tutorials as webpages, we convert the `.ipynb` versions to Markdown.
To generate all Markdown files listed in `.web.yml`, use

```bash
tox -e markdown
```

Again, this will generate files in the `build` directory.
Details on configuring with `.web.yml` are in that file.

Additionally, you can prevent cells from being rendered in Markdown by adding `{"tag": ["md-exclude"]}`
to the cell header in the `.py` file.
For example:

```python
# %% {"tag": ["md-exclude"]}
command.do_not_show()
this_line.will_not_appear()
```

You can also prevent cells from rendering output in Markdown by adding
`{"tag": ["md-exclude-output"]}` to the cell header in the `.py` file.
For example:

```python
# %% {"tag": ["md-exclude-output"]}
command.will_show()
this_line.will_appear()
print(my_object)  # The output will not show
```
