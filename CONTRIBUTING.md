# Contributing to Snorkel Tutorials

We love contributors, so first and foremost, thank you!
We're actively working on our contributing guidelines, so this document is subject to change.
First things first: we adhere to the
[Contributor Covenant Code of Conduct](http://contributor-covenant.org/version/1/4/),
so please read through it before contributing.

### Contents

* [Types of Tutorials](#types-of-tutorials)
* [Dev Setup](#dev-setup)
* [Making Changes to an Existing Tutorial](#making-changes-to-an-existing-tutorial)
* [Adding a New Tutorial](#adding-a-new-tutorial)
* [Testing Changes Locally](#testing-changes-locally)
* [Previewing Changes to the Website](#previewing-changes-to-the-website)


## Types of Tutorials

Currently, we have notebook-based tutorials and script-based tutorials.
Both types are referenced in this guide.
Notebook-based tutorials act as walkthroughs of concepts
(the [`spam` tutorial](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam) is a good example).
Script-based tutorials act more as examples of how to deploy certain Snorkel functionality
(the [`drybell` tutorial](https://github.com/snorkel-team/snorkel-tutorials/tree/master/drybell) is a good example).
We default to using notebook-based tutorials.

## Dev Setup

For dev setup, you  will need to install [`tox`](https://tox.readthedocs.io), and set up a virtualenv with all the requirements.
For example, if you use `pip`, and want to work on the `spam` tutorial:

```bash
python3 -m pip install -U 'tox>=3.13.0,<4.0.0'
python3 -m pip install --upgrade virtualenv
virtualenv -p python3 .env
source .env/bin/activate

python3 -m pip install -r requirements.txt
python3 -m pip install -r spam/requirements.txt  # Change based on tutorial.
```

Start jupyter from the virtualenv to make sure the kernel has all the required dependencies.

## Making Changes to an Existing Tutorial

First, we recommend [posting an issue](https://github.com/snorkel-team/snorkel-tutorials/issues/new)
describing the improvement or fix you want to make.
Once you've worked out details with the maintainers, follow these general steps:

1. Make your changes to the source files
    * For notebook-based tutorials, we recommend making changes to the `.py` version
    then syncing changes with `tox -e my_tutorial_dir -- sync`.
    Alternatively, if you have already run all the cells in your browser, you can select
    `File` &rarr; `Jupytext` &rarr; `Pair Notebook with percent Script` to save the
    outputs directly to the notebook version.
    After saving, unpair the notebook with
    `File` &rarr; `Jupytext` &rarr; `Unpair notebook` so jupyter does not
    keep updating the notebook when all cells haven't been run.
    * For script-based tutorials, just make the changes as you normally would.
1. [Test your changes locally](#testing-changes-locally)
1. Submit a PR!

## Adding a New Tutorial

Before adding a new tutorial, we recommend posting a proposal to the
[Snorkel community forum on Spectrum](https://spectrum.chat/snorkel/tutorials?tab=posts).
Once you've worked out details with the maintainers, follow these general steps:

1. Create a new top-level directory (e.g. `my_tutorial_dir`)
1. Add a tutorial configuration file to your tutorial directory
    * For notebook-based tutorials, add a file called `.notebooks` to your tutorial directory
    and add the base name of each tutorial script/notebook pair (e.g. `my_tutorial`) as a
    separate line in `.notebooks`.
    See `spam` for an [example](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/.notebooks).
    * For script-based tutorials, add a file called `.scripts` to your tutorial directory
    and add the file name of each tutorial script (e.g. `my_tutorial.py`) as a
    separate line in `.scripts`.
    See `drybell` for an [example](https://github.com/snorkel-team/snorkel-tutorials/blob/master/drybell/.scripts).
1. Add a `requirements.txt` to your directory if additional ones are needed
1. Add a command to `[testenv]` in `tox.ini` by copying `spam` and add the `requirements.txt` file if necessary.
Also add the command name to `envlist`.
1. Write your tutorial!
    * For notebook-based tutorials, write your tutorial either as a Python script (e.g. `my_tutorial_dir/my_tutorial.py`) in [Jupytext percent format](https://gist.github.com/mwouts/91f3e1262871cdaa6d35394cd14f9bdc) or a Jupyter notebook.
        * Run `tox -e my_tutorial_dir -- sync` to generate a notebook version from the Python script version
          (or if you have run all cells, you can select
          `File` &rarr; `Jupytext` &rarr; `Pair Notebook with percent Script` to
          save the outputs directly to the notebook version, and then unpair it
          with `File` &rarr; `Jupytext` &rarr; `Unpair notebook` so jupyter does not
          keep updating the notebook when all cells haven't been run).
          Do this to update the notebook whenever changes are made to the tutorial script.
        * Run `tox -e my_tutorial_dir -- sync --py` to generate a Python script version from the notebook version. Run this command to update when changes are made to the tutorial notebook.
    * For script-based tutorials, write your tutoral as a Python script.
1. [Test your changes locally](#testing-changes-locally)
1. Submit a PR! Make sure to include a reference to the Spectrum planning thread.

## Testing Changes Locally

### Testing changes to tutorials

You can test changes to a specific tutorial by running `tox -e my_tutorial` where `my_tutoral` is
replaced by the corresponding environment name in `tox.ini`.
For scripts and notebooks, this will check that they execute without erroring.
For notebooks only, this will also check that any URLs in Markdown cells are reachable and that
the `.ipynb` versions match the `.py` versions.
Travis will also always run `tox -e style` to check code style and formatting, so you sould always
run this locally as well.
Running `tox` on its own will test all tutorials, which can be **extremely slow** since some tutorials
(like `recsys`) take a long time to run with a full dataset.

### Other `tox`-related commands

* To fix code formatting issues, run `tox -e fix`.
* You might need to update packages for an environmnet (for example, if you update `requirements.txt`
dependencies). Use the `-r` command for this. For example, you can run `tox -e spam -r -- sync` to
rebuild the `spam` environment and then run the `sync` script.
* Travis uses the `get_tox_envs.py` script to figure out which `tox` environments it needs to run to
test a PR. Once you've commited your changes, you can preview the environments that Travis will execute
by running `python3 scripts/get_tox_envs.py --plan`.

## Previewing Changes to the Website

All of the tutorials listed in `.web.yml` are rendered on our [website](https://snorkel.org/use-cases/).
Details on configuring with `.web.yml` are in that file.
In order to display tutorials as webpages, we convert the `.ipynb` versions to Markdown.
To generate all Markdown files listed in `.web.yml`, use `tox -e markdown`.
This will generate files in the `build` directory.

You can prevent cells from being rendered in Markdown by adding `{"tag": ["md-exclude"]}`
to the cell header in the `.py` file.
This is useful for confusing setup cells or cells with difficult-to-render outputs.
For example:

```python
# %% {"tag": ["md-exclude"]}
command.do_not_show()
this_line.will_not_appear()
```

You can also prevent cells from rendering output in Markdown by adding
`{"tag": ["md-exclude-output"]}` to the cell header in the `.py` file.
This is useful for cells that display warning messages and other confusing stuff.
For example:

```python
# %% {"tag": ["md-exclude-output"]}
command.will_show()
this_line.will_appear()
print(my_object)  # The output will not show
```
