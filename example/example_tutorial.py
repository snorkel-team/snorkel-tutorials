# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
# ---

# %% [markdown]
# ## Let's write some LFs!
# This should be fun

# %%
from snorkel.labeling.lf import labeling_function


@labeling_function()
def lf_1(x):
    return 1 if x.n_failures > 10 else 0


@labeling_function()
def lf_2(x):
    return 1 if x.n_successes < 2 else 0


# %%
from types import SimpleNamespace

x = SimpleNamespace(n_failures=8, n_successes=1)
assert lf_1(x) == 0
assert lf_2(x) == 1
