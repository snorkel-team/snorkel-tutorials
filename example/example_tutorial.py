# %% [markdown]
# ## Let's write some LFs!
# This should be fun!

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
print("LF 1 output:", lf_1(x))
print("LF 2 output:", lf_2(x))

# %%
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(list(range(10)))
