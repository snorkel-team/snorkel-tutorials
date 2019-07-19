# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: snorkel_tutorials_env
#     language: python
#     name: snorkel_tutorials_env
# ---

# %% [markdown]
# # Introductory Snorkel Tutorial: Spam Detection

# %% [markdown]
# * Nice introductory text
# * Purpose of this tutorial
# * Steps:
#     1. Load data
#     2. Write labeling functions (LFs)
#     3. Combine with Label Model
#     4. Predict with Classifier

# %% [markdown]
# ### Task: Spam Detection

# %% [markdown]
# * Here's what we're trying to do
# * Here's where the data came from
# * Show sample T and F in markdown

# %% [markdown]
# ### Data Splits in Snorkel

# %% [markdown]
# * 4 splits: train, dev, valid, test
# * train is large and unlabeled
# * valid/test is labeled and you don't look at it 
# * best to come up with LFs while looking at data. Options:
#     * look at train for ideas; no labels, but np.
#     * label small subset of train (e.g., 200), call it "dev"
#     * in a pinch, use valid set as dev (note though that valid will no longer be good rep of test)

# %% [markdown]
# ## 1. Load data

# %% [markdown]
# * Start by loading data
# * utility pulls from internet, re-splits, and shuffles
# * for this application, train is videos 1-4, valid/test are video 5

# %%
from utils import load_spam_dataset

df_train, df_dev, df_valid, df_test = load_spam_dataset()


# %% [markdown]
# * Describe fields

# %%
df_train.sample(5)

# %% [markdown]
# ## 2. Write Labeling Functions (LFs)

# %% [markdown]
# * What's an LF
#     * Why are they awesome
# * Can be many types: 
#     * keyword
#     * pattern-match
#     * heuristic
#     * third-party models
#     * distant supervision
#     * crowdworkers (non-expert)

# %% [markdown]
# * Look at 10 examples; got any ideas? 

# %%
from snorkel.labeling.lf import labeling_function

lfs = []

@labeling_function()
def tbd(x):
    pass

# %% [markdown]
# * Apply it with LFApplier
# * Score with lf_summary()
#     * What each column means
# * Now let's make lots more!

# %% [markdown]
# ### i. Keyword LFs

# %% [markdown]
# * Keywords

# %% [markdown]
# ### ii. Pattern-matching LFs

# %% [markdown]
# * Regexes

# %% [markdown]
# ### iii.  Heuristic LFs

# %% [markdown]
# * Length, early comma, etc.
# * SpaCy (preprocessor)

# %% [markdown]
# ### iv. Third-party Model LFs

# %% [markdown]
# * Sentiment classifier (preprocessor)

# %% [markdown]
# ### v. Write your own LFs

# %% [markdown]
# * Make a stub

# %% [markdown]
# ## 3. Combine with Label Model

# %% [markdown]
# * Pretty much copy prose from Spouse tutorial

# %% [markdown]
# * Run LabelModel, get probabilities
# * Look at probabilities (histogram)
# * What if we used this directly as a classifier? (score)
#     * Why we expect classifier we train to generalize better
#     * Look - we're randomly guessing on XX% of the data

# %% [markdown]
# * Can also compare to MV
#     * Does worse

# %% [markdown]
# ## 4. Predict with Classifier

# %% [markdown]
# * Now train classifier
#     * Can use any third-party classifier (plug into your existing pipelines!)
#     * Some libraries natively support probabilistic labels (us, TF); for others, can round.
# * Use bag-of-ngrams as features
# * [Train TF logreg w/ soft labels]
# * Score; see, we do better!
# * Also demonstrate sklearn logreg with hard labels (end model agnostic)
# * Compare with training on dev directly (see, we did better)
#     * And we could do even better with more raw unlabeled data

# %%
