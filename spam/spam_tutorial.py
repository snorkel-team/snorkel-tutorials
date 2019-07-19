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
# * Purpose of this tutorial...
# * Steps:
#     1. Load data
#     2. Write labeling functions (LFs)
#     3. Combine with Label Model
#     4. Predict with Classifier

# %% [markdown]
# ### Task: Spam Detection

# %% [markdown]
# * Here's what we're trying to do
# * Here's where the data came from (cite properly)
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
df_train.sample(5, random_state=1)

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
# Don't truncate text fields in the display
pd.set_option('display.max_colwidth', 0)  

# Display just the text and label
df_dev[["CONTENT", "LABEL"]].sample(10, random_state=123)

# %% [markdown]
# The simplest way to create labeling functions in Snorkel is with the `@labeling_function()` decorator, which wraps a function for evaluating on a single `DataPoint` (in this case, a row of the dataframe).
#
# Looking at samples of our data, we see multiple messages where spammers are trying to get viewers to look at "my channel" or "my video," so we write a simple LF that labels an example as spam if it includes the word "my".

# %%
from snorkel.labeling.lf import labeling_function

# For clarity, we'll define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = 0
SPAM = 1
HAM = 2

# We initialize an empty list that we'll add our LFs to as we create them
lfs = []

@labeling_function()
def keywords_my(x):
    return SPAM if 'my' in x.CONTENT.lower() else ABSTAIN

lfs.append(keywords_my)

# %% [markdown]
# To apply one or more LFs that we've written to a collection of `DataPoints`, we use an `LFApplier`.
#
# Because our `DataPoints` are represented with a Pandas dataframe in this tutorial, we use the `PandasLFApplier` class.

# %%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# %% [markdown]
# The output of the `apply()` method is a sparse label matrix which we generally refer to as `L`.

# %%
L_train

# %% [markdown]
# We can easily calculate the coverage of this LF (i.e., the percentage of the dataset that it labels) as follows:

# %%
coverage = L_train.nnz / L_train.shape[0]
print(f"Coverage: {coverage}")

# %% [markdown]
# To get an estimate of its accuracy, we can label the development set with it and compare that to the few gold labels we do have.

# %%
L_dev = applier.apply(df_dev)

# Note that we don't want to penalize the LF for examples where it abstained, 
# so we filter out both the predictions and the gold labels where the prediction
# is ABSTAIN
L_dev_array = np.asarray(L_dev.todense()).squeeze()
Y_dev_array = df_dev["LABEL"].values
accuracy = ((L_dev_array == Y_dev_array)[L_dev_array != ABSTAIN]).sum() / (L_dev_array != ABSTAIN).sum()
print(f"Accuracy: {accuracy}")

# %% [markdown]
# Alternatively, you can use the helper method `lf_summary` to report the following summary statistics:
# * Polarity: The set of labels this LF outputs
# * Coverage: The fraction of the dataset the LF labels
# * Overlaps: The fraction of the dataset where this LF and at least one other LF label
# * Conflicts: The fraction of the dataset where this LF and at least one other LF label and disagree
# * Correct: The number of `DataPoints` this LF labels correctly (if gold labels are provided)
# * Incorrect: The number of `DataPoints` this LF labels incorrectly (if gold labels are provided)
# * Emp. Acc.: The empirical accuracy of this LF (if gold labels are provided)

# %%
from snorkel.labeling.analysis import lf_summary

lf_names= [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev_array, lf_names=lf_names)

# %% [markdown]
# This LF is fairly accurate, but it only labels a fraction of the dataset.
# If we want to do well on our test set, we'll need more LFs.
#
# In the following subsections, we'll show just a few of the many types of LFs that you could write to generate a training dataset for this problem.

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
#     * Note: no labels are required or used
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
