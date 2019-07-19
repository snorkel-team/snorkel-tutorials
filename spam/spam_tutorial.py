# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: snorkel
#     language: python
#     name: snorkel
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
# * Typically an iterative process
#     * Look at examples for ideas
#     * Write an LF
#     * Check performance on dev set
#     * Balance accuracy/coverage

# %% [markdown]
# ### a) Look at examples for ideas

# %% [markdown]
# * Look at 10 examples; got any ideas?

# %%
# Don't truncate text fields in the display
pd.set_option('display.max_colwidth', 0)  

# Display just the text and label
df_dev[["CONTENT", "LABEL"]].sample(10, random_state=123)

# %%
# for i, x in df_dev.iterrows():
#     if "please" in x.CONTENT:
#         print(x.CONTENT)

# %% [markdown]
# ### b) Write an LF

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
def keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if 'my' in x.CONTENT.lower() else ABSTAIN

lfs.append(keyword_my)

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
# ### c) Check performance on dev set

# %% [markdown]
# We can easily calculate the coverage of this LF by hand (i.e., the percentage of the dataset that it labels) as follows:

# %%
coverage = L_train.nnz / L_train.shape[0]
print(f"Coverage: {coverage}")

# %% [markdown]
# To get an estimate of its accuracy, we can label the development set with it and compare that to the few gold labels we do have.
#
# Note that we don't want to penalize the LF for examples where it abstained, so we filter out both the predictions and the gold labels where the prediction is `ABSTAIN`.

# %%
L_dev = applier.apply(df_dev)
L_dev_array = np.asarray(L_dev.todense()).squeeze()

Y_dev_array = df_dev["LABEL"].values

accuracy = ((L_dev_array == Y_dev_array)[L_dev_array != ABSTAIN]).sum() / (L_dev_array != ABSTAIN).sum()
print(f"Accuracy: {accuracy}")

# %% [markdown]
# Alternatively, you can use the provided `metric_score()` helper method, which allows you to specify a metric to calculate and certain classes to ignore (such as ABSTAIN).

# %%
from snorkel.analysis.metrics import metric_score

# Calculate accuracy, ignore all examples for which the predicted label is ABSTAIN
accuracy = metric_score(golds=Y_dev_array, preds=L_dev_array, metric="accuracy", filter_dict={"preds": [ABSTAIN]})
print(f"Accuracy: {accuracy}")

# %% [markdown]
# You can also use the helper method `lf_summary()` to report the following summary statistics for multiple LFs at once:
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
# ### d) Balance accuracy/coverage

# %% [markdown]
# Often, by looking at the examples that an LF does and doesn't label, we can get ideas for how to improve it.
#
# The helper method `error_buckets()` groups examples by their predicted label and true label, so `buckets[(1, 2)]` will contain the indices of examples that that the LF labeled 1 (SPAM) that were actually of class 2 (HAM).

# %%
from snorkel.analysis.error_analysis import error_buckets

buckets = error_buckets(Y_dev_array, L_dev_array)
df_dev[["CONTENT", "LABEL"]].iloc[buckets[(1, 2)]].head()

# %% [markdown]
# On the other hand, `buckets[(1, 1)]` contains SPAM examples it labeled correctly.

# %%
df_dev[["CONTENT", "LABEL"]].iloc[buckets[(1, 1)]].head()


# %% [markdown]
# Looking at these examples, we may notice that much of the time when "my" is used, it's referring to "my channel". We can update our LF to see how making this change affects accuracy and coverage.

# %%
@labeling_function()
def keywords_my_channel(x):
    return SPAM if 'my channel' in x.CONTENT.lower() else ABSTAIN

lfs = [keywords_my_channel]
applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
lf_names= [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev_array)

# %% [markdown]
# In this case, accuracy does improve a bit, but it was already fairly accurate to begin with, and "tightening" the LF like this causes the coverage drops significantly, so we'll stick with the original LF.

# %% [markdown]
# ## More Labeling Functions

# %% [markdown]
# If a single LF had high enough coverage to label our entire test dataset accurately, then we wouldn't need a classifier at all; we could just use that single simple heuristic to complete the task. But most problems are not that simple. Instead, we usually need to **combine multiple LFs** to label our dataset, both to increase the size of the generated training set (since we can't generate training labels for data points that all LFs abstained on) and to improve the overall accuracy of the training labels we generate by factoring in multiple different signals.
#
# In the following subsections, we'll show just a few of the many types of LFs that you could write to generate a training dataset for this problem.

# %% [markdown]
# ### i. Keyword LFs

# %% [markdown]
# For text applications, some of the simplest LFs to write are often just keyword lookups.

# %%
lfs = []

@labeling_function()
def keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if 'my' in x.CONTENT.lower() else ABSTAIN
lfs.append(keyword_my)

@labeling_function()
def lf_subscribe(x):
    """Spammers ask users to subscribe to their channels."""
    return SPAM if "subscribe" in x.CONTENT else 0
lfs.append(lf_subscribe)

@labeling_function()
def lf_link(x):
    """Spammers post links to their channels."""
    return SPAM if "http" in x.CONTENT.lower() else 0
lfs.append(lf_link)

@labeling_function()
def lf_please(x):
    """Spammers make requests rather than commenting."""
    return SPAM if any([word in x.CONTENT.lower() for word in ["please", "plz"]]) else ABSTAIN
lfs.append(lf_please)

@labeling_function()
def lf_song(x):
    """Hammers actually talk about the video's content."""
    return HAM if "song" in x.CONTENT.lower() else ABSTAIN
lfs.append(lf_song)

# %% [markdown]
# ### ii. Pattern-matching LFs (Regular Expressions)

# %% [markdown]
# If we want a little more control over a keyword search, we can look for regular expressions instead.

# %%
import re

@labeling_function()
def regex_check_out(x):
    """Catch 'check out my video' as well as 'check it out', for example."""
    return SPAM if re.search(r"check.*out", x.CONTENT, flags=re.I) else ABSTAIN

lfs.append(regex_check_out)


# %% [markdown]
# ### iii.  Heuristic LFs

# %% [markdown]
# There may other heuristics or "rules of thumb" that you come up with as you look at the data.
# So long as you can express it in a function, it's a viable LF!

# %%
@labeling_function()
def short_comment(x):
    """Hammer comments are often short, such as 'cool video!'"""
    return HAM if len(x.CONTENT.split()) < 5 else ABSTAIN
lfs.append(short_comment)


@labeling_function()
def short_word_lengths(x):
    """Ham comments tend to have shorter words"""
    words = x.CONTENT.split()
    lengths = [len(word) for word in words]
    mean_word_length = sum(lengths) / len(lengths)
    return HAM if mean_word_length < 4 else ABSTAIN
lfs.append(short_word_lengths)

# %% [markdown]
# ### Adding Preprocessors

# %% [markdown]
# Some LFs rely on fields that aren't present in the raw data, but can be derived from it. We can enrich our data (providing more fields for the LFs to refer to) using `Preprocessors`.
#
# For example, we can use the fantastic NLP tool [spaCy](https://spacy.io/) to add lemmas, part-of-speech (pos) tags, etc. to each token.

# %%
# Download the spacy english model
# ! python -m spacy download en_core_web_sm

# %%
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor
# The SpacyPreprocessor parses the text in text_field and 
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="CONTENT", doc_field="doc", memoize=True)

@labeling_function(preprocessors=[spacy])
def no_names(x):
    """"""
    num_names = sum([token.pos_ == "PROPN" for token in x.doc])
    return HAM if num_names == 0 else ABSTAIN
lfs.append(no_names)

# %% [markdown]
# ### iv. Third-party Model LFs

# %% [markdown]
# We can also utilize other models, including ones trained for other tasks that are related to, but not the same as, the one we care about.
#
# For example, the [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) tool provides a pretrained sentiment analyzer. Our spam classification task is not the same as sentiment classification, but it turns out that SPAM and HAM comments have different distributions of sentiment scores, with HAM having more positive/subjective sentiments.

# %%
import matplotlib.pyplot as plt
from textblob import TextBlob

spam_polarities = [TextBlob(x.CONTENT).sentiment.polarity for i, x in df_dev.iterrows() if x.LABEL == SPAM]
ham_polarities = [TextBlob(x.CONTENT).sentiment.polarity for i, x in df_dev.iterrows() if x.LABEL == HAM]

_ = plt.hist([spam_polarities, ham_polarities])

# %%
from textblob import TextBlob

@labeling_function()
def textblob_polarity(x):
    return 2 if TextBlob(x.CONTENT).sentiment.polarity > 0.3 else 0
lfs.append(textblob_polarity)

@labeling_function()
def textblob_subjectivity(x):
    return 2 if TextBlob(x.CONTENT).sentiment.subjectivity > 0.9 else 0
lfs.append(textblob_subjectivity)

# %% [markdown]
# ### v. Write your own LFs

# %% [markdown]
# This tutorial demonstrates just a handful of the types of LFs that one might write for this task. 
# The strength of LFs is that they provide a flexible abstraction for conveying a huge variety of supervision signals. 
#
# You can uncomment the cell below to write one or more of your own LFs.
# Don't forget to add them to the list of `lfs` so that they are included by the `LFApplier` in the next section.

# %%
# @labeling_function()
# def my_lf(x):
#     pass
# lfs.append(my_lf)

# %% [markdown]
# ### Apply LFs

# %% [markdown]
# With our full set of LFs (including any you wrote), we can now apply these once again with `LFApplier` to get our the label matrices for the `train` and `dev` splits. We'll use the `train` split's label matrix to generate training labels with the Label Model. The `dev` split's label model is primarily helpful for looking at summary statistics.
#
# Note that the `pandas` format provides an easy interface that many practioners are familiar with, but it is also less optimized for scale. For larger datasets, more compute-intensive LFs, or larger LF sets, you may decide to use one of the other supported data formats such as `dask` or `spark` dataframes, and their corresponding applier objects.

# %%
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

lf_names= [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev_array, lf_names=lf_names)

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
