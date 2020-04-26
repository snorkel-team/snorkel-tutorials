# -*- coding: utf-8 -*-
# %% [markdown]
# # 🚀 Snorkel Intro Tutorial: Data Labeling

# %% [markdown]
# In this tutorial, we will walk through the process of using Snorkel to build a training set for classifying YouTube comments as spam or not spam.
# The goal of this tutorial is to illustrate the basic components and concepts of Snorkel in a simple way, but also to dive into the actual process of iteratively developing real applications in Snorkel.
#
# **Note that this is a toy dataset that helps highlight the different features of Snorkel. For examples of high-performance, real-world uses of Snorkel, see [our publications list](https://www.snorkel.org/resources/).**
#
# Additionally:
# * For an overview of Snorkel, visit [snorkel.org](https://snorkel.org)
# * You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/)
#
# Our goal is to train a classifier over the comment data that can predict whether a comment is spam or not spam.
# We have access to a large amount of *unlabeled data* in the form of YouTube comments with some metadata.
# In order to train a classifier, we need to label our data, but doing so by hand for real world applications can often be prohibitively slow and expensive.
#
# In these cases, we can turn to a _weak supervision_ approach, using **_labeling functions (LFs)_** in Snorkel: noisy, programmatic rules and heuristics that assign labels to unlabeled training data.
# We'll dive into the Snorkel API and how we write labeling functions later in this tutorial, but as an example,
# we can write an LF that labels data points with `"http"` in the comment text as spam since many spam
# comments contain links:
#
# ```python
# from snorkel.labeling import labeling_function
#
# @labeling_function()
# def lf_contains_link(x):
#     # Return a label of SPAM if "http" in comment text, otherwise ABSTAIN
#     return SPAM if "http" in x.text.lower() else ABSTAIN
# ```
#
# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/), originally introduced in ["TubeSpam: Comment Spam Filtering on YouTube"](https://ieeexplore.ieee.org/document/7424299/), ICMLA'15 (T.C. Alberto, J.V. Lochter, J.V. Almeida).
#
# 2. **Writing Labeling Functions**: We write Python programs that take as input a data point and assign labels (or abstain) using heuristics, pattern matching, and third-party models.
#
# 3. **Combining Labeling Function Outputs with the Label Model**: We model the outputs of the labeling functions over the training set using a novel, theoretically-grounded [modeling approach](https://arxiv.org/abs/1605.07723), which estimates the accuracies and correlations of the labeling functions using only their agreements and disagreements, and then uses this to reweight and combine their outputs, which we then use as _probabilistic_ training labels.
#
# 4. **Training a Classifier**: We train a classifier that can predict labels for *any* YouTube comment (not just the ones labeled by the labeling functions) using the probabilistic training labels from step 3.

# %% [markdown]
# ### Task: Spam Detection

# %% [markdown]
# We use a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/) that consists of YouTube comments from 5 videos. The task is to classify each comment as being
#
# * **`HAM`**: comments relevant to the video (even very simple ones), or
# * **`SPAM`**: irrelevant (often trying to advertise something) or inappropriate messages
#
# For example, the following comments are `SPAM`:
#
#         "Subscribe to me for free Android games, apps.."
#
#         "Please check out my vidios"
#
#         "Subscribe to me and I'll subscribe back!!!"
#
# and these are `HAM`:
#
#         "3:46 so cute!"
#
#         "This looks so fun and it's a good song"
#
#         "This is a weird video."

# %% [markdown]
# ### Data Splits in Snorkel
#
# We split our data into two sets:
# * **Training Set**: The largest split of the dataset, and the one without any ground truth ("gold") labels.
# We will generate labels for these data points with weak supervision.
# * **Test Set**: A small, standard held-out blind hand-labeled set for final evaluation of our classifier. This set should only be used for final evaluation, _not_ error analysis.
#
# Note that in more advanced production settings, we will often further split up the available hand-labeled data into a _development split_, for getting ideas to write labeling functions, and a _validation split_ for e.g. checking our performance without looking at test set scores, hyperparameter tuning, etc.  These splits are used in some of the other advanced tutorials, but omitted for simplicity here.

# %% [markdown]
# ## 1. Loading Data

# %% [markdown]
# We load the YouTube comments dataset and create Pandas DataFrame objects for the train and test sets.
# DataFrames are extremely popular in Python data analysis workloads, and Snorkel provides native support
# for several DataFrame-like data structures, including Pandas, Dask, and PySpark.
# For more information on working with Pandas DataFrames, see the [Pandas DataFrame guide](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html).
#
# Each DataFrame consists of the following fields:
# * **`author`**: Username of the comment author
# * **`data`**: Date and time the comment was posted
# * **`text`**: Raw text content of the comment
# * **`label`**: Whether the comment is `SPAM` (1), `HAM` (0), or `UNKNOWN/ABSTAIN` (-1)
# * **`video`**: Video the comment is associated with
#
# We start by loading our data.
# The `load_spam_dataset()` method downloads the raw CSV files from the internet, divides them into splits, converts them into DataFrames, and shuffles them.
# As mentioned above, the dataset contains comments from 5 of the most popular YouTube videos during a period between 2014 and 2015.
# * The first four videos' comments are combined to form the `train` set. This set has no gold labels.
# * The fifth video is part of the `test` set.

# %% [markdown] {"tags": ["md-exclude"]}
# This next cell takes care of some notebook-specific housekeeping.
# You can ignore it.

# %% {"tags": ["md-exclude"]}
# %matplotlib inline

import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# Turn off TensorFlow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For reproducibility
os.environ["PYTHONHASHSEED"] = "0"

# %% [markdown] {"tags": ["md-exclude"]}
# If you want to display all comment text untruncated, change `DISPLAY_ALL_TEXT` to `True` below.

# %% {"tags": ["md-exclude"]}
import pandas as pd


DISPLAY_ALL_TEXT = False

pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

# %% [markdown] {"tags": ["md-exclude"]}
# This next cell makes sure a spaCy English model is downloaded.
# If this is your first time downloading this model, restart the kernel after executing the next cell.

# %% {"tags": ["md-exclude"]}
# Download the spaCy english model
# ! python -m spacy download en_core_web_sm

# %%
from utils import load_spam_dataset

df_train, df_test = load_spam_dataset()

# We pull out the label vectors for ease of use later
Y_test = df_test.label.values


# %% [markdown]
# The class distribution varies slightly between `SPAM` and `HAM`, but they're approximately class-balanced.

# %%
# For clarity, we define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = -1
HAM = 0
SPAM = 1

# %% [markdown]
# ## 2. Writing Labeling Functions (LFs)

# %% [markdown]
# ### A gentle introduction to LFs

# %% [markdown]
# **Labeling functions (LFs) help users encode domain knowledge and other supervision sources programmatically.**
#
# LFs are heuristics that take as input a data point and either assign a label to it (in this case, `HAM` or `SPAM`) or abstain (don't assign any label). Labeling functions can be *noisy*: they don't have perfect accuracy and don't have to label every data point.
# Moreover, different labeling functions can overlap (label the same data point) and even conflict (assign different labels to the same data point). This is expected, and we demonstrate how we deal with this later.
#
# Because their only requirement is that they map a data point a label (or abstain), they can wrap a wide variety of forms of supervision. Examples include, but are not limited to:
# * *Keyword searches*: looking for specific words in a sentence
# * *Pattern matching*: looking for specific syntactical patterns
# * *Third-party models*: using an pre-trained model (usually a model for a different task than the one at hand)
# * *Distant supervision*: using external knowledge base
# * *Crowdworker labels*: treating each crowdworker as a black-box function that assigns labels to subsets of the data

# %% [markdown]
# ### Recommended practice for LF development

# %% [markdown]
# Typical LF development cycles include multiple iterations of ideation, refining, evaluation, and debugging.
# A typical cycle consists of the following steps:
#
# 1. Look at examples to generate ideas for LFs
# 1. Write an initial version of an LF
# 1. Spot check its performance by looking at its output on data points in the training set (or development set if available)
# 1. Refine and debug to improve coverage or accuracy as necessary
#
# Our goal for LF development is to create a high quality set of training labels for our unlabeled dataset,
# not to label everything or directly create a model for inference using the LFs.
# The training labels are used to train a separate discriminative model (in this case, one which just uses the comment text) in order to generalize to new, unseen data points.
# Using this model, we can make predictions for data points that our LFs don't cover.
#
# We'll walk through the development of two LFs using basic analysis tools in Snorkel, then provide a full set of LFs that we developed for this tutorial.

# %% [markdown]
# ### a) Exploring the training set for initial ideas

# %% [markdown]
# We'll start by looking at 20 random data points from the `train` set to generate some ideas for LFs.

# %%
df_train[["author", "text", "video"]].sample(20, random_state=2)

# %% [markdown]
# One dominant pattern in the comments that look like spam (which we might know from prior domain experience, or from inspection of a few training data points) is the use of the phrase "check out" (e.g. "check out my channel").
# Let's start with that.

# %% [markdown]
# ### b) Writing an LF to identify spammy comments that use the phrase "check out"

# %% [markdown]
# Labeling functions in Snorkel are created with the
# [`@labeling_function` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.labeling_function.html).
# The [decorator](https://realpython.com/primer-on-python-decorators/) can be applied to _any Python function_ that returns a label for a single data point.
#
# Let's start developing an LF to catch instances of commenters trying to get people to "check out" their channel, video, or website.
# We'll start by just looking for the exact string `"check out"` in the text, and see how that compares to looking for just `"check"` in the text.
# For the two versions of our rule, we'll write a Python function over a single data point that express it, then add the decorator.

# %%
from snorkel.labeling import labeling_function


@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN


@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN


# %% [markdown]
# To apply one or more LFs that we've written to a collection of data points, we use an
# [`LFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.LFApplier.html).
# Because our data points are represented with a Pandas DataFrame in this tutorial, we use the
# [`PandasLFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.PandasLFApplier.html).
# Correspondingly, a single data point `x` that's passed into our LFs will be a [Pandas `Series` object](https://pandas.pydata.org/pandas-docs/stable/reference/series.html).
#
# It's important to note that these LFs will work for any object with an attribute named `text`, not just Pandas objects.
# Snorkel has several other appliers for different data point collection types which you can browse in the [API documentation](https://snorkel.readthedocs.io/en/master/packages/labeling.html).
#
# The output of the `apply(...)` method is a ***label matrix***, a fundamental concept in Snorkel.
# It's a NumPy array `L` with one column for each LF and one row for each data point, where `L[i, j]` is the label that the `j`th labeling function output for the `i`th data point.
# We'll create a label matrix for the `train` set.

# %% {"tags": ["md-exclude-output"]}
from snorkel.labeling import PandasLFApplier

lfs = [check_out, check]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

# %%
L_train

# %% [markdown]
# ### c) Evaluate performance on training set

# %% [markdown]
# We can easily calculate the coverage of these LFs (i.e., the percentage of the dataset that they label) as follows:

# %%
coverage_check_out, coverage_check = (L_train != ABSTAIN).mean(axis=0)
print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
print(f"check coverage: {coverage_check * 100:.1f}%")

# %% [markdown]
# Lots of statistics about labeling functions &mdash; like coverage &mdash; are useful when building any Snorkel application.
# So Snorkel provides tooling for common LF analyses using the
# [`LFAnalysis` utility](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.LFAnalysis.html).
# We report the following summary statistics for multiple LFs at once:
#
# * **Polarity**: The set of unique labels this LF outputs (excluding abstains)
# * **Coverage**: The fraction of the dataset the LF labels
# * **Overlaps**: The fraction of the dataset where this LF and at least one other LF label
# * **Conflicts**: The fraction of the dataset where this LF and at least one other LF label and disagree
# * **Correct**: The number of data points this LF labels correctly (if gold labels are provided)
# * **Incorrect**: The number of data points this LF labels incorrectly (if gold labels are provided)
# * **Empirical Accuracy**: The empirical accuracy of this LF (if gold labels are provided)
#
# For *Correct*, *Incorrect*, and *Empirical Accuracy*, we don't want to penalize the LF for data points where it abstained.
# We calculate these statistics only over those data points where the LF output a label.
# **Note that in our current setup, we can't compute these statistics because we don't have any ground-truth labels (other than in the test set, which we cannot look at). Not to worry—Snorkel's `LabelModel` will estimate them without needing any ground-truth labels in the next step!**

# %%
from snorkel.labeling import LFAnalysis

LFAnalysis(L=L_train, lfs=lfs).lf_summary()

# %% [markdown]
# We might want to pick the `check` rule, since `check` has higher coverage. Let's take a look at 10 random `train` set data points where `check` labeled `SPAM` to see if it matches our intuition or if we can identify some false positives.

# %%
df_train.iloc[L_train[:, 1] == SPAM].sample(10, random_state=1)

# %% [markdown]
# No clear false positives here, but many look like they could be labeled by `check_out` as well.
#
# Let's see 10 data points where `check_out` abstained, but `check` labeled. We can use the`get_label_buckets(...)` to group data points by their predicted label and/or true labels.

# %%
from snorkel.analysis import get_label_buckets

buckets = get_label_buckets(L_train[:, 0], L_train[:, 1])
df_train.iloc[buckets[(ABSTAIN, SPAM)]].sample(10, random_state=1)

# %% [markdown]
# Most of these seem like small modifications of "check out", like "check me out" or "check it out".
# Can we get the best of both worlds?

# %% [markdown]
# ### d) Balance accuracy and coverage

# %% [markdown]
# Let's see if we can use regular expressions to account for modifications of "check out" and get the coverage of `check` plus the accuracy of `check_out`.

# %%
import re


@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


# %% [markdown]
# Again, let's generate our label matrices and see how we do.

# %% {"tags": ["md-exclude-output"]}
lfs = [check_out, check, regex_check_out]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

# %%
LFAnalysis(L=L_train, lfs=lfs).lf_summary()

# %% [markdown]
# We've split the difference in `train` set coverage—this looks promising!
# Let's verify that we corrected our false positive from before.

# %% [markdown]
# To understand the coverage difference between `check` and `regex_check_out`, let's take a look at 10 data points from the `train` set.
# Remember: coverage isn't always good.
# Adding false positives will increase coverage.

# %%
buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
df_train.iloc[buckets[(SPAM, ABSTAIN)]].sample(10, random_state=1)

# %% [markdown]
# Most of these are SPAM, but a good number are false positives.
# **To keep precision high (while not sacrificing much in terms of coverage), we'd choose our regex-based rule.**

# %% [markdown]
# ### e) Writing an LF that uses a third-party model

# %% [markdown]
# The LF interface is extremely flexible, and can wrap existing models.
# A common technique is to use a commodity model trained for other tasks that are related to, but not the same as, the one we care about.
#
# For example, the [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) tool provides a pretrained sentiment analyzer. Our spam classification task is not the same as sentiment classification, but we may believe that `SPAM` and `HAM` comments have different distributions of sentiment scores.
# We'll focus on writing LFs for `HAM`, since we identified `SPAM` comments above.
#
# **A brief intro to `Preprocessor`s**
#
# A [Snorkel `Preprocessor`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.Preprocessor.html#snorkel.preprocess.Preprocessor)
# is constructed from a black-box Python function that maps a data point to a new data point.
# `LabelingFunction`s can use `Preprocessor`s, which lets us write LFs over transformed or enhanced data points.
# We add the [`@preprocessor(...)` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.preprocessor.html)
# to preprocessing functions to create `Preprocessor`s.
# `Preprocessor`s also have extra functionality, such as memoization
# (i.e. input/output caching, so it doesn't re-execute for each LF that uses it).
#
# We'll start by creating a `Preprocessor` that runs `TextBlob` on our comments, then extracts the polarity and subjectivity scores.

# %%
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


# %% [markdown]
# We can now pick a reasonable threshold and write a corresponding labeling function (note that it doesn't have to be perfect as the `LabelModel` will soon help us estimate each labeling function's accuracy and reweight their outputs accordingly):

# %%
@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN


# %% [markdown]
# Let's do the same for the subjectivity scores.
# This will run faster than the last cell, since we memoized the `Preprocessor` outputs.

# %%
@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN


# %% [markdown]
# Let's apply our LFs so we can analyze their performance.

# %% {"tags": ["md-exclude-output"]}
lfs = [textblob_polarity, textblob_subjectivity]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# %%
LFAnalysis(L_train, lfs).lf_summary()

# %% [markdown]
# **Again, these LFs aren't perfect—note that the `textblob_subjectivity` LF has fairly high coverage and could have a high rate of false positives. We'll rely on Snorkel's `LabelModel` to estimate the labeling function accuracies and reweight and combine their outputs accordingly.**

# %% [markdown]
# ## 3. Writing More Labeling Functions

# %% [markdown]
# If a single LF had high enough coverage to label our entire test dataset accurately, then we wouldn't need a classifier at all.
# We could just use that single simple heuristic to complete the task.
# But most problems are not that simple.
# Instead, we usually need to **combine multiple LFs** to label our dataset, both to increase the size of the generated training set (since we can't generate training labels for data points that no LF voted on) and to improve the overall accuracy of the training labels we generate by factoring in multiple different signals.
#
# In the following sections, we'll show just a few of the many types of LFs that you could write to generate a training dataset for this problem.

# %% [markdown]
# ### a) Keyword LFs

# %% [markdown]
# For text applications, some of the simplest LFs to write are often just keyword lookups.
# These will often follow the same execution pattern, so we can create a template and use the `resources` parameter to pass in LF-specific keywords.
# Similar to the [`labeling_function` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.labeling_function.html#snorkel.labeling.labeling_function),
# the [`LabelingFunction` class](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html#snorkel.labeling.LabelingFunction)
# wraps a Python function (the `f` parameter), and we can use the `resources` parameter to pass in keyword arguments (here, our keywords to lookup) to said function.

# %%
from snorkel.labeling import LabelingFunction


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=SPAM):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments post links to other channels."""
keyword_link = make_keyword_lf(keywords=["http"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song"], label=HAM)


# %% [markdown]
# ### b) Pattern-matching LFs (regular expressions)

# %% [markdown]
# If we want a little more control over a keyword search, we can look for regular expressions instead.
# The LF we developed above (`regex_check_out`) is an example of this.

# %% [markdown]
# ### c)  Heuristic LFs

# %% [markdown]
# There may other heuristics or "rules of thumb" that you come up with as you look at the data.
# So long as you can express it in a function, it's a viable LF!

# %%
@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN


# %% [markdown]
# ### d) LFs with Complex Preprocessors

# %% [markdown]
# Some LFs rely on fields that aren't present in the raw data, but can be derived from it.
# We can enrich our data (providing more fields for the LFs to refer to) using `Preprocessor`s.
#
# For example, we can use the fantastic NLP (natural language processing) tool [spaCy](https://spacy.io/) to add lemmas, part-of-speech (pos) tags, etc. to each token.
# Snorkel provides a prebuilt preprocessor for spaCy called `SpacyPreprocessor` which adds a new field to the
# data point containing a [spaCy `Doc` object](https://spacy.io/api/doc).
# For more info, see the [`SpacyPreprocessor` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.nlp.SpacyPreprocessor.html#snorkel.preprocess.nlp.SpacyPreprocessor).
#
#
# If you prefer to use a different NLP tool, you can also wrap that as a `Preprocessor` and use it in the same way.
# For more info, see the [`preprocessor` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.preprocessor.html#snorkel.preprocess.preprocessor).

# %% [markdown] {"tags": ["md-exclude"]}
# If the spaCy English model wasn't already installed, the next cell may raise an exception.
# If this happens, restart the kernel and re-execute the cells up to this point.

# %%
from snorkel.preprocess.nlp import SpacyPreprocessor

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)


# %%
@labeling_function(pre=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


# %% [markdown]
# Because spaCy is such a common preprocessor for NLP applications, we also provide a
# [prebuilt `labeling_function`-like decorator that uses spaCy](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.lf.nlp.nlp_labeling_function.html#snorkel.labeling.lf.nlp.nlp_labeling_function).
# This resulting LF is identical to the one defined manually above.

# %%
from snorkel.labeling.lf.nlp import nlp_labeling_function


@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


# %% [markdown]
# **Adding new domain-specific preprocessors and LF types is a great way to contribute to Snorkel!
# If you have an idea, feel free to reach out to the maintainers or submit a PR!**

# %% [markdown]
# ### e) Third-party Model LFs

# %% [markdown]
# We can also utilize other models, including ones trained for other tasks that are related to, but not the same as, the one we care about.
# The TextBlob-based LFs we created above are great examples of this!

# %% [markdown]
# ## 4. Combining Labeling Function Outputs with the Label Model

# %% [markdown]
# This tutorial demonstrates just a handful of the types of LFs that one might write for this task.
# One of the key goals of Snorkel is _not_ to replace the effort, creativity, and subject matter expertise required to come up with these labeling functions, but rather to make it faster to write them, since **in Snorkel the labeling functions are assumed to be noisy, i.e. innaccurate, overlapping, etc.**
# Said another way: the LF abstraction provides a flexible interface for conveying a huge variety of supervision signals, and the `LabelModel` is able to denoise these signals, reducing the need for painstaking manual fine-tuning.

# %%
lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]

# %% [markdown]
# With our full set of LFs, we can now apply these once again with `LFApplier` to get the label matrices.
# The Pandas format provides an easy interface that many practitioners are familiar with, but it is also less optimized for scale.
# For larger datasets, more compute-intensive LFs, or larger LF sets, you may decide to use one of the other data formats
# that Snorkel supports natively, such as Dask DataFrames or PySpark DataFrames, and their corresponding applier objects.
# For more info, check out the [Snorkel API documentation](https://snorkel.readthedocs.io/en/master/packages/labeling.html).

# %% {"tags": ["md-exclude-output"]}
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

# %%
LFAnalysis(L=L_train, lfs=lfs).lf_summary()

# %% [markdown] {"tags": ["md-exclude"]}
# We see that our labeling functions vary in coverage, how much they overlap/conflict with one another, and almost certainly their accuracies as well.
# We can view a histogram of how many LF labels the data points in our train set have to get an idea of our total coverage.

# %% {"tags": ["md-exclude"]}
import matplotlib.pyplot as plt

# %matplotlib inline


def plot_label_frequency(L):
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    plt.show()


plot_label_frequency(L_train)

# %% [markdown] {"tags": ["md-exclude"]}
# We see that over half of our `train` dataset data points have 2 or fewer labels from LFs.
# Fortunately, the labels we do have can be used to train a classifier over the comment text directly, allowing this final machine learning model to generalize beyond what our labeling functions labeling.

# %% [markdown]
# Our goal is now to convert the labels from our LFs into a single _noise-aware_ probabilistic (or confidence-weighted) label per data point.
# A simple baseline for doing this is to take the majority vote on a per-data point basis: if more LFs voted SPAM than HAM, label it SPAM (and vice versa).
# We can test this with the
# [`MajorityLabelVoter` baseline model](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.baselines.MajorityLabelVoter.html#snorkel.labeling.model.baselines.MajorityLabelVoter).

# %% {"tags": ["md-exclude-output"]}
from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

# %%
preds_train

# %% [markdown]
# However, as we can see from the summary statistics of our LFs in the previous section, they have varying properties and should not be treated identically. In addition to having varied accuracies and coverages, LFs may be correlated, resulting in certain signals being overrepresented in a majority-vote-based model. To handle these issues appropriately, we will instead use a more sophisticated Snorkel `LabelModel` to combine the outputs of the LFs.
#
# This model will ultimately produce a single set of noise-aware training labels, which are probabilistic or confidence-weighted labels. We will then use these labels to train a classifier for our task. For more technical details of this overall approach, see our [NeurIPS 2016](https://arxiv.org/abs/1605.07723) and [AAAI 2019](https://arxiv.org/abs/1810.02840) papers. For more info on the API, see the [`LabelModel` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html#snorkel.labeling.model.label_model.LabelModel).
#
# Note that no gold labels are used during the training process.
# The only information we need is the label matrix, which contains the output of the LFs on our training set.
# The `LabelModel` is able to learn weights for the labeling functions using only the label matrix as input.
# We also specify the `cardinality`, or number of classes.

# %% {"tags": ["md-exclude-output"]}
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

# %%
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

# %% [markdown]
# The majority vote model or more sophisticated `LabelModel` could in principle be used directly as a classifier if the outputs of our labeling functions were made available at test time.
# However, these models (i.e. these re-weighted combinations of our labeling function's votes) will abstain on the data points that our labeling functions don't cover (and additionally, may require slow or unavailable features to execute at test time).
# In the next section, we will instead use the outputs of the `LabelModel` as training labels to train a discriminative classifier **which can generalize beyond the labeling function outputs** to see if we can improve performance further.
# This classifier will also only need the text of the comment to make predictions, making it much more suitable for inference over unseen comments.
# For more information on the properties of the label model, see the [Snorkel documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html#snorkel.labeling.model.label_model.LabelModel).

# %% [markdown] {"tags": ["md-exclude"]}
# Let's briefly confirm that the labels the `LabelModel` produces are indeed probabilistic in nature.
# The following histogram shows the confidences we have that each data point has the label SPAM.
# The points we are least certain about will have labels close to 0.5.

# %% {"tags": ["md-exclude"]}


def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of SPAM")
    plt.ylabel("Number of data points")
    plt.show()


probs_train = label_model.predict_proba(L=L_train)
plot_probabilities_histogram(probs_train[:, SPAM])

# %% [markdown]
# ### Filtering out unlabeled data points

# %% [markdown]
# As we saw earlier, some of the data points in our `train` set received no labels from any of our LFs.
# These data points convey no supervision signal and tend to hurt performance, so we filter them out before training using a
# [built-in utility](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.filter_unlabeled_dataframe.html#snorkel.labeling.filter_unlabeled_dataframe).

# %%
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)

# %% [markdown]
# ## 5. Training a Classifier

# %% [markdown]
# In this final section of the tutorial, we'll use the probabilistic training labels we generated in the last section to train a classifier for our task.
# **The output of the Snorkel `LabelModel` is just a set of labels which can be used with most popular libraries for performing supervised learning, such as TensorFlow, Keras, PyTorch, Scikit-Learn, Ludwig, and XGBoost.**
# In this tutorial, we use the well-known library [Scikit-Learn](https://scikit-learn.org).
# **Note that typically, Snorkel is used (and really shines!) with much more complex, training data-hungry models, but we will use Logistic Regression here for simplicity of exposition.**

# %% [markdown]
# ### Featurization

# %% [markdown]
# For simplicity and speed, we use a simple "bag of n-grams" feature representation: each data point is represented by a one-hot vector marking which words or 2-word combinations are present in the comment text.

# %% {"tags": ["md-exclude-output"]}
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())

# %% [markdown]
# ### Scikit-Learn Classifier

# %% [markdown]
# As we saw in Section 4, the `LabelModel` outputs probabilistic (float) labels.
# If the classifier we are training accepts target labels as floats, we can train on these labels directly (see describe the properties of this type of "noise-aware" loss in our [NeurIPS 2016 paper](https://arxiv.org/abs/1605.07723)).
#
# If we want to use a library or model that doesn't accept probabilistic labels (such as Scikit-Learn), we can instead replace each label distribution with the label of the class that has the maximum probability.
# This can easily be done using the
# [`probs_to_preds` helper method](https://snorkel.readthedocs.io/en/master/packages/_autosummary/utils/snorkel.utils.probs_to_preds.html#snorkel.utils.probs_to_preds).
# We do note, however, that this transformation is lossy, as we no longer have values for our confidence in each label.

# %%
from snorkel.utils import probs_to_preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

# %% [markdown]
# We then use these labels to train a classifier as usual.

# %% {"tags": ["md-exclude-output"]}
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)

# %%
print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")

# %% [markdown]
# **We observe an additional boost in accuracy over the `LabelModel` by multiple points! This is in part because the discriminative model generalizes beyond the labeling function's labels and makes good predictions on all data points, not just the ones covered by labeling functions.
# By using the label model to transfer the domain knowledge encoded in our LFs to the discriminative model,
# we were able to generalize beyond the noisy labeling heuristics**.

# %% [markdown]
# ## Summary

# %% [markdown]
# In this tutorial, we accomplished the following:
# * We introduced the concept of Labeling Functions (LFs) and demonstrated some of the forms they can take.
# * We used the Snorkel `LabelModel` to automatically learn how to combine the outputs of our LFs into strong probabilistic labels.
# * We showed that a classifier trained on a weakly supervised dataset can outperform an approach based on the LFs alone as it learns to generalize beyond the noisy heuristics we provide.

# %% [markdown]
# ### Next Steps

# %% [markdown]
# If you enjoyed this tutorial and you've already checked out the [Getting Started](https://snorkel.org/get-started/) tutorial, check out the [Tutorials](https://snorkel.org/use-cases/) page for other tutorials that you may find interesting, including demonstrations of how to use Snorkel
#
# * As part of a [hybrid crowdsourcing pipeline](https://snorkel.org/use-cases/crowdsourcing-tutorial)
# * For [visual relationship detection over images](https://snorkel.org/use-cases/visual-relation-tutorial)
# * For [information extraction over text](https://snorkel.org/use-cases/spouse-demo)
# * For [data augmentation](https://snorkel.org/use-cases/02-spam-data-augmentation-tutorial)
#
# and more!
# You can also visit the [Snorkel website](https://snorkel.org) or [Snorkel API documentation](https://snorkel.readthedocs.io) for more info!
