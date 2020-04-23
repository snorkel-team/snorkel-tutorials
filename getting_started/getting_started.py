# -*- coding: utf-8 -*-
# %% [markdown] {"tags": ["md-exclude"]}
# # Getting Started with Snorkel

# %% {"tags": ["md-exclude"]}
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("getting_started")

# %% [markdown]
# ## Programmatically Building and Managing Training Data with Snorkel
#
# Snorkel is a system for _programmatically_ building and managing training datasets **without manual labeling**.
# In Snorkel, users can develop large training datasets in hours or days rather than hand-labeling them over weeks or months.
#
# Snorkel currently exposes three key programmatic operations:
# - **Labeling data**, e.g., using heuristic rules or distant supervision techniques
# - **Transforming data**, e.g., rotating or stretching images to perform data augmentation
# - **Slicing data** into different critical subsets for monitoring or targeted improvement
#
# Snorkel then automatically models, cleans, and integrates the resulting training data using novel, theoretically-grounded techniques.

# %% [markdown]
# <img src="img/Overview.png" onerror="this.onerror=null; this.src='/doks-theme/assets/images/layout/Overview.png';" align="center" style="display: block; margin-left: auto; margin-right: auto;">

# %% [markdown]
# In this quick walkthrough, we'll preview the high-level workflow and interfaces of Snorkel using a canonical machine learning problem: classifying spam.
# We'll use a public [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/), and see how **Snorkel can enable training a machine learning model without _any_ hand-labeled training data!**
# For more detailed versions of the sections in this walkthrough, see the corresponding tutorials: ([Spam LFs](https://snorkel.org/use-cases/01-spam-tutorial), [Spam TFs](https://snorkel.org/use-cases/02-spam-data-augmentation-tutorial), [Spam SFs](https://snorkel.org/use-cases/03-spam-data-slicing-tutorial)).

# %% [markdown]
# We'll walk through five basic steps:
#
# 1. **Writing Labeling Functions (LFs):** First, rather than hand-labeling any training data, we'll programmatically label our _unlabeled_ dataset with LFs.
# 2. **Modeling & Combining LFs:** Next, we'll use Snorkel's `LabelModel` to automatically learn the accuracies of our LFs and reweight and combine their outputs into a single, confidence-weighted training label per data point.
# 3. **Writing Transformation Functions (TFs) for Data Augmentation:** Then, we'll augment this labeled training set by writing a simple TF.
# 4. **Writing _Slicing Functions (SFs)_ for Data Subset Selection:** We'll also preview writing an SF to identify a critical subset or _slice_ of our training set.
# 5. **Training a final ML model:** Finally, we'll train an ML model with our training set.
#
# We'll start first by loading the _unlabeled_ comments, which we'll use as our training data, as a Pandas `DataFrame`:

# %%
from utils import load_unlabeled_spam_dataset

df_train = load_unlabeled_spam_dataset()

# %% [markdown]
# ## 1) Writing Labeling Functions
#
# _Labeling functions (LFs)_ are one of the core operators for building and managing training datasets programmatically in Snorkel.
# The basic idea is simple: **a labeling function is a function that outputs a label for some subset of the training dataset**.
# In our example here, each labeling function takes as input a comment data point, and either outputs a label (`SPAM = 1` or `NOT_SPAM = 0`) or abstains from labeling (`ABSTAIN = -1`):

# %%
# Define the label mappings for convenience
ABSTAIN = -1
NOT_SPAM = 0
SPAM = 1

# %% [markdown]
# Labeling functions can be used to represent many heuristic and/or noisy strategies for labeling data, often referred to as [weak supervision](https://www.snorkel.org/blog/weak-supervision).
# The basic idea of labeling functions, and other programmatic operators in Snorkel, is to let users inject domain information into machine learning models in higher level, higher bandwidth ways than manually labeling thousands or millions of individual data points.
# **The key idea is that labeling functions do not need to be perfectly accurate**, and can in fact even be correlated with each other.
# Snorkel will automatically estimate their accuracies and correlations in a [provably consistent way](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly), and then reweight and combine their output labels, leading to high-quality training labels.

# %% [markdown]
# In our text data setting here, labeling functions use:
#
# Keyword matches:

# %%
from snorkel.labeling import labeling_function


@labeling_function()
def lf_keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.text.lower() else ABSTAIN


# %% [markdown]
# Regular expressions:

# %%
import re


@labeling_function()
def lf_regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


# %% [markdown]
# Arbitrary heuristics:

# %%
@labeling_function()
def lf_short_comment(x):
    """Non-spam comments are often short, such as 'cool video!'."""
    return NOT_SPAM if len(x.text.split()) < 5 else ABSTAIN


# %% [markdown]
# Third-party models:

# %%
from textblob import TextBlob


@labeling_function()
def lf_textblob_polarity(x):
    """
    We use a third-party sentiment classification model, TextBlob.

    We combine this with the heuristic that non-spam comments are often positive.
    """
    return NOT_SPAM if TextBlob(x.text).sentiment.polarity > 0.3 else ABSTAIN


# %% [markdown]
# And much more!
# For many more types of labeling functions — including over data modalities beyond text — see the other [tutorials](https://snorkel.org/use-cases/) and [real-world examples](https://snorkel.org/resources/).
#
# In general the process of developing labeling functions is, like any other development process, an iterative one that takes time.
# However, in many cases it can be _orders-of-magnitude_ faster that hand-labeling training data.
# For more detail on the process of developing labeling functions and other training data operators in Snorkel, see the [Introduction Tutorials](https://snorkel.org/use-cases).

# %% [markdown]
# ## 2) Combining & Cleaning the Labels
#
# Our next step is to apply the labeling functions we wrote to the unlabeled training data.
# The result is a *label matrix*, `L_train`, where each row corresponds to a data point and each column corresponds to a labeling function.
# Since the labeling functions have unknown accuracies and correlations, their output labels may overlap and conflict.
# We use the `LabelModel` to automatically estimate their accuracies and correlations, reweight and combine their labels, and produce our final set of clean, integrated training labels:

# %%
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier

# Define the set of labeling functions (LFs)
lfs = [lf_keyword_my, lf_regex_check_out, lf_short_comment, lf_textblob_polarity]

# Apply the LFs to the unlabeled training data
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# Train the label model and compute the training labels
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

# %% [markdown]
# Note that we used the `LabelModel` to label data; however, on many data points, all the labeling functions abstain, and so the `LabelModel` abstains as well.
# We'll filter these data points out of our training set now:

# %%
df_train = df_train[df_train.label != ABSTAIN]

# %% [markdown]
# Our ultimate goal is to use the resulting labeled training data points to train a machine learning model that can **generalize beyond the coverage of the labeling functions and the `LabelModel`**.
# However first we'll explore some of Snorkel's other operators for building and managing training data.

# %% [markdown]
# ## 3) Writing Transformation Functions for Data Augmentation
#
# An increasingly popular and critical technique in modern machine learning is [data augmentation](https://www.snorkel.org/blog/tanda),
# the strategy of artificially *augmenting* existing labeled training datasets by creating transformed copies of the data points.
# Data augmentation is a practical and powerful method for injecting information about domain invariances into ML models via the data, rather than by trying to modify their internal architectures.
# The canonical example is randomly rotating, stretching, and transforming images when training image classifiers — a ubiquitous technique in the field of computer vision today.
# However, data augmentation is increasingly used in a range of settings, including text.
#
# Here, we implement a simple text data augmentation strategy — randomly replacing a word with a synonym.
# We express this as a *transformation function (TF)*:

# %%
import random

import nltk
from nltk.corpus import wordnet as wn

from snorkel.augmentation import transformation_function

nltk.download("wordnet", quiet=True)


def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    lemmas = set().union(*[s.lemmas() for s in wn.synsets(word)])
    return list(set(l.name().lower().replace("_", " ") for l in lemmas) - {word})


@transformation_function()
def tf_replace_word_with_synonym(x):
    """Try to replace a random word with a synonym."""
    words = x.text.lower().split()
    idx = random.choice(range(len(words)))
    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1 :])
        return x


# %% [markdown]
# Next, we apply this transformation function to our training dataset:

# %%
from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier

tf_policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
tf_applier = PandasTFApplier([tf_replace_word_with_synonym], tf_policy)
df_train_augmented = tf_applier.apply(df_train)

# %% [markdown]
# Note that a common challenge with data augmentation is figuring out how to tune and apply different transformation functions to best augment a training set.
# This is most commonly done as an ad hoc manual process; however, in Snorkel, various approaches for using automatically learned data augmentation _policies_ are supported.
# For more detail, see the [Spam TFs tutorial](https://snorkel.org/use-cases/02-spam-data-augmentation-tutorial).

# %% [markdown]
# ## 4) Writing a Slicing Function
#
# Finally, a third operator in Snorkel, *slicing functions (SFs)*, handles the reality that many datasets have certain subsets or _slices_ that are more important than others.
# In Snorkel, we can write SFs to (a) monitor specific slices and (b) improve model performance over them by adding representational capacity targeted on a per-slice basis.
#
# Writing a slicing function is simple.
# For example, we could write one that looks for suspiciously shortened links, which might be critical due to their likelihood of linking to malicious sites:

# %%
from snorkel.slicing import slicing_function


@slicing_function()
def short_link(x):
    """Return whether text matches common pattern for shortened ".ly" links."""
    return int(bool(re.search(r"\w+\.ly", x.text)))


# %% [markdown]
# We can now use Snorkel to monitor the performance over this slice, and to add representational capacity to our model in order to potentially increase performance on this slice.
# For a walkthrough of these steps, see the [Spam SFs tutorial](https://snorkel.org/use-cases/03-spam-data-slicing-tutorial).

# %% [markdown]
# ## 5) Training a Classifier
#
# The ultimate goal in Snorkel is to **create a training dataset**, which can then be plugged into an arbitrary machine learning framework (e.g. TensorFlow, Keras, PyTorch, Scikit-Learn, Ludwig, XGBoost) to train powerful machine learning models.
# Here, to complete this initial walkthrough, we'll train an extremely simple model — a "bag of n-grams" logistic regression model in Scikit-Learn — using the weakly labeled and augmented training set we made with our labeling and transformation functions:

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_text = df_train_augmented.text.tolist()
X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(train_text)

clf = LogisticRegression(solver="lbfgs")
clf.fit(X=X_train, y=df_train_augmented.label.values)

# %% [markdown]
# And that's it — you've trained your first model **without hand-labeling _any_ training data!**
# Next, to learn more about Snorkel, check out the [tutorials](https://snorkel.org/use-cases/), [resources](https://snorkel.org/resources), and [documentation](https://snorkel.readthedocs.io) for much more on how to use Snorkel to power your own machine learning applications.
