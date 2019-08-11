# -*- coding: utf-8 -*-
# %% [markdown]
# # Getting Started with Snorkel

# %% [markdown]
# In this quick walkthrough, we'll preview the high level workflow and interfaces of Snorkel.  For the more detailed version, see the [Introductory Tutorial](#).

# %% [markdown]
# ## Programmatically Building and Managing Training Data with Snorkel
#
# Snorkel is a system for programmatically building and managing training datasets. In Snorkel, users can develop training datasets in hours or days rather than hand-labeling them over weeks or months.
#
# Snorkel currently exposes three key programmatic operations: **labeling data**, for example using heuristic rules or distant supervision techniques; **transforming data**, for example rotating or stretching images to perform data augmentation; and **slicing data** into different critical subsets. Snorkel then automatically models, cleans, and integrates the resulting training data using novel, theoretically-grounded techniques.

# %% [markdown]
# <img src="img/snorkel_ops.png" align="center">

# %% [markdown]
# In this walkthrough, we'll look at a canonical machine learning problem: classifying spam.  We'll use a public [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle; for more details on this dataset see the [Introductory Tutorial](#).
#
# We'll walk through five basic steps:
#
# 1. **Writing Labeling Functions (LFs):** First, rather than hand-labeling any training data, we'll programamtically label our _unlabeled_ dataset with LFs
# 2. **Modeling & Combining LFs:** Next, we'll use the `LabelModel` to automatically learn the accuracies of our LFs and reweight and combine their outputs
# 3. **Writing Transformation Functions (TFs) for Data Augmentation:** Then, we'll augment this labeled training set by writing a simple TF
# 4. **Writing _Slicing Functions (SFs)_ for Data Subset Selection:** Then, we'll write an SF to identify a critical subset or _slice_ of our training set.
# 5. **Training a final ML model:** Finally, we'll train a simple ML model with our training set!
#
# We'll start first by loading the _unlabeled_ comments, which we'll use as our training data, as a pandas `DataFrame`:

# %%
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("intro")

# %%
from utils import load_unlabeled_spam_dataset
df_train = load_unlabeled_spam_dataset()

# %% [markdown]
# # OLD STUFF BELOW HERE

# %% [markdown]
# Each row is one comment consisting of text, author, and date values, as well as an integer id for which YouTube video the comment corresponds to.
# Additionally, since we are looking at the development set, these examples have labels as well- `1` for spam, `0` for ham (not spam).

# %% [markdown]
# ## STEP 1: Writing Labeling Functions
#
# _Labeling functions (LFs)_ are one of the core operators for building and managing training datasets programmatically in Snorkel.
# The basic idea is simple: **a labeling function is a function that labels some subset of the training dataset**.
# That is, each labeling function either outputs `SPAM`, `HAM`, or `ABSTAIN`:

# %% [markdown]
# Labeling functions can be used to represent many heuristic strategies for labeling data.
# **The key idea is that labeling functions do not need to be perfectly accurate**, as Snorkel will automatically estimate their accuracies and correlations, and then reweight and combine their output labels, leading to high-quality training labels.
#
# As a starting example, labeling functions can be based on **matching keywords**, using **regular expressions**, leveraging arbitrary **heuristics**, using **third-party models**, and much more- anything that can be expressed as a function that labels!

# %%
from snorkel.labeling import labeling_function
import re
from textblob import TextBlob

# Define the label mappings for convenience
ABSTAIN = -1
HAM = 0
SPAM = 1

@labeling_function()
def lf_keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

@labeling_function()
def lf_short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN

@labeling_function()
def lf_textblob_polarity(x):
    """
    We use a third-party sentiment classification model, TextBlob,
    combined with the heuristic that ham comments are often positive.
    """
    sentiment = TextBlob(x.text).sentiment
    return HAM if sentiment.polarity > 0.3 else ABSTAIN

lfs = [
    lf_keyword_my,
    lf_regex_check_out,
    lf_short_comment,
    lf_textblob_polarity
]

# %% [markdown]
# For many more types of labeling functions---including over data modalities beyond text---see the other [tutorials](https://github.com/snorkel-team/snorkel-tutorials) and examples at [snorkel.org](http://snorkel.org).
# In general the process of developing labeling functions is, like any other development process, an iterative one that takes time- but that, in many cases, can be orders-of-magnitude faster that hand-labeling training data.
# For more detail on the process of developing labeling functions and other training data operators in Snorkel, see the [full version of this tutorial](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam).

# %% [markdown]
# ## STEP 2: Combining & Cleaning the Labels
#
# Our next step is to apply the labeling functions we wrote to the unlabeled training data; we do this using the `LFApplier` corresponding to our base data class (in this case, the `PandasLFApplier`):

# %%
from snorkel.labeling import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# %% [markdown]
# The result of applying the labeling functions (LFs) to the data is (for each split of the data) a _label matrix_ with rows corresponding to data points, and columns corresponding to LFs.
# We can take a look at some statistics of the LFs on the dev set:

# %%
from snorkel.labeling import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
df_train['label'] = label_model.predict(L=L_train)

# %% [markdown]
# Note that above, we have applied the `LabelModel` to the test set, and see that we get an accuracy score of approximately $85\%$.
# In many Snorkel applications, it is not possible to apply the labeling functions (and therefore the `LabelModel`) at test time, for example due to the labeling functions using features not available at test time, or being overly slow to execute (for an example of this, see the [cross-modal tutorial](), and other examples at [snorkel.org]()).
#
# Here, we _can_ apply the `LabelModel` to the test set, but it leaves a lot to be desired- in part because the labeling functions leave a large portion of the dataset unlabeled:

# %% [markdown]
# Here is where the final step of the pipeline comes in handy- we will now use the probabilistic training labels to train a machine learning model which will generalize beyond---and outperform---the labeling functions.

# %% [markdown]
# ### [Optional] STEP 3: Writing Transformation Functions for Data Augmentation
#
# TODO: Intro text here

# %%
from snorkel.augmentation import transformation_function
import random
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")

def get_synonyms(word):
    """Helper function to get the synonyms of word from Wordnet."""
    lemmas = set().union(*[s.lemmas() for s in wn.synsets(word)])
    return list(set([l.name().lower().replace("_", " ") for l in lemmas]) - {word})

@transformation_function()
def tf_replace_word_with_synonym(x):
    """Try to replace a random word with a synonym."""
    words = x.text.lower().split()
    idx = random.choice(range(len(words)))
    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1:])
        return x


# %% [markdown]
# Now, we apply these to the data:

# %%
from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier 

tf_policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
tf_applier = PandasTFApplier([tf_replace_word_with_synonym], tf_policy)
df_train_augmented = tf_applier.apply(df_train)

# %% [markdown]
# ## STEP 3: Training a Machine Learning Model
#
# In this final step, our goal is to train a machine learning model that generalizes beyond what the labeling functions label, and thereby outperforms the `LabelModel` above.
# In this example, **we use an extremely simple ML model**, but still see this generalization effect occur!
#
# Note that because the output of the Snorkel `LabelModel` is just a set of labels, Snorkel easily integrates with most popular libraries for performing supervised learning: TensorFlow, Keras, PyTorch, Scikit-Learn, Ludwig, XGBoost, etc.
#
# In this tutorial we demonstrate using classifiers from Keras and Scikit-Learn. For simplicity and speed, we use a simple "bag of n-grams" feature representation: each data point is represented by a one-hot vector marking which words or 2-word combinations are present in the comment text, which we compute using a basic Scikit-Learn `CountVectorizer`:

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

text_train = [row.text for i, row in df_train_augmented.iterrows()]
X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(text_train)

clf = LogisticRegression()
clf.fit(X=X_train, y=df_train_augmented.label.values)

# %% [markdown]
# **We observe an additional boost in accuracy over the `LabelModel` by multiple points!
# By using the label model to transfer the domain knowledge encoded in our LFs to the discriminative model,
# we were able to generalize beyond the noisy labeling heuristics**.

# %% [markdown]
# ### Next Steps

# %% [markdown]
# In this tutorial, we demonstrated the basic pipeline of Snorkel, and showed how it can enable us to train high-quality ML models without hand-labeling large training datasets.
#
# **Next, check out the extended version of this tutorial---the [spam tutorial](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam)---which goes into much more detail about the actual process of iterating on labeling functions and other types of operators to build end-to-end ML applications in Snorkel!**
#
# You can also check out the [Snorkel 101 Guide](#) and the [`snorkel-tutorials` table of contents](https://github.com/snorkel-team/snorkel-tutorials#snorkel-tutorials) for other tutorials that you may find interesting, including demonstrations of how to use Snorkel:
#
# * As part of a [hybrid crowdsourcing pipeline](https://github.com/snorkel-team/snorkel-tutorials/tree/master/crowdsourcing)
# * For [scene-graph detection over images](https://github.com/snorkel-team/snorkel-tutorials/tree/master/scene_graph)
# * For [information extraction over text](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spouse)
# * For [data augmentation](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam)
#
# and many more!
# You can also visit the [Snorkel homepage](http://snorkel.org) or [Snorkel API documentation](https://snorkel.readthedocs.io) for more info!
