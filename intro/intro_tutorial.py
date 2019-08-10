# -*- coding: utf-8 -*-
# %% [markdown]
# # Introduction to Snorkel

# %% [markdown]
# In this first Snorkel tutorial, we will walk through the basics of Snorkel, using it to fight YouTube comments spam!

# %% [markdown]
# ### Snorkel Basics
#
# **Snorkel is a system for programmatically building and managing training datasets to rapidly and flexibly fuel machine learning models.**
#
# Today's state-of-the-art machine learning models are more powerful and easy to use than ever before- however, they require massive _training datasets_.
# For example, if we wanted to use one of the latest and greatest machine learning models to classify YouTube comments as spam or not, we'd need to first hand-label a large number of YouTube comments---a *training set*---that our model would learn from.
#
# Building and managing training datasets often requires slow and prohibitively expensive manual effort by domain experts (especially when data is private or requires expensive expert labelers).
# In Snorkel, users instead write **programmatic operations to label, transform, and structure training datasets** for machine learning, without needing to hand label any training data; Snorkel then uses novel, theoretically-grounded modeling techniques to clean and integrate the resulting training data.
# In a wide range of applications---from medical image monitoring to text information extraction to industrial deployments over web data---Snorkel provides a radically faster and more flexible to build machine learning applications; see [snorkel.org](snorkel.org) for more detail on many examples of Snorkel usage!
#
# In this intro tutorial, we'll see how Snorkel can let us train a machine learning model for spam classification _without_ hand-labeling anything but a small test and validation set (i.e., without hand-labeling _any_ training data).

# %% [markdown]
# ### The Snorkel Pipeline

# %% [markdown]
# <img src="img/snorkel_101_pipeline.png" align="left">`

# %% [markdown]
# Snorkel is a system for programmatically building and managing training datasets in a number of ways- we'll start with **labeling** training data.
# Here, the basic pipeline consists of three main steps:
#
# 1. **Writing Labeling Functions:** First, instead of labeling the training data by hand, we will write _labeling functions_, special Python functions that label subsets of the training data heuristically.
#
# 2. **Combining & cleaning the labels:** The labeling functions we write will have varying accuracies, coverages, and correlations- leading to complex overlaps and disagreements. We will use Snorkel's `LabelModel` to automatically reweight and combine the outputs of the labeling functions, resulting in clean, _probabilistic_ training labels.
#
# 3. **Training a machine learning model:** Finally, we'll show how to use the probabilistic training labels from step (2) to train a machine learning model, which we'll show will generalize beyond and outperform the labeling functions!
#
# For much more on Snorkel---including four years of academic papers, applications, and more!---see [snorkel.org](http://snorkel.org).
# You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/).

# %% [markdown]
# ### Example Problem: Classifying YouTube Spam

# %% [markdown]
# <img src="img/snorkel_101_spam.png" width="500px" align="left">

# %% [markdown]
# For this tutorial, we'll focus on a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle that consists of YouTube comments from 5 videos.
# **For a much more detailed version of this tutorial, see the Snorkel [spam tutorial](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam).**
#
# The simple classification task we focus on here is a classic one in the history of machine learning- classifying each comment as being "spam" or "ham" (not spam); more specifically, we aim to train a classifier that outputs one of the following labels for each YouTube comment:
#
# * **`SPAM`**: irrelevant or inappropriate messages, or
# * **`HAM`**: comments relevant to the video
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
#         
# For our task, we have access to a large amount of *unlabeled YouTube comments*, which forms our **training set**.
# We also have access to a small amount of labeled data, which we split into **development set** (for looking at while developing labeling functions), a **validation set** (for model hyperparameter tuning), and a **test set** (for final evaluation).
# We load this data in now:

# %%
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("intro")

# %%
from utils import load_spam_dataset

# Load data- for details see the spam tutorial
df_train, *_ = load_spam_dataset()

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
from snorkel.labeling.lf import labeling_function
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
from snorkel.labeling.apply.pandas import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# %% [markdown]
# The result of applying the labeling functions (LFs) to the data is (for each split of the data) a _label matrix_ with rows corresponding to data points, and columns corresponding to LFs.
# We can take a look at some statistics of the LFs on the dev set:

# %%
from snorkel.labeling.model.label_model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
Y_preds_train = label_model.predict(L=L_train)

# %% [markdown]
# Note that above, we have applied the `LabelModel` to the test set, and see that we get an accuracy score of approximately $85\%$.
# In many Snorkel applications, it is not possible to apply the labeling functions (and therefore the `LabelModel`) at test time, for example due to the labeling functions using features not available at test time, or being overly slow to execute (for an example of this, see the [cross-modal tutorial](), and other examples at [snorkel.org]()).
#
# Here, we _can_ apply the `LabelModel` to the test set, but it leaves a lot to be desired- in part because the labeling functions leave a large portion of the dataset unlabeled:

# %% [markdown]
# Here is where the final step of the pipeline comes in handy- we will now use the probabilistic training labels to train a machine learning model which will generalize beyond---and outperform---the labeling functions.

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

text_train = [row.text for i, row in df_train.iterrows()]
X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(text_train)

clf = LogisticRegression()
clf.fit(X=X_train, y=Y_preds_train)

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
