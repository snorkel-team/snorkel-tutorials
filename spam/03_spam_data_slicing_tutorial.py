# -*- coding: utf-8 -*-
# %% [markdown]
# # ✂️ Snorkel Intro Tutorial: _Data Slicing_
#
# In real-world applications, some model outcomes are often more important than others — e.g. vulnerable cyclist detections in an autonomous driving task, or, in our running **spam** application, potentially malicious link redirects to external websites.
#
# Traditional machine learning systems optimize for overall quality, which may be too coarse-grained.
# Models that achieve high overall performance might produce unacceptable failure rates on critical slices of the data — data subsets that might correspond to vulnerable cyclist detection in an autonomous driving task, or in our running spam detection application, external links to potentially malicious websites.
#
# In this tutorial, we:
# 1. **Introduce _Slicing Functions (SFs)_** as a programming interface
# 1. **Monitor** application-critical data subsets
# 2. **Improve model performance** on slices

# %% [markdown] {"tags": ["md-exclude"]}
# First, we'll set up our notebook for reproducibility and proper logging.

# %% {"tags": ["md-exclude"]}
import logging
import os
import numpy as np
import random
import torch

# For reproducibility
os.environ["PYTHONHASHSEED"] = "0"
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# To visualize logs
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# %% [markdown] {"tags": ["md-exclude"]}
# If you want to display all comment text untruncated, change `DISPLAY_ALL_TEXT` to `True` below.

# %% {"tags": ["md-exclude"]}
import pandas as pd


DISPLAY_ALL_TEXT = False

pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

# %% [markdown]
# _Note:_ this tutorial differs from the labeling tutorial in that we use ground truth labels in the train split for demo purposes.
# SFs are intended to be used *after the training set has already been labeled* by LFs (or by hand) in the training data pipeline.

# %%
from utils import load_spam_dataset

df_train, df_valid, df_test = load_spam_dataset(load_train_labels=True, split_dev=False)


# %% [markdown]
# ## 1. Write slicing functions
#
# We leverage *slicing functions* (SFs), which output binary _masks_ indicating whether an data point is in the slice or not.
# Each slice represents some noisily-defined subset of the data (corresponding to an SF) that we'd like to programmatically monitor.

# %% [markdown]
# In the following cells, we use the [`@slicing_function()`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html#snorkel.slicing.slicing_function) decorator to initialize an SF that identifies shortened links the spam dataset.
# These links could redirect us to potentially dangerous websites, and we don't want our users to click them!
# To select the subset of shortened links in our dataset, we write a regex that checks for the commonly-used `.ly` extension.
#
# You'll notice that the `short_link` SF is a heuristic, like the other programmatic ops we've defined, and may not fully cover the slice of interest.
# That's okay — in last section, we'll show how a model can handle this in Snorkel.

# %%
import re
from snorkel.slicing import slicing_function


@slicing_function()
def short_link(x):
    """Returns whether text matches common pattern for shortened ".ly" links."""
    return bool(re.search(r"\w+\.ly", x.text))


sfs = [short_link]

# %% [markdown]
# ### Visualize slices

# %% [markdown]
# With a utility function, [`slice_dataframe`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html#snorkel.slicing.slice_dataframe), we can visualize data points belonging to this slice in a `pandas.DataFrame`.

# %% {"tags": ["md-exclude-output"]}
from snorkel.slicing import slice_dataframe

short_link_df = slice_dataframe(df_valid, short_link)

# %%
short_link_df[["text", "label"]]

# %% [markdown]
# ## 2. Monitor slice performance with [`Scorer.score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer.score_slices)

# %% [markdown]
# In this section, we'll demonstrate how we might monitor slice performance on the `short_link` slice — this approach is compatible with _any modeling framework_.

# %% [markdown]
# ### Train a simple classifier
# First, we featurize the data — as you saw in the introductory Spam tutorial, we can extract simple bag-of-words features and store them as numpy arrays.

# %%
from sklearn.feature_extraction.text import CountVectorizer
from utils import df_to_features

vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train, Y_train = df_to_features(vectorizer, df_train, "train")
X_valid, Y_valid = df_to_features(vectorizer, df_valid, "valid")
X_test, Y_test = df_to_features(vectorizer, df_test, "test")

# %% [markdown]
# We define a `LogisticRegression` model from `sklearn` and show how we might visualize these slice-specific scores.

# %%
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=Y_train)
print(f"Test set accuracy: {100 * sklearn_model.score(X_test, Y_test):.1f}%")

# %%
from snorkel.utils import preds_to_probs

preds_test = sklearn_model.predict(X_test)
probs_test = preds_to_probs(preds_test, 2)

# %% [markdown]
# ### Store slice metadata in `S`

# %% [markdown]
# We apply our list of `sfs` to the data using an SF applier.
# For our data format, we leverage the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html#snorkel.slicing.PandasSFApplier).
# The output of the `applier` is an [`np.recarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) which stores vectors in named fields indicating whether each of $n$ data points belongs to the corresponding slice.

# %% {"tags": ["md-exclude-output"]}
from snorkel.slicing import PandasSFApplier

applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)

# %% [markdown]
# Now, we initialize a [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer) using the desired `metrics`.

# %%
from snorkel.analysis import Scorer

scorer = Scorer(metrics=["accuracy", "f1"])

# %% [markdown]
# Using the [`score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer.score_slices) method, we can see both `overall` and slice-specific performance.

# %%
scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)

# %% [markdown]
# Despite high overall performance, the `short_link` slice performs poorly here!

# %% [markdown]
# ### Write additional slicing functions (SFs)
#
# Slices are dynamic — as monitoring needs grow or change with new data distributions or application needs, an ML pipeline might require dozens, or even hundreds, of slices.
#
# We'll take inspiration from the labeling tutorial to write additional slicing functions.
# We demonstrate how the same powerful preprocessors and utilities available for labeling functions can be leveraged for slicing functions.

# %%
from snorkel.slicing import SlicingFunction, slicing_function
from snorkel.preprocess import preprocessor


# Keyword-based SFs
def keyword_lookup(x, keywords):
    return any(word in x.text.lower() for word in keywords)


def make_keyword_sf(keywords):
    return SlicingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


keyword_subscribe = make_keyword_sf(keywords=["subscribe"])
keyword_please = make_keyword_sf(keywords=["please", "plz"])


# Regex-based SF
@slicing_function()
def regex_check_out(x):
    return bool(re.search(r"check.*out", x.text, flags=re.I))


# Heuristic-based SF
@slicing_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return len(x.text.split()) < 5


# Leverage preprocessor in SF
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.9


# %% [markdown]
# Again, we'd like to visualize data points in a particular slice. This time, we'll inspect the `textblob_polarity` slice.
#
# Most data points with high-polarity sentiments are strong opinions about the video — hence, they are usually relevant to the video, and the corresponding labels are $0$.
# We might define a slice here for *product and marketing reasons*, it's important to make sure that we don't misclassify very positive comments from good users.

# %% {"tags": ["md-exclude-output"]}
polarity_df = slice_dataframe(df_valid, textblob_polarity)

# %%
polarity_df[["text", "label"]].head()

# %% [markdown]
# We can evaluate performance on _all SFs_ using the model-agnostic [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel-analysis-scorer).

# %%
extra_sfs = [
    keyword_subscribe,
    keyword_please,
    regex_check_out,
    short_comment,
    textblob_polarity,
]

sfs = [short_link] + extra_sfs
slice_names = [sf.name for sf in sfs]

# %% [markdown]
# Let's see how the `sklearn` model we learned before performs on these new slices!

# %% {"tags": ["md-exclude-output"]}
applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)

# %%
scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)

# %% [markdown]
# Looks like some do extremely well on our small test set, while others do decently.
# At the very least, we may want to monitor these to make sure that as we iterate to improve certain slices like `short_link`, we don't hurt the performance of others.
# Next, we'll introduce a model that helps us to do this balancing act automatically!

# %% [markdown]
# ## 3. Improve slice performance
#
# In the following section, we demonstrate a modeling approach that we call _Slice-based Learning,_ which improves performance by adding extra slice-specific representational capacity to whichever model we're using.
# Intuitively, we'd like to model to learn *representations that are better suited to handle data points in this slice*.
# In our approach, we model each slice as a separate "expert task" in the style of [multi-task learning](https://github.com/snorkel-team/snorkel-tutorials/blob/master/multitask/multitask_tutorial.ipynb); for further details of how slice-based learning works under the hood, check out the [code](https://github.com/snorkel-team/snorkel/blob/master/snorkel/slicing/utils.py) (with paper coming soon)!
#
# In other approaches, one might attempt to increase slice performance with techniques like _oversampling_ (i.e. with PyTorch's [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)), effectively shifting the training distribution towards certain populations.
#
# This might work with small number of slices, but with hundreds or thousands or production slices at scale, it could quickly become intractable to tune upsampling weights per slice.

# %% [markdown]
# ### Set up modeling pipeline with [`SliceAwareClassifier`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html)
#
# Snorkel supports performance monitoring on slices using discriminative models from [`snorkel.slicing`](https://snorkel.readthedocs.io/en/master/packages/slicing.html).
# To demonstrate this functionality, we'll first set up a the datasets + modeling pipeline in the PyTorch-based [`snorkel.classification`](https://snorkel.readthedocs.io/en/master/packages/classification.html) package.

# %% [markdown]
# First, we initialize a dataloaders for each split.

# %%
from utils import create_dict_dataloader

BATCH_SIZE = 64


train_dl = create_dict_dataloader(
    X_train, Y_train, "train", batch_size=BATCH_SIZE, shuffle=True
)
valid_dl = create_dict_dataloader(
    X_valid, Y_valid, "valid", batch_size=BATCH_SIZE, shuffle=False
)
test_dl = create_dict_dataloader(
    X_test, Y_test, "test", batch_size=BATCH_SIZE, shuffle=True
)

# %% [markdown]
# We'll now initialize a [`SliceAwareClassifier`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html):
# * `base_architecture`: We define a simple Multi-Layer Perceptron (MLP) in Pytorch to serve as the primary representation architecture. We note that the `BinarySlicingClassifier` is **agnostic to the base architecture** — you might leverage a Transformer model for text, or a ResNet for images.
# * `head_dim`: identifies the final output feature dimension of the `base_architecture`
# * `slice_names`: Specify the slices that we plan to train on with this classifier.

# %%
from snorkel.slicing import SliceAwareClassifier
from utils import get_pytorch_mlp


# Define model architecture
bow_dim = X_train.shape[1]
hidden_dim = bow_dim
mlp = get_pytorch_mlp(hidden_dim=hidden_dim, num_layers=2)

# Init slice model
slice_model = SliceAwareClassifier(
    base_architecture=mlp, head_dim=hidden_dim, slice_names=[sf.name for sf in sfs]
)

# %% [markdown]
# ### Monitor slice performance _during training_
#
# Using Snorkel's [`Trainer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.Trainer.html), we fit to `train_dl`, and validate on `valid_dl`.
#
# We note that we can monitor slice-specific performance during training — this is a powerful way to track especially critical subsets of the data.
# If logging in `Tensorboard` (i.e. [`snorkel.classification.TensorboardWritier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.TensorBoardWriter.html)), we would visualize individual loss curves and validation metrics to debug convegence for specific slices.

# %%
from snorkel.classification import Trainer

# For demonstration purposes, we set n_epochs=2
trainer = Trainer(lr=1e-4, n_epochs=2)
trainer.fit(slice_model, [train_dl, valid_dl])

# %% [markdown]
# ### Representation learning with slices

# %% [markdown]
# To cope with scale, we will attempt to learn and combine many slice-specific representations with an attention mechanism.
# (Please see our [Section 3 of our technical report](https://arxiv.org/abs/1909.06349) for details on this approach).

# %% [markdown]
# First, we'll generate the remaining `S` matrixes with the new set of slicing functions.

# %% {"tags": ["md-exclude-output"]}
applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)

# %% [markdown]
# In order to train using slice information, we'd like to initialize a **slice-aware dataloader**.
# To do this, we can use [`slice_model.make_slice_dataloader`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html#snorkel.slicing.SliceAwareClassifier.predict) to add slice labels to an existing dataloader.
#
# Under the hood, this method leverages slice metadata to add slice labels to the appropriate fields such that it's compatible with the initialized [`SliceClassifier`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html#snorkel-slicing-slicingclassifier).

# %%
train_dl_slice = slice_model.make_slice_dataloader(
    train_dl.dataset, S_train, shuffle=True, batch_size=BATCH_SIZE
)
valid_dl_slice = slice_model.make_slice_dataloader(
    valid_dl.dataset, S_valid, shuffle=False, batch_size=BATCH_SIZE
)
test_dl_slice = slice_model.make_slice_dataloader(
    test_dl.dataset, S_test, shuffle=False, batch_size=BATCH_SIZE
)

# %% [markdown]
# We train a single model initialized with all slice tasks.

# %%
from snorkel.classification import Trainer

# For demonstration purposes, we set n_epochs=2
trainer = Trainer(n_epochs=2, lr=1e-4, progress_bar=True)
trainer.fit(slice_model, [train_dl_slice, valid_dl_slice])

# %% [markdown]
# At inference time, the primary task head (`spam_task`) will make all final predictions.
# We'd like to evaluate all the slice heads on the original task head — [`score_slices`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html#snorkel.slicing.SliceAwareClassifier.score_slices) remaps all slice-related labels, denoted `spam_task_slice:{slice_name}_pred`, to be evaluated on the `spam_task`.

# %%
slice_model.score_slices([valid_dl_slice, test_dl_slice], as_dataframe=True)

# %% [markdown]
# *Note: in this toy dataset, we see high variance in slice performance, because our dataset is so small that (i) there are few data points in the train split, giving little signal to learn over, and (ii) there are few data points in the test split, making our evaluation metrics very noisy.
# For a demonstration of data slicing deployed in state-of-the-art models, please see our [SuperGLUE](https://github.com/HazyResearch/snorkel-superglue/tree/master/tutorials) tutorials.*

# %% [markdown]
# ---
# ## Recap

# %% [markdown]
# This tutorial walked through the process authoring slices, monitoring model performance on specific slices, and improving model performance using slice information.
# This programming abstraction provides a mechanism to heuristically identify critical data subsets.
# For more technical details about _Slice-based Learning,_ please see our [NeurIPS 2019 paper](https://arxiv.org/abs/1909.06349)!

# %%
