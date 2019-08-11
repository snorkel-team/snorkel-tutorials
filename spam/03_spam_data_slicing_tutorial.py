# -*- coding: utf-8 -*-
# %% [markdown]
# # ✂️ _Spam_ — Data Slicing Tutorial
#
# In real-world applications, some model outcomes are often more important than others — e.g. vulnerable cyclist detections in an autonomous driving task, or, in our running **spam** application, potentially malicious link redirects to external websites.
#
# Traditional machine learning systems optimize for overall quality, which may be too coarse-grained: models that achieve high overall performance might produce unacceptable failure rates on critical slices of the data — data subsets that might correspond to vulnerable cyclist detection in an autonomous driving task, or in our running spam detection application, external links to potentially malicious websites.
#
# In this tutorial, we introduce _Slicing Functions (SFs)_ as a programming interface to:
# 1. **Monitor** application-critical data slices
# 2. **Improve model performance** on slices

# %% [markdown]
# First, we'll set up our notebook for reproducibility and proper logging.

# %%
import logging
import os
import pandas as pd
from snorkel.utils import set_seed

# For reproducibility
os.environ["PYTHONHASHSEED"] = "0"
set_seed(111)

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# To visualize logs
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Show full columns for viewing data
pd.set_option("display.max_colwidth", -1)

# Download the spaCy english model for preprocessing
import spacy

# ! python -m spacy download en

# %% [markdown]
# _Note:_ this tutorial differs from the labeling tutorial in that we use ground truth labels in the train split for demo purposes.
# In practice, data slicing is agnostic to the _training labels_ used as inputs — you can use Snorkel-generated labels as inputs to this pipeline.

# %%
from utils import load_spam_dataset

df_train, df_valid, df_test = load_spam_dataset(
    load_train_labels=True, include_dev=False
)


# %% [markdown]
# ## 1. Train a discriminative model
#
# To start, we'll initialize a discriminative model using our [`SnorkelClassifier`](https://snorkel.readthedocs.io/en/redux/source/snorkel.classification.html).
# We'll assume that you are familiar with Snorkel's multitask model — if not, we'd recommend you check out our [Multitask Tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/multitask/multitask_tutorial.ipynb).

# %% [markdown]
# ### Featurize Data
#
# As a first step, we'll featurize the data—as you saw in the introductory Spam tutorial, we can extract simple bag of words features and store them as numpy arrays.

# %%
import torch
from sklearn.feature_extraction.text import CountVectorizer
from snorkel.classification.data import DictDataset, DictDataLoader

vectorizer = CountVectorizer(ngram_range=(1, 1))


def df_to_torch_example(vectorizer, df, fit_train=False):
    words = [row.text for i, row in df.iterrows()]

    if fit_train:
        feats = vectorizer.fit_transform(words)
    else:
        feats = vectorizer.transform(words)
    X = feats.todense()
    Y = df["label"].values
    return X, Y


# %%
X_train, Y_train = df_to_torch_example(vectorizer, df_train, fit_train=True)
X_valid, Y_valid = df_to_torch_example(vectorizer, df_valid, fit_train=False)
X_test, Y_test = df_to_torch_example(vectorizer, df_test, fit_train=False)

# %% [markdown]
# ### Create DataLoaders
#
# Next, we'll use the extracted Tensors to initialize a `DictDataLoader` — as a quick recap, this is a Snorkel-specific class that inherits from the common PyTorch class and supports multiple data fields in the `X_dict` and labels in the `Y_dict`.
#
# In this task, we'd like to store the `bow_features` in our `X_dict`, and we have one set of labels (for now) correpsonding to the `spam_task`.

# %%
BATCH_SIZE = 32


def create_dict_dataloader(X, Y, split, **kwargs):
    """Create a DictDataLoader for bag of words features."""
    ds = DictDataset(
        name="spam_dataset",
        split=split,
        X_dict={"bow_features": torch.FloatTensor(X)},
        Y_dict={"spam_task": torch.LongTensor(Y)},
    )
    return DictDataLoader(ds, **kwargs)


dl_train = create_dict_dataloader(
    X_train, Y_train, split="train", batch_size=BATCH_SIZE, shuffle=True
)
dl_valid = create_dict_dataloader(
    X_valid, Y_valid, split="valid", batch_size=BATCH_SIZE, shuffle=False
)
dl_test = create_dict_dataloader(
    X_test, Y_test, split="test", batch_size=BATCH_SIZE, shuffle=False
)

# %% [markdown]
# We can inspect our datasets to confirm that they have the appropriate fields.

# %%
dl_valid.dataset

# %% [markdown]
# ### Define `SnorkelClassifier`
#
# We define a simple Multi-Layer Perceptron (MLP) architecture to learn from the `bow_features`.
#
# _Note: the following might feel like extra steps to define what is a very simple architecture, but this will lend us additional flexibility later in the pipeline!_

# %% [markdown]
# To start, we define a `module_pool` with all the [PyTorch](https://pytorch.org) modules that we'll want to include in our network.

# %%
import torch.nn as nn

bow_dim = X_train.shape[1]
module_pool = nn.ModuleDict(
    {
        "mlp": nn.Sequential(nn.Linear(bow_dim, bow_dim), nn.ReLU()),
        "prediction_head": nn.Linear(bow_dim, 2),
    }
)

# %% [markdown]
# Then, we specify the desired `task_flow` through each module.

# %%
from snorkel.classification.task import Operation

task_flow = [
    Operation(name="input_op", module_name="mlp", inputs=[("_input_", "bow_features")]),
    Operation(name="head_op", module_name="prediction_head", inputs=[("input_op", 0)]),
]

# %% [markdown]
# With these pieces, we're ready to define a [`Task`](https://snorkel.readthedocs.io/en/redux/source/snorkel.classification.html#module-snorkel.classification.task) in Snorkel for spam classification.

# %%
from functools import partial
from snorkel.analysis import Scorer
from snorkel.classification import (
    Task,
    cross_entropy_from_outputs,
    softmax_from_outputs,
)

spam_task = Task(
    name="spam_task",
    module_pool=module_pool,
    task_flow=task_flow,
    loss_func=partial(cross_entropy_from_outputs, "head_op"),
    output_func=partial(softmax_from_outputs, "head_op"),
    scorer=Scorer(metrics=["accuracy", "f1"]),
)

# %% [markdown]
# We'll initialize a [`SnorkelClassifier`](https://snorkel.readthedocs.io/en/redux/source/snorkel.classification.html) with the `spam_task` we've created, initialize a corresponding [`Trainer`](https://snorkel.readthedocs.io/en/redux/source/snorkel.classification.training.html#module-snorkel.classification.training.trainer), and `fit` to our dataloaders!

# %%
from snorkel.classification import MultitaskClassifier, Trainer

model = MultitaskClassifier([spam_task])
trainer = Trainer(n_epochs=5, lr=1e-4, progress_bar=True)
trainer.fit(model, [dl_train, dl_valid])

# %% [markdown]
# How well does our model do?

# %%
model.score([dl_train, dl_valid], as_dataframe=True)

# %% [markdown]
# ## 2. Perform error analysis
#
# In overall metrics (`f1`, `accuracy`) our model appears to perform well!
# However, we emphasize we might actually be _more_ interested in performance for application-critical subsets, or _slices_.
#
# Let's perform an [`error_analysis`](https://snorkel.readthedocs.io/en/redux/source/snorkel.analysis.html#module-snorkel.analysis.error_analysis) to see where our model makes mistakes.
# We collect the predictions from the model and visualize examples in specific error buckets.

# %%
from snorkel.analysis.error_analysis import get_label_buckets

outputs = model.predict(dl_valid, return_preds=True)
error_buckets = get_label_buckets(
    outputs["golds"]["spam_task"], outputs["preds"]["spam_task"]
)

# %% [markdown]
# For application purposes, we might care especially about false negatives (i.e. true label was $1$, but model predicted $0$) — for the spam task, this error mode might expose users to external links that correspond to malware!

# %%
df_valid[["text", "label"]].iloc[error_buckets[(1, 0)]].head()

# %% [markdown]
# We notice that we're mis-classifying particularly some comments with shortened urls (e.g. `bit.ly/...`) — these links could redirect us to potentially dangerous websites, and we don't want our users to click them!

# %% [markdown]
# ## 3. Monitor data slices
#
# We leverage *slicing functions* (SFs) — an abstraction that shares syntax with *labeling functions*, which you should already be familiar with.
# (If not, please see the [intro tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb).)
# A key difference: whereas labeling functions output labels, slicing functions output binary _masks_ indicating whether an example is in the slice or not.

# %% [markdown]
# In the following cells, we define a slicing function that identifies these shortened links the spam dataset.
# To do so, we write a regex that checks for the commonly-used `.ly` extension.
#
# You'll notice that the slicing function is noisily defined — it doesn't represent the ground truth for all short links.
# Instead, SFs are often heuristics to quickly measure performance over important subsets of the data.

# %%
import re
from snorkel.slicing import slicing_function


@slicing_function()
def short_link(x):
    """Returns whether text matches common pattern for shortened ".ly" links."""
    return bool(re.search(r"\w+\.ly", x.text))


sfs = [short_link]
slice_names = [sf.name for sf in sfs]

# %% [markdown]
# For our $n$ examples and $k$ slices in each split, we apply the SF to our data to create an $n \times k$ matrix. (So far, $k=1$).

# %%
from snorkel.slicing import PandasSFApplier

applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)
S_test = applier.apply(df_test)

# %% [markdown]
# ### Visualize slices with `snorkel.slicing.monitor`

# %% [markdown]
# With a utility function, `slice_dataframe`, we can visualize examples belonging to this slice in a `pandas.DataFrame`.

# %%
from snorkel.slicing import slice_dataframe

short_link_df = slice_dataframe(df_valid, short_link)
short_link_df[["text", "label"]]

# %% [markdown]
# Now, we add labels for this particular slice to an existing dataloader.
# Specifically, `add_slice_labels` will add two sets of labels for each slice:
# * `spam_task_slice:{slice_name}_ind`: an indicator label, which corresponds to the outputs of the slicing functions.
# These indicate whether each example is in the slice (`label=1`)or not (`label=0`).
# * `spam_task_slice:{slice_name}_pred`: a _masked_ set of the original task labels (in this case, labeled `spam_task`) for each slice. Examples that are masked (with `label=-1`) will not contribute to loss or scoring.

# %%
from snorkel.slicing import add_slice_labels

slice_names = [sf.name for sf in sfs]
add_slice_labels(dl_train, spam_task, S_train, slice_names)
add_slice_labels(dl_valid, spam_task, S_valid, slice_names)
add_slice_labels(dl_test, spam_task, S_test, slice_names)

# %%
dl_valid.dataset

# %% [markdown]
# With our updated dataloader, we want to evaluate our model on the defined slice.
# In the `SnorkelClassifier`, we can call `score` with an additional argument, `remap_labels`, to specify that the slice's prediction labels, `spam_task_slice:short_link_pred`, should be mapped to the `spam_task` for evaluation.

# %%
model.score(
    dataloaders=[dl_valid, dl_test],
    remap_labels={"spam_task_slice:short_link_pred": "spam_task"},
    as_dataframe=True,
)

# %% [markdown]
# ### Monitor slices with `SliceScorer`

# %% [markdown]
# If you're using a model other than `SnorkelClassifier`, you can still evaluate on slices using the more general `SliceScorer` class.
#
# We define a `LogisticRegression` model from sklearn and show how we might visualize these slice-specific scores.

# %%
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=Y_train)
sklearn_model.score(X_test, Y_test)

# %%
from snorkel.utils import preds_to_probs
from snorkel.slicing import SliceScorer


preds_test = sklearn_model.predict(X_test)

scorer = Scorer(metrics=["accuracy", "f1"])
scorer = SliceScorer(scorer, slice_names)
scorer.score(
    S=S_test,
    golds=Y_test,
    preds=preds_test,
    probs=preds_to_probs(preds_test, 2),
    as_dataframe=True,
)

# %% [markdown]
# ## 4. Improve slice performance
#
# In classification tasks, we might attempt to increase slice performance with techniques like _oversampling_ (i.e. with PyTorch's [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)).
# This would shift the training distribution to over-represent certain minority populations.
# Intuitively, we'd like to show the model more `short_link` examples so that the representation is better suited to handle them.
#
# A technique like upsampling might work with a small number of slices, but with hundreds or thousands or production slices, it could quickly become intractable to tune upsampling weights per slice.
# In the following section, we show a modeling approach that we call _Slice-based Learning,_ which handles numerous slices using with slice-specific representation learning.

# %% [markdown]
# ### Write additional slicing functions (SFs)
#
# We'll take inspiration from the labeling tutorial to write a few additional `SlicingFunctions`.

# %%
from snorkel.slicing import SlicingFunction, slicing_function, nlp_slicing_function
from snorkel.preprocess import preprocessor


def keyword_lookup(x, keywords):
    return any(word in x.text.lower() for word in keywords)


def make_keyword_sf(keywords):
    return SlicingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_sf(keywords=["subscribe"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_sf(keywords=["please", "plz"])


@nlp_slicing_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    return len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents])


@slicing_function()
def regex_check_out(x):
    return bool(re.search(r"check.*out", x.text, flags=re.I))


@slicing_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return len(x.text.split()) < 5


@slicing_function(pre=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    return len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents])


from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.9


extra_sfs = [
    keyword_subscribe,
    keyword_please,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
]

sfs = [short_link] + extra_sfs

# %%
applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)
S_test = applier.apply(df_test)

# %%
slice_names = [sf.name for sf in sfs]
add_slice_labels(dl_train, spam_task, S_train, slice_names)
add_slice_labels(dl_valid, spam_task, S_valid, slice_names)
add_slice_labels(dl_test, spam_task, S_test, slice_names)

# %% [markdown]
# Like we saw above, we'd like to visualize examples in the slice.

# %%
polarity_df = slice_dataframe(df_valid, textblob_polarity)
polarity_df[["text", "label"]].head()

# %% [markdown]
# ### Representation learning with slices

# %% [markdown]
# To cope with scale, we will attempt to learn and combine many slice-specific representations with an attention mechanism.
# (For details, please see our technical report — coming soon!)
# Using the helper, `convert_to_slice_tasks`, we initialize slice-specific tasks to learn "expert task heads" for each slice, in the style of multi-task learning.
# The original `spam_task` now contains the attention mechanism to then combine these slice experts.

# %%
from snorkel.slicing import convert_to_slice_tasks

slice_tasks = convert_to_slice_tasks(spam_task, slice_names)
slice_tasks

# %%
slice_model = MultitaskClassifier(slice_tasks)

# %% [markdown]
# We train a single model initialized with all slice tasks.
# We note that we can monitor slice-specific performance during training — this is a powerful way to track especially critical subsets of the data.
#
# _Note: This model includes more parameters (corresponding to additional slices), and therefore requires more time to train — we set `num_epochs=1` for demonstration purposes._

# %%
trainer = Trainer(n_epochs=1, lr=1e-4, progress_bar=True)
trainer.fit(slice_model, [dl_train, dl_valid])

# %% [markdown]
# At inference time, the primary task head (`spam_task`) will make all final predictions.
# We'd like to evaluate all the slice heads on the original task head.
# To do this, we use our `remap_labels` API, as we did earlier.
# Note that this time, we map each `ind` head to `None` — it doesn't make sense to evaluate these labels on the base task head.

# %%
Y_dict = dl_valid.dataset.Y_dict
eval_mapping = {label: "spam_task" for label in Y_dict.keys() if "pred" in label}
eval_mapping.update({label: None for label in Y_dict.keys() if "ind" in label})

# %% [markdown]
# _Note: in this toy dataset, we might not see significant gains because our dataset is so small that (i) there are few examples the train split, giving little signal to learn over, and (ii) there are few examples in the test split, making our evaluation metrics very noisy.
# For a demonstration of data slicing deployed in state-of-the-art models, please see our [SuperGLUE](https://github.com/HazyResearch/snorkel-superglue/tree/master/tutorials) tutorials._

# %%
slice_model.score([dl_valid], remap_labels=eval_mapping, as_dataframe=True)

# %% [markdown]
# You've just defined slicing functions to monitor specific slices + improved slice-specific performance.
# For more technical details about _Slice-based Learning,_ stay tuned — our technical report is coming soon!

# %%
