# -*- coding: utf-8 -*-
# %% [markdown]
# # 📈 Snorkel Intro Tutorial: Data Augmentation

# %% [markdown]
# In this tutorial, we will walk through the process of using *transformation functions* (TFs) to perform data augmentation.
# Like the labeling tutorial, our goal is to train a classifier to YouTube comments as `SPAM` or `HAM` (not spam).
# In the [previous tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb),
# we demonstrated how to label training sets programmatically with Snorkel.
# In this tutorial, we'll assume that step has already been done, and start with labeled training data,
# which we'll aim to augment using transformation functions.
#
# %% [markdown] {"tags": ["md-exclude"]}
# * For more details on the task, check out the [labeling tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb)
# * For an overview of Snorkel, visit [snorkel.org](https://snorkel.org)
# * You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/)
#
# %% [markdown]
# Data augmentation is a popular technique for increasing the size of labeled training sets by applying class-preserving transformations to create copies of labeled data points.
# In the image domain, it is a crucial factor in almost every state-of-the-art result today and is quickly gaining
# popularity in text-based applications.
# Snorkel models the data augmentation process by applying user-defined *transformation functions* (TFs) in sequence.
# You can learn more about data augmentation in
# [this blog post about our NeurIPS 2017 work on automatically learned data augmentation](https://snorkel.org/blog/tanda/).
#
# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/).
# 2. **Writing Transformation Functions**: We write Transformation Functions (TFs) that can be applied to training data points to generate new training data points.
# 3. **Applying Transformation Functions to Augment Our Dataset**: We apply a sequence of TFs to each training data point, using a random policy, to generate an augmented training set.
# 4. **Training a Model**: We use the augmented training set to train an LSTM model for classifying new comments as `SPAM` or `HAM`.

# %% [markdown] {"tags": ["md-exclude"]}
# This next cell takes care of some notebook-specific housekeeping.
# You can ignore it.

# %% {"tags": ["md-exclude"]}
import os
import random

import numpy as np

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# Turn off TensorFlow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For reproducibility
seed = 0
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(0)
random.seed(0)

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

# %% [markdown]
# ## 1. Loading Data

# %% [markdown]
# We load the Kaggle dataset and create Pandas DataFrame objects for the `train` and `test` sets.
# The two main columns in the DataFrames are:
# * **`text`**: Raw text content of the comment
# * **`label`**: Whether the comment is `SPAM` (1) or `HAM` (0).
#
# For more details, check out the [labeling tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb).

# %%
from utils import load_spam_dataset

df_train, df_test = load_spam_dataset(load_train_labels=True)

# We pull out the label vectors for ease of use later
Y_train = df_train["label"].values
Y_test = df_test["label"].values


# %%
df_train.head()

# %% [markdown]
# ## 2. Writing Transformation Functions (TFs)
#
# Transformation functions are functions that can be applied to a training data point to create another valid training data point of the same class.
# For example, for image classification problems, it is common to rotate or crop images in the training data to create new training inputs.
# Transformation functions should be atomic e.g. a small rotation of an image, or changing a single word in a sentence.
# We then compose multiple transformation functions when applying them to training data points.
#
# Common ways to augment text includes replacing words with their synonyms, or replacing names entities with other entities.
# More info can be found
# [here](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28) or
# [here](https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610).
# Our basic modeling assumption is that applying these operations to a comment generally shouldn't change whether it is `SPAM` or not.
#
# Transformation functions in Snorkel are created with the
# [`transformation_function` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/augmentation/snorkel.augmentation.transformation_function.html#snorkel.augmentation.transformation_function),
# which wraps a function that takes in a single data point and returns a transformed version of the data point.
# If no transformation is possible, a TF can return `None` or the original data point.
# If all the TFs applied to a data point return `None`, the data point won't be included in
# the augmented dataset when we apply our TFs below.
#
# Just like the `labeling_function` decorator, the `transformation_function` decorator
# accepts `pre` argument for `Preprocessor` objects.
# Here, we'll use a
# [`SpacyPreprocessor`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.nlp.SpacyPreprocessor.html#snorkel.preprocess.nlp.SpacyPreprocessor).

# %%
from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

# %%
import names
from snorkel.augmentation import transformation_function

# Pregenerate some random person names to replace existing ones with
# for the transformation strategies below
replacement_names = [names.get_full_name() for _ in range(50)]


# Replace a random named entity with a different entity of the same type.
@transformation_function(pre=[spacy])
def change_person(x):
    person_names = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]
    # If there is at least one person name, replace a random one. Else return None.
    if person_names:
        name_to_replace = np.random.choice(person_names)
        replacement_name = np.random.choice(replacement_names)
        x.text = x.text.replace(name_to_replace, replacement_name)
        return x


# Swap two adjectives at random.
@transformation_function(pre=[spacy])
def swap_adjectives(x):
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    # Check that there are at least two adjectives to swap.
    if len(adjective_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
        # Swap tokens in positions idx1 and idx2.
        x.text = " ".join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1 : idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2 :].text,
            ]
        )
        return x


# %% [markdown]
# We add some transformation functions that use `wordnet` from [NLTK](https://www.nltk.org/) to replace different parts of speech with their synonyms.

# %% {"tags": ["md-exclude-output"]}
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")


def get_synonym(word, pos=None):
    """Get synonym for word given its part-of-speech (pos)."""
    synsets = wn.synsets(word, pos=pos)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace("_", " ")


def replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement."""
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])


@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
    # Get indices of verb tokens in sentence.
    verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    if verb_idxs:
        # Pick random verb idx to replace.
        idx = np.random.choice(verb_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="v")
        # If there's a valid verb synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
    # Get indices of noun tokens in sentence.
    noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if noun_idxs:
        # Pick random noun idx to replace.
        idx = np.random.choice(noun_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="n")
        # If there's a valid noun synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


@transformation_function(pre=[spacy])
def replace_adjective_with_synonym(x):
    # Get indices of adjective tokens in sentence.
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    if adjective_idxs:
        # Pick random adjective idx to replace.
        idx = np.random.choice(adjective_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="a")
        # If there's a valid adjective synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


# %%
tfs = [
    change_person,
    swap_adjectives,
    replace_verb_with_synonym,
    replace_noun_with_synonym,
    replace_adjective_with_synonym,
]

# %% [markdown]
# Let's check out a few examples of transformed data points to see what our TFs are doing.

# %%
from utils import preview_tfs

preview_tfs(df_train, tfs)

# %% [markdown]
# We notice a couple of things about the TFs.
#
# * Sometimes they make trivial changes (`"website"` to `"web site"` for replace_noun_with_synonym).
#   This can still be helpful for training our model, because it teaches the model to be invariant to such small changes.
# * Sometimes they introduce incorrect grammar to the sentence (e.g. `swap_adjectives` swapping `"young"` and `"more"` above).
#
# The TFs are expected to be heuristic strategies that indeed preserve the class most of the time, but
# [don't need to be perfect](https://arxiv.org/pdf/1901.11196.pdf).
# This is especially true when using automated
# [data augmentation techniques](https://snorkel.org/blog/tanda/)
# which can learn to avoid particularly corrupted data points.
# As we'll see below, Snorkel is compatible with such learned augmentation policies.

# %% [markdown]
# ## 3. Applying Transformation Functions

# %% [markdown]
# We'll first define a `Policy` to determine what sequence of TFs to apply to each data point.
# We'll start with a [`RandomPolicy`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/augmentation/snorkel.augmentation.RandomPolicy.html)
# that samples `sequence_length=2` TFs to apply uniformly at random per data point.
# The `n_per_original` argument determines how many augmented data points to generate per original data point.

# %%
from snorkel.augmentation import RandomPolicy

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True
)

# %% [markdown]
# In some cases, we can do better than uniform random sampling.
# We might have domain knowledge that some TFs should be applied more frequently than others,
# or have trained an [automated data augmentation model](https://snorkel.org/blog/tanda/)
# that learned a sampling distribution for the TFs.
# Snorkel supports this use case with a
# [`MeanFieldPolicy`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/augmentation/snorkel.augmentation.MeanFieldPolicy.html),
# which allows you to specify a sampling distribution for the TFs.
# We give higher probabilities to the `replace_[X]_with_synonym` TFs, since those provide more information to the model.

# %%
from snorkel.augmentation import MeanFieldPolicy

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=2,
    keep_original=True,
    p=[0.05, 0.05, 0.3, 0.3, 0.3],
)

# %% [markdown]
# To apply one or more TFs that we've written to a collection of data points according to our policy, we use a
# [`PandasTFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/augmentation/snorkel.augmentation.PandasTFApplier.html)
# because our data points are represented with a Pandas DataFrame.

# %% {"tags": ["md-exclude-output"]}
from snorkel.augmentation import PandasTFApplier

tf_applier = PandasTFApplier(tfs, mean_field_policy)
df_train_augmented = tf_applier.apply(df_train)
Y_train_augmented = df_train_augmented["label"].values

# %%
print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")

# %% [markdown]
# We have almost doubled our dataset using TFs!
# Note that despite `n_per_original` being set to 2, our dataset may not exactly triple in size,
# because sometimes TFs return `None` instead of a new data point
# (e.g. `change_person` when applied to a sentence with no persons).
# If you prefer to have exact proportions for your dataset, you can have TFs that can't perform a
# valid transformation return the original data point rather than `None` (as they do here).


# %% [markdown]
# ## 4. Training A Model
#
# Our final step is to use the augmented data to train a model. We train an LSTM (Long Short Term Memory) model, which is a very standard architecture for text processing tasks.

# %% [markdown] {"tags": ["md-exclude"]}
# The next cell makes Keras results reproducible. You can ignore it.

# %% {"tags": ["md-exclude"]}
import tensorflow as tf

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

tf.compat.v1.set_random_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# %% [markdown]
# Now we'll train our LSTM on both the original and augmented datasets to compare performance.

# %% {"tags": ["md-exclude-output"]}
from utils import featurize_df_tokens, get_keras_lstm

X_train = featurize_df_tokens(df_train)
X_train_augmented = featurize_df_tokens(df_train_augmented)
X_test = featurize_df_tokens(df_test)


def train_and_test(X_train, Y_train, X_test=X_test, Y_test=Y_test, num_buckets=30000):
    # Define a vanilla LSTM model with Keras
    lstm_model = get_keras_lstm(num_buckets)
    lstm_model.fit(X_train, Y_train, epochs=5, verbose=0)
    preds_test = lstm_model.predict(X_test)[:, 0] > 0.5
    return (preds_test == Y_test).mean()


acc_augmented = train_and_test(X_train_augmented, Y_train_augmented)
acc_original = train_and_test(X_train, Y_train)

# %%
print(f"Test Accuracy (original training data): {100 * acc_original:.1f}%")
print(f"Test Accuracy (augmented training data): {100 * acc_augmented:.1f}%")


# %% [markdown]
# So using the augmented dataset indeed improved our model!
# There is a lot more you can do with data augmentation, so try a few ideas
# out on your own!
