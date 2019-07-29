# -*- coding: utf-8 -*-
# %% [markdown]
# # Snorkel Transformation Functions Tutorial

# %% [markdown]
# In this tutorial, we will walk through the process of using `Snorkel Transformation Functions (TFs)` to classify YouTube comments as `SPAM` or `HAM` (not spam).
# For an overview of Snorkel, visit [snorkel.org](http://snorkel.org).
# You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/).
#
# For our task, we have access to a some labeled data.
# We use **_Transformation Functions_**, to perform data augmentation to get additional training data.
#
# The tutorial is divided into three parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle.
# 2. **Writing Transformation Functions**: We write Transformation Functions (TFs) that can be applied to training examples to generate new training examples.
# 3. **Applying Transformation Functions**: We apply a sequence of TFs to each training data point, using a random policy, to generate an augmented training set.
# 4. **Training An End Model**: We use the augmented training set to train an LSTM model for classifying new comments as `SPAM` or `HAM`.

# %% [markdown]
# ### Task: Spam Detection

# %% [markdown]
# We use a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) that consists of YouTube comments from 5 videos. The task is to classify each comment as being
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

# %% [markdown]
# ### Data Splits in Snorkel
#
# We split our data into 3 sets:
# * **Training Set**: The largest split of the dataset. These are the examples used for training, and also the ones that transformation functions are applied on.
# * **Validation Set**: A labeled set used to tune hyperparameters and/or perform early stopping while training the classifier.
# * **Test Set**: A labeled set for final evaluation of our classifier. This set should only be used for final evaluation, _not_ tuning.

# %% [markdown]
# ## 1. Loading Data

# %% [markdown]
# We load the Kaggle dataset and create Pandas DataFrame objects for each of the sets described above.
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
# * The `dev` set is a random sample of 200 data points from the `train` set with gold labels added.
# * The fifth video is split 50/50 between a validation set (`valid`) and `test` set.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# %%
from utils import load_spam_dataset

df_train, _, df_valid, df_test = load_spam_dataset(delete_train_labels=False)

# We pull out the label vectors for ease of use later
Y_valid = df_valid["label"].values
Y_train = df_train["label"].values
Y_test = df_test["label"].values


# %%
df_train.head()

# %% [markdown]
# ## 2. Writing Transformation Functions
#
# Transformation Functions are functions that can be applied to a training example to create another valid training example. For example, for image classification problems, it is common to rotate or crop images in the training data to create new training inputs.
#
# Our task involves processing text. Some common ways to augment text includes replacing words with their synonyms, or replacing names entities with other entities. Applying these operations to a comment shouldn't change whether it is `SPAM` or not.
#
# Transformation functions in Snorkel are created with the `@transformation_function()` decorator, which wraps a function for taking a single data point and returning a transformed version of the data point.

# %% [markdown]
# We start with a transformation function that uses `wordnet` from [NLTK](https://www.nltk.org/) to replace words with their synonyms, and one that drops the last sentence of multi-sentence comments.

# %%
from snorkel.augmentation.tf import transformation_function
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")


def get_synonyms(word):
    synsets = wn.synsets(word)
    if not synsets:
        return []
    else:
        return [lemma.name() for lemma in synsets[0].lemmas()]


@transformation_function()
def replace_with_synonyms(x, replace_prob=0.05):
    new_words = []
    for word in x.text.split():
        if np.random.rand() < replace_prob:
            synonyms = get_synonyms(word)
            word = synonyms[0] if synonyms else word
        new_words.append(word)
    x.text = " ".join(new_words)
    return x


# Dropping the last sentence of a multi-sentence comment shouldn't change it's spam / ham nature.
@transformation_function()
def drop_last_sentence(x):
    sentences = x.text.split(".")
    if len(sentences) > 1:
        x.text = ". ".join(sentences[:-1])
        return x


# %% [markdown]
# ### Adding `pre` mappers.
# Some TFs rely on fields that aren't present in the raw data, but can be derived from it.
# We can enrich our data (providing more fields for the TFs to refer to) using map functions specified in the `pre` field of the transformation_function decorator (similar to `preprocessor` used for Labeling Functions).
#
# For example, we can use the fantastic NLP tool [spaCy](https://spacy.io/) to add lemmas, part-of-speech (pos) tags, etc. to each token.
# Snorkel provides a prebuilt preprocessor for spaCy called `SpacyPreprocessor` which adds a new field to the
# data point containing a [spaCy `Doc` object](https://spacy.io/api/doc).
# For more info, see the [`SpacyPreprocessor` documentation](https://snorkel.readthedocs.io/en/master/source/snorkel.labeling.preprocess.html#snorkel.labeling.preprocess.nlp.SpacyPreprocessor).
#

# %%
# Download the spaCy english model
# If you see an error in the next cell, restart the kernel
# ! python -m spacy download en_core_web_sm

# %%
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

# We use named entities recognized by spacy to replace them with different entities of the same type.
@transformation_function(pre=[spacy])
def change_person(x):
    persons = [str(ent) for ent in x.doc.ents if ent.label_ == "PERSON"]
    if persons:
        to_replace = np.random.choice(persons)
        replacement = np.random.choice(["Bob", "Alice"])
        x.text = x.text.replace(to_replace, replacement)
        return x


@transformation_function(pre=[spacy])
def change_date(x):
    dates = [str(ent) for ent in x.doc.ents if ent.label_ == "DATE"]
    if dates:
        to_replace = np.random.choice(dates)
        replacement = np.random.choice(["31st December", "01/03/99"])
        x.text = x.text.replace(to_replace, replacement)
        return x


# %% [markdown]
# ## 3. Applying Transformation Functions

# %% [markdown]
# To apply one or more TFs that we've written to a collection of data points, we use a `TFApplier`.
# Because our data points are represented with a Pandas DataFrame in this tutorial, we use the `PandasTFApplier` class. In addition, we can apply multiple TFs in a sequence to each example. A `policy` is used to determine what sequence of TFs to apply to each example. In this case, we just use a `RandomPolicy` that picks 3 TFs at random per example. The `n_per_original` argument determines how many augmented examples to generate per original example.
#

# %%
from snorkel.augmentation.apply import PandasTFApplier
from snorkel.augmentation.policy import RandomPolicy

tfs = [change_person, change_date, drop_last_sentence, replace_with_synonyms]

policy = RandomPolicy(len(tfs), sequence_length=3, n_per_original=2)
tf_applier = PandasTFApplier(tfs, policy)
df_train_augmented = tf_applier.apply(df_train).infer_objects()
Y_train_augmented = df_train_augmented["label"].values

# %%
print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")

# %% [markdown]
# We have more than doubled our dataset using TFs! Note that despite `n_per_original` being set to 2, our dataset does not exactly triple in size, because some TFs keep the example unchanged (e.g. `change_person` when applied to a sentence with no persons).

# %% [markdown]
# ## 4. Training an End Model
#
# Our final step is to use the augmented data to train a model. We train an LSTM (Long Short Term Memory) model, which is a commonly used architecture for text processing tasks.

# %%
import tensorflow as tf


def train_and_test(
    train_set, train_labels, num_buckets=30000, embed_dim=16, rnn_state_size=64
):
    def map_pad_or_truncate(string, max_length=30):
        ids = tf.keras.preprocessing.text.hashing_trick(
            string, n=num_buckets, hash_function="md5"
        )
        return ids[:max_length] + [0] * (max_length - len(ids))

    train_tokens = np.array(list(map(map_pad_or_truncate, train_set.text)))
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim))
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu))
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    lstm_model.compile("Adagrad", "binary_crossentropy", metrics=["accuracy"])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=10, verbose=1, restore_best_weights=True
    )

    valid_tokens = np.array(list(map(map_pad_or_truncate, df_valid.text)))

    lstm_model.fit(
        train_tokens,
        train_labels,
        epochs=50,
        validation_data=(valid_tokens, Y_valid),
        callbacks=[early_stopping],
        verbose=0,
    )

    test_tokens = np.array(list(map(map_pad_or_truncate, df_test.text)))
    test_probs = lstm_model.predict(test_tokens)
    return ((test_probs[:, 0] > 0.5) == (Y_test == 1)).mean()


print(
    f"Test Accuracy when training on original dataset: {train_and_test(df_train, Y_train)}"
)
print(
    f"Test Accuracy when training on augmented dataset: {train_and_test(df_train_augmented, Y_train_augmented)}"
)
