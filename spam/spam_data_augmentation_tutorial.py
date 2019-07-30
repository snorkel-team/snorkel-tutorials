# -*- coding: utf-8 -*-
# %% [markdown]
# # Snorkel Transformation Functions Tutorial

# %% [markdown]
# In this tutorial, we will walk through the process of using Snorkel Transformation Functions (TFs) to classify YouTube comments as `SPAM` or `HAM` (not spam). For more details on the task, check out the main labeling functions [tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/spam_tutorial.ipynb).
# For an overview of Snorkel, visit [snorkel.org](http://snorkel.org).
# You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/).
#
# For our task, we have access to some labeled YouTube comments for training. We generate additional data by transforming the labeled comments using **_Transformation Functions_**.
#
# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle.
# 2. **Writing Transformation Functions**: We write Transformation Functions (TFs) that can be applied to training examples to generate new training examples.
# 3. **Applying Transformation Functions**: We apply a sequence of TFs to each training data point, using a random policy, to generate an augmented training set.
# 4. **Training An End Model**: We use the augmented training set to train an LSTM model for classifying new comments as `SPAM` or `HAM`.

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
# The two main columns in the DataFrames are:
# * **`text`**: Raw text content of the comment
# * **`label`**: Whether the comment is `SPAM` (1) or `HAM` (0).
#
# For more details, check out the labeling functions [tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/spam_tutorial.ipynb).

# %%
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# %%
from utils import load_spam_dataset

df_train, _, df_valid, df_test = load_spam_dataset(load_train_labels=False)

# We pull out the label vectors for ease of use later
Y_valid = df_valid["label"].values
Y_train = df_train["label"].values
Y_test = df_test["label"].values


# %%
df_train.head()

# %% [markdown]
# ## 2. Writing Transformation Functions
#
# Transformation Functions are functions that can be applied to a training example to create another valid training example. For example, for image classification problems, it is common to rotate or crop images in the training data to create new training inputs. Transformation functions should be atomic e.g. a small rotation of an image, or changing a single word in a sentence. We then compose multiple transformation functions when applying them to training examples.
#
# Our task involves processing text. Some [common](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28) [ways](https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610) to augment text includes replacing words with their synonyms, or replacing names entities with other entities. Applying these operations to a comment shouldn't change whether it is `SPAM` or not.
#
# Transformation functions in Snorkel are created with the `@transformation_function()` decorator, which wraps a function that takes in a single data point and returns a transformed version of the data point. If no transformation is possible, the function should return `None`.

# %% [markdown]
# ### Adding `pre` mappers.
# Some TFs rely on fields that aren't present in the raw data, but can be derived from it.
# We can enrich our data (providing more fields for the TFs to refer to) using map functions specified in the `pre` field of the transformation_function decorator (similar to `preprocessor` used for Labeling Functions).
#
# For example, we can use the fantastic NLP tool [spaCy](https://spacy.io/) to add lemmas, part-of-speech (pos) tags, etc. to each token.
# Snorkel provides a prebuilt preprocessor for spaCy called `SpacyPreprocessor` which adds a new field to the
# data point containing a [spaCy `Doc` object](https://spacy.io/api/doc). It uses memoization internally, so it will not reparse the text after applying each TF unless the text's hash changes.
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

# %%
import names
import numpy as np
from snorkel.augmentation.tf import transformation_function

# TFs for replacing a random named entity with a different entity of the same type.
@transformation_function(pre=[spacy])
def change_person(x):
    persons = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]
    if persons:
        to_replace = np.random.choice(persons)
        replacement_name = names.get_full_name()
        x.text = x.text.replace(to_replace, replacement_name)
        return x


# Swap two nouns at random.
@transformation_function(pre=[spacy])
def swap_nouns(x):
    words = [token.text for token in x.doc]
    idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if len(idxs) < 3:
        return None
    idx1, idx2 = sorted(np.random.choice(idxs[:-1], 2))
    x.text = " ".join(
        words[:idx1]
        + [words[idx2]]
        + words[1 + idx1 : idx2]
        + [words[idx1]]
        + words[1 + idx2 :]
    )
    return x


# %% [markdown]
# We add some transformation functions that use `wordnet` from [NLTK](https://www.nltk.org/) to replace different parts of speech with their synonyms.

# %%
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")


def get_synonym(word, pos=None):
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return None
    else:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        return words[0] if words else word


@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
    words = [token.text for token in x.doc]
    idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    if len(idxs) > 1:
        to_replace = np.random.choice(idxs[:-1])
        synonym = get_synonym(words[to_replace], pos="v")
        if synonym and synonym != words[to_replace]:
            x.text = " ".join(words[:to_replace] + [synonym] + words[1 + to_replace :])
            return x


@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
    words = [token.text for token in x.doc]
    idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if len(idxs) > 1:
        to_replace = np.random.choice(idxs[:-1])
        synonym = get_synonym(words[to_replace], pos="n")
        if synonym and synonym != words[to_replace]:
            x.text = " ".join(words[:to_replace] + [synonym] + words[1 + to_replace :])
            return x


@transformation_function(pre=[spacy])
def replace_adjective_with_synonym(x):
    words = [token.text for token in x.doc]
    idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    if len(idxs) > 1:
        to_replace = np.random.choice(idxs[:-1])
        synonym = get_synonym(words[to_replace], pos="a")
        if synonym and synonym != words[to_replace]:
            x.text = " ".join(words[:to_replace] + [synonym] + words[1 + to_replace :])
            return x


# %% [markdown]
# We can try running the TFs on our training data to demonstrate their effect.

# %%
tfs = [
    change_person,
    swap_nouns,
    replace_verb_with_synonym,
    replace_noun_with_synonym,
    replace_adjective_with_synonym,
]

for tf in tfs:
    for i, row in df_train.iterrows():
        transformed_or_none = tf(row)
        if transformed_or_none is not None:
            print(f"TF Name: {tf.name}")
            print(f"Original Text: {row.text}")
            print(f"Transformed Text: {transformed_or_none.text}\n")
            break

# %% [markdown]
# ## 3. Applying Transformation Functions

# %% [markdown]
# To apply one or more TFs that we've written to a collection of data points, we use a `TFApplier`.
# Because our data points are represented with a Pandas DataFrame in this tutorial, we use the `PandasTFApplier` class. In addition, we can apply multiple TFs in a sequence to each example. A `policy` is used to determine what sequence of TFs to apply to each example. In this case, we just use a `RandomPolicy` that picks 3 TFs at random per example. The `n_per_original` argument determines how many augmented examples to generate per original example.
#

# %%
from snorkel.augmentation.apply import PandasTFApplier
from snorkel.augmentation.policy import RandomPolicy

policy = RandomPolicy(len(tfs), sequence_length=2, n_per_original=2, keep_original=True)
tf_applier = PandasTFApplier(tfs, policy)
df_train_augmented = tf_applier.apply(df_train).infer_objects()
Y_train_augmented = df_train_augmented["label"].values

# %%
print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")

# %% [markdown]
# We have almost doubled our dataset using TFs! Note that despite `n_per_original` being set to 2, our dataset may not exactly triple in size, because some TFs keep the example unchanged (e.g. `change_person` when applied to a sentence with no persons).

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
    test_preds = test_probs[:, 0] > 0.5
    return (test_preds == Y_test).mean()


test_accuracy_original = train_and_test(df_train, Y_train)
test_accuracy_augmented = train_and_test(df_train_augmented, Y_train_augmented)

print(f"Test Accuracy when training on original dataset: {test_accuracy_original}")
print(f"Test Accuracy when training on augmented dataset: {test_accuracy_augmented}")
