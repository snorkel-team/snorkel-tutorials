# -*- coding: utf-8 -*-
# %% [markdown]
# # Snorkel Transformation Functions Tutorial

# %% [markdown]
# In this tutorial, we will walk through the process of using Snorkel Transformation Functions (TFs) to classify YouTube comments as `SPAM` or `HAM` (not spam). For more details on the task, check out the main labeling functions [tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb).
# For an overview of Snorkel, visit [snorkel.org](http://snorkel.org).
# You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/).
#
# For our task, we have access to some labeled YouTube comments for training. We generate additional data by transforming the labeled comments using **_Transformation Functions_**.
#
# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle.
# 2. **Writing Transformation Functions**: We write Transformation Functions (TFs) that can be applied to training examples to generate new training examples.
# 3. **Applying Transformation Functions**: We apply a sequence of TFs to each training data point, using a random policy, to generate an augmented training set.
# 4. **Training A Model**: We use the augmented training set to train an LSTM model for classifying new comments as `SPAM` or `HAM`.

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
# For more details, check out the labeling functions [tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb).

# %%
import numpy as np
import os
import random

# For reproducibility
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
random.seed(0)

# Turn off TensorFlow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# %%
from utils import load_spam_dataset

df_train, _, df_valid, df_test = load_spam_dataset(load_train_labels=True)

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
# For more info, see the [`SpacyPreprocessor` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.nlp.SpacyPreprocessor.html#snorkel.preprocess.nlp.SpacyPreprocessor).
#

# %%
# Download the spaCy english model
# If you see an error in the next cell, restart the kernel
# ! python -m spacy download en_core_web_sm

# %%
from snorkel.preprocess.nlp import SpacyPreprocessor

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

# %%
import names
from snorkel.augmentation import transformation_function

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

# %%
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


# %% [markdown]
# We can try running the TFs on our training data to demonstrate their effect.

# %%
import pandas as pd
from collections import OrderedDict

# Prevent truncating displayed sentences.
pd.set_option("display.max_colwidth", 0)
tfs = [
    change_person,
    swap_adjectives,
    replace_verb_with_synonym,
    replace_noun_with_synonym,
    replace_adjective_with_synonym,
]

transformed_examples = []
for tf in tfs:
    for i, row in df_train.sample(frac=1, random_state=2).iterrows():
        transformed_or_none = tf(row)
        # If TF returned a transformed example, record it in dict and move to next TF.
        if transformed_or_none is not None:
            transformed_examples.append(
                OrderedDict(
                    {
                        "TF Name": tf.name,
                        "Original Text": row.text,
                        "Transformed Text": transformed_or_none.text,
                    }
                )
            )
            break
pd.DataFrame(transformed_examples)

# %% [markdown]
# We notice a couple of things about the TFs.
# * Sometimes they make trivial changes (_website_ -> _web site_ for replace_noun_with_synonym). This can still be helpful for training our model, because it teaches the model that these variations have similar meanings.
# * Sometimes they make the sentence less meaningful (e.g. swapping _young_ and _more_ for swap_adjectives).
#
# Data augmentation can be tricky for text inputs, so we expect most TFs to be a little flawed. But these TFs can  be useful despite the flaws; see [this paper](https://arxiv.org/pdf/1901.11196.pdf) for gains resulting from similar TFs.

# %% [markdown]
# ## 3. Applying Transformation Functions

# %% [markdown]
# To apply one or more TFs that we've written to a collection of data points, we use a `TFApplier`.
# Because our data points are represented with a Pandas DataFrame in this tutorial, we use the `PandasTFApplier` class. In addition, we can apply multiple TFs in a sequence to each example. A `policy` is used to determine what sequence of TFs to apply to each example. In this case, we just use a `MeanFieldPolicy` that picks 2 TFs at random per example, with probabilities given by `p`. We give higher probabilities to the replace_X_with_synonym TFs, since those provide more information to the model. The `n_per_original` argument determines how many augmented examples to generate per original example.

# %%
from snorkel.augmentation import MeanFieldPolicy, PandasTFApplier

policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=2,
    keep_original=True,
    p=[0.05, 0.05, 0.3, 0.3, 0.3],
)
tf_applier = PandasTFApplier(tfs, policy)
df_train_augmented = tf_applier.apply(df_train)
Y_train_augmented = df_train_augmented["label"].values

# %%
print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")

# %% [markdown]
# We have almost doubled our dataset using TFs! Note that despite `n_per_original` being set to 2, our dataset may not exactly triple in size, because sometimes TFs return `None` instead of a new example (e.g. `change_person` when applied to a sentence with no persons).

# %% [markdown]
# ## 4. Training A Model
#
# Our final step is to use the augmented data to train a model. We train an LSTM (Long Short Term Memory) model, which is a very standard architecture for text processing tasks.
#
# The next cell makes keras results reproducible. You can ignore it.

# %%
import tensorflow as tf

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

tf.compat.v1.set_random_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# %% [markdown]
# Next, we add some boilerplate code for creating an LSTM model.

# %%
def get_lstm_model(num_buckets, embed_dim=16, rnn_state_size=64):
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim))
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu))
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    lstm_model.compile("Adagrad", "binary_crossentropy", metrics=["accuracy"])
    return lstm_model


# %%
def train_and_test(train_set, Y_train, num_buckets=30000):
    def map_pad_or_truncate(string, max_length=30):
        """Tokenize text, pad or truncate to get max_length, and hash tokens."""
        ids = tf.keras.preprocessing.text.hashing_trick(
            string, n=num_buckets, hash_function="md5"
        )
        return ids[:max_length] + [0] * (max_length - len(ids))

    # Tokenize training text and convert words to integers.
    X_train = np.array(list(map(map_pad_or_truncate, train_set.text)))
    lstm_model = get_lstm_model(num_buckets)

    # Tokenize validation set text and convert words to integers.
    X_valid = np.array(list(map(map_pad_or_truncate, df_valid.text)))

    lstm_model.fit(
        X_train,
        Y_train,
        epochs=25,
        validation_data=(X_valid, Y_valid),
        # Set up early stopping based on val set accuracy.
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_acc", patience=5, verbose=1, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    # Tokenize validation set text and convert words to integers.
    X_test = np.array(list(map(map_pad_or_truncate, df_test.text)))
    probs_test = lstm_model.predict(X_test)
    preds_test = probs_test[:, 0] > 0.5
    return (preds_test == Y_test).mean()


test_accuracy_augmented = train_and_test(df_train_augmented, Y_train_augmented)
test_accuracy_original = train_and_test(df_train, Y_train)

print(f"Test Accuracy when training on original dataset: {test_accuracy_original}")
print(f"Test Accuracy when training on augmented dataset: {test_accuracy_augmented}")
