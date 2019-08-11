# %% [markdown]
# # Recommender Systems Tutorial
# In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to improve recommendations. We consider a setting similar to the [Netflix challenge](https://www.kaggle.com/netflix-inc/netflix-prize-data), but with books instead of movies. We have a set of users and books, and for each user we know the set of books they have interacted with (read or marked as to-read). We don't have the user's ratings for the read books, except in a small number of cases. We also have some text reviews written by users. Our goal is to predict whether a user will read and like any given book. We represent users using the set of books they have interacted with, and train a model to predict a `rating` given the set of books the user interacted with and a new book to be rated (a `rating` of 1 means the user will read and like the book). Note that [Recommender systems](https://en.wikipedia.org/wiki/Recommender_system) is a very well studied area with a wide variety of settings and approaches, and we just focus on one of them.
#
# We will use the [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) dataset, from
# "Item Recommendation on Monotonic Behavior Chains", RecSys'18 (Mengting Wan, Julian McAuley), and "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", ACL'19 (Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley).
# In this dataset, we have user interactions and reviews for Young Adult novels from the Goodreads website, along with metadata (like `title` and `authors`) for the novels.

# %%
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("recsys")

# %% [markdown]
# ## Loading Data

# %% [markdown]
# We start by running the `download_and_process_data` function. The function returns the `df_train`, `df_test`, `df_dev`, `df_val` dataframes, which correspond to our training, test, development, and validation sets. Each of those dataframes has the following fields:
# * `user_idx`: A unique identifier for a user.
# * `book_idx`: A unique identifier for a book that is being rated by the user.
# * `book_idxs`: The set of books that the user has interacted with (read or planned to read).
# * `review_text`: Optional text review written by the user for the book.
# * `rating`: Either `0` (which means the user did not read or did not like the book) or `1` (which means the user read and liked the book). The `rating` field is missing for `df_train`.
# Our objective is to predict whether a given user (represented by the set of book_idxs the user has interacted with) will read and like any given book. That is, we want to train a model that takes a set of `book_idxs` (the user) and a single `book_idx` (the book to rate) and predicts the `rating`.
#
# In addition, `download_and_process_data` also returns the `df_books` dataframe, which contains one row per book, along with metadata for that book (such as `title` and `first_author`).

# %%
from utils import download_and_process_data

(df_train, df_test, df_dev, df_val), df_books = download_and_process_data()

df_books.head()

# %% [markdown]
# We look at a sample of the labeled development set. As an example, we want our final recommendations model to be able to predict that a user who has interacted with `book_idxs` (25743, 22318, 7662, 6857, 83, 14495, 30664, ...) would either not read or not like the book with `book_idx` 22764 (first row), while a user who has interacted with `book_idxs` (3880, 18078, 9092, 29933, 1511, 8560, ...) would read and like the book with `book_idx` 3181 (second row).

# %%
df_dev.sample(frac=1, random_state=12).head()

# %% [markdown]
# ## Writing Labeling Functions

# %% [markdown]
# If a user has interacted with several books written by an author, there is a good chance that the user will read and like other books by the same author. We express this as a labeling function, using the `first_author` field in the `df_books` dataframe.

# %%
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

# %%
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_idx, df_books.first_author))
first_author_to_books_df = df_books.groupby("first_author")[["book_idx"]].agg(set)
first_author_to_books = dict(
    zip(first_author_to_books_df.index, first_author_to_books_df.book_idx)
)


@labeling_function()
def shared_first_author(x):
    author = book_to_first_author[x.book_idx]
    same_author_books = first_author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return POSITIVE if num_read > 15 else ABSTAIN


# %% [markdown]
# We can also leverage the long text reviews written by users to guess whether they liked or disliked a book. For example, the third df_dev entry above has a review with the text '4.5 STARS', which indicates that the user liked the book. We write a simple LF that looks for similar phrases to guess the user's rating of a book. We interpret >= 4 stars to indicate a positive rating, while < 4 stars is negative.

# %%
low_rating_strs = [
    "one star",
    "1 star",
    "two star",
    "2 star",
    "3 star",
    "three star",
    "3.5 star",
    "2.5 star",
    "1 out of 5",
    "2 out of 5",
    "3 out of 5",
]
high_rating_strs = ["5 stars", "five stars", "four stars", "4 stars", "4.5 stars"]


@labeling_function()
def stars_in_review(x):
    if not isinstance(x.review_text, str):
        return ABSTAIN
    for low_rating_str in low_rating_strs:
        if low_rating_str in x.review_text.lower():
            return NEGATIVE
    for high_rating_str in high_rating_strs:
        if high_rating_str in x.review_text.lower():
            return POSITIVE
    return ABSTAIN


# %% [markdown]
# We can also run [TextBlob](https://textblob.readthedocs.io/en/dev/index.html), a tool that provides a pretrained sentiment analyzer, on the reviews, and use its polarity and subjectivity scores to estimate the user's rating for the book.

# %%
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_polarity(x):
    if x.review_text:
        x.blob = TextBlob(str(x.review_text))
    else:
        x.blob = None
    return x


# Label high polarity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    if x.blob:
        if x.blob.polarity > 0.3:
            return POSITIVE
    return ABSTAIN


# Label high subjectivity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def subjectivity_positive(x):
    if x.blob:
        if x.blob.subjectivity > 0.75:
            return POSITIVE
    return ABSTAIN


# Label low polarity reviews as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    if x.blob:
        if x.blob.polarity < 0.0:
            return NEGATIVE
    return ABSTAIN


# %%
from snorkel.labeling import PandasLFApplier, LFAnalysis

lfs = [
    stars_in_review,
    shared_first_author,
    polarity_positive,
    subjectivity_positive,
    polarity_negative,
]

applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
LFAnalysis(L_dev, lfs).lf_summary(df_dev.rating)

# %% [markdown]
# ### Applying labeling functions to the training set
#
# We apply the labeling functions to the training set, and then filter out examples unlabeled by any LF, and combine the rest with the dev set to form our final training set.

# %%
from snorkel.labeling.model.label_model import LabelModel

# Train LabelModel.
L_train = applier.apply(df_train)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)
Y_train_preds = label_model.predict(L_train)

# %%
import pandas as pd
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, Y_train_preds_filtered = filter_unlabeled_dataframe(
    df_train, Y_train_preds, L_train
)
df_train_filtered["rating"] = Y_train_preds_filtered
combined_df_train = pd.concat([df_train_filtered, df_dev], axis=0)

# %% [markdown]
# ### Rating Prediction Model
# We write a Keras model for predicting ratings given a user's book list and a book (which is being rated). The model represents the list of books the user interacted with, `books_idxs`, by learning an embedding for each idx, and averaging the embeddings in `book_idxs`. It learns another embedding for the `book_idx`, the book to be rated. Then it concatenates the two embeddings and uses an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) to compute the probability of the `rating` being 1.

# %%
import numpy as np
import tensorflow as tf
from utils import precision, recall, f1

n_books = len(df_books)


# Keras model to predict rating given book_idxs and book_idx.
def get_model(embed_dim=64, hidden_layer_sizes=[32]):
    # Compute embedding for book_idxs.
    len_book_idxs = tf.keras.layers.Input([])
    book_idxs = tf.keras.layers.Input([None])
    book_idxs_emb = tf.keras.layers.Embedding(n_books, embed_dim)(book_idxs)
    book_idxs_emb = tf.math.divide(
        tf.keras.backend.sum(book_idxs_emb, axis=1), tf.expand_dims(len_book_idxs, 1)
    )
    # Compute embedding for book_idx.
    book_idx = tf.keras.layers.Input([])
    book_idx_emb = tf.keras.layers.Embedding(n_books, embed_dim)(book_idx)
    input_layer = tf.keras.layers.concatenate([book_idxs_emb, book_idx_emb], 1)
    # Build Multi Layer Perceptron on input layer.
    cur_layer = input_layer
    for size in hidden_layer_sizes:
        tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur_layer)
    output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(cur_layer)
    # Create and compile keras model.
    model = tf.keras.Model(
        inputs=[len_book_idxs, book_idxs, book_idx], outputs=[output_layer]
    )
    model.compile(
        "Adagrad", "binary_crossentropy", metrics=["accuracy", f1, precision, recall]
    )
    return model


# %% [markdown]
# We use triples of (`book_idxs`, `book_idx`, `rating`) from our dataframes as training examples. In addition, we want to train the model to recognize when a user will not read a book. To create examples for that, we randomly sample a `book_id` not in `book_idxs` and use that with a `rating` of 0 as a _random negative_ example. We create one such _random negative_ example for every positive (`rating` 1) example in our dataframe so that positive and negative examples are roughly balanced.

# %%
# Generator to turn dataframe into examples.
def get_examples_generator(df):
    def generator():
        for book_idxs, book_idx, rating in zip(df.book_idxs, df.book_idx, df.rating):
            # Remove book_idx from book_idxs so the model can't just look it up.
            book_idxs = tuple(filter(lambda x: x != book_idx, book_idxs))
            yield {
                "len_book_idxs": len(book_idxs),
                "book_idxs": book_idxs,
                "book_idx": book_idx,
                "label": rating,
            }
            if rating == 1:
                # Generate a random negative book_id not in book_idxs.
                random_negative = np.random.randint(0, n_books)
                while random_negative in book_idxs:
                    random_negative = np.random.randint(0, n_books)
                yield {
                    "len_book_idxs": len(book_idxs),
                    "book_idxs": book_idxs,
                    "book_idx": random_negative,
                    "label": 0,
                }

    return generator


def get_data_tensors(df):
    # Use generator to get examples each epoch, along with shuffling and batching.
    padded_shapes = {
        "len_book_idxs": [],
        "book_idxs": [None],
        "book_idx": [],
        "label": [],
    }
    dataset = (
        tf.data.Dataset.from_generator(
            get_examples_generator(df), {k: tf.int64 for k in padded_shapes}
        )
        .shuffle(123)
        .repeat(None)
        .padded_batch(batch_size=256, padded_shapes=padded_shapes)
    )
    tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    return (
        tensor_dict["len_book_idxs"],
        tensor_dict["book_idxs"],
        tensor_dict["book_idx"],
        tensor_dict["label"],
    )


# %% [markdown]
# We now train the model on our combined training data (data labeled by LFs plus dev data).
#
# %%
model = get_model()

train_data_tensors = get_data_tensors(combined_df_train)
val_data_tensors = get_data_tensors(df_val)
model.fit(
    train_data_tensors[:-1],
    train_data_tensors[-1],
    steps_per_epoch=300,
    validation_data=(val_data_tensors[:-1], val_data_tensors[-1]),
    validation_steps=40,
    epochs=30,
    verbose=1,
)
# %% [markdown]
# Finally, we evaluate the model's predicted ratings on our test data.
#
# %%
test_data_tensors = get_data_tensors(df_test)
model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=30)

# %% [markdown]
# ## Summary
#
# In this tutorial, we showed one way to use Snorkel for recommendations. We used books' metadata and review text to create `LF`s that estimate user ratings. We used a `LabelModel` to combine the outputs of those `LF`s. Finally, we trained a model to predict what books a user will read and like (and therefore what books should be recommended to the user) based only on what books the user has interacted with in the past.
#
# Here we demonstrated one way to use Snorkel for training a recommender system. Note, however, that this approach could easily be adapted to take advantage of additional information as it is available (e.g., user profile data, denser user ratings, and so on.)
