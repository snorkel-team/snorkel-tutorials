# %% [markdown]
# # Recommender Systems Tutorial
# In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to to improve recommendations.
# We will use the [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) dataset, from
# "Item Recommendation on Monotonic Behavior Chains", RecSys'18 (Mengting Wan, Julian McAuley), and "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", ACL'19, Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley. In this dataset, we have user ratings and reviews for Young Adult novels from the Goodreads website, along with metadata (like authors) for the novels. We consider the task of predicting whether a user will read and like any given book.

# %%
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("recsys")

# %% [markdown]
# ## Loading Data

# %% [markdown]
# We run the `download_data` function to download and preprocess the data. The function returns the `df_books` dataframe, which contains one row per book, along with metadata for that book. It also contains the `data_train`, `data_test`, `data_val`, `data_val` dataframes, which correspond to our training, test, development, and validation sets. Each of those dataframes has the following fields:
# * `user_idx`: A unique identifier for a user.
# * `book_idx`: A unique identifier for a book that is being rated by the user.
# * `book_idxs`: The set of books that the user has interacted with (read or planned to read).
# * `review_text`: Optional text review written by the user for the book.
# * `rating`: Either `0` (which means the user did not read or did not like the book) or `1` (which means the user read and liked the book). The `rating` field is missing for `data_train`.
# Our objective is to predict whether a given user (represented by the set of book_idxs the user has interacted with) will read and like any given book. That is, we want to train a model that takes a set of `book_idxs` and a `book_idx` as input and predicts the `rating`.

# %%
from data import download_and_process_data

df_books, (data_train, data_test, data_dev, data_val) = download_and_process_data()

data_dev.head()

# %% [markdown]
# If a user has interacted with several books written by an author, there is a good chance that the user will read and like other books by the same author. We express this as a labeling function, using the `first_author` field in the `df_books` dataframe.

# %%
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_idx, df_books.first_author))
first_author_to_books_df = df_books.groupby("first_author")[["book_idx"]].agg(set)
first_author_to_books = dict(
    zip(first_author_to_books_df.index, first_author_to_books_df.book_idx)
)


@labeling_function()
def common_first_author(x):
    author = book_to_first_author[x.book_idx]
    same_author_books = first_author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return 1 if num_read > 10 else -1


# %% [markdown]
# We can also leverage the long text reviews written by users to guess whether they liked or disliked a book. We run [TextBlob](https://textblob.readthedocs.io/en/dev/index.html), a tool that provides a pretrained sentiment analyzer, on the reviews, and use its polarity and subjectivity scores to estimate the user's rating for the book.

# %%
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor()
def textblob_polarity(x):
    if x.review_text:
        x.blob = TextBlob(str(x.review_text))
        x.start_blob = TextBlob(" ".join(x.blob.raw_sentences[:2]))
    else:
        x.blob = None
    return x


textblob_polarity.memoize = True

# Label high polarity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    if x.blob:
        if x.blob.polarity > 0.3:
            return 1
    return -1


# Label high subjectivity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def subjectivity_positive(x):
    if x.blob:
        if x.blob.subjectivity > 0.75:
            return 1
    return -1


# Label low polarity reviews as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    if x.blob:
        if x.blob.polarity < 0.0:
            return 0
    return -1


# %%
from snorkel.labeling import PandasLFApplier, LFAnalysis

lfs = [
    common_first_author,
    polarity_positive,
    subjectivity_positive,
    polarity_negative,
]

applier = PandasLFApplier(lfs)
L_dev = applier.apply(data_dev)
LFAnalysis(L_dev, lfs).lf_summary(data_dev.rating)

# %% [markdown]
# ### Applying labeling functions to the training set
#
# We apply the labeling functions to the training set, and then filter out examples unlabeled by any LF, and combine the rest with the dev set to form our final training set.

# %%
from snorkel.labeling.model.label_model import LabelModel

# Train LabelModel.
L_train = applier.apply(data_train)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)

from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds

Y_dev_prob = label_model.predict_proba(L_dev)
Y_dev_pred = probs_to_preds(Y_dev_prob)

acc = metric_score(data_dev.rating, Y_dev_pred, probs=None, metric="accuracy")
print(f"LabelModel Accuracy: {acc:.3f}")

Y_train_prob = label_model.predict_proba(L_train)
Y_train_preds = probs_to_preds(Y_train_prob)

# %%
import pandas as pd
from snorkel.labeling import filter_unlabeled_dataframe

data_train_filtered, Y_train_prob_filtered = filter_unlabeled_dataframe(
    data_train, Y_train_prob, L_train
)
Y_train_preds_filtered = probs_to_preds(Y_train_prob_filtered)
data_train_filtered["rating"] = Y_train_preds_filtered
combined_data_train = pd.concat([data_train_filtered, data_dev], axis=0)

# %% [markdown]
# ### Rating Prediction Model
# We write a Keras model for predicting ratings given a user's book list and a book (which is being rated). The model represents the list of books the user interacted with, `books_idxs`, by learning an embedding for each idx, and averaging the embeddings in `book_idxs`. It learns another embedding for the `book_idx`, the book to be rated. Then it concatenates the two embeddings and uses an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) to compute the probability of the `rating` being 1.

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=10, verbose=1, restore_best_weights=True
)
n_books = len(df_books)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_feedforward_model():
    num_b = tf.keras.layers.Input([], name="inp_num_b")
    bids = tf.keras.layers.Input([None], name="inp_bids")
    bid = tf.keras.layers.Input([], name="inp_bid")
    bi_emb_dim = 64
    b_emb_dim = 64
    layer_sizes = [32]
    bi_emb = tf.keras.layers.Embedding(n_books, bi_emb_dim, name="bi_emb")(bids)
    b_emb = tf.keras.layers.Embedding(n_books, b_emb_dim, name="b_emb")(bid)
    bi_emb_reduced = tf.math.divide(
        tf.keras.backend.sum(bi_emb, axis=1),
        tf.expand_dims(num_b, 1),
        name="bi_emb_reduced",
    )
    input_layer = tf.keras.layers.concatenate(
        [bi_emb_reduced, b_emb], 1, name="input_layer"
    )
    cur_layer = input_layer
    for size in layer_sizes:
        tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur_layer)
    output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(cur_layer)
    feedforward_model = tf.keras.Model(
        inputs=[num_b, bids, bid], outputs=[output_layer]
    )
    feedforward_model.compile(
        "Adagrad",
        "binary_crossentropy",
        metrics=["accuracy", f1_m, precision_m, recall_m],
    )
    return feedforward_model


padded_shapes = {"num_b": [], "bids": [None], "bid": [], "label": []}


def get_data_tensors(data):
    def generator():
        for bids, bid, rating in zip(data.book_idxs, data.book_idx, data.rating):
            if len(bids) <= 1:
                continue
            yield {"num_b": len(bids), "bids": bids, "bid": bid, "label": rating}
            if rating == 1:
                yield {
                    "num_b": len(bids),
                    "bids": bids,
                    "bid": np.random.randint(0, n_books),
                    "label": 0,
                }

    dataset = (
        tf.data.Dataset.from_generator(generator, {k: tf.int64 for k in padded_shapes})
        .shuffle(123)
        .repeat(None)
        .padded_batch(256, padded_shapes, drop_remainder=False)
    )
    tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    return (
        tensor_dict["num_b"],
        tensor_dict["bids"],
        tensor_dict["bid"],
        tensor_dict["label"],
    )


# %%
feedforward_model = get_feedforward_model()

train_data_tensors = get_data_tensors(combined_data_train)
val_data_tensors = get_data_tensors(data_val)
feedforward_model.fit(
    train_data_tensors[:-1],
    train_data_tensors[-1],
    steps_per_epoch=300,
    validation_data=(val_data_tensors[:-1], val_data_tensors[-1]),
    validation_steps=40,
    # callbacks=[early_stopping],
    epochs=50,
    verbose=1,
)

# %%
test_data_tensors = get_data_tensors(data_test)
feedforward_model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=30)
