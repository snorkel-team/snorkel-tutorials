# %% [markdown]
# # Recommender Systems Tutorial
# In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to to improve recommendations.
# We will use the [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) dataset.
#
# Citations: (Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.  [bibtex]
# Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19).

# %% [markdown]
# ## TODOS
#
# ### Story
# Imagine we have some reviews (but without ratings), and to-read shelves (0 ratings). We use the reviews, and the shelves, to generate labels. We model each user as bag of to-read books. We want to predict which of those books the user will rate as 5.
#
# ### Data
# Map labels to binary (5 is 1, else is 0). Could also try 4, 5 to 1 and else 0.
#
# Leave ratings only for dev and test. Delete rest (or drop and use only the 0 rating pairs).
#
# ### Model:
# Bag of to-read books, plus book, use to predict binary rating.
#
# ### LFs:
# textblob on review. same_author, same_series on to-read books (assume if the user really wants to read an author / series, he will rate it highly).

# %% [markdown]
# ## Loading Goodreads Dataset

# %%
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("recsys")

# %%
# TODO: Move downloading and processing code to data.py
import gdown

if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    # Books
    gdown.download(
        "https://drive.google.com/uc?id=1gH7dG4yQzZykTpbHYsrw2nFknjUm0Mol",
        output=None,
        quiet=None,
    )
    # Interactions
    gdown.download(
        "https://drive.google.com/uc?id=1NNX7SWcKahezLFNyiW88QFPAqOAYP5qg",
        output=None,
        quiet=None,
    )
    # Reviews
    gdown.download(
        "https://drive.google.com/uc?id=1M5iqCZ8a7rZRtsmY5KQ5rYnP9S0bQJVo",
        output=None,
        quiet=None,
    )
    os.chdir("..")

# %%
import calendar
import collections
import gzip
import json
import numpy as np
import pandas as pd
from datetime import datetime

# %%
month_to_int = dict((v, k) for k, v in enumerate(calendar.month_abbr))


def get_timestamp(date_str):
    _, month, day, _, _, year = date_str.split()
    dt = datetime(year=int(year), month=month_to_int[month], day=int(day))
    return datetime.timestamp(dt)


def load_data(file_name, max_to_load=100, filter_dict=None):
    count = 0
    data = []
    filter_dict = filter_dict or {}
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            for k, v in filter_dict.items():
                if d[k] not in v:
                    break
            else:
                count += 1
                data.append(d)
                if (max_to_load is not None) and (count >= max_to_load):
                    break
    return data


# %%
# book_suffix = '_poetry'
book_suffix = "_young_adult"

books = load_data(f"data/goodreads_books{book_suffix}.json.gz", None)
df_books = pd.DataFrame(books)

df_books = df_books[
    [
        "authors",
        "average_rating",
        "book_id",
        "country_code",
        "description",
        "is_ebook",
        "language_code",
        "popular_shelves",
        "publication_year",
        "ratings_count",
        "series",
        "similar_books",
        "text_reviews_count",
        "title",
        "title_without_series",
        "work_id",
    ]
]

df_books = df_books.astype(
    dict(
        average_rating=float,
        book_id=int,
        is_ebook=bool,
        ratings_count=int,
        text_reviews_count=int,
        work_id=int,
    )
)

# Turns author role dict into list of <= 5 authors for simplicity.
df_books.authors = df_books.authors.map(lambda l: [pair["author_id"] for pair in l[:5]])
df_books["first_author"] = df_books.authors.map(lambda l: int(l[0]))

# %%
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

ratings = df_books.ratings_count.values

print(np.percentile(ratings, [10, 50, 90, 95, 99]).astype(int))

plt.hist(ratings.clip(0, 200), bins=100)
plt.show()

# %%
min_ratings = 100
max_ratings = 15000

df_books_filt = df_books[
    (df_books.ratings_count >= min_ratings) & (df_books.ratings_count <= max_ratings)
]

book_id_to_idx = {v: i for i, v in enumerate(df_books_filt.book_id)}
df_books_filt["book_idx"] = df_books_filt.book_id.map(lambda bid: book_id_to_idx[bid])

print(len(df_books), "books ->", len(df_books_filt), "books")
df_books_filt.head()

# %%
filt_book_ids = set(map(str, book_id_to_idx.keys()))
interactions = load_data(f"data/goodreads_interactions{book_suffix}.json.gz", 5000000, dict(book_id=filt_book_ids))
df_interactions = pd.DataFrame(interactions)

df_interactions = df_interactions[
    ["book_id", "is_read", "rating", "review_id", "user_id"]
]

df_interactions = df_interactions.astype(dict(book_id=int, is_read=bool, rating=int))

df_interactions["book_idx"] = df_interactions.book_id.map(
    lambda bid: book_id_to_idx[bid]
)

df_interactions.head()

# %%
user_counts = df_interactions.groupby(["user_id"]).size().values

print(np.percentile(user_counts, [10, 50, 75, 90, 95, 99]).astype(int))

plt.hist(user_counts.clip(0, 200), bins=100)
plt.show()

# %%
user_counts = df_interactions.groupby(["user_id"]).size()
users_filt = user_counts[(user_counts >= 25) & (user_counts <= 200)].index
user_id_to_idx = {v: i for i, v in enumerate(users_filt)}

len(user_counts), len(user_id_to_idx)

# %%
df_interactions_filt = df_interactions[
    df_interactions.user_id.isin(set(user_id_to_idx.keys()))
]
df_interactions_filt["user_idx"] = df_interactions_filt.user_id.map(
    lambda uid: user_id_to_idx[uid]
)

print(len(df_interactions), len(df_interactions_filt))

df_interactions_filt.head()

# %%
df_interactions_nz = df_interactions_filt[df_interactions_filt.rating != 0]
df_interactions_z = df_interactions_filt[df_interactions_filt.rating == 0]

ratings_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
df_interactions_nz["rating_4_5"] = df_interactions_nz.rating.map(
    lambda r: ratings_map[r]
)

ratings_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
df_interactions_nz["rating_5"] = df_interactions_nz.rating.map(lambda r: ratings_map[r])

# %%
reviews = load_data(f"data/goodreads_reviews{book_suffix}.json.gz", None, dict(book_id=filt_book_ids, user_id=set(user_id_to_idx.keys())))
df_reviews = pd.DataFrame(reviews)
df_reviews["book_idx"] = df_reviews.book_id.astype('int').map(book_id_to_idx)
df_reviews["user_idx"] = df_reviews.user_id.map(user_id_to_idx)

# %%
# del df_books
# del df_interactions

print(f"{len(user_id_to_idx)} users")
print(f"{len(book_id_to_idx)} books")
print(f"{len(df_interactions_nz)} non-zero interactions")
print(f"{len(df_interactions_z)} zero interactions")
print(f"{len(df_reviews)} reviews")

n_books = len(book_id_to_idx)

# %%
from sklearn.model_selection import train_test_split

# Input data for FeedForward model
user_to_books = (
    df_interactions_filt.groupby("user_idx")["book_idx"]
    .apply(tuple)
    .reset_index()
    .rename(columns={"book_idx": "book_idxs"})
)
data = user_to_books.merge(df_interactions_nz, on="user_idx", how="inner")[
    ["user_idx", "book_idxs", "book_idx", "rating_4_5"]
].merge(
    df_reviews[['user_idx', 'book_idx', 'review_text']], on=['user_idx', 'book_idx'], how='left'
)
user_idxs = list(user_id_to_idx.values())
user_idxs_train, user_idxs_test = train_test_split(user_idxs, test_size=0.1)
user_idxs_train, user_idxs_dev = train_test_split(user_idxs_train, test_size=0.11111)
user_idxs_train, user_idxs_val = train_test_split(user_idxs_train, test_size=0.125)

data_train = data[data.user_idx.isin(set(user_idxs_train))]
data_test = data[data.user_idx.isin(set(user_idxs_test))]
data_dev = data[data.user_idx.isin(set(user_idxs_dev))]
data_val = data[data.user_idx.isin(set(user_idxs_val))]

# %%
data_train.rating_4_5.mean()

# %% [markdown]
# ### FeedForward Model
# This model learns two embeddings per book. It takes an input a set of book ratings, and a new book, and tries to predict its rating. (So it models a user as a set of book-rating pairs, instead of with a separate embedding).

# %% [markdown]
# A lot of books are part of a series (e.g. Naruto, The Sandman). The book titles are often of the form "<book_name> (series_name #number)". To capture the series, we use a regex to find non-numeric strings in between a '(' and '#'.

# %%
# TODO: This only works for comics
from snorkel.labeling.lf import labeling_function
import re

series_to_books = collections.defaultdict(set)
for book_idx, title in zip(df_books_filt.book_idx, df_books_filt.title):
    match = re.match(".*\(([^0-9]*)#", title)
    if match:
        series_to_books[match.group(1)].add(book_idx)
# Only keep 'series' with at least 10 entries, as they are more likely to be an actual book series.
series_to_books = {k: v for k, v in series_to_books.items() if len(v) > 10}
print(list(series_to_books.keys())[:5])
# Create reverse lookup dictionary.
book_to_series = {
    book_idx: series
    for series, book_set in series_to_books.items()
    for book_idx in book_set
}


@labeling_function(
    resources={"book_to_series": book_to_series, "series_to_books": series_to_books}
)
def many_same_series(x, book_to_series, series_to_books):
    # Abstain if book is not part of a series.
    if x.book_idx not in book_to_series:
        return -1
    same_series_books = series_to_books[book_to_series[x.book_idx]]
    num_read = len(set(x.book_idxs).intersection(same_series_books))
    return 1 if num_read > 4 else -1


# %%
book_to_first_author = dict(zip(df_books_filt.book_idx, df_books_filt.first_author))
author_to_bookd_df = df_books_filt.groupby('first_author')[['book_idx']].agg(set)
author_to_books = dict(zip(author_to_bookd_df.index, author_to_bookd_df.book_idx))

@labeling_function()
def common_first_author(x):
    author = book_to_first_author[x.book_idx]
    same_author_books = author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return 1 if num_read > 6 else -1


# %%
from snorkel.preprocess import preprocessor
from textblob import TextBlob

@preprocessor()
def textblob_polarity(x):
    if x.review_text:
        x.blob = TextBlob(str(x.review_text))
        x.start_blob = TextBlob(' '.join(x.blob.raw_sentences[:2]))
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

# Label reviews with high polarity starting sentences as positive.
@labeling_function(pre=[textblob_polarity])
def starting_polarity_positive(x):
    if x.blob:
        if x.start_blob.polarity > 0.5:
            return 1
    return -1

# Label reviews with low polarity starting sentences as negative.
@labeling_function(pre=[textblob_polarity])
def starting_polarity_negative(x):
    if x.blob:
        if x.start_blob.polarity < -0.2:
            return 0
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

lfs = [common_first_author, starting_polarity_positive, starting_polarity_negative, polarity_positive, subjectivity_positive, polarity_negative]

applier = PandasLFApplier(lfs)
L_dev = applier.apply(data_dev)
LFAnalysis(L_dev, lfs).lf_summary(data_dev.rating_4_5)

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

acc = metric_score(data_dev.rating_4_5, Y_dev_pred, probs=None, metric="accuracy")
print(f"LabelModel Accuracy: {acc:.3f}")

Y_train_prob = label_model.predict_proba(L_train)
Y_train_preds = probs_to_preds(Y_train_prob)

# %%
# Create new training examples using LF.
from snorkel.labeling import filter_unlabeled_dataframe
data_train_filtered, Y_train_prob_filtered = filter_unlabeled_dataframe(data_train, Y_train_prob, L_train)
Y_train_preds_filtered = probs_to_preds(Y_train_prob_filtered)
data_train_filtered['rating_4_5'] = Y_train_preds_filtered
combined_data_train = pd.concat([data_train_filtered, data_dev], axis=0)

# %%
import tensorflow as tf
from tensorflow.keras import backend as K

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=10, verbose=1, restore_best_weights=True
)


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
        for bids, bid, rating in zip(data.book_idxs, data.book_idx, data.rating_4_5):
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
    validation_steps=30,
    # callbacks=[early_stopping],
    epochs=40,
    verbose=1,
)

# %%
test_data_tensors = get_data_tensors(data_test)
feedforward_model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=30)
