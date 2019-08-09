# %% [markdown]
# # Recommender Systems Tutorial
# In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to to improve recommendations.
# We will use the [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) dataset.
#
# Citations: (Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.  [bibtex]
# Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19).

# %% [markdown]
# ## Dataset Overview
# There are three dataframes
# * **Books**: One row per book, about 36000 rows total. Each row has a book_id, and book data including title, description, authors, popular shelves.
# * **Interactions**: One row per user-book pair (sparse). Each row includes a user_id, book_id, and rating. Rating can be 0 to 5. Rating of 1 to 5 means the user read the book and gave it that rating. Rating of 0 means the user had some interaction with the book (like adding it to the 'to-read' shelf) but did not finish reading it (and hence didn't rate it). There are 2.7M interactions, of which half are rating 0.
# * **Reviews**: One row per user-book pair (sparse). The user-book pairs here are a strict subset of those in the interactions data. The row includes user_id, book_id, and a detailed text review. There are about 150K detailed reviews.

# %% [markdown]
# ## Thoughts
#
# ### Task:
# We can do one of two tasks:
# * **Retrieval**: Given a user's previous ratings, predict new books (from the 36000) that the user will read / rate.
# * **Ranking**: Given a user's previous ratings, and a new user-book pair, predict the rating (from 1 to 5) that the user will give the book.
#
# For retrieval, we don't want to 'predict' or recommend all things a user has read. In fact, a rating of 1 or 2 (even 3) is worse than not having read a book. So if we're doing retrieval, we could just use 4s and 5s for prediction, and maybe use 1s and 2s as negatives.
# Ranking seems to be more straightforward, since we can keep 10% of the current interactions as test data, and predict ratings on those pairs. We still have to decide whether to equally punish all prediction errors (5->4 vs. 5->1). Current options are (i) Just measure accuracy, thus treating all errors as equal (ii) Cluster (4,5) and (1,2,3) and make it a binary problem. (Though 4 + 5 would be like 74% of the total ratings, so the problem is unbalanced). (iii) Compute square loss on test set. For training, if predictions are discrete, so square loss is not differentiable, so we'll still need something like softmax.
#
# ### Model Architecture:
# SVD doesn't work well. Mean square is not a good loss when the range is so limited (1 to 5). Instead we use softmax (final dense layer of size 5).
# There are two high level ways of setting up the input-output of the problem.
# * **FeedBack**: Model maps (user_id, book_id) pair to rating. So we have one user embedding per user, one book embedding per book, and the model combines these embeddings to produce rating predictions.
# * **Feedforward** Model maps (set of book ratings, book_id) pair to rating. i.e. now, each user is modeled as a set of past book ratings, instead of having a learned user embedding. This is especially helpful for users who have very little data (out of 380K users, over 280K have 3 or fewer interactions).
#
# The latter is referred to as feedforward because the user embedding is computed on the fly as a feedforward network on the past book ratings.
#
# Both models take really long to train, so I'm not sure I trained to convergence, but they seem to do about equally well. The feedback model seems easier to use alongside labeling functions (since LFs just have to create (user_id, book_id, rating) triples for new (user_id, book_id) pairs, e.g. those with current rating 0). An LF may
# also be more beenficial to users with very few interactions in this model.
#
# ### Data split:
# For the feedforward model, each user is like a single data point. We can randomly split users into train, test, dev sets.
# For the Feedback model, each interaction is a single data point. Plus, we need some data for each user and each books, since we have to learn their embedding. So the data split is on the interactions dataframe.
#
# We could use interaction date (added or updated) for splits (instead of splitting randomly) to make it both more challenging and more realistic. The task then becomes, given past data, predict future interactions. To ensure good coverage per user, maybe we have a separate threshold date per user (using percentile).
#
# ### Metadata for LFS:
# Two kinds of LFS.
# * Those based on book-book similarity (using e.g. author), that use the user's score for similar books to get score for a book.
# * Those based on a user-book relation e.g. a review or shelf. Those use the review words / shelf to guess how a user might rate the book.
#
# #### Book Authors
# Users often read books by same author. Unfortunately, there are also a look of duplicate books, which also end up having the same author, but only one copy of each set should be read by a user.
# Tried an LF where a users rating for a book is predicted to be equal to the user's rating for a book by the same author (abstain if there is no other read book by same author). That LF has a Dev accuracy of 0.58. As a comparison, the model after a lot of training gets accuracy of 0.55, while the MajorityLabelVoter baseline (which always predicts rating 5) gets an accuracy of 0.4.
# Tried on the comic book dataset, where 4s and 5s are more balanced (37% each), and the author LF has 62% accuracy. Unfortunately, the system also hangs occasionally on this dataset, due to size.
#
# #### Average rating, Number of ratings, Text Reviews Count
# Seems useful, but usable directly by the end model. So we can just it there instead of in LFs.
#
# #### Series
# The same series seems to correspond to the same book (maybe different editions). They often have the same title as well. A user should ideally never read or rate two of these, so this doesn't seem useful for an LF.
#
# #### Similar Books
# Not useful for this dataset. It turns out all similar books are from a different genre than this dataset (which is poetry), so no two books in this dataset are marked as similar. There are 150 books that are similar for more than one of our books, so we can use that to get some transitive similarity, but even then it'll be a small number.
#
# #### Book description, Title, Title without series
# TODO: Think about how to use.
#
# #### Language Code
# This is useful for candidate generation, but not for ranking (people will read books of the same language,
# but may not give it the same rating).
#
# #### Book shelves
# Tried building one LF per shelf, which copies the user's mean rating from previously read books from the same shelf. Doesn't do that well (0.45 accuracy on dev set). For comic books, it does better, with 0.485.
#
# #### Book reviews
# TODO: Free form text. Think about how to use.

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
    os.mkdir("data")
    os.chdir("data")
    # Books
    gdown.download(
        "https://drive.google.com/uc?id=1H6xUV48D5sa2uSF_BusW-IBJ7PCQZTS1",
        output=None,
        quiet=None,
    )
    # Interactions
    gdown.download(
        "https://drive.google.com/uc?id=17G5_MeSWuhYnD4fGJMvKRSOlBqCCimxJ",
        output=None,
        quiet=None,
    )
    # Reviews
    gdown.download(
        "https://drive.google.com/uc?id=1FVD3LxJXRc5GrKm97LehLgVGbRfF9TyO",
        output=None,
        quiet=None,
    )
else:
    os.chdir("data")

# %%
import calendar
import collections
import gzip
import json
import numpy as np
import pandas as pd
from datetime import datetime

month_to_int = dict((v, k) for k, v in enumerate(calendar.month_abbr))


def get_timestamp(date_str):
    _, month, day, _, _, year = date_str.split()
    dt = datetime(year=int(year), month=month_to_int[month], day=int(day))
    return datetime.timestamp(dt)


def load_data(file_name, max_to_load=100):
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)
            if (max_to_load is not None) and (count >= max_to_load):
                break
    return data


# book_suffix = '_poetry'
book_suffix = "_comics_graphic"

books = load_data(f"goodreads_books{book_suffix}.json.gz", None)
df_books = pd.DataFrame(books)
# Make book_id column contiguous from 0 to len(df_books) - 1.
book_to_idx = {bid: i for i, bid in enumerate(df_books.book_id.unique())}
df_books.book_id = df_books.book_id.map(book_to_idx)
# Turns author role dict into list of <= 5 authors for simplicity.
df_books.authors = df_books.authors.map(lambda l: [pair["author_id"] for pair in l[:5]])
df_books["first_author"] = df_books.authors.map(lambda l: l[0])
n_books = len(book_to_idx)

interactions = load_data(f"goodreads_interactions{book_suffix}.json.gz", None)
df_interactions = pd.DataFrame(interactions)
df_interactions.book_id = df_interactions.book_id.map(book_to_idx)
# Turn timestamp string into unix timestamp.
df_interactions["timestamp"] = df_interactions.date_updated.map(get_timestamp)
# Make user_id column continguous from 0 to num_users - 1.
user_to_idx = {uid: i for i, uid in enumerate(df_interactions.user_id.unique())}
df_interactions.user_id = df_interactions.user_id.map(user_to_idx)
n_users = len(user_to_idx)

df_interactions_nz = df_interactions[df_interactions.rating != 0]
df_interactions_z = df_interactions[df_interactions.rating == 0]

reviews = load_data(f"goodreads_reviews{book_suffix}.json.gz", None)
df_reviews = pd.DataFrame(reviews)
df_reviews["timestamp"] = df_reviews.date_updated.map(get_timestamp)
df_reviews.book_id = df_reviews.book_id.map(book_to_idx)
df_reviews.user_id = df_reviews.user_id.map(user_to_idx)

user_ctr = collections.Counter(df_interactions.user_id)
book_ctr = collections.Counter(df_interactions.book_id)

print(f"{len(user_to_idx)} users")
print(f"{len(book_to_idx)} books")
print(f"{len(interactions)} interactions")
print(f"{len(reviews)} reviews")

# Split data for FeedBack model
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

csr_interactions = csr_matrix(
    (
        df_interactions.rating.astype("float"),
        (df_interactions.user_id, df_interactions.book_id),
    ),
    shape=(len(user_to_idx), len(book_to_idx)),
)
# Currently using random split. TODO: Split based on time.
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_interactions_nz, test_size=0.1)
df_train, df_dev = train_test_split(df_train, test_size=0.11111)
df_train, df_val = train_test_split(df_train, test_size=0.125)

# Split data for FeedForward model
user_to_books = []
groupby_user = df_interactions_nz.groupby("user_id")
for uid in range(n_users):
    if uid in groupby_user.groups:
        user_df = groupby_user.get_group(uid)[["book_id", "rating"]]
        user_to_books.append(list(zip(user_df.book_id, user_df.rating)))
    else:
        user_to_books.append([])
data = np.array(user_to_books)
data_train, data_test = train_test_split(data, test_size=0.1)
data_train, data_dev = train_test_split(data_train, test_size=0.11111)
data_train, data_val = train_test_split(data_train, test_size=0.125)


# %%
def flatten_column(df, id_vars, column_to_flatten):
    return (
        pd.concat([df[id_vars], df[column_to_flatten].apply(pd.Series)], axis=1)
        .melt(id_vars=id_vars, value_name=column_to_flatten)
        .dropna()
        .drop("variable", axis=1)
    )


# book_by_author = flatten_column(df=df_books, id_vars=['book_id', 'title'] , column_to_flatten='author').sort_values('author')
# book_by_series = flatten_column(df=df_books, id_vars=['book_id', 'title'] , column_to_flatten='series').sort_values('series')

# Create dictionary mapping (user, author) to rating for author based LF.
joined_df = df_train.merge(df_books, on="book_id")

# Group by book attributes by user and look for commonalities.
# e.g. Do books by same author get same rating from a user?

gbu = joined_df[["user_id", "rating", "title", "num_pages"]].groupby("user_id")
user_dfs = []
for user_id in gbu.groups:
    user_df = gbu.get_group(user_id)
    if len(user_df) < 3:
        continue
    user_dfs.append(user_df.drop("user_id", axis=1))
    if len(user_dfs) > 25:
        break


# %% [markdown]
# A lot of books are part of a series (e.g. Naruto, The Sandman). The book titles are often of the form "<book_name> (series_name #number)". To capture the series, we use a regex to find non-numeric strings in between a '(' and '#'.

# %%
import re

# Dictionary mapping user book pairs to ratings
user_book_ratings_dict = dict(
    zip(zip(df_train.user_id, df_train.book_id), df_train.rating)
)

series_to_books = collections.defaultdict(list)
for book_id, title in zip(df_books.book_id, df_books.title):
    match = re.match(".*\(([^0-9]*)#", title)
    if match:
        series_to_books[match.group(1)].append(book_id)
# Only keep 'series' with at least 10 entries, as they are more likely to be an actual book series.
series_to_books = {k: v for k, v in series_to_books.items() if len(v) > 10}
print(list(series_to_books.keys())[:5])
# Create reverse lookup dictionary.
book_to_series = {
    book_id: series
    for series, book_list in series_dict.items()
    for book_id in book_list
}


@labeling_function(
    resources={"book_to_series": book_to_series, "series_to_books": series_to_books}
)
def same_series(x, book_to_series, series_to_books):
    # Abstain if book is not part of a series.
    if x.book_id not in book_to_series:
        return -1
    same_series_books = series_to_books[book_to_series[x.book_id]]
    ratings = [
        user_book_ratings_dict[(x.user_id, book_id)]
        for book_id in same_series_books
        if (x.user_id, book_id) in user_book_ratings_dict
    ]
    return int(np.around(np.array(ratings).mean())) if ratings else -1


# %%
shelves_count = collections.defaultdict(int)
shelves_sum = collections.defaultdict(int)
for shelves in df_books.popular_shelves:
    for d in shelves:
        name = d["name"]
        shelves_count[name] += 1
        shelves_sum[name] += int(d["count"])

shelves_and_counts = sorted(shelves_count.items(), key=lambda x: -x[1])
shelves_and_sums = sorted(shelves_sum.items(), key=lambda x: -x[1])
shelves_and_sums


# %% [markdown]
# As you can see, the first few shelves (like 'poetry' or 'to-read') are not very informative about the books (because all books are poetry). The shelves much lower in the list have very low count. For shelves in between, that provide some information about the book type and occur reasonably often, we can create one LF per shelf, which looks for a user's ratings for other books with the same popular shelf, and uses those to predict a rating.


# %%
from snorkel.labeling.lf import LabelingFunction

df_books.shelves_set = df_books.popular_shelves.map(
    lambda x: set([pair["name"] for pair in x])
)
shelf_rating_dicts = {}
for shelf, _ in shelves_and_sums[5:20]:
    shelf_df = df_books[["book_id"]][df_books.shelves_set.apply(lambda x: shelf in x)]
    shelf_ratings = df_train.merge(shelf_df, on="book_id", how="inner")
    # Select most common rating per user.
    shelf_ratings_mode = shelf_ratings[["user_id", "rating"]].groupby("user_id").mean()
    ratings_dict = dict(zip(shelf_ratings_mode.index, shelf_ratings_mode.rating))
    shelf_rating_dicts[shelf] = ratings_dict


def f(x, ratings_dict):
    return int(np.ceil(ratings_dict.get(x.user_id, -1)))


def get_shelf_lf(shelf):
    name = f"shelf_{shelf}"
    return LabelingFunction(
        name, f=f, resources={"ratings_dict": shelf_rating_dicts[shelf]}
    )


shelf_lfs = [get_shelf_lf(shelf) for shelf in shelf_rating_dicts]


# %%
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_id, df_books.first_author))
user_author_ratings = df_train.merge(
    df_books[["book_id", "first_author"]], on="book_id"
)
user_author_mean_ratings = (
    user_author_ratings[["user_id", "first_author", "rating"]]
    .groupby(["user_id", "first_author"])
    .agg("mean")
    .reset_index()
)
user_author_ratings_dict = dict(
    zip(
        zip(user_author_mean_ratings.user_id, user_author_mean_ratings.first_author),
        user_author_mean_ratings.rating,
    )
)


@labeling_function()
def common_first_author(x):
    author = book_to_first_author[x.book_id]
    if (x.user_id, author) in user_author_ratings_dict:
        return int(np.around(user_author_ratings_dict[(x.user_id, author)]))
    return -1


# %%
from snorkel.labeling.apply import PandasLFApplier
from snorkel.labeling.analysis import LFAnalysis

# same_first_author is slower but more accuarate than common_author (0.68)
# The difference is we take mean and np.around in same_author but pick random entry in common_author.
lfs = [common_first_author, same_series]  # + shelf_lfs

applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
LFAnalysis(L_dev, lfs).lf_summary(df_dev.rating)

# %%
{k: v / float(len(df_dev)) for k, v in Counter(df_dev.rating).items()}

# %%
from snorkel.labeling.model.label_model import LabelModel

# Train LabelModel.
L_train = applier.apply(df_interactions_z)
label_model = LabelModel(cardinality=6, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)

from snorkel.analysis.metrics import metric_score
from snorkel.analysis.utils import probs_to_preds

Y_dev_prob = label_model.predict_proba(L_dev)
Y_dev_pred = probs_to_preds(Y_dev_prob)

acc = metric_score(df_dev.rating, Y_dev_pred, probs=None, metric="accuracy")
print(f"LabelModel Accuracy: {acc:.3f}")

Y_train_prob = label_model.predict_proba(L_train)
Y_train_preds = probs_to_preds(Y_train_prob)

# %%
# Create new training examples using LF.
L_train = applier.apply(df_interactions_z)
new_idxs = np.where(L_train.max(1) != -1)[0]
new_df_train = df_interactions_z.iloc[new_idxs]
new_df_train.rating = L_train[new_idxs]
combined_df_train = pd.concat([df_train, new_df_train], axis=0)

# %% [markdown]
# ### FeedForward Model
# This model learns two embeddings per book. It takes an input a set of book ratings, and a new book, and tries to predict its rating. (So it models a user as a set of book-rating pairs, instead of with a separate embedding).

# %%
import tensorflow as tf

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=10, verbose=1, restore_best_weights=True
)


def get_feedforward_model():
    num_b = tf.keras.layers.Input([], name="inp_num_b")
    bids = tf.keras.layers.Input([None], name="inp_bids")
    ratings = tf.keras.layers.Input([None], name="inp_ratings")
    bid = tf.keras.layers.Input([], name="inp_bid")
    bi_emb_dim = 32
    b_emb_dim = 10
    layer_sizes = [40, 20, 10]
    r_emb = tf.keras.layers.Embedding(6, bi_emb_dim, name="r_emb")(
        ratings
    )  # 6 because ratings are 1 to 5.
    bi_emb = tf.keras.layers.Embedding(n_books, bi_emb_dim, name="bi_emb")(bids)
    b_emb = tf.keras.layers.Embedding(n_books, b_emb_dim, name="b_emb")(bid)
    bir_emb = tf.keras.layers.multiply([r_emb, bi_emb], name="bir_emb")
    bir_emb_reduced = tf.math.divide(
        tf.keras.backend.sum(bir_emb, axis=1),
        tf.expand_dims(num_b, 1),
        name="bir_emb_reduced",
    )
    #### Same as other model.
    input_layer = tf.keras.layers.concatenate(
        [bir_emb_reduced, b_emb], 1, name="input_layer"
    )
    cur_layer = input_layer
    for size in layer_sizes:
        tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur_layer)
    output_layer = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(cur_layer)
    feedforward_model = tf.keras.Model(
        inputs=[num_b, bids, ratings, bid], outputs=[output_layer]
    )
    feedforward_model.compile(
        "Adagrad", "sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return feedforward_model


padded_shapes = {"num_b": [], "bids": [None], "ratings": [None], "bid": [], "label": []}


def parse_row(row, max_len=40):
    np.random.shuffle(row)
    bid, rating = row[-1]
    row = row[:-1][:max_len]
    bids, ratings = zip(*row)
    return {
        "num_b": len(row),
        "bids": bids,
        "ratings": ratings,
        "bid": bid,
        "label": rating - 1,
    }


def get_data_tensors(data_array):
    def generator():
        for row in data_array:
            if len(row) <= 1:
                continue
            yield parse_row(row)

    dataset = (
        tf.data.Dataset.from_generator(generator, {k: tf.int64 for k in padded_shapes})
        .shuffle(123)
        .repeat(None)
        .padded_batch(32, padded_shapes, drop_remainder=False)
    )
    tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    return (
        tensor_dict["num_b"],
        tensor_dict["bids"],
        tensor_dict["ratings"],
        tensor_dict["bid"],
        tensor_dict["label"],
    )


# %%
feedforward_model = get_feedforward_model()

train_data_tensors = get_data_tensors(data_train)
val_data_tensors = get_data_tensors(data_val)
feedforward_model.fit(
    train_data_tensors[:-1],
    train_data_tensors[-1],
    steps_per_epoch=10000,
    validation_data=(val_data_tensors[:-1], val_data_tensors[-1]),
    validation_steps=1000,
    callbacks=[early_stopping],
    epochs=1,
    verbose=1,
)

test_data_tensors = get_data_tensors(data_test)
feedforward_model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=1000)

# %%
feedforward_model.fit(
    train_data_tensors[:-1],
    train_data_tensors[-1],
    steps_per_epoch=10000,
    validation_data=(val_data_tensors[:-1], val_data_tensors[-1]),
    validation_steps=1000,
    callbacks=[early_stopping],
    epochs=1,
    verbose=1,
)
feedforward_model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=1000)


# %% [markdown]
# ### FeedBack Model
# This model learns one embedding per user and one embedding per book. To predict rating, it concatenates the embeddings and passes them through a few hidden fully connected layers and computes a softmax probability distribution over the 5 ratings.

# %%
def get_feedback_model():
    uid = tf.keras.layers.Input([])
    bid = tf.keras.layers.Input([])
    layer_sizes = [40, 20, 10]
    u_emb_dim = 10
    b_emb_dim = 10
    u_emb = tf.keras.layers.Embedding(n_users, u_emb_dim)(uid)
    b_emb = tf.keras.layers.Embedding(n_books, b_emb_dim)(bid)
    input_layer = tf.keras.layers.concatenate([u_emb, b_emb], 1)
    cur_layer = input_layer
    for size in layer_sizes:
        tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur_layer)
    output_layer = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(cur_layer)
    feedback_model = tf.keras.Model(inputs=[uid, bid], outputs=[output_layer])
    feedback_model.compile(
        "Adagrad", "sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return feedback_model


# %%
# Learn on only train data
# feedback_model = get_feedback_model()
feedback_model.fit(
    [df_train.user_id.values, df_train.book_id.values],
    df_train.rating - 1,
    validation_data=((df_val.user_id.values, df_val.book_id.values), df_val.rating - 1),
    callbacks=[early_stopping],
    epochs=20,
    verbose=1,
)
base_score = feedback_model.evaluate(
    [df_test.user_id.values, df_test.book_id.values], df_test.rating - 1
)

# %%
# Learn on combined data
# feedback_model_combined = get_feedback_model()
feedback_model_combined.fit(
    [combined_df_train.user_id.values, combined_df_train.book_id.values],
    combined_df_train.rating - 1,
    validation_data=((df_val.user_id.values, df_val.book_id.values), df_val.rating - 1),
    callbacks=[early_stopping],
    epochs=30,
    verbose=1,
)
lf_score = feedback_model_combined.evaluate(
    [df_test.user_id.values, df_test.book_id.values], df_test.rating - 1
)

# %%
from snorkel.analysis.utils import preds_to_probs

Y_train_probs = preds_to_probs(df_train.rating - 1, 5)

# %%
a = collections.Counter(df_test.rating)
{rating: num / float(sum(a.values())) for rating, num in a.items()}

# %% [markdown]
# ### SVD
# This does really badly.

# %%

csr_interactions = csr_matrix(
    (
        df_interactions.rating.map({1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0}),
        (df_interactions.user_id, df_interactions.book_id),
    ),
    shape=(len(user_to_idx), len(book_to_idx)),
)

U, s, V = svds(csr_interactions, k=64)
Vt = V.T
Us = U * s


def get_prediction(uid, bid):
    u_emb = Us[uid]
    b_emb = Vt[bid]
    return np.dot(u_emb, b_emb)


# Note: This sucks at predictions, as expected.


# %%
for i, (u, b, r) in enumerate(
    zip(df_interactions.user_id, df_interactions.book_id, df_interactions.rating)
):
    pred = get_prediction(u, b)
    print(f"{pred}, {r}")
    if i > 10:
        break
