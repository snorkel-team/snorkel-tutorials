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

if not os.path.exists('data'):
    os.mkdir('data')
    os.chdir('data')
    # Books
    gdown.download('https://drive.google.com/uc?id=1H6xUV48D5sa2uSF_BusW-IBJ7PCQZTS1', output=None, quiet=None)
    # Interactions
    gdown.download('https://drive.google.com/uc?id=17G5_MeSWuhYnD4fGJMvKRSOlBqCCimxJ', output=None, quiet=None)
    # Reviews
    gdown.download('https://drive.google.com/uc?id=1FVD3LxJXRc5GrKm97LehLgVGbRfF9TyO', output=None, quiet=None)
    os.chdir('..')

# %%
import calendar
import collections
import gzip
import json
import numpy as np
import pandas as pd
from datetime import datetime

month_to_int = dict((v,k) for k,v in enumerate(calendar.month_abbr))

def get_timestamp(date_str):
    _, month, day, _, _, year = date_str.split()
    dt = datetime(year=int(year), month=month_to_int[month], day=int(day))
    return datetime.timestamp(dt)

def load_data(file_name, max_to_load = 100):
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

#book_suffix = '_poetry'
book_suffix = '_comics_graphic'

books = load_data(f"data/goodreads_books{book_suffix}.json.gz", None)
df_books = pd.DataFrame(books)
# Make book_id column contiguous from 0 to len(df_books) - 1.
book_to_idx = {bid: i for i, bid in enumerate(df_books.book_id.unique())}
df_books.book_id = df_books.book_id.map(book_to_idx)
# Turns author role dict into list of <= 5 authors for simplicity.
df_books.authors = df_books.authors.map(lambda l: [pair['author_id'] for pair in l[:5]])
df_books['first_author'] = df_books.authors.map(lambda l: l[0])
n_books = len(book_to_idx)

interactions = load_data(f"data/goodreads_interactions{book_suffix}.json.gz", None)
df_interactions = pd.DataFrame(interactions)
df_interactions.book_id = df_interactions.book_id.map(book_to_idx)
# Turn timestamp string into unix timestamp.
df_interactions['timestamp'] = df_interactions.date_updated.map(get_timestamp)
# Make user_id column continguous from 0 to num_users - 1.
user_to_idx = {uid: i for i, uid in enumerate(df_interactions.user_id.unique())}
df_interactions.user_id = df_interactions.user_id.map(user_to_idx)
n_users = len(user_to_idx)
# Map ratings to binary
ratings_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
df_interactions_nz = df_interactions[df_interactions.rating != 0]
df_interactions_z = df_interactions[df_interactions.rating == 0]
df_interactions.rating = df_interactions.rating.map(ratings_map)
df_interactions_nz.rating = df_interactions_nz.rating.map(ratings_map)
df_interactions_z.rating = -1

reviews = load_data(f"data/goodreads_reviews{book_suffix}.json.gz", None)
df_reviews = pd.DataFrame(reviews)
df_reviews['timestamp'] = df_reviews.date_updated.map(get_timestamp)
df_reviews.book_id = df_reviews.book_id.map(book_to_idx)
df_reviews.user_id = df_reviews.user_id.map(user_to_idx)

print(f"{len(user_to_idx)} users")
print(f"{len(book_to_idx)} books")
print(f"{len(interactions)} interactions")
print(f"{len(reviews)} reviews")

from sklearn.model_selection import train_test_split

# Input data for FeedForward model
user_to_books = (df_interactions.groupby('user_id')['book_id'].apply(tuple)
                 .reset_index().rename(columns={'book_id': 'book_ids'}))
data = user_to_books.merge(df_interactions_nz, on='user_id', how='inner')[['book_ids', 'book_id', 'rating']]
data_train, data_test = train_test_split(data, test_size=0.1)
data_train, data_dev = train_test_split(data_train, test_size=0.11111)
data_train, data_val = train_test_split(data_train, test_size=0.125)

# %% [markdown]
# ### FeedForward Model
# This model learns two embeddings per book. It takes an input a set of book ratings, and a new book, and tries to predict its rating. (So it models a user as a set of book-rating pairs, instead of with a separate embedding).

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_feedforward_model():
    num_b = tf.keras.layers.Input([], name='inp_num_b')
    bids = tf.keras.layers.Input([None], name='inp_bids')
    bid = tf.keras.layers.Input([], name='inp_bid')
    bi_emb_dim = 128
    b_emb_dim = 64
    layer_sizes = [40, 20, 10]
    bi_emb = tf.keras.layers.Embedding(n_books, bi_emb_dim, name='bi_emb')(bids)
    b_emb = tf.keras.layers.Embedding(n_books, b_emb_dim, name='b_emb')(bid)
    bi_emb_reduced = tf.math.divide(tf.keras.backend.sum(bi_emb, axis=1), tf.expand_dims(num_b, 1), name='bi_emb_reduced')
    input_layer = tf.keras.layers.concatenate([bi_emb_reduced, b_emb], 1, name='input_layer')
    cur_layer = input_layer
    for size in layer_sizes:
        tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur_layer)
    output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(cur_layer)
    feedforward_model = tf.keras.Model(inputs=[num_b, bids, bid], outputs=[output_layer])
    feedforward_model.compile('Adagrad', 'binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    return feedforward_model

padded_shapes = {'num_b': [], 'bids': [None], 'bid': [], 'label': []}

def get_data_tensors(data):
    def generator():
        for bids, bid, rating in zip(data.book_ids, data.book_id, data.rating):
            if len(bids) <= 1:
                continue
            yield {'num_b': len(bids), 'bids': bids, 'bid': bid, 'label': rating}
            yield {'num_b': len(bids), 'bids': bids, 'bid': np.random.randint(0, n_books), 'label': 0}
    dataset = (
        tf.data.Dataset.from_generator(generator, {k: tf.int64 for k in padded_shapes})
        .shuffle(123)
        .repeat(None)
        .padded_batch(256, padded_shapes, drop_remainder=False)
    )
    tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    return (tensor_dict["num_b"], tensor_dict["bids"], tensor_dict["bid"], tensor_dict["label"])



# %%
#feedforward_model = get_feedforward_model()

train_data_tensors = get_data_tensors(data_train)
val_data_tensors = get_data_tensors(data_val)
feedforward_model.fit(
    train_data_tensors[:-1], train_data_tensors[-1], steps_per_epoch=300,
    validation_data=(val_data_tensors[:-1], val_data_tensors[-1]), validation_steps=30,
    callbacks=[early_stopping], epochs=20, verbose=1)

test_data_tensors = get_data_tensors(data_test)
feedforward_model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=30)

# %%
feedforward_model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=30)
#probs = feedforward_model.predict(test_data_tensors[:-1], steps=30)
#probs.reshape([256, 30])

# %% [markdown]
# A lot of books are part of a series (e.g. Naruto, The Sandman). The book titles are often of the form "<book_name> (series_name #number)". To capture the series, we use a regex to find non-numeric strings in between a '(' and '#'.

# %%
# TODO: This only works for comics
from snorkel.labeling.lf import labeling_function
import re

series_to_books = collections.defaultdict(set)
for book_id, title in zip(df_books.book_id, df_books.title):
    match = re.match('.*\(([^0-9]*)#', title)
    if match:
        series_to_books[match.group(1)].add(book_id)
# Only keep 'series' with at least 10 entries, as they are more likely to be an actual book series.
series_to_books = {k:v for k, v in series_to_books.items() if len(v) > 10}
print(list(series_to_books.keys())[:5])
# Create reverse lookup dictionary.
book_to_series = {book_id: series for series, book_set in series_to_books.items() for book_id in book_set}

@labeling_function(resources={"book_to_series": book_to_series, 'series_to_books': series_to_books})
def many_same_series(x, book_to_series, series_to_books):
    # Abstain if book is not part of a series.
    if x.book_id not in book_to_series:
        return -1
    same_series_books = series_to_books[book_to_series[x.book_id]]
    num_read = len(set(x.book_ids).intersection(same_series_books))
    return 1 if num_read > 1 else -1



# %%
book_to_first_author = dict(zip(df_books.book_id, df_books.first_author))
user_author_ratings = data_train.merge(df_books[['book_id', 'first_author']], on='book_id')
user_author_mean_ratings = user_author_ratings[['user_id', 'first_author', 'rating']].groupby(['user_id', 'first_author']).agg('mean').reset_index()
user_author_ratings_dict = dict(zip(zip(user_author_mean_ratings.user_id, user_author_mean_ratings.first_author), user_author_mean_ratings.rating))

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
lfs = [many_same_series]#, common_first_author]# + shelf_lfs

applier = PandasLFApplier(lfs)
L_dev = applier.apply(data_dev)
LFAnalysis(L_dev, lfs).lf_summary(data_dev.rating)

# %%
{k: v / float(len(data_dev)) for k, v in Counter(data_dev.rating).items()}

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

acc = metric_score(data_dev.rating, Y_dev_pred, probs=None, metric="accuracy")
print(f"LabelModel Accuracy: {acc:.3f}")

Y_train_prob = label_model.predict_proba(L_train)
Y_train_preds = probs_to_preds(Y_train_prob)

# %%
# Create new training examples using LF.
L_train = applier.apply(df_interactions_z)
new_idxs = np.where(L_train.max(1) != -1)[0]
new_data_train = df_interactions_z.iloc[new_idxs]
new_data_train.rating = L_train[new_idxs]
combined_data_train = pd.concat([data_train, new_data_train], axis=0)
