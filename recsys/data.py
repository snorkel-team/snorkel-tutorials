import calendar
import collections
import gdown
import gzip
import json
import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


def maybe_download_files():
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


def get_timestamp(date_str):
    month_to_int = dict((v, k) for k, v in enumerate(calendar.month_abbr))
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


def process_books_data():
    books = load_data(f"data/goodreads_books_young_adult.json.gz", None)
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
            "ratings_count",
            "similar_books",
            "text_reviews_count",
            "title",
        ]
    ]
    df_books = df_books.astype(
        dict(
            average_rating=float,
            book_id=int,
            is_ebook=bool,
            ratings_count=int,
            text_reviews_count=int,
        )
    )
    # Turns author role dict into list of <= 5 authors for simplicity.
    df_books.authors = df_books.authors.map(
        lambda l: [pair["author_id"] for pair in l[:5]]
    )
    df_books["first_author"] = df_books.authors.map(lambda l: int(l[0]))
    min_ratings = 100
    max_ratings = 15000

    df_books = df_books[
        (df_books.ratings_count >= min_ratings)
        & (df_books.ratings_count <= max_ratings)
    ]

    book_id_to_idx = {v: i for i, v in enumerate(df_books.book_id)}
    df_books["book_idx"] = df_books.book_id.map(lambda bid: book_id_to_idx[bid])
    return df_books, book_id_to_idx


def process_interactions_data(book_id_to_idx):
    interactions = load_data(
        f"data/goodreads_interactions_young_adult.json.gz",
        5000000,
        dict(book_id=set(map(str, book_id_to_idx.keys()))),
    )
    df_interactions = pd.DataFrame(interactions)
    df_interactions = df_interactions[
        ["book_id", "is_read", "rating", "review_id", "user_id"]
    ]
    df_interactions = df_interactions.astype(
        dict(book_id=int, is_read=bool, rating=int)
    )
    df_interactions["book_idx"] = df_interactions.book_id.map(book_id_to_idx)
    user_counts = df_interactions.groupby(["user_id"]).size()
    users_filt = user_counts[(user_counts >= 25) & (user_counts <= 200)].index
    user_id_to_idx = {v: i for i, v in enumerate(users_filt)}
    df_interactions = df_interactions[
        df_interactions.user_id.isin(set(user_id_to_idx.keys()))
    ]
    df_interactions["user_idx"] = df_interactions.user_id.map(user_id_to_idx)
    return df_interactions, user_id_to_idx


def process_reviews_data(book_id_to_idx, user_id_to_idx):
    reviews = load_data(
        f"data/goodreads_reviews_young_adult.json.gz",
        None,
        dict(
            book_id=set(map(str, book_id_to_idx.keys())),
            user_id=set(user_id_to_idx.keys()),
        ),
    )
    df_reviews = pd.DataFrame(reviews)
    df_reviews["book_idx"] = df_reviews.book_id.astype("int").map(book_id_to_idx)
    df_reviews["user_idx"] = df_reviews.user_id.map(user_id_to_idx)
    return df_reviews


def split_data(user_idxs, data):
    user_idxs_train, user_idxs_test = train_test_split(user_idxs, test_size=0.1)
    user_idxs_train, user_idxs_dev = train_test_split(user_idxs_train, test_size=0.111)
    user_idxs_train, user_idxs_val = train_test_split(user_idxs_train, test_size=0.125)

    data_train = data[data.user_idx.isin(set(user_idxs_train))].drop("rating", axis=1)
    data_test = data[data.user_idx.isin(set(user_idxs_test))]
    data_dev = data[data.user_idx.isin(set(user_idxs_dev))]
    data_val = data[data.user_idx.isin(set(user_idxs_val))]
    return data_train, data_test, data_dev, data_val


def download_and_process_data():
    maybe_download_files()
    df_books, book_id_to_idx = process_books_data()

    df_interactions, user_id_to_idx = process_interactions_data(book_id_to_idx)
    df_interactions_nz = df_interactions[df_interactions.rating != 0]
    df_interactions_z = df_interactions[df_interactions.rating == 0]
    ratings_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    df_interactions_nz["rating_4_5"] = df_interactions_nz.rating.map(
        lambda r: ratings_map[r]
    )

    df_reviews = process_reviews_data(book_id_to_idx, user_id_to_idx)

    # Compute book_idxs for each user.
    user_to_books = (
        df_interactions.groupby("user_idx")["book_idx"]
        .apply(tuple)
        .reset_index()
        .rename(columns={"book_idx": "book_idxs"})
    )
    data = user_to_books.merge(df_interactions_nz, on="user_idx", how="inner")[
        ["user_idx", "book_idxs", "book_idx", "rating_4_5"]
    ].merge(
        df_reviews[["user_idx", "book_idx", "review_text"]],
        on=["user_idx", "book_idx"],
        how="left",
    )
    data = data.rename(columns={"rating_4_5": "rating"})
    user_idxs = list(user_id_to_idx.values())
    return df_books, split_data(user_idxs, data)
