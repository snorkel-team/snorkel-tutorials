import calendar
import gzip
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

IS_TEST = os.environ.get("TRAVIS") == "true" or os.environ.get("IS_TEST") == "true"

YA_BOOKS_URL = "https://drive.google.com/uc?id=1gH7dG4yQzZykTpbHYsrw2nFknjUm0Mol"
YA_INTERACTIONS_URL = "https://drive.google.com/uc?id=1NNX7SWcKahezLFNyiW88QFPAqOAYP5qg"
YA_REVIEWS_URL = "https://drive.google.com/uc?id=1M5iqCZ8a7rZRtsmY5KQ5rYnP9S0bQJVo"
SMALL_DATA_URL = "https://drive.google.com/uc?id=1_UY4xTbk3o0xjGbVllQZC2bBt-WAwyF_"

BOOK_DATA = "data/goodreads_books_young_adult.json.gz"
INTERACTIONS_DATA = "data/goodreads_interactions_young_adult.json.gz"
REVIEWS_DATA = "data/goodreads_reviews_young_adult.json.gz"
SAMPLE_DATA = "data/sample_data.pkl"


# +
def save_small_sample():
    """Load full data, sample, and dump to file.."""
    (df_train, df_test, df_dev, df_valid), df_books = download_and_process_data()
    df_train = df_train.dropna().sample(frac=0.01)
    df_test = df_test.dropna().sample(frac=0.01)
    df_dev = df_dev.dropna().sample(frac=0.01)
    df_valid = df_valid.dropna().sample(frac=0.01)
    df_all = pd.concat([df_train, df_test, df_dev, df_valid], axis=0)
    df_books = df_books.merge(
        df_all[["book_idx"]].drop_duplicates(), on="book_idx", how="inner"
    )
    with open(SAMPLE_DATA, "wb") as f:
        pickle.dump(df_train, f)
        pickle.dump(df_test, f)
        pickle.dump(df_dev, f)
        pickle.dump(df_valid, f)
        pickle.dump(df_books, f)


def load_small_sample():
    """Load sample data."""
    with open(SAMPLE_DATA, "rb") as f:
        df_train = pickle.load(f)
        df_test = pickle.load(f)
        df_dev = pickle.load(f)
        df_valid = pickle.load(f)
        df_books = pickle.load(f)
        return (df_train, df_test, df_dev, df_valid), df_books


# -


def maybe_download_files(data_dir: str = "data") -> None:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        if IS_TEST:
            # Sample data pickle
            gdown.download(SMALL_DATA_URL, output=SAMPLE_DATA, quiet=None)
        else:
            # Books
            gdown.download(YA_BOOKS_URL, output=BOOK_DATA, quiet=None)
            # Interactions
            gdown.download(YA_INTERACTIONS_URL, output=INTERACTIONS_DATA, quiet=None)
            # Reviews
            gdown.download(YA_REVIEWS_URL, output=REVIEWS_DATA, quiet=None)


def get_timestamp(date_str: str) -> datetime.timestamp:
    month_to_int = dict((v, k) for k, v in enumerate(calendar.month_abbr))
    _, month, day, _, _, year = date_str.split()
    dt = datetime(year=int(year), month=month_to_int[month], day=int(day))
    return datetime.timestamp(dt)


def load_data(
    file_name: str, max_to_load: int = 100, filter_dict: Optional[dict] = None
) -> List[Dict[str, Any]]:
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


def process_books_data(
    book_path: str = BOOK_DATA, min_ratings: int = 100, max_ratings: int = 15000
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    books = load_data(book_path, None)
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

    df_books = df_books[
        (df_books.ratings_count >= min_ratings)
        & (df_books.ratings_count <= max_ratings)
    ]

    book_id_to_idx = {v: i for i, v in enumerate(df_books.book_id)}
    df_books["book_idx"] = df_books.book_id.map(book_id_to_idx)
    return df_books, book_id_to_idx


def process_interactions_data(
    book_id_to_idx: Dict[int, int],
    interactions_path: str = INTERACTIONS_DATA,
    min_user_count: int = 25,
    max_user_count: int = 200,
    max_to_load: int = 5_000_000,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    interactions = load_data(
        interactions_path,
        max_to_load,
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
    user_mask = (user_counts >= min_user_count) & (user_counts <= max_user_count)
    users_filt = user_counts[user_mask].index
    user_id_to_idx = {v: i for i, v in enumerate(users_filt)}
    df_interactions = df_interactions[
        df_interactions.user_id.isin(set(user_id_to_idx.keys()))
    ]
    df_interactions["user_idx"] = df_interactions.user_id.map(user_id_to_idx)
    return df_interactions, user_id_to_idx


def process_reviews_data(
    book_id_to_idx: Dict[int, int],
    user_id_to_idx: Dict[int, int],
    reviews_path: str = REVIEWS_DATA,
) -> pd.DataFrame:
    reviews = load_data(
        reviews_path,
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


def split_data(user_idxs, data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    user_idxs_train, user_idxs_test = train_test_split(user_idxs, test_size=0.05)
    user_idxs_train, user_idxs_dev = train_test_split(user_idxs_train, test_size=0.01)
    user_idxs_train, user_idxs_val = train_test_split(user_idxs_train, test_size=0.01)

    data_train = data[data.user_idx.isin(set(user_idxs_train))].drop("rating", axis=1)
    data_test = data[data.user_idx.isin(set(user_idxs_test))]
    data_dev = data[data.user_idx.isin(set(user_idxs_dev))]
    data_val = data[data.user_idx.isin(set(user_idxs_val))]
    return data_train, data_test, data_dev, data_val


def download_and_process_data() -> Tuple[Tuple[pd.DataFrame, ...], pd.DataFrame]:
    logging.info("Downloading raw data")
    maybe_download_files()
    if IS_TEST:
        return load_small_sample()
    logging.info("Processing book data")
    df_books, book_id_to_idx = process_books_data()
    logging.info("Processing interaction data")
    df_interactions, user_id_to_idx = process_interactions_data(book_id_to_idx)
    df_interactions_nz = df_interactions[df_interactions.rating != 0]
    ratings_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    df_interactions_nz["rating_4_5"] = df_interactions_nz.rating.map(ratings_map)
    logging.info("Processing review data")
    df_reviews = process_reviews_data(book_id_to_idx, user_id_to_idx)
    logging.info("Joining interaction data")
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
    return split_data(user_idxs, data), df_books


def recall_batch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = K.sum(K.round(y_true * y_pred))
    all_positives = K.sum(y_true)
    return true_positives / (all_positives + K.epsilon())


def precision_batch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = K.sum(K.round(y_true * y_pred))
    predicted_positives = K.sum(K.round(y_pred))
    return true_positives / (predicted_positives + K.epsilon())


def f1_batch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec = precision_batch(y_true, y_pred)
    rec = recall_batch(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def get_n_epochs() -> int:
    return 2 if IS_TEST else 30
