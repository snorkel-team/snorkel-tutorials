import os
import subprocess
from typing import Tuple
import urllib
import shutil

import pandas as pd


LABEL_MAPPING = {
    "Negative": 0,
    "Positive": 1,
    "I can't tell": 2,
    "Neutral / author is just sharing information": 2,
    "Tweet not related to weather condition": 2,
}

def download_files():
    reload = any(not os.path.exists(os.path.join("data", filename)) for filename in ("weather-non-agg-DFE.csv", "weather-evaluated-agg-DFE.csv"))    
    if reload:
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.mkdir("data")
        
        urllib.request.urlretrieve("https://www.dropbox.com/s/94d2wsrrwh1ioyd/weather-non-agg-DFE.csv", os.path.join("data", "weather-non-agg-DFE.csv"))
        urllib.request.urlretrieve("https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/weather-evaluated-agg-DFE.csv", os.path.join("data", "weather-evaluated-agg-DFE.csv"))
  

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if os.path.basename(os.getcwd()) != "crowdsourcing":
        raise ValueError("Function must be called from crowdsourcing/ directory.")
    download_files()


    gold_labels = pd.read_csv("data/weather-evaluated-agg-DFE.csv")
    gold_labels = gold_labels.set_index("tweet_id", drop=False)
    labeled = gold_labels[
        (gold_labels["is_the_category_correct_for_this_tweet:confidence"] == 1)
        & (
            (gold_labels.sentiment == "Positive")
            | (gold_labels.sentiment == "Negative")
        )
    ]
    labeled = labeled.sample(frac=1, random_state=123)  # Shuffle data points.

    crowd_labels = pd.read_csv("data/weather-non-agg-DFE.csv")
    # Keep only the tweets with available ground truth.
    crowd_labels = crowd_labels.join(
        labeled, on=["tweet_id"], lsuffix=".raw", rsuffix=".gold", how="inner"
    )
    crowd_labels = crowd_labels[["tweet_id", "worker_id", "emotion"]]
    crowd_labels.emotion = crowd_labels.emotion.map(LABEL_MAPPING)
    crowd_labels = crowd_labels.rename(columns=dict(emotion="label"))
    crowd_labels = crowd_labels.set_index("tweet_id")
    crowd_labels = crowd_labels[crowd_labels["label"] != 2]

    df_dev = labeled[:50]
    df_dev = df_dev[["tweet_id", "tweet_text", "sentiment"]]
    df_dev.sentiment = df_dev.sentiment.map(LABEL_MAPPING).values
    # Remove half the labels
    crowd_labels = crowd_labels.drop(df_dev[: int(len(df_dev) / 2)].tweet_id)

    df_test = labeled[50:100]
    df_test = df_test[["tweet_id", "tweet_text", "sentiment"]]
    df_test.sentiment = df_test.sentiment.map(LABEL_MAPPING).values
    crowd_labels = crowd_labels.drop(df_test.tweet_id)

    df_train = labeled[100:][["tweet_id", "tweet_text"]]
    # Remove half the labels
    crowd_labels = crowd_labels.drop(df_train[: int(len(df_train) / 2)].tweet_id)

    return crowd_labels, df_train, df_dev, df_test

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "snorkel-tutorials":
        os.chdir("crowdsourcing")
    download_files()