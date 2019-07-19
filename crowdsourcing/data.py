import os
import subprocess
from typing import Tuple

import pandas as pd


def load_weather_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.basename(os.getcwd()) == "snorkel-tutorials":
        os.chdir("crowdsourcing")
    subprocess.run(["bash", "download-data.sh"], check=True)
    # Load raw crowdsourcing data
    raw_crowd_answers = pd.read_csv("data/weather-agg-DFE.csv")
    # Load ground truth crowdsourcing data
    gold_labels = pd.read_csv("data/weather-evaluated-agg-DFE.csv")
    gold_labels = gold_labels.set_index("tweet_id", drop=False)
    # Filter out low-confidence answers
    gold_labels = gold_labels[["sentiment"]][
        (gold_labels.correct_category == "Yes")
        & (gold_labels.correct_category_conf == 1)
    ]
    # Keep only the tweets with available ground truth
    crowd_labeled_tweets = raw_crowd_answers.join(
        gold_labels,
        on=["tweet_id"],
        lsuffix=".raw",
        rsuffix=".gold",
        how="inner",
    )
    # Clean up columns
    crowd_labeled_tweets = crowd_labeled_tweets[
        ["tweet_id.raw", "tweet_body.raw", "worker_id", "emotion"]
    ]
    crowd_labeled_tweets.columns = ["tweet_id", "tweet_body", "worker_id", "emotion"]
    # Return crowd answers and labels
    return crowd_labeled_tweets, gold_labels
