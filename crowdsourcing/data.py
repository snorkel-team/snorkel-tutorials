import os
import subprocess
from typing import Tuple

import pandas as pd
import numpy as np

emotions = {
    "I can't tell": -1,
    "Positive": 0,
    "Negative": 1,
    "Neutral / author is just sharing information": 2,
    "Tweet not related to weather condition": 3,
}


def load_data() -> Tuple[
    pd.DataFrame,
    Tuple[
        Tuple[pd.DataFrame, np.ndarray],
        Tuple[pd.DataFrame, np.ndarray],
        Tuple[pd.DataFrame, np.ndarray],
        pd.DataFrame,
    ],
]:
    if os.path.basename(os.getcwd()) != "crowdsourcing":
        raise ValueError("Function must be called from crowdsourcing/ directory.")
    subprocess.run(["bash", "download-data.sh"], check=True)

    gold_labels = pd.read_csv("data/weather-evaluated-agg-DFE.csv")
    gold_labels = gold_labels.set_index("tweet_id", drop=False)
    labeled = gold_labels[
        gold_labels["is_the_category_correct_for_this_tweet:confidence"] == 1
    ]
    labeled = labeled.sample(frac=1, random_state=123)  # Shuffle items.

    crowd_answers = pd.read_csv("data/weather-non-agg-DFE.csv")
    # Keep only the tweets with available groundtruth.
    crowd_answers = crowd_answers.join(
        labeled, on=["tweet_id"], lsuffix=".raw", rsuffix=".gold", how="inner"
    )
    crowd_answers = crowd_answers[["tweet_id", "worker_id", "emotion"]]
    crowd_answers.emotion = crowd_answers.emotion.map(emotions)

    dev_df = labeled[:50]
    dev_df, dev_labels = (
        dev_df[["tweet_id", "tweet_text"]],
        dev_df["sentiment"].map(emotions).values,
    )

    val_df = labeled[50:100]
    val_df, val_labels = (
        val_df[["tweet_id", "tweet_text"]],
        val_df["sentiment"].map(emotions).values,
    )

    test_df = labeled[100:150]
    test_df, test_labels = (
        test_df[["tweet_id", "tweet_text"]],
        test_df["sentiment"].map(emotions).values,
    )

    train_df = labeled[150:][["tweet_id", "tweet_text"]]
    return (
        crowd_answers,
        ((dev_df, dev_labels), (val_df, val_labels), (test_df, test_labels), train_df),
    )
