import os
import subprocess
from typing import Tuple

import pandas as pd


answer_mapping = {
    "I can't tell": -1,
    "Negative": 0,
    "Positive": 1,
    "Neutral / author is just sharing information": 2,
    "Tweet not related to weather condition": 3,
}


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if os.path.basename(os.getcwd()) != "crowdsourcing":
        raise ValueError("Function must be called from crowdsourcing/ directory.")
    subprocess.run(["bash", "download-data.sh"], check=True)

    gold_labels = pd.read_csv("data/weather-evaluated-agg-DFE.csv")
    gold_labels = gold_labels.set_index("tweet_id", drop=False)
    labeled = gold_labels[
        (gold_labels["is_the_category_correct_for_this_tweet:confidence"] == 1)
        & (
            (gold_labels.sentiment == "Positive")
            | (gold_labels.sentiment == "Negative")
        )
    ]
    labeled = labeled.sample(frac=1, random_state=123)  # Shuffle items.

    crowd_answers = pd.read_csv("data/weather-non-agg-DFE.csv")
    # Keep only the tweets with available groundtruth.
    crowd_answers = crowd_answers.join(
        labeled, on=["tweet_id"], lsuffix=".raw", rsuffix=".gold", how="inner"
    )
    crowd_answers = crowd_answers[["tweet_id", "worker_id", "emotion"]]
    crowd_answers.emotion = crowd_answers.emotion.map(answer_mapping)
    crowd_answers = crowd_answers.rename(columns=dict(emotion="answer"))
    crowd_answers = crowd_answers.set_index("tweet_id")

    df_dev = labeled[:50]
    df_dev = df_dev[["tweet_id", "tweet_text", "sentiment"]]
    df_dev.sentiment = df_dev.sentiment.map(answer_mapping).values
    crowd_answers = crowd_answers.drop(df_dev[:int(len(df_dev) / 2)].tweet_id)

    df_test = labeled[50:100]
    df_test = df_test[["tweet_id", "tweet_text", "sentiment"]]
    df_test.sentiment = df_test.sentiment.map(answer_mapping).values
    crowd_answers = crowd_answers.drop(df_test.tweet_id)

    df_train = labeled[100:][["tweet_id", "tweet_text"]]
    crowd_answers = crowd_answers.drop(df_train[:int(len(df_train) / 2)].tweet_id)

    return crowd_answers, df_train, df_dev, df_test
