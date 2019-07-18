import glob
import os
import subprocess
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


def load_spam_dataset():
    # TODO:
    # Add reference to dataset
    # Send email to dataset owner: tuliocasagrande < AT > acm.org
    subprocess.call(["bash", "download_data.sh"], shell=True)
    filenames = sorted(glob.glob("raw_data/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        df["VIDEO_ID"] = [i] * len(df)
        df = df.rename(columns={"CLASS": "LABEL"})
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle order
        df['LABEL'] = df['LABEL'].map({0: 2, 1: 1})
        dfs.append(df)

    df_train_dev = pd.concat(dfs[:4])
    df_train, df_dev = train_test_split(
        df_train_dev,
        test_size=0.1,
        random_state=123,
        stratify=df_train_dev.LABEL
    )
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test,
        test_size=0.5,
        random_state=123,
        stratify=df_valid_test.LABEL,
    )

    # TODO: Drop the label column for train
    return df_train, df_dev, df_valid, df_test


if __name__ == "__main__":
    load_spam_dataset()