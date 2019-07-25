import glob
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_spam_dataset():
    if os.path.basename(os.getcwd()) == "snorkel-tutorials":
        os.chdir("spam")
    # TODO:
    # Add reference to dataset
    # Send email to dataset owner: tuliocasagrande < AT > acm.org
    subprocess.call("bash download_data.sh", shell=True)
    filenames = sorted(glob.glob("data/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    df_dev = df_train.sample(100, random_state=123)
    df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    return df_train, df_dev, df_valid, df_test


if __name__ == "__main__":
    load_spam_dataset()
