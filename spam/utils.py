import glob
import subprocess

import pandas as pd
from sklearn.model_selection import train_test_split


def load_spam_dataset():
    # TODO:
    # Add reference to dataset
    # Send email to dataset owner: tuliocasagrande < AT > acm.org
    subprocess.call("bash spam/download_data.sh", shell=True)
    filenames = sorted(glob.glob("spam/data/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video_id"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    df_dev = df_train.sample(100, random_state=123)
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    # TODO: Drop the label column for train
    return df_train, df_dev, df_valid, df_test


if __name__ == "__main__":
    load_spam_dataset()
