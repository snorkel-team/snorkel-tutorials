import glob
import os
import subprocess

import pandas as pd


def load_unlabeled_spam_dataset():
    """Load spam training dataset without any labels."""
    if os.path.basename(os.getcwd()) == "snorkel-tutorials":
        os.chdir("getting_started")
    try:
        subprocess.run(["bash", "download_data.sh"], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
        raise e
    filenames = sorted(glob.glob("data/Youtube*.csv"))
    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Remove comment_id, label fields
        df = df.drop("comment_id", axis=1)
        df = df.drop("label", axis=1)
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)
    return pd.concat(dfs)
