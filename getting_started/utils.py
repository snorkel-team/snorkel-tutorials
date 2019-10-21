import glob
import os
import subprocess
import sys

import pandas as pd

FILES = (
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
DIRECTORY = "getting_started"

sys.path.insert(
    0, os.path.split(os.path.dirname(__file__))[0]
)  # so we can import from utils
from snorkle_example_utils.download_files import download_files


def load_unlabeled_spam_dataset():
    """Load spam training dataset without any labels."""
    if os.path.basename(os.getcwd()) == "snorkel-tutorials":
        os.chdir("getting_started")

    download_files(FILES, DATA_URL, DIRECTORY)

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
