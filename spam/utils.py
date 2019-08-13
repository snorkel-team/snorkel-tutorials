import glob
import os
import subprocess

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from snorkel.analysis import Scorer
from snorkel.classification.data import DictDataset, DictDataLoader
from snorkel.classification.task import Operation
from snorkel.classification import Task


def load_spam_dataset(load_train_labels: bool = False, include_dev: bool = True):
    if os.path.basename(os.getcwd()) == "snorkel-tutorials":
        os.chdir("spam")
    # TODO:
    # Add reference to dataset
    # Send email to dataset owner: tuliocasagrande < AT > acm.org
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

    if include_dev:
        df_dev = df_train.sample(100, random_state=123)

    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    if include_dev:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_valid, df_test


def df_to_torch_features(vectorizer, df, fit_train=False):
    """Convert pandas DataFrame containing spam data to bag-of-words PyTorch features."""
    words = [row.text for i, row in df.iterrows()]

    if fit_train:
        feats = vectorizer.fit_transform(words)
    else:
        feats = vectorizer.transform(words)
    X = feats.todense()
    Y = df["label"].values
    return X, Y


def create_dict_dataloader(X, Y, split, **kwargs):
    """Create a DictDataLoader for bag-of-words features."""
    ds = DictDataset(
        name="spam_dataset",
        split=split,
        X_dict={"bow_features": torch.FloatTensor(X)},
        Y_dict={"spam_task": torch.LongTensor(Y)},
    )
    return DictDataLoader(ds, **kwargs)


def create_spam_task(bow_dim):
    """Create a Snorkel Task specifying the task flow for a simple Multi-layer Perceptron."""

    # Define a `module_pool` with all the PyTorch modules that we'll want to include in our network
    module_pool = nn.ModuleDict(
        {
            "mlp": nn.Sequential(nn.Linear(bow_dim, bow_dim), nn.ReLU()),
            "prediction_head": nn.Linear(bow_dim, 2),
        }
    )

    # Specify the desired `op_sequence` through each module
    op_sequence = [
        Operation(
            name="input_op", module_name="mlp", inputs=[("_input_", "bow_features")]
        ),
        Operation(name="head_op", module_name="prediction_head", inputs=["input_op"]),
    ]

    # Define a Snorkel Task
    spam_task = Task(
        name="spam_task",
        module_pool=module_pool,
        op_sequence=op_sequence,
        scorer=Scorer(metrics=["accuracy", "f1"]),
    )

    return spam_task
