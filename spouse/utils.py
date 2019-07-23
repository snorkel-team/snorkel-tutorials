import os
import pickle
import subprocess
from typing import Tuple

import numpy as np

import pandas as pd


def load_data() -> Tuple[
    Tuple[pd.DataFrame, np.ndarray], pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]
]:
    """
    Returns:
        dev_df, dev_labels: Development set examples and 1D labels ndarray.
        train_df: Training set examples dataframe.
        test_df, test_labels: Test set examples dataframe and 1D labels ndarray.
    """
    subprocess.run(["bash", "download_data.sh"], check=True)
    with open(os.path.join("data", "dev_data.pkl"), "rb") as f:
        dev_df = pickle.load(f)
        dev_labels = pickle.load(f)

    with open(os.path.join("data", "train_data.pkl"), "rb") as f:
        train_df = pickle.load(f)

    with open(os.path.join("data", "test_data.pkl"), "rb") as f:
        test_df = pickle.load(f)
        test_labels = pickle.load(f)

    # Convert labels to {0, 1} format from {-1, 1} format.
    dev_labels = (1 + dev_labels) // 2
    test_labels = (1 + test_labels) // 2
    return ((dev_df, dev_labels), train_df, (test_df, test_labels))
