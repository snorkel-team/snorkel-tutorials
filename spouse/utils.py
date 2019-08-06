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
        df_dev, Y_dev: Development set examples and 1D labels ndarray.
        df_train: Training set examples dataframe.
        df_test, Y_test: Test set examples dataframe and 1D labels ndarray.
    """
    subprocess.run(["bash", "download_data.sh"], check=True)
    with open(os.path.join("data", "dev_data.pkl"), "rb") as f:
        df_dev = pickle.load(f)
        Y_dev = pickle.load(f)

    with open(os.path.join("data", "train_data.pkl"), "rb") as f:
        df_train = pickle.load(f)

    with open(os.path.join("data", "test_data.pkl"), "rb") as f:
        df_test = pickle.load(f)
        Y_test = pickle.load(f)

    # Convert labels to {0, 1} format from {-1, 1} format.
    Y_dev = (1 + Y_dev) // 2
    Y_test = (1 + Y_test) // 2
    return ((df_dev, Y_dev), df_train, (df_test, Y_test))
