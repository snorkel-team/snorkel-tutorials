import os
import pickle
import subprocess
from typing import Tuple

import numpy as np

import pandas as pd

IS_TRAVIS = "TRAVIS" in os.environ


def load_data() -> Tuple[
    Tuple[pd.DataFrame, np.ndarray], pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]
]:
    """
    Returns:
        df_dev, Y_dev: Development set examples and 1D labels ndarray.
        df_train: Training set examples dataframe.
        df_test, Y_test: Test set examples dataframe and 1D labels ndarray.
    """
    try:
        subprocess.run(["bash", "download_data.sh"], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
        raise e
    with open(os.path.join("data", "dev_data.pkl"), "rb") as f:
        df_dev = pickle.load(f)
        Y_dev = pickle.load(f)

    with open(os.path.join("data", "train_data.pkl"), "rb") as f:
        df_train = pickle.load(f)
        if IS_TRAVIS:
            # Reduce train set size to speed up travis.
            df_train = df_train.iloc[:2000]

    with open(os.path.join("data", "test_data.pkl"), "rb") as f:
        df_test = pickle.load(f)
        Y_test = pickle.load(f)

    # Convert labels to {0, 1} format from {-1, 1} format.
    Y_dev = (1 + Y_dev) // 2
    Y_test = (1 + Y_test) // 2
    return ((df_dev, Y_dev), df_train, (df_test, Y_test))


def get_n_epochs() -> int:
    return 3 if IS_TRAVIS else 30
