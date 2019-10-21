import os
import pickle
import subprocess
from typing import Tuple
import sys

import numpy as np
import shutil

import pandas as pd



sys.path.insert(0, os.path.split(os.path.dirname(__file__))[0]) # so we can import from utils
from snorkle_example_utils.download_files import download_files

IS_TEST = os.environ.get("TRAVIS") == "true" or os.environ.get("IS_TEST") == "true"
DATA_URL="https://www.dropbox.com/s/jmrvyaqew4zp9cy/spouse_data.zip?dl=1"
FILES=( "train_data.pkl", "dev_data.pkl", "test_data.pkl", "dbpedia.pkl" )
DIRECTORY = "spouse"

def load_data() -> Tuple[
    Tuple[pd.DataFrame, np.ndarray], pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]
]:
    """
    Returns:
        df_dev, Y_dev: Development set data points and 1D labels ndarray.
        df_train: Training set data points dataframe.
        df_test, Y_test: Test set data points dataframe and 1D labels ndarray.
    """
    
    download_files(FILES, DATA_URL, DIRECTORY)
    with open(os.path.join("data", "dev_data.pkl"), "rb") as f:
        df_dev = pickle.load(f)
        Y_dev = pickle.load(f)

    with open(os.path.join("data", "train_data.pkl"), "rb") as f:
        df_train = pickle.load(f)
        if IS_TEST:
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
    return 3 if IS_TEST else 30
