import os
import pickle
import subprocess


def load_data(data_dir):
    """
    Args:
        data_dir: Name of directory containing pickle files.

    Returns:
        dev_df, dev_labels: Development set examples dataframe and 1D labels np.array.
        train_df: Training set examples dataframe.
        test_df, test_labels: Test set examples dataframe and 1D labels np.array.
    """
    subprocess.run(["bash", "download_data.sh"], check=True)
    with open(os.path.join(data_dir, "dev_data.pkl"), "rb") as f:
        dev_df = pickle.load(f)
        dev_labels = pickle.load(f)

    with open(os.path.join(data_dir, "train_data.pkl"), "rb") as f:
        train_df = pickle.load(f)

    with open(os.path.join(data_dir, "test_data.pkl"), "rb") as f:
        test_df = pickle.load(f)
        test_labels = pickle.load(f)
    return ((dev_df, dev_labels), train_df, (test_df, test_labels))
