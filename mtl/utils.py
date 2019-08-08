import numpy as np
from sklearn.model_selection import train_test_split


def make_circle_dataset(N, R, **kwargs):
    X = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2 < R).astype(int)
    return split_data(X, Y, **kwargs)


def make_inv_circle_dataset(N, R, **kwargs):
    X = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2 > R).astype(int)
    return split_data(X, Y, **kwargs)


def make_square_dataset(N, R, **kwargs):
    X = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
    Y = ((abs(X[:, 0]) < R / 2) * (abs(X[:, 1]) < R / 2)).astype(int)
    return split_data(X, Y, **kwargs)


def split_data(X, Y, splits=[0.8, 0.1, 0.1], seed=123):
    """Split data twice using sklearn train_test_split helper."""
    assert len(splits) == 3

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=splits[2], random_state=seed
    )
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train, test_size=splits[1] / sum(splits[:2]), random_state=seed
    )

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)
