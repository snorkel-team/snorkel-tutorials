from typing import Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


DataSplits = Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]


def make_circle_dataset(n: int, r: float, **kwargs: Any) -> DataSplits:
    X = np.random.uniform(0, 1, size=(n, 2)) * 2 - 1
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2 < r).astype(int)
    return split_data(X, Y, **kwargs)


def make_inv_circle_dataset(n: int, r: float, **kwargs: Any) -> DataSplits:
    X = np.random.uniform(0, 1, size=(n, 2)) * 2 - 1
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2 > r).astype(int)
    return split_data(X, Y, **kwargs)


def make_square_dataset(n: int, r: float, **kwargs: Any) -> DataSplits:
    X = np.random.uniform(0, 1, size=(n, 2)) * 2 - 1
    Y = ((abs(X[:, 0]) < r / 2) * (abs(X[:, 1]) < r / 2)).astype(int)
    return split_data(X, Y, **kwargs)


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    splits: Tuple[float] = (0.8, 0.1, 0.1),
    seed: int = 123,
) -> DataSplits:
    """Split data twice using sklearn train_test_split helper."""
    assert len(splits) == 3

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=splits[2], random_state=seed
    )
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train, test_size=splits[1] / sum(splits[:2]), random_state=seed
    )

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)
