import numpy as np
from sklearn.model_selection import train_test_split


def make_circle_data(N, R, **kwargs):
    circle_data = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
    circle_labels = (circle_data[:, 0] ** 2 + circle_data[:, 1] ** 2 < R).astype(int)
    return split_data(circle_data, circle_labels, **kwargs)


def make_inv_circle_data(N, R, **kwargs):
    circle_data = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
    circle_labels = (circle_data[:, 0] ** 2 + circle_data[:, 1] ** 2 > R).astype(int)
    return split_data(circle_data, circle_labels, **kwargs)


def make_square_data(N, R, **kwargs):
    square_data = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
    square_labels = (
        (abs(square_data[:, 0]) < R / 2) * (abs(square_data[:, 1]) < R / 2)
    ).astype(int)
    return split_data(square_data, square_labels, **kwargs)


def split_data(data, labels, splits=[0.8, 0.1, 0.1], seed=123):
    """Split data twice using sklearn train_test_split helper."""
    assert len(splits) == 3

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=splits[2], random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=splits[1] / sum(splits[:2]), random_state=seed
    )

    return (
        {"train": X_train, "valid": X_val, "test": X_test},
        {"train": y_train, "valid": y_val, "test": y_test},
    )
