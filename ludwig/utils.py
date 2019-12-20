import numpy as np


def indices_to_one_hot(data):
    """Convert an iterable of indices to one-hot encoded labels."""
    nb_classes = len(np.unique(data))
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
