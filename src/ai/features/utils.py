import numpy as np


def mean_pooling_and_normalization(x):
    o = np.mean(x, axis=0)
    return o / np.linalg.norm(o)


def gauss_norm(x: np.ndarray) -> np.ndarray:
    x = (x - x.mean()) / x.std()
    return x
