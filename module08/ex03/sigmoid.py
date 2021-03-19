import numpy as np
from math import exp

def sigmoid_(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be an numpy.ndarray, a vector
    Returns: 
        The sigmoid value as a numpy.ndarray.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if x.size == 0:
        return None
    x = x.astype(np.float)
    if x.ndim == 0:
        x = np.array(x, ndmin=1)
    return (1 / (1 + (np.exp(x * -1))))


# x = np.array(2)
# print(sigmoid_(x))
    