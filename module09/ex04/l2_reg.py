import numpy as np

def iterative_l2(theta: np.ndarray) -> float:
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
      theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
      The L2 regularization as a float.
      None if theta in an empty numpy.ndarray.
    Raises:
      This function should not raise any Exception.
    """
    if theta.size == 0:
        return None
    array_add = 0
    for i in range(1, theta.size):
        array_add += theta[i]**2
    return float(array_add)


def l2(theta: np.ndarray) -> float:
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
      theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
      The L2 regularization as a float.
      None if theta in an empty numpy.ndarray.
    Raises:
      This function should not raise any Exception.
    """
    if theta.size == 0:
        return None
    array_sum = theta.transpose().dot(theta)
    return float(array_sum - theta[0]**2)


# x = np.array([3,0.5,-6])
# print(iterative_l2(x))
# print(l2(x))