import numpy as np


def sigmoid_(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return None
    x = x.astype(np.float)
    if x.ndim == 0:
        x = np.array(x, ndmin=1)
    return (1 / (1 + (np.exp(x * -1))))

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception. 
    """
    if (x.ndim == 1):
        x = x.reshape(-1, 1)
    x_plus = np.column_stack((np.full((x.shape[0], theta.shape[0] - x.shape[1]) , 1), x)) 
    x_theta = x_plus.dot(theta).reshape(-1, 1)
    return sigmoid_(x_theta)

# x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
# theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# print(logistic_predict_(x, theta))