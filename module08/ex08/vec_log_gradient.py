import numpy as np 

def sigmoid_(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return None
    x = x.astype(np.float)
    if x.ndim == 0:
        x = np.array(x, ndmin=1)
    return (1 / (1 + (np.exp(x * -1))))

def logistic_predict(x, theta):
    if (x.ndim == 1):
        x = x.reshape(-1, 1)
    x_plus = np.column_stack((np.full((x.shape[0], theta.shape[0] - x.shape[1]) , 1), x)) 
    x_theta = x_plus.dot(theta).reshape(-1, 1)
    return sigmoid_(x_theta)

def vec_log_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatible dimensions.
    Args:
      x: has to be an numpy.ndarray, a matrix of dimension m * n.
      y: has to be an numpy.ndarray, a vector of dimension m * 1.
      theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
      The gradient as a numpy.ndarray, a vector of dimensions n * 1, containing the result of the formula for all j.
      None if x, y, or theta are empty numpy.ndarray.
      None if x, y and theta do not have compatible dimensions.
    Raises:
      This function should not raise any Exception.
    """
    if x.ndim == 1 or y.ndim == 1 or theta.ndim == 1:
        x = np.array(x, ndmin=2)
        y = np.array(y, ndmin=2)
        theta = np.array(theta, ndmin=2)
    if x.size == 0 or y.size == 0 or theta.size == 0 \
            or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None
    y_hat = logistic_predict(x, theta)
    x_plus = np.column_stack((np.full((x.shape[0], theta.shape[0] - x.shape[1]) , 1), x)) 
    return x_plus.transpose().dot(np.subtract(y_hat, y)) / x.shape[0]


# y1 = np.array([1])
# x1 = np.array([4])
# theta1 = np.array([[2], [0.5]])

# y3 = np.array([[0], [1], [1]])
# x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
# theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

# print(vec_log_gradient(x3, y3, theta3))