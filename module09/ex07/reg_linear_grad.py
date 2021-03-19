import numpy as np

def theta0(theta):
    theta[0] = 0
    return theta

def reg_linear_grad(y, x, theta, lambda_):
    """
    Args:
      y: has to be a numpy.ndarray, a vector of dimension m * 1.
      x: has to be a numpy.ndarray, a matrix of dimesion m * n.
      theta: has to be a numpy.ndarray, a vector of dimension n * 1.
      lambda_: has to be a float.
    Returns:
      A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
      None if y, x, or theta are empty numpy.ndarray.
      None if y, x or theta does not share compatibles dimensions.
    Raises:
      This function should not raise any Exception.
    """
    x_plus = np.column_stack((np.full((x.shape[0], theta.shape[0] - x.shape[1]) , 1), x)) # add intercept
    arr_size = y.shape[0] 
    return (x_plus.transpose().dot(np.subtract(x_plus.dot(theta) , y)) + lambda_ * theta0(theta))  / arr_size


# x = np.array([
#       [ -6,  -7,  -9],
#       [ 13,  -2,  14],
#       [ -7,  14,  -1],
#       [ -8,  -4,   6],
#       [ -5,  -9,   6],
#       [  1,  -5,  11],
#       [  9, -11,   8]])
# y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
# theta = np.array([[7.01], [3], [10.5], [-6]])

# # Example 1.1:
# print(reg_linear_grad(y, x, theta, 0.0))