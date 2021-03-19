import numpy as np
from lib.prediction import *

def vec_gradient(x: np.ndarray , y: np.ndarray , theta: np.ndarray):
	"""Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have the compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a matrice of dimension (m, n).
	y: has to be an numpy.ndarray, a vector of dimension (m, 1).
	theta: has to be an numpy.ndarray, a vector of dimension (n, 1).
	Returns:
	The gradient as a numpy.ndarray, a vector of dimensions (n, 1), containg the result of the formula for all j.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	x_plus = add_intercept(x)
	x_theta = x_plus.dot(theta)
	x_theta_minus_y = np.subtract(x_theta, y)
	x_transpose = x_plus.transpose()
	return x_transpose.dot(x_theta_minus_y) / x.size


# X = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
# Y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
# theta = np.array([2, 0.7])
# print(vec_gradient(X, Y, theta))

