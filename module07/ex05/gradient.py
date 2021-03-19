import numpy as np

def add_intercept(x):
	if x.ndim == 1:
		return np.array([[1.0, elem] for elem in x])
	else:
		lst = []
		for elem in x:
			tmp = elem.tolist()
			tmp.insert(0, 1.0)
			lst.append(tmp)
		return np.array(lst)

def gradient(x: np.ndarray , y: np.ndarray , theta: np.ndarray):
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
	return x_transpose.dot(x_theta_minus_y) / len(x.tolist())


# x = np.array([
# 	      [ -6,  -7,  -9],
#         [ 13,  -2,  14],
#         [ -7,  14,  -1],
#         [ -8,  -4,   6],
#         [ -5,  -9,   6],
#         [  1,  -5,  11],
#         [  9, -11,   8]])
# y = np.array([2, 14, -13, 5, 12, 4, -19])
# theta = np.array([0, 3,0.5,-6])
# print(gradient(x, y, theta))

