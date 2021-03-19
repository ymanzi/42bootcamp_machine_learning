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
	x_theta = predict(x, theta)
	x_theta_minus_y = [e1 - e2 for e1, e2 in zip(x_theta, y)]
	x_transpose = T(x)
	gradient_vec = matrix_mul(x_transpose, [[elem / (len(x_theta_minus_y))] for elem in x_theta_minus_y])
	return np.array(gradient_vec)


X = np.array([
	[ -6,  -7,  -9],
        [ 13,  -2,  14],
        [ -7,  14,  -1],
        [ -8,  -4,   6],
        [ -5,  -9,   6],
        [  1,  -5,  11],
        [  9, -11,   8]])

Y = np.array([2, 14, -13, 5, 12, 4, -19])
# theta = np.array([0, 3, 0.5, -6])
theta = np.array([0, 0, 0, 0])
print(vec_gradient(X, Y, theta))

