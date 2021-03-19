import numpy as np

def simple_gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a 2 * 1 vector.
	Returns:
	The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	gradient = []
	theta_list = []
	for one,two in zip(x,y):
		theta_list.append(theta[1]*one + theta[0] - two)
	gradient.append(sum(theta_list) / x.size)
	grandient_1_list = []
	for one, two in zip(theta_list, x):
		grandient_1_list.append(one * two)
	gradient.append(sum(grandient_1_list) / x.size)
	return np.array([[elem] for elem in gradient])


x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta = np.array([2, 0.7])
print(simple_gradient(x, y, theta))