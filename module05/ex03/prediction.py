import numpy as np

def simple_predict(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray) \
		or not isinstance(theta, np.ndarray) \
		or x.ndim != 1 or theta.ndim != 1:
		return None 

	y_hat = np.zeros(x.shape)
	for i, elem in enumerate(x):
		y_hat[i] = theta[0] + theta[1] * elem
	return y_hat

# x = np.arange(1,6)
# theta1 = np.array([-3, 1])
# ret = simple_predict(x, theta1)
# print(ret)