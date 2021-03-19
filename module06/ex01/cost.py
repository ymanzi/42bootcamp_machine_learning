import numpy as np

def cost_(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""Computes the half mean squared error of two non-empty numpy.ndarray,
			without any for loop. The two arrays must have the same dimensions.
		Args:
		y: has to be an numpy.ndarray, a vector.
		y_hat: has to be an numpy.ndarray, a vector.
		Returns:
		The half mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.ndarray.
		None if y and y_hat does not share the same dimensions.
		Raises:
		This function should not raise any Exceptions.
	"""

	tmp = [pow((e1 - e2), 2)/ (2 * len(y)) for e1, e2 in zip(y, y_hat)]
	return sum(tmp)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(cost_(X, X))