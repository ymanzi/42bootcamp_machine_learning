import numpy as np

def minmax(x):
	"""Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
	Args:
	x: has to be an numpy.ndarray, a vector.
	Returns:
	x' as a numpy.ndarray. 
	None if x is a non-empty numpy.ndarray.
	Raises:
	This function shouldn't raise any Exception.
	"""
	array_min = min(x)
	array_max = max(x)
	diff_max_min = array_max - array_min
	f = lambda x: (x - array_min) / diff_max_min
	return np.array(list(map(f, x)))

X = np.array([2, 14, -13, 5, 12, 4, -19])
print(minmax(X))
