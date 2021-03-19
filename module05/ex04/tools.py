import numpy as np

def add_intercept(x):
	"""Adds a column of 1's to the non-empty numpy.ndarray x.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
	Returns:
		X as a numpy.ndarray, a vector of dimension m * 2.
		None if x is not a numpy.ndarray.
		None if x is a empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if x.ndim == 1:
		return np.array([[1.0, elem] for elem in x])
	else:
		lst = []
		for elem in x:
			tmp = elem.tolist()
			tmp.insert(0, 1.0)
			lst.append(tmp)
		return np.array(lst)
	# 

# x = np.arange(1,10) #.reshape((3,3))
# # add_intercept(x)
# print(add_intercept(x))


