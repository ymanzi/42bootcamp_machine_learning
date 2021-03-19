import numpy as np

def zscore(x):
	"""Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
	Args:
	x: has to be an numpy.ndarray, a vector.
	Returns:
	x' as a numpy.ndarray. 
	None if x is a non-empty numpy.ndarray.
	Raises:
	This function shouldn't raise any Exception.
	"""
	def mean(x):
		return float(sum(x) / len(x))

	def std(x):
		mean = float(sum(x) / len(x))
		f = lambda x: (x - mean)**2
		tmp_lst = list(map(f, x))
		return float(sum(tmp_lst) / len(x)) ** (0.5)

	mean = mean(x)
	standard_deviation = std(x)
	f = lambda x: (x - mean) / standard_deviation 
	return np.array(list(map(f, x)))

X = np.array([2, 14, -13, 5, 12, 4, -19])
print(zscore(X))