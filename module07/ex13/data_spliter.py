import numpy as np
from random import shuffle

def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set, while respecting the given proportion of examples to be kept in the traning set.
	Args:
	x: has to be an numpy.ndarray, a matrix of dimension m * n.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	proportion: has to be a float, the proportion of the dataset that will be assigned to the training set.
	Returns:
	(x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray
	None if x or y is an empty numpy.ndarray.
	None if x and y do not share compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
		return None
	random_zip = np.array(list(zip(x, y)), dtype=list)
	np.random.shuffle(random_zip)
	new_x = random_zip[:, 0]
	new_y = random_zip[:, 1]
	proportion_position = int(x.shape[0] * proportion)
	ret_array = []
	ret_array.append(new_x[:proportion_position])
	ret_array.append(new_x[proportion_position:])
	ret_array.append(new_y[:proportion_position])
	ret_array.append(new_y[proportion_position:])
	return np.array(ret_array, dtype=np.ndarray)
	

x1 = np.array([ [  1,  42],
                [300,  10],
                [ 59,   1],
                [300,  59],
                [ 10,  42]])
y = np.array([0,1,0,1,0])
# Example 1:
print(data_spliter(x1, y, 0.5)[0])