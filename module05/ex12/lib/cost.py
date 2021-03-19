import numpy as np

def predict(x: np.ndarray, theta: np.ndarray):
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

	x_matrix = add_intercept(x).tolist()
	new_m = []
	for elem in x_matrix:
		tmp = []
		for elem2 in theta[0]:
			tmp.append(0)
		new_m.append(tmp)
	for v1_x in range(len(x_matrix)):
		for v1_y in range(len(x_matrix[0])):
			for v2_y in range(len(theta[0])):
				new_m[v1_x][v2_y] += x_matrix[v1_x][v1_y] * theta[v1_y][v2_y]
	# new_m = [elem for lst in new_m for elem in lst]
	return np.array(new_m)

def cost_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
	"""
	Description:
		Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
	Args:
	y: has to be an numpy.ndarray, a vector.
	y_hat: has to be an numpy.ndarray, a vector.
	Returns:
		J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
		None if there is a dimension matching problem between X, Y or theta.
	Raises:
		This function should not raise any Exception.
	"""
	if y.ndim == 1:
		y = [[elem] for elem in y]
	if y_hat.ndim == 1:
		y_hat = [[elem] for elem in y_hat]
	y = [elem for lst in y for elem in lst]
	y_hat = [elem for lst in y_hat for elem in lst]

	tmp = [pow((e1 - e2), 2)/ (2 * len(y)) for e1, e2 in zip(y, y_hat)]
	return np.array([[elem] for elem in tmp])
		# ... your code here ...

def cost_(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""
	Description:
		Calculates the value of cost function.
	Args:
	y: has to be an numpy.ndarray, a vector.
	y_hat: has to be an numpy.ndarray, a vector.
	Returns:
		J_value : has to be a float.
		None if there is a dimension matching problem between X, Y or theta.
	Raises:
		This function should not raise any Exception.
	"""
	if y.ndim == 1:
		y = [[elem] for elem in y]
	if y_hat.ndim == 1:
		y_hat = [[elem] for elem in y_hat]
	y = [elem for lst in y for elem in lst]
	y_hat = [elem for lst in y_hat for elem in lst]

	tmp = [pow((e1 - e2), 2)/ (2 * len(y)) for e1, e2 in zip(y, y_hat)]
	sum = 0
	for elem in tmp:
		sum += elem
	return sum

# x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
# theta1 = np.array([[2.], [4.]])
# y_hat1 = predict(x1, theta1)
# y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# print(cost_elem_(y1, y_hat1))
# print(cost_(y1, y_hat1))

# x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
# theta2 = np.array([[0.05], [1.], [1.], [1.]])
# y_hat2 = predict(x2, theta2)
# y2 = np.array([[19.], [42.], [67.], [93.]])
# print(cost_elem_(y2, y_hat2))
# print(cost_(y2, y_hat2))

# x3 = np.array([0, 15, -9, 7, 12, 3, -21])
# theta3 = np.array([[0.], [1.]])
# y_hat3 = predict(x3, theta3)
# y3 = np.array([2, 14, -13, 5, 12, 4, -19])
# print(cost_(y3, y3))