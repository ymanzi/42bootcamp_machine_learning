import numpy as np
import matplotlib.pyplot as plt

def plot_with_cost(x: np.ndarray, y: np.ndarray , theta: np.ndarray):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
		Nothing.
	Raises:
	This function should not raise any Exception.
	"""
	plt.scatter(x, y) #draw the multiple points
	x_matrix = [[1.0, float(elem)] for elem in x]
	theta = [[elem] for elem in theta]

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
	y_hat = np.array([elem for lst in new_m for elem in lst])
	for i, elem in enumerate(y_hat):
		plt.vlines(x=x[i], ymin=y[i], ymax=elem, colors='green', ls='--', lw=2)
	cost = 2 * sum([pow((e1 - e2), 2)/ (2 * len(y)) for e1, e2 in zip(y, y_hat)])

	plt.plot(x, y_hat, color="#00ff00")
	plt.xlabel("X")
	plt.ylabel("Y")
	title = "Cost : " + str(cost)[:9]
	plt.title(title)
	plt.show()

# x = np.arange(1,6)
# y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# theta1= np.array([14, 0])
# plot_with_cost(x, y, theta1)