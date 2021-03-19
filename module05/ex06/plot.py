import numpy as np
import matplotlib.pyplot as plt

def plot(x: np.ndarray, y: np.ndarray , theta: np.ndarray):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
		Nothing.
	Raises:
		This function should not raise any Exceptions.
	"""
	# for i in range(x.shape[0]):
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
	new_m = np.array([elem for lst in new_m for elem in lst])

	plt.plot(x, new_m, color="#00ff00")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

x = np.arange(1,6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
theta1 = np.array([-1.5, 2])
plot(x, y, theta1)