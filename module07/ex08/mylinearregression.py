import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
	"""
	Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self,  thetas, alpha=0.001, max_iter=10):
		self.alpha = alpha
		self.max_iter = max_iter
		if isinstance(thetas, list):
			thetas = np.array(thetas)
		if thetas.ndim != 1:
			thetas = np.array([elem for lst in thetas for elem in lst])
		self.thetas = thetas

	def add_intercept(self, x):
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
		ret = x
		while (ret.shape[1] != self.thetas.shape[0]):
			# print(ret.shape[1], "  ", self.thetas.shape[0])
			if x.ndim == 1:
				ret = np.array([[1.0, elem] for elem in ret])
			else:
				lst = []
				for elem in ret:
					tmp = elem.tolist()
					tmp.insert(0, 1.0)
					lst.append(tmp)
				ret = np.array(lst)
		return ret

	def predict_(self, x):
		x_plus = self.add_intercept(x)
		if x.shape[1] == 1:
			return np.array([[elem] for elem in x_plus.dot(self.thetas)])
		return x_plus.dot(self.thetas)

	def cost_elem_(self, x, y):
		predicted_y = self.predict_(x)
		ret = np.array([(e1 - e2)**2 / (2 * y.size) \
			for e1, e2 in zip(predicted_y, y)])
		if ret.shape[1] == 1:
			ret = np.array([elem for lst in ret for elem in lst])
		return ret

	def cost_(self, x, y):
		return sum(self.cost_elem_(x, y).tolist())
	
	def get_gradient(self, x: np.ndarray , y: np.ndarray):
		if y.ndim > 1:
			y = np.array([elem for lst in y for elem in lst])
		x_plus = self.add_intercept(x)
		x_theta = x_plus.dot(self.thetas)
		x_theta_minus_y = np.subtract(x_theta, y)
		return x_plus.transpose().dot(x_theta_minus_y) / len(x.tolist())

	def fit_(self, x, y, alpha = 0.1, n_cycle=1000):
		if y.ndim > 1:
			y = np.array([elem for lst in y for elem in lst])
		x_plus = self.add_intercept(x)
		for i in range(n_cycle):
			x_theta = x_plus.dot(self.thetas)
			x_theta_minus_y = np.subtract(x_theta, y)
			grad = x_plus.transpose().dot(x_theta_minus_y) / len(x.tolist())
			self.thetas = np.subtract(self.thetas, alpha * grad)
		return self.thetas
	
	def mse_(self, x, y):
		predicted_y = self.predict_(x)
		ret = np.array([(e1 - e2)**2 / (y.size) \
			for e1, e2 in zip(predicted_y, y)])
		if ret.shape[1] == 1:
			ret = np.array([elem for lst in ret for elem in lst])
		return sum(ret)

	def plot_line(self, x: np.ndarray, y: np.ndarray):
		predicted_values = self.predict_(x)
		plt.scatter(x, y, color="orange") #draw the multiple points
		cost = self.cost_(x, y)
		plt.plot(x, predicted_values, color="blue", marker="o")
		plt.xlabel("X")
		plt.ylabel("Y")
		title = "Cost : " + str(cost)[:9]
		plt.title(title)
		plt.show()

	def plot_points(self, x_values, expected_values):
		predicted_values = self.predict_(x_values)
		# print(np.array([[elem] for elem in predicted_values]))
		# print(expected_values)
		plt.plot(x_values, predicted_values, linestyle="",marker="o", color="orange")
		plt.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
		# plt.title("Cost: " + str(cost))
		plt.show()
	# def plot_points(self, x: np.ndarray, y: np.ndarray):
	# 	predicted_values = self.predict_(x)
		# plt.scatter(x_values, predicted_values, color="orange") #draw the multiple points
	# 	cost = self.cost_(x, y)
	# 	plt.scatter(x, predicted_values, color="blue", marker="x")
	# 	plt.xlabel("X")
	# 	plt.ylabel("Y")
	# 	title = "Cost : " + str(cost)[:9]
	# 	plt.title(title)
		# plt.show()


# x = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
# y = np.array([[23.], [48.], [218.]])

# lr = MyLinearRegression(np.array([[1.], [1.], [1.], [1.], [1]]))
# lr.fit_(x, y)
# print(lr.cost_elem_(x, y))