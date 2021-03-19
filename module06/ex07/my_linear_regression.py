import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
	"""
	Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self,  thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
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
		if x.ndim == 1:
			return np.array([[1.0, elem] for elem in x])
		else:
			lst = []
			for elem in x:
				tmp = elem.tolist()
				tmp.insert(0, 1.0)
				lst.append(tmp)
			return np.array(lst)

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

	def fit_(self, x, y):
		for i in range(self.max_iter):
			x_plus = self.add_intercept(x)
			x_theta = self.predict_(x)
			gradient = x_plus.transpose().dot(np.subtract(x_theta, y)) / x.size
			self.thetas[0] -= self.alpha * gradient[0][0]
			self.thetas[1] -= self.alpha * gradient[1][0]
	
	def mse_(self, x, y):
		predicted_y = self.predict_(x)
		ret = np.array([(e1 - e2)**2 / (y.size) \
			for e1, e2 in zip(predicted_y, y)])
		if ret.shape[1] == 1:
			ret = np.array([elem for lst in ret for elem in lst])
		return sum(ret)

	def plot(self, x: np.ndarray, y: np.ndarray):
		predicted_values = self.predict_(x)
		plt.scatter(x, y, color="orange") #draw the multiple points
		for i, elem in enumerate(predicted_values):
			plt.vlines(x=x[i], ymin=y[i], ymax=elem, colors='green', ls='--', lw=2)
		cost = self.cost_(x, y)

		plt.plot(x, predicted_values, color="blue", marker="o")
		plt.xlabel("X")
		plt.ylabel("Y")
		title = "Cost : " + str(cost)[:9]
		plt.title(title)
		plt.show()


# x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
# y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

# lr = MyLinearRegression([1, 1], 5e-8, 1500000)
# lr.fit_(x,y)
# print(lr.thetas)