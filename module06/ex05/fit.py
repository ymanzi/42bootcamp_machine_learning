import numpy as np
from prediction import vec_gradient

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
		Fits the model to the training dataset contained in x and y.
	Args:
		x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
	Returns:
		new_theta: numpy.ndarray, a vector of dimension 2 * 1.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exception.
	"""
	for i in range(max_iter):
		gradient = vec_gradient(x, y, theta)
		theta = np.array([theta[0] - (alpha * gradient[0]), theta[1] - (alpha * gradient[1])])
		# theta[1] = theta[1] - (alpha * gradient[1])
	return theta

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1])
theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
print(theta1)