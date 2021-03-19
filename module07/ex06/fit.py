import numpy as np

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

def gradient(x: np.ndarray , y: np.ndarray , theta: np.ndarray):
	x_plus = add_intercept(x)
	x_theta = x_plus.dot(theta)
	x_theta_minus_y = np.subtract(x_theta, y)
	x_transpose = x_plus.transpose()
	return x_transpose.dot(x_theta_minus_y) / len(x.tolist())

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
		grad = gradient(x, y, theta)
		new_theta = []
		for g, t in zip(grad, theta):
			t -= alpha * g
			new_theta.append(t)
		theta = np.array(new_theta)
		# theta = np.array([theta[0] - (alpha * gradient[0]), theta[1] - (alpha * gradient[1])])
		# theta[1] = theta[1] - (alpha * gradient[1])
	return theta

x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])
theta1 = fit_(x, y, theta,  alpha = 0.0005, max_iter=42000)
print(theta1)