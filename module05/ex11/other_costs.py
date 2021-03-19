import numpy as np
from math import sqrt, pow

def mse_(y : np.ndarray, y_hat : np.ndarray) -> float:
	"""
	Description:
		Calculate the MSE between the predicted output and the real output.
	Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
	Returns:
		mse: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	tmp = [pow((e1 - e2), 2)/ (len(y)) for e1, e2 in zip(y, y_hat)]
	return sum(tmp)

def mse_elem(y: np.ndarray, y_hat: np.ndarray):
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	tmp = [pow((e1 - e2), 2)/ (len(y)) for e1, e2 in zip(y, y_hat)]
	return tmp

def rmse_(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""
	Description:
		Calculate the RMSE between the predicted output and the real output.
	Args:
	        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
	Returns:
		rmse: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	tmp = [pow((e1 - e2), 2)/ (len(y)) for e1, e2 in zip(y, y_hat)]
	return sqrt(sum(tmp))

def rmse_elem(y: np.ndarray, y_hat: np.ndarray):
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	tmp = [pow((e1 - e2), 2)/ (len(y)) for e1, e2 in zip(y, y_hat)]
	return tmp


def mae_(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""
	Description:
		Calculate the MAE between the predicted output and the real output.
	Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
	Returns:
		mae: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	tmp = [abs(e2 - e1)/ (len(y)) for e1, e2 in zip(y, y_hat)]
	return sum(tmp)

def mae_elem(y: np.ndarray, y_hat: np.ndarray):
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	tmp = [abs(e2 - e1)/ (len(y)) for e1, e2 in zip(y, y_hat)]
	return tmp

def r2score_(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""
	Description:
		Calculate the R2score between the predicted output and the output.
	Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
	Returns:
		r2score: has to be a float.
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exceptions.
	"""
	if (y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape):
		return None
	y_minus_yhat = [pow(e2 - e1, 2) for e1, e2 in zip(y, y_hat)]
	mean_y = sum(y.tolist()) / y.size
	y_minus_means = [pow(e1 - mean_y, 2) for e1 in y]

	return (1 - (sum(y_minus_yhat) / sum(y_minus_means)))

# Example 1:
x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])

# Mean squared error
## your implementation
print(r2score_(x,y))