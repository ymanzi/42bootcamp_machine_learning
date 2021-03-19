import numpy as np
from random import shuffle

def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float):
	if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
		return None
	random_zip = list(zip(x.tolist(), y))
	shuffle(random_zip)
	new_x = []
	new_y = []
	for e1, e2 in random_zip:
		new_x.append(e1)
		new_y.append(e2)
	new_x = np.array(new_x)
	new_y = np.array(new_y)
	proportion_position = int(x.shape[0] * proportion)
	ret_array = []
	ret_array.append(new_x[:proportion_position])
	ret_array.append(new_x[proportion_position:])
	ret_array.append(new_y[:proportion_position])
	ret_array.append(new_y[proportion_position:])
	return np.array(ret_array, dtype=np.ndarray)

class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, n_cycle=1000, penalty='l2'):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.theta = np.array(theta).reshape(-1, 1)
        self.penalty = penalty

    def theta0(self, theta):
        theta[0] = 0
        return theta

    def sigmoid_(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return None
        x = x.astype(np.float)
        if x.ndim == 0:
            x = np.array(x, ndmin=1)
        return (1 / (1 + (np.exp(x * -1))))
    
    def predict_(self, x):
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x)) 
        x_theta = x_plus.dot(self.theta).reshape(-1, 1)
        return self.sigmoid_(x_theta)

    def cost_(self, x: np.ndarray, y: np.ndarray, lamdba_=0.0, eps=1e-15):
        y_hat = self.predict_(x)
        ones = np.ones(y.shape)
        arr_size = y.shape[0]
        if (self.penalty == 'l2'):
            l2 = float(self.theta.transpose().dot(self.theta) - self.theta[0]**2)
        elif self.penalty == 'none':
            l2_cor = 0
        log_loss_array = y.transpose().dot(np.log(y_hat + eps)) + (ones - y).transpose().dot(np.log(ones - y_hat + eps))
        return np.sum(log_loss_array) / ((-1) * arr_size) + lamdba_ * l2 / (2 * arr_size) 

    def fit_(self, x: np.ndarray, y: np.ndarray, alpha = 0.0001, n_cycle=10000, lambda_ = 0.0):
        # if y.ndim > 1:
        #     y = np.array([elem for lst in y for elem in lst])
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x))  # add intercept
        for i in range(n_cycle):
            y_hat = self.predict_(x)
            if self.penalty == 'l2':
                l2_cor = lambda_ * self.theta0(self.theta)
            elif self.penalty == 'none':
                l2_cor = 0
            gradient = (x_plus.transpose().dot(np.subtract(y_hat, y)) + l2_cor) / y.shape[0]   # to give us the direction to a better theta-i
            self.theta = self.theta - alpha * gradient # improve theta with alpha-step
        return self.theta

    def add_polynomial_features(self, x: np.ndarray, power: int) -> np.ndarray:
        if x.size == 0:
            return None
        copy_x = x
        for nb in range(2, power + 1):
            x = np.column_stack((x, copy_x ** nb))
        return x