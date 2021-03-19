import numpy as np

def theta0(theta):
    theta[0] = 0
    return theta

def sigmoid_(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return None
    x = x.astype(np.float)
    if x.ndim == 0:
        x = np.array(x, ndmin=1)
    return (1 / (1 + (np.exp(x * -1))))


def reg_logistic_grad(y, x, theta, lambda_):
    x_plus = np.column_stack((np.full((x.shape[0], theta.shape[0] - x.shape[1]) , 1), x))
    y_hat = sigmoid_(x_plus.dot(theta))
    res = x_plus.transpose().dot(np.subtract(y_hat, y)) + lambda_ * theta0(theta)
    return res / x.shape[0]

# x = np.array([[0, 2, 3, 4], 
#               [2, 4, 5, 5], 
#               [1, 3, 2, 7]])
# y = np.array([[0], [1], [1]])
# theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

# print(reg_logistic_grad(y, x, theta, 0.0))