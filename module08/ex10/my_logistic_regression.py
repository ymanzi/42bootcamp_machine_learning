import numpy as np

class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.theta = np.array(theta).reshape(-1, 1)

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

    def cost_(self, x: np.ndarray, y: np.ndarray, eps=1e-15):
        y_hat = self.predict_(x)
        ones = np.ones(y.shape)
        log_loss_array = y.transpose().dot(np.log(y_hat + eps)) + (ones - y).transpose().dot(np.log(ones - y_hat + eps))
        return np.sum(log_loss_array) / ((-1) * y.shape[0])

    def fit_(self, x: np.ndarray, y: np.ndarray, alpha = 0.0001, n_cycle=10000):
        # if y.ndim > 1:
        #     y = np.array([elem for lst in y for elem in lst])
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x))  # add intercept
        for i in range(n_cycle):
            y_hat = self.predict_(x)
            gradient = x_plus.transpose().dot(np.subtract(y_hat, y))  # to give us the direction to a better theta-i
            self.theta = self.theta - alpha * gradient # improve theta with alpha-step
        return self.theta

    def add_polynomial_features(self, x: np.ndarray, power: int) -> np.ndarray:
        if x.size == 0:
            return None
        copy_x = x
        for nb in range(2, power + 1):
            x = np.column_stack((x, copy_x ** nb))
        return x

# X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
# Y = np.array([[1], [0], [1]])
# mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])

# print(mylr.predict_(X))
# print(mylr.cost_(X,Y))
# mylr.fit_(X, Y)
# print(mylr.theta)
# print(mylr.predict_(X))
# print(mylr.cost_(X,Y))