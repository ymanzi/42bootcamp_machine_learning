import numpy as np 
from mylinearregression import MyLinearRegression as MLR

class MyRidge(MLR):
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """
    def __init__(self,  theta, alpha=0.001, n_cycle=1000, lambda_=0.5):
        super().__init__(theta, alpha, n_cycle)
        self.lambda_ = lambda_

    def theta0(self, theta):
        theta[0] = 0
        return theta

    def add_polynomial_features(self, x: np.ndarray, power: int) -> np.ndarray:
        if x.size == 0:
            return None
        copy_x = x
        for nb in range(2, power + 1):
            x = np.column_stack((x, copy_x ** nb))
        return x

    def cost_(self, x, y):
        array_size = y.shape[0]
        l2 = float(self.theta.transpose().dot(self.theta) - self.theta[0]**2)
        return (sum(self.cost_elem_(x, y)) + self.lambda_ * l2 ) / (2 * array_size)

    def gradient_(self, y, x):
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x)) # add intercept
        arr_size = y.shape[0] 
        return (x_plus.transpose().dot(np.subtract(x_plus.dot(self.theta) , y)) + self.lambda_ * self.theta0(self.theta))  / arr_size

    def fit_(self, x, y, alpha= 0.001, n_cycle = 10000):
        if y.ndim > 1:
            y = np.array([elem for lst in y for elem in lst])
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x)) # add intercept
        arr_size = y.shape[0]
        for i in range(n_cycle):
            x_theta = x_plus.dot(self.theta)
            gradient = (x_plus.transpose().dot(np.subtract(x_plus.dot(self.theta) , y)) \
                + self.lambda_ * self.theta0(self.theta))  / arr_size
            self.theta = np.subtract(self.theta, alpha * gradient)
        return self.theta

