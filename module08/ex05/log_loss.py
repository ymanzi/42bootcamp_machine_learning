import numpy as np

def sigmoid_(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return None
    x = x.astype(np.float)
    if x.ndim == 0:
        x = np.array(x, ndmin=1)
    return (1 / (1 + (np.exp(x * -1))))


def logistic_predict(x, theta):
    if (x.ndim == 1):
        x = x.reshape(-1, 1)
    x_plus = np.column_stack((np.full((x.shape[0], theta.shape[0] - x.shape[1]) , 1), x)) 
    x_theta = x_plus.dot(theta).reshape(-1, 1)
    return sigmoid_(x_theta)


def log_loss_(y: np.ndarray, y_hat: np.ndarray , eps=1e-15):
    if y.shape[0] != y_hat.shape[0]:
        return None
    log_loss_array = np.array([elem_y * np.log(elem_y_hat + eps) \
            + (1 - elem_y) * np.log(1 - elem_y_hat + eps)  for elem_y, elem_y_hat in zip(y, y_hat)])
    return np.sum(log_loss_array) / ((-1) * y.shape[0])

# y3 = np.array([[0], [1], [1]])
# x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
# theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# y_hat3 = logistic_predict(x3, theta3)
# print(log_loss_(y3, y_hat3))
