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


def vec_log_loss_(y: np.ndarray, y_hat: np.ndarray , eps=1e-15):
    if y.shape[0] != y_hat.shape[0]:
        return None
    ones = np.ones(y.shape)
    # log_loss_array = np.array([y * np.log(y_hat + eps) \
    #         + (1 - y) * np.log(1 - y_hat + eps)  for y, y_hat in zip(y, y_hat)])
    log_loss_array = y.transpose().dot(np.log(y_hat + eps)) + (ones - y).transpose().dot(np.log(ones - y_hat + eps))
    return np.sum(log_loss_array) / ((-1) * y.shape[0])

# y3 = np.array([[0], [1], [1]])
# x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
# theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# y_hat3 = logistic_predict(x3, theta3)
# print(vec_log_loss_(y3, y_hat3))
