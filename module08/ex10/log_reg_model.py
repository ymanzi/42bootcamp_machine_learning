import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MLR
from data_spliter import data_spliter

def one_versus_all_train(data):
    cat_unique = np.unique(data[2])
    val = np.ones(data[0].shape[0])
    prob = np.zeros(data[0].shape[0])
    test = val.astype(float)
    y_train = data[2]
    x_train = data[0]
    theta_list = []
    for planete in cat_unique:
        verif = y_train == planete
        y_zero_train = verif.astype(float)
        mlr = MLR(np.ones(x_train.shape[1] + 1))
        mlr.fit_(x_train, y_zero_train, alpha= 3/10000000, n_cycle=100000)
        y_hat = mlr.predict_(x_train)
        for i, planete_prob in enumerate(y_hat.tolist()):
            # print(prob[i],"\n", planete_prob)
            if prob[i] < planete_prob[0]:
                val[i] = planete
                prob[i] = planete_prob[0]
        theta_list.append(mlr.theta)
    ret_list = []
    ret_list.append(val)
    ret_list.append(theta_list)
    return ret_list

def one_versus_all_test(data, result):
    cat_unique = np.unique(data[2])
    val = np.ones(data[1].shape[0])
    prob = np.zeros(data[1].shape[0])
    test = val.astype(float)
    y_test = data[3]
    x_test = data[1]
    for j, planete in enumerate(cat_unique):
        verif = y_test == planete
        y_zero_test = verif.astype(float)
        mlr = MLR(result[j])
        # mlr.fit_(x_train, y_zero_train, alpha= 4/1000000, n_cycle=1000000)
        y_hat = mlr.predict_(x_test)
        for i, planete_prob in enumerate(y_hat.tolist()):
            if prob[i] < planete_prob[0]:
                val[i] = planete
                prob[i] = planete_prob[0]
    return val


def cost_(y_hat: np.ndarray, y: np.ndarray, eps=1e-15):
        ones = np.ones(y.shape)
        log_loss_array = y.transpose().dot(np.log(y_hat + eps)) + (ones - y).transpose().dot(np.log(ones - y_hat + eps))
        return np.sum(log_loss_array) / ((-1) * y.shape[0])

data = pd.read_csv("../resources/solar_system_census.csv")
data = np.array(data.drop(data.columns[0], axis=1))

y = pd.read_csv("../resources/solar_system_census_planets.csv")
y = np.array(y.drop(y.columns[0], axis=1))

# ret = data_spliter(data, y, 0.6)

# with open('test.npy', 'wb') as f:
#     np.save(f, ret, allow_pickle=True)

with open('test.npy', 'rb') as f:
    ret = np.load(f, allow_pickle=True)

ret_train = one_versus_all_train(ret)
y_hat_train = ret_train[0]
# print("thetas: ", ret_train[1])
y_hat_test = one_versus_all_test(ret, ret_train[1])

print(y_hat_train.shape, "  train  ", ret[2].shape)
print(y_hat_test.shape, "  test  ", ret[3].shape)

print("train cost: ", cost_(y_hat_train, ret[2]))
print("test cost: ", cost_(y_hat_test, ret[3]))

# x_train = ret[0]
# y_train = ret[2]

# x_test = ret[1]
# y_test = ret[3]

# a = (y_train == 0.0)
# b = (y_test == 0.0)

# y_zero_train = a.astype(float)
# y_zero_test = b.astype(float)

# mlr = MLR(np.ones(x_train.shape[1] + 1))

# x_pol = mlr.add_polynomial_features(x_train, 1)
# mlr.fit_(x_pol, y_zero_train, alpha= 3/1000000, n_cycle=10000)
# print (x_train.shape)
# y_hat = mlr.predict_(x_pol)

# print(list(zip(y_zero_train.tolist(), y_hat.tolist())))
# print(mlr.cost_(x_pol, y_zero_train))
# print(mlr.cost_(x_test, y_zero_test))
