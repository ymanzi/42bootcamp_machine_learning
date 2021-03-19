import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MLR
from my_logistic_regression import data_spliter

def check_positive_negative(y: np.ndarray, y_hat: np.ndarray, categorie):
    dic_pos_neg = { "true positives" : 0,
                    "false positives": 0,
                    "true negatives": 0,
                    "false negatives": 0}
    for e_real, e_predict in zip(y, y_hat):
        if e_real == e_predict and e_real == categorie:
            dic_pos_neg["true positives"] += 1
        elif e_real == e_predict and e_real != categorie:
            dic_pos_neg["true negatives"] += 1
        elif e_real != e_predict and e_real == categorie:
            dic_pos_neg["false negatives"] += 1
        elif e_real != e_predict and e_predict == categorie:
            dic_pos_neg["false positives"] += 1
    return dic_pos_neg

def add_polynomial_features_extend(x, power):
    if x.size == 0:
        return None
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    columns_list = []
    for i in range(x.shape[1]):
        columns_list.append(x[:,i])
    for nb in range(2, power + 1):
        for elem in columns_list:
            x = np.column_stack((x, elem ** nb))
    return x

def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
    if x.size == 0:
        return None
    copy_x = x
    for nb in range(2, power + 1):
        x = np.column_stack((x, copy_x ** nb))
    return x

def one_versus_all_train(data, lambda_):
    cat_unique = np.unique(data[2])
    val = np.ones(data[0].shape[0])
    prob = np.zeros(data[0].shape[0])
    test = val.astype(float)
    y_train = data[2]
    x_train = add_polynomial_features_extend(data[0], 3)
    theta_list = []
    for planete in cat_unique:
        verif = y_train == planete
        y_zero_train = verif.astype(float)
        mlr = MLR(np.ones(x_train.shape[1] + 1))
        mlr.fit_(x_train, y_zero_train, alpha= 5e-15, n_cycle=100000, lambda_=lambda_)
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

def accuracy_score_(y: np.ndarray, y_hat: np.ndarray):
    result = np.array([e1 == e2 for e1, e2 in zip(y, y_hat)]).astype(int)
    return np.sum(result) / result.size

def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false positives"])

def recall_score_(y, y_hat, pos_label=1):
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false negatives"])


def f1_score_(y, y_hat, pos_label=1):
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return (2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label)) /\
         (precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label))

def cost_(y_hat: np.ndarray, y: np.ndarray, eps=1e-15):
        ones = np.ones(y.shape)
        log_loss_array = y.transpose().dot(np.log(y_hat + eps)) + (ones - y).transpose().dot(np.log(ones - y_hat + eps))
        return np.sum(log_loss_array) / ((-1) * y.shape[0])

# data = pd.read_csv("../resources/solar_system_census.csv")
# data = np.array(data.drop(data.columns[0], axis=1))

# y = pd.read_csv("../resources/solar_system_census_planets.csv")
# y = np.array(y.drop(y.columns[0], axis=1))

# ret = data_spliter(data, y, 0.6)

# with open('test.npy', 'wb') as f:
#     np.save(f, ret, allow_pickle=True)

with open('test.npy', 'rb') as f:
    ret = np.load(f, allow_pickle=True)

ret_train = one_versus_all_train(ret, 0)
y_hat_train = ret_train[0]
# print("thetas: ", ret_train[1])
y_hat_test = one_versus_all_test(ret, ret_train[1])

# print(y_hat_train.shape, "  train  ", ret[2].shape)
# print(y_hat_test.shape, "  test  ", ret[3].shape)
print(y_hat_test)

print("train f1_score: ", f1_score_(ret[2], y_hat_train, pos_label=2.0))
print("test f1_score: ", f1_score_(ret[3], y_hat_test, pos_label=2.0))