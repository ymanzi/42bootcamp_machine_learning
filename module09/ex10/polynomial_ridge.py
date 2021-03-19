import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ridge import MyRidge as MR
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

def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
    if x.size == 0:
        return None
    copy_x = x
    for nb in range(2, power + 1):
        x = np.column_stack((x, copy_x ** nb))
    return x

# file_data = pd.read_csv("../resources/spacecraft_data.csv")
# y = np.array(file_data["Sell_price"]).reshape(-1, 1)
# x = np.array(file_data[["Age","Thrust_power","Terameters"]])
# ret_split = data_spliter(x, y, 0.6)
# with open('test.npy', 'wb') as f:
#     np.save(f, ret_split, allow_pickle=True)

with open('test.npy', 'rb') as f:
    ret_split = np.load(f, allow_pickle=True)

x_train = ret_split[0]
y_train = ret_split[2]

lambda_list = []
mse_list = []

x_train = add_polynomial_features(x_train, 3)

for tmp_lambda in np.arange(0, 1.1, 0.1):
    myridge = MR(np.ones((y_train.shape[0] + 1, y_train.shape[1])), lambda_= tmp_lambda)
    myridge.fit_(x_train, y_train, 3e-13, 1000)
    lambda_list.append(tmp_lambda)
    mse_list.append(myridge.mse_ (x_train, y_train))
# print(mse_list)

plt.plot(lambda_list, mse_list)
plt.show()
