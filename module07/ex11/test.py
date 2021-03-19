import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MLR

def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
	if x.size == 0:
		return None
	copy_x = x
	for nb in range(1, power + 1):
		x = np.column_stack((x, copy_x ** nb))
	return x

data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])
reg = MLR([1.0, 1.0,   1.0,  1.0])

# power = 1
# reg.fit_(X,Y, alpha = 6.5e-5, n_cycle = 1000000)
# print("cost ", power, " : ", reg.cost_(X, Y), "\n")

power = 2
reg.thetas = np.full(3 ** power + 1, 1.0)
P = add_polynomial_features(X, power)
reg.fit_(P, Y, alpha = 1e-9, n_cycle = 5000)
print("cost ", power, " : ", reg.cost_(P, Y), "\n")

# power = 3
# reg.thetas = np.full(4 * power + 1, 1.0)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 3e-13, n_cycle = 100000)
# print("cost ", power, " : ", reg.cost_(reg.predict_(P), Y), "\n")

# power = 4
# reg.thetas = np.full(4 * power, 1)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 1e-17, n_cycle = 1000000)
# print("cost ", power, " : ", reg.cost_(reg.predict_(P), Y), "\n")

# power = 5
# reg.thetas = np.full(power * 3 + 1, 1.0)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 1e-22, n_cycle = 30000)
# print("cost ", power, " : ", reg.cost_(reg.predict_(P), Y), "\n")

# power = 6
# reg.thetas = np.full(power * 3 + 1, 1.0)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 1e-26, n_cycle = 6000)
# print("cost ", power, " : ", reg.cost_(reg.predict_(P), Y), "\n\n")

# power = 7
# reg.thetas = np.full(power * 3 + 1, 1.0)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 1e-32, n_cycle = 60000)
# print("cost ", power, " : ", reg.cost_(reg.predict_(P), Y), "\n\n")

# power = 8
# reg.thetas = np.full(30, 1.0)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 1e-35, n_cycle = 600000)
# print("cost ", power, " : ", reg.cost_(P, Y), "\n\n")

# power = 9
# reg.thetas = np.full(34, 1.0)
# P = add_polynomial_features(X, power)
# reg.fit_(P, Y, alpha = 1e-40, n_cycle = 60000)
# print("cost ", power, " : ", reg.cost_(P, Y), "\n\n")

# power = 10
# reg.thetas = np.full(34 , 1)
# P = add_polynomial_features(X, power)
# # print(P.shape)
# reg.fit_(P, Y, alpha = 1.3e-44, n_cycle = 100000)
# print("cost ", power, " : ", reg.cost_(P, Y), "\n")