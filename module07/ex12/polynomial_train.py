import numpy as np
import matplotlib.pyplot as plt

def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
	if x.size == 0:
		return None
	copy_x = x
	for nb in range(2, power + 1):
		x = np.column_stack((x, copy_x ** nb))
	return x

x = np.arange(1,11).reshape(-1,1)
y = np.array([[ 1.39270298],
           [ 3.88237651],
           [ 4.37726357],
           [ 4.63389049],
           [ 7.79814439],
           [ 6.41717461],
           [ 8.63429886],
           [ 8.19939795],
           [10.37567392],
           [10.68238222]])

# plt.scatter(x,y)
# plt.show()

from mylinearregression import MyLinearRegression as MyLR

# Build the model:
x_ = add_polynomial_features(x, 3)
my_lr = MyLR(np.ones(4).reshape(-1,1))
print(my_lr.thetas)
my_lr.fit_(x_, y, alpha=1e-6, n_cycle=5000000)
print(my_lr.thetas)


# Plot:
## To get a smooth curve, we need a lot of data points
continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
x_ = add_polynomial_features(continuous_x, 3)
y_hat = my_lr.predict_(x_)

plt.scatter(x,y)
plt.plot(continuous_x, y_hat, color='orange')
plt.show()