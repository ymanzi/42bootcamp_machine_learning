import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MLR

data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])
my_lreg = MLR([383.69840485, -24.29091709,   5.67467907,  -2.66502708])

# print(X.shape)

my_lreg.fit_(X,Y, alpha = 6.5e-5, n_cycle = 60000)
print(my_lreg.thetas)
print(my_lreg.mse_(X,Y))

X = np.array(data[['Age']])
my_lreg.plot_points(X, Y)