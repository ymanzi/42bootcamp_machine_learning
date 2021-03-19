import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power given in argument.  
    Args:
      x: has to be an numpy.ndarray, a matrix of dimension m * n.
      power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
      The matrix of polynomial features as a numpy.ndarray, of dimension m * (np), containg the polynomial feature values for all training examples.
      None if x is an empty numpy.ndarray.
    Raises:
      This function should not raise any Exception.
    """
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

# x = np.arange(1,11).reshape(5, 2)
# print(add_polynomial_features(x, 5))
