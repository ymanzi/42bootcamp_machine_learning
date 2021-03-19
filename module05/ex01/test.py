#!/usr/bin/python

from matrix import Matrix
from matrix import Test

v1 = [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
v2 = [[0.0], [0.0]]

v3 = Matrix(v1) * Matrix(v2)
print(v3)

# m1 =  Matrix([[0., 2., 4.], [1., 3., 5.]])
# print(v3)

