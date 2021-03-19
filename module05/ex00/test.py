#!/usr/bin/python

from vector import Vector

v1 = [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
v2 = [[3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]
v3 = Vector(v1).T()

print(v3.shape)


