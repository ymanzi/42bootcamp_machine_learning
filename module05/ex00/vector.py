#/goinfre/miniconda3/bin/python
#!/usr/bin/python

class Vector:
	def __init__(self, values):
		""" 
			Vector class that allow to make operation between them 
			self.values
			self.shape 
		"""

		if isinstance(values, int):
			self.values = [ [float(elem)] for elem in range(values) ]
		elif (isinstance(values, list) and (len(values) > 0) 
				and (isinstance(values[0], float))):
			self.values = [[float(elem)] for elem in values]
		elif (isinstance(values, list) and (len(values) > 0) 
				and (isinstance(values[0], list))):
			self.values = values
		else:
			raise TypeError("Wrong Init Variables")
		self.shape = (len(self.values), len(self.values[0]))
		
	def __str__(self):
		return ("Vector {}".format(self.values))

	def __repr__(self):
		print(self)

	def __add__(self, oth):
		""" vector + int or vector + vector operation """
		if (isinstance(oth, Vector) and self.shape == oth.shape):
			return Vector([[a1 + a2 for a1, a2 in zip(e1, e2)] for e1, e2 in zip(self.values, oth.values)] )
		elif (isinstance(oth, int)):
			return Vector([[elem2 + oth for elem2 in lst] for lst in self.values])
		else:
			raise TypeError("Wrong variables for addition")
	
	def __radd__(self, oth):
		""" int + vector operation """
		return self + oth

	def __sub__(self, oth):
		""" vector - int or vector - vector operation """
		if (isinstance(oth, Vector) and self.shape == oth.shape):
			return Vector([[a1 - a2 for a1, a2 in zip(e1, e2)] for e1, e2 in zip(self.values, oth.values)] )
		elif (isinstance(oth, int)):
			return Vector([[elem2 - oth for elem2 in lst] for lst in self.values])
		else:
			raise TypeError("Wrong variables for substraction")
	
	def __rsub__(self, oth):
		""" int - vector operation """
		return ((-1) * self + oth)
	
	def __truediv__(self, oth):
		""" vector / int operation """
		if (isinstance(oth, int) and oth != 0):
			return Vector([[elem2 / oth for elem2 in lst] for lst in self.values])
		else:
			raise TypeError("Wrong variables for division")
	
	def __rtruediv__(self, oth):
		""" int / vector operation """
		return self / oth
	
	def __mul__(self, oth):
		""" vector * int operation """
		if (isinstance(oth, int)):
			return Vector([[elem2 * oth for elem2 in lst] for lst in self.values])
		else:
			raise TypeError("Wrong variables for addition")
	
	def __rmull__(self, oth):
		""" int * vector operation """
		return self * oth

	def dot(self, oth):
		"""
			method which produce a dot product between two vectors of same dimensions.
		"""	
		if (isinstance(oth, Vector) and self.shape == oth.shape):
			return Vector([[a1 * a2 for a1, a2 in zip(e1, e2)] for e1, e2 in zip(self.values, oth.values)] )
		else:
			raise TypeError("Only for mult of 2 vectors of same size")

	def T(self):
		"""
			method which converts a vector 
			into its transpose (i.e. a column vector 
			into a row vector, or a row vector into a column vector).
		"""
		tmp_v = []
		check = True
		for lst in self.values:
			for i in range(len(lst)):
				if check:
					tmp_v.append([])
				tmp_v[i].append(lst[i])
			check = False
		return Vector(tmp_v)