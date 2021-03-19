import pandas as pd
import os

class FileLoader:
	def __init__(self):
		pass

	def load(self, path):
		"""
			load(path) : 
				takes as an argument the file path of the dataset to load, 
				displays a message specifying the dimensions of the dataset (e.g. 340 x 500) 
				and returns the dataset loaded as a pandas.DataFrame.
		"""
		if os.path.exists(path) and os.path.isfile(path):
			file = pd.read_csv(path)
			print("Loading dataset of dimensions {} x {}".format(file.shape[0], file.shape[1]))
		else:
			file = None
		return file
	
	def display(self, df, n):
		"""
			display(df, n) : 
				takes a pandas.DataFrame and an integer as arguments,
				displays the first n rows of the dataset if n is positive, 
				or the last n rows if n is negative.
		"""
		if (n > 0):
			print(df.head(n))
		elif (n < 0):
			print(df.tail(-n))