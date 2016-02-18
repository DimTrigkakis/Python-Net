import numpy as np
import math

class incredibleMathFunctions(object):

##### Math functions
	@staticmethod
	def linearTransform(x,w):
		return np.dot(x,w)

	@staticmethod
	def reluFunction(x):
		return x if x > 0 else 0

	@staticmethod
	def reluMfunction(X):
		fvectorized = np.vectorize(incredibleMathFunctions.reluFunction,otypes=[np.float])
		return fvectorized(X)

	@staticmethod
	def reluGradFunction(x):
		return 1 if x > 0 else 0

	@staticmethod
	def reluGradMFunction(X):
		fvectorized = np.vectorize(incredibleMathFunctions.reluGradfunction,otypes=[np.float])
		return fvectorized(X)

	@staticmethod
	def softmaxFunction(x,Sum):

		if Sum != 0:
			return (math.exp(x)/Sum)
		else:
			return 1.0

	@staticmethod
	def softmaxMFunction(X):

		expVectorized = np.vectorize(math.exp,otypes=[np.float])
		try:
			expX = expVectorized(X)
		except OverflowError:
			expX = 0
			print("Overflow error occured in imf, line 44 for the exp function")

		Sum = np.sum(expX)

		fvectorized = np.vectorize(incredibleMathFunctions.softmaxFunction,otypes=[np.float])
		return fvectorized(X,Sum)