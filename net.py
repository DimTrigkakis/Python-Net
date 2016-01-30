import numpy as np
from matplotlib import pyplot as plt
import random

##### Parameters
N1,N2 = 2,1
randomWeightsRange = 1
alpha = 0.1
#####

## Neural Network functions
def gfunction(x):

	return x if x > 0 else 0

def gMfunction(X):
	fvectorized = np.vectorize(gfunction,otypes=[np.float])
	return fvectorized(X)

def innerInitializeWeights(W):
	
	size = W.shape
	for i in range(size[0]):
		for j in range(size[1]):
			W[i,j] = random.uniform(-randomWeightsRange,randomWeightsRange)

def initializeWeights(W):

	for i in range(len(W)):
		innerInitializeWeights(W[i])

	return

def batchUpdate(myImages,Activations,Weights,alpha):

	for i in range(len(myImages)):
		# Calculate error
		E += 1

	# Backpropagate error to calculate deltas in each layer
	# Calculate weight updates
	# Update weights based on alpha
	
	
	return

#####
def showWeight(W):

	size = W.shape
	newImage = np.empty([size[0],size[1],3],np.float64)
	difference = W.max() - W.min()

	if difference == 0:
		return

	for i in range(0,size[0]):
		for j in range(0,size[1]):			
			a = ((W[i,j]-W.min())*255)/255
			r,g,b = a,random.randint(0,255),a
			newImage[i,j] = r,g,b

	plt.imshow(newImage, interpolation="nearest")
	plt.xlim([0,size[0]])
	plt.show()

#####
def activateNetwork(input,W):
	
	print "In network"
	print input.shape,W[0].shape
	Act1 = gMfunction(np.dot(input,W[0]))
	
	##### adding the bias term to the activation
	Act1 = np.append(Act1,gfunction(1))
	print Act1.shape,W[1].shape
	Act2 = gMfunction(np.dot(Act1,W[1]))

	return input,Act1,Act2
#####

print("A neural net implementation for the Deep Learning class by Fuxin Lee, by Dimitris Trigkakis")

h, w = 2, 2
myImage = np.empty([h, w, 3], dtype = np.float64)

for i in range(0,h):
	for j in range(0,w):
		r,g,b = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]
		myImage[i,j] = r,g,b

plt.imshow(myImage, interpolation='nearest')
plt.show()

##### Network initialization
# ----weights include the bias weights
Win = np.empty([(h*w*3+1),N1], dtype = np.float64)
Wout = np.empty([N1+1,N2], dtype = np.float64)
Weights = [Win,Wout]
initializeWeights(Weights)

##### add bias term
myImageFlattened = myImage.flatten()
myImageFlattened = np.append(myImageFlattened,(1))

Activations = activateNetwork(myImageFlattened,Weights)

print Activations[0]
print Activations[1]
print Activations[2]

##### Backpropagation
batchSize = 1

for i in range(batchSize):
	myImages.append(myImage)

batchUpdate(myImages,Activations,Weights,alpha)

		




