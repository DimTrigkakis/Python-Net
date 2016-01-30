import numpy as np
from matplotlib import pyplot as plt
import random
import math

##### Parameters
N1,N2 = 2,10
randomWeightsRange = 1
C = 10 
alpha = 0.1
#####

## Neural Network functions
def gfunction(x):

	return x if x > 0 else 0

def gMfunction(X):
	fvectorized = np.vectorize(gfunction,otypes=[np.float])
	return fvectorized(X)


def gSoftmaxfunction(x,Sum):

	return (math.exp(x)/Sum)

def gMSoftmaxfunction(X):

	expVectorized = np.vectorize(math.exp,otypes=[np.float])
	expX = expVectorized(X)
	Sum = np.sum(expX)

	fvectorized = np.vectorize(gSoftmaxfunction,otypes=[np.float])
	return fvectorized(X,Sum)

def innerInitializeWeights(W):
	
	size = W.shape
	for i in range(size[0]):
		for j in range(size[1]):
			W[i,j] = random.uniform(-randomWeightsRange,randomWeightsRange)

def initializeWeights(W):

	for i in range(len(W)):
		innerInitializeWeights(W[i])

	return

# target activation is a class from 1 to C
def crossEntropyLoss(finalActivations,targetActivation):
	
	oneHot = []
	for i in range(1,C+1):
		if (targetActivation == i):
			oneHot.append(1)
		else:
			oneHot.append(0)
	
	loss = -math.log(np.dot(oneHot,finalActivations))
	
	return loss

def batchUpdate(myImages,myLabels,Activations,Weights,alpha):

	Loss = 0
	for i in range(len(myImages)):
		# Calculate the loss
		Loss += crossEntropyLoss(Activations[2],myLabels[i])

	
	# Backpropagate loss to calculate deltas in each layer
	

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
	
	Act1 = gMfunction(np.dot(input,W[0]))
	
	##### adding the bias term to the activation
	Act1 = np.append(Act1,gfunction(1))

	Act2 = np.dot(Act1,W[1])
	Act2 = gMSoftmaxfunction(Act2)
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

myImages = []
myLabels = []
for i in range(batchSize):
	myImages.append(myImage)
	myLabels.append(random.randint(1,10))

batchUpdate(myImages,myLabels,Activations,Weights,alpha)

		




