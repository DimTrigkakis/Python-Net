import numpy as np
from matplotlib import pyplot as plt
import random
import math

##### Parameters
C = 2
N1,N2 = 3,C
randomWeightsRange = 1
alpha = 0.1
#####

## Setting the random seed for testing purposes
random.seed(6) ## This lets 2/3 neurons in the hidden layer have non-zero values after relu activation

## Debugging print functionality

myDebug = True
def show(*T):
	if (myDebug):
		for o in list(T):
			print o,
		print

## Neural Network functions
def gfunction(x):
	return x if x > 0 else 0

def gMfunction(X):
	fvectorized = np.vectorize(gfunction,otypes=[np.float])
	return fvectorized(X)


def ggradfunction(x):
	return 1 if x > 0 else 0

def ggradMfunction(X):
	fvectorized = np.vectorize(ggradfunction,otypes=[np.float])
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
	
	return loss,oneHot

def batchUpdate(myImages,myLabels,Activations,Weights,alpha):

	W0 = []
	W1 = []
	finalWeights = [W0, W1]
	for i in range(len(myImages)):
		# Calculate the loss

		Loss,oneHot = crossEntropyLoss(Activations[2],myLabels[i])

		E = []


		a0Shape = int(Activations[0].shape[0])
		activations0 = np.reshape(Activations[0],(a0Shape, 1))

		a1Shape = int(Activations[1].shape[0])
		activations1 = np.reshape(Activations[1],(a1Shape, 1))

		a2Shape = int(Activations[2].shape[0])
		activations2 = np.reshape(Activations[2],(a2Shape, 1))

		for j in range(len(Activations[2])):
			if (oneHot[j] == 1):
				E.append(-math.log(Activations[2][j]))
			else:
				E.append(-math.log(1-Activations[2][j]))

		outputDeltas = []

		for j in range(len(Activations[2])):
			print(oneHot[j],Activations[2][j])
			outputDeltas.append(oneHot[j] - Activations[2][j])

		outputDeltas = np.reshape(np.asarray(outputDeltas),(len(outputDeltas),1))

		WeightUpdatesHidden = np.copy(Weights[1]) # second set of weights
		size = WeightUpdatesHidden.shape

		WeightUpdatesHidden = activations1.dot(outputDeltas.transpose())

		# weightupdateshidden is deltas multiplied by the weights, just multiply with gradients and activations now
		# define the hidden deltas

		WeightUpdatesInput = np.copy(Weights[0]) # second set of weights
		size = WeightUpdatesInput.shape


		WeightUpdatesTemp = Weights[1].dot(outputDeltas)
		WeightUpdatesTemp = np.delete(WeightUpdatesTemp,3,0) # hardcoded 3 !!!! ALERT
		
		
		derivatives = ggradMfunction(activations1)
		derivatives = np.delete(derivatives,3,0)

		temp = np.multiply(WeightUpdatesTemp,derivatives)

		WeightUpdatesInput = activations0.dot(temp.transpose())
		
		print temp
		# remove last bias term from outputDeltas

		
		if (i == 0):
			# zeros for this 
			finalWeights[0] = np.zeros(Weights[0].shape)
			finalWeights[0] = np.add(finalWeights[0],WeightUpdatesInput)
			finalWeights[1] = np.zeros(Weights[1].shape)
			finalWeights[1] = np.add(finalWeights[1],WeightUpdatesHidden)
		else:
			finalWeights[0] = np.add(WeightUpdatesInput,finalWeights[0])
			finalWeights[1] = np.add(WeightUpdatesHidden,finalWeights[1])

		print "Initial Weights",Weights
		print "my updated weights",finalWeights

		m = -0.99

		updatedWeight = np.subtract(Weights,np.multiply(finalWeights,m))
	
		print updatedWeight


	return updatedWeight

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

print("A neural net implementation for the Deep Learning class by Fuxin Lee, by Viktor Milleghere (D.T.)")

h, w = 2, 2
myImage = np.empty([h, w, 3], dtype = np.float64)

for i in range(0,h):
	for j in range(0,w):
		r,g,b = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]
		myImage[i,j] = r,g,b

# plt.imshow(myImage, interpolation='nearest')
# plt.show()

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

show("Network Activations ---v")
show("Activations[0] ---v\n",Activations[0])
show("Activations[1] ---v\n",Activations[1])
show("Activations[2] ---v\n",Activations[2])

##### Backpropagation
batchSize = 1

myImages = []
myLabels = []
for i in range(batchSize):
	myImages.append(myImage)
	myLabels.append(random.randint(1,C))

newWeights = batchUpdate(myImages,myLabels,Activations,Weights,alpha)
Activations = activateNetwork(myImageFlattened,newWeights)


show("Network Activations ---v")
show("Activations[0] ---v\n",Activations[0])
show("Activations[1] ---v\n",Activations[1])
show("Activations[2] ---v\n",Activations[2])