from matplotlib import pyplot as plt
import random,math,numpy as np
from incredibleMathFunctions import incredibleMathFunctions as imf

class Layer(object):
	def __init__(self,myNN,name,inputSize,ourSize):

		self.type = [name[1],name[2]]
		self.inputSize = inputSize
		self.ourSize = ourSize

		if not (self.type[0] == "Input" or self.type[0] == "Output"):
			# inputSize+1 for the bias
			self.weights = myNN.initializeWeights(inputSize+1,ourSize)
		else:
			self.weights = None

		self.activations = None
		self.netActivations = None
		self.deltas = None

	def showNetActivations(self):

		show("Net activations ",self.netActivations)

	def showActivations(self):

		show("Final activations ",self.activations)

	def showWeights(self):

		show("Weights",self.weights)

class neuralNet(object):

	########## Static functions
	##### Callable weight function to initialize layer weights

	def initializeWeights(self,inputSize,outputSize):
	
		W = np.empty((inputSize,outputSize),dtype = np.float64)

		for i in range(inputSize):
			for j in range(outputSize):
				W[i,j] = random.uniform(-self.randomWeightsRange,self.randomWeightsRange)

		return W

	########## Member functions
	##### Network member variable initialization

	def __init__(self,N=[[2,"Input","Empty"],[2,"Linear Transform","Relu"],[2,"Linear Transform","Softmax"],["CrossEntropyLoss"]],randomWeightsRange=1,alpha=0.1,mi=0,batchSize=1):

		self.randomWeightsRange = randomWeightsRange
		self.alpha = alpha
		self.mi = mi
		self.batchSize = batchSize
		self.layerList = []

		for n in range(0,len(N)-1):
			if (n < len(N) and 0 < n):
				l = Layer(self,N[n],N[n-1][0],N[n][0])
			if (n == 0):
				l = Layer(self,N[n],None,N[n][0])
			
			self.layerList.append(l)

		self.lossFunction = N[len(N)-1]
		self.activated = False	

	def transparent(self,weights=False):

		show("\n### Network transparency")
		for layer in self.layerList:
			show("### ### For layer ",layer.type)
			layer.showNetActivations()
			layer.showActivations()
			if (weights == True):
				layer.showWeights()

	##### One hot encoding, where target activations are in the range 1 to C

	def oneHotEncoding(self,targetActivation):

		oneHot = []
		myList = self.layerList
		for i in range(1,myList[len(myList)-1].ourSize+1):
			if (targetActivation == i):
				oneHot.append(1)
			else:
				oneHot.append(0)

		return oneHot

	##### Loss function

	def crossEntropyLoss(self,targetActivation):
	
		oneHot = self.oneHotEncoding(targetActivation)
		loss = -math.log(np.dot(oneHot,self.layerList[len(self.layerList)-1].activations))
	
		return loss

	##### Feedforward activation

	def activateLayer(self,currLayer,prevLayer):

		prevActivations = prevLayer.activations

		#####
		if (currLayer.type[0] == "Linear Transform"):
			currLayer.netActivations = imf.linearTransform(prevActivations,currLayer.weights)

		##### transform the net activations based on the activation function
		if (currLayer.type[1] == "Relu"):
			currLayer.activations = imf.reluMfunction(currLayer.netActivations)
			currLayer.activations = np.append(currLayer.activations,imf.reluFunction(1))
		elif (currLayer.type[1] == "Softmax"): # Assuming softmax is output layer, so no bias to be added
			currLayer.activations = imf.softmaxMFunction(currLayer.netActivations)
		elif (currLayer.type[1] == "Empty"):
			currLayer.activations = currLayer.netActivations
			currLayer.activations = np.append(currLayer.activations,1)

	def activateNetwork(self,input):

		for l in range(len(self.layerList)):

			currLayer = self.layerList[l]

			if (l == 0):
				currLayer.activations = input
			else:
				prevLayer = self.layerList[l-1]
				self.activateLayer(currLayer,prevLayer)

		self.activated = True

	##### Delta calculation

	def calculateDeltas(self,currLayer,prevLayer,label):

		layerType = currLayer.type[1]

		if (layerType == "Softmax"):
			currLayer.deltas = np.subtract(currLayer.activations,oneHotEncoding(label))

		elif (layerType == "Relu"):

			# These will grab the deltas of the previous layer and multiply by the current weights
			WeightUpdatesTemp = Weights.dot(outputDeltas)
			WeightUpdatesTemp = np.delete(WeightUpdatesTemp,WeightUpdatesTemp.size-1,0)
		
			# Relu derivatives, matrix wise
			derivatives = self.ggradMfunction(Activations)
			derivatives = np.delete(derivatives,WeightUpdatesTemp.size-1,0)

			# Multiply the sum of deltas at a node with the relu gradient
			temp = np.multiply(WeightUpdatesTemp,derivatives)

			return temp

	##### Batch update for the network

	def batchUpdate(self,myImages,myLabels):

		LossSum = 0

		for i in range(len(myImages)):

			print("Batch update initiating")

			self.activateNetwork(myImages[i])
			Loss = self.crossEntropyLoss(myLabels[i])
			LossSum + Loss

			E = []

			for l in range(len(self.layerList)-1,0,-1):

				currLayer = self.layerList[l]
				print(currLayer.type[0])
				if (l == len(self.layerList)-1):
					self.calculateDeltas(currLayer,None)
				else:
					prevLayer = self.layerList[l+1]
					self.calculateDeltas(currLayer,prevLayer)



			outputDeltas = self.calculateDeltas("SoftmaxCrossentropy",Activations[2],myOutLabels=myLabels[i])
			WeightUpdatesHidden = activations[1].dot(outputDeltas.transpose())

			outputDeltasHidden = self.calculateDeltas("Relu",activations[1],Weights=Weights[1],outputDeltas=outputDeltas)
			WeightUpdatesInput = activations[0].dot(outputDeltasHidden.transpose())
		
			if (i == 0): # Batch update will simply add up the weights without updating until we are done
				finalWeights[0] = np.zeros(Weights[0].shape)
				finalWeights[0] = np.add(finalWeights[0],WeightUpdatesInput)
			finalWeights[1] = np.zeros(Weights[1].shape)
			finalWeights[1] = np.add(finalWeights[1],WeightUpdatesHidden)
		else: # 
			finalWeights[0] = np.add(WeightUpdatesInput,finalWeights[0])
			finalWeights[1] = np.add(WeightUpdatesHidden,finalWeights[1])

		# We do not want to update the biases
		sizes = finalWeights[0].shape,finalWeights[1].shape
		finalWeights[0][sizes[0][0]-1] = np.zeros(sizes[0][1])
		finalWeights[1][sizes[1][0]-1] = np.zeros(sizes[1][1])


		# Updating the weight using mi value (for momentum)
		updatedWeight = np.subtract(Weights,np.multiply(finalWeights,alpha))
		try:
			updatedWeight = np.add(updatedWeight,np.multiply(updatedWeightPrevious,mi))
		except NameError:
			updatedWeightPrevious = np.zeros(updatedWeight.shape)

		updatedWeightPrevious = np.array(finalWeights)

		return updatedWeight,LossSum/self.batchSize


## Debugging functionality, outside class

myDebug = True
def show(*T):
	if (myDebug):
		for o in list(T):
			print(o,)
		print()

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

##### End of neural net implementation <3