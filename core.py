import random,numpy as np
import neuralNet as nn

##### Parameters
h, w = 32, 32
N1=2
N2=2
randomWeightsRange=1.0
alpha=0.01
mi=0,
batchSize=1
## Setting the random seed for testing purposes
random.seed(0)
##### Assisting functions

def unpickle():
	try:
	    import cPickle as pickle
	except:
	    import pickle

	fo = open("cifar_2class", 'rb')
	dict = pickle.load(fo)
	fo.close()
	return dict

#####

print("A neural net implementation for the Deep Learning class by Fuxin Lee, by Viktor Milleghere")

myCollection = unpickle()
training,trainingL,testing,testingL = myCollection['train_data'],myCollection['train_labels'],myCollection['test_data'],myCollection['test_labels']

myTrainingImages = []
myTrainingLabels = []

for k in range(int(training.size/3072)):

	#myTrainingImage = np.reshape(training[k],(32,32,3),order='F')
	myTrainingLabel = trainingL[k]+1

	#myImage = np.empty([32,32,3])
	#if (k % 500 == 0):
	#	print("Training data load at",(100.0*k/(training.size/3072.0)),"percent")
	#for i in range(0,w):
	#	for j in range(0,h):
	#		r,g,b = [myTrainingImage[i,j,0]/255.0,myTrainingImage[i,j,1]/255.0,myTrainingImage[i,j,2]/255.0]
	#		myImage[i,j] = [r,g,b]

	myTrainingImages.append(training[k])
	myTrainingLabels.append(myTrainingLabel[0])

myTestingImages = []
myTestingLabels = []

for k in range(int(testing.size/3072)):
	myTestingImage = testing[k].reshape((32,32,3))
	myTestingLabel = testingL[k]+1

	#myImage = np.empty([32,32,3])
	#if (k % 500 == 0):
	#	print("Testing data load at",(100.0*k/(testing.size/3072.0)),"percent")
	#for i in range(0,w):
	#	for j in range(0,h):
	#		r,g,b = [myTestingImage[i,j,0]/255.0,myTestingImage[i,j,1]/255.0,myTestingImage[i,j,2]/255.0]
	#		myImage[i,j] = [r,g,b]

	myTestingImages.append(testing[k])
	myTestingLabels.append(myTestingLabel[0])

print("Data loading completed successfully")

#Before batch training, we validate some important things

myFakeImage = np.array(([[[0.8,0.2,1.0]]]))
myFakeImage = myFakeImage.flatten()

myNN = nn.neuralNet(randomWeightsRange=randomWeightsRange,alpha=alpha,mi=mi,batchSize=batchSize)
myNN.activateNetwork(myFakeImage)
# myNN.transparent()

newWeights,Loss = myNN.batchUpdate([myFakeImage],[1])
myNN.activateNetwork(myFakeImage)
# myNN.transparent()

exit()

# Verify improvements on the testing set
sumC = 0
sum = 0
for i in range(0,int(testing.size/3072)):

	myImageFlattened = (testing[i]).flatten()
	myImageFlattened = np.multiply(myImageFlattened,1/255.0)
	myImageFlattened = np.append(myImageFlattened,(1))
	Activations = myNN.activateNetwork(myImageFlattened,Weights)
	if (Activations[2][0] > Activations[2][1]):
		classPrediction = 1
	else:
		classPrediction = 2

	if (classPrediction == testingL[i]+1):
		sumC += 1
	sum += 1

print("Initially testing error ratio is ",(1.0*sumC)/sum)



epochs = 100
for epoch in range(epochs):
	LossSum = 0
	##### Backpropagation
	random.shuffle(training)
	for batches in range(0,int(training.size/(3072*batchSize))):

		myBatchImages = []
		myBatchLabels = []
		for i in range(0,batchSize):

			##### add bias term before importing for training
			myImageFlattened = (myTrainingImages[batches*batchSize+i]).flatten()
			myImageFlattened = np.multiply(myImageFlattened,1/255.0)
			myImageFlattened = np.append(myImageFlattened,(1))

			myBatchImages.append(myImageFlattened)
			myBatchLabels.append(myTrainingLabels[batches*batchSize+i])
			#myLabels.append(random.randint(1,C))

		newWeights,Loss = myNN.batchUpdate(myBatchImages,myBatchLabels,Weights,alpha)
		Weights = newWeights

		LossSum += Loss
	print("Final training loss is averaging ",LossSum/(1.0*batches))
	LossSum = 0

	# Verify improvements on the testing set
	sumC = 0
	sum = 0
	for i in range(0,int(testing.size/3072)):

		myImageFlattened = (testing[i]).flatten()
		myImageFlattened = np.multiply(myImageFlattened,1/255.0)
		myImageFlattened = np.append(myImageFlattened,(1))

		Activations = myNN.activateNetwork(myImageFlattened,newWeights)
		if (Activations[2][0] > Activations[2][1]):
			classPrediction = 1
		else:
			classPrediction = 2

		if (classPrediction == testingL[i]+1):
			sumC += 1
		sum += 1

		testActivations = myNN.activateNetwork(myImageFlattened,Weights)
		Loss,kop = myNN.crossEntropyLoss(testActivations[2],testingL[i]+1)
		LossSum += Loss
	print("Final testing loss is averaging ",LossSum/(1.0*int(testing.size/3072)))

	print("Testing error ratio is ",1-(1.0*sumC)/sum,"for epoch",epoch)

	# Verify improvements on the training set (proper algorithm should improve this more)
	sumC = 0
	sum = 0
	for i in range(0,int(training.size/3072)):

		myImageFlattened = (training[i]).flatten()
		myImageFlattened = np.multiply(myImageFlattened,1/255.0)
		myImageFlattened = np.append(myImageFlattened,(1))

		Activations = myNN.activateNetwork(myImageFlattened,newWeights)
		if (Activations[2][0] > Activations[2][1]):
			classPrediction = 1
		else:
			classPrediction = 2
		#print(Activations[2][0],Activations[2][1],classPrediction,trainingL[i]+1)

		if (classPrediction == trainingL[i]+1):
			sumC += 1
		sum += 1

	print("Training error ratio is ",1-(1.0*sumC)/sum,"for epoch",epoch)
