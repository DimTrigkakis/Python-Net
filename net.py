import numpy as np
from matplotlib import pyplot as plt
import random

print("A neural net implementation for the Deep Learning class by Fuxin Lee, by Dimitris Trigkakis")

w, h = 512, 512
myImage = np.empty([h, w, 3], dtype = np.uint8)

for i in range(0,h):
	for j in range(0,w):
		r,g,b = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
		myImage[i,j] = r,g,b

plt.imshow(myImage, interpolation='nearest')
plt.show()



		




