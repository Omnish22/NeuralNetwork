import torch 
import matplotlib.pyplot as plt 
import numpy as np 

image = plt.imread("dog.jpg")
inputLayer = np.array(image).reshape(1,-1)
inputLayer = torch.from_numpy(inputLayer)

w1 = torch.rand(150738,200) #784 features and 200 units in first layer
w2 = torch.rand(200,10) # 10 outputs 

hiddenLayer1 = torch.matmul(inputLayer,w1)
outputLayer = torch.matmul(hiddenLayer1,w2)

print(outputLayer)