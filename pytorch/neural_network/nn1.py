import torch


inputLayer = torch.rand(1,4)

# 4 layer nn
w1 = torch.rand(4,4)
w2 = torch.rand(4,4)
w3 = torch.rand(4,4)

layer1 = torch.matmul(inputLayer,w1)
layer2 = torch.matmul(layer1,w2)
outputLayer = torch.matmul(layer2,w3)

weight = torch.matmul(torch.matmul(w1,w2),w3)
netoutput = torch.matmul(inputLayer,weight)

print(outputLayer)
print(netoutput)

''' both the outputs are same '''