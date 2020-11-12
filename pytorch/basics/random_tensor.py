import torch

random_tensor = torch.rand(3,3)
print(random_tensor)
print(random_tensor.shape)

# ones matrix
ones_tensor = torch.ones(3,3)
identity = torch.eye(3)

elementWise = ones_tensor * identity
print("element wise\n",elementWise)

matrixMul = torch.matmul(ones_tensor,identity)
print("matrix multiplication\n", matrixMul)