import torch 

x = torch.rand(1000,1000, requires_grad=True)
y = torch.rand(1000,1000, requires_grad=True)
z = torch.rand(1000,1000, requires_grad=True)

q = torch.matmul(x,y)
f = z * q  

mean_f = torch.mean(f)

mean_f.backward()

print(x.grad)