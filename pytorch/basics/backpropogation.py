import torch

x = torch.tensor(4.,requires_grad=True)
y = torch.tensor(-3.,requires_grad=True)
z = torch.tensor(5.,requires_grad=True)

q = x + y
f = q * z 

# for differentiate f w.r.t any value
f.backward()

print("Gradient w.r.t x is ",x.grad)
print("Gradient w.r.t y is ",y.grad)
print("Gradient w.r.t z is ",z.grad)