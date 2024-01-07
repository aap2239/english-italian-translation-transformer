import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)


z = y * y * 2
print(z)

z1 = z.mean()
print(z1)
v = torch.tensor([0.1, 0.1, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)
