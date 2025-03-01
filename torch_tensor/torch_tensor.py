import torch

x = torch.tensor([[0, 0, 1],[1, 1, 1],[0, 0, 0]])
print(x)

x[0][0] = 5
print(x)