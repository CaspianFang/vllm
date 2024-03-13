import torch

x = torch.rand(3,5)
w_a = torch.rand(5,1,2,5)
w_b = torch.rand(5,1,4,2)
indeices = torch.tensor([[2,3],[0,4],[1,3]])


y = torch.rand(3,4)

print(y)

for batch_index in range(x.shape[0]):
        for indicies_index in range(indeices.shape[-1]):
            if (indeices[batch_index][indicies_index] > -1):
                y[batch_index] += x[batch_index] @ (w_a[indeices[batch_index][indicies_index]].squeeze().transpose(-1, -2) @ w_b[indeices[batch_index][indicies_index]].squeeze().transpose(-1, -2))
print(y)