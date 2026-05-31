import torch

from f0_encoder import F0Encoder

model = F0Encoder()

x = torch.randn(2, 1, 400)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)
