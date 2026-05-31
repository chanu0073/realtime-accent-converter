import torch

from text_prosody_encoder import TextProsodyEncoder

model = TextProsodyEncoder()

text_feats = torch.randn(2, 192, 400)
f0_feats   = torch.randn(2, 192, 400)

y = model(text_feats, f0_feats)

print("Text :", text_feats.shape)
print("F0   :", f0_feats.shape)
print("Out  :", y.shape)
