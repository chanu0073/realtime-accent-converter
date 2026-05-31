import torch

from speaker_encoder import SpeakerEncoder

model = SpeakerEncoder()

mel = torch.randn(2, 80, 400)

g = model(mel)

print("Input :", mel.shape)
print("Output:", g.shape)
