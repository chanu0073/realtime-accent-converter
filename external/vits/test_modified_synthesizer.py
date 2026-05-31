import torch

from models import SynthesizerTrn
from text.symbols import symbols

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SynthesizerTrn(
    len(symbols),
    513,
    32,
    192,
    192,
    768,
    2,
    6,
    3,
    0.1,
    "1",
    [3, 7, 11],
    [[1,3,5],[1,3,5],[1,3,5]],
    [8,8,2,2],
    512,
    [16,16,4,4],
    n_speakers=0,
    gin_channels=256
).to(device)

print("Model created")

B = 2

text = torch.randint(
    0,
    len(symbols),
    (B,50)
).to(device)

text_lengths = torch.LongTensor(
    [50,45]
).to(device)

spec = torch.randn(
    B,
    513,
    400
).to(device)

spec_lengths = torch.LongTensor(
    [400,380]
).to(device)

f0 = torch.randn(
    B,
    1,
    50
).to(device)

ref_mel = torch.randn(
    B,
    80,
    400
).to(device)

with torch.no_grad():

    outputs = model(
        text,
        text_lengths,
        spec,
        spec_lengths,
        f0,
        ref_mel
    )

print("Forward pass successful")

for i, item in enumerate(outputs):
    if torch.is_tensor(item):
        print(i, item.shape)
    else:
        print(i, type(item))
