import torch

from native_text_encoder import NativeTextEncoder

model = NativeTextEncoder(
    n_vocab=101,
    out_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1
)

text = torch.randint(
    0,
    100,
    (2, 50)
)

text_lengths = torch.LongTensor(
    [50, 45]
)

f0 = torch.randn(
    2,
    1,
    400
)

x, m, logs, mask = model(
    text,
    text_lengths,
    f0
)

print("fused :", x.shape)
print("m     :", m.shape)
print("logs  :", logs.shape)
print("mask  :", mask.shape)
