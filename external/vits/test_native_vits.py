import torch

from native_vits import NativeVITS


model = NativeVITS()

text = torch.randint(
    low=0,
    high=100,
    size=(2, 50)
)

text_lengths = torch.LongTensor(
    [50, 45]
)

f0 = torch.randn(
    2,
    1,
    400
)

ref_mel = torch.randn(
    2,
    80,
    400
)

outputs = model(
    text,
    text_lengths,
    f0,
    ref_mel
)

print()
print("===== Native VITS Test =====")
print()

for k, v in outputs.items():
    print(
        f"{k:15s}",
        v.shape
    )
