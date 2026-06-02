from native_synthesizer import NativeSynthesizer
from utils import get_hparams_from_file
from text.symbols import symbols

import torch


hps = get_hparams_from_file(
    "configs/ljs_base.json"
)

model = NativeSynthesizer(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

text = torch.randint(
    0,
    len(symbols),
    (2, 100)
)

text_lengths = torch.LongTensor(
    [100, 80]
)

mel = torch.randn(
    2,
    80,
    400
)

mel_lengths = torch.LongTensor(
    [400, 350]
)

f0 = torch.randn(
    2,
    1,
    400
)

spec = torch.randn(
    2,
    513,
    400
)

spec_lengths = torch.LongTensor(
    [400, 350]
)

mel = torch.randn(
    2,
    80,
    400
)

mel_lengths = torch.LongTensor(
    [400, 350]
)

f0 = torch.randn(
    2,
    1,
    400
)

out = model(
    text,
    text_lengths,

    spec,
    spec_lengths,

    mel,
    mel_lengths,

    f0
)

print("Forward successful")
