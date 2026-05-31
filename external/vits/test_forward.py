import torch
from torch.utils.data import DataLoader

import utils
from data_utils import (
    TextAudioLoader,
    TextAudioCollate
)
from models import SynthesizerTrn
from text.symbols import symbols


hps = utils.get_hparams_from_file(
    "./configs/ljs_base.json"
)

dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=TextAudioCollate()
)

batch = next(iter(loader))

x, x_lengths, spec, spec_lengths, y, y_lengths = batch

model = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

with torch.no_grad():
    outputs = model(
        x,
        x_lengths,
        spec,
        spec_lengths
    )

print("Forward pass successful")

for i, item in enumerate(outputs):
    print(i, type(item))
