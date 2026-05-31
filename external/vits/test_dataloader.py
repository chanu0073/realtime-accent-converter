from torch.utils.data import DataLoader

from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
)

import utils

hps = utils.get_hparams_from_file(
    "./configs/ljs_base.json"
)

dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

collate_fn = TextAudioCollate()

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

batch = next(iter(loader))

print("Batch contents:")

for i, item in enumerate(batch):
    try:
        print(i, item.shape)
    except:
        print(i, type(item))
