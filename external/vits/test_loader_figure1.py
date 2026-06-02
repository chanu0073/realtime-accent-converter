from data_utils_figure1 import TextAudioLoader
from data_utils_figure1 import TextAudioCollate

from utils import get_hparams_from_file

from torch.utils.data import DataLoader


hps = get_hparams_from_file(
    "configs/ljs_base.json"
)

dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

print("Dataset size:", len(dataset))

item = dataset[0]

print("\nSingle item\n")

for i, x in enumerate(item):
    print(i, x.shape)

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=TextAudioCollate()
)

batch = next(iter(loader))

print("\nBatch\n")

for i, x in enumerate(batch):
    print(i, x.shape)
