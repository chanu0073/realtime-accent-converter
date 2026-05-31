from data_utils import TextAudioLoader
import utils

hps = utils.get_hparams_from_file(
    "./configs/ljs_base.json"
)

dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nReturned items:")

for i, item in enumerate(sample):
    try:
        print(
            f"{i}: {item.shape}"
        )
    except:
        print(
            f"{i}: {type(item)}"
        )
