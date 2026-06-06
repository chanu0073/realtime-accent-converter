import os
import glob
import random


ROOT = "/path/to/augmented_dataset"

OUTPUT_TRAIN = "filelists/native_train.txt"
OUTPUT_VAL = "filelists/native_val.txt"

VAL_SIZE = 100


all_entries = []


metadata_files = glob.glob(
    os.path.join(ROOT, "batch_*", "metadata.csv")
)

print("Found metadata files:", len(metadata_files))


for meta_path in metadata_files:

    batch_dir = os.path.dirname(meta_path)

    wav_dir = os.path.join(
        batch_dir,
        "wavs"
    )

    with open(meta_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split("|")

            if len(parts) != 3:
                continue

            wav_name, speaker_id, text = parts

            wav_path = os.path.join(
                wav_dir,
                wav_name
            )

            entry = f"{wav_path}|{text}"

            all_entries.append(entry)


print("Total samples:", len(all_entries))

random.shuffle(all_entries)

val_entries = all_entries[:VAL_SIZE]
train_entries = all_entries[VAL_SIZE:]


os.makedirs("filelists", exist_ok=True)

with open(OUTPUT_TRAIN, "w") as f:
    for x in train_entries:
        f.write(x + "\n")

with open(OUTPUT_VAL, "w") as f:
    for x in val_entries:
        f.write(x + "\n")

print("Train:", len(train_entries))
print("Val:", len(val_entries))
