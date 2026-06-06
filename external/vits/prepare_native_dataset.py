import os
import random


DATASET_ROOT = "datasets/augmented_dataset"

METADATA_FILE = os.path.join(
    DATASET_ROOT,
    "metadata.csv"
)

TRAIN_FILE = "filelists/native_train.txt"
VAL_FILE = "filelists/native_val.txt"

VAL_RATIO = 0.02
SEED = 1234


def main():

    random.seed(SEED)

    samples = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("|")

            if len(parts) != 3:
                print(
                    f"Skipping malformed line: {line}"
                )
                continue

            wav_name = parts[0]
            speaker_id = parts[1]
            text = parts[2]

            wav_path = os.path.join(
                DATASET_ROOT,
                "wavs",
                wav_name
            )

            if not os.path.exists(wav_path):

                print(
                    f"Missing wav: {wav_path}"
                )
                continue

            samples.append(
                (
                    wav_path,
                    speaker_id,
                    text
                )
            )

    print(
        f"Loaded {len(samples)} samples"
    )

    random.shuffle(samples)

    val_size = int(
        len(samples) * VAL_RATIO
    )

    val_samples = samples[:val_size]

    train_samples = samples[val_size:]

    os.makedirs(
        "filelists",
        exist_ok=True
    )

    with open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for wav_path, speaker_id, text in train_samples:

            f.write(
                f"{wav_path}|{text}\n"
            )

    with open(
        VAL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for wav_path, speaker_id, text in val_samples:

            f.write(
                f"{wav_path}|{text}\n"
            )

    print(
        f"Train samples: {len(train_samples)}"
    )

    print(
        f"Val samples: {len(val_samples)}"
    )

    print(
        f"Saved {TRAIN_FILE}"
    )

    print(
        f"Saved {VAL_FILE}"
    )


if __name__ == "__main__":
    main()
