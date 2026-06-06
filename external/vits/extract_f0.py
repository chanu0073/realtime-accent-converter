import os
from pathlib import Path

import torch
import librosa
import pyworld as pw
import numpy as np


SR = 22050

DATASET_ROOT = (
    "/home/chanu/Files/projects/AIR/realtime-accent-converter/"
    "datasets/augmented_dataset"
)

wavs = list(
    Path(DATASET_ROOT).rglob("*.wav")
)

print(f"Found: {len(wavs)} wavs")


success = 0
failed = 0


for idx, wav_path in enumerate(wavs):

    wav_path = str(wav_path)

    save_path = wav_path.replace(
        ".wav",
        ".f0.pt"
    )

    # ----------------------------------
    # Skip existing files
    # ----------------------------------

    if os.path.exists(save_path):
        continue

    try:

        y, sr = librosa.load(
            wav_path,
            sr=SR
        )

        if len(y) == 0:
            raise RuntimeError(
                "Empty audio"
            )

        f0, t = pw.dio(
            y.astype(np.float64),
            sr
        )

        f0 = pw.stonemask(
            y.astype(np.float64),
            f0,
            t,
            sr
        )

        f0 = torch.FloatTensor(
            f0
        )

        torch.save(
            f0,
            save_path
        )

        success += 1

    except Exception as e:

        failed += 1

        print(
            f"[FAILED] {wav_path}"
        )

        print(e)

    if idx % 100 == 0:

        print(
            f"[{idx}/{len(wavs)}] "
            f"success={success} "
            f"failed={failed}"
        )

print("\nDone")

print(
    f"Generated: {success}"
)

print(
    f"Failed: {failed}"
)