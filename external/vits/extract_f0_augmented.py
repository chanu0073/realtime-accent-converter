import os
import glob

import torch
import librosa
import pyworld as pw
import numpy as np


SR = 22050


wavs = glob.glob(
    "/path/to/augmented_dataset/batch_*/wavs/*.wav"
)

print("Found:", len(wavs))


for idx, wav_path in enumerate(wavs):

    y, sr = librosa.load(
        wav_path,
        sr=SR
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

    f0 = torch.FloatTensor(f0)

    save_path = wav_path.replace(
        ".wav",
        ".f0.pt"
    )

    torch.save(
        f0,
        save_path
    )

    if idx % 500 == 0:
        print(idx, "/", len(wavs))

print("Done")
