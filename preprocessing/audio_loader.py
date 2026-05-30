from pathlib import Path

import librosa
import numpy as np


class AudioLoader:
    """
    Loads audio files into a standardized format.

    Standard:
    - 16 kHz
    - Mono
    - float32
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load(self, audio_path: str) -> np.ndarray:
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, _ = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True
        )

        return waveform.astype(np.float32)

    def get_duration(self, waveform: np.ndarray) -> float:
        return len(waveform) / self.sample_rate