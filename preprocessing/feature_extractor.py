import librosa
import numpy as np


class FeatureExtractor:
    """
    Extracts speech features used throughout the project.

    Features:
    - Mel Spectrogram
    - Pitch (F0)
    - Energy
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract_mel(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract 80-bin Mel Spectrogram.

        Returns:
            shape -> (n_mels, time_frames)
        """

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        return mel_db.astype(np.float32)

    def extract_f0(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract pitch contour.

        Returns:
            shape -> (time_frames,)
        """

        f0, _, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        f0 = np.nan_to_num(f0)

        return f0.astype(np.float32)

    def extract_energy(self, waveform: np.ndarray) -> np.ndarray:
        """
        Frame-level energy.

        Returns:
            shape -> (time_frames,)
        """

        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )

        return rms.squeeze().astype(np.float32)

    def extract_all(self, waveform: np.ndarray):
        """
        Extract all features at once.
        """

        return {
            "mel": self.extract_mel(waveform),
            "f0": self.extract_f0(waveform),
            "energy": self.extract_energy(waveform),
        }