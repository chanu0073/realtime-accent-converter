import matplotlib.pyplot as plt
import librosa.display
import numpy as np


class Visualizer:

    @staticmethod
    def plot_mel(mel: np.ndarray):
        """
        Display Mel Spectrogram.
        """

        plt.figure(figsize=(12, 4))

        librosa.display.specshow(
            mel,
            sr=16000,
            hop_length=320,
            x_axis="time",
            y_axis="mel"
        )

        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_f0(f0: np.ndarray):
        """
        Display Pitch contour.
        """

        plt.figure(figsize=(12, 4))

        f0_plot = f0.copy()
        f0_plot[f0_plot == 0] = np.nan
        plt.plot(f0_plot)

        plt.title("Pitch (F0)")
        plt.xlabel("Frame")
        plt.ylabel("Hz")

        plt.tight_layout()
        plt.show()