import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    """
    Produces speaker embedding g.

    Input:
        [B, 80, T]
        (mel spectrogram)

    Output:
        [B, 256]
    """

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, mel):
        x = self.conv(mel)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)

        return x