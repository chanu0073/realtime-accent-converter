import torch
import torch.nn as nn


class F0Encoder(nn.Module):
    """
    Encodes frame-level F0 contour into latent prosody features.

    Input:
        [B, 1, T]

    Output:
        [B, hidden_channels, T]
    """

    def __init__(self, hidden_channels=192):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, hidden_channels, kernel_size=3, padding=1)
        )

    def forward(self, f0):
        return self.net(f0)