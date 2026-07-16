import torch
import torch.nn as nn


class F0Encoder(nn.Module):
    """
    Encodes frame-level F0 contour into latent prosody features.

    Input:
        [B, 1, T]   — raw F0 in Hz (0 = unvoiced)

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

    def forward(self, f0, f0_lengths=None):
        f0 = torch.log1p(f0)

        if f0_lengths is not None:
            B, C, T = f0.shape
            mask = torch.arange(T, device=f0.device).unsqueeze(0) < f0_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            f0 = f0 * mask

        return self.net(f0)
