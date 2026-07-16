import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    """
    Produces speaker embedding g.

    Input:
        [B, 80, T]
        (mel spectrogram)

        [B]
        (mel_lengths — number of valid frames per sample)

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

        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, mel, mel_lengths=None):
        x = self.conv(mel)

        if mel_lengths is not None:
            B, C, T = x.shape
            mask = torch.arange(T, device=x.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            x = x * mask
            denom = mel_lengths.float().unsqueeze(1).unsqueeze(2).clamp(min=1)
            x = x.sum(dim=2, keepdim=True) / denom
        else:
            x = x.mean(dim=2, keepdim=True)

        x = x.squeeze(-1)
        x = self.proj(x)

        return x
