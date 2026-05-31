import torch
import torch.nn as nn


class TextProsodyEncoder(nn.Module):
    """
    Fuses text features and F0 features.

    text_feats : [B, 192, T]
    f0_feats   : [B, 192, T]

    output     : [B, 192, T]
    """

    def __init__(self, hidden_channels=192):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv1d(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, text_feats, f0_feats):
        x = torch.cat([text_feats, f0_feats], dim=1)
        return self.fusion(x)