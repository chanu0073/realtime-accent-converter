import math
import torch
from torch import nn

import commons
import attentions

from f0_encoder import F0Encoder
from text_prosody_encoder import TextProsodyEncoder


class NativeTextEncoder(nn.Module):
    """
    Figure-1 Encoder

    Text
      ↓
    TextEncoder
      ↓
    text_features

    F0
      ↓
    F0Encoder
      ↓
    f0_features

    text_features + f0_features
      ↓
    TextProsodyEncoder
      ↓
    fused_features

      ↓
    m_p, logs_p
    """

    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # --------------------------------------------------
        # Original VITS text embedding
        # --------------------------------------------------

        self.emb = nn.Embedding(
            n_vocab,
            hidden_channels
        )

        nn.init.normal_(
            self.emb.weight,
            0.0,
            hidden_channels ** -0.5
        )

        # --------------------------------------------------
        # Original VITS transformer encoder
        # --------------------------------------------------

        self.text_encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )

        # --------------------------------------------------
        # Figure-1 additions
        # --------------------------------------------------

        self.f0_encoder = F0Encoder(
            hidden_channels=hidden_channels
        )

        self.text_prosody_encoder = TextProsodyEncoder(
            hidden_channels=hidden_channels
        )

        # --------------------------------------------------
        # Original VITS projection
        # --------------------------------------------------

        self.proj = nn.Conv1d(
            hidden_channels,
            out_channels * 2,
            1
        )

    def forward(
        self,
        text,
        text_lengths,
        f0
    ):
        """
        text:
            [B, T_text]

        text_lengths:
            [B]

        f0:
            [B, 1, T_audio]

        Returns:
            fused_features,
            m,
            logs,
            x_mask
        """

        # ==================================================
        # Text path
        # ==================================================

        x = self.emb(text)

        x = x * math.sqrt(
            self.hidden_channels
        )

        x = torch.transpose(
            x,
            1,
            -1
        )  # [B,H,T]

        x_mask = torch.unsqueeze(
            commons.sequence_mask(
                text_lengths,
                x.size(2)
            ),
            1
        ).to(x.dtype)

        text_features = self.text_encoder(
            x * x_mask,
            x_mask
        )

        # ==================================================
        # F0 path
        # ==================================================

        f0_features = self.f0_encoder(
            f0
        )

        # ==================================================
        # Align lengths
        # ==================================================

        text_len = text_features.size(2)
        f0_len = f0_features.size(2)

        if f0_len != text_len:

            f0_features = torch.nn.functional.interpolate(
                f0_features,
                size=text_len,
                mode="linear",
                align_corners=False
            )

        # ==================================================
        # Fuse text + prosody
        # ==================================================

        fused_features = self.text_prosody_encoder(
            text_features,
            f0_features
        )

        fused_features = fused_features * x_mask

        # ==================================================
        # Original VITS prior projection
        # ==================================================

        stats = self.proj(
            fused_features
        ) * x_mask

        m, logs = torch.split(
            stats,
            self.out_channels,
            dim=1
        )

        return (
            fused_features,
            m,
            logs,
            x_mask
        )
