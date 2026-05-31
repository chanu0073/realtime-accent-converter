import torch
from torch import nn

from native_text_encoder import NativeTextEncoder
from speaker_encoder import SpeakerEncoder


class NativeVITS(nn.Module):

    def __init__(
        self,
        n_vocab=101,
        hidden_channels=192,
        inter_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        speaker_embedding_dim=256
    ):
        super().__init__()

        self.enc_p = NativeTextEncoder(
            n_vocab=n_vocab,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )

        self.speaker_encoder = SpeakerEncoder()

        self.speaker_proj = nn.Linear(
            speaker_embedding_dim,
            hidden_channels
        )

    def forward(
        self,
        text,
        text_lengths,
        f0,
        ref_mel
    ):

        fused, m, logs, mask = self.enc_p(
            text,
            text_lengths,
            f0
        )

        speaker_embed = self.speaker_encoder(
            ref_mel
        )

        speaker_cond = self.speaker_proj(
            speaker_embed
        )

        return {
            "fused": fused,
            "m": m,
            "logs": logs,
            "mask": mask,
            "speaker_embed": speaker_embed,
            "speaker_cond": speaker_cond
        }