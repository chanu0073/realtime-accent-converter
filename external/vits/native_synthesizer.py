import math
import torch
from torch import nn

import commons
import monotonic_align

from models import (
    PosteriorEncoder,
    ResidualCouplingBlock,
    Generator,
    StochasticDurationPredictor,
    DurationPredictor,
)

from native_text_encoder import NativeTextEncoder
from speaker_encoder import SpeakerEncoder


class NativeSynthesizer(nn.Module):
    """
    Figure-1 Native TTS Model
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=256,
        use_sdp=True,
        n_layers_q=3,
        use_spectral_norm=False,
    ):
        super().__init__()

        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        # --------------------------------------------------
        # Figure 1 modules
        # --------------------------------------------------

        self.enc_p = NativeTextEncoder(
            n_vocab=n_vocab,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            out_channels=inter_channels,
        )

        self.speaker_encoder = SpeakerEncoder(
            embedding_dim=256
        )

        self.spk_proj = nn.Linear(
            256,
            gin_channels
        )

        # --------------------------------------------------
        # Original VITS modules
        # --------------------------------------------------

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
        )

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                inter_channels,
                192,
                3,
                0.5,
                4,
                gin_channels=gin_channels,
            )
        else:
            self.dp = DurationPredictor(
                inter_channels,
                256,
                3,
                0.5,
                gin_channels=gin_channels,
            )

    def forward(
        self,
        text,
        text_lengths,

        spec,
        spec_lengths,

        mel,
        mel_lengths,

        f0,
    ):

        # ----------------------------------------
        # Figure 1 encoder
        # ----------------------------------------

        fused, m_p, logs_p, x_mask = self.enc_p(
            text,
            text_lengths,
            f0,
        )

        spk_emb = self.speaker_encoder(mel)

        g = self.spk_proj(spk_emb)
        g = g.unsqueeze(-1)

        # ----------------------------------------
        # Posterior encoder
        # ----------------------------------------

        z, m_q, logs_q, y_mask = self.enc_q(
            spec,
            spec_lengths,
            g=g,
        )

        z_p = self.flow(
            z,
            y_mask,
            g=g,
        )

        # ----------------------------------------
        # Alignment (same as VITS)
        # ----------------------------------------

        with torch.no_grad():

            s_p_sq_r = torch.exp(-2 * logs_p)

            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p,
                [1],
                keepdim=True,
            )

            neg_cent2 = torch.matmul(
                -0.5 * (z_p ** 2).transpose(1, 2),
                s_p_sq_r,
            )

            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2),
                (m_p * s_p_sq_r),
            )

            neg_cent4 = torch.sum(
                -0.5 * (m_p ** 2) * s_p_sq_r,
                [1],
                keepdim=True,
            )

            neg_cent = (
                neg_cent1
                + neg_cent2
                + neg_cent3
                + neg_cent4
            )

            attn_mask = (
                torch.unsqueeze(x_mask, 2)
                * torch.unsqueeze(y_mask, -1)
            )

            attn = (
                monotonic_align.maximum_path(
                    neg_cent,
                    attn_mask.squeeze(1),
                )
                .unsqueeze(1)
                .detach()
            )

        # ----------------------------------------
        # Duration loss
        # ----------------------------------------

        w = attn.sum(2)

        if self.use_sdp:

            l_length = self.dp(
                fused,
                x_mask,
                w,
                g=g,
            )

            l_length = (
                l_length
                / torch.sum(x_mask)
            )

        else:

            logw_ = torch.log(
                w + 1e-6
            ) * x_mask

            logw = self.dp(
                fused,
                x_mask,
                g=g,
            )

            l_length = torch.sum(
                (logw - logw_) ** 2,
                [1, 2],
            ) / torch.sum(x_mask)

        # ----------------------------------------
        # Expand prior
        # ----------------------------------------

        m_p = torch.matmul(
            attn.squeeze(1),
            m_p.transpose(1, 2),
        ).transpose(1, 2)

        logs_p = torch.matmul(
            attn.squeeze(1),
            logs_p.transpose(1, 2),
        ).transpose(1, 2)

        # ----------------------------------------
        # Decoder
        # ----------------------------------------

        z_slice, ids_slice = commons.rand_slice_segments(
            z,
            mel_lengths,
            self.segment_size,
        )

        audio = self.dec(
            z_slice,
            g=g,
        )

        return (
            audio,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (
                z,
                z_p,
                m_p,
                logs_p,
                m_q,
                logs_q,
            ),
        )
