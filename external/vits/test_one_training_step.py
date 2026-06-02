import torch
from torch.utils.data import DataLoader

from utils import get_hparams_from_file
from data_utils_figure1 import (
    TextAudioLoader,
    TextAudioCollate
)

from native_synthesizer import NativeSynthesizer
from text.symbols import symbols


CONFIG_PATH = "configs/ljs_base.json"


def main():

    print("Loading config...")

    hps = get_hparams_from_file(
        CONFIG_PATH
    )

    print("Loading dataset...")

    dataset = TextAudioLoader(
        hps.data.training_files,
        hps.data
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=TextAudioCollate(),
        num_workers=0
    )

    print("Loading model...")

    model = NativeSynthesizer(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,

        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,

        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,

        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,

        gin_channels=256,
        use_sdp=True,
    )

    model.train()

    print("Getting batch...")

    (
        text,
        text_lengths,

        spec,
        spec_lengths,

        mel,
        mel_lengths,

        wav,
        wav_lengths,

        f0,
        f0_lengths,
    ) = next(iter(loader))

    print("\nBatch shapes")
    print("text         ", text.shape)
    print("spec         ", spec.shape)
    print("mel          ", mel.shape)
    print("wav          ", wav.shape)
    print("f0           ", f0.shape)

    print("\nRunning forward pass...")

    (
        y_hat,
        l_length,
        attn,
        ids_slice,
        x_mask,
        z_mask,
        (
            z,
            z_p,
            m_p,
            logs_p,
            m_q,
            logs_q
        )
    ) = model(
        text,
        text_lengths,

        spec,
        spec_lengths,

        mel,
        mel_lengths,

        f0
    )

    print("Forward OK")

    print("\nOutput shapes")
    print("y_hat        ", y_hat.shape)
    print("attn         ", attn.shape)
    print("z            ", z.shape)
    print("z_p          ", z_p.shape)

    print("\nCreating dummy loss...")

    loss = (
        y_hat.abs().mean()
        + z.abs().mean()
        + z_p.abs().mean()
    )

    print("Loss =", loss.item())

    print("\nRunning backward pass...")

    loss.backward()

    print("Backward OK")

    print("\nSUCCESS")
    print("Forward + Backward completed")


if __name__ == "__main__":
    main()
