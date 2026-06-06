import torch

from native_synthesizer import NativeSynthesizer
from utils import get_hparams_from_file
from text.symbols import symbols


print("Loading config...")

hps = get_hparams_from_file(
    "configs/native_vits.json"
)

print("Creating model...")

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

    gin_channels=hps.model.gin_channels,
    use_sdp=hps.model.use_sdp,
)

model.eval()

print("Creating dummy inputs...")

text = torch.randint(
    0,
    len(symbols),
    (1, 100)
)

text_lengths = torch.LongTensor(
    [100]
)

mel = torch.randn(
    1,
    80,
    400
)

mel_lengths = torch.LongTensor(
    [400]
)

f0 = torch.randn(
    1,
    1,
    400
)

print("Running inference...")

with torch.no_grad():

    y_hat, x_mask, latent = model.infer(
        text,
        text_lengths,

        mel,
        mel_lengths,

        f0
    )

print("\nInference successful\n")

print("Output shapes\n")

print("y_hat      ", y_hat.shape)
print("x_mask     ", x_mask.shape)

z, z_p, m_p, logs_p = latent

print("z           ", z.shape)
print("z_p         ", z_p.shape)
print("m_p         ", m_p.shape)
print("logs_p      ", logs_p.shape)