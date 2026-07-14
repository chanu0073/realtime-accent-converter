import torch
import torchaudio

from torch.utils.data import DataLoader

from utils import get_hparams_from_file, load_checkpoint
from data_utils_figure1 import (
    TextAudioLoader,
    TextAudioCollate
)

from native_synthesizer import NativeSynthesizer
from text.symbols import symbols
from mel_processing import mel_spectrogram_torch


CONFIG_PATH = "configs/native_vits.json"

CHECKPOINT_PATH = "logs/figure1_nogan/G_10000.pth"


print("Loading config...")

hps = get_hparams_from_file(CONFIG_PATH)

device = "cuda"


print("Loading dataset...")

dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=TextAudioCollate(),
    num_workers=0
)

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
).to(device)


print("Loading checkpoint...")

model, _, _, _ = load_checkpoint(
    CHECKPOINT_PATH,
    model,
    None
)

model.eval()


text = text.to(device)
text_lengths = text_lengths.to(device)

spec = spec.to(device)
spec_lengths = spec_lengths.to(device)

mel = mel.to(device)
mel_lengths = mel_lengths.to(device)

f0 = f0.to(device)


print("Running TRAINING forward pass...")

with torch.no_grad():

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

audio = y_hat[0][0].cpu()

print("\nWaveform samples")

print(audio[:50])

print("\nAudio stats")
print(audio.min())
print(audio.max())
print(audio.abs().mean())
print(audio.shape)

print("\nLatent stats")

print("z mean abs:", z.abs().mean())
print("z_p mean abs:", z_p.abs().mean())

print("m_p mean abs:", m_p.abs().mean())
print("logs_p mean abs:", logs_p.abs().mean())

print("m_q mean abs:", m_q.abs().mean())
print("logs_q mean abs:", logs_q.abs().mean())

generated_mel = mel_spectrogram_torch(
    y_hat.squeeze(1),
    hps.data.filter_length,
    hps.data.n_mel_channels,
    hps.data.sampling_rate,
    hps.data.hop_length,
    hps.data.win_length,
    hps.data.mel_fmin,
    hps.data.mel_fmax
)

print("\nGenerated mel stats")

print("generated mel abs mean:",
      generated_mel.abs().mean())

print("real mel abs mean:",
      mel.abs().mean())

torchaudio.save(
    "decoder_only.wav",
    (audio * 100).unsqueeze(0),
    hps.data.sampling_rate
)

print("\nSaved decoder_only.wav")