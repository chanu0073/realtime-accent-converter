import torch
import torchaudio
import librosa
import pyworld as pw
import numpy as np

from text import cleaned_text_to_sequence
from text.symbols import symbols

from native_synthesizer import NativeSynthesizer
from utils import get_hparams_from_file, load_checkpoint
from phonemizer.backend import EspeakBackend


CONFIG_PATH = "configs/native_vits.json"

CHECKPOINT_PATH = "logs/figure1_nogan/G_3000.pth"

REFERENCE_WAV = "/host/home/dc/lv01-server/accent_conversion/augmented_dataset/batch_00001/wavs/utt_000001.wav"

OUTPUT_WAV = "generated.wav"

TEXT = "Hello how are you today"


print("Loading config...")

hps = get_hparams_from_file(
    CONFIG_PATH
)

device = "cuda"


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

for name, param in model.named_parameters():
    print(name, param.abs().mean().item())
    break

model.eval()

backend = EspeakBackend(
    language="en-us",
    preserve_punctuation=False,
    with_stress=True
)


print("Preparing text...")

phonemes = backend.phonemize(
    [TEXT],
    strip=True
)[0]

print("Phonemes:")
print(phonemes)

text_norm = cleaned_text_to_sequence(
    phonemes
)

text = torch.LongTensor(
    text_norm
).unsqueeze(0).to(device)

text_lengths = torch.LongTensor(
    [text.size(1)]
).to(device)


print("Loading reference wav...")

wav, sr = librosa.load(
    REFERENCE_WAV,
    sr=hps.data.sampling_rate
)

wav_torch = torch.FloatTensor(
    wav
).unsqueeze(0)


print("Creating mel...")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=hps.data.sampling_rate,
    n_fft=hps.data.filter_length,
    hop_length=hps.data.hop_length,
    win_length=hps.data.win_length,
    n_mels=hps.data.n_mel_channels,
)

mel = mel_transform(
    wav_torch
)

mel = torch.log(
    torch.clamp(mel, min=1e-5)
)

mel = mel.to(device)

mel_lengths = torch.LongTensor(
    [mel.size(2)]
).to(device)


print("Extracting F0...")

f0, t = pw.dio(
    wav.astype(np.float64),
    sr
)

f0 = pw.stonemask(
    wav.astype(np.float64),
    f0,
    t,
    sr
)

f0 = torch.FloatTensor(
    f0
).unsqueeze(0).unsqueeze(0).to(device)


print("Running inference...")

with torch.no_grad():

    y_hat, attn, y_mask, latent = model.infer(
        text,
        text_lengths,

        mel,
        mel_lengths,

        f0,

        noise_scale_w=1.0,
        length_scale=3.0
    )

audio = y_hat[0][0].cpu()

print(audio.min())
print(audio.max())
print(audio.abs().mean())

print("Saving audio...")

torchaudio.save(
    OUTPUT_WAV,
    audio.unsqueeze(0),
    hps.data.sampling_rate
)

print("DONE")
print("Saved:", OUTPUT_WAV)