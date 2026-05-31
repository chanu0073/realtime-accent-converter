import torch
import soundfile as sf

import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


CHECKPOINT = "./logs/ljs_test/G_1000.pth"
CONFIG = "./logs/ljs_test/config.json"

TEXT = "Hello, this is a test of VITS speech synthesis."


# -----------------------
# Load config
# -----------------------

hps = utils.get_hparams_from_file(CONFIG)

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Build model
# -----------------------

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).to(device)

net_g.eval()

print("Model created")


# -----------------------
# Load checkpoint
# -----------------------

utils.load_checkpoint(
    CHECKPOINT,
    net_g,
    None
)

print("Checkpoint loaded")


# -----------------------
# Text → sequence
# -----------------------

seq = text_to_sequence(
    TEXT,
    hps.data.text_cleaners
)

if hps.data.add_blank:
    seq = [item for s in seq for item in (0, s)][1:]

x = torch.LongTensor(seq).unsqueeze(0).to(device)
x_lengths = torch.LongTensor([x.size(1)]).to(device)

print("Token count:", x.size(1))


# -----------------------
# Inference
# -----------------------

with torch.no_grad():

    audio, attn, y_mask, extras = net_g.infer(
        x,
        x_lengths,
        noise_scale=0.667,
        length_scale=1.0,
        noise_scale_w=0.8
    )

audio = audio[0, 0].cpu().numpy()

print("Generated samples:", len(audio))


# -----------------------
# Save wav
# -----------------------

sf.write(
    "test.wav",
    audio,
    hps.data.sampling_rate
)

print("Saved test.wav")
