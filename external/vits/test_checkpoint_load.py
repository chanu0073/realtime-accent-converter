import torch
import utils

from models import SynthesizerTrn
from text.symbols import symbols


CHECKPOINT = "./logs/ljs_test/G_1000.pth"
CONFIG = "./logs/ljs_test/config.json"


hps = utils.get_hparams_from_file(CONFIG)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

print("Model created")

utils.load_checkpoint(
    CHECKPOINT,
    net_g,
    None
)

print("Checkpoint loaded successfully")

total_params = sum(p.numel() for p in net_g.parameters())

print(f"Parameters: {total_params:,}")
