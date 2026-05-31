import utils

from models import SynthesizerTrn
from text.symbols import symbols

hps = utils.get_hparams_from_file(
    "./configs/ljs_base.json"
)

model = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: "
    f"{total_params:,}"
)
