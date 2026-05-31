import logging

logging.getLogger("numba").setLevel(
    logging.WARNING
)

import torch
from torch.utils.data import DataLoader
from text.symbols import symbols

import utils

from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
)

from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)

from losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)

from mel_processing import (
    mel_spectrogram_torch,
)


device = "cuda"


# -------------------------
# Load config
# -------------------------
hps = utils.get_hparams_from_file(
    "configs/ljs_base.json"
)

# -------------------------
# Dataset
# -------------------------
dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

collate_fn = TextAudioCollate()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

batch = next(iter(loader))

(
    x,
    x_lengths,
    spec,
    spec_lengths,
    y,
    y_lengths,
) = batch

x = x.to(device)
x_lengths = x_lengths.to(device)

spec = spec.to(device)
spec_lengths = spec_lengths.to(device)

y = y.to(device)
y_lengths = y_lengths.to(device)

print("Batch loaded")

# -------------------------
# Models
# -------------------------
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size //
    hps.data.hop_length,
    **hps.model
).to(device)

net_d = MultiPeriodDiscriminator(
    hps.model.use_spectral_norm
).to(device)

print("Models created")

# -------------------------
# Optimizers
# -------------------------
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    lr=hps.train.learning_rate,
    betas=hps.train.betas,
)

optim_d = torch.optim.AdamW(
    net_d.parameters(),
    lr=hps.train.learning_rate,
    betas=hps.train.betas,
)

# -------------------------
# Forward pass
# -------------------------
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
        logs_q,
    ),
) = net_g(
    x,
    x_lengths,
    spec,
    spec_lengths,
)

import commons

y = commons.slice_segments(
    y,
    ids_slice * hps.data.hop_length,
    hps.train.segment_size
)

print("GT segment:", y.shape)
print("Generated :", y_hat.shape)

print("Generator forward OK")

# -------------------------
# Discriminator
# -------------------------
y_d_hat_r, y_d_hat_g, _, _ = net_d(
    y,
    y_hat.detach(),
)

loss_disc, _, _ = discriminator_loss(
    y_d_hat_r,
    y_d_hat_g,
)

optim_d.zero_grad()
loss_disc.backward()
optim_d.step()

print(
    f"Discriminator step OK "
    f"(loss={loss_disc.item():.4f})"
)

# -------------------------
# Generator
# -------------------------
y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(
    y,
    y_hat,
)

loss_gen, _ = generator_loss(
    y_d_hat_g
)

loss_fm = feature_loss(
    fmap_r,
    fmap_g,
)

loss_kl = kl_loss(
    z_p,
    logs_q,
    m_p,
    logs_p,
    z_mask,
)

loss_g = (
    loss_gen
    + loss_fm
    + loss_kl
    + l_length.mean()
)

optim_g.zero_grad()
loss_g.backward()
optim_g.step()

print(
    f"Generator step OK "
    f"(loss={loss_g.item():.4f})"
)

print("\nSUCCESS")
print("Forward pass")
print("Backward pass")
print("Optimizer step")
print("All working")
