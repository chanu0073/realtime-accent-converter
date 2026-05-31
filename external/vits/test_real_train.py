import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import commons
import utils

from data_utils import (
    TextAudioLoader,
    TextAudioCollate
)

from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator
)

from losses import (
    discriminator_loss,
    generator_loss,
    feature_loss,
    kl_loss
)

from mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch
)

from text.symbols import symbols


DEVICE = "cuda"


# -----------------------------
# Load config
# -----------------------------
hps = utils.get_hparams_from_file(
    "configs/ljs_base.json"
)

# -----------------------------
# Dataset
# -----------------------------
dataset = TextAudioLoader(
    hps.data.training_files,
    hps.data
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=TextAudioCollate(),
    num_workers=0
)

batch = next(iter(loader))

(
    x,
    x_lengths,
    spec,
    spec_lengths,
    y,
    y_lengths
) = batch

x = x.to(DEVICE)
x_lengths = x_lengths.to(DEVICE)

spec = spec.to(DEVICE)
spec_lengths = spec_lengths.to(DEVICE)

y = y.to(DEVICE)
y_lengths = y_lengths.to(DEVICE)

print("Batch loaded")


# -----------------------------
# Models
# -----------------------------
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).to(DEVICE)

net_d = MultiPeriodDiscriminator().to(DEVICE)

print("Models created")


# -----------------------------
# Optimizers
# -----------------------------
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    lr=hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps
)

optim_d = torch.optim.AdamW(
    net_d.parameters(),
    lr=hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps
)


# -----------------------------
# Generator Forward
# -----------------------------
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
) = net_g(
    x,
    x_lengths,
    spec,
    spec_lengths
)

print("Generator forward OK")


# -----------------------------
# Mel Processing
# -----------------------------
mel = spec_to_mel_torch(
    spec,
    hps.data.filter_length,
    hps.data.n_mel_channels,
    hps.data.sampling_rate,
    hps.data.mel_fmin,
    hps.data.mel_fmax
)

y_mel = commons.slice_segments(
    mel,
    ids_slice,
    hps.train.segment_size // hps.data.hop_length
)

y_hat_mel = mel_spectrogram_torch(
    y_hat.squeeze(1),
    hps.data.filter_length,
    hps.data.n_mel_channels,
    hps.data.sampling_rate,
    hps.data.hop_length,
    hps.data.win_length,
    hps.data.mel_fmin,
    hps.data.mel_fmax
)

y_seg = commons.slice_segments(
    y,
    ids_slice * hps.data.hop_length,
    hps.train.segment_size
)

print("GT segment :", y_seg.shape)
print("Generated  :", y_hat.shape)


# ==================================================
# Discriminator Step
# ==================================================
optim_d.zero_grad()

y_d_hat_r, y_d_hat_g, _, _ = net_d(
    y_seg,
    y_hat.detach()
)

loss_disc, _, _ = discriminator_loss(
    y_d_hat_r,
    y_d_hat_g
)

loss_disc.backward()
optim_d.step()

print(
    f"Discriminator step OK "
    f"(loss={loss_disc.item():.4f})"
)


# ==================================================
# Generator Step
# ==================================================
optim_g.zero_grad()

y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(
    y_seg,
    y_hat
)

loss_dur = torch.sum(l_length)

loss_mel = (
    F.l1_loss(
        y_mel,
        y_hat_mel
    )
    * hps.train.c_mel
)

loss_kl = (
    kl_loss(
        z_p,
        logs_q,
        m_p,
        logs_p,
        z_mask
    )
    * hps.train.c_kl
)

loss_fm = feature_loss(
    fmap_r,
    fmap_g
)

loss_gen, _ = generator_loss(
    y_d_hat_g
)

loss_gen_all = (
    loss_gen
    + loss_fm
    + loss_mel
    + loss_dur
    + loss_kl
)

loss_gen_all.backward()
optim_g.step()

print(
    f"Generator step OK "
    f"(loss={loss_gen_all.item():.4f})"
)


# ==================================================
# Save Checkpoints
# ==================================================
torch.save(
    net_g.state_dict(),
    "test_G_0.pth"
)

torch.save(
    net_d.state_dict(),
    "test_D_0.pth"
)

print("\nCheckpoint save OK")
print("Saved:")
print("  test_G_0.pth")
print("  test_D_0.pth")

print("\nSUCCESS")
print("One complete training iteration executed.")