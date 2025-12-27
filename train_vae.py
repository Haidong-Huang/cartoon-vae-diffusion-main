import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import UnlabeledImageFolder, set_seed
from vae import VAE

# --- 0) Reproducibility ---
SEED = 42
set_seed(SEED)

# --- 1) Hyperparameters ---
DATA_ROOT  = "CartoonSet"
BATCH_SIZE = 64
IMAGE_SIZE = 64
Z_DIM      = 512
LR         = 1e-3
EPOCHS     = 50
KL_WEIGHT  = 1.0

RUN_DIR   = "runs"
MODEL_DIR = "models/vae"
# CKPT_PATH = os.path.join(MODEL_DIR, f"vae_z{Z_DIM}.ckpt.pth")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 2) Data ---
dataset = UnlabeledImageFolder(DATA_ROOT, image_size=IMAGE_SIZE)
# generator for reproducible shuffling
g = torch.Generator()
g.manual_seed(SEED)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    generator=g
)

# --- 3) Model & Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VAE(in_channels=3, image_size=IMAGE_SIZE, z_dim=Z_DIM).to(device)
opt    = optim.Adam(model.parameters(), lr=LR)

# --- 4) Loss function ---
mse = nn.MSELoss(reduction='sum')
def loss_fn(recon, x, mu, logvar):
    recon_loss = mse(recon, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KL_WEIGHT*kl, recon_loss, kl

# --- 5) Training loop ---
for epoch in range(1, EPOCHS+1):
    model.train()
    total, tot_rec, tot_kl = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for x in pbar:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, rec_l, kl_l = loss_fn(recon, x, mu, logvar)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total  += loss.item()
        tot_rec+= rec_l.item()
        tot_kl += kl_l.item()
        pbar.set_postfix({
            "L": total/len(dataset),
            "R": tot_rec/len(dataset),
            "K": tot_kl/len(dataset)
        })
model_name = f'{MODEL_DIR}/vae_zdim{Z_DIM}.pth'
torch.save(model.state_dict(), model_name)
print(f"→ Saved checkpoint: {model_name}")
