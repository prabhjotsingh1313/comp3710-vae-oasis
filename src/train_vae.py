from pickle import TRUE
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
import time, numpy as np, matplotlib.pyplot as plt

# Build dataset & splits
IMG_SIZE   = 128
BATCH_SIZE = 32
VAL_FRAC   = 0.1

dataset = PngFolderDataset(DATA_DIRS, img_size=IMG_SIZE)
n_total = len(dataset)
n_val   = int(n_total * VAL_FRAC)
n_train = n_total - n_val
train_set, val_set = random_split(dataset, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))

# Safe loaders (no multiprocessing in Colab)
NUM_WORKERS = 0
PERSISTENT  = False
use_cuda    = torch.cuda.is_available()
device      = torch.device('cuda' if use_cuda else 'cpu')

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=use_cuda, persistent_workers=PERSISTENT)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=use_cuda, persistent_workers=PERSISTENT)

print(f"Train {len(train_set)} | Val {len(val_set)} | Device {device}")

# ----- train -----
USE_AMP   = TRUE      # keep False for stability first; you can set True later
Z_DIM     = 64
EPOCHS    = 40
LR        = 1e-4
BETA_MAX  = 0.5
WARMUP_EP = 10
CLIP_NORM = 2.0

model  = ConvVAE(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and use_cuda))

best_val = float('inf'); t0 = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    tr_loss=tr_rec=tr_kl=n_tr=0
    beta = min(1.0, epoch / max(1, WARMUP_EP)) * BETA_MAX

    for x in train_loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(USE_AMP and use_cuda)):
            x_logits, mu, logvar = model(x)
            loss, rec, kl = vae_loss_logits(x_logits, x, mu, logvar, beta=beta)

        if USE_AMP and use_cuda:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

        bs = x.size(0)
        tr_loss += loss.item()*bs; tr_rec += rec*bs; tr_kl += kl*bs; n_tr += bs

    # validation
    model.eval()
    va_loss=va_rec=va_kl=n_va=0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            x_logits, mu, logvar = model(x)
            loss, rec, kl = vae_loss_logits(x_logits, x, mu, logvar, beta=BETA_MAX)
            bs = x.size(0)
            va_loss += loss.item()*bs; va_rec += rec*bs; va_kl += kl*bs; n_va += bs

    tr_loss/=max(1,n_tr); tr_rec/=max(1,n_tr); tr_kl/=max(1,n_tr)
    va_loss/=max(1,n_va); va_rec/=max(1,n_va); va_kl/=max(1,n_va)
    print(f"Epoch {epoch:03d} | train {tr_loss:.4f} (rec {tr_rec:.4f}, kl {tr_kl:.4f}) | "
          f"val {va_loss:.4f} (rec {va_rec:.4f}, kl {va_kl:.4f}) | {time.time()-t0:.1f}s")

    if epoch % 5 == 0 or epoch == 1:
        save_recons_grid(model, val_loader, device, os.path.join(OUT_DIR, f"recons_epoch{epoch:03d}.png"))

    if va_loss < best_val - 1e-4:
        best_val = va_loss
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "vae_oasis_best.pth"))

torch.save(model.state_dict(), os.path.join(OUT_DIR, "vae_oasis_last.pth"))
print("Saved:", os.path.join(OUT_DIR, "vae_oasis_best.pth"), "and", os.path.join(OUT_DIR, "vae_oasis_last.pth"))

# Manifold and a final recon grid
save_umap(model, val_loader, device, os.path.join(OUT_DIR, "latent_umap.png"))
save_recons_grid(model, val_loader, device, os.path.join(OUT_DIR, "recons_final.png"))
print("Saved UMAP + recon grid in:", OUT_DIR)
