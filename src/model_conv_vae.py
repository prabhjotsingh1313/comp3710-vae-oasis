import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    """128x128 VAE producing LOGITS (no final Sigmoid). Use BCEWithLogits."""
    def __init__(self, img_size=128, z_dim=32):
        super().__init__()
        self.img_size = img_size
        self.enc = nn.Sequential(
            nn.Conv2d(1,  32, 3, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True), # 64x64
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True), # 32x32
            nn.Conv2d(64,128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True), # 16x16
            nn.Conv2d(128,256,3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 8x8
            nn.Conv2d(256,512,3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 8x8
        )
        feat = 512 * (img_size//16) * (img_size//16)  # 512*8*8 for 128
        self.mu     = nn.Linear(feat, z_dim)
        self.logvar = nn.Linear(feat, z_dim)
        self.fc     = nn.Linear(z_dim, feat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),  # 32x32
            nn.ConvTranspose2d(128, 64,4,2,1), nn.BatchNorm2d(64),  nn.ReLU(True),  # 64x64
            nn.ConvTranspose2d(64,  32,4,2,1), nn.BatchNorm2d(32),  nn.ReLU(True),  # 128x128
            nn.Conv2d(32, 1, 3, 1, 1),  # logits out
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        mu     = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -8.0, 8.0)  # clamp for stability
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc(z).view(z.size(0), 512, self.img_size//16, self.img_size//16)
        return self.dec(h)  # logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_logits(x_logits, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta*kl, rec.item(), kl.item()
