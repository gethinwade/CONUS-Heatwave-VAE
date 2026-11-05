
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset
import config

SEED = 42


# ===========
# MODEL SETUP
# ===========

class SampleDS(Dataset):
    def __init__(self, X, idx):
        self.X = X
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.X[self.idx[i]], dtype=np.float32).copy())


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, input_dim, input_shape, hidden_dim, kernel_size, stride, padding):
        super(SpatialTemporalEncoder, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.enc_conv1 = nn.Sequential(
            nn.Conv3d(self.input_dim, self.hidden_dim // 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // 8),
            nn.LeakyReLU(),
        )
        self.max_pool1 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.enc_conv2 = nn.Sequential(
            nn.Conv3d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // 4),
            nn.LeakyReLU(),
        )
        self.max_pool2 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.enc_conv3 = nn.Sequential(
            nn.Conv3d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // 2),
            nn.LeakyReLU(),
        )
        self.max_pool3 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.enc_conv4 = nn.Sequential(
            nn.Conv3d(self.hidden_dim // 2, self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim),
        )
        self.max_pool4 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.max_pool1(x)
        x = self.enc_conv2(x)
        x = self.max_pool2(x)
        x = self.enc_conv3(x)
        x = self.max_pool3(x)
        x = self.enc_conv4(x)
        x = self.max_pool4(x)
        return x


class SpatialTemporalDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, input_shape, kernel_size, stride, padding):
        super(SpatialTemporalDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.input_shape = input_shape
        
        self.up_kernel = (3, 2, 2)
        self.up_stride = (1, 2, 2)
        self.up_padding = (0, 0, 0)

        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim, self.hidden_dim // 2, kernel_size=self.up_kernel, stride=self.up_stride, padding=self.up_padding),
            nn.BatchNorm3d(self.hidden_dim // 2),
            nn.LeakyReLU(),
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=self.up_kernel, stride=self.up_stride, padding=self.up_padding),
            nn.BatchNorm3d(self.hidden_dim // 4),
            nn.LeakyReLU(),
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim // 4, self.hidden_dim // 8, kernel_size=self.up_kernel, stride=self.up_stride, padding=self.up_padding),
            nn.BatchNorm3d(self.hidden_dim // 8),
            nn.LeakyReLU(),
        )
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim // 8, self.input_dim, kernel_size=self.up_kernel, stride=self.up_stride, padding=self.up_padding),
        )

    def forward(self, x):
        x = self.dec_conv4(x)
        x = self.dec_conv3(x)
        x = self.dec_conv2(x)
        x = self.dec_conv1(x)
        return x


class VAEConv3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, input_shape, kernel_size, stride, padding, apply_sigmoid=False):
        super(VAEConv3D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.apply_sigmoid = apply_sigmoid

        self.spatial_temporal_encoder = SpatialTemporalEncoder(self.input_dim, self.input_shape, self.hidden_dim, self.kernel_size, self.stride, self.padding)
        self.flatten_dim, self.tl, self.hl, self.wl = self._calculate_flatten_dim()
        self.flatten_encoder = nn.Flatten(start_dim=1)

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)

        self.fc_decoder = nn.Linear(self.latent_dim, self.flatten_dim)
        self.unflatten_decoder = nn.Unflatten(dim=1, unflattened_size=(self.hidden_dim, self.tl, self.hl, self.wl))
        self.spatial_temporal_decoder = SpatialTemporalDecoder(self.hidden_dim, self.input_dim, self.input_shape, self.kernel_size, self.stride, self.padding)
        self.final_activation = nn.Sigmoid()

    def _calculate_flatten_dim(self):
        t, h, w = self.input_shape
        for layer in self.spatial_temporal_encoder.children():
            if isinstance(layer, nn.Sequential):
                for l in layer:
                    if isinstance(l, nn.Conv3d):
                        t = (t + 2*l.padding[0] - l.dilation[0]*(l.kernel_size[0] - 1) - 1) // l.stride[0] + 1
                        h = (h + 2*l.padding[1] - l.dilation[1]*(l.kernel_size[1] - 1) - 1) // l.stride[1] + 1
                        w = (w + 2*l.padding[2] - l.dilation[2]*(l.kernel_size[2] - 1) - 1) // l.stride[2] + 1
                    elif isinstance(l, nn.MaxPool3d):
                        t = (t + 2*l.padding[0] - l.dilation*(l.kernel_size[0] - 1) - 1) // l.stride[0] + 1
                        h = (h + 2*l.padding[1] - l.dilation*(l.kernel_size[1] - 1) - 1) // l.stride[1] + 1
                        w = (w + 2*l.padding[2] - l.dilation*(l.kernel_size[2] - 1) - 1) // l.stride[2] + 1
            elif isinstance(layer, nn.MaxPool3d):
                t = (t + 2*layer.padding[0] - layer.dilation*(layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
                h = (h + 2*layer.padding[1] - layer.dilation*(layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
                w = (w + 2*layer.padding[2] - layer.dilation*(layer.kernel_size[2] - 1) - 1) // layer.stride[2] + 1
        return self.hidden_dim*t*h*w, t, h, w

    def encode(self, x):
        h = self.spatial_temporal_encoder(x)
        h = self.flatten_encoder(h)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def decode(self, z):
        z = self.fc_decoder(z)
        z = self.unflatten_decoder(z)
        z = self.spatial_temporal_decoder(z)
        if self.apply_sigmoid:
            z = self.final_activation(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z

    def reparameterize(self, mu, logvar):
        epsilon = 1e-6
        std = torch.sqrt(torch.exp(logvar).clamp(min=epsilon))
        eps = torch.randn_like(std)
        return eps*std + mu


# ==============
# MODEL TRAINING
# ==============

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon = nn.L1Loss(reduction="mean")(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


def train_vae(model, train_loader, val_loader, device, epochs=100, lr=1e-3, clip=1.0, beta_max=1.0):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        beta = beta_max * min(1.0, epoch / epochs)
        train_total = 0.0
        for data in train_loader:
            data = data.to(device)
            recon, mu, logvar, z = model(data)
            loss, recon_l, kld = vae_loss(recon, data, mu, logvar, beta)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            opt.zero_grad()
            train_total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, mu, logvar, z = model(data)
                loss, _, _ = vae_loss(recon, data, mu, logvar, beta)
                val_total += loss.item()

        n_tr = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        print(f"epoch {epoch+1}/{epochs}  beta={beta:.3f}  "
              f"train={train_total/n_tr:.4f}  val={val_total/n_val:.4f}")

    return model


def latent_health(model, loader, device):
    """Per-dim KL on the val set: active-dim count, mean recon, mean KL."""
    model.eval()
    kl_dims, recon_sum, n = [], 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon, mu, logvar, z = model(data)
            logvar = torch.clamp(logvar, min=-10, max=10)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_dims.append(kl_per_dim.sum(dim=0).cpu().numpy())
            recon_sum += nn.L1Loss(reduction="sum")(recon, data).item()
            n += data.shape[0]
    kl_by_dim = np.sum(kl_dims, axis=0) / n
    active = int((kl_by_dim > 0.01).sum())
    return {
        "active_dims": active,
        "val_recon_per_elem": recon_sum / (n * int(np.prod(loader.dataset[0].shape))),
        "val_kl_mean": float(kl_by_dim.mean()),
        "kl_by_dim": kl_by_dim,
    }


def run_sweep(train_loader, val_loader, device, latents, betas, epochs=50, lr=1e-3, hidden=256):
    results = []
    for latent in latents:
        for beta_max in betas:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            model = VAEConv3D(
                input_dim=7, hidden_dim=hidden, latent_dim=latent,
                input_shape=(11, config.TARGET_H, config.TARGET_W),
                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                apply_sigmoid=False,
            ).to(device)
            model = train_vae(model, train_loader, val_loader, device,
                              epochs=epochs, lr=lr, clip=1.0, beta_max=beta_max)
            health = latent_health(model, val_loader, device)
            row = {"latent": latent, "beta": beta_max, "lr": lr, "hidden": hidden,
                   "active_dims": health["active_dims"],
                   "val_recon_per_elem": health["val_recon_per_elem"],
                   "val_kl_mean": health["val_kl_mean"]}
            print(row)
            results.append(row)
    return results


def train_vae_final(model, train_loader, val_loader, device, epochs=100, lr=1e-3, clip=1.0, beta_max=0.1, warmup=25, patience=5, factor=0.5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=factor, patience=patience)

    for epoch in range(epochs):
        model.train()
        beta = beta_max * min(1.0, epoch / warmup)
        train_total = 0.0
        for data in train_loader:
            data = data.to(device)
            recon, mu, logvar, z = model(data)
            loss, recon_l, kld = vae_loss(recon, data, mu, logvar, beta)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            opt.zero_grad()
            train_total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, mu, logvar, z = model(data)
                loss, _, _ = vae_loss(recon, data, mu, logvar, beta)
                val_total += loss.item()

        n_tr = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        val_epoch = val_total / n_val
        scheduler.step(val_epoch)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{epochs}  beta={beta:.3f}  lr={cur_lr:.1e}  "
              f"train={train_total/n_tr:.4f}  val={val_epoch:.4f}")

    return model