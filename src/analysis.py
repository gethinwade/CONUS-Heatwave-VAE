
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Rectangle

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec


SEED = 42


class HeatwaveArrayDataset(Dataset):
    def __init__(self, X, dates):
        self.samples = X
        self.dates = np.array([str(d) for d in dates])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), self.dates[idx]


def make_analysis_loader(X, dates, batch_size=32, shuffle=False):
    ds = HeatwaveArrayDataset(X, dates)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def reconstruct_data(model, loader, device):
    recon_x, z_vectors, date_labels = [], [], []
    num_used = len(loader.dataset)
    model.eval()
    with torch.no_grad():
        for batch, dates in loader:
            batch = batch.float().to(device)
            recon_batch, mu, logvar, z = model(batch)
            recon_x.append(recon_batch.cpu().numpy())
            z_vectors.append(z.cpu().numpy())
            date_labels.append(np.array(dates))
    recon_x = np.concatenate(recon_x, axis=0)
    z_vectors = np.concatenate(z_vectors, axis=0)
    date_labels = np.concatenate(date_labels, axis=0)

    n_processed = len(recon_x)
    x = loader.dataset.samples[:n_processed]
    date_labels = date_labels[:n_processed]
    recon_x = recon_x[:n_processed]
    return x, recon_x, z_vectors, date_labels



def fit_latent_pca(recon_df, n_components=None):
    """Fit PCA on train-split latent z, project all splits into it.
    Stores per-split projections and the fitted PCA on recon_df."""
    z_train = recon_df["train"]["z"]
    n_components = n_components or z_train.shape[1]
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(z_train)
    for p in ("train", "val", "test"):
        recon_df[p]["pca"] = pca.transform(recon_df[p]["z"])
    recon_df["pca_model"] = pca
    return recon_df


def plot_scree(recon_df, n_show=20):
    """Two-panel: per-component variance (bars) and cumulative variance (line),
    from the train-fit PCA."""
    pca = recon_df["pca_model"]
    evr = pca.explained_variance_ratio_ * 100
    k = min(n_show, len(evr))
    comps = np.arange(1, k + 1)
    cum = np.cumsum(evr[:k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(comps, evr[:k], color="tab:blue")
    ax1.set_title("Scree Plot")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (%)")

    ax2.plot(comps, cum, color="tab:blue", marker="o")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance (%)")

    fig.tight_layout()
    return fig


def compute_latent_tsne(recon_df, perplexity=50):
    """t-SNE on the raw 64-d latent (no PCA pre-reduction).
    Fits on all splits jointly so they share one embedding, then splits back."""
    periods = ["train", "val", "test"]
    combined_z = np.concatenate([recon_df[p]["z"] for p in periods], axis=0)

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=SEED, method="exact")
    emb = tsne.fit_transform(combined_z)

    sizes = [len(recon_df[p]["z"]) for p in periods]
    splits = np.split(emb, np.cumsum(sizes)[:-1], axis=0)
    for p, e in zip(periods, splits):
        recon_df[p][f"tsne_{perplexity}"] = e
    return recon_df


def plot_tsne_by_split(recon_df, perplexity=50):
    """Scatter of the t-SNE embedding, colored by train/val/test."""
    periods = ["train", "val", "test"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for p, c, m in zip(periods, colors, markers):
        e = recon_df[p][f"tsne_{perplexity}"]
        ax.scatter(e[:, 0], e[:, 1], s=10, alpha=0.6, c=c, marker=m, label=p)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Latent t-SNE (perplexity={perplexity})")
    ax.legend(markerscale=2, framealpha=0.6)
    fig.tight_layout()
    return fig


def plot_tsne_by_season(recon_df, perplexity=50, periods=("train", "val", "test")):
    """Same t-SNE embedding, colored by season of onset (all splits pooled)."""
    season_list = ["DJF", "MAM", "JJA", "SON"]
    season_colors = ["cornflowerblue", "mediumseagreen", "orange", "indianred"]

    emb = np.concatenate([recon_df[p][f"tsne_{perplexity}"] for p in periods], axis=0)
    dates = np.concatenate([recon_df[p]["date_labels"] for p in periods], axis=0)
    months = np.array([int(str(d)[5:7]) for d in dates])
    season_idx = np.array([(m % 12) // 3 for m in months])

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (s, c) in enumerate(zip(season_list, season_colors)):
        sel = season_idx == i
        ax.scatter(emb[sel, 0], emb[sel, 1], s=12, alpha=0.7, c=c, label=s)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Latent t-SNE by season (perplexity={perplexity})")
    ax.legend(markerscale=2, framealpha=0.6)
    fig.tight_layout()
    return fig




def swapped_index_diagnostic(model, loader, device, n=64):
    """Decode sample A's latent z with sample B's pooling indices.
    Compares the swapped reconstruction to both sources to see which
    pathway (latent vs index-skip) drives the output."""
    model.eval()

    batch = next(iter(loader))
    x = (batch[0] if isinstance(batch, (list, tuple)) else batch).float().to(device)
    m = min(n, x.shape[0] // 2)
    xa, xb = x[:m], x[m:2*m]

    with torch.no_grad():
        mu_a, logvar_a, idx_a = model.encode(xa)
        mu_b, logvar_b, idx_b = model.encode(xb)
        za = model.reparameterize(mu_a, logvar_a)

        recon_normal = model.decode(za, idx_a)
        recon_swapped = model.decode(za, idx_b)

    xa_ = xa.cpu().numpy(); xb_ = xb.cpu().numpy()
    rn = recon_normal.cpu().numpy(); rs = recon_swapped.cpu().numpy()

    d_to_A = np.mean(np.abs(rs - xa_), axis=(1, 2, 3, 4))
    d_to_B = np.mean(np.abs(rs - xb_), axis=(1, 2, 3, 4))
    d_normal = np.mean(np.abs(rn - xa_), axis=(1, 2, 3, 4))

    print(f"normal recon error (z_A + idx_A vs A):     {d_normal.mean():.4f}")
    print(f"swapped recon → distance to A (z source):  {d_to_A.mean():.4f}")
    print(f"swapped recon → distance to B (idx source): {d_to_B.mean():.4f}")
    print()
    if d_to_A.mean() < d_to_B.mean():
        print("Swapped recon resembles A (the z source) → latent carries the reconstruction.")
    else:
        print("Swapped recon resembles B (the index source) → the skip carries it.")
    return {"d_normal": d_normal, "d_to_A": d_to_A, "d_to_B": d_to_B}




# =============================================== RECREATION CODE =============================================== #


def calculate_r2_score(originals, reconstructions, variables):
    orig_flat = originals.reshape(originals.shape[0], -1)
    recon_flat = reconstructions.reshape(reconstructions.shape[0], -1)
    overall_r2 = r2_score(orig_flat, recon_flat)

    channel_r2 = {}
    for c, cname in enumerate(variables):
        orig_c = originals[:, c, ...].reshape(originals.shape[0], -1)
        recon_c = reconstructions[:, c, ...].reshape(reconstructions.shape[0], -1)
        channel_r2[cname] = r2_score(orig_c, recon_c)

    return overall_r2, channel_r2


def plot_combined_tsne_figure(recon_df, perplexity=50):
    periods = ["train", "val", "test"]
    markers = ["o", "s", "^"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    season_colors = ["cornflowerblue", "mediumseagreen", "orange", "indianred"]
    season_list = ["DJF", "MAM", "JJA", "SON"]
    season_to_int = {s: i for i, s in enumerate(season_list)}
    subplot_labels = {"train": "A", "val": "B", "test": "C",
                      "combined": "D", "distribution": "E", "seasons": "F"}
    label_fs, title_fs, legend_fs = 15, 20, 12

    fig = plt.figure(figsize=(16, 8))
    mosaic = [
        ["train", "train", "val", "val", "distribution", "distribution", "distribution"],
        ["train", "train", "val", "val", "seasons", "seasons", "seasons"],
        ["test", "test", "combined", "combined", "seasons", "seasons", "seasons"],
        ["test", "test", "combined", "combined", "seasons", "seasons", "seasons"],
    ]
    ax = fig.subplot_mosaic(mosaic)
    ref = ax["train"]

    # Train / val / test scatter panels
    for i, p in enumerate(periods):
        a = ax[p]
        if p != "train":
            a.sharex(ref); a.sharey(ref)
        a.text(0.05, 0.95, subplot_labels[p], transform=a.transAxes,
               fontsize=title_fs, va="top", ha="left")
        t = recon_df[p][perplexity]["tsne"]
        a.scatter(t[:, 0], t[:, 1], alpha=0.6, c=colors[i], marker=markers[i], s=10, label=p)
        if p == "train":
            a.set_ylabel("t-SNE 2", fontsize=label_fs)
        if p == "test":
            a.set_xlabel("t-SNE 1", fontsize=label_fs)
            a.set_ylabel("t-SNE 2", fontsize=label_fs)

    # Combined panel
    ad = ax["combined"]
    for i, p in enumerate(periods):
        t = recon_df[p][perplexity]["tsne"]
        ad.scatter(t[:, 0], t[:, 1], alpha=0.6, c=colors[i], marker=markers[i], s=5, label=p)
    ad.sharex(ref); ad.sharey(ref)
    ad.set_xlabel("t-SNE 1", fontsize=label_fs)
    ad.legend(fontsize=legend_fs, markerscale=2, framealpha=0.5, labelspacing=0.2, handletextpad=0.1)
    ad.text(0.05, 0.95, subplot_labels["combined"], transform=ad.transAxes,
            fontsize=title_fs, va="top", ha="left")

    # Seasonal panel with GMM ellipses
    t = recon_df["test"][perplexity]["tsne"]
    dates = recon_df["test"]["date_labels"]
    gmm = recon_df["test"][perplexity]["gmm"]
    months = np.array([int(str(d)[5:7]) for d in dates])
    seasons = np.array([season_list[(m % 12) // 3] for m in months])
    season_idx = np.array([season_to_int[s] for s in seasons])

    asea = ax["seasons"]
    asea.text(0.02, 0.97, subplot_labels["seasons"], transform=asea.transAxes,
              fontsize=title_fs, va="top", ha="left")
    asea.scatter(t[:, 0], t[:, 1], c=season_idx,
                 cmap=mcolors.ListedColormap(season_colors), s=15, alpha=0.7)
    asea.sharex(ref); asea.sharey(ref)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=s,
                          markerfacecolor=c, markersize=6)
               for s, c in zip(season_list, season_colors)]
    asea.legend(handles=handles, loc="best", fontsize=legend_fs, markerscale=1.5,
                framealpha=0.5, labelspacing=0.2, handletextpad=0.1)
    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        vals, vecs = np.linalg.eigh(cov)
        angle = np.arctan2(vecs[0, 1], vecs[0, 0]) * 180 / np.pi
        width, height = 2 * np.sqrt(vals)
        asea.add_patch(Ellipse(mean, width, height, angle=angle,
                               edgecolor="black", facecolor="none", lw=2))
        asea.text(mean[0], mean[1], str(i + 1), fontsize=15, ha="center", va="center", weight="bold")
    asea.set_xlabel("t-SNE 1", fontsize=label_fs)

    # Seasonal distribution bars
    adist = ax["distribution"]
    labels = recon_df["test"][perplexity]["labels"]
    n_clusters = int(labels.max()) + 1
    counts = np.zeros((n_clusters, len(season_list)))
    for cid in range(n_clusters):
        cs = seasons[labels == cid]
        for i, s in enumerate(season_list):
            counts[cid, i] = np.sum(cs == s)
    pct = counts / counts.sum(axis=1, keepdims=True) * 100

    bar_w = 0.2
    x = np.arange(n_clusters)
    for i, s in enumerate(season_list):
        bars = adist.bar(x + i * bar_w, pct[:, i], width=bar_w, color=season_colors[i], label=s)
        for j, bar in enumerate(bars):
            adist.text(bar.get_x() + bar.get_width() / 2, 1, f"{pct[j, i]:.1f}",
                       ha="center", va="bottom", fontsize=10, color="black", rotation=90)
    adist.set_xticks(x + 1.5 * bar_w)
    adist.set_xticklabels([f"#{i+1}" for i in range(n_clusters)])
    adist.set_ylabel("%", rotation=0, fontsize=label_fs, labelpad=10)
    adist.legend(ncols=2, framealpha=0.5, labelspacing=0.2, handletextpad=0.1)
    adist.text(0.02, 0.97, subplot_labels["distribution"], transform=adist.transAxes,
               fontsize=title_fs, va="top", ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def compute_cluster_composites(input_x, tsne_data, centroids, n_samples):
    distances = np.linalg.norm(tsne_data[:, np.newaxis] - centroids, axis=2)
    if isinstance(n_samples, int) and distances.shape[0] < n_samples:
        n_samples = distances.shape[0]
    elif n_samples == "all":
        n_samples = distances.shape[0]
    closest_indices = np.argsort(distances, axis=0)[:n_samples]

    num_clusters = centroids.shape[0]
    composite_array = np.zeros((num_clusters, *input_x.shape[1:]), dtype=np.float32)
    for k in range(num_clusters):
        composite_array[k] = np.mean(input_x[closest_indices[:, k]], axis=0)
    return composite_array, closest_indices


var_cmaps = {
    "t2m": "bwr", "msl": "Spectral_r", "u10": "PuOr", "v10": "PuOr",
    "tcc": "BrBG", "ssr": "RdYlGn_r", "r": "RdYlBu_r",
    "stream250": "seismic", "z500": "Spectral_r",
}


def plot_cluster_composites(recon_df, variables, domain, period="test",
                            perplexity=50, n_samples=100, dpi=300):
    min_lon, max_lon, min_lat, max_lat = domain
    input_x = recon_df[period]["x"]
    tsne = recon_df[period][perplexity]["tsne"]
    gmm = recon_df[period][perplexity]["gmm"]
    centroids = gmm.means_
    n_clusters = gmm.n_components
    n_steps = input_x.shape[2]
    timestep_labels = [f"t={o}" if o == 0 else f"t{o:+}" for o in range(-5, 6)]

    var_labels = [v.replace("stream", "stream250").replace("z", "z500") for v in variables]
    sample_set, _ = compute_cluster_composites(input_x, tsne, centroids, n_samples)

    figs = {}
    for v, var_name in enumerate(var_labels):
        cmap = var_cmaps.get(var_name, "bwr")
        fig = plt.figure(figsize=(n_clusters * 2.2, n_steps * 0.7))
        spec = gridspec.GridSpec(n_steps, n_clusters + 1,
                                 width_ratios=[1] * n_clusters + [0.1],
                                 hspace=0.01, wspace=0.1)

        v_data = sample_set[:, v, :, :, :].ravel()
        vmin, vmax = np.percentile(v_data, [1, 99])
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        lons = np.linspace(min_lon, max_lon, sample_set.shape[-1])
        lats = np.linspace(min_lat, max_lat, sample_set.shape[-2])
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        img = None
        for t in range(n_steps):
            for i in range(n_clusters):
                ax = fig.add_subplot(spec[t, i], projection=ccrs.PlateCarree())
                img = ax.contourf(lon_grid, lat_grid, sample_set[i, v, t],
                                  levels=10, cmap=cmap, transform=ccrs.PlateCarree(),
                                  norm=divnorm)
                ax.coastlines(linewidth=0.3, color="black")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_ylabel(timestep_labels[t], fontsize=13, labelpad=15,
                                  rotation=0, verticalalignment="center")
                if t == 5:
                    ax.add_patch(Rectangle((min_lon, min_lat), max_lon - min_lon,
                                           max_lat - min_lat, linewidth=4,
                                           edgecolor="black", facecolor="none",
                                           transform=ccrs.PlateCarree()))
                if t == 0:
                    ax.set_title(f"#{i+1}", fontsize=16)

        cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.025])
        cbar = fig.colorbar(img, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(var_name, fontsize=12, labelpad=4)
        cbar.ax.tick_params(labelsize=8)
        figs[var_name] = fig
    return figs