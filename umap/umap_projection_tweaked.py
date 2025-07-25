
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (0.05, '#440053'),
    (0.1, '#404388'),
    (0.15, '#2a788e'),
    (0.3, '#21a784'),
    (0.5, '#78d151'),
    (1, "#24fd9f"),
], N=256)

def using_mpl_scatter_density(fig, x, y, xlabel="UMAP 1", ylabel="UMAP 2", title = "UMAP"):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

embedding = np.load("umap_embedding_tweaked.npy")

x = embedding[:, 0]
y = embedding[:, 1]

# fig = plt.figure(figsize=(10, 8), dpi=600)
# using_mpl_scatter_density(fig, x, y, title="UMAP Clinical Trial Density-Params Tweaked")
# plt.savefig("umap_density_tweaked.png", dpi=600)


plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    marker='o',
    c = "blue",
    alpha = 0.05,
    s = 0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of Stability - tweaked, simple plot', fontsize=12);
plt.savefig("umap_tweaked")
