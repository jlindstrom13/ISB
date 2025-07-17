# UMAP of stability data plot, but with outliers from filtered_dist_numfeatures, cutoff =13 highlighted
import zipfile
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import umap
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import math

trials_above_cutoff = pd.read_pickle("trials_above_cutoff4.pkl")

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

embedding = np.load("umap_embedding.npy")

x = embedding[:, 0]
y = embedding[:, 1]

fig = plt.figure(figsize=(10, 8), dpi=600)
using_mpl_scatter_density(fig, x, y, title="UMAP Clinical Trial Density")
plt.savefig("umap_stability_outliers.png", dpi=600)

# new stuff to try and get NCTID:
stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
stability_df_filtered = stability_df.loc[:, na_percent <= q3]

nct_ids = stability_df_filtered['nctid'].reset_index(drop=True) # just nct_ids

umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"]) 
umap_df["nct_id"] = nct_ids # combining umap1,2 and nctids


umap_df["is_outlier"] = umap_df["nct_id"].isin(trials_above_cutoff)
outlier_points = umap_df[umap_df["is_outlier"]]

plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = "lightgray",
    alpha = 0.05,
    s = 1)
plt.scatter(
    outlier_points["UMAP1"],
    outlier_points["UMAP2"],
    c="purple",
    s=1,
    alpha=0.05,
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of the Stability data, 11656 Outliers Highlighted (Cutoff 4)', fontsize=11);
plt.savefig("umap_stability_outliers4")
