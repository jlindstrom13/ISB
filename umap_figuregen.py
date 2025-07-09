# Using umap_stability.py output file (umap_embedding.npy) create a graph

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

embedding = np.load("umap_embedding.npy")

x = embedding[:, 0]
y = embedding[:, 1]

fig = plt.figure(figsize=(10, 8), dpi=600)
using_mpl_scatter_density(fig, x, y, title="UMAP Clinical Trial Density")
plt.savefig("umap_density_plot.png", dpi=600)

# new stuff to try and get NCTID:
stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
stability_df_filtered = stability_df.loc[:, na_percent <= q3]

nct_ids = stability_df_filtered['nctid'].reset_index(drop=True) # just nct_ids

umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"]) 
umap_df["nct_id"] = nct_ids # combining umap1,2 and nctids

# choosing phase 1, make graph that plots phase 1 in purple
# note: stability dataframe doesn't actually say if its phase 1.. just stability score

aact_df = pd.read_pickle("aact_20250626.pkl")

phase1_df = aact_df[aact_df['phase'] == 'PHASE1']
umap_df["is_phase1"] = umap_df["nct_id"].isin(phase1_df["nct_id"])
phase1_points = umap_df[umap_df["is_phase1"]]

phase2_df = aact_df[aact_df['phase'] == 'PHASE2']
umap_df["is_phase2"] = umap_df["nct_id"].isin(phase2_df["nct_id"])
phase2_points = umap_df[umap_df["is_phase2"]]

phase3_df = aact_df[aact_df['phase'] == 'PHASE3']
umap_df["is_phase3"] = umap_df["nct_id"].isin(phase3_df["nct_id"])
phase3_points = umap_df[umap_df["is_phase3"]]


#Scatter plot of all trials as gray, and then phase one as purple

plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = "lightgray",
    alpha = 0.05,
    s = 1)
plt.scatter(
    phase1_points["UMAP1"],
    phase1_points["UMAP2"],
    c="purple",
    s=1,
    alpha=0.05,
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Stability dataset, Phase 1 Highlighted', fontsize=12);
plt.savefig("umap_phase1")

#PHASE 2
plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = "lightgray",
    alpha = 0.05,
    s = 1)
plt.scatter(
    phase2_points["UMAP1"],
    phase2_points["UMAP2"],
    c="blue",
    s=1,
    alpha=0.05,
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Stability dataset, Phase 2 Highlighted', fontsize=12);
plt.savefig("umap_phase2")


#PHASE 3
plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = "lightgray",
    alpha = 0.05,
    s = 1)
plt.scatter(
    phase3_points["UMAP1"],
    phase3_points["UMAP2"],
    c="green",
    s=1,
    alpha=0.05,
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Stability dataset, Phase 3 Highlighted', fontsize=12);
plt.savefig("umap_phase3")