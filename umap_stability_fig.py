# creating a scatterplot with different color dots for what the stability score is
# one feature chosen

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0.0, '#ffffff'),
#     (0.9, '#440053'),
#     (0.96, '#404388'),
#     (0.97, '#2a788e'),
#     (0.98, '#21a784'),
#     (0.99, '#78d151'),
#     (1, "#24fd9f"),
# ], N=256)

# def using_mpl_scatter(fig, x, y, values, xlabel="UMAP 1", ylabel="UMAP 2", title="UMAP", cmap=white_viridis):
#     ax = fig.add_subplot(1, 1, 1)
#     scatter = ax.scatter(x, y, c=values, cmap=cmap, vmin=0, vmax=1, s=5)
#     fig.colorbar(scatter, label='Value (0–1)')
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)


embedding = np.load("umap_embedding.npy")

x = embedding[:, 0]
y = embedding[:, 1]

# new stuff to try and get NCTID:
stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
stability_df_filtered = stability_df.loc[:, na_percent <= q3]

nct_ids = stability_df_filtered['nctid'].reset_index(drop=True) # just nct_ids

umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"]) 
umap_df["nct_id"] = nct_ids # combining umap1,2 and nctids


#print(stability_df_filtered.columns)

umap_df = umap_df.merge(
    stability_df_filtered[['nctid', 'studies:study_type']],
    left_on='nct_id',
    right_on='nctid',
    how='left'
).drop(columns='nctid') 

print(umap_df.head())

# fig = plt.figure(figsize=(8, 6))
# using_mpl_scatter(fig, umap_df['UMAP1'], umap_df['UMAP2'], umap_df['studies:study_type'])
# plt.savefig("umap_stability_studytype.png")
 


plt.figure(figsize=(8, 6))
plt.scatter(
    umap_df['UMAP1'],
    umap_df['UMAP2'],
    c='lightgray',
    alpha=0.05,
    s=1
)


cutoff = 0.95
highlight = umap_df[umap_df['studies:study_type'] < cutoff]

plt.scatter(
    highlight['UMAP1'],
    highlight['UMAP2'], 
    c='red',
    alpha=0.3,
    s=1
)

# cutoffhigh = 0.98
# highlight = umap_df[umap_df['studies:study_type'] < cutoffhigh]

# plt.scatter(
#     highlight['UMAP1'],
#     highlight['UMAP2'], 
#     c='blue',
#     alpha=0.3
# )

plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP projection with points Stability < Red {cutoff} highlighted')
plt.savefig("umap_studytype_cutoff")

count_below_cutoff = (umap_df['studies:study_type'] < cutoff).sum()
print(f"Number of points with studies:study_type stability < {cutoff}: {count_below_cutoff}")

# plt.figure()
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c = "lightgray",
#     alpha = 0.05,
#     s = 1)
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Stability dataset, Phase 3 Highlighted', fontsize=12);
# plt.savefig("umap_phase3")