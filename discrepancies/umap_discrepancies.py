# UMAP but highlight the NCTs that are from the discrepancies paper

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

#58 NCTs
discrepant_nct = pd.read_pickle("discrepant_unique_nctids.pkl")

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

umap_df["is_discrepant"] = umap_df["nct_id"].isin(discrepant_nct)
discrepant_points = umap_df[umap_df["is_discrepant"]]

random_sample = umap_df.sample(n=58, random_state=42)


plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c="#f5f5f5",
    alpha = 0.01,
    s = 1)
plt.scatter(
    discrepant_points["UMAP1"],
    discrepant_points["UMAP2"],
    c="purple",
    s=2,
    alpha=0.2,
    )
plt.scatter(
    random_sample["UMAP1"],
    random_sample["UMAP2"],
    c="green",
    s=2,
    alpha=0.2,
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of Stability data, Discrepancies between Publication & Trial Highlighted', fontsize=10);
plt.savefig("umap_stability_discrepancies")
