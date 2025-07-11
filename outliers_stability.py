
import zipfile
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import umap
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Identify outliers from the umap embedding of stability data set
# Identify outliers from the stability data set


# STABILITY DATA
# Method 1: Just use the Q1-1.5*IQR method

stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
stability_df_filtered = stability_df.loc[:, na_percent <= q3]


nct_ids = stability_df_filtered['nctid'].reset_index(drop=True) # just nct_ids

# Df with NO NCT
stability_df_no_nct = stability_df_filtered.select_dtypes(include=[np.number])


Q1 = stability_df_no_nct.quantile(0.25)
Q3 = stability_df_no_nct.quantile(0.75)
IQR = Q3 - Q1

outlier_df = (stability_df_no_nct  < (Q1 - 1.5 * IQR)) | (stability_df_no_nct  > (Q3 + 1.5 * IQR))

outlier_counts = outlier_df.sum()

print(outlier_counts)

row_outliers = outlier_df.any(axis=1)

print(f"\nTotal unique outlier rows: {row_outliers.sum()}")

# Optional: Extract just those rows
outlier_df = stability_df_no_nct [row_outliers]

print(f"head of outlier df:")
print(outlier_df.head(3))

#Add NCTIDs back
outlier_nctids = nct_ids[row_outliers]
outlier_df = outlier_df.copy() 
outlier_df["nctid"] = outlier_nctids.values
print(f"head of outlier df with NCTIDs:")
print(outlier_df.head(3))

# Conclusion... above ^^ method produces way too many outlier rows...

#EMBEDDING DATA
# Method 1: Just use the Q1-1.5*IQR method
embedding = np.load("umap_embedding.npy")

x = embedding[:, 0]
y = embedding[:, 1]

umap_df = pd.DataFrame({"UMAP1": x, "UMAP2": y})

# IQR  on each axis
Q1 = umap_df.quantile(0.25)
Q3 = umap_df.quantile(0.75)
IQR = Q3 - Q1

outlier_df_umap = (
    (umap_df < (Q1 - 1.5 * IQR)) |
    (umap_df > (Q3 + 1.5 * IQR))
).any(axis=1)  # True if outlier in either x or y

# Print result
print(f"Total UMAP IQR outliers: {outlier_df_umap.sum()}")


plt.figure(figsize=(8, 6))
plt.scatter(x, y, c="lightgray", s=1, alpha=0.1)
plt.scatter(x[outlier_df_umap], y[outlier_df_umap], c="red", s=1, alpha=0.05)
plt.gca().set_aspect('equal', 'datalim')
plt.title("UMAP with IQR Outliers Highlighted")
plt.savefig("umap_outliers_iqr.png")

# EMBEDDING
# Method 2: Local Outlier Factor (LOF) 
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)  
outlier_flags = lof.fit_predict(embedding)  # -1 for outliers, 1 for inliers

# Get outlier indices
outlier_indices = np.where(outlier_flags == -1)[0]

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c='lightgray', s=1, alpha=0.1)
plt.scatter(embedding[outlier_indices, 0], embedding[outlier_indices, 1], c='red', s=1, alpha=0.05)
plt.gca().set_aspect('equal', 'datalim')
plt.title("UMAP with LOF Outliers Highlighted")
plt.legend()
plt.savefig("umap_lof_outliers.png")

print(f"Total UMAP LOF outliers: {len(outlier_indices)}")

lof_outlier_nctids = nct_ids.iloc[outlier_indices].values

print("Sample LOF UMAP outlier NCTIDs:", lof_outlier_nctids[:10])

# Method 3: Log transform, then use IQR or +-3 Z Score (np.abs((x - mean) / std)) 
