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

# take ALL g transformed stability data, figure out how much has less than 0.5,
# create plot fo the distribution of outlier features 
stability_transformed = pd.read_pickle("stability_transformed.pkl")
nct_ids = pd.read_pickle("nct_ids.pkl")

# not filtered
stability_lessthan_05 = stability_transformed <0.5
count_lessthan_05 = stability_lessthan_05.sum(axis=0)
print(count_lessthan_05.shape)

count_lessthan_05.to_csv('count_lessthan05_all.tsv', sep="\t")


#stability_transformed_with_id = stability_transformed.copy()
#stability_transformed_with_id['nctid'] = nct_ids


outlier_binary_matrix = (stability_transformed < 0.5).astype(int)
outlier_binary_matrix['nctid'] = nct_ids.values

print(outlier_binary_matrix['brief_summaries:description'].sum())

print(outlier_binary_matrix.head())

# Plot: x: # outlier (stability <0.5) features, y: log(#trials)
num_outlier_features = outlier_binary_matrix.drop(columns=['nctid']).sum(axis=1) # drop nctid, sum along rows 

counts = num_outlier_features.value_counts().sort_index()

cutoff = 24
plt.figure(figsize=(10,6))
plt.plot(counts.index, counts.values, marker='o')  # log1p adds one to help w 0 count
plt.axvline(x=cutoff, color = 'red') 
plt.xlabel("Number of Outlier Features per Trial")
plt.ylabel("Number of Trials, log scale")
plt.yscale('log') 
plt.title("Dist. of Outlier Features per Trial")
plt.grid(True)
plt.tight_layout()
plt.savefig("dist_outlierfeatures.png")


cutoff = 24
trials_above_cutoff = outlier_binary_matrix.loc[num_outlier_features > cutoff, 'nctid']
print(f"Trials with more than {cutoff} outlier features:")
print(trials_above_cutoff)

