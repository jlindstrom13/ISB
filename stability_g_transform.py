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


# ...so simply log(0.5)/log(median)


stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')


# remove columns with too many NA
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
stability_df_filtered = stability_df.loc[:, na_percent <= q3]


nct_ids = stability_df_filtered['nctid'].reset_index(drop=True) # just nct_ids

# Df with NO NCT
stability_df_no_nct = stability_df_filtered.select_dtypes(include=[np.number])


# function
def median_rescale_to_half(col):
    median_val = col.median() # should avoid NaNs automatically
    if median_val <= 0:
        return col  # skip invalid or non-positive medians
    power = np.log(0.5) / np.log(median_val)
    return col ** power

stability_transformed = stability_df_no_nct.apply(median_rescale_to_half)

column_to_plot = stability_df_no_nct.columns[:9] 

for col in column_to_plot:
    plt.figure(figsize=(6,4))
    sns.histplot(stability_transformed[col], bins=100, kde=False, color='green')
    plt.yscale("log")
    plt.title(f'Transformed Histogram: {col}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.savefig("transformed_stability.png")