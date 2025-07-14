# ...so simply log(0.5)/log(median)
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

stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
# na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
# q3 = np.percentile(na_percent, 75)
# stability_df_filtered = stability_df.loc[:, na_percent <= q3]

nct_ids = stability_df['nctid'].reset_index(drop=True) # just nct_ids

# Df with NO NCT
stability_df_no_nct = stability_df.select_dtypes(include=[np.number])

median_val = stability_df_no_nct['studies:acronym'].median()
print(median_val)

# function to transform
def median_rescale_to_95(col):
    median_val = col.median() # should avoid NaNs automatically
    if median_val <= 0:
        return col  # skip transformation if median less than or = 0
    if median_val == 1: # skip transform if median=1
        return col
    power = np.log(0.95) / np.log(median_val)
    return col ** power

stability_transformed = stability_df_no_nct.apply(median_rescale_to_95)

print(stability_transformed['studies:acronym'].head())


pdf_path = "all_transformed_histograms2.pdf"

with PdfPages(pdf_path) as pdf:
    for col in stability_transformed.columns:
       
        plt.figure(figsize=(6,4))
        sns.histplot(stability_transformed[col], bins=100, kde=False, color='green')
        plt.yscale("log")
        plt.title(f'All Transformed Histogram, version 2: {col}')
        plt.xlim(0,1)
        plt.xlabel('Value')
        plt.ylabel('Count (log scale)')
        pdf.savefig()
        plt.close()
