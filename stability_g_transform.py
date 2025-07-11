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


# ...so simply log(0.5)/log(median)

stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
# na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
# q3 = np.percentile(na_percent, 75)
# stability_df_filtered = stability_df.loc[:, na_percent <= q3]


nct_ids = stability_df['nctid'].reset_index(drop=True) # just nct_ids

# Df with NO NCT
stability_df_no_nct = stability_df.select_dtypes(include=[np.number])


# function
def median_rescale_to_half(col):
    median_val = col.median() # should avoid NaNs automatically
    if median_val <= 0:
        return col  # skip invalid or non-positive medians
    power = np.log(0.95) / np.log(median_val)
    return col ** power

stability_transformed = stability_df_no_nct.apply(median_rescale_to_half)


# Drop plots that a) have lots of values below say 0.5
# b) ones that have none below say 0.5

pdf_path = "select_transformed_histograms.pdf"

with PdfPages(pdf_path) as pdf:
    for col in stability_transformed.columns:
        values = stability_transformed[col]
        prop_below_05 = (values < 0.5).mean()
        if prop_below_05 > 0.1 or prop_below_05 == 0: # code this out if u want all
            continue # skips these... # code this out if u want all
        
        #plot the rest of the graphs
        plt.figure(figsize=(6,4))
        sns.histplot(stability_transformed[col], bins=100, kde=False, color='green')
        plt.yscale("log")
        plt.title(f'Select Transformed Histogram: {col}')
        plt.xlabel('Value')
        plt.ylabel('Count (log scale)')
        pdf.savefig()
        plt.close()


# CREATE CUMULATIVE HISTOGRAM: sum of the 54 feature plots
bins = np.linspace(0, 1, 101) 
cumulative_counts = np.zeros(len(bins) - 1)

# Loop over columns, get histogram counts, and sum
for col in stability_transformed.columns:
    values = stability_transformed[col]
    prop_below_05 = (values < 0.5).mean()
    if prop_below_05 > 0.1 or prop_below_05 == 0: # code this out if u want all
            continue 
    counts, _ = np.histogram(stability_transformed[col], bins=bins)
    cumulative_counts += counts

# Plot cumulative histogram
plt.figure(figsize=(8, 5))
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.bar(bin_centers, cumulative_counts, width=bins[1]-bins[0], color='blue', alpha=0.7)
plt.yscale("log")
plt.xlabel('Value')
plt.ylabel('Count (log scale)')
plt.title('Cumulative Histogram of 54 Transformed Columns')
plt.savefig("cumulative_stability.png")




