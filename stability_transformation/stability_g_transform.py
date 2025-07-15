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


# ...so simply log(0.5)/log(median)

stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')

# remove columns with too many NA
# na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
# q3 = np.percentile(na_percent, 75)
# stability_df_filtered = stability_df.loc[:, na_percent <= q3]

nct_ids = stability_df['nctid'].reset_index(drop=True) # just nct_ids
nct_ids.to_pickle("nct_ids.pkl")

# Df with NO NCT
stability_df_no_nct = stability_df.select_dtypes(include=[np.number])


# function
def median_rescale_to_95(col):
    median_val = col.median() # should avoid NaNs automatically
    if median_val <= 0:
        return col 
    power = np.log(0.95) / np.log(median_val)
    return col ** power

stability_transformed = stability_df_no_nct.apply(median_rescale_to_95)

# Drop plots that a) have lots of values below say 0.5
# b) ones that have none below say 0.5

# Sum up amount of trials less than 0.5 for each feature
stability_lessthan_05 = stability_transformed <0.5
print(f"true or false for stability scores is < 0.5{stability_lessthan_05.head(3)}")
count_lessthan_05 = stability_lessthan_05.sum(axis=0)
print(f"how many trials are less than 0.5 per feature{count_lessthan_05.head(20)}")
print(count_lessthan_05.shape)

count_lessthan_05.to_csv('count_stability', sep="\t")



plt.scatter(count_lessthan_05+1, count_lessthan_05+1, c="blue", s=15, alpha=0.2)
plt.xscale("log")
plt.yscale("log")
plt.title("Scatterplot stability transformed")
plt.savefig("scatter_transformed.png")



# max_count = count_lessthan_05.max()

# sns.histplot(count_lessthan_05+1,  
#              color='blue')
# plt.xlabel("Number of trials less than 0.5")
# plt.xscale("log")
# plt.ylabel("Number of Features that Have that Number of Trials")
# plt.title("number of features with what number of trials less than 0.5")
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.savefig("features_vs_number.png")
# plt.close()



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
        plt.xlim(0,1)
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

# PLOT Number of Features vs the Fraction of Trials that are outliers according to above cutoff

# below_05 = stability_transformed < 0.5  # get number of outliers (values/trials below 0.5)

# fraction_trials_below_05 = below_05.mean(axis=0)  # count how many trials below 0.5 for each feature
# print(fraction_trials_below_05)
# print(fraction_trials_below_05.shape)

# sns.histplot(fraction_trials_below_05, 
#              bins=np.linspace(0, 1, 101),  # same binning as before
#              color='blue')
# plt.xlabel("Fraction of Trials Below 0.5")
# plt.xlim(0, 0.1)
# plt.ylabel("Number of Features that Have that Fraction")
# plt.title("Number of Features vs Fraction of Trials Below 0.5")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("features_vs_fraction.png")
# plt.close()


