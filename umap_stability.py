# UMAP dimension reduction for stability score clinical trial data

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#read in data
stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')  # or sep='|'

# filter out top 25% of rows with most missing data
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
stability_df_filtered = stability_df.loc[:, na_percent <= q3]

# don't include NCTID dtype='object'
stability_df_no_nct = stability_df_filtered.select_dtypes(include=[np.number])


