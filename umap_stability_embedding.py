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

#make a copy with NCTID:
#stability_with_id = stability_df_filtered.copy()
#nct_ids = stability_with_id['nct_id']

# don't include NCTID dtype='object'
stability_df_no_nct = stability_df_filtered.select_dtypes(include=[np.number])

# Apply mean imputation to fill missing values with MEAN
imputer = SimpleImputer(strategy='mean')
stability_imputed = imputer.fit_transform(stability_df_no_nct)

# construct a umap object
reducer = umap.UMAP()

#standardization:  center so mean of each column = 0, normalize by dividing each    
# variable by lenght of vector ( values from 0-1)

scaler = StandardScaler()
stability_standardized = scaler.fit_transform(stability_imputed)

# take a subset of first 10,000 to make it faster
# stability_subset = stability_standardized[:10000]

embedding = reducer.fit_transform(stability_standardized)
#shape_output = embedding.shape
#print(shape_output) --> should be (x,2) numpy array

#save embedding so you don't have to rerun it each time
np.save("umap_embedding.npy", embedding)
