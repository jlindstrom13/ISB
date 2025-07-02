import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')  # or sep='|'

stability_clean = stability_df

# data cleaning
stability_clean = stability_df.drop(columns=['nctid'], errors='ignore')
stability_clean = stability_clean.select_dtypes(include=[np.number]).dropna()

pca=PCA(n_components=2)
pca.fit(stability_clean)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)