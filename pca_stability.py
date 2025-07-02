# https://pca4ds.github.io/handling-missing-values.html

import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')  # or sep='|'

stability_clean = stability_df

# data cleaning: addressing missing values
stability_clean =stability_clean.select_dtypes(include=[np.number])

# Apply mean imputation to fill missing values with MEAN
imputer = SimpleImputer(strategy='mean')
stability_imputed = imputer.fit_transform(stability_clean)

# try it not normalized
pca=PCA(n_components=2)
pca.fit(stability_imputed)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))
#standardization:  center so mean of each column = 0, normalize by dividing each    
# variable by lenght of vector ( values from 0-1)

scaler = StandardScaler()
stability_standardized = scaler.fit_transform(stability_imputed)

pca=PCA(n_components=2)
pca.fit(stability_standardized)

print("(Standardized) Explained variance ratio:", pca.explained_variance_ratio_)
print("(Standardized) Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Component Number')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.xticks(range(1, 3))
plt.grid(True, axis='y')
plt.savefig("scree_plot")

loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=stability_clean.columns)
print(loadings.sort_values('PC1', ascending=False).head(10))