# https://pca4ds.github.io/handling-missing-values.html

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#read in data
stability_df = pd.read_csv('/15TB_2/gglusman/clinicaltrials/stability.txt.gz', sep='\t')  # or sep='|'

print(stability_df.head(4))

#Histogram of percent of NAs in one column
na_percent = stability_df.isna().mean().sort_values(ascending=False) * 100
q3 = np.percentile(na_percent, 75)
print(f"Drop all columns with % NA greater than q3:  {q3}")

stability_df_filtered = stability_df.loc[:, na_percent <= q3]

# don't include NCTID dtype='object'
stability_df_no_nct = stability_df_filtered.select_dtypes(include=[np.number])

# Apply mean imputation to fill missing values with MEAN
imputer = SimpleImputer(strategy='mean')
stability_imputed = imputer.fit_transform(stability_df_no_nct)

# try it not normalized
# pca=PCA(n_components=2)
# pca.fit(stability_imputed)
# print("Explained variance ratio:", pca.explained_variance_ratio_)
#print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

#standardization:  center so mean of each column = 0, normalize by dividing each    
# variable by lenght of vector ( values from 0-1)

scaler = StandardScaler()
stability_standardized = scaler.fit_transform(stability_imputed)

pca=PCA(n_components=2)
pca_result = pca.fit_transform(stability_standardized)

print("(Standardized) Explained variance ratio:", pca.explained_variance_ratio_)
print("(Standardized) Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

#Scree Plot:
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Component Number')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.xticks(range(1, 3))
plt.grid(True, axis='y')
plt.savefig("scree_plot")

loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=stability_df_no_nct.columns)
print(loadings.sort_values('PC1', ascending=False).head(10))

plt.figure(figsize=(8, 5))
plt.hist(na_percent, bins=30, color='steelblue', edgecolor='black')
plt.axvline(q3, color='red', linestyle='--', linewidth=2, label=f'75th percentile: {q3:.1f}%')
plt.xlabel('% NA')
plt.ylabel('# of Columns')
plt.title('Histogram of % NA per Column')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("na_percent_hist.png")

# Plot of PC1 vs PC2
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3, s=2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC1 vs PC2')
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_pc2.png")



outlier_idx = np.argmin(pca_result[:, 0])
print("Outlier trial index:", outlier_idx)

nct_id = stability_df.iloc[outlier_idx]['nctid']
print("Outlier NCT:", nct_id)