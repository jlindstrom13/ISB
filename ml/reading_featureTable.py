# Code to read in the FeatureTable from /15TB_2/gglusman/clinicaltrials/buildFeatureTable.py

import pandas as pd

file_path = "/15TB_2/gglusman/clinicaltrials/featureTable.tsv.gz"

df = pd.read_csv(file_path, sep="\t", compression="gzip", dtype=str)

print(df.head())


df.to_pickle("featureTable.pkl")