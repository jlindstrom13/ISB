import pandas as pd

file_path = "matched_featureTable_labeled.pkl"

df = pd.read_pickle(file_path)

print(df.head())

print(df.columns)
print(df.shape)