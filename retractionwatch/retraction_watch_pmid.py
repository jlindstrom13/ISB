import pandas as pd

df = pd.read_csv("retraction_watch.csv")

print(df[['RetractionPubMedID', 'OriginalPaperPubMedID', 'Reason']].head())

print(df.columns)