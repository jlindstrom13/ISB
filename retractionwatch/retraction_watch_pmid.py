import pandas as pd

df = pd.read_csv("retraction_watch.csv")

print(df[['RetractionPubMedID', 'OriginalPaperPubMedID', 'Reason']].head())

print(df.columns)

count_with_pmid = df['OriginalPaperPubMedID'].notnull().sum()

print(f"Number of entries with OriginalPaperPubMedID: {count_with_pmid}")


pd.Series( df['OriginalPaperPubMedID']).to_pickle("retracted_pmids.pkl")
