# Labeling 0 1 the matched cases and controls

import pandas as pd


df = pd.read_pickle("matched_cases_controls.pkl")

df_features = pd.read_pickle("featureTable.pkl")

print(f"df_features column names {df_features.columns}")


cases = df[['cases']].copy()
cases.columns = ['nct_id']
cases['label'] = 1

controls = df[['controls']].copy()
controls.columns = ['nct_id']
controls['label']=0

labels_df = pd.concat([cases, controls], ignore_index=True)

df_features_labeled = df_features.merge(labels_df, on='nct_id', how='inner')  # inner join keeps only matched rows

print(df_features_labeled['label'].value_counts())

df_features_labeled.to_pickle("matched_featureTable_labeled.pkl")
