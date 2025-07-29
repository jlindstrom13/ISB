# Labeling 0 1 the matched cases and controls

import pandas as pd

file_path = "matched_cases_controls.pkl"

df = pd.read_pickle(file_path)

path = "featureTable.pkl"

df_features = pd.read_pickle(path)


cases = df[['cases']].copy()
cases.columns = ['nctid']
cases['label'] = 1

controls = df[['controls']].copy()
controls.columns = ['nctid']
controls['label']=0

labeled_df = pd.concat([cases, controls], ignore_index=True)


labeled_matched_features = df_features[df_features['nct_id'].isin(labeled_df['nctid'])]

print(labeled_matched_features.shape)