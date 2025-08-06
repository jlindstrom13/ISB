# Label Clinical Trial Feature Table Using Defined Criteria
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

# Load the feature table
df = pd.read_pickle("featureTable.pkl")


# Load lists of "untrustworthy" trial NCTs: 
# - Discrepant trials
# - Retracted trials (PubMed and ClinicalTrials.gov)
# - Trials flagged as unstable (from stability transformation)
# - Retracted Trials: RetractionWatch flagged trials

discrepant_nctids = pd.read_pickle("/users/jlindstr/code/discrepancies/discrepant_unique_nctids.pkl")
retracted_pm_nctids = pd.read_pickle("/users/jlindstr/code/retractions/retracted_pm_ncts.pkl")
retracted_ct_nctids = pd.read_pickle("/users/jlindstr/code/retractions/retracted_ct_ncts.pkl")
stability_nctids = pd.read_pickle("/users/jlindstr/code/stability_transformation/trials_above_cutoff4.pkl")
rwatch_nctids = pd.read_pickle("/users/jlindstr/code/retractions/r_watch_ncts.pkl")

print(f" length of discrepant trials: {len(discrepant_nctids)}") 
print(f" length of pm retracted trials: {len(retracted_pm_nctids)}") 
print(f" length of ct retracted trials: {len(retracted_ct_nctids)}") 
print(f" length of unstable trials: {len(stability_nctids)}") 
print(f" length of rwatch trials: {len(rwatch_nctids)}") 

# Combine all untrustworthy trials into a single set using union
all_ncts_1 = set(discrepant_nctids) | set(retracted_pm_nctids) | set(stability_nctids) | set(retracted_ct_nctids) | set(rwatch_nctids)

print(f" length of untrustworthy trials: {len(all_ncts_1)}") #11,848 currently....

# Label the feature table
# - 1 = untrustworthy (case)
# - NaN = unlabeled (will be used later for controls and production
df['label'] = np.nan  # set all to NA
df.loc[df['nct_id'].isin(all_ncts_1), 'label'] = 1

# Save labeled feature table to pkl for other ML uses
df.to_pickle("featureTable_labeled.pkl")
print(f"Length of labeled featuretable: {len(df)}")

# To be used in matching.py and Gwênlyn's matchCasesControls.py:
cases_df = df[df['label'] == 1].copy()

print(f"Number of trials labeled cases:{len(cases_df)}")

cases_df.to_pickle("cases_df.pkl")

# Extract trials without a label (NaN) to use as production/unlabeled set
# note... is this too many? where are trials labeled 0? 
df_production = df[df['label'].isna()].copy()

df_production.to_pickle("production.pkl")

print(df_production.shape)


# Investigating where the 8 missing NCTs are...
missing_ncts = all_ncts_1 - set(df['nct_id'])
print(f"Missing NCTS: {missing_ncts}")
print(len(missing_ncts))  


missing_in_rwatch = missing_ncts.intersection(set(rwatch_nctids))
print("Missing NCTs that are in rwatch_nctids:", missing_in_rwatch)
print("Missing NCTs in discrepant_nctids:", missing_ncts.intersection(set(discrepant_nctids)))
print("Missing NCTs in retracted_pm_nctids:", missing_ncts.intersection(set(retracted_pm_nctids)))
print("Missing NCTs in retracted_ct_nctids:", missing_ncts.intersection(set(retracted_ct_nctids)))
print("Missing NCTs in stability_nctids:", missing_ncts.intersection(set(stability_nctids)))