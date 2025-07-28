# Use pkl of discrepant NCTs to label the feature table 0-1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random


df = pd.read_pickle("featureTable.pkl")

# TO ADD:
 # retractions/retracted_ncts.pkl   discrepancies/discrepant_unique_nctids.pkl
 # stability_transformation/trials_above_cutoff4.pkl
 # retractions/r_watch_ncts.pkl (143)

discrepant_nctids = pd.read_pickle("/users/jlindstr/code/discrepancies/discrepant_unique_nctids.pkl")
retracted_nctids = pd.read_pickle("/users/jlindstr/code/retractions/retracted_ncts.pkl")
stability_nctids = pd.read_pickle("/users/jlindstr/code/stability_transformation/trials_above_cutoff4.pkl")
rwatch_nctids = pd.read_pickle("/users/jlindstr/code/retractions/r_watch_ncts.pkl")

print(f" length of discrepant trials: {len(discrepant_nctids)}") 
print(f" length of retracted trials: {len(retracted_nctids)}") 
print(f" length of unstable trials: {len(stability_nctids)}") 
print(f" length of rwatch trials: {len(rwatch_nctids)}") 



all_ncts_1 = set(discrepant_nctids) | set(retracted_nctids) | set(stability_nctids)

print(f" length of untrustworthy trials: {len(all_ncts_1)}") #11,848 currently....

df['label'] = np.nan  # set ALL to NA

df.loc[df['nct_id'].isin(all_ncts_1), 'label'] = 1


# Choose random trials to label as 0 (aka OK)... same amount as untrustworthy:

num_ncts = len(all_ncts_1)

unlabeled_nctids = df.loc[df['label'].isna(), 'nct_id']

np.random.seed(36)  
nct_0 = np.random.choice(unlabeled_nctids, size=num_ncts, replace=False)

df.loc[df['nct_id'].isin(nct_0), 'label'] = 0

# Save df to pkl for other ML uses
df.to_pickle("featureTable_labeled.pkl")

# To be used in matching.py :
cases_df = df[df['label'] == 1].copy()
controls_df = df[df['label'] == 0].copy()

cases_df.to_pickle("cases_df.pkl")
controls_df.to_pickle("controls_df.pkl")