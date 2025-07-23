# Use pkl of discrepant NCTs to label the feature table 0-1
# Random forest ML below
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

# Printing 4 random "untrustworthy" NCTs
ncts_list = list(all_ncts_1)

random_5_ncts = random.sample(ncts_list, 5)

print(f"Random 5 NCTs from untrustworthy label: {random_5_ncts}")

