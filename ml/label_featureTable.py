# Use pkl of discrepant NCTs to label the feature table 0-1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

np.random.seed(42)  
nct_0 = np.random.choice(unlabeled_nctids, size=num_ncts, replace=False)

df.loc[df['nct_id'].isin(nct_0), 'label'] = 0

# drop NCT ID column and label column
X = df.drop(columns=["nct_id", "label"])

# label variable only
y = df["label"]


# First ML attempt.... using random forest
X = X[y.notna()]
y = y[y.notna()].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# confusion matrix aka 2x2 or contingency table
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Contingency Table')
plt.savefig("contingency_table.png")


#Training accuracy
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")


#Test accuracy

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")