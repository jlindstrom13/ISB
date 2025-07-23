import argparse
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap



# Command line arguments: 
parser = argparse.ArgumentParser(description="Train with custom epochs")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (max_iter)")
parser.add_argument("--k_features", type=int, default=96, help="Number of features to select with chi2")
args = parser.parse_args()

# Loading data:
df = pd.read_pickle("featureTable_labeled.pkl") #160 columns/ features

# Splitting data:
X = df.drop(columns=["nct_id", "label"])
y = df["label"]
X = X[y.notna()]
y = y[y.notna()].astype(int)

# Dropping empty columns:
empty_cols = X.columns[X.isna().all()]
print("Dropping empty columns:", empty_cols.tolist())
X = X.drop(columns=empty_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

# Pipeline with chi squared + neural network
nn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('minmax', MinMaxScaler()),     # Instead of standard schalar bc chi2 cant take negs
    ('chi2', SelectKBest(score_func=chi2, k=96)),         
    ('nn', MLPClassifier(hidden_layer_sizes=(64, 32), 
                         activation='relu', 
                         learning_rate="adaptive",
                         learning_rate_init=1e-4,
                         solver='adam', 
                         early_stopping=True,
                         max_iter=args.epochs, 
                         random_state=36))
])

# Training
nn_pipeline.fit(X_train, y_train)

y_pred = nn_pipeline.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Training accuracy
y_train_pred = nn_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

with open("nn_training_log.txt", "a") as f: #the "a" means append mode
    f.write(f"Epochs: {args.epochs}, # of Features: {args.k_features}, Test Acc: {accuracy_score(y_test, y_pred):.4f}, Train Accuracy: {train_accuracy:.4f}\n")


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
plt.title('Neural Net Contingency Table')
plt.savefig("nn_contingency_table.png")
plt.close()


# Get a confidence score:
probs = nn_pipeline.predict_proba(X_test)

probs_df = pd.DataFrame(probs, columns=["prob_0", "prob_1"], index=X_test.index)

nct_ids = df.loc[X_test.index, "nct_id"].reset_index(drop=True)
probs_df = probs_df.reset_index(drop=True)
probs_df["nct_id"] = nct_ids

#probs_df = probs_df.sort_values("prob_1", ascending=False)

# explainer = shap.KernelExplainer(nn_pipeline.predict,X_train)

# shap_values = explainer.shap_values(X_test,nsamples=100)

# shap.summary_plot(shap_values,X_test,feature_names=features)


# SHAP plot
# background = shap.sample(X_train, 100)  

# def model_predict(X):
#     return nn_pipeline.predict_proba(X)

# explainer = shap.KernelExplainer(model_predict, background)
# X_test_sample = X_test[:50]
# shap_values = explainer.shap_values(X_test_sample)  # Optional: limit to first 50 for speed

# X_test_transformed = nn_pipeline[:-1].transform(X_test_sample)  # all steps before 'nn'
# shap.summary_plot(shap_values[1], X_test_transformed) # [1] = class 1
# plt.savefig("nn_shap_summary_plot.png")
# plt.close()