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

# SHAP Plot..

X_background = X_train.sample(100, random_state=36)
X_sample = X_test.sample(50, random_state=36)

# Preprocess these manually (impute, scale, select features)
preprocessor = Pipeline([
    ('imputer', nn_pipeline.named_steps['imputer']),
    ('minmax', nn_pipeline.named_steps['minmax']),
    ('chi2', nn_pipeline.named_steps['chi2'])
])

# Sample background and sample data
X_background = X_train.sample(100, random_state=36)
X_sample = X_test.sample(50, random_state=36)

# Preprocessing pipeline to transform data consistently
preprocessor = Pipeline([
    ('imputer', nn_pipeline.named_steps['imputer']),
    ('minmax', nn_pipeline.named_steps['minmax']),
    ('chi2', nn_pipeline.named_steps['chi2'])
])


X_background = X_train.sample(100, random_state=36)
X_sample = X_test.sample(100, random_state=36)

preprocessor = Pipeline([
    ('imputer', nn_pipeline.named_steps['imputer']),
    ('minmax', nn_pipeline.named_steps['minmax']),
    ('chi2', nn_pipeline.named_steps['chi2'])
])

X_background_transformed = preprocessor.transform(X_background)
X_sample_transformed = preprocessor.transform(X_sample)

def predict_fn(X):
    return nn_pipeline.named_steps['nn'].predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(predict_fn, X_background_transformed)

shap_values = explainer.shap_values(X_sample_transformed, nsamples=100)

selected_features = X.columns[nn_pipeline.named_steps['chi2'].get_support()]

X_sample_df = pd.DataFrame(X_sample_transformed, columns=selected_features)

plt.figure(figsize=(8, 8))
shap.summary_plot(shap_values, X_sample_df, max_display=20, show=False)

plt.title("SHAP Plot for Neural Net, Untrustworthy Trials")
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("nn_shap_1.png")
plt.close()