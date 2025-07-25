# ML using Naive Bayes Classifier and Increasing Number of Controls. Potentially random oversampling 
import argparse
import pandas as pd
from imblearn.pipeline import Pipeline
import shap
from shap import KernelExplainer
from imblearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np


# CHANGE: want to increase number of controls ....


# Loading data:
df = pd.read_pickle("featureTable_labeled_3xcontrol.pkl") #160 columns/ features

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

# Pipeline for Naive Bayes

naive_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('minmax', MinMaxScaler()),
    ('chi2', SelectKBest(chi2, k=20)),
    ('oversample', RandomOverSampler(random_state=36)),
    ('nb', GaussianNB())
])

naive_pipeline.fit(X_train, y_train)


y_pred = naive_pipeline.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Training accuracy
y_train_pred = naive_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# confusion matrix aka 2x2 or contingency table
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# # SHAP Plot 
# Step 1: Transform X using the pipeline (excluding the model)

X_background = X_train.sample(100, random_state=36)
X_sample = X_test.sample(100, random_state=36)

preprocessor = Pipeline([
    ('imputer', naive_pipeline.named_steps['imputer']),
    ('minmax', naive_pipeline.named_steps['minmax']),
    ('chi2', naive_pipeline.named_steps['chi2'])
])

X_background_transformed = preprocessor.transform(X_background)
X_sample_transformed = preprocessor.transform(X_sample)


def predict_fn(X):
    return naive_pipeline.named_steps['nb'].predict_proba(X)[:, 1] #probability of class 1: untrustworthy

explainer = shap.KernelExplainer(predict_fn, X_background_transformed)

shap_values = explainer.shap_values(X_sample_transformed, nsamples=100)

selected_features = X.columns[naive_pipeline.named_steps['chi2'].get_support()]

X_sample_df = pd.DataFrame(X_sample_transformed, columns=selected_features)


plt.figure(figsize=(8, 4))
shap.summary_plot(shap_values, X_sample_df, max_display= 20)
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.title("SHAP Plot for Naive Bayes, Untrustworthy Trials", fontsize=12)
plt.tight_layout()
plt.savefig("naive_shap_1.png")
plt.close()