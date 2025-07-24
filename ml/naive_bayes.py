# ML using Naive Bayes Classifier and Increasing Number of Controls. Potentially random oversampling 
import argparse
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# CHANGE: want to increase number of controls ....


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

# Pipeline for Naive Bayes

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)

naive_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
    ('scaler', StandardScaler())
    ('nb', GaussianNB())
])

naive_pipeline.fit(X_train, y_train)