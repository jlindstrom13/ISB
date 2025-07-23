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

df = pd.read_pickle("featureTable_labeled.pkl") #160 columns/ features

# Split data 
X = df.drop(columns=["nct_id", "label"])
y = df["label"]
X = X[y.notna()]
y = y[y.notna()].astype(int)

# cant have a column of just NAs 
empty_cols = X.columns[X.isna().all()]
print("Dropping empty columns:", empty_cols.tolist())
X = X.drop(columns=empty_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

# Pipeline with scaling + neural network
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
                         max_iter=500, 
                         random_state=36))
])

# training
nn_pipeline.fit(X_train, y_train)

y_pred = nn_pipeline.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Training accuracy
y_train_pred = nn_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

