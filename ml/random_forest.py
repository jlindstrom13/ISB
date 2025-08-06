import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import shap
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV


# Loading data:
df = pd.read_pickle("matched_featureTable_labeled.pkl") #160 columns/ features

# Splitting data into x and y:
X = df.drop(columns=["nct_id", "label"])
y = df["label"]
mask = y.notna()
X = X[mask]
y = y[mask].astype(int)

# delete empty columns (can't replace with median! when imputing)
empty_cols = X.columns[X.isna().all()]
print("Dropping empty columns from full dataset:", empty_cols.tolist())
X = X.drop(columns=empty_cols)
X = X.drop(columns=['studies:fdaaa801_violation'])
#splitting into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

# RF pipeline
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(random_state=36, 
                                  n_estimators=400, 
                                  max_depth =None,
                                  max_features = 'sqrt',
                                  min_samples_leaf =  5,
                                  min_samples_split =  20))
])
#'rf__max_depth': None, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 5, 'rf__min_samples_split': 20, 'rf__n_estimators': 400



# uses gridsearch CV to see which params maximize train and test error
# param_grid = {
#     'rf__n_estimators': [200, 300, 400],
#     'rf__max_depth': [8, 10, 12, None],
#     'rf__min_samples_split': [5, 10, 20],
#     'rf__min_samples_leaf': [1, 2, 5],
#     'rf__max_features': ['sqrt', 'log2']
# }

# grid = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid.fit(X_train, y_train)

# print("Best params:", grid.best_params_)
# print("Best CV accuracy:", grid.best_score_)

# Training
rf_pipeline.fit(X_train, y_train)

y_pred = rf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# confusion matrix aka 2x2 or contingency table
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)


# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=['Predicted 0', 'Predicted 1'],
#             yticklabels=['Actual 0', 'Actual 1'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Matched Random Forest Contingency Table')
# plt.savefig("Contingency_tables/rf_contingency_matched_k.png")
# plt.close()


#Training accuracy
y_train_pred = rf_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

#Test accuracy

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# # Importances
rf_model = rf_pipeline.named_steps['rf']
# importances = rf_model.feature_importances_
# feature_names = X.columns
# feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
# for feat, imp in feat_imp[:10]:
#     print(f"{feat}: {imp:.4f}")

# making shap plot- using imputed values

valid_columns = X_train.columns # just the columns that were kept for training!

X_train_imputed = rf_pipeline.named_steps['imputer'].transform(X_train)
X_test_imputed = rf_pipeline.named_steps['imputer'].transform(X_test)

X_test_sample = pd.DataFrame(X_test_imputed[:100], columns=valid_columns)

# SHAP explainer and values
explainer = shap.TreeExplainer(rf_model, X_train_imputed)
shap_values = explainer(X_test_sample)

# Plot summary
plt.figure(figsize=(8, 4))
shap.summary_plot(shap_values[:, :, 1], X_test_sample, max_display= 20)
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.title("Matched Random Forest, Untrustworthy Trials, v2", fontsize=12)
plt.tight_layout()
plt.savefig("rf_shap_matched2.png")
plt.close()


# Production time!

production = pd.read_pickle("production.pkl")

X_prod = production.drop(columns=["nct_id", "label"] + empty_cols.tolist(),  errors='ignore')
X_prod = X_prod.drop(columns=['studies:fdaaa801_violation'], errors='ignore')

# Prediction
prod_preds = rf_pipeline.predict(X_prod)
prod_pred_probs = rf_pipeline.predict_proba(X_prod)

# Save predictions
production["predicted_label"] = prod_preds
production["prob_0"] = prod_pred_probs[:, 0]
production["prob_1"] = prod_pred_probs[:, 1]

# Export
production.to_pickle("production_predictions/rf_production_predictions2.pkl")
production.to_csv("production_predictions/rf_production_predictions2.csv", index=False)


