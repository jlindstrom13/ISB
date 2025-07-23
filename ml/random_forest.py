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


# Loading data:
df = pd.read_pickle("featureTable_labeled.pkl") #160 columns/ features

# Splitting data into x and y:
X = df.drop(columns=["nct_id", "label"])
y = df["label"]
mask = y.notna()
X = X[mask]
y = y[mask].astype(int)

# Dropping empty columns:
empty_cols = X.columns[X.isna().all()]
print("Dropping empty columns:", empty_cols.tolist())
X = X.drop(columns=empty_cols)

#splitting into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

# RF pipeline
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(random_state=36))
])


# Training
rf_pipeline.fit(X_train, y_train)

y_pred = rf_pipeline.predict(X_test)
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
plt.title('Random Forest Contingency Table')
plt.savefig("rf_contingency_table.png")
plt.close()


#Training accuracy
y_train_pred = rf_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

#Test accuracy

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Importances
rf_model = rf_pipeline.named_steps['rf']
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for feat, imp in feat_imp[:10]:
    print(f"{feat}: {imp:.4f}")

# making shap plot- using imputed values
X_train_imputed = rf_pipeline.named_steps['imputer'].transform(X_train)
X_test_imputed = rf_pipeline.named_steps['imputer'].transform(X_test)

X_test_sample = pd.DataFrame(X_test_imputed[:100], columns=X.columns)

# SHAP explainer and values
explainer = shap.TreeExplainer(rf_model, X_train_imputed)
shap_values = explainer(X_test_sample)

# Plot summary
plt.figure(figsize=(8, 4))
shap.summary_plot(shap_values[:, :, 1], X_test_sample, max_display= 20)
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("rf_shap_1.png")
plt.close()



