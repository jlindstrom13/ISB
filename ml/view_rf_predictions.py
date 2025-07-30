# View pkl file with RF predictions
import pandas as pd


file_path = "rf_production_predictions.pkl"

df = pd.read_pickle(file_path)

print(df.shape)

# Top 5 most confident in class 0
top_prob_0 = df.sort_values(by="prob_0", ascending=False).head(5)

# Top 5 most confident in class 1
top_prob_1 = df.sort_values(by="prob_1", ascending=False).head(5)

print("Top 5 most confident in class 0, Okay:")
print(top_prob_0[["nct_id", "prob_0", "prob_1"]])

print("\nTop 5 most confident in class 1, Untrustworthy:")
print(top_prob_1[["nct_id", "prob_0", "prob_1"]])