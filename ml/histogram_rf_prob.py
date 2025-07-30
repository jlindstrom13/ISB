import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_pickle("rf_production_predictions.pkl")

plt.figure(figsize=(8, 5))
sns.histplot(df["prob_1"], bins=100, kde=False)

plt.xlabel("Probability of Being Untrustworthy (prob_1)", fontsize=12)
plt.ylabel("Count, log scale", fontsize=12)
plt.yscale("log")
plt.title("Distribution of Predicted Probabilities for Untrustworthy", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_prob1_distribution.png")
