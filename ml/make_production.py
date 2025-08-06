import pandas as pd

# Load full feature table and matched trials
df = pd.read_pickle("featureTable_labeled.pkl")
matched = pd.read_pickle("matched_cases_controls.pkl")

# Exclude all NCTs that were used in training (cases + controls)
training_ncts = set(matched['nct_id'])
production_df = df[~df['nct_id'].isin(training_ncts)].copy()

# Confirm it's only unlabeled rows
assert production_df['label'].isna().all(), "Some labeled trials are in production set!"

# Save production set
production_df.to_pickle("production.pkl")
print(f"Saved production set with {len(production_df)} trials.")


n_cases = sum(matched['label'] == 1)
n_controls = sum(matched['label'] == 0)
print(f"  Cases: {n_cases}")
print(f"  Controls: {n_controls}")