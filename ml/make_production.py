import pandas as pd

# Load full feature table and matched trials
df = pd.read_pickle("featureTable_labeled.pkl")
matched = pd.read_pickle("matched_cases_controls.pkl")

print(matched.columns)
# Exclude all NCTs that were used in training (cases + controls)
training_ncts = set(matched['cases']).union(set(matched['controls']))
production_df = df[~df['nct_id'].isin(training_ncts)].copy()

# Save production 
production_df.to_pickle("production.pkl")
print(f"Number of trials saved to production: {len(production_df)} ")

# END product of production file


# Exploration/ Checking to make sure training and production NCTs are unique...
n_cases = matched['cases'].notna().sum()
n_controls = matched['controls'].notna().sum()
print(f"Cases: {n_cases}")
print(f"Controls: {n_controls}")
print(f" total cases and controls: {n_cases+n_controls}")


training_ncts = set(matched['cases']) | (set(matched['controls']))
print(f"Unique training NCTs: {len(training_ncts)}")

cases_set = set(matched['cases'].dropna())
controls_set = set(matched['controls'].dropna())

overlap_ncts = cases_set & controls_set


print(f"Number of NCTs in both cases and controls: {len(overlap_ncts)}")

for nct in list(overlap_ncts)[:5]:
    print(f"Overlap ex: {nct}")
