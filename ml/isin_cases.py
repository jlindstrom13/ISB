import pandas as pd


cases= pd.read_pickle("cases_df.pkl")


nct_id = "NCT01143480"

if nct_id in cases["nct_id"].values:
    print(f"{nct_id} is in cases_df.")
else:
    print(f"{nct_id} is NOT in cases_df.")