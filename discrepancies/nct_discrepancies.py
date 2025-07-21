import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

# Making a list of ALL NCTs identified as having a discrepancy (there will be duplicates)
nct_ids = [
    # Death or survival an outcome
    "NCT00379769", "NCT00095238",

    # Death or survival NOT an outcome
    "NCT00313313", "NCT00092521", "NCT00441545", "NCT00463866", "NCT00489736", "NCT00081770",

    # SAE reporting discrepancies (>0 SAEs diff than publication)
    "NCT00323492", "NCT00097591", "NCT00075218", "NCT00603798", "NCT00092521", "NCT00087516",
    "NCT00462748", "NCT00460564", "NCT00094458", "NCT00355615", "NCT00402324", "NCT00271817",
    "NCT00095238", "NCT00285012", "NCT00313313", "NCT00393718", "NCT00727857", "NCT00102440",
    "NCT00494013", "NCT00450437", "NCT00092677",

    # SAE reported in CT.gov – count missing in publication
    "NCT00080301", "NCT00487240", "NCT00003869", "NCT00308711", "NCT00046228", "NCT00115765",
    "NCT00318461", "NCT00129402", "NCT00300482", "NCT00225277", "NCT00441545", "NCT00552669",

    # SAE reported in CT.gov – 0 in publication
    "NCT00975507", "NCT00236938", "NCT00360490", "NCT00363142",

    # Zero SAEs reported in CT.gov
    "NCT00029172",

    # Primary outcome discrepancy – inconsistent values
    "NCT00095238", "NCT00833794", "NCT00879697", "NCT00462748", "NCT00294515", "NCT00406783",
    "NCT00257660", "NCT00104247", "NCT00552669",

    # Minor differences / unclear impact
    "NCT00287053", "NCT00029172", "NCT00494013", "NCT00422734", "NCT00806403", "NCT00852917",
    "NCT00337727", "NCT00452426", "NCT00313820", "NCT00432237", "NCT01218958", "NCT00308711",
    "NCT00886600"
]

unique_nctids = list(set(nct_ids))

pd.Series(unique_nctids).to_pickle("discrepant_unique_nctids.pkl")

print(len(unique_nctids))


death_outcome_reported = {
    "NCT00379769", "NCT00095238", "NCT00240331", "NCT00092677", "NCT00158600", "NCT00806403",
    "NCT00552669", "NCT00144339", "NCT00046228", "NCT00105443", "NCT00003869", "NCT00080301",
    "NCT00115765", "NCT00075218"
}

death_outcome_not_reported = {
    "NCT00313313", "NCT00092521", "NCT00441545", "NCT00463866", "NCT00489736", "NCT00081770",
    "NCT00152763", "NCT00402324", "NCT00852917", "NCT00479713", "NCT00094458", "NCT00289848",
    "NCT00285012", "NCT00132808", "NCT00127712"
}

sae_discrepancy_diff_count = {
    "NCT00323492", "NCT00097591", "NCT00075218", "NCT00603798", "NCT00092521", "NCT00087516",
    "NCT00462748", "NCT00460564", "NCT00094458", "NCT00355615", "NCT00402324", "NCT00271817",
    "NCT00095238", "NCT00285012", "NCT00313313", "NCT00393718", "NCT00727857", "NCT00102440",
    "NCT00494013", "NCT00450437", "NCT00092677"
}

sae_discrepancy_count_missing = {
    "NCT00080301", "NCT00487240", "NCT00003869", "NCT00308711", "NCT00046228", "NCT00115765",
    "NCT00318461", "NCT00129402", "NCT00300482", "NCT00225277", "NCT00441545", "NCT00552669"
}

sae_discrepancy_zero_reported = {
    "NCT00975507", "NCT00236938", "NCT00360490", "NCT00363142"
}

zero_saes_ctgov = {"NCT00029172"}

primary_outcome_discrepancy = {
    "NCT00095238", "NCT00833794", "NCT00879697", "NCT00462748", "NCT00294515", "NCT00406783",
    "NCT00257660", "NCT00104247", "NCT00552669"
}

primary_outcome_minor_diff = {
    "NCT00287053", "NCT00029172", "NCT00494013", "NCT00422734", "NCT00806403", "NCT00852917",
    "NCT00337727", "NCT00452426", "NCT00313820", "NCT00432237", "NCT01218958", "NCT00308711",
    "NCT00886600"
}


all_nctids = (
    death_outcome_reported | death_outcome_not_reported |
    sae_discrepancy_diff_count | sae_discrepancy_count_missing |
    sae_discrepancy_zero_reported | zero_saes_ctgov |
    primary_outcome_discrepancy | primary_outcome_minor_diff
) # | means union and gets all unique ones


df = pd.DataFrame({
    "nctid": list(all_nctids)
}) 

# TRUE / FALSE in each column for if that nctid has that criteria
df["death_outcome_reported"] = df["nctid"].isin(death_outcome_reported)
df["sae_discrepancy_diff_count"] = df["nctid"].isin(sae_discrepancy_diff_count)
df["sae_discrepancy_count_missing"] = df["nctid"].isin(sae_discrepancy_count_missing)
df["sae_discrepancy_zero_reported"] = df["nctid"].isin(sae_discrepancy_zero_reported)
df["zero_saes_ctgov"] = df["nctid"].isin(zero_saes_ctgov)
df["primary_outcome_discrepancy"] = df["nctid"].isin(primary_outcome_discrepancy)
df["primary_outcome_minor_diff"] = df["nctid"].isin(primary_outcome_minor_diff)




