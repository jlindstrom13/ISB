# Making venn diagram of all retracted NCTS
# - RWatch
# - PubMed
# - Clinicaltrials.gov

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn3_circles

retracted_pm_nctids = pd.read_pickle("/users/jlindstr/code/retractions/retracted_pm_ncts.pkl")
retracted_ct_nctids = pd.read_pickle("/users/jlindstr/code/retractions/retracted_ct_ncts.pkl")
rwatch_nctids = pd.read_pickle("/users/jlindstr/code/retractions/r_watch_ncts.pkl")

print(f" length of pm retracted trials: {len(retracted_pm_nctids)}") 
print(f" length of ct retracted trials: {len(retracted_ct_nctids)}") 
print(f" length of rwatch trials: {len(rwatch_nctids)}") 

# Combine all untrustworthy trials into a single set using union
all_ncts = set(retracted_pm_nctids) | set(retracted_ct_nctids) | set(rwatch_nctids)

# Labels for clarity
A = set(retracted_pm_nctids)
B = set(retracted_ct_nctids)
C = set(rwatch_nctids)

# Plot
plt.figure(figsize=(6, 6))
venn = venn3([A, B, C], set_labels=('PubMed', 'ClinicalTrials.gov', 'Retraction Watch'))

# Optional: Customize appearance
for subset_id in ('100', '010', '001', '110', '101', '011', '111'):
    patch = venn.get_patch_by_id(subset_id)
    if patch:
        patch.set_alpha(0.7)

plt.title("Overlap of Retracted Trials (NCT IDs)")
plt.tight_layout()
plt.savefig("venn_retractions")


only_ct = B- (A|C)
for nct in sorted(only_ct)[:10]:
    print(nct)