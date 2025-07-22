# Creating table to label retractions (NCT | PMID | 0/1)

from sqlite_utils import Database
import pandas as pd

pubmedDbFile = "/ssd/sqlite/PubMed.db"
pubmed = Database(pubmedDbFile)


# first function to get pmid by mesh DO.... value
def get_pmid_by_type(type_mesh_DO):
    return list(
        row[0]
        for row in pubmed.execute(
            "SELECT pmid FROM type WHERE type = ?;", [type_mesh_DO]
        )
    )


papers = {}  # Dictionary that maps NCT : list of PMIDs (could be multiple)
trials = {}  # Dictionary that maps PMID: list of NCT

for pmid, acc in pubmed.execute("SELECT pmid, acc FROM acc;"):
    if not acc.startswith("NCT"):
        continue  # skips over acc values that don't have NCT start
    if acc not in papers:
        papers[acc] = []  # if first appearance of acc in papers, add it in
    papers[acc].append(pmid)  # adds pmid
    if pmid not in trials:
        trials[pmid] = []  # Note: can't be { } can't have dict of dict
    trials[pmid].append(acc)

# here we have two dictionaries... now need to create table
rows = []
for NCT, pmid in papers.items():
    for each_pmid in pmid:
        rows.append({"NCT": NCT, "PMID": each_pmid})

table = pd.DataFrame(rows)


# Add a column of 0 and 1s for retracted papers
# 1 = retracted pmid, 0 = not retracted pmid
# Multiple of the same NCT listed, maps to diff publications

retracted_pmids = set(
    get_pmid_by_type("D016441")
)  # retracted & change to set (no repeats)

table["retracted"] = table["PMID"].isin(retracted_pmids).astype(int)

print(table.head())

print(len(table))


# ct_pmid = set(get_pmid_by_type("D016430")) # ct mesh

# table["MeSH_ct"] = table["PMID"].isin(ct_pmid).astype(int)

# print(table[table["MeSH_ct"] == 0].head())
# conclusion: there are some trials not labeled CT MeSH but do have an NCT number associated w trial
print(table['retracted'].value_counts())



# Going other way... table of multiple of same pmid mapping to diff ncts
# silly.. this actually is the same exact table as above
rows = []
for pmid, nct_list in trials.items():
    for each_nct in nct_list:
        rows.append({"PMID": pmid, "NCT": each_nct})

pmid_nct_table = pd.DataFrame(rows)

pmid_nct_table["retracted"] = pmid_nct_table["PMID"].isin(retracted_pmids).astype(int)

print(pmid_nct_table.head())
print(pmid_nct_table["retracted"].value_counts())


retracted_ncts = pmid_nct_table[pmid_nct_table["retracted"] == 1]["NCT"].unique()

pd.Series(retracted_ncts).to_pickle("retracted_ncts.pkl")
