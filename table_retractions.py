# Creating table to label retractions (NCT | PMID | 0/1)

from sqlite_utils import Database
import pandas as pd

pubmedDbFile = "/ssd/sqlite/PubMed.db"
pubmed = Database(pubmedDbFile)

papers = {} # Dictionary that maps NCT : list of PMIDs (could be multiple)
trials = {} # Dictionary that maps PMID: list of NCT 

for pmid, acc in pubmed.execute("SELECT pmid, acc FROM acc;"):
	if not acc.startswith("NCT"): continue #skips over acc values that don't have NCT start
	if acc not in papers: papers[acc] = [] #if first appearance of acc in papers, add it in
	papers[acc].append(pmid) #adds pmid 
	if pmid not in trials: trials[pmid] = []  #Note: can't be { } can't have dict of dict
	trials[pmid].append(acc)

# here we have two dictionaries... now need to create table 
rows = []
for NCT, pmid in papers.items():
	for each_pmid in pmid:
	    rows.append({"NCT": NCT, "PMID": each_pmid})
		
table = pd.DataFrame(rows)
print(table.head())
print(len(table))

# Add a column of 0 and 1s for retracted papers
# 1 = retracted pmid, 0 = not retracted pmid

# first function to get pmid by mesh DO.... value
def get_pmid_by_type(type_mesh_DO):
	return list(
		row[0] for row in pubmed.execute(
			"SELECT pmid FROM type WHERE type = ?;", [type_mesh_DO]))

# test= get_pmid_by_type("D016441") # Retractions: DO16441
# print(f"trying to use function {test[5]}")

retracted_pmids = set(get_pmid_by_type("D016441")) # retracted & change to set (no repeats)

table["retracted"] = table["PMID"].isin(retracted_pmids).astype(int)

print(table.head())


