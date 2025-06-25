# Using pubmed.db and ct.gov, label all ct as FAIL if associated with retracted paper

# code from new_retracted.py

from sqlite_utils import Database

pubmedDbFile = "/ssd/sqlite/PubMed.db"
pubmed = Database(pubmedDbFile)

papers = {} # Dictionary that maps NCT : list of PMIDs (could be multiple)
trials = {} # Dictionary that maps PMID: list of NCT

#first, only examine retracted papers
retracted=set(row[0] for row in pubmed.execute("SELECT pmid FROM type WHERE type = 'D016441';")
)


for pmid, acc in pubmed.execute("SELECT pmid, acc FROM acc;"):
	if pmid not in retracted: continue #ignore pmid not in retracted
	if not acc.startswith("NCT"): continue 
	if acc not in papers: papers[acc] = [] 
	papers[acc].append(pmid) 
	if pmid not in trials:
		trials[pmid] = []  
	trials[pmid].append(acc)
	
## HELP! How do i access rows from clinicaltrials/zips/ all the tables of .txt files??

ct_path = "/users/jlindstr/clinicaltrials/zips"
ct_folder = Database(ct_path) # I know this isn't going to work bc it isnt' a database 