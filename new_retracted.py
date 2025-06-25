# Using pubmed.db, create two dictionaries to map NCT --> published papers and the reverse

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

print(f"Number of clinical trials that have 1+ retracted papers associated with it: {len(papers)}")
print(f"Number of retracted papers that have 1+ trials associated with it: {len(trials)}")
print(f"Entry 5 in papers: {list(papers.items())[4]}")
print(f"Entry 5 in trials: {list(trials.items())[4]}")
