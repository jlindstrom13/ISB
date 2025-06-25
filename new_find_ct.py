from sqlite_utils import Database

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


print(f"Fifth entry in papers")
print(list(papers.items())[4])

print(f"10th entry in trials")
print(list(trials.items())[9])

print(f"Length of papers dict:{len(papers)}")
print(f"Length of trials dict:{len(trials)}")
