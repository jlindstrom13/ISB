# Creating table to label retractions (NCT | PMID | 0/1)

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
