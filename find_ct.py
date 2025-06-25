from sqlite_utils import Database

pubmedDbFile = "/ssd/sqlite/PubMed.db"
pubmed = Database(pubmedDbFile)

def get_pmid_by_type(type_do):
	return list(pubmed.query("SELECT pmid FROM type WHERE type = ?;", [type_do]))

def get_nct_by_type(type_do):
	query="SELECT *  FROM acc JOIN type ON acc.pmid = type.pmid WHERE type.type=?;"
	return list(pubmed.query(query, [type_do]))

#main

ct_pmids = get_pmid_by_type("D016430")
print(f"Total clinical trials from PubMed.db (Clinical Trial D016430):{len(ct_pmids)}")
print(ct_pmids[0])

nct_matches = get_nct_by_type("D016430")
print(f"Clinical trials with NCT IDs: {len(nct_matches)}")
for row in nct_matches[:6]:
        print(row)
retracted = get_nct_by_type("D016441")
print(f" Retracted: {len(retracted)}")
for row in retracted[:6]:
        print(row)
