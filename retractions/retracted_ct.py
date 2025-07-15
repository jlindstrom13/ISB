# -*- coding: utf-8 -*-
# Using PubMed.db and clinical trial zips, find clinical trials that are retracted 

import sqlite3

db_path = "/ssd/sqlite/PubMed.db" 
zip_dir = "/clinicaltrials/zips"

connection = sqlite3.connect(db_path)
cursor= connection.cursor()

cursor.execute("SELECT DISTINCT pmid FROM type WHERE type = 'D016441';")
retracted_pmids = set(str(row[0]) for row in cursor.fetchall())
print(f"Found {len(retracted_pmids)} retracted PMIDs from PubMed.")
for pmid in list(retracted_pmids)[:3]:
    print(pmid)


