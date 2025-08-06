# Using clinical trial zips, find clinical trials that are retracted 

import sqlite3
import zipfile
import pandas as pd

aact = '20250626'

def readTable(aact, tablename, usefields):
	usefields.append('nct_id')
	file = '/users/jlindstr/clinicaltrials/zips/'+tablename+'.txt'
	try:
		table = pd.read_csv(file, sep='|', usecols=usefields)
		#print('read', tablename, 'from file', flush=True)
	except:
		zipfilename = f'/users/jlindstr/clinicaltrials/zips/{aact}.zip' #'zips/'+aact+'.zip'
		zf = zipfile.ZipFile(zipfilename)
		#print('reading', tablename, 'from zip', flush=True)
		with zf.open(tablename+'.txt') as f:
			table = pd.read_csv(f, sep='|', usecols=usefields)
	return table

df = readTable(aact, 'retractions', ['nct_id','pmid'])
for index, row in df.iterrows():
	nctid = row['nct_id']

print(df.head())

num_retracted_trials = df['nct_id'].nunique()
print(f'Total unique retracted trials: {num_retracted_trials}')

unique_ncts = df['nct_id'].unique()

unique_ncts_series = pd.Series(unique_ncts)

# Save to a pickle file
unique_ncts_series.to_pickle('retracted_ct_ncts.pkl')