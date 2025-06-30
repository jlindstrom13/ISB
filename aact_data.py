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

# calls the function readTable
# input aact always
# table name: 'study_references' column names: ['nct_id', etc.]
refs = readTable(aact, 'study_references', ['nct_id','pmid','reference_type'])
for index, row in refs.iterrows():
	nctid = row['nct_id']


retractions = readTable(aact, 'retractions', ['nct_id','pmid'])
for index, row in retractions.iterrows():
	nctid = row['nct_id']


print("retractions has", len(retractions), "rows")
print(retractions.head())
