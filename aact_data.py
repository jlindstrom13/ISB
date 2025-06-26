aact = '20250626'

def readTable(aact, tablename, usefields):
	usefields.append('nct_id')
	file = '/jlindstr/clinicaltrials/zips/'+tablename+'.txt'
	try:
		table = pd.read_csv(file, sep='|', usecols=usefields)
		#print('read', tablename, 'from file', flush=True)
	except:
		zipfilename = 'zips/'+aact+'.zip'
		zf = zipfile.ZipFile(zipfilename)
		#print('reading', tablename, 'from zip', flush=True)
		with zf.open(tablename+'.txt') as f:
			table = pd.read_csv(f, sep='|', usecols=usefields)
	return table

refs = readTable(aact, 'study_references', fields['nct_id','pmid','reference_type'])
for index, row in refs.iterrows():
	nctid = row['nct_id']
