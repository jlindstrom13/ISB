#  Creating table to label retrospective studies (NCT | 0/1)

# Studies.txt → study_first_posted_date and 
# start_date where start_date_type = ACTUAL (instead of estimated)

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


start_dates = readTable(aact, 'studies.txt', ['nct_id','study_first_posted_date', 'study_first_posted_date_type', 'start_date', 'start_date_type'])
for index, row in start_dates.iterrows():
	nctid = row['nct_id']


print(start_dates.head())
