#  Creating table to label retrospective studies (NCT | 0/1)

# Studies.txt → study_first_posted_date and 
# start_date where start_date_type = ACTUAL (instead of estimated)

import zipfile
import pandas as pd
import matplotlib.pyplot as plt


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


df = readTable(aact, 'studies', ['nct_id','study_first_posted_date', 'study_first_posted_date_type', 'start_date', 'start_date_type'])
for index, row in df.iterrows():
	nctid = row['nct_id']

print(df.head())

type= df['study_first_posted_date_type'].value_counts()
print(f'number of counts for this column{type}')

# so 2/5 of these are estimated first posted dates...

# looking at which are missing start_date or study_first_posted
missing_start = df['start_date'].isna().sum()
missing_posted = df['study_first_posted_date'].isna().sum()

print("Trials missing start_date:", missing_start)
print("Trials missing study_first_posted_date:", missing_posted)

df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['study_first_posted_date'] = pd.to_datetime(df['study_first_posted_date'], errors='coerce')

# Drop rows w/ na
new_df = df.dropna(subset=['start_date', 'study_first_posted_date'])

new_df.plot.scatter(
	x='start_date', 
	y='study_first_posted_date',
	alpha=0.3, 
	title='Start vs. Posted Date')
plt.plot(new_df['start_date'], new_df['start_date'], 'r--', label='start = posted')

plt.savefig("dates_plot.png")

latest_start = new_df['start_date'].max()
print("Latest start date:", latest_start)