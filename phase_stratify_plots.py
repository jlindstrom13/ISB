# To do: create plots stratified on phase for time post time start difference

import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

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


def plot_hist(data, column, filename, bins=100, xlabel='', ylabel='', title='', log=False):
	plt.figure()
	plt.hist(data[column], bins=100, log=log)
	plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='x = 0')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(filename)
	plt.close()

df = readTable(aact, 'studies', ['nct_id','study_first_posted_date', 'study_first_posted_date_type', 'start_date', 'start_date_type', 'phase'])
for index, row in df.iterrows():
	nctid = row['nct_id']

df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['study_first_posted_date'] = pd.to_datetime(df['study_first_posted_date'], errors='coerce')

# Drop rows w/ na
new_df = df.dropna(subset=['start_date', 'study_first_posted_date', 'phase'])
new_df.loc[:,'diff_days'] = (new_df['study_first_posted_date'] - new_df['start_date']).dt.days

phase1_df = new_df[new_df['phase'] == 'PHASE1']

plot_hist(
    phase1_df,
    column='diff_days',
    filename='phase1_diff.png',
    xlabel='Posted Date - Start Date (days)',
    ylabel='Number of Trials',
    title='Phase 1 Posted and Start Date Difference'
)


# future work.... continue for other phases and get rid of extreme outliers and put in log