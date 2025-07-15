#  Created table to label retrospective studies (NCT | 0/1)
# Plotted start vs posted date, and difference between

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

plt.figure()
new_df.plot.scatter(
	x='start_date', 
	y='study_first_posted_date',
	alpha=0.02, 
	s=8,
	title='Start vs. Posted Date')
plt.plot(new_df['start_date'], new_df['start_date'], 'r--', label='start = posted')

plt.savefig("dates_plot.png", dpi=600)

latest_start = new_df['start_date'].max()
print("Latest start date:", latest_start)

# zoomed in plot
new_df.plot.scatter(
	x='start_date', 
	y='study_first_posted_date',
	alpha=0.02, 
	s=8,
	title='Start vs. Posted Date')
# adds line
plt.plot(new_df['start_date'], new_df['start_date'], 'r--', label='start = posted')
# set axis
plt.xlim(pd.to_datetime('1980-01-01'), pd.to_datetime('2035-01-01'))
plt.ylim(pd.to_datetime('1980-01-01'), pd.to_datetime('2035-01-01'))

plt.savefig("zoomed_dates.png", dpi=600)
plt.close()

new_df['diff_days'] = (new_df['study_first_posted_date'] - new_df['start_date']).dt.days

plot_hist(
    new_df,
    column='diff_days',
    filename='diff_start_posted.png',
    xlabel='Posted Date - Start Date (days)',
    ylabel='Number of Trials',
    title='Posted and Start Date Difference'
)

short_df = new_df[(new_df['diff_days'] > -500) & (new_df['diff_days'] < 500)]
plot_hist(
    short_df,
    column='diff_days',
    filename='diff_start_post_no_outlier.png',
    xlabel='Posted Date - Start Date (days)',
    ylabel='Number of Trials',
    title='Posted and Start Date Difference No Outliers'
)

# log scale
plot_hist(
    short_df,
    column='diff_days',
    filename='log_diff_start_post_log.png',
    xlabel='Posted Date - Start Date (days)',
    ylabel='Log Number of Trials',
    title='Posted and Start Date Difference +/- 500 (Log Scale)',
    log=True
)

df_120 = new_df[(new_df['diff_days'] > -120) & (new_df['diff_days'] < 120)]
plot_hist(
    df_120,
    column='diff_days',
    filename='log_diff_start_post_120.png',
    xlabel='Posted Date - Start Date (days)',
    ylabel='Log Number of Trials',
    title='Posted and Start Date Difference +/- 120 (Log Scale)',
    log=True
)


df_60 = new_df[(new_df['diff_days'] > -60) & (new_df['diff_days'] < 60)]

plot_hist(
    df_60,
    column='diff_days',
    filename='log_diff_start_post_60.png',
    xlabel='Posted Date - Start Date (days)',
    ylabel='Log Number of Trials',
    title='Posted and Start Date Difference +/- 60 (Log Scale)',
    log=True
)
