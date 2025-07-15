import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import datetime

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

def using_mpl_scatter_density(fig, x, y):
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	density = ax.scatter_density(x, y, cmap=white_viridis)
	fig.colorbar(density, label='Number of points per pixel')
	ax.set_xlabel("Start Date")
	ax.set_ylabel("Study First Posted Date")
	ax.set_title("Density of Start vs. Posted Dates")
	ax.xaxis.set_major_formatter(FuncFormatter(format_date))
	ax.yaxis.set_major_formatter(FuncFormatter(format_date))
	start_ordinal = datetime.date(2000, 1, 1).toordinal()
	end_ordinal = datetime.date(2030, 1, 1).toordinal()
	ax.set_xlim(start_ordinal, end_ordinal)
	ax.set_ylim(start_ordinal, end_ordinal)


def format_date(x, _):
    try:
        return datetime.date.fromordinal(int(x)).strftime('%Y')
    except:
        return ''

df = readTable(aact, 'studies', ['nct_id','study_first_posted_date', 'start_date'])
for index, row in df.iterrows():
	nctid = row['nct_id']

df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['study_first_posted_date'] = pd.to_datetime(df['study_first_posted_date'], errors='coerce')

# Drop rows w/ na
heatdf = df.dropna(subset=['start_date', 'study_first_posted_date'])

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (0.05, '#440053'),
    (0.15, '#404388'),
    (0.3, '#2a788e'),
    (0.5, '#21a784'),
    (0.7, '#78d151'),
    (1, '#fde624'),
], N=256)

x = heatdf['start_date'].map(pd.Timestamp.toordinal)
y = heatdf['study_first_posted_date'].map(pd.Timestamp.toordinal)

fig = plt.figure()
using_mpl_scatter_density(fig, x, y)
plt.savefig("density_start_vs_posted.png", dpi=600)




