# examining csv file....

import pandas as pd

# Load the CSV file
df = pd.read_csv("retraction-watch-data/retraction-watch-data.csv")

# View the first few rows
print(df.head())