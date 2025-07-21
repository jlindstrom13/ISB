# examining csv file....

import pandas as pd

# Load the CSV file
df = pd.read_csv("retraction_watch.csv")

# View the first few rows
print(df.head())