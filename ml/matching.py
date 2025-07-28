# Matching cases and controls on start date
# https://pypi.org/project/epydemiology/0.1.4/


import pandas as pd
import epydemiology as epy

cases_df = pd.read_pickle("cases_df.pkl")
controls_df = pd.read_pickle("controls_df.pkl")

matched_df = epy.phjSelectCaseControlDataset(
    phjCasesDF = cases_df,
    phjPotentialControlsDF = controls_df,
    phjUniqueIdentifierVarName = 'nct_id',
    phjMatchingVariablesList = ['studies:study_first_posted_date_type:ACTUAL'],
    phjControlsPerCaseInt = 1,
    phjPrintResults = False)

print(matched_df.head(10))
