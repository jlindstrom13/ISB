import pandas as pd
import pickle as pkl

file_path = "/15TB_2/gglusman/clinicaltrials/matched_controls2.txt"

df = pd.read_csv(file_path, sep="\t", dtype=str)

df.to_pickle("matched_cases_controls.pkl")
