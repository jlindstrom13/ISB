# Reads in .txt file of matched cases and controls, saves to pkl
import pandas as pd
import pickle as pkl

#file_path = "/15TB_2/gglusman/clinicaltrials/matched_controls3.txt" Old, pre adding in more retracted trials to training
file_path = "matched_controls3.txt"

df = pd.read_csv(file_path, sep="\t", dtype=str)

df.to_pickle("matched_cases_controls.pkl")

print(f"Number of matched controls: {len(df)}") #13,162
