import pandas as pd
import numpy as np
import os

df = pd.read_pickle("performances_RNS_bandpass_new.p")

# check only PE's that have more than 5 seizures

PE_run = [i for i in range(df.shape[0]) if df["cm"].iloc[i][1, :].sum()>3]
df_run = df.iloc[PE_run]

# 9 subjects and 23 PE's
edf_paths = []
for idx, row in df_run.iterrows():
    sub = row["sub"]
    PE = row["PE"]
    PATH_EDF = f"X:\\RNS_DataBank\\PITT\\PIT-{sub}\\iEEG\\PIT-{sub}_PE{PE}-1_EOF_SZ-VK.EDF"
    if os.path.exists(PATH_EDF) is True:
        edf_paths.append(PATH_EDF)

df_run["edf_file"] = edf_paths
df_run.to_csv("edf_files_to_run.csv")

# get PE names
print(df)
