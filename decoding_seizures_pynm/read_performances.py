import pandas as pd
import os

df = pd.read_pickle(os.path.dirname(__file__) + '\\..\\performances_RNS_bandpass_new.p')

print(df)


df_settings = pd.read_csv(os.path.dirname(__file__) + "\\..\\iESPNet_bandpass_parameters_PythonMerge.csv")