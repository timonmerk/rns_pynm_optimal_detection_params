import pandas as pd

df_subjects = pd.read_csv('/mnt/Nexus2/iESPnet/Data/Metadatafiles/subjects_info_zeropadall_nothalamus.csv')


RNSIDS = df_subjects.rns_deid_id

RNSIDS[18]  # Victoria  patient number 19  'PIT-RNS7525'

RNSIDS[22]  # patient number 23  'PIT-RNS9183'

RNSIDS[21] # patient number 22  'PIT-RNS8973'

RNSIDS[3] # number 4 'PIT-RNS1529'

RNSIDS[10] # number 11 'PIT-RNS2227'

RNSIDS[4] # number 5 'PIT-RNS1534'




# use 'PIT-RNS7525', (Victoria's patient list 19) had very good performance

# RNS7525	EP1255	Pitt	1	0	Nexus3/EpilepsySurgery_DataBank_2015-2019/2019/2019-05-21_EP1255/	1	2	0	neocortex	6/11/19	lesional	temporal	na	left_temporal_fcd	0	0		

# calculate features for that subject, and check which ones Victoria used


#There is surprising suuuper high performance for within subject performance.
#Idea: make figures already,

#1. Time Series Plot Seizure
#2. Feature plot for all channels


# Select 5 or 6 patients where the performance was high
# Next application: identify epochs that had certain type of modulation
# try to predict those --> needds to be a different approach

