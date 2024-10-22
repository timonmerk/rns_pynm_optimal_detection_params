import os
import pickle

PATH_SWEEP = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\rns_pynm_optimal_detection_params\RNS_PE_param_sweep_noCV"
files = [os.path.join(PATH_SWEEP, i) for i in os.listdir(PATH_SWEEP) if "_per_PE.p" in i]


patient_recordings = []
patient_recording_duration = []
dur_on = []

for f in files:
    with open(f, 'rb') as handle:
        b = pickle.load(handle)
    recs = []
    durs = []
    dur_on = []
    for PE in b.keys():
        recs.append(len(list(b.keys())))
        durs.append(b[PE])