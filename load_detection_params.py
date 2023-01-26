import pandas as pd
import os
import mne
from datetime import datetime
import pickle
import numpy as np
from sklearn import metrics
from joblib import Memory
from joblib import Parallel, delayed
import glob
from itertools import product

def get_detection_pattern(df_settings, sub='RNS1529', str_PE='20151215'):
    df_settings_sub = df_settings.query("rns_id == @sub")
    df_sett_sub_PE = df_settings_sub.loc[df_settings_sub["timestamp"].apply(pd.to_datetime).dt.strftime('%Y%m%d') == str_PE]

    pattern_list = []

    if df_sett_sub_PE.shape[0] != 0:
        
        for idx, row in df_sett_sub_PE.query("cm_min_frequency_shape == 'Sinusoid'").iterrows():
            str_detector = None
            if row["pattern"] == "Pattern A":
                str_detector = "A"
            else:
                str_detector = "B"
            if row["detector"] == "First Detector":
                str_detect = f"{str_detector}1"
            else:
                str_detect = f"{str_detector}2"    
            if str_detect not in pattern_list:
                pattern_list.append(str_detect)
    return pattern_list

def get_predictions_RNS(PATH_OUT, df_RNS_annot,PE, PATH_EDF, pattern_used):

    l_pred = []
    l_sez = []

    raw = mne.io.read_raw_edf(PATH_EDF)

    data = raw.get_data()

    for idx_segment, segment in enumerate(raw.annotations):

        segment_type = segment["description"]
        if segment_type != "eof":
            continue

        annot = df_RNS_annot[df_RNS_annot.rns_deid_id == sub]

        PE__ = PE[8:PE.find("_EOF")]
        PE_annot_map = f"PIT-{sub}_PE{PE__}"
        eof = segment["onset"]

        annot_pe = annot[annot.compiled_file_nm.str.contains(PE_annot_map)]
        if annot_pe.shape[0] == 0:
            annot_pe = annot[annot.compiled_file_nm.str.contains(PE_annot_map[:-2])]
            if  annot_pe.shape[0] == 0:
                annot_pe = annot[annot.compiled_file_nm.str.contains(PE_annot_map[4:-2])]
                if annot_pe.shape[0] == 0:
                    annot_pe = annot[annot.compiled_file_nm.str.contains(PE_annot_map[4:])]
        if annot_pe.shape[0] == 0:
            detected = None

        query_segment = annot_pe.query("stop_secs_dec > @eof-1 and stop_secs_dec < @eof+1")

        detected = False
        sz = False
        if query_segment.shape[0]:
            for episode in pattern_used: #
                if episode in query_segment["name"].unique() or "Therapy Delivered" in query_segment["name"].unique():
                    detected = True
                    if "sz" in raw.annotations[idx_segment-1]["description"] and idx_segment != 0:
                        sz = True
        l_pred.append(detected)
        l_sez.append(sz)

    return l_pred, l_sez

def get_per_PE(run_):
    sub = run_[0]
    PE = run_[1]
    PATH_EDF = f"X:\\RNS_DataBank\\PITT\\PIT-{sub}\\iEEG\\PIT-{sub}_PE{PE}-1_EOF_SZ-VK.EDF"
    try:
        pattern_used = get_detection_pattern(df_settings, sub=sub, str_PE=PE)
    except Exception:
        return  # might be that this detection does not exist

    if len(pattern_used) > 0:
        l_pred, l_sez = get_predictions_RNS(PATH_OUT, df_RNS_annot,PE, PATH_EDF, pattern_used)
        # check here performances
        cm_ = metrics.confusion_matrix(
            l_sez,
            l_pred
        )
        if cm_.shape[0] != 1:   # check for more than one class (no seizures) present
            return {
                    "sub" : sub,
                    "PE" : PE,
                    "cm" : cm_,
                    "cm_normed" : metrics.confusion_matrix(
                        l_sez,
                        l_pred,
                        normalize="true"
                    )
                }


PATH_OUT = r"X:\Users\timon\RNSOut_pynm_good"
df_RNS_annot = pd.read_csv(r"X:\Users\timon\RNS_Detect_Annotations\Pitt_Ecogs_events_updated_01212022.csv")

df_settings = pd.read_csv("iESPNet_bandpass_parameters_PythonMerge.csv")

subjects_all = df_settings["rns_id"].unique()

PEs_paths =  glob.glob(f"X:\\RNS_DataBank\\PITT\\PIT-*\\iEEG\\PIT-*_PE*-1_EOF_SZ-VK.EDF")
l_run = []

for sub in subjects_all:

    PEs_str = np.unique(
        [
            os.path.basename(f)[os.path.basename(f).find("_PE")+3: os.path.basename(f).find("-1_EOF")]
            for f in PEs_paths if sub in f
        ]
    )
    #get_per_PE((sub, PEs_str[1]))
    for p in PEs_str:
        l_run.append((sub, p))

location = './cachedir'
memory = Memory(location, verbose=0)
costly_compute_cached = memory.cache(get_per_PE)
def data_processing_mean_using_cache(run_):
    """Compute the mean of a column."""
    return costly_compute_cached(run_)
results = Parallel(n_jobs=30)(
    delayed(data_processing_mean_using_cache)(run_)
    for run_ in l_run
)
memory.clear(warn=False)

df = pd.DataFrame([r for r in results if r is not None])
df.to_pickle("performances_RNS_bandpasse_new.p")
# store now confusion matrices for each PE

from sklearn import metrics
from matplotlib import pyplot as plt

plt.figure()
metrics.ConfusionMatrixDisplay(df["cm"].sum())
plt.show()

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=df["cm"].sum(),
    display_labels=["NoSz", "Sz"]
)
disp.plot()
disp.ax_.set_title("First Bandpass Detector")
