import py_neuromodulation as nm
#from py_neuromodulation import (
#    nm_analysis,
#    nm_plots,
#    nm_stats,
#)
import time
from scipy import stats
import pickle
import os
import pandas as pd
import seaborn as sb
import mne
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics, linear_model


PATH_OUT = r"X:\Users\timon\RNSOut_pynm_good"
df_RNS_annot = pd.read_csv(r"X:\Users\timon\RNS_Detect_Annotations\Pitt_Ecogs_events_updated_01212022.csv")
PATH_EDF = r"X:\RNS_DataBank\PITT\PIT-RNS1529\iEEG\PIT-RNS1529_PE20151215-1_EOF_SZ-VK.EDF"
PATH_EDF = r"X:\RNS_DataBank\PITT\PIT-RNS1529\iEEG\PIT-RNS1529_PE20151118-1_EOF_SZ-VK.EDF"
#PATH_EDF = r"X:\RNS_DataBank\PITT\PIT-RNS1529\iEEG\PIT-RNS1529_PE20160120-1_EOF_SZ-VK.EDF"
#PATH_EDF = r"X:\RNS_DataBank\PITT\PIT-RNS1529\iEEG\PIT-RNS1529_PE20160608-1_EOF_SZ-VK.EDF"



#THIS ONE FOR FIGURE
#PATH_EDF = r"X:\RNS_DataBank\PITT\PIT-RNS1529\iEEG\PIT-RNS1529_PE20160811-1_EOF_SZ-VK.EDF"  # USE  E219

sub = 'RNS1529'

list_PE = [f for f in os.listdir(os.path.join(PATH_OUT)) if sub in f and ".p" in f]

PE = list_PE[5]

with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
    b = pickle.load(handle)

raw = mne.io.read_raw_edf(PATH_EDF)
data = raw.get_data()

for segment_nr in b.keys():
    #segment_nr = "E219" 
    annot = df_RNS_annot[df_RNS_annot.rns_deid_id == sub]

    PE__ = PE[8:PE.find("_EOF")]
    PE_annot_map = f"PIT-{sub}_PE{PE__}"
    eof = b[segment_nr]["time"].iloc[-1]/1000

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
    if query_segment.shape[0]:
        for episode in ['Long Episode']: #['A1', 'A2', 'B1', 'B2', "Long Episode"]: #
            if episode in query_segment["name"].unique():
                detected = True
                print(b[segment_nr]["sz"].sum())
                if b[segment_nr]["sz"].sum() == 0:
                    print("here")
    # append detected
    # NOW: add the detection time instead

# is there a LE that was not a seizure?

#PE2: 2462


msk_seg = np.logical_and(
    raw.times > query_segment.iloc[0]["start_secs_dec"],
    raw.times < query_segment.iloc[0]["stop_secs_dec"]
)

stims = np.array(query_segment.query("name == 'Therapy Delivered'")["start_time"])

plt.figure(figsize=(10, 5), dpi=300)
offset = 0.001
for i in range(4):
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
#for stim in stims:

dets = np.array(query_segment.query("name == 'A1' or name == 'A2' or name == 'B1' or name == 'B2'")["start_time"])
dets_duration = np.array(query_segment.query("name == 'A1' or name == 'A2' or name == 'B1' or name == 'B2'")["duration"])

y_start = 0.0035
delta_y = 0.00035
delta_x = 5
for idx, det in enumerate(dets):
    plt.annotate(
        "", xy=(det, y_start), xytext=(det-delta_x, y_start+delta_y),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"),
    )
    if idx == 0:
        label = "Detection"
    else:
        label = None
    plt.axvspan(det, det-dets_duration[idx], alpha=0.5, color="blue", label=label)

y_start = 0.0035
delta_x = 5
delta_y = 0.0005

for idx, stim in enumerate(stims):
    plt.annotate(
        "", xy=(stim, y_start), xytext=(stim-delta_x, y_start+delta_y),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    )
    if idx == 0:
        label = "Stimulation"
    else:
        label = None
    plt.axvspan(stim, stim+1, alpha=0.5, color='red', label=label)

plt.annotate(
    "Therapy Inhibited Max Therapy Limit",
    xy=(48.7, y_start), xytext=(48.7-22, y_start+delta_y*1.5),
    arrowprops=dict(arrowstyle="->", lw=2, color="green"),
)

plt.annotate(
    "Magnet Swipe",
    xy=(59.920, y_start), xytext=(59.920-10, y_start+delta_y),
    arrowprops=dict(arrowstyle="->", lw=2, color="green"),
)

plt.xlim(0, 90)
#plt.ylim(0, 0.001*4)
plt.yticks(np.arange(0, offset*4, offset), [f"Channel {idx+1}" for idx in range(4)])
plt.legend(bbox_to_anchor=(1.04, 1))


plt.tight_layout()
plt.xlabel("Time [s]")
plt.title("Exemplary Stimulation and Detection Events\n\n\n\n")

plt.savefig(
    "ExeplaryStimDetect_InlcludeMagnet.pdf",
    bbox_inches="tight",
)






# Now without Any Detections

plt.figure(figsize=(10, 5), dpi=300)
offset = 0.001
for i in range(4):
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
    plt.plot(raw.times[msk_seg]-raw.times[msk_seg][i], i*offset + data[:, msk_seg][i, :], color='black', linewidth=0.1)
#for stim in stims:

plt.xlim(0, 90)
#plt.ylim(0, 0.001*4)
plt.yticks(np.arange(0, offset*4, offset), [f"Channel {idx+1}" for idx in range(4)])


plt.tight_layout()
plt.xlabel("Time [s]")
plt.title("Optimal Setting Detections")

plt.savefig(
    "ExeplaryStimDetect_InlcludeMagnet_withoutDetect.pdf",
    bbox_inches="tight",
)





# check for feature distributions:
    ch_features = [
        "ch1_fft_theta",
        "ch1_fft_alpha",
        "ch1_fft_low beta",
        "ch1_fft_high beta",
        "ch1_fft_low gamma"
    ]

    l_df = []
    for PE in list_PE:
        df_PE = pd.DataFrame()
        for idx, ch_feat in enumerate(ch_features):
            with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
                b = pickle.load(handle)
                feature_name = ch_feat # 'ch1_fft_theta' #"ch1_fft_high beta"
                f_arr = np.concatenate([np.array(b[seg][feature_name]) for seg in b.keys()])
            df_PE[ch_feat] = f_arr
        df_PE["PE"] = PE[8:16]
        l_df.append(df_PE.reset_index())
    df_plt = pd.concat(l_df).reset_index()
    df_plt["PE"] = df_plt["PE"].astype(str)

    plt.figure()
    for idx, ch_name in enumerate(ch_features):
        plt.subplot(5, 1, idx+1)
        sb.histplot(
            x=ch_name,
            #y=ch_name,
            hue="PE",
            data=df_plt.query('PE == "20161220" or PE == "20171115"'),
            stat="percent",
            palette="viridis",
            common_norm=False,
            kde=True,
            pthresh=.05, pmax=.9,
            #ax=axes[idx]
            #multiple="fill",#"dodge",
            #bins=50,
            #ax=axes[idx]
        )
        plt.yticks([], [])
        plt.xticks([], [])
        plt.xlabel("")
        if idx != 0:
            plt.legend('',frameon=False)
        plt.title(ch_name)

    plt.savefig(f"hist_features_{sub}.pdf", bbox_inches="tight",)
