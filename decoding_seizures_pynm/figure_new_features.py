import mne
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import pickle
import os
from scipy import stats

file_read = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\rns_pynm_optimal_detection_params\PIT-RNS7525_PE20190611-1_EOF_SZ-VK.EDF"

PATH_OUT = r"C:\Users\ICN_admin\Documents\Datasets\Boston Epilepsy RNS\RNSOut_pynm_more_features"
PE = "RNS7525_20190611-1_EOF_SZ-VK.p"

# plot 07:'sz_on'
ONSET = 302.304
SZ = 332.869
OFFSET = 392.424

with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
    b = pickle.load(handle)

epoch_num = "E4"

raw = mne.io.read_raw_edf(file_read)

idx_seg = np.where(np.logical_and(raw.times > ONSET, raw.times < OFFSET))[0]
idx_on = np.where(raw.times>332.9)[0][0]

time_sz = (idx_on - idx_seg[0])*0.004


cols_plt = [col for col in b[epoch_num].columns if "ch" in col]
cols_plt_sorted = [_[4:] for _ in cols_plt]
idx_sorted = np.argsort(cols_plt_sorted)
cols_sortex = np.sort(cols_plt_sorted)

plt.figure(figsize=(7,7), dpi=300)
plt.imshow(
    stats.zscore(
        b[epoch_num].iloc[:, idx_sorted].fillna(0),
        axis=0,
        nan_policy='omit'
    ).T[::-1],
    #b[epoch_num][cols_plt].T[::-1],
    aspect="auto",
    interpolation="gaussian"
)
plt.yticks(np.arange(0, len(cols_sortex), 1), cols_sortex[::-1], size=2)
plt.xlabel("Time [s]")
plt.gca().invert_yaxis()
plt.title("Computed Features")
plt.clim(-1, 1)
plt.axvline(x=time_sz, label="Seizure Onset", color="red")
plt.legend(loc = "upper right")
plt.savefig("Features_more_slim.pdf", bbox_inches="tight")

