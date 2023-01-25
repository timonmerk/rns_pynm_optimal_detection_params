import mne
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import pickle
import os
from scipy import stats

""" 1. Figure: 
Raw Time Series Seizure Across all channels

2. Figure: 
Feature Plot Below Sz ON / OFF plot for certain amount of duration

3. Figure Performance:
Include more patients
 """

PLT_TIME_SERIES = True

PATH_OUT = "/mnt/4TB/timon/RNSOut_pynm_good"
file_read = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS7525/iEEG/PIT-RNS7525_PE20190611-1_EOF_SZ-VK.EDF"

if PLT_TIME_SERIES is True:
    raw = mne.io.read_raw_edf(file_read)

    #raw.plot(block=True)

    dat = raw.get_data()

    # plot 07:'sz_on'
    ONSET = 302.304
    SZ = 332.869
    OFFSET = 392.424

    idx_seg = np.where(np.logical_and(raw.times > ONSET, raw.times < OFFSET))[0]
    idx_on = np.where(raw.times>332.9)[0][0]

    times = np.arange(0, idx_seg.shape[0]*0.004, 0.004)
    dat_plt = dat[:, idx_seg]
    time_sz = (idx_on - idx_seg[0])*0.004

    y_sep = 750 
    hue_colors = sb.color_palette("viridis", 5)

    plt.figure(figsize=(6,3), dpi=300)
    for idx in [0, 1, 2, 3]:
        plt.plot(times, dat_plt[idx] + y_sep*idx, linewidth=0.2, color="black") # label=f"channel {idx}"

    plt.axvline(x=time_sz, label="Seizure Onset", color=hue_colors[0])

    plt.legend(loc = "upper right")
    plt.xlabel("Time [s]")
    plt.yticks(np.arange(0, 750*4, 750), [f"Channel {idx}" for idx in range(4)])
    plt.title("Example Seizure Time Series")
    plt.tight_layout()
    plt.savefig(
        "TimeSeriesFigure.pdf",
        bbox_inches="tight",
    )


# Get here the py_nm features for that segment
PE = "RNS7525_20190611-1_EOF_SZ-VK.p"

with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
    b = pickle.load(handle)

epoch_num = "E4"

cols_plt = [col for col in b[epoch_num].columns if "ch" in col]


plt.imshow(stats.zscore(b[epoch_num][cols_plt], axis=0).T, aspect="auto")
#plt.clim(-1, 1)
plt.yticks(np.arange(0, len(cols_plt), 1), cols_plt)
plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()




# PLT COMBINED
plt.figure(figsize=(12,6), dpi=300)

plt.subplot(121)
for idx in [0, 1, 2, 3]:
    plt.plot(times, dat_plt[idx] + y_sep*idx, linewidth=0.075, color="black") # label=f"channel {idx}"

plt.axvline(x=time_sz, label="Seizure Onset", color="red")

plt.legend(loc = "upper right")
plt.xlabel("Time [s]")
plt.yticks(np.arange(0, 750*4, 750), [f"Channel {idx+1}" for idx in range(4)])
plt.title("Example Seizure Time Series")

plt.subplot(122)
plt.imshow(stats.zscore(b[epoch_num][cols_plt], axis=0).T[::-1], aspect="auto")
#plt.clim(-1, 1)
plt.yticks(np.arange(0, len(cols_plt), 1), cols_plt)
plt.xlabel("Time [s]")
plt.gca().invert_yaxis()
plt.title("Computed Features")

plt.axvline(x=time_sz, label="Seizure Onset", color="red")
plt.legend(loc = "upper right")

plt.tight_layout()

plt.savefig(
    "TimeSeriesFigure_combined_features.pdf",
    bbox_inches="tight",
)

plt.show()