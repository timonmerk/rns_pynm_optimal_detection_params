# Idea here: load a feature dataframe and plot pyneuromodulation features
# over segements

import mne
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import os
import pickle
import pandas as pd

from scipy import stats

PATH_OUT = "/mnt/4TB/timon/RNSOut_pynm_good"


list_PE = [
    "RNS7525_20190611-1_EOF_SZ-VK.p",
    "RNS7525_20190625-1_EOF_SZ-VK.p",
    "RNS7525_20190806-1_EOF_SZ-VK.p"
]

out_ = []

for PE in list_PE:
    with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
        b = pickle.load(handle)

    # plot_segment(b, epoch_num="E5")

    # first idea: just average all features for each segments, don't normalize

    for epoch_str in b.keys():
        out_.append(b[epoch_str].mean())

df_out = pd.DataFrame(out_)


# calculate mean features across channels
df_plt_mean = pd.DataFrame()
df_features = df_out[[col for col in df_out.columns if "ch" in col]]

df_features = stats.zscore(df_features)
df_plt_mean["FFT theta"] = df_features[[f for f in df_features.columns if "theta" in f]].mean(axis=1)
df_plt_mean["FFT alpha"] = df_features[[f for f in df_features.columns if "alpha" in f]].mean(axis=1)
df_plt_mean["FFT low beta"] = df_features[[f for f in df_features.columns if "low beta" in f]].mean(axis=1)
df_plt_mean["FFT high beta"] = df_features[[f for f in df_features.columns if "high beta" in f]].mean(axis=1)
df_plt_mean["FFT low gamma"] = df_features[[f for f in df_features.columns if "low gamma" in f]].mean(axis=1)
df_plt_mean["FFT Line-Length"] = df_features[[f for f in df_features.columns if "Line" in f]].mean(axis=1)
df_plt_mean["seizure"] = df_out["sz"] > 0
df_melt = pd.melt(df_plt_mean, id_vars = "seizure")

plt.figure(figsize=(6, 4), dpi=300)

sb.boxplot(
    df_melt,
    x="variable",
    y="value",
    hue="seizure",
    palette="viridis",
    showmeans=True,
    meanprops={"markeredgecolor":"black", "markerfacecolor":"white"}
)

plt.xticks(rotation = 90)
plt.ylabel("Normalized Feature Value")
plt.legend(loc="upper right", title="Seizure")
plt.title("Examplary subject mean feature values")

plt.savefig(
    "BoxPlot_Mean_Features.pdf",
    bbox_inches="tight",
)

###################################################
# OLD:



df_plt = stats.zscore(df_features)
df_plt["seizure"] = df_out["sz"] > 0
df_melt = pd.melt(df_plt, id_vars = "seizure")
plt.figure(figsize=(6, 4), dpi=300)
sb.boxplot(df_melt, x="variable", y="value", hue="seizure", palette="viridis")


plt.imshow(df_plt.T.iloc[:, :100], aspect='auto')


df_melt = pd.melt(df_out[[col for col in df_out.columns if "ch" in col or col == "sz"]], id_vars = "sz")
df_melt["sz"] = df_melt["sz"] >0

plt.figure(figsize=(6, 4), dpi=300)
sb.boxplot(df_melt, x="variable", y="value", hue="sz", palette="viridis")


# But try out feature time series plot as well

