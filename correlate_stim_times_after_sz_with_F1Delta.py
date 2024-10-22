import pandas as pd

import seaborn as sb

import numpy as np

from py_neuromodulation import nm_stats

from matplotlib import pyplot as plt

df_stim_times = pd.read_csv("stim_times_all.csv")

df_first_stim = df_stim_times.query("first_stim == True")

df_outcome = pd.read_csv("outcome_table.csv")

# plot
#sb.histplot(x="num_detect_after_stim", data=df_first_stim, bins=np.arange(30), hue="sub", multiple="stack")

df_first_stim = df_first_stim.query("time_stim_after_szon >-90 and time_stim_after_szon<90")

df_mean_detect_after_stim = df_first_stim.groupby(["sub", "PE"])[["num_detect_after_stim", "time_stim_after_szon"]].mean().reset_index()

# load additionally the F1 performances
df_per = pd.read_csv("decoding_performances_reported.csv")

rows = []
for sub in df_per["sub"].unique():
    for PE in df_per.query("sub == @sub")["PE"].unique():
        F1_RNS = df_per.query("sub == @sub and PE == @PE and type=='RNS_settings'")["f1"].iloc[0]
        F1_sweep_RNS = df_per.query("sub == @sub and PE == @PE and type=='pynm_param_sweep'")["f1"].iloc[0]
        diff_F1 = F1_sweep_RNS - F1_RNS
        # ok, now I have to check what the corresponding mean_detections were
        df_detect_correct_sub = df_mean_detect_after_stim.query("sub == @sub", engine="python")
        PE_str = str(PE)
        df_correct_PE = df_detect_correct_sub.query("PE.str.contains(@PE_str)")

        if df_correct_PE.shape[0] == 0:
            continue
        else:
            mean_detect_after_stim = df_correct_PE.iloc[0]["num_detect_after_stim"]
            time_stim_after_szon = df_correct_PE.iloc[0]["time_stim_after_szon"]

            int_PE = int(PE)
            sz_red = df_outcome.query("sub.str.contains(@sub) and PE == @int_PE").iloc[0]["Sz_red"]
            rows.append({
                "sub" : sub,
                "PE" : PE,
                "F1_RNS" : F1_RNS,
                "F1_sweep_RNS" : F1_sweep_RNS,
                "diff_F1" : diff_F1,
                "mean_detect_after_stim" : mean_detect_after_stim,
                "time_stim_after_szon" : time_stim_after_szon,
                "sz_red" : sz_red
            })

df_plt = pd.DataFrame(rows)


plt.figure(figsize=(4,3), dpi=300)
sb.regplot(x="mean_detect_after_stim", y="diff_F1", data=df_plt[df_plt["mean_detect_after_stim"]<15])
plt.title("Do better parametrization predict\nif seizures are stopped?")
plt.tight_layout()

plt.figure(figsize=(4,3), dpi=300)
sb.regplot(x="sz_red", y="diff_F1", data=df_plt)
plt.title("Do better parametrization predict\n seizure reduction?")
plt.tight_layout()

# now with F1 directly:
plt.figure(figsize=(4,3), dpi=300)
sb.regplot(x="mean_detect_after_stim", y="F1_RNS", data=df_plt)
plt.title("Do better seizure prediction performance\n predict detections after sz_on?")
plt.tight_layout()

# now with F1 directly:
plt.figure(figsize=(4,3), dpi=300)
sb.regplot(x="mean_detect_after_stim", y="F1_RNS", data=df_plt[df_plt["mean_detect_after_stim"]<15])
plt.title("Do better seizure prediction performance\n predict detections after sz_on?")
plt.tight_layout()

# 
plt.figure(figsize=(4,3), dpi=300)
sb.boxplot(y="F1_RNS", data=df_plt, x="responder", palette="viridis")
sb.stripplot(y="F1_RNS", data=df_plt, x="responder")
plt.title("Responders=Sz_red>0.25")
plt.tight_layout()

df_plt["responder"] = df_plt["sz_red"]>0.25
plt.figure(figsize=(4,3), dpi=300)
sb.boxplot(y="diff_F1", data=df_plt, x="responder", palette="viridis")
sb.stripplot(y="diff_F1", data=df_plt, x="responder")
plt.title("Responders=Sz_red>0.25")
plt.tight_layout()


df_plt["good_parametrized"] =  df_plt["diff_F1"] > df_plt["diff_F1"].median()
plt.figure(figsize=(4,3), dpi=300)
sb.boxplot(y="sz_red", data=df_plt, x="good_parametrized", palette="viridis")
sb.stripplot(y="sz_red", data=df_plt, x="good_parametrized")
plt.title("Good parametrized>Median delta F1")
plt.tight_layout()

# time

plt.figure(figsize=(4,3), dpi=300)
sb.regplot(x="time_stim_after_szon", y="diff_F1", data=df_plt[df_plt["mean_detect_after_stim"]<15])
plt.title("Better parametrization leads to\n earlier stimulation")
plt.tight_layout()

plt.figure(figsize=(4,3), dpi=300)
sb.regplot(x="time_stim_after_szon", y="F1_RNS", data=df_plt[df_plt["mean_detect_after_stim"]<15])
plt.title("Better parametrization leads to\n earlier stimulation")
plt.tight_layout()

# 1. run obvious correltation:
# Sz_red with delta F1 

plt.figure()
sb.regplot(x="sz_red", y="diff_F1", data=df_plt.groupby("sub").mean())

plt.figure()
sb.regplot(x="sz_red", y="F1_RNS", data=df_plt.groupby("sub").mean())


plt.figure()
sb.regplot(x="sz_red", y="F1_RNS", data=df_plt)

plt.figure()
sb.regplot(x="sz_red", y="mean_detect_after_stim", data=df_plt)

plt.figure()
sb.regplot(x="sz_red", y="time_stim_after_szon", data=df_plt)

plt.figure()
sb.regplot(x="mean_detect_after_stim", y="diff_F1", data=df_plt)

sb.regplot(x="time_stim_after_szon", y="diff_F1", data=df_plt)

sb.regplot(x="mean_detect_after_stim", y="diff_F1", data=df_plt)

sb.regplot(x="mean_detect_after_stim", y="F1_sweep_RNS", data=df_plt)

sb.regplot(x="mean_detect_after_stim", y="F1_RNS", data=df_plt[df_plt["mean_detect_after_stim"]  < 15])