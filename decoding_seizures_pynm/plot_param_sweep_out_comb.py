import pickle
import os
import numpy as np
import pandas as pd
import xgboost
from sklearn import model_selection, metrics, linear_model, ensemble, svm, discriminant_analysis
import seaborn as sb
from matplotlib import pyplot as plt
from scipy import stats
from py_neuromodulation import nm_plots

PATH_SWEEP = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\rns_pynm_optimal_detection_params"
PATH_SWEEP = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\rns_pynm_optimal_detection_params\RNS_PE_param_sweep_noCV"
files = [os.path.join(PATH_SWEEP, i) for i in os.listdir(PATH_SWEEP) if "_per_PE.p" in i]
df_f1_res = []

def get_f1_score(m: np.array):
    if len(m.shape) == 5:
        tp = m[:, :, :, 1, 1]
        tn = m[:, :, :, 0, 0]
        fp = m[:, :, :, 0, 1]
        fn = m[:, :, :, 1, 0]
    else:
        tp = m[1, 1]
        tn = m[0, 0]
        fp = m[0, 1]
        fn = m[1, 0]

    precision = np.nan_to_num(tp / (tp + fp))
    recall = np.nan_to_num(tp / (tp + fn))

    F1 = np.nan_to_num((2 * precision * recall) / (precision + recall))
    return F1

EVAL_ML = False


for f in files:
    sub = os.path.basename(f)[:7]
    with open(f, 'rb') as handle:
        b = pickle.load(handle)


    for PE in b.keys():
        print(f"sub: {sub} PE: {PE}")
        # calculate f1 score

        F1 = get_f1_score(b[PE]["cm_out"])
        best_feature_idx, best_time_idx, best_amplitude_idx = np.unravel_index(np.nanargmax(F1), F1.shape)
        best_F1_param_sweep = F1[best_feature_idx, best_time_idx, best_amplitude_idx]

        #best_F1_param_sweep = np.mean(b[PE]["f1_score_test"])

        # read now RNS param from pick
        df_RNS_per = pd.read_pickle("performances_RNS_bandpass_new.p")

        cm_RNS = df_RNS_per.query("sub == @sub and PE == @PE")["cm"].iloc[0]
        F1_RNS = get_f1_score(cm_RNS)

        # load now the features and calculate f1 score from those
        #PATH_FEATURES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\RNSOut_pynm_more_features"
        #folder_name = f"PIT-{sub}-{PE}-1_EOF_SZ-VK"
        #feature_name = f"PIT-{sub}-{PE}-1_EOF_SZ-VK_FEATURES.CSV"
        #PATH_FEATURE = os.path.join(PATH_FEATURES, folder_name, feature_name)
        #df_f = pd.read_csv(PATH_FEATURE)

        PATH_FEATURES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\predictions_sweep\folder_features"
        df_f = pd.read_pickle(os.path.join(PATH_FEATURES, f"{sub}_{PE}.p"))
        df_comb = []
        for seg in df_f.keys():
            df_add = df_f[seg][[c for c in df_f[seg].columns if c!="time_aligned_s"]]
            df_comb.append(np.array(df_add.mean()))
        df_feature = pd.DataFrame(data=np.vstack(df_comb), columns=df_add.columns)

        X = df_feature[[f for f in df_feature.columns if "ch" in f]]
        y = df_feature["sz"]>0

        
        if EVAL_ML is True:
            model_list = [
                linear_model.LogisticRegression(),
                discriminant_analysis.LinearDiscriminantAnalysis(),
                svm.SVC(),
                ensemble.RandomForestClassifier(n_jobs=40),
                xgboost.XGBClassifier(n_jobs=50),
            ]
            ml_list_names = ["LM", "LDA", "SVC", "RF", "XGBOOST"]
            f1_list = []
            for model in model_list:
                pr_ = model_selection.cross_val_predict(
                    estimator=model,
                    cv=model_selection.KFold(n_splits=3),
                    X=X[[f for f in X.columns if "ch" in f]],  # ch
                    y=y
                )

                f1_list.append(metrics.f1_score(y, pr_))

            list_add = [best_F1_param_sweep, F1_RNS] + f1_list
            list_names = ["pynm_param_sweep", "RNS_settings"] + ml_list_names

        else:
            list_add = [best_F1_param_sweep, F1_RNS]
            list_names = ["pynm_param_sweep", "RNS_settings"]

        for f1, type in zip(
            list_add,
            list_names
        ):
            df_f1_res.append(
                {
                    "type": type,
                    "f1" : f1,
                    "sub" : sub,
                    "PE" : PE
                }
            )

df_res = pd.DataFrame(df_f1_res)
df_res.to_csv("decoding_performances_reported.csv")

# performance report paper:
print(df_res.query("type == 'RNS_settings'").groupby("sub").mean().mean())
print(df_res.query("type == 'RNS_settings'").groupby("sub").mean().std())

print(df_res.query("type == 'pynm_param_sweep'").groupby("sub").mean().mean())
print(df_res.query("type == 'pynm_param_sweep'").groupby("sub").mean().std())

print(df_res.query("type == 'XGBOOST'").groupby("sub").mean().mean())
print(df_res.query("type == 'XGBOOST'").groupby("sub").mean().std())

print(df_res.query("type == 'LM'").groupby("sub").mean().mean())
print(df_res.query("type == 'LM'").groupby("sub").mean().std())

print(df_res.query("type == 'SVC'").groupby("sub").mean().mean())
print(df_res.query("type == 'SVC'").groupby("sub").mean().std())

# PLOT 1
ax = plot_df_subjects(
    df=df_res.query('type.str.contains("pynm_param_sweep") or type.str.contains("RNS_settings") or type.str.contains("XGB")').groupby(["sub", "type"]).mean().reset_index(),
    x_col="type",
    y_col="f1",
    figsize_tuple=(5,6)
)
ax.set_ylabel("F1 score")
ax.set_title("Seizure detection comparison")
plt.tight_layout()
plt.savefig(
    "COMP_CrossValidated_Param_Sweep.pdf",
    bbox_inches="tight",
)

# PLOT 2
ax = plot_df_subjects(
    df=df_res.query('type.str.contains("XGB") or type.str.contains("LM") or type.str.contains("SVC")').groupby(["sub", "type"]).mean().reset_index(),
    x_col="type",
    y_col="f1",
    figsize_tuple=(5,6)
)
ax.set_ylabel("F1 score")
ax.set_title("Seizure detection comparison")
plt.tight_layout()
plt.savefig(
    "COMP_ML_methods_param_sweep.pdf",
    bbox_inches="tight",
)

# PLOT 3: evaluate here the params in RNS_PE_param_sweep_noCV
ax = nm_plots.plot_df_subjects(
    df=df_res.query('type.str.contains("pynm_param_sweep") or type.str.contains("RNS_settings")').groupby(["sub", "type"]).mean().reset_index(),
    x_col="type",
    y_col="f1",
    figsize_tuple=(5,6)
)
ax.set_ylabel("F1 score")
ax.set_title("Seizure detection comparison")
plt.tight_layout()
plt.savefig(
    "NoCrossValParamSweep.pdf",
    bbox_inches="tight",
)



plt.figure(figsize=(4,3), dpi=300)
sb.barplot(
    x="sub", y="f1", hue="type", data=df_res, palette="viridis"
)
plt.xticks(rotation=90)
plt.ylabel("F1 score")
plt.legend(loc = "upper right", bbox_to_anchor=(1.04, 1))
plt.title("F1 performance comparison")
plt.tight_layout()
plt.savefig(
    "F1_performances_comp.pdf",
    bbox_inches="tight",
)
plt.show()


df_res.groupby(["type", "sub"]).mean()

sb.boxplot(
    x="type", y="f1", data=df_res
)

plt.figure(figsize=(4, 3), dpi=300)
sb.boxplot(
    x="type", y="f1", data=df_res.query('type.str.contains("py_nm")'), palette="viridis"
)
plt.show()

