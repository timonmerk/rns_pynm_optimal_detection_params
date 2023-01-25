import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_plots,
    nm_stats,
)
from scipy import stats
import pickle
import os
import pandas as pd
import seaborn as sb

import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics, linear_model
import xgboost

PATH_OUT = "/mnt/4TB/timon/RNSOut_pynm_good"
sub_name = "PIT-RNS0427"

# read pickle file with whole PE
# extract features for each segment

def plot_segment(b, epoch_num: str = "E1", norm=True):
    if norm == True:
        plt.imshow(stats.zscore(b[epoch_num], axis=0).T, aspect="auto")
    else:
        plt.imshow(b[epoch_num].T, aspect="auto")
    plt.colorbar()
    plt.clim(-1, 1)
    plt.show()

pr_ = []
label_ = []

for sub in ['RNS7525', 'RNS9183', 'RNS8973', 'RNS1529', 'RNS2227', 'RNS1534']:
    list_PE = [f for f in os.listdir(os.path.join(PATH_OUT)) if sub in f and ".p" in f]

    out_ = []

    for PE in list_PE:
        with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
            b = pickle.load(handle)

        # plot_segment(b, epoch_num="E5")

        # first idea: just average all features for each segments, don't normalize

        for epoch_str in b.keys():
            out_.append(b[epoch_str].mean())

    df_out = pd.DataFrame(out_)

    feature_used = df_out[[col for col in df_out.columns if "ch" in col]]

    label = df_out["sz_on"] > 0

    estimator = xgboost.XGBClassifier()
    #estimator = linear_model.LogisticRegression()

    pr = model_selection.cross_val_predict(
        estimator=estimator,
        X = feature_used,
        y = label,
        cv = model_selection.KFold(shuffle=False, n_splits=5)
    )

    pr_.append(pr)
    label_.append(label)


for sub in np.arange(6):
    print(metrics.balanced_accuracy_score(label_[sub], pr_[sub]))

#RNS7525: 0.96
#RNS9183: 0.93
#RNS8973: 0.97
#RNS1529: 0.96
#RNS2227: 0.78
#RNS1534: 0.9

m_ = []
for m in np.arange(6):
    m_.append(metrics.confusion_matrix(label_[m], pr_[m], normalize='true'))

arr = np.array(m_).mean(axis=0)

fig = plt.figure(figsize=(4,4), dpi=300)
ax = fig.add_subplot(111)
metrics.ConfusionMatrixDisplay(
    arr
).plot(ax=ax)
plt.xlabel("Predicted Seizure")
plt.ylabel("Predicted Seizure")
plt.title("Mean patient confusion matrix")
#cbar = plt.colorbar()
#cbar.set_label("Normalized Accuracy")
plt.savefig(
    "ConfusionMatrix_MeanSubjects.pdf",
    bbox_inches="tight",
)




#####################################
###: Now Try Across Patient Analysis


def get_sub_data(sub):
    list_PE = [f for f in os.listdir(os.path.join(PATH_OUT)) if sub in f and ".p" in f]

    out_ = []

    for PE in list_PE:
        with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
            b = pickle.load(handle)

        # plot_segment(b, epoch_num="E5")

        # first idea: just average all features for each segments, don't normalize

        for epoch_str in b.keys():
            out_.append(b[epoch_str].mean())

    df_out = pd.DataFrame(out_)

    feature_used = df_out[[col for col in df_out.columns if "ch" in col]]

    label = df_out["sz_on"] > 0
    return feature_used, label

subs = ['RNS7525', 'RNS9183', 'RNS8973', 'RNS1529', 'RNS2227', 'RNS1534']

labels = []
features = []

for sub in subs:
    f, l = get_sub_data(sub)
    f["sub"] = sub
    labels.append(l)
    features.append(f)

df_f = pd.concat(features)
labels = pd.concat(labels)

ls = []
prs = []

for sub in subs:

    test_index = np.where(df_f["sub"] == sub)[0]
    train_index = np.where(df_f["sub"] != sub)[0]

    X_train, X_test = df_f.iloc[train_index], df_f.iloc[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

    estimator = xgboost.XGBClassifier()
    estimator.fit(X_train[[c for c in X_train.columns if "ch" in c]], y_train)

    pr = estimator.predict(X_test[[c for c in X_test.columns if "ch" in c]])
    prs.append(pr)
    ls.append(y_test)

    print(metrics.balanced_accuracy_score(y_true=y_test, y_pred=pr))








##############################################
#old:

metrics.balanced_accuracy_score(label, pr)

# Check why the results are so crazy good
df_melt = pd.melt(df_out[[col for col in df_out.columns if "ch" in col or col == "sz"]], id_vars = "sz")
df_melt["sz"] = df_melt["sz"] >0

sb.boxplot(df_melt, x="variable", y="value", hue="sz")

# plot only line length:
df_melt = pd.melt(df_out[[col for col in df_out.columns if "Line" in col or col == "sz"]], id_vars = "sz")
df_melt["sz"] = df_melt["sz"] >0
sb.boxplot(df_melt, x="variable", y="value", hue="sz")


# init analyzer
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=sub_name
)

arr = stats.zscore(feature_reader.feature_arr)
arr_background = arr[arr["ch1_LineLength"]<np.percentile(arr["ch1_LineLength"], q=75)]

plt.imshow(arr_background.T, aspect="auto")
plt.yticks(np.arange(arr_background.shape[1]), feature_reader.feature_arr.columns)
plt.show()

plt.imshow(arr.T, aspect="auto")
plt.yticks(np.arange(arr.shape[1]), feature_reader.feature_arr.columns)
plt.colorbar()
plt.clim(-1, 1)
plt.show()