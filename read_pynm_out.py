import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import xgboost
import pandas as pd

import seaborn as sb
from scipy import stats

from sklearn.utils import class_weight
from sklearn import metrics
from sklearn import linear_model

PATH_OUT = "/mnt/4TB/timon/RNSOut_pynm"
cols_use_norm = None 
READ_FILES = False


if READ_FILES == True:
    files = [f for f in os.listdir(PATH_OUT) if f.endswith(".p")]

    list_df = []
    for f in files:
        with open(os.path.join(PATH_OUT, f), 'rb') as f_:
            d = pickle.load(f_)
        for k in d.keys():
            df_add = d[k]
            if cols_use_norm == None:
                cols_use_norm = [c for c in df_add.columns if c.startswith("ch")]
            df_add["sub"] = f[:f.find("_")]
            df_add["PE"] = f[f.find("_")+1:-2]

            df_add[cols_use_norm] = (df_add[cols_use_norm] - df_add[cols_use_norm].mean())/df_add[cols_use_norm].std(ddof=0)

            list_df.append(d[k])
    df_all = pd.concat(list_df)
    df_all.to_csv("df_all.csv")
else:
    df_all = pd.read_csv("df_all.csv")    

cols_use = [c for c in df_all.columns if c.startswith("ch") or c.startswith("sub")]
df_use = df_all[cols_use]

label = df_all["sz"]
X = df_use.dropna().reset_index()
y = np.array(label) 

y_pr = []
y_te = []

for sub_test in np.unique(X["sub"]):
    print(sub_test)
    test_index = np.where(X["sub"] == sub_test)[0]
    train_index = np.where(X["sub"] != sub_test)[0]

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.drop("sub", axis=1)
    X_test = X_test.drop("sub",  axis=1)

    X_train = X_train.drop("index", axis=1)
    X_test = X_test.drop("index",  axis=1)

    X_train = stats.zscore(np.array(X_train))
    X_test = stats.zscore(np.array(X_test))
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    #model = xgboost.XGBClassifier()
    model = linear_model.LogisticRegression(class_weight="balanced")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pr.append(y_pred)
    y_te.append(y_test)

metrics.balanced_accuracy_score(
    y_pred=np.concatenate(y_pr),
    y_true=np.concatenate(y_te)
)
