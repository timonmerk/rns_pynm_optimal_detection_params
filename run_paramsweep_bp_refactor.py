import py_neuromodulation as nm
import ast 
import time
from scipy import stats
import pickle
import os
import pandas as pd
import seaborn as sb

import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics, linear_model
from sklearn import preprocessing
import xgboost
from joblib import Memory
from joblib import Parallel, delayed

def get_bp_det_arr(segment_nr, feature_name, b, thr_bandpass = 8, duration_min_ = 3, scale=False):

    if scale == True:
        data = preprocessing.MinMaxScaler().fit_transform(np.array(b[segment_nr][feature_name]).reshape(-1, 1))
    else:
        data = b[segment_nr][feature_name]
    above_thr = (data > thr_bandpass)

    in_detection = False
    det_counter = 0
    l_det = []
    det_arr = np.zeros(above_thr.shape[0])

    for idx, val in enumerate(above_thr):
        if val == True:
            det_counter += 1
            in_detection = True
        if val == False:
            if in_detection == True:
                in_detection = False
                if det_counter >= duration_min_:
                    l_det.append((idx-det_counter, det_counter))
                    for i in np.arange(idx-det_counter, idx, 1):
                        det_arr[i] = 1
            det_counter = 0
    return det_arr, l_det

def get_seizure_GT(b):
    sz_ground_truth = []
    for seg in b.keys():
        sz_ground_truth.append(b[seg]["sz_on"].sum() > 0)
    return sz_ground_truth

def get_confusion_matrix(sz_ground_truth, LEs_RNS):
    idx_not_none = [idx for idx, i in enumerate(LEs_RNS) if i is not None]

    # get confusion matrix
    cm_PE = metrics.confusion_matrix(
        np.array(np.array(sz_ground_truth)[idx_not_none]*1.0).astype(int),
        np.array(np.array(LEs_RNS)[idx_not_none]*1.0).astype(int)
    )
    return cm_PE

def get_cm_for_pynm_predict_params(b, feature_name, thr_bandpass=7.8, duration_min_=3):
    out_pr = []

    for segment_nr in b.keys():
        det_arr, l_det = get_bp_det_arr(
            segment_nr,
            feature_name,
            b,
            thr_bandpass=thr_bandpass,
            duration_min_=duration_min_
        )
        if len(l_det) > 0:
            out_pr.append(1)
        else:
            out_pr.append(0)
    return out_pr

def make_parameter_sweep_bp(
    b,
    sz_ground_truth,
    feature_name,
    param_duration_min = np.arange(1, 50, 1),
):

    param_thr_bandpass = get_thr_arrays(feature_name)
    cm_param_out = np.zeros([param_duration_min.shape[0], param_thr_bandpass.shape[0], 2, 2])
    for i, param_dur in enumerate(param_duration_min):
        for j, param_thr in enumerate(param_thr_bandpass):
            out_pr_param_set = get_cm_for_pynm_predict_params(
                b,
                feature_name,
                thr_bandpass=param_thr,
                duration_min_=param_dur
            )

            cm_params = get_confusion_matrix(sz_ground_truth, out_pr_param_set)

            cm_param_out[i, j, :, :] = cm_params
    return cm_param_out

def get_thr_arrays(feature_name):
    with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
        b = pickle.load(handle)

    vals_PE_cur = np.concatenate([b[seg][feature_name] for seg in b.keys()])
    min_cur_PE = np.nanmin(vals_PE_cur)
    max_cur_PE = np.nanmax(vals_PE_cur)
    bins = np.arange(min_cur_PE, max_cur_PE, (max_cur_PE - min_cur_PE)/20)
    return bins

PATH_OUT = r"X:\Users\timon\RNSOut_pynm_more_features_2"

df_per_RNS = pd.read_csv("edf_files_to_run.csv")
subjects_use = df_per_RNS["sub"].unique()
out_save_dict = {}

for sub in subjects_use:
    out_save_dict[sub] = {}

    list_PE = [f for f in os.listdir(os.path.join(PATH_OUT)) if sub in f and ".p" in f]
    
    for PE in list_PE:
        PE_str = PE[-10:-2]
        if os.path.exists(f'{sub}_per_PE.p'):
            print(f"skipping {sub}_per_PE.p")
            continue
        out_save_dict[sub][PE_str] = {}

        with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
            b = pickle.load(handle)

            sz_ground_truth = get_seizure_GT(b)

            col_use = [f for f in b["E1"].columns if "fft" in f]

            param_duration_min = np.arange(1, 50, 1)
            #param_thr_bandpass = np.arange(0, 1, 0.05) OR
            #res = make_parameter_sweep_bp(b, sz_ground_truth, col_use[0])

            location = './cachedir'
            memory = Memory(location, verbose=0)
            costly_compute_cached = memory.cache(make_parameter_sweep_bp)
            def data_processing_mean_using_cache(b, sz_ground_truth, column):
                """Compute the mean of a column."""
                return costly_compute_cached(b, sz_ground_truth, column)
            results = Parallel(n_jobs=len(col_use))(
                delayed(data_processing_mean_using_cache)(b, sz_ground_truth, col)
                for col in col_use
            )
            memory.clear(warn=False)

            out_all = np.array(results)

            PE_select = int(PE_str)

            cm_rns = df_per_RNS.query("sub == @sub and PE == @PE_select ")['cm'].iloc[0]

            out_save_dict[sub][PE_str]["out_all"] = out_all
            out_save_dict[sub][PE_str]["RNS_PER_cm"] = cm_rns

    with open(f'{sub}_per_PE.p', 'wb') as handle:
        pickle.dump(out_save_dict[sub], handle, protocol=pickle.HIGHEST_PROTOCOL)
