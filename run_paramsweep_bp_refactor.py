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
from numba import njit

@njit
def get_bp_det_arr(
    data, thr_bandpass=8, duration_min_=3,
    inv: bool = False
):

    if inv == False:
        thr_crossed = data > thr_bandpass
    else:
        thr_crossed = data < thr_bandpass

    in_detection = False
    det_counter = 0
    l_det = []
    det_arr = np.zeros(thr_crossed.shape[0])

    for idx, val in enumerate(thr_crossed):
        if val == True:
            det_counter += 1
            in_detection = True
        if val == False:
            if in_detection == True:
                in_detection = False
                if det_counter >= duration_min_:
                    l_det.append((idx - det_counter, det_counter))
                    for i in np.arange(idx - det_counter, idx, 1):
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
        np.array(np.array(sz_ground_truth)[idx_not_none] * 1.0).astype(int),
        np.array(np.array(LEs_RNS)[idx_not_none] * 1.0).astype(int),
    )
    return cm_PE


def get_cm_for_pynm_predict_params(
    b,
    feature_name,
    inv: bool,
    thr_bandpass=7.8,
    duration_min_=3,
    ):
    out_pr = []
    det_time = []

    for segment_nr in b.keys():
        det_arr, l_det = get_bp_det_arr(
            np.array(b[segment_nr][feature_name]),
            thr_bandpass=thr_bandpass,
            duration_min_=duration_min_,
            inv=inv
        )

        if len(l_det) > 0:
            out_pr.append(1)
        else:
            out_pr.append(0)
        if np.nonzero(det_arr)[0].shape != 0:
            det_time.append(None)
        else:
            det_time.append(np.nonzero(det_arr)[0][0])
    return out_pr, det_time


def make_parameter_sweep_bp(
    b,
    sz_ground_truth,
    feature_name,
    param_duration_bandpass=np.arange(1, 30, 2),
    inv: bool = False
):
    param_thr_bandpass_arr = get_thr_arrays(
        feature_name,
        b,
        num_bins=NUM_BINS_THR
    )
    cm_param_out = np.empty(
        [param_duration_bandpass.shape[0], param_thr_bandpass_arr.shape[0], 2, 2]
    )
    det_timepoint_param_out = np.empty(
        [param_duration_bandpass.shape[0], param_thr_bandpass_arr.shape[0]], dtype=object
    )
    cm_param_out[:] = None
    #for inv in [True, False]:
    for i, param_dur in enumerate(param_duration_bandpass):
        for j, param_thr in enumerate(param_thr_bandpass_arr):
            try:
                out_pr_param_set, det_time_points = get_cm_for_pynm_predict_params(
                    b, feature_name, thr_bandpass=param_thr, duration_min_=param_dur,
                    inv=inv
                )

                cm_params = get_confusion_matrix(sz_ground_truth, out_pr_param_set)
                #if inv == True:
                cm_param_out[i, j, :, :] = cm_params
                det_timepoint_param_out[i, j] = det_time_points
                #else:   
                #    cm_param_out[1, i, j, :, :] = cm_params
                #    det_timepoint_param_out[1, i, j] = det_time_points
            except Exception as e:
                print(e)
                continue
    return [cm_param_out, det_timepoint_param_out, param_thr_bandpass_arr]


def get_thr_arrays(
    feature_name: str,
    dict_feature: dict,
    num_bins: int = 20
):
    vals_PE_cur = np.concatenate([dict_feature[seg][feature_name] for seg in dict_feature.keys()])

    min_cur_PE = np.nanmin(vals_PE_cur)
    max_cur_PE = np.nanmax(vals_PE_cur)
    bins = np.arange(min_cur_PE, max_cur_PE, (max_cur_PE - min_cur_PE) / num_bins)

    return bins

if __name__ == "__main__":
    PATH_OUT = r"X:\Users\timon\RNSOut_pynm_more_features_2"
    PATH_OUT = (
        r"C:\Users\ICN_admin\Downloads\predictions_sweep\predictions_sweep\folder_features"
    )

    df_per_RNS = pd.read_csv("edf_files_to_run.csv")
    subjects_use = df_per_RNS["sub"].unique()

    with Parallel(n_jobs=45) as parallel:
        for sub in subjects_use[::-1]:
            out_save_dict = {}

            list_PE = [f for f in os.listdir(os.path.join(PATH_OUT)) if sub in f and ".p" in f]

            for PE in list_PE[::-1]:
                PE_str = PE[-10:-2]
                #if os.path.exists(f"{sub}_per_PE.p"):
                #    print(f"skipping {sub}_per_PE.p")
                #    continue
                out_save_dict[PE_str] = {}

                with open(os.path.join(PATH_OUT, PE), "rb") as handle:
                    B = pickle.load(handle)

                    sz_ground_truth = get_seizure_GT(B)

                    #cm, det_time = make_parameter_sweep_bp(
                    #    b,
                    #    sz_ground_truth,
                    #    col_use[0],
                    #    param_thr_bandpass,
                    #    True
                    #)

                    # RUN THE whole feature estimation in a 3 fold CV;
                    # LIMIT the range of b
                    
                    arrs_ = np.arange(len(sz_ground_truth))
                    
                    cv = model_selection.KFold(n_splits=3, shuffle=False)
                    cm_test = []
                    f1_score = []
                    f1_score_tr = []
                    for train, test in cv.split(arrs_):

                        b_train = {k: v for k, v in B.items() if int(k[1:]) in train}
                        b_test = {k: v for k, v in B.items() if int(k[1:]) in test}

                        sz_ground_truth_train = np.array(sz_ground_truth)[train]
                        sz_ground_truth_test = np.array(sz_ground_truth)[test]

                        location = "./cachedir"
                        memory = Memory(location, verbose=0)
                        costly_compute_cached = memory.cache(make_parameter_sweep_bp)

                        def data_processing_mean_using_cache(
                            b_train, sz_ground_truth_train, column, param_duration_bandpass, inv):
                            """Compute the mean of a column."""
                            return costly_compute_cached(
                                b_train,
                                sz_ground_truth_train,
                                column,
                                param_duration_bandpass,
                                inv
                            )

                        col_use = [f for f in B[list(B.keys())[0]].columns if "fft" in f]
                        param_duration_bandpass = np.arange(1, 30, 2)
                        NUM_BINS_THR = 10

                        #col_use = ["ch1_fft_theta", "ch1_fft_alpha"]
                        #param_duration_bandpass = np.arange(1, 30, 15)
                        #NUM_BINS_THR = 20
                        

                        res = parallel(
                                delayed(data_processing_mean_using_cache)(
                                    b_train, sz_ground_truth, col, param_duration_bandpass, inv
                            )
                            for inv in [True, False]
                            for col in col_use
                        )
                        memory.clear(warn=False)
                        # first 24 columns inverted, next ones not

                        l_cm = []
                        l_t = []
                        thr_arr_features = []
                        for r in res:
                            l_cm.append(r[0])
                            l_t.append(r[1])
                            thr_arr_features.append(r[2])
                        cm_out = np.array(l_cm)
                        t_out = np.array(l_t)

                        # calculate first f1 score and select then the best params
                        tp  = cm_out[:, :, :, 1, 1]
                        tn = cm_out[:, :, :, 0, 0]
                        fp = cm_out[:, :, :, 0, 1]
                        fn = cm_out[:, :, :, 1, 0]

                        precision = np.nan_to_num(tp / (tp + fp))
                        recall = np.nan_to_num(tp / (tp + fn))

                        F1 = np.nan_to_num((2 * precision * recall) / (precision + recall))

                        best_feature_idx, best_time_idx, best_amplitude_idx = np.unravel_index(
                            np.nanargmax(F1),
                            tp.shape
                        )

                        # first half of features in col_use are inverted; second inv = True
                        if best_feature_idx >= len(col_use):
                            inv_used_best = False
                            best_feature = col_use[best_feature_idx - len(col_use)]
                            thr_arr = thr_arr_features[best_feature_idx - len(col_use)]
                        else:
                            inv_used_best = True
                            # best feature index stays the same
                            best_feature = col_use[best_feature_idx]
                            thr_arr = thr_arr_features[best_feature_idx]

                        param_thr = thr_arr[best_amplitude_idx]
                        param_dur = param_duration_bandpass[best_time_idx]

                        best_F1_param_sweep = F1[best_feature_idx, best_time_idx, best_amplitude_idx]
                        f1_score_tr.append(best_F1_param_sweep)

                        # select here best params of cm_out; predict those on all the b_test dict values
                        out_pr_param_set, det_time_points = get_cm_for_pynm_predict_params(
                            b_test, best_feature, thr_bandpass=param_thr, duration_min_=param_dur,
                            inv=inv_used_best
                        )

                        c_test = get_confusion_matrix(sz_ground_truth, out_pr_param_set)
                        tp  = c_test[1, 1] + 0.00001
                        tn = c_test[0, 0] + 0.00001
                        fp = c_test[0, 1] + 0.00001
                        fn = c_test[1, 0] + 0.00001

                        precision = np.nan_to_num(tp / (tp + fp))
                        recall = np.nan_to_num(tp / (tp + fn))

                        F1 = np.nan_to_num((2 * precision * recall) / (precision + recall))


                        f1_score.append(F1)

                    PE_select = int(PE_str)

                    cm_rns = df_per_RNS.query("sub == @sub and PE == @PE_select ")["cm"].iloc[0]

                    out_save_dict[PE_str]["f1_score_test"] = f1_score
                    out_save_dict[PE_str]["f1_score_train"] = f1_score_tr
                    out_save_dict[PE_str]["RNS_PER_cm"] = cm_rns
                    #out_save_dict[PE_str]["det_time_pynm"] = t_out

            with open(f"{sub}_per_PE.p", "wb") as handle:
                pickle.dump(out_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
