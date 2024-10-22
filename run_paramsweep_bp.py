import py_neuromodulation as nm

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

def plot_segment(b, epoch_num: str = "E1", norm=True):
    if norm == True:
        plt.imshow(stats.zscore(b[epoch_num], axis=0).T, aspect="auto")
    else:
        plt.imshow(b[epoch_num].T, aspect="auto")
    plt.colorbar()
    plt.clim(-1, 1)
    plt.show()


def plot_detections(f_plt, det_arr):

    # det_arr: bool array of shape f_plt

    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(np.array(f_plt))
    plt.fill_between(
        np.arange(f_plt.shape[0]),
        np.min(f_plt),
        np.max(f_plt),
        det_arr,
        alpha=0.5,
        color="red",
        edgecolor=None,
        label="detections"
    )
    plt.legend()
    plt.xlabel("Time [s]")

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

def get_LE_status_RNS(sub, PE, segment_nr):

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
        for episode in ['A1', 'A2', 'B1', 'B2', 'Long Episode']:
            if episode in query_segment["name"].unique():
                detected = True

        #if "Long Episode" in query_segment["name"].unique():
        #    detected = True
        #else:
        #    detected = False

    return detected

def get_RNS_detections(b, sub, PE):
    LEs_RNS = []
    for seg in b.keys():
        LEs_RNS.append(get_LE_status_RNS(sub, PE, seg))
    return LEs_RNS

def get_seizure_GT(b):
    sz_ground_truth = []
    for seg in b.keys():
        sz_ground_truth.append(b[seg]["sz_on"].sum() > 0)
    return sz_ground_truth

def plot_ground_truth_vs_RNS_episodes(sz_ground_truth, LEs_RNS):
    plt.figure()
    plt.plot(sz_ground_truth, label="Ground Truth")
    plt.plot(LEs_RNS, label="RNS LE")
    plt.legend()

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

def plot_param_sweep_bp(param_duration_min, param_thr_bandpass, cm_out, MSK_OUT_FP_NOT_BEING_ZERO=False):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(np.abs(cm_out).T, aspect='auto')
    plt.xticks(np.arange(param_duration_min.shape[0])[::5], np.round(param_duration_min[::5]/10, 2))
    plt.yticks(np.arange(param_thr_bandpass.shape[0])[::2], np.round(param_thr_bandpass*100, 2)[::2])
    cbar = plt.colorbar()
    cbar.set_label("Accuracy\n")
    plt.title("RNS Bandpass Rythmic Activity Detector\n"+"$0.9\cdot TPR + 0.1\cdot TNR$")
    plt.title("RNS Bandpass Rythmic Activity Detector")
    plt.gca().invert_yaxis()
    #plt.clim(0.8, 0.95)
    #plt.gca().invert_xaxis()
    plt.ylabel("Threshold absolute value [%]")
    plt.xlabel("Minimum duration for detection [s]")
    plt.tight_layout()
    plt.savefig(
    f"Example_Bandpass_Detector_ParameterSweep_PE20160811_MASK_{str(MSK_OUT_FP_NOT_BEING_ZERO)}.pdf",
    bbox_inches="tight",
)

def get_thr_arrays(feature_name):
    with open(os.path.join(PATH_OUT, PE), 'rb') as handle:
        b = pickle.load(handle)

    index_current_PE = list_PE.index(PE)
    index_next_PE = index_current_PE + 1
    vals_PE_cur = np.concatenate([b[seg][feature_name] for seg in b.keys()])
    min_cur_PE = np.nanmin(vals_PE_cur)
    max_cur_PE = np.nanmax(vals_PE_cur)
    if index_next_PE == len(list_PE):
        # get default array of thresholds
        bins = np.arange(min_cur_PE, max_cur_PE, (max_cur_PE - min_cur_PE)/20)
    else:
        with open(os.path.join(PATH_OUT, list_PE[index_next_PE]), 'rb') as handle:
            b_next = pickle.load(handle)
            vals_PE_next = np.concatenate([b_next[seg][feature_name] for seg in b_next.keys()])
            min_next_PE = np.nanmin(vals_PE_next)
            max_next_PE = np.nanmax(vals_PE_next)
        min_all = np.min((min_cur_PE, min_next_PE))
        max_all = np.max((max_cur_PE, max_next_PE))
        bins = np.arange(min_all, max_all, (max_all - min_all)/20)
    return bins

PATH_OUT = r"X:\Users\timon\RNSOut_pynm_good"
PATH_OUT = r"C:\Users\ICN_admin\Documents\Datasets\Boston Epilepsy RNS\RNSOut_pynm_more_features"

#df_RNS_annot = pd.read_csv(r"X:\Users\timon\RNS_Detect_Annotations\Pitt_Ecogs_events_updated_01212022.csv")

subjects_use = ['RNS1529', 'RNS8973', 'RNS7525', 'RNS9183', 'RNS2227', 'RNS1534']

df_per_RNS = pd.read_pickle("performances_RNS_bandpass_new.p")

out_save_dict = {}

for sub in subjects_use[::-1]:
    out_save_dict[sub] = {}

    list_PE = [f for f in os.listdir(os.path.join(PATH_OUT)) if sub in f and ".p" in f]
    
    for PE in list_PE:
        per_RNS = df_per_RNS.query("sub == @sub and PE == @PE")
        if per_RNS.shape[0] == 0:
            continue
        out_save_dict[sub][PE] = {}

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
            # Now get the max
            TNR = out_all[:, :, :, 0, 0] / (out_all[:, :, :, 0, 0] + out_all[:, :, :, 0, 1])
            TPR = out_all[:, :, :, 1, 1] / (out_all[:, :, :, 1, 0] + out_all[:, :, :, 1, 1])
            weight_N = 0.5
            weight_P = 0.5
            cm_out = TNR * weight_N + TPR * weight_P
            max_feature, max_x, max_y = np.unravel_index(np.nanargmax(np.abs(cm_out)), cm_out.shape)
            max_feature_name = col_use[max_feature]
            print(out_all[max_feature, max_x, max_y])

            out_all_SZ = out_all
            out_all_SZ[out_all_SZ[:, :, :, 1, 0] != 0, :, :] = None
            TNR = out_all[:, :, :, 0, 0] / (out_all[:, :, :, 0, 0] + out_all[:, :, :, 0, 1])
            TPR = out_all[:, :, :, 1, 1] / (out_all[:, :, :, 1, 0] + out_all[:, :, :, 1, 1])
            weight_N = 0.5
            weight_P = 0.5
            cm_out = TNR * weight_N + TPR * weight_P
            max_feature_allsz, max_x_allsz, max_y_allsz = np.unravel_index(np.nanargmax(np.abs(cm_out)), cm_out.shape)


            #plot_param_sweep_bp(
            #    param_duration_min,
            #    param_thr_bandpass,
            #    cm_out[max_feature]
            #)
            param_thr_bandpass = get_thr_arrays(col_use[max_feature])
            out_save_dict[sub][PE]["out_all"] = out_all
            out_save_dict[sub][PE]["cm_out"] = cm_out
            out_save_dict[sub][PE]["cm_best_within"] = out_all[max_feature, max_x, max_y]
            out_save_dict[sub][PE]["cm_best_within_all_sz"] = out_all[max_feature_allsz, max_x_allsz, max_y_allsz]
            out_save_dict[sub][PE]["best_param_duration"] = param_duration_min[max_x]
            out_save_dict[sub][PE]["best_param_thr"] = param_thr_bandpass[max_y]
            out_save_dict[sub][PE]["best_feature_select"] = max_feature_name
            out_save_dict[sub][PE]["thr_arr"] = param_thr_bandpass
            out_save_dict[sub][PE]["RNS_PER_cm"] = df_per_RNS.query("sub == @sub and PE == @PE")["cm"]
            out_save_dict[sub][PE]["RNS_PER_cm_norm"] = df_per_RNS.query("sub == @sub and PE == @PE")["cm_normed"]

    with open(f'{sub}_per_within_and_next_PE.p', 'wb') as handle:
        pickle.dump(out_save_dict[sub], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
# save results

# Now compare the confusion matrices for both cases
# optimal params for 5.6 thr and 4 time
df_RNS_BP = pd.read_pickle("performances_RNS_bandpass.p")
df_RNS_BP.query("sub == @sub and PE == @PE")["cm"]

max_feature, max_x, max_y = np.unravel_index(np.nanargmax(np.abs(out_arr)), out_arr.shape)
print(param_duration_min[max_x])
print(param_thr_bandpass[max_y])
print(cm_param_out[max_x, max_y])

# but now, also get the best param from the RNS


fig = plt.figure(figsize=(8,4), dpi=300)
ax = fig.add_subplot(121)
metrics.ConfusionMatrixDisplay(
    cm_param_out[max_x, max_y, :, :]
).plot(ax=ax)
plt.xlabel("Predicted Seizure")
plt.ylabel("True Seizure")
plt.title("Best param confusion matrix")

ax = fig.add_subplot(122)
metrics.ConfusionMatrixDisplay(
    cm_PE
).plot(ax=ax)
plt.xlabel("Predicted Seizure")
plt.ylabel("True Seizure")
plt.title("RNS device parameter")
plt.tight_layout()

plt.savefig(
    "Example_ConfusionMatrixComparison_PE20160811.pdf",
    bbox_inches="tight",
)


# Now plot detections for the run I previosly identified
# segment: RNS1529_PE20160811 segment E219

#E219
det_arr, l_det = get_bp_det_arr(
    "E219",
    feature_name,
    thr_bandpass=param_duration_min[max_x],
    duration_min_=param_thr_bandpass[max_y]
)


LE_detect_status = get_LE_status_RNS(sub, PE, segment_nr)

f_arr = np.concatenate([np.array(b[seg][feature_name]) for seg in b.keys()])
