from dataclasses import dataclass
from operator import ne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import multiprocessing
import mne
import os
from scipy import stats
import pickle
from joblib import Memory
from joblib import Parallel, delayed

import IO
import tf_plot_spec
import remove_artifact
import pynm_init_features_rns

import py_neuromodulation as py_nm


def get_segment_stim_times(annot_df: pd.DataFrame, row: pd.Series, raw_times: np.array):
    PE_number = row["annot_epoch_str"]
    print(PE_number)

    PE_number_next = f"E{int(PE_number[1:])+1}"
    if int(PE_number[1:]) == 0:
        start_time = 0
    else:
        start_time = float(
            annot_df.query("annot_epoch_str == @PE_number & annot == 'eof'")["time"]
        )
    if (
        annot_df.query("annot_epoch_str == @PE_number_next & annot == 'eof'")[
            "time"
        ].shape[0]
        == 0
    ):
        stop_time = raw_times[-1]
    else:
        stop_time = float(
            annot_df.query("annot_epoch_str == @PE_number_next & annot == 'eof'")[
                "time"
            ]
        )
    return start_time, stop_time, PE_number


def process_PE(t):
    #try:

    PATH_EDF = t[0]
    PATH_ANNOT = t[1]
    sub = t[2]
    PE = t[3]

    dict_epochs_out = {}

    annot_df = tf_plot_spec.SpectrumReader.read_annot(
        PATH_ANNOT
    )

    #smb://132.183.240.28/nexus2/RNS_DataBank/PITT/PIT-RNS0427/iEEG
    raw = mne.io.read_raw_edf(PATH_EDF)
    raw_times = raw.times
    raw_data = raw.get_data()

    stream = pynm_init_features_rns.init_pynm_rns_stream()

    # data will be saved at the PATH_OUT in folder sub_name
    if RUN == True:
        #return None
        stream.run(
            data=raw_data[:, :],
            folder_name=f"PIT-{sub}-{PE}",
            out_path_root=PATH_OUT,
        )

    # read stream to add seizure onset information:
    #PATH_OUT = r"X:\Users\timon\RNSOut_pynm_good"
    feature_reader = py_nm.nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT,
        feature_file=f"PIT-{sub}-{PE}"
    )

    arr = feature_reader.feature_arr
    arr["time_s"] = arr["time"] / 1000

    for index, row in annot_df.iterrows():

        if row["annot"] != "eof":
            continue
        start_time, stop_time, PE_number = get_segment_stim_times(
            annot_df, row, raw_times
        )

        segment_nr = row["annot_epoch_str"]
        if start_time == stop_time:
            print("Error: start time == stop time")
            continue

        try:
            dat_epoch, times, idx_start = tf_plot_spec.SpectrumReader.get_data_raw_epoch(
                raw_data, raw.times, start_time, stop_time
            )
        except Exception as e:
            print("error retrieving the segment end")
            continue

        (
            dat_no_artifact,
            bool_mask_skip,
            times_no_artifact,
        ) = remove_artifact.get_stim_clip_cleaned(dat_epoch, times, 250)

        # currently data is estimated every 4 ms
        # idea now: get samples every 100ms, since fs = 250, get every 25th value
        times_resampled = times[::250] # skip first 1s samples

        feature_arr_seg = arr.query('time_s >= @start_time and time_s <= @stop_time')

        sz_row = annot_df.query(
            "(annot == 'sz_on' or annot == 'sz_on_r' or annot == 'sz_on_l' or annot == 'sz_l' or annot == 'sz_r') and annot_epoch_str == @segment_nr"
        )

        feature_arr_seg["sz_on"] = 0
        feature_arr_seg["sz"] = 0
        if sz_row.shape[0] == 0:
            label = 0
        else:
            time_sz = float(sz_row.iloc[0]["time"] - start_time)
        
            # point the index where the time onset is
            feature_arr_seg["time_aligned_s"] = feature_arr_seg["time_s"] - feature_arr_seg.iloc[0]["time_s"]
            pos_sz_on = feature_arr_seg.query("time_aligned_s >= @time_sz").index[0]
            pos_sz = feature_arr_seg.query("time_aligned_s >= @time_sz").index
            feature_arr_seg.loc[pos_sz_on, "sz_on"] = 1
            feature_arr_seg.loc[pos_sz, "sz"] = 1

        #set all values that have mask_used being zero to None
        for ch_idx in np.arange(1, 5, 1):
            ch_name = f"ch{ch_idx}"
            cols = [col for col in feature_arr_seg.columns if col.startswith(ch_name)]

            mask_used = bool_mask_skip[ch_idx-1][::250]
            if mask_used.shape[0] > feature_arr_seg.shape[0]:
                mask_used = mask_used[:feature_arr_seg.shape[0]]

            feature_arr_seg.loc[np.logical_not(mask_used), cols] = None

        dict_epochs_out[segment_nr] = feature_arr_seg

    # save the dict to pic
    with open(os.path.join(PATH_OUT, f'{sub}_{PE}.p'), 'wb') as handle:
        pickle.dump(dict_epochs_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #except Exception:
    #    print(f"skipping {t}")

if __name__ == "__main__":

    PATH_OUT = r'X:\Users\timon\RNSOut_pynm_more_features_2'
    PATH_RAW_EDF = r'X:\RNS_DataBank\PITT'

    RUN = True

    df_to_run = pd.read_csv("edf_files_to_run.csv")
    process_args = []
    for idx, row in df_to_run.iterrows():
        PATH_RUN = row["edf_file"]
        sub = row["sub"]
        PE = str(row["PE"])
        PATH_RAW_DATA = os.path.dirname(PATH_RUN)

        PATH_ANNOT = [
            os.path.join(PATH_RAW_DATA, f)
            for f in os.listdir(PATH_RAW_DATA)
            if "-1_EOF_SZ-VK.TXT" in f and PE in f
        ]
        if len([f for f in os.listdir(PATH_OUT) if sub in f and PE in f])>0:
            continue
        if len(PATH_ANNOT) == 1:
            process_args.append(
                (
                    PATH_RUN,
                    PATH_ANNOT[0],
                    sub,
                    PE
                )
            )

    
    #process_PE(process_args[5])

    location = './cachedir'
    memory = Memory(location, verbose=0)
    costly_compute_cached = memory.cache(process_PE)

    def data_processing_mean_using_cache(b):
        """Compute the mean of a column."""
        return costly_compute_cached(b)

    results = Parallel(n_jobs=len(process_args))(
        delayed(data_processing_mean_using_cache)(p)
        for p in process_args
    )
    memory.clear(warn=False)
