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
    try:
        file_read = t[0]
        annot_files = t[1]
        PATH_RAW_DATA = t[2]
        patient_subfolder = t[3]

        fn = os.path.basename(file_read)
        sub = fn[4:fn.find("_PE")]
        PE = fn[fn.find("_PE")+3:fn.find(".EDF")]

        dict_epochs_out = {}

        annot_table_pe = [
            i
            for i in annot_files
            if sub in i and PE[:-len(str("_EOF_SZ-VK"))] in i
        ]

        if len(annot_table_pe) == 0:
            print(f"skipping {file_read}")
            return
        else:
            annot_table_pe = annot_table_pe[0]

        annot_df = tf_plot_spec.SpectrumReader.read_annot(
            os.path.join(
                PATH_RAW_DATA,
                f"PIT-{patient_subfolder}",
    #            "iEEG",
                annot_table_pe
            )
        )

        #smb://132.183.240.28/nexus2/RNS_DataBank/PITT/PIT-RNS0427/iEEG
        raw = mne.io.read_raw_edf(file_read)
        raw_times = raw.times
        raw_data = raw.get_data()

        stream = pynm_init_features_rns.init_pynm_rns_stream()

        # data will be saved at the PATH_OUT in folder sub_name
        if RUN == True:
            #return None
            stream.run(
                data=raw_data,
                folder_name=f"PIT-{patient_subfolder}-{PE}",
                out_path_root=PATH_OUT,
            )

        # read stream to add seizure onset information:
        #PATH_OUT = r"X:\Users\timon\RNSOut_pynm_good"
        feature_reader = py_nm.nm_analysis.Feature_Reader(
            feature_dir=PATH_OUT,
            feature_file=f"PIT-{patient_subfolder}-{PE}"
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
    except Exception:
        print("skipping")

if __name__ == "__main__":
    PATH_OUT = r'X:\Users\timon\RNSOut_pynm_more_features'
    PATH_RAW_EDF = r'X:\RNS_DataBank\PITT'

    RUN = True

    patient_list = ['RNS9183', 'RNS8973', 'RNS1529', 'RNS2227', 'RNS1534', 'RNS7525']

    # get the annotation and edf files each of those patients

    list_edfs = []
    list_annots = []

    for patient in patient_list:
        PATH_RAW_DATA = os.path.join(PATH_RAW_EDF, f"PIT-{patient}", "iEEG") 
        annot_files_sub = [
            os.path.join(PATH_RAW_DATA, f)
            for f in os.listdir(PATH_RAW_DATA)
            if "SZ-VK.TXT" in f
        ]
        edf_files_sub = [os.path.join(PATH_RAW_DATA, f[:-4]+".EDF") for f in annot_files_sub]
        list_edfs.append(edf_files_sub)
        list_annots.append(annot_files_sub)

    process_args = []
    for idx_sub, f in enumerate(list_edfs):
        for idx_edf, f_ in enumerate(f):
            if os.path.exists(os.path.join(PATH_RAW_EDF, f"PIT-{patient_list[idx_sub]}", "iEEG", f_)):
                print("exists")
                print(f_)

                pp = os.path.basename(f_)[:-4]
                ppp = pp[:pp.find("_PE")]+"-"  + pp[pp.find("_PE")+3:]
                PATH_AT_DRIVE = os.path.join(PATH_OUT, ppp)

                #if os.path.exists(PATH_AT_DRIVE) is True:
                #    continue
                process_args.append(
                    (
                        list_edfs[idx_sub][idx_edf],
                        [list_annots[idx_sub][idx_edf]],
                        PATH_RAW_EDF,
                        patient_list[idx_sub]
                    )
                )
            else:
                print("NOT")
                print(f_)

    """ for f in process_args:
        try:
            process_PE(f)
        except Exception:
            print(f)
            print("continue")
    """
    #for idx, p in enumerate(process_args):
    #    process_PE(p)
    #pool = multiprocessing.Pool(15)
    #pool.map(process_PE, process_args)

    location = './cachedir'
    memory = Memory(location, verbose=0)
    costly_compute_cached = memory.cache(process_PE)

    def data_processing_mean_using_cache(b):
        """Compute the mean of a column."""
        return costly_compute_cached(b)

    results = Parallel(n_jobs=10)(
        delayed(data_processing_mean_using_cache)(p)
        for p in process_args
    )
    memory.clear(warn=False)


    """ for idx_sub in range(len(list_edfs)):
        for idx_edf in range(len(list_edfs[idx_sub])):
            process_PE(
                (
                    list_edfs[idx_sub][idx_edf],
                    [list_annots[idx_sub][idx_edf]],
                    PATH_RAW_EDF,
                    patient_list[idx_sub]
                )
            )
    """