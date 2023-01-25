import os
import time
import numpy as np
import mne
from matplotlib import pyplot as plt
import pandas as pd


class SpectrumReader:
    def __init__(
        self,
        PATH_SPEC_PRECOM: str = "/home/timonmerk/Documents/Data/Victoria iESPnet/Spec_Pitt/PITT",
        PATH_RAW_DATA: str = "/home/timonmerk/Documents/Data/Victoria iESPnet/PIT-RNS0427/iEEG",
    ) -> None:
        self.PATH_SPEC_PRECOM = PATH_SPEC_PRECOM
        self.PATH_RAW_DATA = PATH_RAW_DATA
    
    def get_center_sub_pe(self, epoch_str: str):
        '''
        epoch_str in format 
        '''
        center = epoch_str[:-3].split("_")[0]
        sub = epoch_str[:-3].split("_")[1]
        pe = epoch_str[:-3].split("_")[2]
        return center, sub, pe


    def read_spectrum(self, file_name: str="PIT_RNS0427_20190115-1_E30.npy"):
        spec_ = np.load(
            os.path.join(self.PATH_SPEC_PRECOM, file_name),
            allow_pickle=True,
        )

        self.spec = spec_.reshape(-1,)[
            0
        ]["spectrogram"]
        return self.spec

    def read_raw_time_series(
        self, f_name: str = "PIT-RNS0427_PE20190115-1_EOF_SZ-VK.EDF"
    ):
        # get the time series
        self.raw = mne.io.read_raw_edf(os.path.join(self.PATH_RAW_DATA, f_name))
        return self.raw

    @staticmethod
    def read_annot(file_annot_path):
        #file_annot_path = os.path.join(self.PATH_RAW_DATA, annot_file)

        file_annot = open(file_annot_path, "r")

        df = pd.DataFrame()

        time_ = []
        annot_ = []
        annot_epoch_str = []

        cnt_annot = 0
        for line in file_annot.readlines()[1:]:
            # print(line)
            fname = line.rstrip().split(",")  # using rstrip to remove the \n
            if fname[1] == '':
                annot = fname[2]
            else:
                annot = fname[1]
            if annot == "eof":
                cnt_annot += 1

            time_.append(float(fname[0]))
            annot_.append(annot)
            annot_epoch_str.append(f"E{cnt_annot}")
        df["time"] = time_
        df["annot"] = annot_
        df["annot_epoch_str"] = annot_epoch_str
        #self.annot_df = df
        return df

    def get_times_around_programming_epoch(self, epoch_: str = "E30", annot_df: pd.DataFrame=None):
        epoch_next = f"E{int(epoch_[1:])+1}"

        if annot_df is None:
            annot_df = self.annot_df
        
        if epoch_ == "E0":
            time_start = 0
        else:
            time_start = float(
                annot_df.query('annot=="eof" & annot_epoch_str == @epoch_')["time"]
            )
        time_start_next = float(
            annot_df.query('annot=="eof" & annot_epoch_str == @epoch_next')["time"]
        )
        sz_on_epoch = float(
            annot_df.query('annot=="sz_on" & annot_epoch_str == @epoch_')["time"]
        )

        return time_start, time_start_next, sz_on_epoch

    @staticmethod
    def get_data_raw_epoch(data, times, time_start, time_start_next):
        idx_start = np.where((times > time_start) == 1)[0][0]
        idx_end = np.where((times > time_start_next) == 1)[0][0]

        dat_epoch = data[:, idx_start:idx_end]
        times = times[idx_start:idx_end] - times[idx_start]
        return dat_epoch, times, idx_start

    @staticmethod
    def plot_epoch(dat_epoch, spec, times_epoch, sz_onset_time:float=None, save_path:str=None):
        # check time segment for E30: starting from 5344.92 to 5525.12

        plt.ioff()

        plt.figure(figsize=(8, 8), dpi=300)
        plt.subplot(211)
        plt.plot(times_epoch, dat_epoch, color="black")

        if sz_onset_time is not None:
            plt.axvline(
                x=sz_onset_time,
                label="Seizure Onset",
                color="red",
            )
            plt.legend()
        plt.xlim(0, times_epoch.max())
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        xticks = plt.xticks()[0]

        plt.subplot(212)
        plt.imshow(spec, aspect="auto")
        plt.gca().invert_yaxis()
        plt.ylabel("Frequency [Hz]")
        plt.title("Time Frequency Plot")
        
        #plt.xticks(
        #    np.arange(0, spec.shape[1]-spec.shape[1]/(xticks.shape[0]), spec.shape[1]/(xticks.shape[0])),
        #    xticks[:-1]
        #)
        #plt.xlabel("Time [s]")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
