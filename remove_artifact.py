import numpy as np


def get_stim_clip_cleaned(
    ts:np.array, 
    time: np.array, 
    sfreq:float
):
    cleaned_ts = []
    masks = []
    times_no_artifact = []
    for i in range(4):
        mask_clip = get_bool_mask_stim_artifact(ts[i], time)
        mask_stim = get_bool_mask_threshold_artifact(ts[i])
        mask = np.logical_and(mask_clip, mask_stim)
        masks.append(mask)
        cleaned_ts.append(ts[i][mask])
        times_no_artifact.append(
            np.arange(0, ts[i][mask].shape[0] / sfreq, 1 / sfreq)
        )
    return cleaned_ts, masks, times_no_artifact

def get_bool_mask_threshold_artifact(
    ts: np.array,
):
    """
    Simple Artifact Rejection method removing max value.
    RNS saturations might be epoch specific, but the maximum within a single epoch should be same
    """
    time_index_clip = np.where((ts == np.max(ts)) | (ts == np.min(ts)))[0]
    mask_clip = np.ones(ts.shape[0])
    mask_clip[time_index_clip] = 0
    mask_clip = mask_clip.astype('bool')
    return mask_clip
 
def get_bool_mask_stim_artifact(
    ts: np.array,
    time: np.array, 
    samples_consecutive_artifact: int = 12,
    samples_skip_rebound: int = 500
):
    """
    The Stimulation induces a flat line artifact in the time series.
    Minimum of non changing data for 'samples_consecutive_artifact'is here masked.
    'Samples_skip_rebound' add's samples to be exluded after the flat line artifact due to 
    high amplitude rebound effect. 
    """
    stim_segments = []
    stim_seg = []

    for idx, val in enumerate(np.diff(ts)):
        if val == 0:
            stim_seg.append(time[idx])
            if (idx + samples_skip_rebound) < time.shape[0]: 
                stim_seg.append(time[idx+samples_skip_rebound])
        if val != 0 and len(stim_seg) > samples_consecutive_artifact:
            if len(stim_segments) == 0:
                stim_segments.append(stim_seg)
            else:
                diff_to_last_stim = stim_seg[0] - stim_segments[-1][-1]
                if diff_to_last_stim < 0.1:
                    stim_segments[-1].append(stim_seg[-1])  # append to last previous stim segment
                else:
                    stim_segments.append(stim_seg)
                    
        if val != 0:
            stim_seg = []

    bool_mask_skip = np.ones(time.shape[0])
    for seg in stim_segments:
        bool_mask_skip[np.where((time>seg[0]) & (time<seg[-1]))[0]] = 0
    bool_mask_skip = bool_mask_skip.astype(bool, copy=False)

    return bool_mask_skip
