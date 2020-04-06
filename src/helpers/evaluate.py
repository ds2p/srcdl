"""
Copyright (c) 2020 CRISP

Evalute several metrics

:author: Andrew H. Song
"""

import numpy as np
import scipy.signal

import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

from src.generators.generate import generate_interpolated_Dictionary
from .convolution import *

from tqdm import tqdm

"""
Copyright (c) 2020 CRISP

Helpers for several evaluations

:author: Andrew H. Song
"""

def findpeaks(signal, threshold, tflag=False):

    """

    Find peaks from the given signal that cross a given threshold
    The peaks might not be separated further than the template length

    Inputs
    ======

    signal: signal in an array form
    threshold: threshold for identifying peaks
    tflag: if true, find peaks greater than threshold.
           if false, find peaks smaller than threshold.

    Outputs
    =======

    peak_index: an array of peak indices

    """
    numOfseg = len(signal.keys())
    peak_index = {}

    if tflag and threshold <0:
        print("Trying to find peaks above negative threshold. Are you sure?")

    if tflag == 0 and threshold >0:
        print("Trying to find peaks below positive threshold. Are you sure?")

    if not tflag:
        threshold = -threshold

    for idx in np.arange(numOfseg):
        if not tflag:
            sig = -signal[idx]
        else:
            sig = signal[idx]

        indices = np.where(sig>threshold)[0]

        if len(indices)>0:
            peak_index[idx]=[]

            last_idx = indices[-1]
            curridx = indices[0]

            # Better algorithm required
            while curridx <= last_idx:
                temp_idx = []
                while sig[curridx] > threshold:
                    temp_idx.append(curridx)
                    curridx += 1
                maxidx = np.argmax(abs(sig[temp_idx]))
                peak_index[idx].append(maxidx + temp_idx[0])

                nextidx = np.where(indices >= curridx)[0] # Find the next threshold crossing
                if not len(nextidx):
                    break
                curridx = indices[nextidx[0]]

    return peak_index


def recovery_error(dict1, dict2):

    """
    Compute the error between the corresponding columns of the two dictionaries.

    Inputs
    ======

    dict1: dictionary 1
    dict2: dictionary 2

    Outputs
    =======

    err_distance: error distance between the filters

    """
    assert(np.shape(dict1)==np.shape(dict2)), "Dictionaries are of different dim!"
    filternum = np.shape(dict1)[1]

    err_distance = np.zeros(filternum)

    dict1 = dict1/np.linalg.norm(dict1, axis=0)
    dict2 = dict2/np.linalg.norm(dict2, axis=0)

    for i in np.arange(filternum):
        diff = 1-np.power(np.matmul(np.transpose(dict1[:,i]), dict2[:,i]),2)
        # Numerical issue
        if abs(diff)< 1e-6:
            diff = 0

        err_distance[i] = np.sqrt(diff)

    return err_distance

def recovery_error_interp(dict1, dict2, numOfsubgrids):
    """

    Compute the error between the corresponding columns of the two dictionaries.

    Inputs
    ======

    dict1: dictionary 1
    dict2: dictionary 2

    Outputs
    =======

    err_distance: error distance between the filters

    """
    assert(np.shape(dict1)==np.shape(dict2)), "Dictionaries are of different dim!"
    filternum = np.shape(dict1)[1]

    offset = 5
    offsets = np.arange(-offset,offset+1,dtype=int)

    err_distance = np.zeros(filternum)

    dict1 = dict1/np.linalg.norm(dict1, axis=0)
    dict2 = dict2/np.linalg.norm(dict2, axis=0)

    dict1_interpolated, _ = generate_interpolated_Dictionary(dict1, numOfsubgrids)
    numOfinterp = int(dict1_interpolated.shape[1]/filternum)

    indices = np.zeros(filternum, dtype=int)
    for i in np.arange(filternum):
        diff = 1
        idx = 0
        for j in np.arange(numOfinterp):
            temp = []
            for o in offsets:
                temp.append(np.dot(np.roll(dict1_interpolated[:,i*numOfinterp+j],o),dict2[:,i]))

            temp_diff = 1-np.power(np.max(temp), 2)

            if temp_diff<diff:
                idx = j
                diff = temp_diff

        err_distance[i] = np.sqrt(diff)
        indices[i] = idx
    return err_distance, indices


def compute_hit_error(fs, code, event_indices, numOfelements, dictionary, slen):

    """
    Useful for simulation
    """

    numOfsamples = dictionary.shape[0]

    code_dense = np.zeros((numOfelements, slen))
    interp_factor = int(dictionary.shape[1]/numOfelements)

    hit_error = np.zeros(numOfelements)
    recovered_indices = {fidx:np.array([]) for fidx in np.arange(numOfelements)}

    for fidx in np.arange(numOfelements):
        if interp_factor ==1:
            adjusted_indices = code[fidx]['idx']-(numOfsamples-1)
            recovered_indices[fidx] = adjusted_indices*1/fs
        else:
            for idx in np.arange(interp_factor):
                if len(code[fidx*interp_factor + idx]['idx'])>0:
                    adjusted_indices = code[fidx*interp_factor + idx]['idx'] -(numOfsamples-1) + idx/interp_factor
                    recovered_indices[fidx] = np.append(recovered_indices[fidx], adjusted_indices*1/fs)
            recovered_indices[fidx] = np.sort(recovered_indices[fidx])
        # Mismatch (For now, just make it negative)
        if len(event_indices[fidx]) != len(recovered_indices[fidx]):
            print("Mismatch")
            hit_error[fidx] = -100
        else:
            err = np.median(np.abs(event_indices[fidx] - recovered_indices[fidx]))
            hit_error[fidx] = err

    return hit_error, recovered_indices

def estimateError(signal, channel, startidx, endidx):
    """
    Estimate the standard deviation of the given portion of the data

    Inputs
    ======
    signal:
    channel: channel of the signal to examine
    startidx: starting index of the signal, in terms of the index
    endidx: end index of the signal

    Outputs
    =======
    enorm: standard deviation of the signal
    """

    enorm = np.linalg.norm(signal[channel, startidx:endidx])/np.sqrt(endidx - startidx)
    return enorm


def matchTruth(true_timestamps, d, coeffs, segment_indices, offset, threshrange, polarity):

    """

    match the number of intracellular spikes with the identified spikes. Used for spiksorting application

    Inputs
    ======

    signal: (slen)x(numOfwindows) signal
    peakidx: peak indices of truth signal
    offset: range from the intracellular peak to search for peaks
    filters: dictionary
    coeffs: (clen x filternum)x(numOfwindows)
    threshrange: list of threshold values for false alarm

    Outputs
    =======

    icpeakloc: (signal length)x(num of windows) peak locations of intracellular signal
    truemiss: (num Of filters)x(num of thresholds) a list of true miss rates
    falsealarm: (num Of filters)x(num of thresholds) a list of false alarm rates
    nonzerocoeffs: (num Of filters)x(num of thresholds) number of nonzero coefficients
    match: (num of filters)x(num of thresholds) number of  matches
    """

    dlen, numOfelements = np.shape(d)

    threshlen = len(threshrange)
    truemiss = np.zeros((numOfelements, threshlen))
    falsealarm = np.zeros((numOfelements, threshlen))
    nonzerocoeffs = np.zeros((numOfelements, threshlen))
    match = np.zeros((numOfelements, threshlen))

    fa_coeffs = {t:{} for t in threshrange}
    true_coeffs = {t:{} for t in threshrange}
    print("Computing error statistics for threshold of {} ~ {} with interval {:.3f}".format(threshrange[0], threshrange[-1], threshrange[1]-threshrange[0]))
    for tidx, threshold in enumerate(tqdm(threshrange)):
        fa = fa_coeffs[threshold]

        thresholded_coeffs = convolve_threshold_all(d, coeffs, threshold, polarity)

        true_match = {true_idx:[] for true_idx in true_timestamps}
        numOfmatch = np.zeros(numOfelements, dtype=int)
        numOfnonzerocoeffs = np.zeros(numOfelements, dtype=int)

        for key in coeffs.keys():
            fa[key]={fidx:{'idx':np.array([], dtype=int), 'amp': np.array([])} for fidx in np.arange(numOfelements)}
            for fidx in np.arange(numOfelements):
                if len(coeffs[key][fidx]['idx'])>0:
                    coeffs_ts_start = segment_indices[key][0]

                    indices = thresholded_coeffs[key][fidx]['idx']
                    numOfnonzerocoeffs[fidx] += len(indices)

                    for idx_iter, idx_value in enumerate(indices):
                        timestamp = idx_value + coeffs_ts_start - (dlen-1)
                        match_ts = true_timestamps[(timestamp < true_timestamps) & (true_timestamps<timestamp+offset)]

                        if len(match_ts)>0:    # Corresponding intracellular exists
                            for elem in match_ts:
                                true_match[elem].append(fidx)
                        else:   # The code corresponds to the false alarm
                            fa[key][fidx]['idx'] = np.append(fa[key][fidx]['idx'], idx_value)
                            amp = thresholded_coeffs[key][fidx]['amp'][idx_iter]
                            fa[key][fidx]['amp'] = np.append(fa[key][fidx]['amp'], amp)

        for key, value in true_match.items():
            for fidx in np.arange(numOfelements):
                if fidx in value:
                    numOfmatch[fidx] += 1

        truemiss[:, tidx] = len(true_timestamps) - numOfmatch
        falsealarm[:, tidx] = numOfnonzerocoeffs - numOfmatch
        nonzerocoeffs[:, tidx] = numOfnonzerocoeffs
        match[:, tidx] = numOfmatch

        fa_coeffs[threshold] = fa
        true_coeffs[threshold] = true_match

    return truemiss, falsealarm, nonzerocoeffs, match, fa_coeffs, true_coeffs


def matchTruth_ell1(true_timestamps, d, coeffs, segment_indices, offset, threshrange):

    """

    match the number of intracellular spikes with the identified spikes for ADCG

    Inputs
    ======

    signal: (slen)x(numOfwindows) signal
    peakidx: peak indices of truth signal
    offset: range from the intracellular peak to search for peaks
    filters: dictionary
    coeffs: (clen x filternum)x(numOfwindows)
    threshrange: list of threshold values for false alarm

    Outputs
    =======

    icpeakloc: (signal length)x(num of windows) peak locations of intracellular signal
    truemiss: (num Of filters)x(num of thresholds) a list of true miss rates
    falsealarm: (num Of filters)x(num of thresholds) a list of false alarm rates
    nonzerocoeffs: (num Of filters)x(num of thresholds) number of nonzero coefficients
    match: (num of filters)x(num of thresholds) number of  matches
    """

    dlen, numOfelements = np.shape(d)

    threshlen = len(threshrange)
    truemiss = np.zeros((numOfelements, threshlen))
    falsealarm = np.zeros((numOfelements, threshlen))
    nonzerocoeffs = np.zeros((numOfelements, threshlen))
    match = np.zeros((numOfelements, threshlen))

    fa_coeffs = {t:{} for t in threshrange}
    true_coeffs = {t:{} for t in threshrange}

    print("Computing error statistics for threshold of {} ~ {} with interval {:.3f}".format(threshrange[0], threshrange[-1], threshrange[1]-threshrange[0]))
    for tidx, threshold in enumerate(tqdm(threshrange)):

        fa = fa_coeffs[threshold]

        true_match = {true_idx:[] for true_idx in true_timestamps}
        numOfmatch = np.zeros(numOfelements, dtype=int)
        numOfnonzerocoeffs = np.zeros(numOfelements, dtype=int)

        # Iterate through data
        for key in coeffs.keys():
            coeffs_seg = coeffs[key]
            thresholded_coeffs={fidx:{} for fidx in range(numOfelements)}
            for fidx in range(numOfelements):
                if len(coeffs_seg[fidx]['amp'])>0:
                    indices = np.where(coeffs_seg[fidx]['amp'] * np.min(d[:, fidx]) < threshold)[0]
                else:
                    indices = []

                if len(indices)>0:
                    thresholded_coeffs[fidx]['idx'] = coeffs_seg[fidx]['idx'][indices]
                    thresholded_coeffs[fidx]['amp'] = coeffs_seg[fidx]['amp'][indices]
                else:
                    thresholded_coeffs[fidx]['idx'] = []
                    thresholded_coeffs[fidx]['amp'] = []

            fa[key]={fidx:{'idx':np.array([], dtype=int), 'amp': np.array([])} for fidx in np.arange(numOfelements)}
            for fidx in np.arange(numOfelements):
                if len(coeffs_seg[fidx]['idx'])>0:
                    # Need to align the local segment indices with the global indices
                    coeffs_ts_start = segment_indices[key][0]

                    indices = thresholded_coeffs[fidx]['idx']
                    numOfnonzerocoeffs[fidx] += len(indices)

                    for idx_iter, idx_value in enumerate(indices):
                        timestamp = idx_value + coeffs_ts_start
                        match_ts = true_timestamps[(timestamp < true_timestamps) & (true_timestamps<timestamp+offset)]

                        if len(match_ts)>0:    # Corresponding intracellular exists
                            for elem in match_ts:
                                true_match[elem].append(fidx)
                        else:   # The code corresponds to the false alarm
                            fa[key][fidx]['idx'] = np.append(fa[key][fidx]['idx'], idx_value)
                            amp = thresholded_coeffs[fidx]['amp'][idx_iter]
                            fa[key][fidx]['amp'] = np.append(fa[key][fidx]['amp'], amp)

        for key, value in true_match.items():
            for fidx in np.arange(numOfelements):
                if fidx in value:
                    numOfmatch[fidx] += 1

        truemiss[:, tidx] = len(true_timestamps) - numOfmatch
        falsealarm[:, tidx] = (numOfnonzerocoeffs - numOfmatch)/numOfnonzerocoeffs
        nonzerocoeffs[:, tidx] = numOfnonzerocoeffs
        match[:, tidx] = numOfmatch

        fa_coeffs[threshold] = fa
        true_coeffs[threshold] = true_match

    return truemiss, falsealarm, nonzerocoeffs, match, fa_coeffs, true_coeffs
