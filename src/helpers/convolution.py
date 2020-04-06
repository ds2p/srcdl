"""
Copyright (c) 2020 CRISP

Helpers for convolution related functionalities

:author: Andrew H. Song
"""

import numpy as np
import scipy

def sc_cholesky(lower, d, newelem, sig):

    """
    Efficient implementation of the least squares step to compute sparse code using Cholesky decomposition

    Inputs
    ======

    lower: lower triangular matrix
    d: current dictionary (set of regressors)
    newelem: new dictionary element to be added
    sig: signal to regress against

    Outputs
    =======

    lower_new: newly computed lower triangluar matrix
    sparse_code: sparse code for the newly updated dictionary

    """

    dim = np.shape(lower)[0]

    if d.size==0:
        lower_new = lower # assuming lower is just 1
        sparse_code = [np.matmul(np.transpose(newelem), sig)]
    else:
        if dim ==1:
            lower = [lower]

        temp = np.matmul(np.transpose(d), newelem)
        tempvec = scipy.linalg.solve_triangular(lower, temp, lower = True)

        ###################################
        # Construct lower triangular matrix
        ###################################
        lower_new = np.zeros((dim + 1, dim + 1))
        lower_new[:dim, :dim] = lower
        lower_new[dim, :dim] = np.transpose(tempvec)
        lower_new[dim, dim] = np.sqrt(1 - np.matmul(np.transpose(tempvec), tempvec))

        d_new = np.zeros((d.shape[0], d.shape[1]+1))
        d_new[:,:d.shape[1]] = d
        d_new[:, -1] = newelem

        temp = np.matmul(np.transpose(d_new), sig)
        sparse_code = scipy.linalg.cho_solve([lower_new, 1], temp)

    return lower_new, sparse_code

def sc_cholesky_efficient(lower, dictionary, start_indices, filter_indices, sig):

    """

    Efficient implementation of the least squares step to compute sparse code using Cholesky decomposition.
    This version utilizes the convolutional nature of the matrix to speed things up and hence faster than "sc_cholesky"
    For small signal length, won't likely see any speedups. Effective for longer signal

    Inputs
    ======
    lower: lower triangular matrix
    dictionary: Dictionary of atoms
    start_indices: Active index set of first indices (non-zero filter values) for the selected atoms
    filter_indices: Active index set of the selected atoms
    sig: signal

    Outputs
    =======
    lower_new: newly computed lower triangluar matrix
    sparse_code: sparse code for the newly updated dictionary

    """

    dim = np.shape(lower)[0]
    slen = len(sig)
    numOfsamples = dictionary.shape[0]
    numOfactives = len(start_indices)

    def normalize_atom(elem, slen, start_idx):
        numOfsamples = len(elem)
        if start_idx <0:
            elem_normalized = elem/np.linalg.norm(elem[-start_idx:])
        elif start_idx > slen - numOfsamples:
            elem_normalized = elem/np.linalg.norm(elem[:slen-start_idx])
        else:
            elem_normalized = elem

        return elem_normalized

    def dot_product(elem, y, start_idx):
        slen = len(y)
        numOfsamples = len(elem)

        if start_idx < 0:
            result = np.dot(elem[-(numOfsamples+start_idx): ], y[:numOfsamples + start_idx])
        elif start_idx > slen - numOfsamples:
            result = np.dot(elem[:slen-start_idx], y[start_idx:])
        else:
            result = np.dot(elem, y[start_idx:start_idx + numOfsamples])
        return result

    if len(start_indices)==1:
        lower_new = lower # assuming lower is just 1
        elem = dictionary[:, filter_indices[0]]
        start_idx = start_indices[0]
        elem_normalized = normalize_atom(elem, slen, start_idx)
        sparse_code = [dot_product(elem_normalized, sig, start_idx)]
    else:
        if dim ==1:
            lower = [lower]

        elem_new = dictionary[:, filter_indices[-1]]
        start_idx_new = start_indices[-1]
        elem_new_normalized = normalize_atom(elem_new, slen, start_idx_new)

        v = np.zeros(numOfactives-1)
        cho_temp = np.zeros(numOfactives)

        for idx, start_idx in enumerate(start_indices[:-1]):
            elem = dictionary[:, filter_indices[idx]]
            elem_normalized = normalize_atom(elem, slen, start_idx)

            offset = start_idx - start_idx_new

            if abs(offset) < numOfsamples: # the filter do not overlap
                if offset > 0:
                    if start_idx_new > slen - numOfsamples:
                        v[idx] = np.dot(elem_new_normalized[offset: offset + slen - start_idx], elem_normalized[: slen - start_idx])
                    elif start_idx < 0:
                        v[idx] = np.dot(elem_new_normalized[-(numOfsamples+start_idx_new):], elem_normalized[-(numOfsamples+start_idx): start_idx_new - start_idx])
                    else:
                        v[idx] = np.dot(elem_new_normalized[offset:], elem_normalized[: numOfsamples - offset])
                else:
                    if start_idx > slen - numOfsamples:
                        v[idx] = np.dot(elem_new_normalized[: slen - start_idx_new], elem_normalized[-offset: -offset + slen -start_idx_new])
                    elif start_idx_new < 0:
                        v[idx] = np.dot(elem_new_normalized[-(numOfsamples+start_idx_new):start_idx-start_idx_new], elem_normalized[-(numOfsamples+start_idx):])
                    else:
                        v[idx] = np.dot(elem_new_normalized[: numOfsamples + offset], elem_normalized[-offset:])

            cho_temp[idx] = dot_product(elem_normalized, sig, start_idx)

        cho_temp[-1] = dot_product(elem_new_normalized, sig, start_idx_new)
        w = scipy.linalg.solve_triangular(lower, v, lower = True)

        ###################################
        # Construct lower triangular matrix
        ###################################
        lower_new = np.zeros((dim + 1, dim + 1))
        lower_new[:dim, :dim] = lower
        lower_new[dim, :dim] = np.transpose(w)
        lower_new[dim, dim] = np.sqrt(1 - np.linalg.norm(w)**2)

        sparse_code = scipy.linalg.cho_solve([lower_new, 1], cho_temp)

    return lower_new, sparse_code


def getSignalIndices(dlen, indices):
    """
    Extract the signal for which the corresponding coefficients are non-zero

    """

    arrindices = np.zeros(dlen*np.size(indices), dtype=int)
    for i, value in enumerate(indices):
        arrindices[i*dlen:(i+1)*dlen] = np.arange(value, value+dlen)

    return arrindices

def code_sparse(dense_coeffs, numOfelements):
    """
    Sparse representation of the dense coeffs

    Inputs
    ======
    dense_coeffs: array_like. (numOfelements * clen)
        This array contains many zero elements

    Outputs
    =======
    sparse_coeffs: dictionary
        Each key represents a filter
        The corresponding value is a 2-D array
            First row: Nonzero indices
            Second row: Ampltiudes corresponding to nonzero indices
    """

    sparse_coeffs={}
    clen = int(len(dense_coeffs)/numOfelements)
    for fidx in np.arange(numOfelements):
        indices = np.nonzero(dense_coeffs[fidx*clen:(fidx+1)*clen])[0]

        temp = {}
        # If no nonzero components
        if len(indices)==0:
            temp['idx'] = np.array([], dtype=int)
            temp['amp'] = np.array([])
            sparse_coeffs[fidx] = temp
        else:
            temp['idx'] = indices
            temp['amp'] = dense_coeffs[indices + fidx*clen]
            sparse_coeffs[fidx] = temp

    return sparse_coeffs

def code_dense(sparse_coeffs, clen):
    """
    Dense representation of the coeffs

    Inputs
    =======
    sparse_coeffs: code in sparse format

    Outputs
    ======
    dense_coeffs: array_like. (numOfelements * clen)
        This array contains many zero elements

    """

    numOfelements = len(sparse_coeffs.keys())
    dense_coeffs = np.zeros(clen*numOfelements)
    for fidx in np.arange(numOfelements):
        indices = sparse_coeffs[fidx]['idx']

        if len(indices)>0:
            amps = sparse_coeffs[fidx]['amp']
            dense_coeffs[fidx*clen+indices] = amps
    return dense_coeffs


def reconstruct(d, sparse_coeffs, slen):
    """
    Reconstruct the signal given the dictionary and codes (sparse format)

    Output
    ======
    reconstructed: dictionary
        Each key corresponds to a filter and the value to a reconstructed signal array

    recontstructed_data: array
        Reconstructed signal

    """
    if len(d.shape)==1:
        dlen = len(d)
        d = d.reshape((dlen,1))
        numOfelements = 1
    else:
        dlen, numOfelements = d.shape

    reconstructed = {}
    reconstructed_data = np.zeros(slen)
    for fidx in np.arange(numOfelements):

        clen = slen+dlen-1
        dense_coeffs = code_dense(sparse_coeffs, clen)

        convolved_sig = convolve_from_start(d[:,fidx], dense_coeffs[fidx*clen:(fidx+1)*clen])

        reconstructed[fidx] = convolved_sig[dlen-1:]
        reconstructed_data += convolved_sig[dlen-1:]

    return reconstructed, reconstructed_data

## Class of methods for thresholded convolution
def convolve_threshold(d, coeffs, threshold, polarity):
    """
    Identify the coefficients resulting in convolved signal crossing the given threshold (for a single window)

    """

    numOfelements = d.shape[1]

    thresholded_coeffs={}
    for fidx in np.arange(numOfelements):
        temp = {'idx':[], 'amp':[]}

        element = d[:, fidx]
        temp_coeffs = coeffs[fidx]

        if len(temp_coeffs['idx'])>0:
            for idx, idx_value in enumerate(temp_coeffs['idx']):
                amp = temp_coeffs['amp'][idx]
                convolved_signal = element*amp
                if polarity==0:
                    if min(convolved_signal)<threshold or max(convolved_signal)>-threshold:
                        temp['idx'].append(idx_value)
                        temp['amp'].append(amp)
                else:
                    if max(convolved_signal)>threshold or min(convolved_signal)<-threshold:
                        temp['idx'].append(idx_value)
                        temp['amp'].append(amp)

        temp['idx'] = np.array(temp['idx'])
        temp['amp'] = np.array(temp['amp'])

        thresholded_coeffs[fidx] = temp

    return thresholded_coeffs

def convolve_threshold_all(d, coeffs, threshold, polarity):
    """
    Identify the coefficients resulting in convolved signal crossing the given threshold (for multiple windows)

    Inputs
    ======
    polarity: 0 indicates below threshold. 1 indicates above threshold.

    Outputs
    =======
    thresholded_coeffs: dictionary
        thresholded coefficients in sparse format
    """

    # Normalize the dictionary
    normalized_d = d/np.linalg.norm(d,axis=0)

    threshold_coeffs = {key:{} for key in coeffs.keys()}

    for key in coeffs.keys():
        threshold_coeffs[key] = convolve_threshold(normalized_d, coeffs[key], threshold, polarity)

    return threshold_coeffs

def convolve_threshold_all_interp(d, coeffs, threshold, polarity, indices):
    """
    Identify the coefficients resulting in convolved signal crossing the given threshold (for multiple windows)
    For two dictionaries

    Inputs
    ======
    polarity: 0 indicates below threshold. 1 indicates above threshold.
    indices: binary vector, denotes different dictionary to be used

    Outputs
    =======
    thresholded_coeffs: dictionary
        thresholded coefficients in sparse format
    """

    from src.generators.generate import generate_Dictionary

    normalized_d = d/np.linalg.norm(d,axis=0)

    # For now interp 10
    interpolate =10

    interval = 1/int(interpolate)
    delay_arr = np.arange(interval, 1, interval)
    d_interpolated = generate_Dictionary(d, delay_arr)

    normalized_d_interpolated = d_interpolated/np.linalg.norm(d_interpolated,axis=0)

    threshold_coeffs = {key:{} for key in coeffs.keys()}

    for key in coeffs.keys():
        # print("KEY ",key)
        if indices[key]==0:
            threshold_coeffs[key] = convolve_threshold(normalized_d_interpolated, coeffs[key], threshold, polarity)
        else:
            # print("Greedy ", normalized_d.shape)
            threshold_coeffs[key] = convolve_threshold(normalized_d, coeffs[key], threshold, polarity)

    return threshold_coeffs
