"""
Copyright (c) 2020 CRISP

Convolutional Orthogonal Matching Pursuit

:author: Andrew H. Song
"""

import numpy as np
import sys
import pickle
import os
import scipy.signal

# Append path
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

from . import BaseCSC
from src.helpers.convolution import sc_cholesky, sc_cholesky_efficient

class COMP(BaseCSC):
	def __init__(self, dlen, error_tol, sparsity_tol, pflag):
		BaseCSC.__init__(self, dlen, error_tol, sparsity_tol, pflag)

	def terminate(self, numOfiter, numOfmaxcoeffs, err_residual, err_bound):
		return (err_residual< err_bound) or (numOfiter >= numOfmaxcoeffs)

	def extractCode_seg(self, y_seg, dictionary, sparsity=None, err=None, boundary=1):

		"""
		Given data segment, extract convolutional codes

		Inputs
		======
		y_seg: A segment of data
		boundary: boolean (default=1)
			If 1, accounts for truncated templates as well (clen = slen + dlen - 1)
			If 0, accounts for whole templates only (cldn = slen - dlen + 1)

		"""
		assert(len(np.where(abs(np.linalg.norm(dictionary,axis=0)-1)>1e-6)[0]) == 0), "Not normalized"

		numOfsamples, numOfelements = dictionary.shape

		slen = len(y_seg)
		clen = slen + numOfsamples - 1
		coeffs = np.zeros(clen*numOfelements)

		numOfmaxcoeffs = self.sparsity_tol if sparsity is None else sparsity
		err_bound = self.error_tol if err is None else err

		chosen_vals = np.zeros(numOfelements)
		chosen_idx = np.zeros(numOfelements, dtype=np.int)

		residual = y_seg
		err_residual = np.linalg.norm(residual)/np.sqrt(np.size(residual))

		# Dictionary to collect expanding set of dictionary
		temp_idx = np.zeros(numOfmaxcoeffs, dtype=np.int)
		dictionary_active = np.zeros((slen, numOfmaxcoeffs))

		iternum = 0
		lower_mat = [1]

		while not self.terminate(iternum, numOfmaxcoeffs, err_residual, err_bound):
			#######################
			# Selection step
			#
			# This step can be fruther sped up with careful book-keeping of residuals
			#######################
			for idx in np.arange(numOfelements):
				d = dictionary[:, idx]
				cross = abs(scipy.signal.correlate(residual, d, mode='full'))

				if boundary:
					t_start = int(np.floor(numOfsamples/2))
					t_end = slen + t_start
				else:
					t_start = numOfsamples-1
					t_end = slen - t_start

				cross = cross/self.computeNorm(d, slen)
				m = np.argmax(cross[t_start:t_end]) + t_start

				chosen_idx[idx] = m
				chosen_vals[idx] = cross[m]

			filter_idx = np.argmax(chosen_vals) # Returns the filter with the highest inner product
			coeff_idx = chosen_idx[filter_idx] # index within the chosen filter

			#######################
			# Projection step
			#######################

			# placeholder for coefficients
			temp_idx[iternum] = filter_idx*clen + coeff_idx

			if coeff_idx < numOfsamples-1:	# Boundary condition
				offset = coeff_idx + 1
				elem = dictionary[-offset:,filter_idx]/np.linalg.norm(dictionary[-offset:,filter_idx])
				elem = np.pad(elem, (0, slen-len(elem)), 'constant')
			elif coeff_idx > slen - 1:	# Boundary condition
				offset = numOfsamples-(coeff_idx-(slen-1))
				elem = dictionary[:offset,filter_idx]/np.linalg.norm(dictionary[:offset,filter_idx])
				elem = np.pad(elem, (slen-len(elem), 0), 'constant')
			else:	# Valid correlation. Entire support of the dictionary lies within the signal
				start_idx = coeff_idx - (numOfsamples-1)
				elem = dictionary[:, filter_idx]/np.linalg.norm(dictionary[:, filter_idx])
				elem = np.pad(elem, (start_idx, slen - numOfsamples - start_idx), 'constant')

			dictionary_active[:, iternum] = elem
			[lower_mat, sparse_code] = sc_cholesky(lower_mat, dictionary_active[:, :iternum], elem, y_seg)

			residual = y_seg - np.matmul(dictionary_active[:, :iternum+1], sparse_code)
			coeffs[temp_idx[:iternum+1]] = sparse_code

			iternum += 1

		err_residual = np.linalg.norm(residual)
		return coeffs, err_residual

	def extractCode_seg_eff(self, y_seg, dictionary, sparsity=None, err=None, boundary=1):

		"""
		Given data segment, extract convolutional codes
		Uses efficient implementation of cholesky decomposition, which should be much faster for long signals

		Inputs
		======
		y_seg: A segment of data
		boundary: boolean (default=1)
			If 1, accounts for truncated templates as well (clen = slen + dlen - 1)
			If 0, accounts for whole templates only (cldn = slen - dlen + 1)
		"""
		assert(len(np.where(abs(np.linalg.norm(dictionary,axis=0)-1)>1e-6)[0]) == 0), "Not normalized"

		numOfsamples, numOfelements = dictionary.shape

		slen = len(y_seg)
		clen = slen + numOfsamples - 1
		coeffs = np.zeros(clen*numOfelements)

		numOfmaxcoeffs = self.sparsity_tol if sparsity is None else sparsity
		err_bound = self.error_tol if err is None else err

		chosen_vals = np.zeros(numOfelements)
		chosen_idx = np.zeros(numOfelements, dtype=np.int)

		residual = y_seg
		err_residual = np.linalg.norm(residual)/np.sqrt(np.size(residual))

		# Dictionary to collect expanding set of dictionary
		temp_idx = np.zeros(self.sparsity_tol, dtype=np.int)

		# Active index sets
		filter_indices = []
		start_indices = []

		iternum = 0
		lower_mat = [1]

		while not self.terminate(iternum, numOfmaxcoeffs, err_residual, err_bound):
			#######################
			# Selection step
			#
			# This step can be fruther sped up with careful book-keeping of residuals
			#######################
			for idx in np.arange(numOfelements):
				d = dictionary[:, idx]
				cross = abs(scipy.signal.correlate(residual, d, mode='full'))

				if boundary:
					t_start =int(np.floor(numOfsamples/2))
					t_end = slen + t_start
				else:
					t_start = numOfsamples-1
					t_end = slen - t_start

				cross = cross/self.computeNorm(d, slen)

				m = np.argmax(cross[t_start:t_end]) + t_start

				chosen_idx[idx] = m
				chosen_vals[idx] = cross[m]

			filter_idx = np.argmax(chosen_vals) # Returns the filter with the highest inner product
			filter_indices.append(filter_idx)
			coeff_idx = chosen_idx[filter_idx] # index within the chosen filter

			#######################
			# Projection step
			#######################

			# placeholder for coefficients
			temp_idx[iternum] = filter_idx*clen + coeff_idx

			# Pad the dictionary element to match the signal length
			start_idx = coeff_idx - numOfsamples + 1
			start_indices.append(start_idx)

			[lower_mat, sparse_code] = sc_cholesky_efficient(lower_mat, dictionary, start_indices, filter_indices, y_seg)

			residual = np.copy(y_seg)
			for i, filter_idx in enumerate(filter_indices):
				elem = dictionary[:, filter_idx]
				start_idx = start_indices[i]
				coeff = sparse_code[i]

				if start_idx < 0:
					residual[:numOfsamples+start_idx] -= coeff * elem[-(numOfsamples+start_idx):]
				elif start_idx > slen - numOfsamples:
					residual[start_idx:] -= coeff * elem[:slen - start_idx]
				else:
					residual[start_idx:start_idx + numOfsamples] -= coeff*elem

			coeffs[temp_idx[:iternum+1]] = sparse_code

			iternum += 1

		err_residual = np.linalg.norm(residual)
		return coeffs, err_residual
