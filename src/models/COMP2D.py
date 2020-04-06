"""
Copyright (c) 2020 CRISP

2D implementation of Convolutional Orthogonal Matching Pursuit

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
from src.helpers.convolution import *


class COMP2D(BaseCSC):
	def __init__(self, dlen, numOfelements, error_tol, sparsity_tol, pflag):
		BaseCSC.__init__(self, dlen, numOfelements, error_tol, sparsity_tol, pflag)

	def terminate(self, numOfiter, numOfmaxcoeffs, err_residual, err_bound):
		return (err_residual< err_bound) or (numOfiter >= numOfmaxcoeffs)

	def extractCode_seg(self, y_seg, dictionary, sparsity=None, err=None, debug=0):

		"""
		Given data segment, extract convolutional codes in 2D.
		# Each code corresponds to the mean of the kernel
		For now, won't worry about the boundary conditions
		(Technically CMP for now)

		Inputs
		======
		y_seg: A segment of data
		boundary: boolean (default=1)
			If 1, accounts for truncated templates as well (clen = slen + dlen - 1)
			If 0, accounts for whole templates only (cldn = slen - dlen + 1)

		"""
		# assert(len(np.where(abs(np.linalg.norm(dictionary,axis=0)-1)>1e-6)[0]) == 0), "Not normalized"
		assert dictionary.shape[0]==dictionary.shape[1], "Square kernel required"
		assert y_seg.shape[0]==y_seg.shape[1], "Square data input required"

		numOfsamples, _, numOfelements = dictionary.shape

		slen = y_seg.shape[0]
		clen = slen - numOfsamples + 1

		coeffs = np.zeros((clen, clen, numOfelements))

		numOfmaxcoeffs = self.sparsity_tol if sparsity is None else sparsity
		err_bound = self.error_tol if err is None else err

		chosen_vals = np.zeros(numOfelements)
		chosen_idx = np.zeros(numOfelements, dtype=np.int)

		residual = np.copy(y_seg)
		err_residual = np.linalg.norm(residual)/np.sqrt(np.size(residual))
		prev_err_residual = err_residual
		if debug:
			print("Initial residual ", err_residual)

		# Dictionary to collect expanding set of dictionary
		temp_idx = np.zeros(numOfmaxcoeffs, dtype=np.int)
		dictionary_active = np.zeros((slen, numOfmaxcoeffs))

		iternum = 0
		lower_mat = [1]

		while not self.terminate(iternum, numOfmaxcoeffs, err_residual, err_bound):
			if debug:
				print("--------------")
				print("Iter ",iternum)

			for idx in np.arange(numOfelements):
				d = dictionary[:, :, idx]

				cross = abs(scipy.signal.correlate2d(residual, d, mode='valid'))
				m = np.argmax(cross)

				chosen_idx[idx] = m
				chosen_vals[idx] = cross[np.unravel_index(m, (clen, clen))]

			filter_idx = np.argmax(chosen_vals) # Returns the filter with the highest inner product
			coeff_idx = np.unravel_index(chosen_idx[filter_idx], (clen, clen)) # index within the chosen filter

			template = dictionary[:, :, filter_idx]
			coeff = chosen_vals[filter_idx]

			prev_err_residual = np.linalg.norm(residual)/np.sqrt(np.size(residual))

			temp_residual = np.copy(residual)
			temp_residual[coeff_idx[0] : coeff_idx[0] + numOfsamples, coeff_idx[1] : coeff_idx[1] + numOfsamples] -= template * coeff

			if np.linalg.norm(temp_residual)/np.sqrt(np.size(temp_residual)) > prev_err_residual:
				if debug:
					print("The error residual increased! Terminating...")
				break

			coeffs[coeff_idx[0], coeff_idx[1], filter_idx] = coeff
			residual[coeff_idx[0] : coeff_idx[0] + numOfsamples, coeff_idx[1] : coeff_idx[1] + numOfsamples] -= template * coeff
			err_residual = np.linalg.norm(residual)/np.sqrt(np.size(residual))

			if debug:
				print("filter {} coeff loc {}, {} amp {} res {}".format(filter_idx, coeff_idx[0], coeff_idx[1], coeff, err_residual))
			iternum += 1

		err_residual = np.linalg.norm(residual)
		return coeffs, err_residual
