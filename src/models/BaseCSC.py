"""
Copyright (c) 2020 CRISP

The abstract parent class for Convolutional Sparse Coder

:author: Andrew H. Song
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import sys
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

from src.helpers.convolution import code_sparse

from tqdm import tqdm
from dask import delayed
from dask import compute
import dask.bag as db

class BaseCSC(metaclass=ABCMeta):
	def __init__(self, dlen, error_tol, sparsity_tol, pflag):
		self.dlen = dlen
		self.error_tol = error_tol
		self.sparsity_tol = sparsity_tol
		self.pflag = pflag

	def set_error(self, error):
		self.error_tol = error

	def set_sparsity(self, sparsity):
		self.sparsity_tol = sparsity

	def computeNorm(self, delem, slen):
		"""
		Compute norm of the all possible timeshifts of the dictionary

		Inputs
		======
		delem: array-like. dictionary element

		"""
		numOfsamples = delem.shape[0]
		clen = slen + numOfsamples - 1
		norms = np.zeros(clen)
		for idx in np.arange(clen):
			if idx<numOfsamples-1:
				norms[idx] = np.linalg.norm(delem[-(idx+1):],2)
			elif idx>slen-1:
				dlen = numOfsamples-(idx-(slen-1))
				norms[idx] = np.linalg.norm(delem[:dlen],2)
			else:
				norms[idx] = 1

		return norms

	def extractCode(self, y_seg_set, d, indices, boundary=1):
		"""
		Extract the sparse codes from the data.
		Intended to give more flexbility over various error/sparsity thresholds

		Inputs
		======
		y_seg_set:
			A set of segmented data.
			These either can be equal or different lengths
		indices:
			indicates which group the segment is associated with.
			sparsity level will be different for each segment
		"""

		numOfelements = d.shape[1]
		coeffs = {}

		if self.pflag:	# Parallel implementation via Dask
			output = []
			for k, y_seg in tqdm(y_seg_set.items()):
				if indices is None:
					a = delayed(self.extractCode_seg_eff)(y_seg, d, sparsity=None, boundary=boundary)
				else:
					a = delayed(self.extractCode_seg_eff)(y_seg, d, sparsity=indices[k], boundary=boundary)
				output.append(a)
			o = compute(*(output))
			coeffs = {i:code_sparse(o[i], numOfelements) for i in np.arange(np.shape(o)[0])}
		else:	# Sequential implementation
			for k, y_seg in tqdm(y_seg_set.items()):
				if indices is None:
					c, _ = self.extractCode_seg(y_seg, d, sparsity=None, boundary=boundary)
				else:
					c, _ = self.extractCode_seg(y_seg, d, sparsity=indices[k], boundary=boundary)
				sparse_c = code_sparse(c, numOfelements)
				coeffs[k] = sparse_c
		return coeffs


	@abstractmethod
	def extractCode_seg(self, y_seg, dictionary):
		pass
