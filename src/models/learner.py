"""
Copyright (c) 2020 CRISP

The main class - convolutional learner

:author: Andrew H. Song
"""

import numpy as np
import sys
import pickle
import os
import time

# Append path
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

from src.helpers.evaluate import *
from . import generateCSC, generateCDL
from src.generators.generate import generate_interpolated_Dictionary

class learner:
	def __init__(self, dlen, numOfelements, error_tol, sparsity_tol, pflag, csc_type, cdl_type):
		self.error_tol = error_tol
		self.sparsity_tol = sparsity_tol
		self.pflag = pflag
		self.csc = generateCSC(dlen, error_tol, sparsity_tol, pflag, csc_type)
		self.cdl = generateCDL(dlen, numOfelements, cdl_type)
		self.d_distance = None
		self.init_d = None

	def getDictionary(self):
		return self.d

	def initializeDictionary(self, d):
		self.init_d = d

	def setDictionary(self, d):
		self.d = d

	def train(self, y_train, hyp, indices=None):
		assert(self.init_d is not None), "Dictionary not initialized"
		d = np.copy(self.init_d)

		numOfsubgrids = hyp['interpolate']
		boundary = hyp['boundary']
		numOfiterations = hyp['numOfiterations']

		self.d_distance = np.zeros((d.shape[1], numOfiterations))

		for i in np.arange(numOfiterations):
			print("=========================================")
			print("Iteration number {}/{}".format(i+1, numOfiterations))

			d_interpolated, interpolator = generate_interpolated_Dictionary(d, numOfsubgrids, kind='cubic')
			s= time.time()

			coeffs = self.csc.extractCode(y_train, d_interpolated, indices, boundary)

			print("Elapsed time {:.4f} seconds".format(time.time()-s))

			d, _, _, _ = self.cdl.updateDictionary(y_train, d, coeffs, interpolator)

			error = recovery_error(self.init_d, d)
			self.d_distance[:, i] = error
			print("Dictionary error ", error)

		print("Finished training")
		return d


	def train_and_save(self, y_train, y_train_idx, path, hyp, indices=None):

		"""
		Train and save
		"""

		d_updated = self.train(y_train, hyp, indices)
		filename = os.path.join(path, 'results','results_train')

		results={}
		results['init_d'] = self.init_d
		results['d'] = d_updated
		results['d_distance'] = self.d_distance
		results['segment_indices'] = y_train_idx
		results['interpolate'] = hyp['interpolate']

		with open(filename,'wb') as f:
			pickle.dump(results, f)

	def predict(self, y_test, hyp, indices, error_tol=None, sparsity_tol=None):
		"""
		Based on the learned dictionary, estimate the sparse codes
		Allows for re-configuring error_tol and sparsity_tol

		Inputs
		======
		y_test: test data
		error_tol: Error tolerance. If not specified, use error tolerance used in training
		sparsity_tol: Sparsity tolerance, If not specified, usee sparsity tolerance used in testing

		Outputs
		=======
		code: Sparse codes for the test data

		"""

		numOfsubgrids = hyp['interpolate']
		boundary = hyp['boundary']

		d_interpolated, interpolator = generate_interpolated_Dictionary(self.d, numOfsubgrids, kind='cubic')

		code = self.csc.extractCode(y_test, d_interpolated, indices, boundary)
		print("Finished prediction")
		return code

	def predict_and_save(self, y_test, y_test_idx, path, hyp, indices=None):
		"""
		Run prediction and save the results

		"""

		code = self.predict(y_test, hyp, indices)
		filename = os.path.join(path, 'results','results_test')

		results={}
		results['code'] = code
		results['segment_indices'] = y_test_idx

		f = open(filename,'wb')
		pickle.dump(results, f)
		f.close()
