"""
Copyright (c) 2020 CRISP

Convolutional K-SVD for 2D SMLM. Assume single filter

:author: Andrew H. Song
"""

from . import BaseCDL

import numpy as np
from src.helpers.convolution import *

from scipy.sparse.linalg import svds
import scipy.linalg


class CKSVD2D(BaseCDL):
	def __init__(self, dlen, numOfelements):
		BaseCDL.__init__(self, dlen, numOfelements)

	def updateDictionary(self, y_seg_set, d, coeffs, numOfiterations=1):
		"""
		Shift invariant dictionary implementation with interpolation.
		For now, we ignore the templates on the boundary

		Inputs
		================
		interpolator: Dictionary
			Key corresponds to fractional delay. Value corresponds to the shifted interpolator function
		"""

		from scipy.interpolate import griddata

		assert(len(y_seg_set.keys())==len(coeffs.keys())), "The dimension of data and coeff need to match"

		d_updated = np.copy(d)

		numOfelements = d.shape[2]
		numOfinterp = int(np.sqrt(numOfelements))

		y_extracted_set = np.array([]).reshape(self.dlen, self.dlen,-1)
		coeffs_set = []
		delay_set = np.array([]).reshape(2,-1)

		# Collecting the extracted data segments
		for key, y_seg in y_seg_set.items():
			slen = y_seg.shape[0]
			clen = slen + self.dlen - 1

			x, y, delay_indices = np.where(coeffs[key]!=0)
			for x_loc, y_loc, delay_idx in zip(x,y,delay_indices):
				data = y_seg[x_loc:x_loc + self.dlen, y_loc:y_loc + self.dlen]
				data = data[..., np.newaxis]

				y_extracted_set = np.append(y_extracted_set, data, axis=2)
				delay_set = np.append(delay_set, np.array(np.unravel_index(delay_idx, (numOfinterp, numOfinterp))).reshape(2,-1), axis=1)
				coeffs_set = np.append(coeffs_set, coeffs[key][x_loc, y_loc, delay_idx])

		# Update the dictionary
		denominator = 0
		numerator = np.zeros((self.dlen, self.dlen))
		y_extracted_shift_set = np.array([]).reshape(self.dlen, self.dlen,-1)

		for idx in range(len(coeffs_set)):
			y_extracted = y_extracted_set[..., idx]
			x_delay, y_delay = delay_set[..., idx]

			grid_original = [(i,j) for i in np.arange(self.dlen) for j in np.arange(self.dlen)]

			if x_delay ==0:
				x_grid = np.arange(self.dlen)
			else:
				x_grid = x_delay/numOfinterp + np.arange(self.dlen-1)
				x_grid = np.append(x_grid, 0)

			if y_delay ==0:
				y_grid = np.arange(self.dlen)
			else:
				y_grid = y_delay/numOfinterp + np.arange(self.dlen-1)
				y_grid = np.append(y_grid, 0)

			grid_new = [(i,j) for i in x_grid for j in y_grid]

			y_extracted_shift = griddata(grid_original, y_extracted.reshape(1,-1).flatten(), grid_new, method='cubic')
			y_extracted_shift = y_extracted_shift.reshape(self.dlen, self.dlen)

			y_extracted_shift_set = np.append(y_extracted_shift_set, y_extracted_shift[...,np.newaxis], axis=2)

			coeff = coeffs_set[idx]
			denominator += coeff**2
			numerator += coeff * y_extracted_shift

		d_updated = numerator/denominator
		d_updated = d_updated/np.linalg.norm(d_updated)
		return d_updated[..., np.newaxis], y_extracted_set, y_extracted_shift_set


	def compute_interp_matrix(self, interpolator, dlen):
		"""
		For no interpolator case, the result should just be an identity matrix

		Inputs
		======
		interpolator: array

		"""
		interplen = len(interpolator)
		assert np.mod(interplen,2)==1, "Interpolator legnth must be odd"
		assert interplen<=dlen, "Interpolator length must be less than dictionary template length"

		interpolator_flipped = np.flip(interpolator, axis=0)

		start_clip = int((dlen-1)/2)
		end_clip = start_clip + dlen
		mtx = np.zeros((dlen, 2*dlen-1))

		for idx in np.arange(dlen):
			start_idx = start_clip+idx-int(interplen/2)
			end_idx = start_idx + interplen
			mtx[idx, start_idx : end_idx] = interpolator_flipped

		shift_mat = mtx[:, start_clip:end_clip]

		return shift_mat

	def compute_diff_matrix(self, interpolator_i, interpolator_j, i, j, dlen):
		mtx_i = self.compute_interp_matrix(interpolator_i, dlen)
		mtx_j = self.compute_interp_matrix(interpolator_j, dlen)

		interplen = len(interpolator_i)

		if j - i >= interplen:
			diff_matrix = np.zeros((dlen, dlen))
		else:
			offset = abs(i-j)

			if offset < dlen:
				temp_mtx_1 = np.zeros((dlen + offset, dlen))
				temp_mtx_2 = np.zeros((dlen + offset, dlen))

				if j>i:
					temp_mtx_1[:dlen, :] = mtx_i
					temp_mtx_2[offset:, :] = mtx_j
				else:
					temp_mtx_1[offset:, :] = mtx_i
					temp_mtx_2[:dlen, :] = mtx_j

				diff_matrix = np.matmul(temp_mtx_1.T, temp_mtx_2)

		return diff_matrix
