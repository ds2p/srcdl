"""
Copyright (c) 2020 CRISP

Convolutional K-SVD with interpolation

:author: Andrew H. Song
"""

from . import BaseCDL

import numpy as np
from src.helpers.convolution import *
import scipy.linalg


class CKSVD(BaseCDL):
	def __init__(self, dlen, numOfelements):
		BaseCDL.__init__(self, dlen, numOfelements)

	def updateDictionary(self, y_seg_set, d, coeffs, interpolator=None, numOfiterations=1):
		"""
		Shift invariant dictionary implementation with interpolation.
		For now, we ignore the templates on the boundary

		Inputs
		================
		coeffs: Sparse representation of the codes
		interpolator: Dictionary
			Key corresponds to fractional delay. Value corresponds to the shifted interpolator function
		"""

		assert(len(y_seg_set.keys())==len(coeffs.keys())), "The dimension of data and coeff need to match"

		if len(interpolator)==0:
			interpolator={}
			delta_fn = np.zeros(d.shape[0])
			delta_fn[int(d.shape[0]/2)] = 1
			interpolator[0] = delta_fn

		d_updated = np.copy(d)

		for base_fidx in np.arange(self.numOfelements):
			# Variable to track signal_extracted (required also for coefficient update)
			y_extracted_set = {}
			indices_set = {}
			coeffs_set = {}
			delay_set = {}

			# Collecting the extracted data segments
			for key, y_seg in y_seg_set.items():
				slen = len(y_seg)
				clen = slen + self.dlen - 1

				numOfinterp = len(interpolator.keys())

				coeffs_seg = {}
				filter_delay_indices = {}
			    # Collapse the interpolated codes together
				for fidx in np.arange(self.numOfelements):
					coeffs_seg[fidx] = {'idx':np.array([], dtype=int), 'amp':np.array([])}
					dense_code = np.zeros(clen)
					delay_indices = -np.ones(clen, dtype=int)

					for interp_idx in range(numOfinterp):
						j = fidx * numOfinterp + interp_idx

						dense_code[coeffs[key][j]['idx']] += coeffs[key][j]['amp']
						delay_indices[coeffs[key][j]['idx']] = interp_idx

					nonzero_indices = np.where(abs(dense_code)>1e-6)[0]

					coeffs_seg[fidx]['idx'] = nonzero_indices
					coeffs_seg[fidx]['amp'] = dense_code[nonzero_indices]

					filter_delay_indices[fidx] = np.array([i for i in delay_indices if i>-1])

				########################
				# Construct error signal and extract the segments
				########################
				if len(coeffs_seg[base_fidx]['idx'])>0:
					temp_indices = coeffs_seg[base_fidx]['idx']
					# We don't want to use the templates at the boundary
					indices = np.array([i for i in range(len(temp_indices)) if temp_indices[i]>= self.dlen-1 and temp_indices[i]<=slen-1])

					if len(indices)>0:
						indices_set[key] = coeffs_seg[base_fidx]['idx'][indices]
						coeffs_set[key] = coeffs_seg[base_fidx]['amp'][indices]
						delay_set[key] = filter_delay_indices[base_fidx][indices]

						patch_indices = getSignalIndices(self.dlen, indices_set[key]) - (self.dlen - 1)
						residual = np.copy(y_seg)
						for fidx in np.arange(self.numOfelements):
							# Subtract the contributions from others
							if fidx != base_fidx:
								convolved_sig = np.zeros(clen)
								for i, (idx, amp) in enumerate(zip(coeffs_seg[fidx]['idx'], coeffs_seg[fidx]['amp'])):
									if (idx >= self.dlen-1 and idx <= slen-1):	# We don't want to use the templates at the boundary
										mtx = self.compute_interp_matrix(interpolator[filter_delay_indices[fidx][i]], self.dlen)
										convolved_sig[idx : idx + self.dlen] += amp * np.matmul(mtx, d_updated[:, fidx])

								residual -= convolved_sig[self.dlen-1:]

						y_extracted_set[key] = residual[patch_indices].reshape((self.dlen,-1), order='F')

			print("Updating Filter {}".format(base_fidx))

			##############################
			# Update the filters
			##############################
			if len(y_extracted_set.keys()) > 0:

				denominator = np.zeros((self.dlen, self.dlen))
				numerator = np.zeros(self.dlen)

				for key in y_extracted_set.keys():

					y_extracted_seg = y_extracted_set[key]

					for i, (idx_1, coeff_1, delay_1) in enumerate(zip(indices_set[key], coeffs_set[key], delay_set[key])):
						numerator += coeff_1 * np.matmul(np.transpose(self.compute_interp_matrix(interpolator[delay_1], self.dlen)), y_extracted_seg[:, i])

						for idx_2, coeff_2, delay_2 in zip(indices_set[key], coeffs_set[key], delay_set[key]):
							if abs(idx_1 - idx_2) < self.dlen:
								denominator += coeff_1 * coeff_2 * self.compute_diff_matrix(interpolator[delay_1], interpolator[delay_2], idx_1, idx_2, self.dlen)

				elem = np.matmul(np.linalg.inv(denominator), numerator)
				d_updated[:,base_fidx] = elem/np.linalg.norm(elem)

			else:
				print("Non matching!")
				pass

		return d_updated, indices_set, coeffs_set, y_extracted_set

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
		"""
		Multiply two matrices for the denominator of the updated dictionary
		"""
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
