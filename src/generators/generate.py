"""
Copyright (c) 2020 CRISP

Generates interpolated dictionary

:author: Andrew H. Song
"""

import numpy as np
import pickle
import scipy

def generate_interpolated_Dictionary(d, numOfsubgrids, normalize=1, kind='cubic'):
	"""
	Generates interpolated dictionary given the original dictionary and number of subgrids
	(TODO): Incorporate with generate_sinc_Dictionary

	Inputs
	======
	numOfsubgrids: integer
		Specifies number of subgrids. For example if 10, divides the original sampling grid into 10 equal partitions
	kind: 'cubic' (default), 'linear', or 'sinc'
		Interpolator function

	Outputs
	=======
	d_interpolated: 2-D array
		Interpolated dictionary
	interpolator: Dictionary
		Dictionary of interpolator functions

	"""

	interpolator = {}
	if numOfsubgrids<=1:
		return d, interpolator

	print("Interpolating with 1/{} sub-grid".format(numOfsubgrids))

	interval = 1/numOfsubgrids
	delay_arr = np.arange(interval, 1, interval)

	numOfsamples, numOfelements = d.shape
	assert((0 not in delay_arr) and (1 not in delay_arr)), "Only non-integer delays are allowed"
	assert np.mod(numOfsamples,2)==1, "The filter length must be odd."

	numOfdelays = len(delay_arr)

	# Extra element is for the original template
	d_interpolated = np.zeros((numOfsamples, numOfelements*(numOfdelays+1)))
	d_interpolated[:, np.arange(numOfelements)*(numOfdelays + 1)] = d

	# The first interpolator should be a shifted delta function (to produce the original element)
	delta_fn = np.zeros(numOfsamples)
	delta_fn[int(numOfsamples/2)] = 1
	interpolator[0] = delta_fn

	for didx, delay in enumerate(delay_arr,1):
		if kind == 'cubic':
			x_interp = np.linspace(-2,2,5,endpoint=True) - delay
			f_interp = []
			for idx in x_interp:
				if abs(idx)>=2:
					f_interp.append(0)
				elif 1<=abs(idx) and abs(idx)<2:
					f_new = -0.5*np.power(abs(idx),3) + 2.5*np.power(abs(idx),2) - 4*abs(idx)+2
					f_interp.append(f_new)
				else:
					f_new = 1.5*np.power(abs(idx),3) - 2.5*np.power(abs(idx),2) + 1
					f_interp.append(f_new)
		elif kind == 'linear':
			x_interp = np.linspace(-1,1,3,endpoint=True) - delay
			f_interp = []
			for idx in x_interp:
				if abs(idx)>=1:
					f_interp.append(0)
				else:
					f_new = 1 - abs(idx)
					f_interp.append(f_new)
		elif kind == 'sinc':
			if np.mod(numOfsamples, 2)==0:
				x = np.arange(numOfsamples) - int(numOfsamples/2)+1
			else:
				x = np.arange(numOfsamples) - int(numOfsamples/2)

			f_interp = np.sinc(x-delay)
		else:
			raise NotImplementedError("This interpolator is not implemented!")

		interpolator[didx] = f_interp

		for fidx in np.arange(numOfelements):
			elem = d[:,fidx]
			d_interpolated[:, fidx*(numOfdelays+1)+didx] = scipy.signal.convolve(elem, f_interp, mode='same')

	if normalize:
		d_interpolated = d_interpolated/np.linalg.norm(d_interpolated, axis=0)

	return d_interpolated, interpolator


def generate_interpolated_Dictionary_2D(d, x_delay_arr, y_delay_arr, normalize=1, kind='cubic'):
	"""
	Interpolation of the dictionary for 2D.
	For now, doesn't return the interpolators
	x_delay_arr and y_delay_arr need to have 0 in them.
	"""

	from scipy.interpolate import griddata

	assert np.array_equal(x_delay_arr, y_delay_arr), "Delays in both directions should be same"

	if len(x_delay_arr)==0:
		return d

	numOfsamples, _, numOfelements = d.shape
	assert np.mod(numOfsamples,2)==1, "The filter length must be odd."
	numOfdelays = len(x_delay_arr)

	# Extra element is for the original template
	d_interpolated = np.zeros((numOfsamples, numOfsamples, numOfelements*numOfdelays**2))

	grid_original = [(i,j) for i in np.arange(numOfsamples) for j in np.arange(numOfsamples)]

	for x_delay_idx, x_delay in enumerate(x_delay_arr):
		for y_delay_idx, y_delay in enumerate(y_delay_arr):

			index = x_delay_idx * numOfdelays + y_delay_idx
			if x_delay_idx ==0:
				x_grid = np.arange(numOfsamples)
			else:
				x_grid = (1-x_delay) + np.arange(numOfsamples-1)
				x_grid = np.insert(x_grid,0,0)

			if y_delay_idx == 0:
				y_grid = np.arange(numOfsamples)
			else:
				y_grid = (1-y_delay) + np.arange(numOfsamples-1)
				y_grid = np.insert(y_grid,0,0)

			grid_new = [(i,j) for i in x_grid for j in y_grid]

			for fidx in range(numOfelements):
				itp = griddata(grid_original, d[..., fidx].reshape(1,-1).flatten(), grid_new, method='cubic')
				d_interpolated[..., fidx * numOfdelays**2 + index] = itp.reshape(numOfsamples, numOfsamples)

	if normalize:
		for idx in range(numOfelements*(numOfdelays+1)):
			d_interpolated[..., idx] = d_interpolated[..., idx]/np.linalg.norm(d_interpolated[..., idx])

	return d_interpolated
