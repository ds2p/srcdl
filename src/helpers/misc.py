"""
Copyright (c) 2020 CRISP

Auxilary helper functions

:author: Andrew H. Song
"""

import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

import numpy as np
from scipy.io import loadmat

from .evaluate import findpeaks

import numpy as np
import pickle

def preprocessData(folder_name, config, data_type='train'):
	"""
	Data pre-processing pipeline. Operations include:
	1) Read the data
	2) Estimate the background noise level based on the data

	Two modes are available: 'external' or 'normal'
	'external': Assumes the data has already been segmented (possibly of different lengths)
		In this mode, 'data.mat' file needs to have 'signal': The collection of segemented data and 'idx': the collection of indices corresponding to the ID of the segments
	'normal': Assumes the data has not been segmented

	Outputs
	=======
	data: processed data
	data_idx: dictionary of [start index, end index] for each segment of the data
	noise: background noise level

	"""

	mode = config['mode']

	filename = os.path.join(PATH, 'experiments', folder_name, 'data','data.mat')
	dinfo = loadmat(filename)

	if mode=='external':	# If the data has already been segmented
		print("Data reading: External mode")
		# data structure validation
		assert(('signal' in dinfo.keys()) and ('idx' in dinfo.keys())), "Either signal or seg_idx fields do not exist"

		data = {}
		data_idx = {}

		data_temp = dinfo['signal'].flatten()
		data_idx_temp = dinfo['idx'].flatten()

		for idx in np.arange(len(data_temp)):
			data[idx] = np.array(data_temp[idx].flatten())
			data_idx[idx] = np.array(data_idx_temp[idx].flatten())

		noise = config['noise']

	elif mode=='normal':	# If not segmented previously
		print("Data reading: Normal mode (equal size windows)")
		assert('signal' in dinfo.keys()), "Signal field does not exist!"

		if data_type=='train':
			starttime = config['starttime_train']
			endtime = config['endtime_train']
		else:
			starttime = config['starttime_test']
			endtime = config['endtime_test']

		assert(('noise' in config) or ('starttime_noise' in config)), "Need to supply the noise argument"

		if 'noise' in config:
			noise = config['noise']
		else:
			noise = estimateError(dinfo['signal'][config['channel'],
				int(config['starttime_noise']*config['Fs']):int(config['endtime_noise']*config['Fs'])])

		signal = np.array(dinfo['signal'])
		assert((starttime*config['Fs']<signal.shape[1]) and (endtime*config['Fs']<signal.shape[1])), "start or endtime exceeds the length of the signal"

		extracted = signal[config['channel'], int(starttime*config['Fs']):int(endtime*config['Fs'])]

		win = int(config['win']*config['Fs'])

		numOfwin = int(np.ceil(len(extracted)/win))
		data = {}
		data_idx = {}

		for widx in np.arange(numOfwin):
			if widx == numOfwin -1:
				data[widx] = extracted[widx*win:]
				data_idx[widx] = [starttime + widx*win, starttime + len(extracted) - 1]
			else:
				data[widx] = extracted[widx*win:(widx+1)*win]
				data_idx[widx] = [starttime + widx*win, starttime + (widx+1)*win-1]
	else:
		raise NotImplementedError("Not implemented! Use either 'external' or 'normal' mode.")

	return data, data_idx, noise

def initializeDictionary(data, config, path=None):
	"""

	Initialize dictionary

	Inputs
	======
	config: model configuration file
	folder_name: Required if importing the initial templates

	"""

	from sklearn.cluster import KMeans
	from sklearn.decomposition import PCA

	dlen = config['dlen']
	numOfelements = config['numOfelements']

	if config['init_mode'] == 'external':
		assert(path), "No path given!"

		filename = os.path.join(path, 'data','d_init.mat')
		dinfo = loadmat(filename)

		d = np.array(dinfo['d'])
		assert d.shape==(dlen, numOfelements), "The initial dictionary not the right shape: Should be ({},{}), but got ({}, {})".format(dlen, numOfelements, d.shape[0], d.shape[1])
	else:
		assert(('peakloc' in config.keys()) and ('threshold' in config.keys()) and ('polarity' in config.keys())), "Some of the keys are missing"

		threshold = config['threshold']
		polarity = config['polarity']
		init_iterations = config['init_iterations']
		peakloc = config['peakloc']

		assert(peakloc<dlen), "Peak location cannot be larger than the length of the template"

		if 'threshold' in config:
			threshold = config['threshold']


		peak_index = findpeaks(data, threshold, polarity)

		front = peakloc
		back = dlen - front

		waveforms = np.array([]).reshape((-1, dlen))

		for segidx in peak_index:
			slen = len(data[segidx])
			peaks = peak_index[segidx]

			for pidx in peaks:
				# Check boundary conditions
				if (pidx-front>=0) and (pidx+back < slen):
				    peak = data[segidx][pidx]
				    sig = data[segidx][pidx-front:pidx+back]

				    if (polarity == 0 and min(sig) == peak) or (polarity == 1 and max(sig) == peak):
				        waveforms = np.vstack((waveforms, sig.reshape((-1, dlen))))

		# Minimum correlation templates
		if config['init_mode']=='corr':

			assert('init_iterations' in config.keys()), "init_iterations key missing"

			numOfwaveforms = waveforms.shape[0]
			minIndices = np.zeros(numOfelements)
			min_corr = 1

			print("Waveforms ", numOfwaveforms)

			for i in np.arange(init_iterations):
			    idx = np.random.choice(numOfwaveforms, numOfelements, replace=0)
			    temp = waveforms[idx, :]

			    corr = np.triu(np.corrcoef(temp), k=1)
			    corr = np.max(abs(corr))
			    if corr < min_corr:
			        min_corr = corr
			        min_indices = idx
			d = np.transpose(waveforms[min_indices,:])
		# K-means clustering
		elif config['init_mode']=='cluster':

			print("Initialization via clustering")
			# Temporary measure
			waveforms_PCA = waveforms

			# K-means clustering
			kmeans = KMeans(n_clusters=numOfelements)
			kmeans.fit(waveforms_PCA)
			d = np.transpose(kmeans.cluster_centers_)
		else:
			raise NotImplementedError("Not implemented")

	dict_init = d/np.linalg.norm(d, 2, 0)
	return dict_init

def estimateError(signal):
    """
    Estimate the standard deviation of the given portion of the data

    Inputs
    ======
    signal:

    Outputs
    =======
    enorm: standard deviation of the signal
    """

    enorm = np.linalg.norm(signal)/np.sqrt(len(signal))
    return enorm
