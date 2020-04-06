"""
Copyright (c) 2020 CRISP

Simulation data generator

:author: Andrew H. Song
"""

import sys
import os

# Append path
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

import numpy as np
import scipy

def generate_Simulated_continuous(numOfevents, T, fs, dictionary, filter_length, amps=[0,1]):
	"""
	Generate continuous data and its sampled version.
	For now, assume that we know the templates. These templates start from -5 to 5

	Inputs
	======
	dictionary: a dictionary of continuous functions
	filter_length: filter length (in seconds)

	amps: array (two elements)
		lower boudn and upper bound for the amplitudes


	Outputs
	=======
	signal: array_like
		Generated signal

	"""
	assert(len(amps)==2 and amps[0]<amps[1]), "Wrong amplitude arguments"

	numOfelements = len(dictionary.keys())

	signal = np.zeros(T*fs)
	interval = 1/fs

	# Generate event indices
	events_indices = np.zeros((numOfelements, numOfevents))

	for fidx in np.arange(numOfelements):
		events_idx = np.sort(T*np.random.rand(numOfevents))

		# Event index generation
		idx_diff = np.where(events_idx[1:] - events_idx[:-1]<filter_length)[0]
		condition = len(idx_diff) == 0 and (events_idx[0] > filter_length) and (events_idx[-1] < T - filter_length)

		while not condition:
			if events_idx[0] <= filter_length:
				new_idx = T*np.random.rand()
				events_idx[0] = new_idx
			elif events_idx[-1]>= T-filter_length:
				new_idx = T*np.random.rand()
				events_idx[-1] = new_idx
			else:
				for i in idx_diff:
					new_idx = T*np.random.rand()
					events_idx[i+1] = new_idx

			events_idx = np.sort(events_idx)

			idx_diff = np.where(events_idx[1:] - events_idx[:-1]<filter_length)[0]
			condition = len(idx_diff) == 0 and (events_idx[0] > filter_length) and (events_idx[-1] < T - 1*filter_length)

		events_indices[fidx,:] = events_idx

		# Signal generation
		for idx, event_timestamp in enumerate(events_idx):
			start_sample = 0
			amp = np.random.uniform(amps[0], amps[1])

			# Find the closest sample to the event starting point
			start_sample = int(np.ceil(event_timestamp * fs))

			# Distance between the template starting point (continuous) and the first sample grid
			delta = start_sample * interval - event_timestamp

			maxamp = -100
			filter_length_in_samples = filter_length * fs
			filter_realization = np.zeros(filter_length_in_samples)

			for sidx in np.arange(filter_length_in_samples):
				ts = -filter_length_in_samples/2 * interval + delta + sidx*interval
				point = dictionary[fidx](ts)
				filter_realization[sidx] = point
				if point>maxamp:
					maxamp = point
			signal[start_sample : start_sample + filter_length_in_samples] += filter_realization/maxamp*amp

	return signal, events_indices


def generate_cdl_snr():
	pass
