"""
Copyright (c) 2020 CRISP

Extract/Display results

:author: Andrew H. Song
"""

import os
import yaml
import numpy as np
import click
import pickle
import time
import h5py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH,"..",".."))

from scipy.io import loadmat
from src.plotter.plot import *
from src.helpers.misc import *
from src.helpers.evaluate import matchTruth,  matchTruth_ell1
from src.helpers.convolution import reconstruct, convolve_threshold_all
from src.generators.generate import generate_interpolated_Dictionary

@click.group(chain=True)
def extract_results():
	pass

@extract_results.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def display_results(folder_name):

	####################################
	# load model parameters
	print("load model parameters.")
	filename = os.path.join(PATH, 'experiments', folder_name, 'config','config_model.yml')
	file = open(filename, "rb")
	config_m = yaml.load(file)
	file.close()

	####################################
	# load data parameters
	print("load data parameters.")
	filename = os.path.join(PATH, 'experiments', folder_name, 'config','config_data.yml')
	file = open(filename, "rb")
	config_d = yaml.load(file)
	file.close()

	####################################
	# load results
	print("load results")
	filename = os.path.join(PATH, 'experiments', folder_name, 'results','results_train')
	file = open(filename, "rb")
	results = pickle.load(file)
	file.close()

	####################################
	if 'init_d' not in results:
		init_d = None
	else:
		init_d = results['init_d']

	drawFilters(results['d'], init_d)

	if 'd_distance' in results:
		drawDictError(results['d_distance'])
	plt.show()

@extract_results.command()
def display_spikesorting_errorcurve():
	"""
	Comparsion of the error curves for the spike sorting application.
	The results for CBP and ADCG have been obtained from other scripts
	The dataset is from

	Henze DA et al., "Intracellular features predicted by extracellular recordings in the hippocampus in vivo",
	Journal of Neurophysiology, 2000

	"""
	folder_names={
		'cbp': "spikesorting_cbp",
		'COMP': "spikesorting_no_interp",
		"COMP_interp10": "spikesorting_interp",
		"adcg": "spikesorting_adcg",
	}

	true_miss_list = {}
	false_alarm_list = {}

	for key, folder_name in folder_names.items():

		EXPERIMENT_PATH = os.path.join(PATH, 'experiments', folder_name)

		numOftruth = 621
		print("===========================================")
		print("Method: ", key)
		if key == 'cbp':
			filename = os.path.join(EXPERIMENT_PATH, 'results','true_miss.mat')
			info = loadmat(filename)
			truemiss = info['true_miss'].flatten()

			filename = os.path.join(EXPERIMENT_PATH, 'results','false_alarm.mat')
			info = loadmat(filename)
			false_alarm_list[key] = info['false_alarm'].flatten()
			true_miss_list[key] = truemiss
		elif 'adcg' in key:
			####################################
			# load model parameters
			print("load model parameters.")
			filename = os.path.join(EXPERIMENT_PATH, 'config','config_model.yml')
			file = open(filename, "rb")
			config_m = yaml.load(file)
			file.close()

			####################################
			# load data parameters
			print("load data parameters.")
			filename = os.path.join(EXPERIMENT_PATH, 'config','config_data.yml')
			file = open(filename, "rb")
			config_d = yaml.load(file)
			file.close()

			numOfelements = config_m['numOfelements']

			####################################
			# misc
			print("load CSC results")
			filename = os.path.join(PATH, 'experiments','spikesorting_no_interp', 'results','results_test')
			file = open(filename, "rb")
			results = pickle.load(file)
			file.close()

			# temporary mesasure to load indices
			segment_indices = results['segment_indices']

			####################################
			# # load dictionary
			print("load dictionary")

			filename = os.path.join(EXPERIMENT_PATH, 'data/d_init.mat')
			info = loadmat(filename)
			d = info['d']
			d = d/np.linalg.norm(d, axis=0)

			print("load CSC results")
			filename = os.path.join(EXPERIMENT_PATH, 'results','results_test.hdf5')
			file = h5py.File(filename, 'r')

			# Reshaping the hdf5 into appropriate format
			code = {idx:{} for idx in range(len(file.keys()))}
			for idx in range(len(file.keys())):

				code[idx] = {fidx:{} for fidx in range(numOfelements)}
				results = file['{}'.format(idx+1)]

				for fidx in range(numOfelements):
					code[idx][fidx]['idx'] = np.array([results['filter'][idx,0] for idx in range(results['filter'].shape[0]) if results['filter'][idx,1] == fidx+1])
					code[idx][fidx]['amp'] = np.array([results['amp'][idx] for idx in range(results['filter'].shape[0]) if results['filter'][idx,1] == fidx+1])

			####################################
			# load test data
			data_type='test'
			y_test, y_test_idx, noise = preprocessData(folder_name, config_d, data_type)

			####################################
			# load ground truth (For now, support only peaks)
			print("Load truth")
			filename = os.path.join(EXPERIMENT_PATH, 'data/truth.mat')
			mat = loadmat(filename)
			true_timestamps = mat['true_ts'].flatten()

			thresh_range = np.linspace(config_m['thresh_min'], config_m['thresh_max'], config_m['thresh_num'])
			truemiss, falsealarm, nonzerocoeffs, match, fa_coeffs, true_coeffs = matchTruth_ell1(true_timestamps, d, code, segment_indices, config_m['offset'], thresh_range)
			false_alarm_list[key] = nonzerocoeffs - match
			true_miss_list[key] = truemiss

		elif 'COMP' in key:
			####################################
			# load model parameters
			print("load model parameters.")
			filename = os.path.join(EXPERIMENT_PATH, 'config','config_model.yml')
			file = open(filename, "rb")
			config_m = yaml.load(file)
			file.close()

			####################################
			# load data parameters
			print("load data parameters.")
			filename = os.path.join(EXPERIMENT_PATH, 'config','config_data.yml')
			file = open(filename, "rb")
			config_d = yaml.load(file)
			file.close()

			####################################
			# # load dictionary
			filename = os.path.join(EXPERIMENT_PATH, 'results/results_train')
			file = open(filename,'rb')
			info = pickle.load(file)
			file.close()

			d = info['d']
			if config_m['interpolate']>0:
				d, _ = generate_interpolated_Dictionary(d, config_m['interpolate'])
				print("Interpolated with interval {}".format(1/config_m['interpolate']))

			print("load CSC results")
			filename = os.path.join(EXPERIMENT_PATH, 'results','results_test')
			file = open(filename, "rb")
			results = pickle.load(file)
			file.close()

			code = results['code']
			seg_indices = results['segment_indices']

			####################################
			# load test data
			data_type='test'
			y_test, y_test_idx, noise = preprocessData(folder_name, config_d, data_type)

			####################################
			# load ground truth (For now, support only peaks)
			print("Load truth")
			filename = os.path.join(EXPERIMENT_PATH, 'data/truth.mat')
			mat = loadmat(filename)
			true_timestamps = mat['true_ts'].flatten()

			thresh_range = np.linspace(config_m['thresh_min'], config_m['thresh_max'], config_m['thresh_num'])
			_, _, nonzerocoeffs, match, fa_coeffs, true_coeffs = matchTruth(true_timestamps, d, code, seg_indices, config_m['offset'], thresh_range,config_m['polarity'])

			numOfelements = config_m['numOfelements']

			truemiss = np.zeros((numOfelements, len(thresh_range)))
			falsealarm = np.zeros((numOfelements, len(thresh_range)))

			numOfmem = int(d.shape[1]/numOfelements)
			for idx in np.arange(numOfelements):
				numOfmatch = np.sum(match[idx*numOfmem:(idx+1)*numOfmem, :],0)
				numOfnonzerocoeffs = np.sum(nonzerocoeffs[idx*numOfmem:(idx+1)*numOfmem, :],0)
				truemiss[idx,:] = numOftruth - numOfmatch
				falsealarm[idx,:] = numOfnonzerocoeffs - numOfmatch

			true_miss_list[key] = truemiss
			false_alarm_list[key] = falsealarm

		else:
			raise NotImplementedError("This approach not implemented")

	title = 'Comparison between different methods'
	keys={'cbp': 'CBP', 'COMP':'COMP', 'COMP_interp10':'COMP-INTERP', 'adcg':'ADCG'}

	print("Drawing the result")
	statistics = {'true_miss_list': true_miss_list, 'false_alarm_list': false_alarm_list}
	hyp = {
		'numOfelements': config_m['numOfelements'],
		'title': title,
		'keys': keys,
		'path': os.path.join(PATH, 'experiments'),
		'true_miss_threshold': 200
	}
	drawStatistics_comparison(statistics, hyp)

	result = {}
	result['keys'] = keys
	result['tm'] = true_miss_list
	result['fa'] = false_alarm_list

	filename = os.path.join(PATH,'experiments/spikesorting_comparison_result')
	with open(filename,'wb') as f:
		pickle.dump(result, f)

# @extract_results.command()
# @click.argument("keys",nargs=-1)
# @click.option("--folder_name", default="", help="folder name in experiment directory")
# def display_reconstruct(folder_name, keys):
# 	"""
# 	Reconstruct based on CSC/CDL result
#
# 	Inputs
# 	======
# 	keys: array_like. index of windows for signal reconstruction
#
# 	"""
#
# 	seg = []
# 	for key in keys:
# 		seg.append(int(float(key)))
#
# 	####################################
# 	# load model parameters
# 	print("load model parameters.")
# 	filename = os.path.join(PATH, 'experiments', folder_name, 'config','config_model.yml')
# 	file = open(filename, "rb")
# 	config_m = yaml.load(file)
# 	file.close()
#
# 	####################################
# 	# load data parameters
# 	print("load data parameters.")
# 	filename = os.path.join(PATH, 'experiments', folder_name, 'config','config_data.yml')
# 	file = open(filename, "rb")
# 	config_d = yaml.load(file)
# 	file.close()
#
# 	####################################
# 	# load results
#
# 	print("load CDL results")
# 	filename = os.path.join(PATH, 'experiments', folder_name, 'results','results_train')
# 	file = open(filename, "rb")
# 	results = pickle.load(file)
# 	file.close()
#
# 	d = results['d']
#
# 	print("load CSC results")
# 	filename = os.path.join(PATH, 'experiments', folder_name, 'results','results_test')
# 	file = open(filename, "rb")
# 	results = pickle.load(file)
# 	file.close()
#
# 	code = results['code']
#
# 	####################################
# 	# load data
# 	data_type='test'
# 	y_test, y_test_idx, noise = preprocessData(folder_name, config_d, data_type)
#
# 	####################################
# 	# Reconstruct signals
# 	reconstructed = {}
# 	for key in seg:
# 		slen = len(y_test[key])
# 		convolved_signal = reconstruct(d, code[key], slen)
# 		reconstructed[key] = convolved_signal
#
# 	####################################
# 	# load ground truth (For now, support only peaks)
# 	print("Load truth")
# 	filename = os.path.join(PATH, 'experiments', folder_name, 'data/truth.mat')
# 	mat = loadmat(filename)
# 	true_timestamps = mat['true_ts'].flatten()
#
# 	truth = {}
# 	for key in seg:
# 		timestamps = y_test_idx[key]
# 		truth_ts = true_timestamps[(timestamps[0] < true_timestamps) & (true_timestamps <= timestamps[-1])]
# 		truth[key] = truth_ts -timestamps[0]
#
# 	drawReconstructed(reconstructed, y_test, y_test_idx, truth)
# 	plt.show()


if __name__=="__main__":
	extract_results()
