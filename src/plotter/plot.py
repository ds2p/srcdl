"""
Copyright (c) 2020 CRISP

Helper functions for plotting the results

:author: Andrew H. Song
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib as mpl
import os

sns.set_style('white')

def drawFilters(d, init_d=None):

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	numOfelements = d.shape[1]

	fig, ax = plt.subplots(figsize=(6,6))
	for idx in np.arange(numOfelements):
		plt.plot(d[:, idx], label='Element {}'.format(idx), color=colors[idx], linewidth=3)

	if init_d is not None:
		for idx in np.arange(numOfelements):
			plt.plot(init_d[:, idx], label='Init Element {}'.format(idx), linestyle='-.', color=colors[idx])

	plt.title("Dictionary elements", fontsize=20)
	plt.xlabel('Samples', fontsize=15)
	plt.ylabel('Normalized amplitude', fontsize=15)
	plt.legend(fontsize=12)
	ax.tick_params(axis='both', labelsize=15)

	plt.draw()

def drawDictError(errors):
	"""
	Plots the dictionary distance from the initial dictionary as a function of iterations

	"""

	numOfelements = errors.shape[0]
	fig, ax = plt.subplots(figsize=(8,6))
	for idx in np.arange(numOfelements):
		plt.plot(errors[idx,:], label='Element {}'.format(idx))

	plt.title("Dictionary error")
	plt.xlabel('Iterations', fontsize=15)
	plt.ylabel('Errors', fontsize=15)
	plt.legend(fontsize=12)
	ax.tick_params(axis='both', labelsize=15)

	plt.draw()

def drawCodeDistribution(code):

	# Bit artificial
	for key, value in code.items():
		numOfelements = len(value.keys())
		break

	coeffs_dist = {idx:np.array([]) for idx in np.arange(numOfelements)}
	for key,value in code.items():
		for fidx in np.arange(numOfelements):
			coeffs_dist[fidx] = np.concatenate((coeffs_dist[fidx],value[fidx]['amp']))

	for fidx in np.arange(numOfelements):
		total_count = len(coeffs_dist[fidx])

		fig, ax = plt.subplots(figsize=(6,6))
		plt.hist(coeffs_dist[fidx], bins=100)
		ax.set_title("Filter {} total number {}".format(fidx, total_count))
		plt.draw()



def drawReconstructed(reconstructed, data, data_idx, truth):
	"""
	Draw reconstructed signal from the learned dictionary/codes
	First panel is the actual signal. Second panel is the reconstructed signal + True timestamp marks

	Inputs
	======
	truth: array_like
		Timestamps of the truth signal, if the truth is available.
	"""

	assert(len(reconstructed.keys())<=20), "Too many figure plots! Try reducing the number"

	for key in reconstructed.keys():
		residual = data[key]

		fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(10,7))
		ax1.plot(data[key])
		ax1.tick_params(axis='both', labelsize=14)
		numOfelements = len(reconstructed[key].keys())
		for idx in np.arange(numOfelements):
			residual -= reconstructed[key][idx]
			ax2.plot(reconstructed[key][idx])
		ax2.plot(truth[key],5*np.ones(len(truth[key])),'ro')
		ax2.set_xlim([0, len(data[key])])

		# print("Truth for {} ".format(key), truth[key])

		residual = np.linalg.norm(residual)/np.sqrt(len(residual))

		ax2.tick_params(axis='both', labelsize=14)
		ax1.set_title("True signal for {}".format(key), fontsize=17)
		ax2.set_title("Reconstructed signal for {0}: {1} ~ {2}. err: {3:.2f}".format(key, data_idx[key][0], data_idx[key][-1], residual), fontsize=17)
	plt.draw()

def drawStatistics_comparison(statistics, hyp):


	true_miss_list = statistics['true_miss_list']
	false_alarm_list = statistics['false_alarm_list']

	numOfelements = hyp['numOfelements']
	title = hyp['title']
	keys = hyp['keys']
	path = hyp['path']
	true_miss_threshold = hyp['true_miss_threshold']

	fig, ax = plt.subplots(figsize=(7,6))

	## These handles changing matplotlib background to unify fonts and fontsizes
	# pgf_with_latex = {                      # setup matplotlib to use latex for output
	# "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	# "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	# "font.sans-serif": [],
	# "font.monospace": [],
	# "pgf.preamble": [
	# r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
	# r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
	#         ]
	#     }
	# mpl.rcParams.update(pgf_with_latex)

	fontsize = 19

	for key in true_miss_list.keys():
		if key=='cbp':
			plt.plot(true_miss_list[key], false_alarm_list[key], '-', linewidth=3, label='{}'.format(keys[key]))
		else:
			for idx in np.arange(numOfelements):
				if len(np.where(true_miss_list[key][idx]<true_miss_threshold)[0])>0:
					plt.plot(true_miss_list[key][idx,:], false_alarm_list[key][idx,:], '-', linewidth=3, label='{}'.format(keys[key]))

	ax.tick_params(axis='both', labelsize=18)
	ax.set_xlabel('Number of true miss', fontsize=15)
	ax.set_ylabel('Number of false alarm', fontsize=15)
	plt.legend(fontsize=19, frameon=False)
	ax.grid(False)
	plt.ylim([0,60])
	plt.xlim([0,80])

	plt.savefig(os.path.join(path, 'error_curve.pdf'), bbox_inches='tight')
	plt.savefig(os.path.join(path, 'error_curve.png'), bbox_inches='tight')


def drawCode(cbp_list, omp_list=None):
	numOfwin = cbp_list.shape[0]
	numOfelements = cbp_list.shape[1]

	fig, ax = plt.subplots(4,1,sharex=True, figsize=(10,10))

	for idx in np.arange(numOfelements):
		ax[idx].scatter(np.arange(numOfwin), cbp_list[:,idx], label='cbp', color='blue', marker='o', s=40)
		ax[idx].scatter(np.arange(numOfwin), omp_list[:,idx], label='comp', color='red', marker='x', s=40)
		ax[idx].set_title("Filter {} CBP total {} COMP total {}".format(idx, int(np.sum(cbp_list[:,idx])), int(np.sum(omp_list[:,idx]))))
		ax[idx].tick_params(axis='both', labelsize=14)
		ax[idx].legend(fontsize=15)

	ax[-1].scatter(np.arange(numOfwin), np.sum(cbp_list, 1), color='blue', marker='o', s=40)
	ax[-1].scatter(np.arange(numOfwin), np.sum(omp_list, 1), color='red', marker='x', s=40)
	ax[-1].set_title('Total number')
	ax[-1].tick_params(axis='both', labelsize=14)
	plt.draw()

	plt.draw()
