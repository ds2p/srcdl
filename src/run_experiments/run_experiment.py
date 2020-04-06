"""
Copyright (c) 2020 CRISP

functions to run robust spectral analysis on simulated/real data

:author: Andrew H. Song
"""

import os
import yaml
import numpy as np
import click
import pickle
import time
import sys
import h5py

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(PATH)

from src.models.learner import *
from src.helpers.misc import preprocessData, initializeDictionary
from src.helpers.evaluate import *
from src.models.CKSVD import *
from src.models.COMP import *
from src.generators.make_dataset import *
from src.generators.generate import generate_interpolated_Dictionary

import matplotlib.pyplot as plt

from scipy.io import loadmat
from dask import delayed, compute
import dask.multiprocessing

import logging

@click.group(chain=True)
def run_experiment():
    pass

@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def train(folder_name):

    EXPERIMENT_PATH = os.path.join(PATH, 'experiments', folder_name)
    logname = os.path.join(EXPERIMENT_PATH, 'reports','log')
    print(logname)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s  %(levelname)-10s %(message)s', filename = logname)

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
    # load data
    print("load train data.")

    data_type='train'
    y_train, y_train_idx, noise = preprocessData(folder_name, config_d, data_type)
    print("Number of segments: ", len(y_train.keys()))

    for idx in range(len(y_train.keys())):
        if len(y_train[idx])<=41:
            print("Len ", len(y_train[idx]), " IDX: ", idx)

    if 'train_noise_prop' in config_m:
        noise_adjusted = noise*config_m['train_noise_prop']
    else:
        noise_adjusted = noise

    if config_m['sparsity_flexible']:
        filename = os.path.join(EXPERIMENT_PATH, 'data','sparsity_level.mat')
        info = loadmat(filename)
        indices = info['sparsity'].flatten()
        assert len(y_train)==len(indices), "The number of segments and the number of specified sparsity level have to match: Data seg: {}, sparsity seg: {}".format(len(y_train), len(indices))
    else:
        indices = None

    ####################################
    # Build the model
    print("Building the model")
    instance = learner(config_m['dlen'],
        config_m['numOfelements'],
        noise_adjusted,
        config_m['train_sparsity_tol'],
        config_m['parallel'],
        config_m['csc_type'],
        config_m['cdl_type']
        )

    ####################################
    # load dictionary
    d = initializeDictionary(y_train, config_m, EXPERIMENT_PATH)
    instance.initializeDictionary(d/np.linalg.norm(d, axis=0))

    #####################################
    # Train and save
    instance.train_and_save(y_train,
                            y_train_idx,
                            EXPERIMENT_PATH,
                            config_m,
                            indices=indices)

@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def predict(folder_name):

    EXPERIMENT_PATH = os.path.join(PATH, 'experiments', folder_name)

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
    # load test data
    print("load test data.")
    filename = os.path.join(EXPERIMENT_PATH, 'data','data.mat')

    data_type='test'
    y_test, y_test_idx, noise = preprocessData(folder_name, config_d, data_type)

    if 'test_noise_prop' in config_m:
        noise_adjusted = noise*config_m['test_noise_prop']
    else:
        noise_adjusted = noise
    print("Test: Noise {} Noise adjusted {}".format(noise, noise_adjusted))

    ####################################
    # load dictionary
    ####################################
    print("load dictionary")
    ## Uncomment the block below to load initial dictionary
    #d = initializeDictionary(y_test, config_m, EXPERIMENT_PATH)

    ##############################
    # For trained dictionary
    filename = os.path.join(EXPERIMENT_PATH, 'results/results_train')
    file = open(filename,'rb')
    info = pickle.load(file)
    file.close()

    d = info['d']
    if config_m['interpolate'] <=1:
        print("No interpolation performed")
    else:
        print("Interpolated with interval {}".format(1/config_m['interpolate']))

    #####################################
    # Load sparsity level
    if config_m['sparsity_flexible']:
        filename = os.path.join(EXPERIMENT_PATH, 'data','sparsity_level.mat')
        info = loadmat(filename)
        indices = info['sparsity'].flatten()
        assert len(y_test)==len(indices), "The number of segments and the number of specified sparsity level have to match: Data seg: {}, sparsity seg: {}".format(len(y_train), len(indices))
    else:
        indices = None

    ####################################
    # Build the model
    print("Building the model")
    instance = learner(config_m['dlen'],
        d.shape[1],
        noise_adjusted,
        config_m['test_sparsity_tol'],
        config_m['parallel'],
        config_m['csc_type'],
        config_m['cdl_type']
        )

    # Initialize the dictionary
    instance.setDictionary(d)
    print("============================")
    print("Prediction started")
    s = time.time()
    instance.predict_and_save(y_test, y_test_idx, EXPERIMENT_PATH, config_m, indices)
    e = time.time()
    print("Elapsed time {:.4f} seconds".format(e-s))


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def train_simulation(folder_name):

    """
    For training simulation data
    """

    logname = os.path.join(PATH, 'experiments', folder_name, 'reports','log')
    print(logname)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s  %(levelname)-10s %(message)s', filename = logname)

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
    # Generate data
    ####################################
    # Noise
    noise_vars = np.arange(config_d['noise_start'], config_d['noise_end']+config_d['noise_step']-0.001, config_d['noise_step'])
    noise_vars = [round(noise,3) for noise in noise_vars]
    print(noise_vars)
    ####################################
    # load data
    print("Generate train data.")
    fs = config_d['Fs']
    filter_length = 1
    amps = [1,2]
    numOfevents = config_m['numOfevents']
    T = config_d['T']

    # Pre-defined dictionary for now
    factor = 10/filter_length
    dictionary = {}
    dictionary[0] = lambda x: (factor*x)*np.exp(-(factor*x)**2)*np.cos(2*np.pi*(factor*x)/4)
    dictionary[1] = lambda x: (factor*x)*np.exp(-(factor*x)**2)


    for noisevar in noise_vars:
        print("Noise ", noisevar)
        for i in np.arange(config_d['numOftrials']):
            print("Iteration ",i)
            realization={}

            truth, event_indices = generate_Simulated_continuous(numOfevents, T, fs, dictionary ,filter_length, amps)

            signal = truth + noisevar*np.random.randn(T*fs)

            filename = os.path.join(PATH, 'experiments', folder_name, 'data','T_{}_noise_{}_num_{}_{}.hdf5'.format(T,noisevar, config_m['numOfevents'], i))
            with h5py.File(filename,'w') as f:
                # dset = f.create_dataset("data", data = signal[:-1])
                dset = f.create_dataset("data", data = signal)
                dset.attrs['fs'] = fs
                dset.attrs['T'] = config_d['T']
                dset.attrs['numOfevents'] = config_m['numOfevents']
                dset.attrs['indices'] = event_indices

    print("Data generated")

    ######################################
    # Acutal experiment
    ######################################

    dlen = config_m['dlen']
    if config_m['interpolate']>0:
        interval = 1/int(config_m['interpolate'])
        delay_arr = np.arange(interval, 1, interval)

    sparsity_tol = config_m['numOfevents']*config_m['numOfelements']

    #######################
    # Initialize dictionary
    #######################
    factor = 10
    ts = np.linspace(-5/factor, 5/factor, config_m['dlen'], endpoint=True)
    d_true_discrete = np.zeros((config_m['dlen'], config_m['numOfelements']))
    for fidx in range(config_m['numOfelements']):
        d_true_discrete[:,fidx] = dictionary[fidx](ts)

    errors = {'basic': np.zeros((len(noise_vars), config_d['numOftrials'], config_m['numOfelements'])), 'delay': np.zeros((len(noise_vars),config_d['numOftrials'], config_m['numOfelements']))}
    init_errors = {'basic': np.zeros((len(noise_vars), config_d['numOftrials'], config_m['numOfelements'])), 'delay': np.zeros((len(noise_vars),config_d['numOftrials'], config_m['numOfelements']))}


    d_init_set = {}
    d_train_set = {}
    d_train_interp_set= {}

    #####################
    # Loop through different noise variances
    #####################

    for noise_idx, noise in enumerate(noise_vars):

        d_init_set[noise_idx] = {}
        d_train_set[noise_idx] = {}
        d_train_interp_set[noise_idx] = {}

        print("--------------------------")
        for tidx  in np.arange(config_d['numOftrials']):
            print("Noise var {} Trial {}".format(noise, tidx))
            filename = os.path.join(PATH, 'experiments', folder_name, 'data','T_{}_noise_{}_num_{}_{}.hdf5'.format(T,noisevar, config_m['numOfevents'], tidx))
            with h5py.File(filename,'r') as f:
                signal = f['data'][:]

            #######################
            # Initialize dictionary
            #######################
            init_d = d_true_discrete + config_m['init_noise'] * np.random.randn(dlen, config_m['numOfelements'])
            init_d = init_d/np.linalg.norm(init_d,axis=0)

            #Recovery error
            print("Recovery error ", recovery_error(init_d, d_true_discrete))

            init_errors['basic'][noise_idx,tidx,:] = recovery_error(init_d, d_true_discrete)
            init_errors['delay'][noise_idx,tidx,:] = recovery_error(init_d, d_true_discrete)

            ###################
            # CDL without delay
            ###################
            csc = COMP(dlen, 1e-5, sparsity_tol, 0)
            cdl = CKSVD(dlen, config_m['numOfelements'])
            print("COMP")
            distance_original = np.zeros((config_m['numOfelements'], config_m['numOfiterations']))

            d_train = np.copy(init_d)

            for idx in np.arange(config_m['numOfiterations']):
                print("Noise ", noise," Iteration ", idx, " for Trial ", tidx)
                distance_original[:, idx] = recovery_error_translate(d_true_discrete, d_train)
                code_original, err = csc.extractCode_seg_projection_eff(signal, d_train, boundary=0)
                code_original = code_sparse(code_original, config_m['numOfelements'])
                d_train, _, _, _ = cdl.updateDictionary({0: signal}, d_train, {0: code_original}, {}, 1)

            #################
            # CDL with delay
            #################
            print("COMP with delay")
            distance_interp = np.zeros((config_m['numOfelements'], config_m['numOfiterations']))

            d_train_interp = np.copy(init_d)

            for idx in np.arange(config_m['numOfiterations']):
                print("Noise ", noise," Iteration ", idx, " for Trial ", tidx)
                distance_interp[:, idx], interp_indices = recovery_error_interp(d_true_discrete, d_train_interp, delay_arr)

                [d_interpolated, interpolator] = generate_interpolated_Dictionary(d_train_interp, config_m['interpolate'])

                code_original, err_interp = csc.extractCode_seg_projection_eff(signal, d_interpolated, boundary=0)
                code_original = code_sparse(code_original, d_interpolated.shape[1])

                d_train_interp, _, _, _ = cdl.updateDictionary({0: signal}, d_train_interp, {0: code_original}, interpolator,1)

            print("Original ", distance_original)
            print("Interp ", distance_interp)

            d_init_set[noise_idx][tidx] = init_d
            d_train_set[noise_idx][tidx] = d_train
            d_train_interp_set[noise_idx][tidx] = d_train_interp

            errors['basic'][noise_idx,tidx,:] = distance_original[:,-1]
            errors['delay'][noise_idx,tidx,:] = distance_interp[:,-1]

    info={}
    info['noise'] = noise_vars
    info['numOfevents'] = config_m['numOfevents']
    info['numOfiterations']  = config_m['numOfiterations']
    info['T'] = config_d['T']
    info['errors'] = errors
    info['init_errors'] = init_errors
    info['d_init'] = d_init_set
    info['d_train'] = d_train_set
    info['d_train_interp'] = d_train_interp_set

    filename = os.path.join(PATH, 'experiments', folder_name, 'results','CDL_snr')
    with open(filename, 'wb') as f:
        pickle.dump(info, f)

if __name__=="__main__":
    run_experiment()
