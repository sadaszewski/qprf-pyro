import os
from scipy.io import loadmat
import numpy as np
import pickle


def load_input_file(input_filename):
    name, ext = os.path.splitext(input_filename)
    ext = ext.lower()
    if ext == '.mat':
        data = loadmat(input_filename)
    else:
        raise ValueError('Unsupported input format')
    return data


def resolve_stimulus_variable(data, variable_name):
    if variable_name not in data:
        raise ValueError('Specified variable not found')
    var = data[variable_name]
    while var.dtype == np.object:
        var = var[0]
    if len(var.shape) != 3:
        raise ValueError('Expected data in (height, width, time_steps) shape')
    if not np.issubdtype(var.dtype, np.floating):
        raise ValueError('Expected data to be in normalized floating-point format')
    if np.min(var) < 0 or np.max(var) > 1:
        raise ValueError('Data contains values outside of [0, 1] range')
    return var


def resolve_readings_variable(data, variable_name):
    if variable_name not in data:
        raise ValueError('Specified variable not found')
    var = data[variable_name]
    while var.dtype == np.object:
        var = var[0]
    if len(var.shape) != 2:
        raise ValueError('Expected data in (num_points, time_steps) shape')
    if not np.issubdtype(var.dtype, np.floating):
        raise ValueError('Expected data to be in floating-point format')
    return var


def load_signal_lookup_pickle(lookup_filename):
    with open(lookup_filename, 'rb') as f:
        return pickle.load(f)
