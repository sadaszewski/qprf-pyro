#
# Copyright (C) Stanislaw Adaszewski, 2019
#


import os
from scipy.io import loadmat
import numpy as np
import pickle
import torch


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


def load_signal_lookup_pickle(lookup_filename, dtype, device):
    with open(lookup_filename, 'rb') as f:
        res = pickle.load(f)
    for name in ['lut', 'y', 'x', 'rfsize']:
        res[name] = torch.tensor(res[name], dtype=dtype, device=device)
    return res


def detrend_readings(data, order=4):
    if len(data.shape) != 2:
        raise ValueError('Expected data in (vertices, time_points) shape')
    data = data.copy()
    a = np.linspace(0, 1, data.shape[1])
    b = []
    for i in range(0, order+1):
        b.append(a ** i)
    b = np.array(b).T
    c = np.linalg.pinv(b)
    d = np.matmul(b, c)

    for i in range(len(data)):
        data[i] -= np.matmul(d, data[i])

    return data
