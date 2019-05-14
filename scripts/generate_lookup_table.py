from argparse import ArgumentParser
import torch
from scipy.io import loadmat, \
    savemat
import os
import numpy as np
import pickle


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--input-filename', '-i', type=str, required=True)
    parser.add_argument('--output-filename', '-o', type=str,
        default='signal_lookup_table.pickle')
    parser.add_argument('--stimulus-variable-name', '-v', type=str,
        default='stimulus')
    parser.add_argument('--y-begin', '-yb', type=float, default=0)
    parser.add_argument('--x-begin', '-xb', type=float, default=0)
    parser.add_argument('--rfsize-begin', '-rb', type=float, default=1)
    parser.add_argument('--y-end', '-ye', type=float, default=99)
    parser.add_argument('--x-end', '-xe', type=float, default=99)
    parser.add_argument('--rfsize-end', '-re', type=float, default=99)
    parser.add_argument('--y-samples', '-ys', type=int, default=26)
    parser.add_argument('--x-samples', '-xs', type=int, default=26)
    parser.add_argument('--rfsize-samples', '-rs', type=int, default=26)
    parser.add_argument('--device', '-d', type=str,
        default='cpu:0')
    parser.add_argument('--dtype', '-dt', type=str,
        choices=['float', 'double'], default='float')
    return parser


def load_input_file(input_filename, args):
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


def make_gaussian(syy, sxx, y, x, rfsize, device):
    rfsize_sq = rfsize ** 2
    tmp = -0.5 * \
        ( torch.pow(syy - y, 2) / rfsize_sq + \
        torch.pow(sxx - x, 2) / rfsize_sq )
    pdf = torch.exp(tmp) / ( 2 * np.pi * rfsize_sq )
    return pdf


def make_stimulus_grid(height, width, device, dt):
    sy = torch.arange(0, height,
        dtype=dt, device=device)
    sx = torch.arange(0, width,
        dtype=dt, device=device)
    syy, sxx = torch.meshgrid(sy, sx)
    return syy, sxx


def make_sampling_ranges(args):
    y = np.linspace(args.y_begin, args.y_end, args.y_samples)
    x = np.linspace(args.x_begin, args.x_end, args.x_samples)
    rfsize = np.linspace(args.rfsize_begin, args.rfsize_end, args.rfsize_samples)
    return y, x, rfsize


def make_sampling_tasks(y, x, rfsize):
    i = np.arange(0, len(y))
    k = np.arange(0, len(x))
    m = np.arange(0, len(rfsize))
    ii, kk, mm = np.meshgrid(i, k, m, indexing='ij')
    tasks = zip(ii.ravel(), kk.ravel(), mm.ravel())
    tasks = list(tasks)
    return tasks


def generate_lookup_entry(stimulus, syy, sxx, y, x, rfsize, device):
    g = make_gaussian(syy, sxx, y, x, rfsize, device)
    # print('g:', g.shape, g.dtype)
    # print('stimulus:', stimulus.shape, stimulus.dtype)
    res = torch.sum(stimulus * g, (1, 2))
    return res


def generate_lookup_table(stimulus, device, dt, args):
    syy, sxx = make_stimulus_grid(stimulus.shape[1], stimulus.shape[2], device, dt)

    y, x, rfsize = make_sampling_ranges(args)

    res = np.zeros((len(y), len(x), len(rfsize), stimulus.shape[0]))

    tasks = make_sampling_tasks(y, x, rfsize)

    for t, (i, k, m) in enumerate(tasks):
        y_1 = y[i]
        x_1 = x[k]
        rfsize_1 = rfsize[m]
        if t and t % 1000 == 0:
            print('task {}/{}, y_1: {}, x_1: {}, rfsize_1: {}'.format(t, len(tasks), y_1, x_1, rfsize_1))
        signal = generate_lookup_entry(stimulus, syy, sxx, y_1, x_1, rfsize_1, device)
        res[i, k, m, :] = signal.cpu()

    return res, y, x, rfsize


def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    print('device:', device)

    dt = torch.float if args.dtype == 'float' else torch.double
    print('dt:', dt)

    data = load_input_file(args.input_filename, args)
    stimulus = resolve_stimulus_variable(data, args.stimulus_variable_name)
    print('stimulus raw:', stimulus.shape, stimulus.dtype)

    print('stimulus permuted:', stimulus.shape, stimulus.dtype)
    stimulus = stimulus.transpose([2, 0, 1])

    stimulus = torch.tensor(stimulus, dtype=dt, device=device)

    lut, y, x, rfsize = generate_lookup_table(stimulus, device, dt, args)

    res = {
        'lut': lut,
        'lut_dimensions': ['y', 'x', 'rfsize', 't'],
        'y': y,
        'x': x,
        'rfsize': rfsize
    }

    # savemat('signal_lookup_table.mat', res)

    res ['args'] = args

    with open(args.output_filename, 'wb') as f:
        pickle.dump(res, f)

    print('Done.')


if __name__ == '__main__':
    main()
