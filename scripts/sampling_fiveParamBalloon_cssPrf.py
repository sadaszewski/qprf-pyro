from argparse import ArgumentParser
import os
from qprf_pyro import load_input_file, \
    resolve_stimulus_variable, \
    resolve_readings_variable, \
    load_signal_lookup_pickle, \
    get_priors, \
    create_params, \
    five_param_balloon, \
    css_prf, \
    iid_noise, \
    detrend_readings, \
    get_param_names, \
    get_samples, \
    get_dists
import torch
import pyro
import pyro.infer
import pyro.optim
from collections import defaultdict
from scipy.io import savemat
import pdb


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--input-filename', '-i', type=str, required=True)
    # parser.add_argument('--lookup-table-filename', '-t', type=str, required=True)
    parser.add_argument('--output-filename', '-o', type=str, required=True)
    parser.add_argument('--stimulus-variable-name', '-sv', type=str,
        default='stimulus')
    parser.add_argument('--readings-variable-name', '-rv', type=str,
        default='data')
    parser.add_argument('--repetition-time', '-rt', type=float,
        default=1.0)
    parser.add_argument('--subdivisions', '-s', type=int,
        default=20)
    parser.add_argument('--device', '-d', type=str,
        default='cpu:0')
    parser.add_argument('--data-type', '-dt', type=str,
        default='float', choices=['float', 'double'])
    parser.add_argument('--detrend-order', '-do', type=int,
        default=4)
    parser.add_argument('--number-of-steps', '-n', type=int,
        default=2500)
    parser.add_argument('--signal-lookup-pickle', '-l', type=str,
        required=True)
    return parser



def model(signal_lookup_pickle, time_steps,
    repetition_time, subdivisions):

    priors = get_priors(signal_lookup_pickle)
    dists = get_dists(priors, signal_lookup_pickle)
    model_params = get_samples(dists)
    res = five_param_balloon(signal_lookup_pickle, model_params,
        priors['noise_stdev'], repetition_time, subdivisions)
    res = css_prf(res, model_params)
    res = iid_noise(res, priors['noise_stdev'])
    return res


def guide(signal_lookup_pickle, time_steps,
    repetition_time, subdivisions):

    params = create_params(signal_lookup_pickle)
    dists = get_dists(params, signal_lookup_pickle)
    model_params = get_samples(dists)
    res = five_param_balloon(signal_lookup_pickle, model_params,
        params['noise_stdev'], repetition_time, subdivisions)
    res = css_prf(res, model_params)
    res = iid_noise(res, params['noise_stdev'])
    return res


def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float \
        if args.data_type == 'float' \
        else torch.double

    data = load_input_file(args.input_filename)
    stimulus = resolve_stimulus_variable(data,
        args.stimulus_variable_name)
    readings = resolve_readings_variable(data,
        args.readings_variable_name)

    stimulus = stimulus.transpose([2, 0, 1])
    stimulus = torch.tensor(stimulus,
        dtype=dtype, device=device)

    readings = detrend_readings(readings, args.detrend_order)
    readings = torch.tensor(readings,
        dtype=dtype, device=device)

    time_points = stimulus.shape[0]

    signal_lookup_pickle = load_signal_lookup_pickle(args.signal_lookup_pickle,
        dtype=dtype, device=device)

    param_names = get_param_names()

    pyro.clear_param_store()
    conditioned_model = pyro.condition(model, data = { 'read': readings[0] })
    svi = pyro.infer.SVI(model=conditioned_model,
        guide=guide,
        optim=pyro.optim.SGD({ 'lr': 0.001, 'momentum': 0.1 }),
        loss=pyro.infer.Trace_ELBO())

    # torch.autograd.set_detect_anomaly(True)

    traces = defaultdict(lambda: [])
    for i in range(args.number_of_steps):
        print('i:', i)
        loss = svi.step(signal_lookup_pickle, time_points,
            args.repetition_time, args.subdivisions)
        print('loss:', loss)
        traces['loss'].append(loss)
        for name in param_names:
            if name == 'noise_mean':
                continue
            traces[name].append(pyro.param(name).item())

    savemat(args.output_filename, { 'traces': traces })

if __name__ == '__main__':
    main()
