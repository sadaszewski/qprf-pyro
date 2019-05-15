from argparse import ArgumentParser
import os
from scipy.io import loadmat
from qprf_pyro import load_input_file, \
    resolve_stimulus_variable, \
    resolve_readings_variable, \
    TruncatedNormal, \
    LeftTruncatedNormal, \
    get_ranges_info, \
    get_priors, \
    create_params, \
    get_dists
import torch
import pyro
import pyro.infer
import pyro.distributions as dist
from torch.distributions import constraints


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
    return parser





def model(stimulus_lookup_pickle, time_steps, device):
    priors = get_priors(stimulus_lookup_pickle)
    model_params = get_samples(priors, time_steps)
    res = five_param_balloon(stimulus_lookup_pickle, device, model_params)
    return res


def guide(stimulus_lookup_pickle, time_steps, device):
    params = create_params(stimulus_variable_name)
    model_params = get_samples(params, time_steps)
    res = five_param_balloon(stimulus_lookup_pickle, device, model_params)
    return res





def exec_():
    pyro.infer.SVI(model=five_param_balloon,
        guide=five_param_balloon,
        )


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
    readings = torch.tensor(readings,
        dtype=dtype, device=device)




if __name__ == '__main__':
    main()
