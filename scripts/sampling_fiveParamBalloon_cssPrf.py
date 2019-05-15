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





def get_samples(params, time_steps):
    kappa_d, gamma_d, grubb_d, tau_d, rho_d, x_d, y_d, \
    rfsize_d, expt_d, gain_d, noise_d = \
        get_dists(params)

    kappa = pyro.sample('kappa', kappa_d)
    gamma = pyro.sample('gamma', gamma_d)
    grubb = pyro.sample('grubb', grubb_d)
    tau = pyro.sample('tau', tau_d)
    rho = pyro.sample('rho', rho_d)

    y = pyro.sample('y', y_d)
    x = pyro.sample('x', x_d)
    rfsize = pyro.sample('rfsize', rfsize_d)
    expt = pyro.sample('expt', expt_d)
    gain = pyro.sample('gain', gain_d)

    noise = pyro.sample('noise', noise_d, time_steps)

    return kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain, noise


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


def five_param_balloon_css_prf(stimulus_lookup_pickle,
    device, model_params):

    kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain, noise = \
        model_params

    slp = stimulus_lookup_pickle

    y_1 = np.where(y >= slp['y'])[0][0]
    x_1 = np.where(x >= slp['x'])[0][0]
    rfsize_1 = np.where(rfsize >= slp['rfsize'])[0][0]

    y_2 = min(slp['lut'].shape[0] - 1, y_1 + 1)
    x_2 = min(slp['lut'].shape[1] - 1, x_1 + 1)
    rfsize_2 = min(slp['lut'].shape[2] - 1, rfsize_1 + 1)

    y_frac = (y - slp['y'][y_1]) / \
        (slp['y'][y_2] - slp['y'][y_1])
    x_frac = (x - slp['x'][x_1]) / \
        (slp['x'][x_2] - slp['x'][x_1])
    rfsize_frac = (rfsize - slp['rfsize'][rfsize_1]) / \
        (slp['rfsize'][rfsize_2] - slp['rfsize'][rfsize_1])

    a = slp['lut'][y_1, x_1, rfsize_1]
    b = slp['lut'][y_1, x_1, rfsize_2]
    c = slp['lut'][y_1, x_2, rfsize_1]
    d = slp['lut'][y_1, x_2, rfsize_2]
    e = slp['lut'][y_2, x_1, rfsize_1]
    f = slp['lut'][y_2, x_1, rfsize_2]
    g = slp['lut'][y_2, x_2, rfsize_1]
    h = slp['lut'][y_2, x_2, rfsize_2]

    a = b * rfsize_frac + a * (1.0 - rfsize_frac)
    b = d * rfsize_frac + c * (1.0 - rfsize_frac)
    c = f * rfsize_frac + e * (1.0 - rfsize_frac)
    d = h * rfsize_frac + g * (1.0 - rfsize_frac)

    a = b * x_frac + a * (1.0 - x_frac)
    b = d * x_frac + c * (1.0 - x_frac)

    a = b * y_frac + a * (1.0 - y_frac)

    z = np.resample(a, len(a) * args.subdivisions)
    y = torch.zeros(len(z), dtype=dtype)

    s = 0.0
    f = 1.0
    v = 1.0
    d = 1.0
    t_step = args.repetition_time / args.subdivisions
    V0 = 0.02
    k1 = 7 * rho
    k2 = 2
    k3 = 2 * rho - 0.2

    for t in range(len(z)):
        y[t] = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
        ds = z[t] - kappa * s - gamma * (f - 1)
        df = s
        dv = (f - np.pow(v, (1 / grubb))) / tau
        dq = (f * (1 - np.pow((1 - rho), (1 / f))) / rho - np.pow(v, (1 / grubb)) * q / v) / tau
        s += ds * t_step
        f += df * t_step
        v += dv * t_step
        q += dq * t_step

    y = np.copysign(np.pow(np.abs(y), expt.astype(np.double)) * gain, y)
    # y = y + noise
    with pyro.plate('y_plate', len(y)):
        pyro.sample('y', dist.Normal(y, pyro.param('noise_stdev')))
        return y


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
