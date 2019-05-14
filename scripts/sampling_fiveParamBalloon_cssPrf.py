from argparse import ArgumentParser
import os
from scipy.io import loadmat
from qprf_pyro import load_input_file, \
    resolve_stimulus_variable, \
    resolve_readings_variable, \
    TruncatedNormal, \
    LeftTruncatedNormal
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


def create_params(signal_lookup_pickle):

    y = signal_lookup_pickle['y']
    x = signal_lookup_pickle['x']
    rfsize = signal_lookup_pickle['rfsize']

    min_y, max_y = np.min(y), np.max(y)
    min_x, max_x = np.min(x), np.max(x)
    min_rfsize, max_rfsize = np.min(rfsize), np.max(rfsize)

    kappa_mean = pyro.param('kappa_mean', torch.tensor(0.65),
        constraints=constraints.positive)
    gamma_mean = pyro.param('gamma_mean', torch.tensor(0.41),
        constraints=constraints.positive)
    tau_mean = pyro.param('tau_mean', torch.tensor(0.98),
        constraints=constraints.positive)
    grubb_mean = pyro.param('grubb_mean', torch.tensor(0.32),
        constraints=constraints.positive)
    rho_mean = pyro.param('rho_mean', torch.tensor(0.34),
        constraints=constraints.positive)

    y_mean = pyro.param('y_mean', torch.tensor((min_y + max_y) / 2),
        constraints=constraints.interval(min_y, max_y))
    x_mean = pyro.param('x_mean', torch.tensor((min_x + max_x) / 2),
        constraints=constraints.interval(min_x, max_x))
    rfsize_mean = pyro.param('rfsize_mean', torch.tensor((min_rfsize + max_rfsize) / 2),
        constraints=constraints.interval(min_rfsize, max_rfsize))

    expt_mean = pyro.param('expt_mean', torch.tensor(1.0),
        constraints=constraints.positive)
    gain_mean = pyro.param('gain_mean', torch.tensor(1.0),
        constraints=constraints.positive)

    y_stdev = pyro.param('y_stdev', torch.tensor(1000.0 * (max_y - min_y)),
        constraints=constraints.positive)
    x_stdev = pyro.param('x_stdev', torch.tensor(1000.0 * (max_x - min_x)),
        constraints=constraints.positive)
    rfsize_stdev = pyro.param('rfsize_stdev', torch.tensor(1000.0 * (max_rfsize - min_rfsize)),
        constraints=constraints.positive)
    expt_stdev = pyro.param('expt_stdev', torch.tensor(1000.0),
        constraints=constraints.positive)
    gain_stdev = pyro.param('gain_stdev', torch.tensor(1000.0),
        constraints=constraints.positive)

    noise_mean = 0.0
    noise_stdev = pyro.param('noise_stdev', torch.tensor(0.0001),
        constraints=constraints.positive)


    params = {}
    for k, v in locals().items():
        if k.endswith('_mean') or k.endswith('_stdev'):
            params[k] = v

    return params


def get_samples(params, time_steps):
    for var_name in ['kappa', 'gamma', 'grubb',
        'tau', 'rho', 'x', 'y', 'rfsize', 'noise',
        'expt', 'gain']:
        for param_name in ['mean', 'stdev']:
            name = var_name + '_' + param_name
            locals()[name] = params[name]

    kappa = pyro.sample('kappa', dist.Normal(kappa_mean, kappa_stdev))
    gamma = pyro.sample('gamma', dist.Normal(gamma_mean, gamma_stdev))
    grubb = pyro.sample('grubb', dist.Normal(grubb_mean, grubb_stdev))
    tau = pyro.sample('tau', dist.Normal(tau_mean, tau_stdev))
    rho = pyro.sample('rho', dist.Normal(rho_mean, rho_stdev))

    x = pyro.sample('x', TruncatedNormal(x_mean, x_stdev, min_x, max_x))
    y = pyro.sample('y', TruncatedNormal(y_mean, y_stdev, min_y, max_y))
    rfsize = pyro.sample('rfsize', LeftTruncatedNormal(rfsize_mean, rfsize_stdev, 0.0))
    expt = pyro.sample('expt', LeftTruncatedNormal(expt_mean, expt_stdev, 0.0))
    gain = pyro.sample('gain', LeftTruncatedNormal(gain_mean, gain_stdev, 0.0))

    noise = pyro.sample('noise', dist.HalfNormal(noise_stdev), time_steps)

    return kappa, gamma, grubb, tau, rho, x, y, rfsize, expt, gain, noise


def five_param_balloon(params, stimulus_lookup_pickle, device, dtype, time_steps, args):
    # kappa, gamma, grubb, tau, rho
    # y, x, rfsize


    kappa, gamma, grubb, tau, rho, x, y, rfsize, expt, gain, noise = \
        get_samples(params, time_steps)


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
    y = y + noise

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
