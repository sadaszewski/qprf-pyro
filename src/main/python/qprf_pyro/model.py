import numpy as np
import pyro
from torch.distributions import constraints
import torch
import pyro.distributions as dist
from .distributions import TruncatedNormal, \
    LeftTruncatedNormal


def get_param_names():
    res = []
    for var_name in ['kappa', 'gamma', 'grubb',
        'tau', 'rho', 'y', 'x', 'rfsize',
        'expt', 'gain', 'noise']:
        for stat in ['mean', 'stdev']:
            name = var_name + '_' + stat
            res.append(name)
    return res


def get_ranges_info(signal_lookup_pickle):
    y = signal_lookup_pickle['y']
    x = signal_lookup_pickle['x']
    rfsize = signal_lookup_pickle['rfsize']

    min_y, max_y = torch.min(y), torch.max(y)
    min_x, max_x = torch.min(x), torch.max(x)
    min_rfsize, max_rfsize = torch.min(rfsize), torch.max(rfsize)

    return min_y, max_y, min_x, max_x, min_rfsize, max_rfsize


def get_priors(signal_lookup_pickle):
    min_y, max_y, min_x, max_x, min_rfsize, max_rfsize = \
        get_ranges_info(signal_lookup_pickle)

    res = {
        'kappa_mean': 0.65,
        'gamma_mean': 0.41,
        'tau_mean': 0.98,
        'grubb_mean': 0.32,
        'rho_mean': 0.34,

        'y_mean': (min_y + max_y) / 2,
        'x_mean': (min_x + max_x) / 2,
        'rfsize_mean': (min_rfsize + max_rfsize) / 2,

        'expt_mean': 1.0,
        'gain_mean': 200.0,

        'kappa_stdev': 0.015,
        'gamma_stdev': 0.002,
        'tau_stdev': 0.0568,
        'grubb_stdev': 0.0015,
        'rho_stdev': 0.0024,

        'y_stdev': 2.0 * (max_y - min_y),
        'x_stdev': 2.0 * (max_x - min_x),
        'rfsize_stdev': 2.0 * (max_rfsize - min_rfsize),
        'expt_stdev': 0.1,
        'gain_stdev': 1.0,

        'noise_mean': 0.0,
        'noise_stdev': 0.0001
    }

    lut = signal_lookup_pickle['lut']

    res = { k: v \
        if isinstance(v, torch.Tensor) \
        else torch.tensor(v, dtype=lut.dtype, device=lut.device) \
        for k, v in res.items() }

    return res


def create_params(signal_lookup_pickle):
    min_y, max_y, min_x, max_x, min_rfsize, max_rfsize = \
        get_ranges_info(signal_lookup_pickle)

    priors = get_priors(signal_lookup_pickle)

    def positive_param(name):
        return pyro.param(name, priors[name],
            constraint=constraints.positive)

    def interval_param(name, min_, max_):
        return pyro.param(name, priors[name],
            constraint=constraints.interval(min_, max_))

    kappa_mean, gamma_mean, tau_mean, grubb_mean, rho_mean = \
        (positive_param(name) for name in ['kappa_mean', 'gamma_mean',
            'tau_mean', 'grubb_mean', 'rho_mean'])

    kappa_stdev, gamma_stdev, tau_stdev, grubb_stdev, rho_stdev = \
        (positive_param(name) for name in ['kappa_stdev', 'gamma_stdev',
            'tau_stdev', 'grubb_stdev', 'rho_stdev'])

    y_mean = interval_param('y_mean', min_y, max_y)
    x_mean = interval_param('x_mean', min_x, max_x)
    rfsize_mean = interval_param('rfsize_mean', min_rfsize, max_rfsize)

    expt_mean = positive_param('expt_mean')
    gain_mean = positive_param('gain_mean')

    y_stdev = positive_param('y_stdev')
    x_stdev = positive_param('x_stdev')
    rfsize_stdev = positive_param('rfsize_stdev')

    expt_stdev = positive_param('expt_stdev')
    gain_stdev = positive_param('gain_stdev')

    noise_mean = 0.0
    noise_stdev = positive_param('noise_stdev')

    params = {}
    for k, v in locals().items():
        if k.endswith('_mean') or k.endswith('_stdev'):
            params[k] = v

    return params


def get_dists(param_values, signal_lookup_pickle):
    min_y, max_y, min_x, max_x, min_rfsize, max_rfsize = \
        get_ranges_info(signal_lookup_pickle)

    p = lambda: 0
    for var_name in ['kappa', 'gamma', 'grubb',
        'tau', 'rho', 'x', 'y', 'rfsize', 'noise',
        'expt', 'gain']:
        for param_name in ['mean', 'stdev']:
            name = var_name + '_' + param_name
            setattr(p, name, param_values[name])

    kappa = LeftTruncatedNormal(p.kappa_mean, p.kappa_stdev, 0.0)
    gamma = LeftTruncatedNormal(p.gamma_mean, p.gamma_stdev, 0.0)
    grubb = LeftTruncatedNormal(p.grubb_mean, p.grubb_stdev, 0.0)
    tau = LeftTruncatedNormal(p.tau_mean, p.tau_stdev, 0.0)
    rho = LeftTruncatedNormal(p.rho_mean, p.rho_stdev, 0.0)

    y = TruncatedNormal(p.y_mean, p.y_stdev, min_y, max_y)
    x = TruncatedNormal(p.x_mean, p.x_stdev, min_x, max_x)
    rfsize = TruncatedNormal(p.rfsize_mean, p.rfsize_stdev, min_rfsize, max_rfsize)
    expt = LeftTruncatedNormal(p.expt_mean, p.expt_stdev, 0.0)
    gain = LeftTruncatedNormal(p.gain_mean, p.gain_stdev, 0.0)

    return kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain


def get_samples(dists):
    kappa_d, gamma_d, grubb_d, tau_d, rho_d, x_d, y_d, \
    rfsize_d, expt_d, gain_d = dists

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

    return kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain
