import numpy as np
import pyro
import torch
from .interpolation import signal_interpolation
from torch.nn.functional import interpolate
import pdb
import pyro.distributions as dist


def five_param_balloon(stimulus_lookup_pickle,
    param_samples, noise_stdev, repetition_time, subdivisions):
    kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain = \
        param_samples

    # print('stimulus_lookup_pickle:', stimulus_lookup_pickle)
    # print('y: {}, x: {}, rfsize: {}'.format(y, x, rfsize))
    # print('type(y):', type(y), 'type(lut):', type(stimulus_lookup_pickle['lut']))
    z = signal_interpolation(stimulus_lookup_pickle, \
        y, x, rfsize)

    total_time = len(z)

    z = interpolate(z.view(1, 1, -1), size=len(z) * subdivisions,
        mode='linear', align_corners=True).view(-1)
    read = torch.zeros(len(z), dtype=z.dtype, device=z.device)

    def mktensor(val):
        return torch.tensor(val, dtype=z.dtype, device=z.device)

    s = mktensor(0.)
    f = mktensor(1.)
    v = mktensor(1.)
    q = mktensor(1.)
    t_step = mktensor(repetition_time / subdivisions)
    V0 = mktensor(.02)
    k1 = 7. * rho
    k2 = mktensor(2.)
    k3 = 2. * rho - 0.2

    # pdb.set_trace()

    for t in range(len(z)):
        read[t] = V0 * ( k1 * (1. - q) + \
            k2 * (1. - q / v) + \
            k3 * (1. - v) )
        ds = z[t] - kappa * s - gamma * (f - 1.)
        df = s
        dv = ( f - torch.pow( v, (1. / grubb) ) ) / tau
        dq = ( f * ( 1. - torch.pow( (1. - rho), (1. / f) ) ) / rho - \
            torch.pow( v, (1. / grubb) ) * q / v ) / tau
        s = s + ds * t_step
        f = f + df * t_step
        v = v + dv * t_step
        q = q + dq * t_step

    read = interpolate(read.view(1, 1, -1), size=total_time,
        mode='linear', align_corners=True).view(-1)

    return read


def css_prf(read, param_samples):
    _, _, _, _, _, _, _, _, expt, gain = param_samples
    read = torch.sign(read) * torch.pow(torch.abs(read), expt) * gain
    return read


def iid_noise(read, noise_stdev):
    with pyro.plate('noise_plate', len(read)):
        read = pyro.sample('read', dist.Normal(read, noise_stdev))
    return read
