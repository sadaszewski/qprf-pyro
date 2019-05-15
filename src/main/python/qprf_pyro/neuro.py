import numpy as np
import pyro
import torch
from .interpolation import signal_interpolation


def five_param_balloon_css_prf(stimulus_lookup_pickle,
    param_samples, noise_stdev, repetition_time, subdivisions):
    kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain = \
        param_samples

    z = signal_interpolation(stimulus_lookup_pickle, \
        y, x, rfsize)

    z = np.resample(z, len(z) * subdivisions)
    read = torch.zeros(len(z), dtype=dtype)

    s = 0.0
    f = 1.0
    v = 1.0
    d = 1.0
    t_step = repetition_time / subdivisions
    V0 = 0.02
    k1 = 7 * rho
    k2 = 2
    k3 = 2 * rho - 0.2

    for t in range(len(z)):
        read[t] = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
        ds = z[t] - kappa * s - gamma * (f - 1)
        df = s
        dv = (f - np.pow(v, (1 / grubb))) / tau
        dq = (f * (1 - np.pow((1 - rho), (1 / f))) / rho - np.pow(v, (1 / grubb)) * q / v) / tau
        s += ds * t_step
        f += df * t_step
        v += dv * t_step
        q += dq * t_step

    read = np.copysign(np.pow(np.abs(read), expt.astype(np.double)) * gain, read)
    with pyro.plate('read_plate', len(read)):
        read = pyro.sample('read', dist.Normal(read, noise_stdev))
    return read
