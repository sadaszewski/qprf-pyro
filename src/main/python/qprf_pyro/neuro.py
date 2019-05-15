import numpy as np
import pyro
import torch


def signal_interpolation(stimulus_lookup_pickle, y, x, rfsize):
    slp = stimulus_lookup_pickle
    lut = slp['lut']

    y_2 = np.where(y <= slp['y'])[0][0]
    x_2 = np.where(x <= slp['x'])[0][0]
    rfsize_2 = np.where(rfsize <= slp['rfsize'])[0][0]

    y_1 = max(0, y_2 - 1)
    x_1 = max(0, x_2 - 1)
    rfsize_1 = max(0, rfsize_2 - 1)

    dy = slp['y'][y_2] - slp['y'][y_1]
    dx = slp['x'][x_2] - slp['x'][x_1]
    drfsize = slp['rfsize'][rfsize_2] - slp['rfsize'][rfsize_1]

    y_frac = (y - slp['y'][y_1]) / dy \
        if dy > 0 else 0
    x_frac = (x - slp['x'][x_1]) / dx \
        if dx > 0 else 0
    rfsize_frac = (rfsize - slp['rfsize'][rfsize_1]) / drfsize \
        if drfsize > 0 else 0

    a = lut[y_1, x_1, rfsize_1]
    b = lut[y_1, x_1, rfsize_2]
    c = lut[y_1, x_2, rfsize_1]
    d = lut[y_1, x_2, rfsize_2]
    e = lut[y_2, x_1, rfsize_1]
    f = lut[y_2, x_1, rfsize_2]
    g = lut[y_2, x_2, rfsize_1]
    h = lut[y_2, x_2, rfsize_2]

    a = b * rfsize_frac + a * (1.0 - rfsize_frac)
    b = d * rfsize_frac + c * (1.0 - rfsize_frac)
    c = f * rfsize_frac + e * (1.0 - rfsize_frac)
    d = h * rfsize_frac + g * (1.0 - rfsize_frac)

    a = b * x_frac + a * (1.0 - x_frac)
    b = d * x_frac + c * (1.0 - x_frac)

    a = b * y_frac + a * (1.0 - y_frac)
    return a


def five_param_balloon_css_prf(stimulus_lookup_pickle,
    device, param_samples, noise_stdev):

    kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain = \
        param_samples


    z = signal_interpolation(stimulus_lookup_pickle, \
        y, x, rfsize)

    z = np.resample(z, len(z) * args.subdivisions)
    read = torch.zeros(len(z), dtype=dtype)

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
        read = pyro.sample('read', dist.Normal(read, pyro.param('noise_stdev')))
    return read
