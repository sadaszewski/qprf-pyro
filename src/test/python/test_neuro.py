#
# Copyright (C) Stanislaw Adaszewski, 2019
#


from qprf_pyro import five_param_balloon, \
    css_prf, \
    iid_noise, \
    load_signal_lookup_pickle, \
    get_priors, \
    get_dists, \
    get_samples
import torch
from scipy.io import savemat


_slp = None


def test_00_load_lookup_pickle():
    global _slp
    _slp = load_signal_lookup_pickle('signal_lookup_table.pickle')
    for name in ['lut', 'y', 'x', 'rfsize']:
        _slp[name] = torch.tensor(_slp[name])
    # _slp['lut'] = torch.tensor(_slp['lut'])


def test_01_neuro():
    priors = get_priors(_slp)
    dists = get_dists(priors, _slp)

    param_samples = get_samples(dists)
    print('param_samples:', param_samples)

    # param_samples['expt'] = torch.tensor(1.)
    # param_samples['gain'] = torch.tensor(1.)

    noise_stdev = 0.001
    repetition_time = 1.51
    subdivisions = 20

    _slp['lut'] = torch.zeros(_slp['lut'].shape, dtype=_slp['lut'].dtype)
    mid = _slp['lut'].shape[3]//2
    _slp['lut'][..., mid:mid+100] = 1.

    read = five_param_balloon(_slp, param_samples, noise_stdev,
        repetition_time, subdivisions)

    read = css_prf(read, param_samples)

    read = iid_noise(read, noise_stdev)

    read = read.cpu().numpy()

    savemat('five_param_balloon_css_prf.mat', { 'read': read })
