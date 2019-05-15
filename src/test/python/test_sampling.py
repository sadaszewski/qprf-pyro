from qprf_pyro import load_input_file, \
    resolve_stimulus_variable, \
    resolve_readings_variable, \
    load_signal_lookup_pickle, \
    get_ranges_info, \
    get_priors, \
    create_params, \
    get_dists, \
    TruncatedNormal, \
    LeftTruncatedNormal, \
    get_samples, \
    signal_interpolation
import numpy as np
import pyro


_data_filename = 'examples/simulated_run_9/simulated_run.mat'
_lookup_filename = 'signal_lookup_table.pickle'
_slp = None


def test_00_load_data():
    data = load_input_file(_data_filename)
    stimulus = resolve_stimulus_variable(data, 'stimulus')
    readings = resolve_readings_variable(data, 'data')
    assert len(stimulus.shape) == 3
    assert len(readings.shape) == 2
    assert stimulus.shape[2] == readings.shape[1]


def test_01_load_lookup_pickle():
    global _slp
    _slp = load_signal_lookup_pickle(_lookup_filename)
    assert 'lut' in _slp
    assert 'lut_dimensions' in _slp
    assert _slp['lut_dimensions'][:3] == ['y', 'x', 'rfsize']
    assert 'x' in _slp
    assert 'y' in _slp
    assert 'rfsize' in _slp
    slp_1 = dict(_slp)
    del slp_1['lut']
    print(slp_1)


def test_02_get_ranges_info():
    fake_slp = {
        'x': np.random.random(10),
        'y': np.random.random(10),
        'rfsize': np.random.random(10)
    }
    min_y, max_y, min_x, max_x, min_rfsize, max_rfsize = \
        get_ranges_info(fake_slp)
    assert min_y == np.min(fake_slp['y'])
    assert max_y == np.max(fake_slp['y'])
    assert min_x == np.min(fake_slp['x'])
    assert max_x == np.max(fake_slp['x'])
    assert min_rfsize == np.min(fake_slp['rfsize'])
    assert max_rfsize == np.max(fake_slp['rfsize'])


def test_03_get_priors():
    priors = get_priors(_slp)
    print(priors)
    for name in ['kappa', 'gamma', 'grubb', 'tau', 'rho',
        'x', 'y', 'rfsize', 'expt', 'gain']:
        for stat in ['mean', 'stdev']:
            assert (name + '_' + stat) in priors


def test_04_create_params():
    params = create_params(_slp)
    print(params)
    for var_name in ['kappa', 'gamma', 'grubb', 'tau', 'rho',
        'x', 'y', 'rfsize', 'expt', 'gain']:
        for stat in ['mean', 'stdev']:
            name = var_name + '_' + stat
            assert name in params
            assert pyro.param(name) == params[name]


def test_05_get_dists():
    priors = get_priors(_slp)
    params = create_params(_slp)
    assert priors.keys() == params.keys()
    print(priors.keys())
    model_dists = get_dists(priors, _slp)
    guide_dists = get_dists(params, _slp)
    # assert model_dists.keys() == guide_dists.keys()
    assert len(model_dists) == 10
    assert len(guide_dists) == 10
    # print('model_dists:', model_dists)
    # print('guide_dists:', guide_dists)
    kappa, gamma, grubb, tau, rho, y, x, rfsize, expt, gain = \
        model_dists
    assert isinstance(kappa, LeftTruncatedNormal)
    assert isinstance(gamma, LeftTruncatedNormal)
    assert isinstance(grubb, LeftTruncatedNormal)
    assert isinstance(tau, LeftTruncatedNormal)
    assert isinstance(rho, LeftTruncatedNormal)
    assert isinstance(y, TruncatedNormal)
    assert isinstance(x, TruncatedNormal)
    assert isinstance(rfsize, TruncatedNormal)
    assert isinstance(expt, LeftTruncatedNormal)
    assert isinstance(gain, LeftTruncatedNormal)


def test_06_get_samples():
    params = create_params(_slp)
    guide_dists = get_dists(params, _slp)
    samples = get_samples(guide_dists)
    assert len(samples) == 10
    print('samples:', samples)
