from qprf_pyro import load_input_file, \
    resolve_stimulus_variable, \
    resolve_readings_variable, \
    load_signal_lookup_pickle, \
    get_ranges_info
import numpy as np


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
    del _slp['lut']
    print(_slp)


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
