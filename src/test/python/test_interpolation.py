from qprf_pyro import signal_interpolation, \
    load_signal_lookup_pickle
import numpy as np


_slp = None


def _signal_interpolation_debug(signal_lookup_pickle, y, x, rfsize):
    slp = signal_lookup_pickle
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

    print('slp[\'x\']:', slp['x'])
    print('slp[\'y\']:', slp['y'])
    print('slp[\'rfsize\']:', slp['rfsize'])

    print(('y: {}, x: {}, rfsize: {}\n' +
        'y_1: {}, x_1: {}, rfsize_1: {}\n' +
        'y_2: {}, x_2: {}, rfsize_2: {}\n' +
        'dy: {}, dx: {}, drfsize: {}\n' +
        'y_frac: {}, x_frac: {}, rfsize_frac: {}').format(
        y, x, rfsize,
        y_1, x_1, rfsize_1,
        y_2, x_2, rfsize_2,
        dy, dx, drfsize,
        y_frac, x_frac, rfsize_frac))

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


def test_00_load_lookup_pickle():
    global _slp
    _slp = load_signal_lookup_pickle('signal_lookup_table.pickle')


def test_01_signal_interpolation():
    y = 16
    x = 81
    rfsize = 27
    interpolated = signal_interpolation(_slp, y, x, rfsize)
    check = _signal_interpolation_debug(_slp, y, x, rfsize)
    print('len(interpolated):', len(interpolated))
    print('len(check):', len(check))
    ok = (check == interpolated).all()
    assert ok is True
