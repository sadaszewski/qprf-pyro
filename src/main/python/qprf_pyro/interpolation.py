#
# Copyright (C) Stanislaw Adaszewski, 2019
#


import numpy as np


def signal_interpolation(stimulus_lookup_pickle, y, x, rfsize):
    slp = stimulus_lookup_pickle
    lut = slp['lut']

    y_2 = (y <= slp['y']).nonzero()[0][0]
    x_2 = (x <= slp['x']).nonzero()[0][0]
    rfsize_2 = (rfsize <= slp['rfsize']).nonzero()[0][0]

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
