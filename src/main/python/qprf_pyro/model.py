import numpy as np


def get_ranges_info(signal_lookup_pickle):
    y = signal_lookup_pickle['y']
    x = signal_lookup_pickle['x']
    rfsize = signal_lookup_pickle['rfsize']

    min_y, max_y = np.min(y), np.max(y)
    min_x, max_x = np.min(x), np.max(x)
    min_rfsize, max_rfsize = np.min(rfsize), np.max(rfsize)

    return min_y, max_y, min_x, max_x, min_rfsize, max_rfsize
