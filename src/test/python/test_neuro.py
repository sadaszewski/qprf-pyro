from qprf_pyro import five_param_balloon_css_prf, \
    load_signal_lookup_pickle, \
    get_priors, \
    get_dists, \
    get_samples


_slp = None


def test_00_load_lookup_pickle():
    global _slp
    _slp = load_signal_lookup_pickle('signal_lookup_table.pickle')


def test_01_neuro():
    priors = get_priors(_slp)
    dists = get_dists(priors, _slp)
    
    param_samples = get_samples(dists)
    noise_stdev = 0.01
    repetition_time = 1.51
    subdivisions = 20

    five_param_balloon_css_prf(_slp, param_samples, noise_stdev,
        repetition_time, subdivisions)
