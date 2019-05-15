import torch
import pyro
import pyro.distributions as dist


class TruncatedNormal(dist.Rejector):
    def __init__(self, loc, scale, min_x, max_x):
        propose = dist.Normal(loc, scale)

        def log_prob_accept(x):
            return (x > min_x and x < max_x).type_as(x).log()

        log_scale = torch.log(propose.cdf(max_x) - propose.cdf(min_x))
        super().__init__(propose, log_prob_accept, log_scale)


class LeftTruncatedNormal(dist.Rejector):
    def __init__(self, loc, scale, min_x):
        propose = dist.Normal(loc, scale)

        def log_prob_accept(x):
            return (x > min_x).type_as(x).log()

        log_scale = torch.log(1.0 - propose.cdf(min_x))
        super().__init__(propose, log_prob_accept, log_scale)
