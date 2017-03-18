from scipy.stats import norm
from ..utils import se


def zconfint(mean, sd_error, alpha=0.05):
    """
    Compute z-confidence intervals for given mean, standart error and alpha
    """
    z = abs(norm.ppf(alpha/2))
    return mean - z*sd_error, mean + z*sd_error
