import numpy as np


def bl_mean(mu, std, alpha, n):
    """Compute BL's expected value for a given stats."""
    var__alpha = (alpha - 1) / 2
    var__beta = var__alpha * std ** 2
    var = var__beta / (var__alpha - 1)
    return np.exp(mu + var / 2) * (alpha / n)


def bl_mean_from_array(x):
    """Compute BL's expected value for a given array."""
    n = len(x)
    nonzero_x = x[x > 0]
    log_nonzero_x = np.log(nonzero_x)

    mu = log_nonzero_x.mean()
    std = log_nonzero_x.std()
    alpha = log_nonzero_x.shape[0]

    return bl_mean(mu, std, alpha, n)