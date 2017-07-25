# -*- coding: utf-8 -*-
import numpy as np

from scipy.stats.kde import gaussian_kde
from scipy.stats import norm


def KL(a, b, normalize=True):
    """
    Compute K-L divergence for two arrays of values.

    At first compute probability density function with Gaussian KDE, then
    apply it to linear space of values to get probability distribution. At last
    compute K-L divergence for given probability distributions.

    Parameters
    ----------
    a, b : list
        Arrays of values
    normalize : bool
        Perform or not normalization on probability distribution

    Return
    ------
    float
        Value of K-L divergence for two probability distributions of given data
    """
    a, b = np.array(a), np.array(b)

    x = np.linspace(
        min(a.min(), b.min()) - 1,
        max(a.max(), b.max()) + 1,
        100
    )

    p = gaussian_kde(a)(x)
    q = gaussian_kde(b)(x)

    if normalize:
        p = p/np.sum(p)
        q = q/np.sum(q)

    return np.sum(np.where(p != 0, (p) * np.log(p / q), 0))
