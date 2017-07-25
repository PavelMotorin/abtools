# -*- coding: utf-8 -*-
import numpy as np

from scipy.stats.kde import gaussian_kde
from scipy.stats import norm


def se(std, n):
    """
    Compute standart error for given std and array length.
    """
    return std / np.sqrt(n)


def notintersect(a, b):
    """
    Check for intersection of two intervals defined by pair of values.
    """
    b_in_a = (a[0] <= b[0] <= a[1]) or (a[0] <= b[1] <= a[1])
    a_in_b = (b[0] <= a[1] <= b[1]) or (b[0] <= a[0] <= b[1])
    return not (b_in_a or a_in_b)


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
