import numpy as np
from scipy.stats import norm


def se(std, n):
    """
    Compute standart error for given std and array length
    """
    return std / np.sqrt(n)


def significance_probability(a, b):
    return max(2 * norm.cdf(abs(a.mean() - b.mean()) /
               (se(a, len(a)) + se(b, len(b)))) - 1, 0)


def significance(a, b):
    return not((a[0] <= b[0] <= a[1]) or (a[0] <= b[1] <= a[1])
               or (b[0] <= a[1] <= b[1]) or (b[0] <= a[0] <= b[1]))
