from collections import Counter

import numpy as np

from scipy.stats import normaltest
from numpy.random import choice

from .models import BLModel
from .distributions import Bernoulli, Lognormal, Normal


def naive_model_selector(x):
    """
    Select model by simple rule.

    Choose between BernoulliModel, LognormalModel and BL model,
    if no one is suitable NormalModel is selected.

    Parameters
    ----------
        x: list or numpy's array

    Returns
    -------
        model:
            Bayesian model inhereted from Distribution
    """
    if set(x) == set([0, 1]):
        model = Bernoulli(x)
    elif Counter(x).most_common(1)[0][0] == 0 and np.min(x) >= 0:
        model = BLModel(x)
    elif np.min(x) > 0:
        model = Lognormal(x)
    else:
        model = Normal(x)
    print(model.__class__.__name__, 'automatically selected.')
    return model
