from collections import Counter

import numpy as np

from scipy.stats import normaltest
from numpy.random import choice

from .models import BernoulliModel, LognormalModel, ARPUModel, NormalModel


def naive_model_selector(x):
    if set(x) == set([0, 1]):
        model = BernoulliModel(x)
    elif Counter(x).most_common(1)[0][0] == 0 and np.min(x) >= 0:
        model = ARPUModel(x)
    elif np.min(x) > 0:
        model = LognormalModel(x)
    else:
        model = NormalModel(x)
    print(model.__class__.__name__, 'automatically selected.')
    return model
