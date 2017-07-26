# -*- coding: utf-8 -*-

import scipy.stats
import numpy as np
from math import sqrt


class BaseModel:
    def __init__(self, x):
        raise NotImplementedError()

    def rvs(self, samples=100000):
        return self._rvs(samples)

    def _rvs(self, samples):
        raise NotImplementedError()


class StatTest:
    def __init__(self, a, b):
        raise NotImplementedError()

    def test(self, alpha=0.05, tail='both', a_name='E(A)', b_name='E(B)'):
        p = self.probability()
        p_critical = self.critical(alpha, tail)
        if p > p_critical[1]:
            return ' < '.join([a_name, b_name])
        elif p < p_critical[0]:
            return ' > '.join([a_name, b_name])
        else:
            return ' = '.join([a_name, b_name])

    def critical(self, alpha=0.05, tail='both'):
        if tail == 'left':
            return [alpha, 1]
        elif tail == 'right':
            return [0, 1 - alpha]
        else:
            return [alpha / 2, 1 - alpha / 2]

    def probability(self):
        return self._probability()

    def _probability(self):
        raise NotImplementedError()
