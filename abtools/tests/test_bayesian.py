# -*- coding: utf-8 -*-
import unittest
import numpy as np
from scipy import stats

from abtools.bayesian.models import BernoulliModel, LognormalModel


class TestModels(unittest.TestCase):

    def test_Bernoulli_model(self):
        a = stats.bernoulli(0.05).rvs(1000)
        model = BernoulliModel(a)
        model.fit()

    def test_lognorm_model(self):
        a = stats.lognorm(1).rvs(1000)
        model = LognormalModel(a)
        model.fit()

if __name__ == '__main__':
    unittest.main()
