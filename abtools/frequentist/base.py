# -*- coding: utf-8 -*-
"""Defines base class for statistical tests."""
import numpy as np


class BaseTest(object):
    """
    Asd.

    asd
    """

    def __init__(self, a, b, alpha=0.05):
        """
        Init.

        dfg
        """
        a, b = np.array(a), np.array(b)
        self.name = 'Base Test'
        self.alpha = alpha
        self.diff = b.mean() - a.mean()
        self.statistic = self.compute_test_statistic(a, b)
        self.critical = self.compute_critical(a, b)
        self.p_value = self.compute_p_value(a, b)

        self.sign = self.p_value <= self.alpha
        self.confidence_intervals = self.compute_confidence_intervals(a, b)

        if self.statistic > self.critical:
            self.result = 'E(A) < E(B)'
        elif self.statistic < -self.critical:
            self.result = 'E(A) > E(B)'
        else:
            self.result = 'E(A) = E(B)'

    def compute_test_statistic(self, a, b):
        raise NotImplementedError

    def compute_critical(self, a, b):
        raise NotImplementedError

    def compute_p_value(self, a, b):
        raise NotImplementedError

    def compute_confidence_intervals(self, a, b):
        raise NotImplementedError

    def summary(self):
        """Print test summary."""
        print("Observed difference of means (E(B) - E(A)) = %.4f"
              % self.diff)
        print("Test statistic is %.4f and critical value is %.4f"
              % (self.statistic, self.critical))

        print("Two-sided p-value = %.2f" % self.p_value)

        print("Null hypotesis is %s with %.2f significance level"
              % (('not ' * (1 * (not self.sign)) + 'rejected'), self.alpha))

        print(self.result)
