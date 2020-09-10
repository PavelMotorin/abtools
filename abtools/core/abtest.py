# -*- coding: utf-8 -*-

from itertools import combinations
from tqdm import tqdm

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from .base import StatTest
from .model_selection import naive_model_selector

CLRS_LT = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
CLRS_DK = ['#2980b9', '#c0392b', '#27ae60', '#f1c40f']

__all__ = [
    'ABtest',
    'ZTest',
    'PermutationTest',
    'BTest',
    # 'TTest'
]


class ABtest(object):
    """
    AB test class.
    Class.
    """
    def __init__(self, groups, model=naive_model_selector, samples=5000, group_names=None, alpha=0.05):
        print('ABtest for %d groups' % len(groups))
        if group_names:
          self.models = {
                'group%d' % (i + 1): model(group)
                for i, group in enumerate(groups)
            }
        else:
          self.models = {
              group_names[i]: model(group)
              for i, group in enumerate(groups)
          }
        self.alpha = alpha
        self.samples = samples

    def test(self):
        raise NotImplementedError

    def test_all(self):

        self.deltas = [
            (self.models[b].rvs(self.samples) -
             self.models[a].rvs(self.samples), (a, b))
            for a, b in combinations(sorted(self.models.keys()), 2)
        ]

        self.means = np.array([
            (name, np.mean(self.models[name].rvs(self.samples)))
            for name in self.models
        ])

        best_ind = np.argmax([mean[0] for mean in self.means])
        worst_ind = np.argmin([mean[0] for mean in self.means])

        self.probabilities = [
            (np.mean(delta[0] < 0), np.mean(delta[0] > 0), delta[1])
            for delta in self.deltas
        ]

        # put computed probabilities to DataFrame
        self.probabilities_df = pd.DataFrame(
            index=sorted(self.models.keys()),
            columns=sorted(self.models.keys())
        )
        for p in self.probabilities:
            self.probabilities_df.loc[p[2][0], p[2][1]] = p[1]
            self.probabilities_df.loc[p[2][1], p[2][0]] = p[0]
        means = pd.Series(self.means[:, 1], self.means[:, 0])
        self.probabilities_df['mean'] = means
        self.probabilities_df.loc['mean'] = means
        self.probabilities_df = self.probabilities_df.fillna('-')

        return self

    def plot(self):
        n = len(self.deltas)
        fig, axs = plt.subplots(n, figsize=(10, 3*n))
        fig.subplots_adjust(hspace=0.3)
        if n == 1:
            axs = [axs]
        for ax, delta in zip(axs, self.deltas):
            ax.hist(delta[0], bins=20, alpha=0.75)
            ax.set_title('%s vs %s' % delta[1])
            ax.axvline(0, c='r', linestyle='--')
            ax.text(
                0.0,
                0.8,
                """
                mean $\Delta$ = %.4f
                $P$(%s > %s) = %.2f
                $P$(%s > %s) = %.2f
                """ % (
                        np.mean(delta[0]),
                        delta[1][0], delta[1][1],
                        np.mean(delta[0] < 0),
                        delta[1][1], delta[1][0],
                        np.mean(delta[0] > 0),

                      ),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes
            )


class PermutationTest(object):
    """
    Implementation of Fisher's permutation test.
    A Fisher's exact permutation test is a type of statistical significance
    test in which the distribution of the test statistic under the
    null hypothesis is obtained by calculating all possible values of the test
    statistic under rearrangements of the labels on the observed data points.
    In other words, the method by which treatments are allocated to subjects in
    an experimental design is mirrored in the analysis of that design. If the
    labels are exchangeable under the null hypothesis, then the resulting tests
    yield exact significance levels.
    Parameters
    ----------
    a, b : {list, ndarray}
        Observed data for two groups, where a - control group,  b - test group.
    alpha : float
        Significance level.
    Returns
    -------
    p_value : float
        Two-sided p-value.
    sign : bool
        Rejected or not null hypotesis with given significance level.
    References
    ----------
    [1] - Fisher, R. A. (1935). The design of experiments. 1935.
    Oliver and Boyd, Edinburgh.
    [2] - Ernst, M. D. (2004). Permutation methods: basis for exact inference.
    Statistical Science, 19(4), 676-685
    """

    def __init__(self, a, b, alpha=0.05):
        a, b = np.array(a), np.array(b)
        self.name = 'Permutation Test'
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
        return self.diff

    def compute_critical(self, a, b):
        z = np.concatenate([a, b])
        n = len(a)
        null_dist = []
        # start permutations
        for j in tqdm(range(10000)):
            np.random.shuffle(z)
            null_dist.append(np.mean(z[:n]) - np.mean(z[n:]))

        self.null_dist = np.array(null_dist)
        return np.abs(np.percentile(self.null_dist, 1 - self.alpha / 2))

    def compute_p_value(self, a, b):
        return np.mean(np.abs(self.diff) < np.abs(self.null_dist))

    def compute_confidence_intervals(self, a, b):
        ci = [np.percentile(self.null_dist, p*100)
              for p in [0 + self.alpha/2, 1 - self.alpha/2]]
        return ci

    def plot(self, plt_kwds={}, hist_kwds={}):
        """Plot null distribution of permutation test."""
        fig = plt.figure(**plt_kwds)
        plt.hist(
            self.null_dist,
            50,
            histtype='step',
            color=CLRS_LT[1],
            label='Null distribution',
            **hist_kwds
        )

        plt.legend()
        ci_lbl = ('%d%% confidence interval' % ((1-self.alpha) * 100))
        plt.axvline(self.diff, label='Observed difference')
        plt.axvline(self.confidence_intervals[0], linestyle='--')
        plt.axvline(self.confidence_intervals[1], linestyle='--', label=ci_lbl)

        plt.title("Fisher's permutation test")
        plt.legend(bbox_to_anchor=(1.05, .65))

        
class StudentTest(object):

    def __init__(self, a, b, alpha=0.05):
        """
        
        Calculate the T-test for the means of two independent samples of scores. 
        This is a two-sided test for the null hypothesis that 
        2 independent samples have identical average (expected) values. 
        This test assumes that the populations have identical variances by default.
        Parameters
        ----------
        a, b : {list, ndarray}
            Observed data for two groups, where a - control group,  b - test group.
        alpha : float
            Significance level.
        Returns
        -------
        p_value : float
            Two-sided p-value.
        sign : bool
            Rejected or not null hypotesis with given significance level.
        """
        a, b = np.array(a), np.array(b)
        self.name = 'Student Test'
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

    
    def ci(self, x):
        x = np.array(x)
        n = len(x)
        m, se = np.mean(x), scipy.stats.sem(x)
        h = se * scipy.stats.t.ppf(1-self.alpha/2, df=n-1)
        return m-h, m+h

    def compute_test_statistic(self, a, b):
        a_mean = a.mean()
        avar = a.var(ddof=1)
        na = a.size
        adf = na - 1

        b_mean = b.mean()
        bvar = b.var(ddof=1)
        nb = b.size
        bdf = nb - 1

        t = ((b_mean - a_mean)) / np.sqrt(avar/na + bvar/nb)
        self.df = ((avar/na + bvar/nb)**2 /
                   (avar**2/(na**2*adf) + bvar**2/(nb**2*bdf)))
        return t

    def compute_critical(self, a, b):
        return scipy.stats.t.ppf(1 - self.alpha / 2, df=self.df)

    def compute_p_value(self, a, b):
        return 2 * scipy.stats.t.cdf(-np.abs(self.statistic), df=self.df)

    def compute_confidence_intervals(self, a, b):
        return self.ci(a), self.ci(b)


class ZTest(StatTest):
    """
    Test for mean based on normal distribution, one or two samples.
    In the case of two samples, the samples are assumed to be independent.
    Parameters
    ----------
    a, b : {list, ndarray}
        Observed data for two groups, where a - control group,  b - test group.
    alpha : float
        Significance level.

    Returns
    -------
    p_value : float
        Two-sided p-value.
    sign : bool
        Rejected or not null hypotesis with given significance level.
    """

    def __init__(self, a, b, alpha=0.05):
        self.mu = b.mean() - a.mean()
        self.sigma = np.sqrt(a.std()**2 / len(a) + b.std()**2 / len(b))
        self.alpha = alpha

    def _probability(self):
        return scipy.stats.norm.cdf(0, -self.mu, self.sigma)
    
    def ci(self, x):
        x = np.array(x)
        m, se = np.mean(x), scipy.stats.sem(x)
        h = se * scipy.stats.norm.ppf(1 - self.alpha / 2)
        return m - h, m + h
    
    def compute_confidence_intervals(self, a, b):
        self.significance = max(2*scipy.stats.norm.cdf(abs(a.mean() - b.mean()) /
                                (scipy.stats.sem(a) + scipy.stats.sem(b))) - 1, 0)
        return self.ci(a), self.ci(b)


class BTest(StatTest):
    
    """
    Perform a test that the probability of success is p.
    This is an exact, two-sided test of the null hypothesis that the probability of success in a Bernoulli experiment is p.
    Parameters
        ----------
        a, b : {list, ndarray}
            Observed data for two groups, where a - control group,  b - test group.
        model_name : object
            Model to fit the variables distribution.
        random_size : integer
            Number of random variates
    """
    
    def __init__(self, a, b, model_name, random_size=100000):
        self.diff = model_name(b).rvs(random_size) - model_name(a).rvs(random_size)

    def _probability(self):
        return (self.diff > 0).mean()
