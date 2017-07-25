# -*- coding: utf-8 -*-

from itertools import combinations

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from .models import BernoulliModel, LognormalModel, ARPUModel
from .base import StatTest


class ABtest(object):
    """
    AB test class.

    Class.
    """
    def __init__(self, model, groups, samples=100, alpha=0.05):
        print('ABtest for %d groups' % len(groups))
        self.models = {
            'group%d' % (i + 1): model(group)
            for i, group in enumerate(groups)
        }

        self.samples = samples

    def estimate(self):

        for model in self.models.values():
            model.fit(samples=self.samples)

        self.deltas = [
            (self.models[b].trace - self.models[a].trace, (a, b))
            for a, b in combinations(sorted(self.models.keys()), 2)
        ]

        self.means = np.array([
            (name, np.mean(self.models[name].trace))
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

class ZTest(StatTest):
    def __init__(self, a, b):
        self.mu = b.mean() - a.mean()
        self.sigma = sqrt(a.std()**2 / len(a) + b.std()**2 / len(b))

    def _probability(self):
        return scipy.stats.norm.cdf(0, -self.mu, self.sigma)

class BTest(StatTest):
    def __init__(self, a, b, model_name, random_size=100000):
        self.diff = model_name(b).rvs(random_size) - model_name(a).rvs(random_size)

    def _probability(self):
        return (self.diff > 0).mean()
