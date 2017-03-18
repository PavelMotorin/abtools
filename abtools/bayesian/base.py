from __future__ import absolute_import

import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt
import seaborn as sns


class BaseABModel(object):

    def __init__(self, name='Base A/B Model', auto_init=True):

        self.model = pm.Model()
        self.name = name
        self.auto_init = True

    def init_model(self, n_init=50000, plot=True):
        """
        Performs initialization of model with ADVI.

        """
        with self.model:
            self.means, self.sds, self.elbos = pm.variational.advi(n=n_init)

        if plot:
            fig = plt.figure(figsize=(8, 8))
            plt.plot(self.elbos, c=sns.xkcd_rgb["denim blue"])
            plt.title('ELBO on each ADVI iteration')
            plt.xlabel('# iteration')
            plt.ylabel('ELBO')
            fig.show()

    def fit(self, step=pm.Metropolis, samples=50000, n_init=50000, n_jobs=1):
        """
        Fit model
        """

        if self.auto_init:
            self.init_model(n_init, False)
        else:
            self.means = None

        with self.model:

            step = step()

            self.trace = pm.sample(
                draws=samples,
                step=step,
                njobs=n_jobs,
                start=self.means
            )

    def ppc(self, samples=20000):
        posterior_samples = pm.sample_ppc(
            self.trace,
            samples=samples,
            model=self.model
        )

        return posterior_samples

    def plot_result(self, varnames, burn_in, ref_val=None):
        return pm.plot_posterior(
            self.trace[burn_in:],
            varnames=varnames,
            color='#87ceeb',
            ref_val=ref_val
        )
