from __future__ import absolute_import

import numpy as np
import pymc3 as pm


class BaseModel(object):

    def __init__(self, name='Base Model', auto_init=True):

        self.model = pm.Model()
        self.name = name
        self.auto_init = True

    def init_model(self, init, n_init=10000, plot=True):
        """
        Performs initialization of model with ADVI or MAP.

        """
        with self.model:
            if init.upper() == 'ADVI':
                self.means, self.stds, self.elbos = pm.variational.advi(n=n_init)
            elif init.upper() == 'MAP':
                self.means = pm.find_MAP()

    def fit(self, step=pm.Metropolis, init='MAP', samples=10000,
            burn_in=0.05, thin=1, n_init=10000, n_jobs=1, sample_ppc=False):
        """
        Fit model
        """

        if self.auto_init:
            self.init_model(init, n_init, False)
        else:
            self.means = None

        with self.model:

            step = step()

            trace = pm.sample(
                draws=samples,
                step=step,
                njobs=n_jobs,
                start=self.means
            )
            self.trace = trace[int(samples * burn_in)::thin]
            if sample_ppc:
                self.posterior = pm.sample_ppc(self.trace)

        return self

    def plot_result(self, varnames, ref_val=None):
        return pm.plot_posterior(
            self.trace,
            varnames=varnames,
            color='#87ceeb',
            ref_val=ref_val
        )

    def summary(self):
        return pm.df_summary(self.trace)
