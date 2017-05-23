# -*- coding: utf-8 -*-

import pymc3 as pm


class BaseModel(object):
    """
    Base model class

    Prameters
    ---------
    data : dict
        Dictionary contains named data arrays
    name : str
        Name of a model
    auto_init : bool
        Perform of not automatic initialization for a model

    """
    def __init__(self, a, b=None, auto_init=True):

        self.model = pm.Model()
        self.auto_init = auto_init
        self.build_model(a, b)

    def build_model(self, a, b):
        raise NotImplementedError

    def init_model(self, init, n_init=10000):
        """
        Performs initialization of model with ADVI or MAP.

        Set model starting values (means) after initialization.

        Parameters
        ----------
        init : str
            Name of pymc3 initialization method
        n_init : int
            Number of iterations
        """
        with self.model:
            if init.upper() == 'ADVI':
                self.means, self.stds, self.elbos = pm.variational.advi(n=n_init)
            elif init.upper() == 'MAP':
                self.means = pm.find_MAP()

    def fit(self, step=pm.Metropolis, init='MAP', samples=10000,
            burn_in=0.25, thin=1, n_init=10000, n_jobs=1, sample_ppc=False):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        step : pymc3 step method
            Uses Metropolis sampler by default. Can be changed to NUTS.
        init : str {'ADVI', 'MAP'}
            Name of pymc3 initialization method:
            * ADVI : Run ADVI to estimate posterior mean and diagonal covariance matrix.
            * MAP : Use the MAP as starting point.
        samples : int
            Number pf sampler for draw from posterior
        burn_in : float
            Percentage of values that will be discarded in final trace
        thin : int
            Step in final trace to avoid autocorrelation in samples
        n_init : int
            Number of init iterations for ADVI
        n_jobs : int
            Number of traces that will be sampled in parallel
        sample_ppc : bool
            Sample (or not) posterior predictive samples for a model.
        """

        if self.auto_init:
            self.init_model(init, n_init)
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

    def burn_in(self, burn_in, thin=1):
        self.trace = self.trace[int(len(self.trace) * burn_in)::thin]
        print('%d samples left' % len(self.trace))

    def plot_result(self, varnames, ref_val=None):
        return pm.plot_posterior(
            self.trace,
            varnames=varnames,
            color='#87ceeb',
            ref_val=ref_val
        )

    def summary(self):
        return pm.df_summary(self.trace)

    def sample_ppc(self):
        with self.model:
            self.posterior = pm.sample_ppc(self.trace)
