# -*- coding: utf-8 -*-

import pymc3 as pm


class BaseModel(object):
    """
    Base model class

    Parameters
    ---------
    data : dict
        Dictionary contains named data arrays
    name : str
        Name of a model
    auto_init : bool
        Perform of not automatic initialization for a model

    """
    def __init__(self, a):
        self.model = pm.Model()
        self.build_model(a)

    def build_model(self, a):
        """Build model with likelihoods and priors."""
        raise NotImplementedError

    def fit(self, samples=10000, burn_in=0.25, thin=1, n_jobs=1):
        """
        Inference button.

        Perform sampling of posterior distribution via selected step method.

        Parameters
        ----------
        step : pymc3 step method
            Uses Metropolis sampler by default. Can be changed to NUTS.
        init : str {'ADVI', 'MAP'}
            Name of pymc3 initialization method:
            * ADVI : Estimate posterior mean for starting point.
            * MAP : Use the MAP as starting point.
        samples : int
            Number pf sampler for draw from posterior
        burn_in : float
            Percentage of values that will be discarded in final trace
        thin : int
            Thinning rate. With value 5 drops every fifth sample from trace to
            avoid autocorrelation problem.
        n_init : int
            Number of init iterations for ADVI
        n_jobs : int
            Number of traces that will be sampled in parallel
        sample_ppc : bool
            Sample (or not) posterior predictive samples for a model.

        """


        with self.model:

            trace = pm.sample(
                draws=samples,
                step=pm.Metropolis(),
                njobs=n_jobs,
                start=pm.find_MAP()
            )
            self.trace = trace[int(samples * burn_in)::thin]


        return self

    def estimated_var_trace(self):
        return self.trace[self.estimated_var]

    def plot_result(self, varnames, ref_val=None):
        """
        Plot posterior distributions.

        Parameters
        ----------
        varnames: list of str
            List of model variable names which need to plot.
        ref_val: float or int
            Reference value that is x-point for vertical helper line on a hist.

        """
        return pm.plot_posterior(
            self.trace,
            varnames=varnames,
            color='#87ceeb',
            ref_val=ref_val
        )

    def sample_ppc(self):
        """Sample posterior predictive distributions."""
        with self.model:
            self.posterior = pm.sample_ppc(self.trace)
