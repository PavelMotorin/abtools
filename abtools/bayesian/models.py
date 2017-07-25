# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

from .base import BaseModel


__all__ = [
    'BernoulliModel',
    # 'WaldModel',
    'LognormalModel'
]


class BernoulliModel(BaseModel):
    """
    Bernoulli model with Bernoulli likelihood
    """
    estimated_var = 'p'

    def build_model(self, x_obs):

        x_obs = np.array(x_obs)

        with self.model:
            p = pm.Uniform('p', 0, 1)
            x = pm.Bernoulli('x', p=p, observed=x_obs)


class LognormalModel(BaseModel):
    r"""
    Model with log-Normal likelihood.

    Model heavy-tail distributed data.
    x ~ logN(\mu, \tau)
    where \mu is Normal distributed and \tau is Gamma distributed according to
    conjurate priors for log-Normal distribution.

    This model is most stable to outliers and small data size.

    Parameters
    ----------

    Examples
    --------
    Simple usage example with artificial data:

    """
    estimated_var = 'm'

    def build_model(self, x):

        x_obs = np.array(x)

        m = x_obs.mean()
        v = x_obs.var()

        init_mu = np.log(m / np.sqrt(1 + v / (m ** 2)))
        init_tau = 1 / np.log(1 + v / (m ** 2))

        with self.model:

            tau = pm.Gamma('tau', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu = pm.Normal('mu', init_mu, init_tau ** (-2) * 2)

            x = pm.Lognormal('x', mu=mu, tau=tau, observed=x_obs)

            m = pm.Deterministic('m', np.exp(mu + 1/(2 * tau)))

            var = pm.Deterministic('var', (np.exp(1/tau - 1) *
                                           np.exp(2*mu - 1/tau)))
