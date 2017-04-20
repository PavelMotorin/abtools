from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pymc3 as pm

from .base import BaseModel


class BinaryModel(BaseModel):
    """
    Binary model with Bernoulli likelihood
    """
    def __init__(self, X_obs, auto_init=True):

        super(BinaryModel, self).__init__(
            'Binary A/B model',
            auto_init
        )

        X_obs = np.array(X_obs)

        with self.model:
            p = pm.Uniform('$p$', 0, 1)

            X = pm.Bernoulli('$X$', p=p, observed=X_obs)

    def plot_params(self):
        return super(BinaryModel, self).plot_result(
            ['$p$']
        )


class WaldModel(BaseModel):
    """
    Heavy Tailed model with Inverse Gaussian (Wald) likelihood
    """
    def __init__(self, X_obs, lower=.01, upper=1000, auto_init=True):

        super(WaldModel, self).__init__(
            'Heavy Tailed Inverse Gaussian A/B model',
            auto_init
        )

        X_obs = np.array(X_obs)

        mu_0 = np.mean(X_obs)
        sigma_0 = np.std(X_obs) * 10

        with self.model:

            alpha = pm.Uniform('$\\alpha$', lower=lower, upper=upper)
            lam = pm.Exponential('$\\lambda$', lam=alpha)
            mu = pm.Gamma('$\\mu$', mu=mu_0, sd=sigma_0)

            X = pm.Wald('$X$', mu=mu, lam=lam, observed=X_obs)

            variance = pm.Deterministic('$\\sigma^2$', (mu**3/lam))

    def plot_params(self):
        return super(WaldModel, self).plot_result(
            [
                '$\\mu$', '$\\lam$', '$\\alpha$'
            ]
        )


class LognormalModel(BaseModel):
    """
    Heavy Tailed model with Log Normal likelihood
    """
    def __init__(self, X_obs, lower=.01, upper=1000, auto_init=True):

        super(LognormalModel, self).__init__(
            'Heavy Tailed Log Normal A/B model',
            auto_init
        )

        X_obs = np.array(X_obs)

        mu_0 = np.mean(np.log(X_obs))
        sigma_0 = np.std(np.log(X_obs)) * 10

        with self.model:

            alpha = pm.Uniform('$\\alpha$', lower=lower, upper=upper)
            beta = pm.Uniform('$\\beta$', lower=lower, upper=upper)
            tau = pm.Gamma('$\\tau$', alpha=alpha, beta=beta)
            mu_l = pm.Gamma('$\\mu_{ln(X)}$', mu=mu_0, sd=sigma_0)

            X = pm.Lognormal('$X$', mu=mu_l, tau=tau, observed=X_obs)

            mu = pm.Deterministic('$\\mu$', np.exp(mu_l+1/(2 * tau)))
            variance = pm.Deterministic(
                '\\sigma^2$',
                (np.exp(1/tau - 1) * np.exp(2*mu_l - 1/tau))
            )

    def plot_params(self):
        return super(LognormalModel, self).plot_result(
            [
                '$\\mu$', '$\\sigma^2$'
            ]
        )
