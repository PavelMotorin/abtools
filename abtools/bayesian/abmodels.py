# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

from .base import BaseModel


class BinaryABModel(BaseModel):
    """
    Binary model with Bernoulli likelihood
    """

    def build_model(self, a, b):
        a_obs = np.array(a)
        b_obs = np.array(b)

        with self.model:
            p_a = pm.Uniform('$p_A$', 0, 1)
            p_b = pm.Uniform('$p_B$', 0, 1)

            a = pm.Bernoulli('$A$', p=p_a, observed=a_obs)
            b = pm.Bernoulli('$B$', p=p_b, observed=b_obs)

            a_var = pm.Deterministic('$A_{\\sigma^2}$', p_a * (1 - p_a))
            b_var = pm.Deterministic('$B_{\\sigma^2}$', p_b * (1 - p_b))

            delta_p = pm.Deterministic('$\Delta_p$', p_b - p_a)
            delta_std = pm.Deterministic(
                    '$\\Delta_{\\sigma}$',
                    np.sqrt(b_var) - np.sqrt(a_var)
            )

            effect_size = pm.Deterministic(
                'Effect size',
                delta_p / np.sqrt((a_var + b_var) / 2)
            )

    def plot_deltas(self):
        return self.plot_result(
            ['$\Delta_p$', '$\\Delta_{\\sigma}$', 'Effect size'],
            ref_val=0
        )

    def plot_params(self):
        return self.plot_result(['$p_A$', '$p_B$'])


class WaldABModel(BaseModel):
    r"""
    A/B model with Inverse Gaussian (Wald) likelihoods.

    Model for comparing posterior means of two heavy-tail distributed groups.
        A ~ IG(\mu_a, \lambda_a)
        B ~ IG(\mu_b, \lambda_b)
    where \mu is Gamma distributed and \lambda is Uniform distributed.

    Parameters
    ----------
    data : dict
        Dictionary with named arrays of observed values. Must contains
        following keys:
        - A, B - non-zero continuous observations

    Examples
    --------
    Simple usage example with artificial data:

    >>> from scipy.stats import lognorm
    >>> from abtools.bayesian import WaldABModel
    >>> a_rev = lognorm.rvs(1.03, size=1000)
    >>> b_rev = lognorm.rvs(1.05, size=1000)
    >>> data = {'A': a, 'B': b}
    >>> model = WaldABModel(data)
    >>> model.fit()
    >>> model.summary()

    """

    def build_model(self, a, b):

        a_obs = np.array(a)
        b_obs = np.array(b)

        m = np.mean([a_obs.mean(), b_obs.mean()])
        v = np.mean([a_obs.var(), b_obs.var()])
        x_max = max(a_obs.max(), b_obs.max())

        with self.model:

            lam_a = pm.Uniform('$\\lambda_A$', 0, x_max)
            mu_a = pm.Gamma('$\\mu_A$', mu=m, sd=2*v**(1/2))

            lam_b = pm.Uniform('$\\lambda_B$', 0, x_max)
            mu_b = pm.Gamma('$\\mu_B$', mu=m, sd=2*v**(1/2))

            a = pm.Wald('$A$', mu=mu_a, lam=lam_a, observed=a_obs)
            b = pm.Wald('$B$', mu=mu_b, lam=lam_b, observed=b_obs)

            a_var = pm.Deterministic('$A_{\\sigma^2}$', (mu_a**3/lam_a))
            b_var = pm.Deterministic('$B_{\\sigma^2}$', (mu_b**3/lam_b))

            delta_mu = pm.Deterministic('$\\Delta_{\\mu}$', mu_b - mu_a)

            delta_sigma = pm.Deterministic(
                    '$\\Delta_{\\sigma}$',
                    np.sqrt(b_var) - np.sqrt(a_var)
            )
            effect_size = pm.Deterministic(
                'Effect size',
                delta_mu / np.sqrt((a_var + b_var) / 2)
            )

    def plot_deltas(self):
        return self.plot_result(
            ['$\\Delta_{\\mu}$', '$\\Delta_{\\sigma}$', 'Effect size'],
            ref_val=0
        )

    def plot_params(self):
        return self.plot_result(['$\\mu_A$', '$\\mu_B$'])


class LognormalABModel(BaseModel):
    r"""
    A/B model with log-Normal likelihoods.

    Model for comparing posterior means of two heavy-tail distributed groups.
    A ~ logN(\mu_a, \tau_a)
    B ~ logN(\mu_b, \tau_b)
    where \mu is Normal distributed and \tau is Gamma distributed according to
    conjurate priors for log-Normal distribution.

    This model is most stable to outliers and small data size.

    Parameters
    ----------
    data : dict
        Dictionary with named arrays of observed values. Must contains
        following keys:
        - A, B - non-zero continuous observations

    Examples
    --------
    Simple usage example with artificial data:

    >>> from scipy.stats import lognorm
    >>> from abtools.bayesian import LognormalABModel
    >>> a_rev = lognorm.rvs(1.03, size=1000)
    >>> b_rev = lognorm.rvs(1.05, size=1000)
    >>> data = {'A': a, 'B': b}
    >>> model = LognormalABModel(data)
    >>> model.fit()
    >>> model.summary()

    """

    def build_model(self, a, b):
        a_obs, b_obs = np.array(a), np.array(b)

        m = (a_obs.mean() + b_obs.mean()) / 2
        v = (a_obs.var() + a_obs.var()) / 2

        init_mu = np.log(m / np.sqrt(1 + v / (m ** 2)))
        init_tau = 1 / np.log(1 + v / (m ** 2))

        with self.model:

            tau_a = pm.Gamma('$\\tau_A$', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu_l_a = pm.Normal('$\mu_{ln(A)}$', init_mu, init_tau ** (-2) * 2)

            tau_b = pm.Gamma('$\\tau_B$', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu_l_b = pm.Normal('$\mu_{ln(B)}$', init_mu, init_tau ** (-2) * 2)

            a = pm.Lognormal('$A$', mu=mu_l_a, tau=tau_a, observed=a_obs)
            b = pm.Lognormal('$B$', mu=mu_l_b, tau=tau_b, observed=b_obs)

            mu_a = pm.Deterministic('$\\mu_A$', np.exp(mu_l_a+1/(2 * tau_a)))
            mu_b = pm.Deterministic('$\\mu_B$', np.exp(mu_l_b+1/(2 * tau_b)))

            a_var = pm.Deterministic(
                '$A_{\\sigma^2}$',
                (np.exp(1/tau_a - 1) * np.exp(2*mu_l_a - 1/tau_a))
            )
            b_var = pm.Deterministic(
                '$B_{\\sigma^2}$',
                (np.exp(1/tau_b - 1) * np.exp(2*mu_l_b - 1/tau_b))
            )
            delta_mu = pm.Deterministic('$\\Delta_{\\mu}$', mu_b - mu_a)

            delta_sigma = pm.Deterministic(
                '$\\Delta_{\\sigma}$',
                np.sqrt(b_var) - np.sqrt(a_var)
            )

            effect_size = pm.Deterministic(
                'Effect size',
                delta_mu / np.sqrt((a_var + b_var) / 2)
            )

    def plot_deltas(self):
        return self.plot_result(
            ['$\\Delta_{\\mu}$', '$\\Delta_{\\sigma}$', 'Effect size'],
            ref_val=0
        )

    def plot_params(self):
        return self.plot_result(['$\\mu_A$', '$\\mu_B$'])
