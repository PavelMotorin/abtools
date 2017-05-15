# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm

from .base import BaseModel


__all__ = [
    'WaldARPUABModel',
    'LognormalARPUABModel'
]


class WaldARPUABModel(BaseModel):
    """
    ARPU = C * ARPPU where C with Binary likelihood,
    ARPPU with Inverse Gaussian (Wald)
    """
    def build_model(self, a, b):

        a_obs_r, b_obs_r = np.array(a['revenue']), np.array(b['revenue'])
        a_obs_c, b_obs_c = np.array(a['conversion']), np.array(b['conversion'])

        x_min = min(a_obs_r.min(), b_obs_r.min())
        x_max = max(a_obs_r.max(), b_obs_r.max())

        with self.model:

            # priors
            lam_a = pm.Uniform('$\\lambda_A$', 0, x_max)
            mu_a = pm.Uniform('$\\mu_A$', x_min, x_max)

            lam_b = pm.Uniform('$\\lambda_B$', 0, x_max)
            mu_b = pm.Uniform('$\\mu_B$', x_min, x_max)

            p_a = pm.Uniform('$p_A$', 0, 1)
            p_b = pm.Uniform('$p_B$', 0, 1)


            # likelihoods
            a_r = pm.Wald('$A_R$', mu=mu_a, lam=lam_a, observed=a_obs_r)
            b_r = pm.Wald('$B_R$', mu=mu_b, lam=lam_b, observed=b_obs_r)

            a_c = pm.Bernoulli('$A_C$', p=p_a, observed=a_obs_c)
            b_c = pm.Bernoulli('$B_C$', p=p_b, observed=b_obs_c)

            # deterministic stats
            a_arpu = pm.Deterministic('$A_{ARPU}$', mu_a * p_a)
            b_arpu = pm.Deterministic('$B_{ARPU}$', mu_b * p_b)
            delta_conv = pm.Deterministic('$\Delta_C$', p_b - p_a)
            delta_arppu = pm.Deterministic('$\Delta_{ARPPU}$', mu_b - mu_a)
            delta_arpu = pm.Deterministic('$\Delta_{ARPU}$', b_arpu - a_arpu)

            a_var = pm.Deterministic('$A_{\\sigma^2}$', mu_a ** 3 / lam_a)
            b_var = pm.Deterministic('$B_{\\sigma^2}$', mu_b ** 3 / lam_b)

            delta_sigma = pm.Deterministic(
                    '$\Delta_{\\sigma}$',
                    np.sqrt(b_var) - np.sqrt(a_var)
            )
            effect_size = pm.Deterministic(
                'Effect size',
                delta_arppu / np.sqrt((a_var + b_var) / 2)
            )

    def plot_deltas(self):
        return self.plot_result(
            [
             '$\Delta_C$', '$\Delta_{ARPPU}$', '$\Delta_{ARPU}$',
             'Effect size', '$\Delta_{\\sigma}$'
            ],
            ref_val=0
        )

    def plot_params(self):
        return self.plot_result(
            [
                '$p_A$', '$p_B$',
                '$\\mu_A$', '$\\mu_B$',
                '$A_{ARPU}$', '$B_{ARPU}$'
            ]
        )


class LognormalARPUABModel(BaseModel):
    """
    Mixed A/B ARPU model with log-Normal likelihood for a revenue.

    ARPU model formalizes like follows ARPU = C * ARPPU,
    where C is conversion and ARPPU - expected value of revenue.
    In this model C has a Bernoulli likehood and Uniform prior for $p$, where
    $p$ is conversion probability.ARPPU has a log-Normal likelihood and also
    Uniform priors for $\mu$ and $\tau$.

    Parameters
    ----------

    data : dict
        Dictionary with named arrays of observed values. Must contains
        following keys:
        - A_rev, B_rev - non-zero revenue continuous observations
        - A_conv, B_conv - conversion binary [0, 1] observations

    Examples
    --------

    Simple usage example with artificial data:

    >>> from scipy.stats import bernoulli, lognorm
    >>> from abtools.bayesian import LognormalARPUABModel
    >>> a_conv = bernoulli.rvs(0.05, size=5000)
    >>> b_conv = bernoulli.rvs(0.06, size=5000)
    >>> a_rev = lognorm.rvs(1.03, size=1000)
    >>> b_rev = lognorm.rvs(1.05, size=1000)
    >>> a = {'revenue': a_rev, 'conversion': a_conv}
    >>> b = {'revenue': b_rev, 'conversion': b_conv}
    >>> model = LognormalARPUABModel(a, b)
    >>> model.fit()
    >>> model.summary()
    """
    def build_model(self, a, b):
        """
        Build ARPU model for compartion of two groups
        """
        # get data from given dict
        a_obs_r, b_obs_r = np.array(a['revenue']), np.array(b['revenue'])
        a_obs_c, b_obs_c = np.array(a['conversion']), np.array(b['conversion'])

        # pool groups statistics
        m = (a_obs_r.mean() + b_obs_r.mean()) / 2
        v = (a_obs_r.var() + a_obs_r.var()) / 2
        # init values to make optimization more easy and speed up convergence
        init_mu = np.log(m / np.sqrt(1 + v / (m ** 2)))
        init_tau = 1 / np.log(1 + v / (m ** 2))

        with self.model:

            tau_a = pm.Gamma('$\\tau_A$', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu_l_a = pm.Normal('$\mu_{ln(A)}$', init_mu, init_tau ** (-2) * 2)

            tau_b = pm.Gamma('$\\tau_B$', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu_l_b = pm.Normal('$\mu_{ln(B)}$', init_mu, init_tau ** (-2) * 2)

            a = pm.Lognormal('$A$', mu=mu_l_a, tau=tau_a, observed=a_obs_r)
            b = pm.Lognormal('$B$', mu=mu_l_b, tau=tau_b, observed=b_obs_r)

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

            p_a = pm.Uniform('$p_A$', 0, 1)
            p_b = pm.Uniform('$p_B$', 0, 1)

            a_c = pm.Bernoulli('$A_C$', p=p_a, observed=a_obs_c)
            a_c = pm.Bernoulli('$B_C$', p=p_b, observed=b_obs_c)

            a_arpu = pm.Deterministic('$A_{ARPU}$', mu_a * p_a)
            b_arpu = pm.Deterministic('$B_{ARPU}$', mu_b * p_b)

            delta_conv = pm.Deterministic('$\Delta_C$', p_b - p_a)
            delta_arppu = pm.Deterministic('$\Delta_{ARPPU}$', mu_b - mu_a)
            delta_arpu = pm.Deterministic('$\Delta_{ARPU}$', b_arpu - a_arpu)

            delta_sigma = pm.Deterministic(
                '$\\Delta_{\\sigma}$',
                np.sqrt(b_var) - np.sqrt(a_var)
            )

            effect_size = pm.Deterministic(
                'Effect size',
                delta_arppu / np.sqrt((a_var + b_var) / 2)
            )

    def plot_deltas(self):
        return self.plot_result(
            [
             '$\Delta_C$', '$\Delta_{ARPPU}$', '$\Delta_{ARPU}$',
             'Effect size', '$\Delta_{\\sigma}$'
            ],
            ref_val=0
        )

    def plot_params(self):
        return self.plot_result(
            [
                '$p_A$', '$p_B$',
                '$\\mu_A$', '$\\mu_B$',
                '$A_{ARPU}$', '$A_{ARPU}$'
            ]
        )
