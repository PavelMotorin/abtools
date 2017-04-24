from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pymc3 as pm

from .base import BaseModel


class BinaryABModel(BaseModel):
    """
    Binary model with Bernoulli likelihood
    """
    def __init__(self, A_obs, B_obs, auto_init=True):

        super(BinaryABModel, self).__init__(
            'Binary A/B model',
            auto_init
        )

        A_obs, B_obs = np.array(A_obs), np.array(B_obs)

        with self.model:
            p_A = pm.Uniform('$p_A$', 0, 1)
            p_B = pm.Uniform('$p_B$', 0, 1)

            A_C = pm.Bernoulli('$A_C$', p=p_A, observed=A_obs)
            B_C = pm.Bernoulli('$B_C$', p=p_B, observed=B_obs)

            delta = pm.Deterministic('$\Delta_p$', p_B - p_A)

    def plot_deltas(self):
        return super(BinaryABModel, self).plot_result(
            ['$\Delta_p$'],
            ref_val=0
        )

    def plot_params(self):
        return super(BinaryABModel, self).plot_result(
            ['$p_A$', '$p_B$']
        )


class WaldABModel(BaseModel):
    """
    Heavy Tailed model with Inverse Gaussian (Wald) likelihood
    """
    def __init__(self, A_obs, B_obs, uncertainty=.3, auto_init=True):

        super(WaldABModel, self).__init__(
            'Heavy Tailed Inverse Gaussian A/B model',
            auto_init
        )

        A_obs, B_obs = np.array(A_obs), np.array(B_obs)

        x_min = min(A_obs.min(), B_obs.min()) * (1. - uncertainty)
        x_max = max(A_obs.max(), B_obs.max()) * (1. + uncertainty)

        with self.model:

            lam_a = pm.Uniform('$\\lambda_A$', 0, x_max)
            mu_a = pm.Uniform('$\\mu_A$',x_min, x_max)

            lam_b = pm.Uniform('$\\lambda_B$', 0, x_max)
            mu_b = pm.Uniform('$\\mu_B$', x_min, x_max)

            A = pm.Wald('$A$', mu=mu_a, lam=lam_a, observed=A_obs)
            B = pm.Wald('$B$', mu=mu_b, lam=lam_b, observed=B_obs)

            A_variance = pm.Deterministic('$A_{\\sigma^2}$', (mu_a**3/lam_a))
            B_variance = pm.Deterministic('$B_{\\sigma^2}$', (mu_b**3/lam_b))

            delta_mu = pm.Deterministic('$\\Delta_{\\mu}$', mu_b - mu_a)

            delta_sigma = pm.Deterministic('$\\Delta_{\\sigma}$',
                                         np.sqrt(B_variance) -
                                         np.sqrt(A_variance))
            effect_size = pm.Deterministic(
                'Effect size',
                delta_mu / np.sqrt((A_variance + B_variance) / 2)
            )

    def plot_deltas(self):
        return super(WaldABModel, self).plot_result(
            ['$\\Delta_{\\mu}$', '$\\Delta_{\\sigma}$', 'Effect size'],
            ref_val=0
        )

    def plot_params(self):
        return super(WaldABModel, self).plot_result(
            [
                '$\\mu_A$', '$\\mu_B$'
            ]
        )


class WaldARPUABModel(BaseModel):
    """
    ARPU = C * ARPPU where C with Binary likelihood,
    ARPPU with Inverse Gaussian (Wald)
    """
    def __init__(self, A_obs_C, B_obs_C, A_obs_R, B_obs_R,
                 uncertainty=.3, auto_init=True):

        super(WaldARPUABModel, self).__init__(
            'ARPU Wald A/B model',
            auto_init
        )

        A_obs_R, B_obs_R = np.array(A_obs_R), np.array(B_obs_R)
        A_obs_C, B_obs_C = np.array(A_obs_C), np.array(B_obs_C)

        x_min = min(A_obs_R.min(), B_obs_R.min()) * (1. - uncertainty)
        x_max = max(A_obs_R.max(), B_obs_R.max()) * (1. + uncertainty)

        with self.model:

            # priors
            lam_a = pm.Uniform('$\\lambda_A$', 0, x_max)
            mu_a = pm.Uniform('$\\mu_A$',x_min, x_max)
            p_a = pm.Uniform('$p_A$', 0, 1)

            lam_b = pm.Uniform('$\\lambda_B$', 0, x_max)
            mu_b = pm.Uniform('$\\mu_B$', x_min, x_max)
            p_b = pm.Uniform('$p_B$', 0, 1)

            # likelihoods
            A_R = pm.Wald('$A_R$', mu=mu_a, lam=lam_a, observed=A_obs_R)
            B_R = pm.Wald('$B_R$', mu=mu_b, lam=lam_b, observed=B_obs_R)

            A_C = pm.Bernoulli('$A_C$', p=p_a, observed=A_obs_C)
            B_C = pm.Bernoulli('$B_C$', p=p_b, observed=B_obs_C)

            # deterministic stats
            A_arpu = pm.Deterministic('$A_{ARPU}$', mu_a * p_a)
            B_arpu = pm.Deterministic('$B_{ARPU}$', mu_b * p_b)
            delta_conv = pm.Deterministic('$\Delta_C$', p_b - p_a)
            delta_arppu = pm.Deterministic('$\Delta_{ARPPU}$', mu_b - mu_a)
            delta_arpu = pm.Deterministic('$\Delta_{ARPU}$', B_arpu - A_arpu)

            A_var = pm.Deterministic('$A_{\\sigma^2}$', mu_a ** 3 / lam_a)
            B_var = pm.Deterministic('$B_{\\sigma^2}$', mu_b ** 3 / lam_b)

            delta_sigma = pm.Deterministic('$\Delta_{\\sigma}$',
                                           np.sqrt(B_var) - np.sqrt(A_var))
            effect_size = pm.Deterministic(
                'Effect size',
                delta_arppu / np.sqrt((A_var + B_var) / 2)
            )

    def plot_deltas(self):
        return super(WaldARPUABModel, self).plot_result(
            [
             '$\Delta_C$', '$\Delta_{ARPPU}$', '$\Delta_{ARPU}$',
             'Effect size', '$\Delta_{\\sigma}$'
            ],
            ref_val=0
        )

    def plot_params(self):
        return super(WaldARPUABModel, self).plot_result(
            [
                '$p_A$', '$p_B$',
                '$\\mu_A$', '$\\mu_B$',
                '$A_{ARPU}$', '$B_{ARPU}$'
            ]
        )


class LognormalABModel(BaseModel):
    """
    Heavy Tailed model with Log Normal likelihood
    """
    def __init__(self, A_obs, B_obs, uncertainty=.3, auto_init=True):

        super(LognormalABModel, self).__init__(
            'Heavy Tailed Log Normal A/B model',
            auto_init
        )

        A_obs, B_obs = np.array(A_obs), np.array(B_obs)

        x_min = np.log(min(A_obs.min(), B_obs.min()) * (1. - uncertainty))
        x_max = np.log(max(A_obs.max(), B_obs.max()) * (1. + uncertainty))

        with self.model:

            tau_a = pm.Uniform('$\\lambda_A$', 0, x_max)
            mu_l_a = pm.Uniform('$\\mu_{ln(A)}$', x_min, x_max)

            tau_b = pm.Uniform('$\\lambda_B$', 0, x_max)
            mu_l_b = pm.Uniform('$\\mu_{ln(B)}$', x_min, x_max)

            A = pm.Lognormal('$A$', mu=mu_l_a, tau=tau_a, observed=A_obs)
            B = pm.Lognormal('$B$', mu=mu_l_b, tau=tau_b, observed=B_obs)

            mu_a = pm.Deterministic('$\\mu_A$', np.exp(mu_l_a+1/(2 * tau_a)))
            mu_b = pm.Deterministic('$\\mu_B$', np.exp(mu_l_b+1/(2 * tau_b)))

            A_variance = pm.Deterministic(
                '$A_{\\sigma^2}$',
                (np.exp(1/tau_a - 1) * np.exp(2*mu_l_a - 1/tau_a))
            )
            B_variance = pm.Deterministic(
                '$B_{\\sigma^2}$',
                (np.exp(1/tau_b - 1) * np.exp(2*mu_l_b - 1/tau_b))
            )
            delta_mu = pm.Deterministic('$\\Delta_{\\mu}$', mu_b - mu_a)

            delta_sigma = pm.Deterministic(
                '$\\Delta_{\\sigma}$',
                np.sqrt(B_variance) - np.sqrt(A_variance)
            )

            effect_size = pm.Deterministic(
                'Effect size',
                delta_mu / np.sqrt((A_variance + B_variance) / 2)
            )

    def plot_deltas(self):
        return super(LognormalABModel, self).plot_result(
            ['$\\Delta_{\\mu}$', '$\\Delta_{\\sigma}$', 'Effect size'],
            ref_val=0
        )

    def plot_params(self):
        return super(LognormalABModel, self).plot_result(
            [
                '$\\mu_A$', '$\\mu_B$'
            ]
        )


class LognormalARPUABModel(BaseModel):
    """
    ARPU = C * ARPPU where C with Binary likelihood,
    ARPPU with Log Normal
    """
    def __init__(self, A_obs_C, B_obs_C, A_obs_R, B_obs_R,
                 lower=.01, upper=1000, auto_init=True):

        super(LognormalARPUABModel, self).__init__(
            'ARPU Log Normal A/B model',
            auto_init
        )

        A_obs_R, B_obs_R = np.array(A_obs_R), np.array(B_obs_R)
        A_obs_C, B_obs_C = np.array(A_obs_C), np.array(B_obs_C)

        x_min = np.log(min(A_obs_R.min(), B_obs_R.min()) * (1. - uncertainty))
        x_max = np.log(max(A_obs_R.max(), B_obs_R.max()) * (1. + uncertainty))

        with self.model:

            tau_a = pm.Uniform('$\\lambda_A$', 0, x_max)
            mu_l_a = pm.Uniform('$\\mu_{ln(A)}$', x_min, x_max)

            tau_b = pm.Uniform('$\\lambda_B$', 0, x_max)
            mu_l_b = pm.Uniform('$\\mu_{ln(B)}$', x_min, x_max)

            A = pm.Lognormal('$A$', mu=mu_l_a, tau=tau_a, observed=A_obs_R)
            B = pm.Lognormal('$B$', mu=mu_l_b, tau=tau_b, observed=B_obs_R)

            mu_a = pm.Deterministic('$\\mu_A$', np.exp(mu_l_a+1/(2 * tau_a)))
            mu_b = pm.Deterministic('$\\mu_B$', np.exp(mu_l_b+1/(2 * tau_b)))

            A_variance = pm.Deterministic(
                '$A_{\\sigma^2}$',
                (np.exp(1/tau_a - 1) * np.exp(2*mu_l_a - 1/tau_a))
            )
            B_variance = pm.Deterministic(
                '$B_{\\sigma^2}$',
                (np.exp(1/tau_b - 1) * np.exp(2*mu_l_b - 1/tau_b))
            )

            p_a = pm.Uniform('$p_A$', 0, 1)
            p_b = pm.Uniform('$p_B$', 0, 1)

            A_C = pm.Bernoulli('$A_C$', p=p_a, observed=A_obs_C)
            B_C = pm.Bernoulli('$B_C$', p=p_b, observed=B_obs_C)

            A_arpu = pm.Deterministic('$A_{ARPU}$', mu_a * p_a)
            B_arpu = pm.Deterministic('$B_{ARPU}$', mu_b * p_b)

            delta_conv = pm.Deterministic('$\Delta_C$', p_b - p_a)
            delta_arppu = pm.Deterministic('$\Delta_{ARPPU}$', mu_b - mu_a)
            delta_arpu = pm.Deterministic('$\Delta_{ARPU}$', B_arpu - A_arpu)

            delta_sigma = pm.Deterministic(
                '$\\Delta_{\\sigma}$',
                np.sqrt(B_variance) - np.sqrt(A_variance)
            )

            effect_size = pm.Deterministic(
                'Effect size',
                delta_arppu / np.sqrt((A_variance + B_variance) / 2)
            )

    def plot_deltas(self):
        return super(LognormalARPUABModel, self).plot_result(
            [
             '$\Delta_C$', '$\Delta_{ARPPU}$', '$\Delta_{ARPU}$',
             'Effect size', '$\Delta_{\\sigma}$'
            ],
            ref_val=0
        )

    def plot_params(self):
        return super(LognormalARPUABModel, self).plot_result(
            [
                '$p_A$', '$p_B$',
                '$\\mu_A$', '$\\mu_B$',
                '$A_{ARPU}$', '$A_{ARPU}$'
            ]
        )
