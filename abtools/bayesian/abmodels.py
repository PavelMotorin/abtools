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
    def __init__(self, A_obs, B_obs, lower=.01, upper=1000, auto_init=True):

        super(WaldABModel, self).__init__(
            'Heavy Tailed Inverse Gaussian A/B model',
            auto_init
        )

        A_obs, B_obs = np.array(A_obs), np.array(B_obs)

        mu_a_0, mu_b_0 = np.mean(A_obs), np.mean(B_obs)
        sigma_a_0, sigma_b_0 = np.std(A_obs) * 10, np.std(B_obs) * 10

        with self.model:

            alpha_a = pm.Uniform('$\\alpha_A$', lower=lower, upper=upper)
            lam_a = pm.Exponential('$\\lambda_A$', lam=alpha_a)
            mu_a = pm.Gamma('$\\mu_A$', mu=mu_a_0, sd=sigma_a_0)

            alpha_b = pm.Uniform('$\\alpha_B$', lower=lower, upper=upper)
            lam_b = pm.Exponential('$\\lambda_B$', lam=alpha_b)

            mu_b = pm.Gamma('$\\mu_B$', mu=mu_b_0, sd=sigma_b_0)

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
                 lower=.01, upper=1000, auto_init=True):

        super(WaldARPUABModel, self).__init__(
            'ARPU Wald A/B model',
            auto_init
        )

        A_obs_R, B_obs_R = np.array(A_obs_R), np.array(B_obs_R)
        A_obs_C, B_obs_C = np.array(A_obs_C), np.array(B_obs_C)

        mu_a_0, mu_b_0 = np.mean(A_obs_R), np.mean(B_obs_R)
        sigma_a_0, sigma_b_0 = np.std(A_obs_R) * 10, np.std(B_obs_R) * 10

        with self.model:

            # priors
            alpha_a = pm.Uniform('$\\alpha_A$', lower=lower, upper=upper)
            lam_a = pm.Exponential('$\\lambda_A$', lam=alpha_a)
            mu_a = pm.Gamma('$\\mu_A$', mu=mu_a_0, sd=sigma_a_0)
            p_a = pm.Uniform('$p_A$', 0, 1)

            alpha_b = pm.Uniform('$\\alpha_B$', lower=lower, upper=upper)
            lam_b = pm.Exponential('$\\lambda_B$', lam=alpha_b)
            mu_b = pm.Gamma('$\\mu_B$', mu=mu_b_0, sd=sigma_b_0)
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
    def __init__(self, A_obs, B_obs, lower=.01, upper=1000, auto_init=True):

        super(LognormalABModel, self).__init__(
            'Heavy Tailed Log Normal A/B model',
            auto_init
        )

        A_obs, B_obs = np.array(A_obs), np.array(B_obs)

        mu_a_0, mu_b_0 = np.mean(np.log(A_obs)), np.mean(np.log(B_obs))
        sigma_a_0, sigma_b_0 = np.std(np.log(A_obs)) * 10, np.std(np.log(B_obs)) * 10

        with self.model:

            alpha_a = pm.Uniform('$\\alpha_A$', lower=lower, upper=upper)
            beta_a = pm.Uniform('$\\beta_A$', lower=lower, upper=upper)
            tau_a = pm.Gamma('$\\lambda_A$', alpha=alpha_a, beta=beta_a)
            mu_l_a = pm.Gamma('$\\mu_{ln(A)}$', mu=mu_a_0, sd=sigma_a_0)

            alpha_b = pm.Uniform('$\\alpha_B$', lower=lower, upper=upper)
            beta_b = pm.Uniform('$\\beta_B$', lower=lower, upper=upper)
            tau_b = pm.Gamma('$\\lambda_B$', alpha=alpha_b, beta=alpha_b)

            mu_l_b = pm.Gamma('$\\mu_{ln(B)}$', mu=mu_b_0, sd=sigma_b_0)

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

        mu_a_0, mu_b_0 = np.mean(A_obs_R), np.mean(B_obs_R)
        sigma_a_0, sigma_b_0 = np.std(A_obs_R) * 10, np.std(B_obs_R) * 10

        with self.model:

            alpha_a = pm.Uniform('$\\alpha_A$', lower=lower, upper=upper)
            beta_a = pm.Uniform('$\\beta_A$', lower=lower, upper=upper)
            tau_a = pm.Gamma('$\\lambda_A$', alpha=alpha_a, beta=beta_a)
            mu_l_a = pm.Gamma('$\\mu_{ln(A)}$', mu=mu_a_0, sd=sigma_a_0)

            alpha_b = pm.Uniform('$\\alpha_B$', lower=lower, upper=upper)
            beta_b = pm.Uniform('$\\beta_B$', lower=lower, upper=upper)
            tau_b = pm.Gamma('$\\lambda_B$', alpha=alpha_b, beta=alpha_b)
            mu_l_b = pm.Gamma('$\\mu_{ln(B)}$', mu=mu_b_0, sd=sigma_b_0)

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
