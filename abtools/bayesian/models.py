from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pymc3 as pm

from .base import BaseABModel


class ConversionModel(BaseABModel):
    """
    Conversion model with Bernoulli likelihood
    """
    def __init__(self, observed_A_conversion, observed_B_conversion,
                 std_coeff=2, auto_init=True):

        super(ConversionModel, self).__init__(
            'Conversion A/B model',
            auto_init
        )

        observed_A_conversion = np.array(observed_A_conversion)
        observed_B_conversion = np.array(observed_B_conversion)

        mu_init_A = observed_A_conversion.mean()
        mu_init_B = observed_B_conversion.mean()

        sd_init_A = observed_A_conversion.std() * std_coeff
        sd_init_B = observed_B_conversion.std() * std_coeff

        with self.model:
            p_A = pm.Gamma('$p_A$', mu=mu_init_A, sd=sd_init_A)
            p_B = pm.Gamma('$p_B$', mu=mu_init_B, sd=sd_init_B)

            A_C = pm.Bernoulli('$A_C$', p=p_A, observed=observed_A_conversion)
            B_C = pm.Bernoulli('$B_C$', p=p_B, observed=observed_B_conversion)

            delta = pm.Deterministic('$\Delta_C$', p_B - p_A)

    def plot_deltas(self, burn_in):
        return super(ConversionModel, self).plot_result(
            ['$\Delta_C$'],
            burn_in, ref_val=0
        )

    def plot_params(self, burn_in):
        return super(ConversionModel, self).plot_result(
            ['$p_A$', '$p_B$'],
            burn_in
        )


class ARPPUModel(BaseABModel):
    """
    ARPPU model with Wald likelihood
    """
    def __init__(self,
                 observed_A_arppu, observed_B_arppu,
                 upper_prior_bound=1000,
                 std_coeff=2, auto_init=True):

        super(ARPPUModel, self).__init__(
            'ARPPU A/B model',
            auto_init
        )

        observed_A_arppu = np.array(observed_A_arppu)
        observed_B_arppu = np.array(observed_B_arppu)

        mu_arppu_init_a = np.mean(observed_A_arppu)
        mu_arppu_init_b = np.mean(observed_B_arppu)
        std_arppu_init_a = np.std(observed_A_arppu) * std_coeff
        std_arppu_init_b = np.std(observed_B_arppu) * std_coeff

        with self.model:

            alpha0_a = pm.Uniform('$\\alpha_0^A$', lower=1,
                                  upper=upper_prior_bound)
            lam_a = pm.Exponential('$\\lambda_A$', lam=alpha0_a)
            A_arppu = pm.Gamma('$A_{ARPPU}$', mu=mu_arppu_init_a,
                               sd=std_arppu_init_a)
            alpha0_b = pm.Uniform('$\\alpha_0^B$', lower=1,
                                  upper=upper_prior_bound)
            lam_b = pm.Exponential('$\\lambda_B$', lam=alpha0_b)

            B_arppu = pm.Gamma('$B_{ARPPU}$', mu=mu_arppu_init_b,
                               sd=std_arppu_init_b)

            A_arppu_lhd = pm.Wald('A_arppu_lhd', mu=A_arppu, lam=lam_a,
                                  observed=observed_A_arppu)
            B_arppu_lhd = pm.Wald('B_arppu_lhd', mu=B_arppu, lam=lam_b,
                                  observed=observed_B_arppu)

            A_arppu_var = pm.Deterministic('$A_{ARPPU} var$', (A_arppu**3/lam_a))
            B_arppu_var = pm.Deterministic('$B_{ARPPU} var$', (B_arppu**3/lam_b))

            delta_arppu = pm.Deterministic('$\Delta_{ARPPU}$',
                                           B_arppu - A_arppu)
            delta_std = pm.Deterministic('$\Delta_{std}$',
                                         np.sqrt(B_arppu_var) - np.sqrt(A_arppu_var))
            effect_size = pm.Deterministic(
                'effect size',
                delta_arppu / np.sqrt((A_arppu_var + B_arppu_var) / 2)
            )

    def plot_deltas(self, burn_in):
        return super(ARPUModel, self).plot_result(
            ['$\Delta_{ARPPU}$', '$\Delta_{std}$', 'effect size'],
            burn_in, ref_val=0
        )

    def plot_params(self, burn_in):
        return super(ARPUModel, self).plot_result(
            [
                '$A_{ARPPU}$', '$A_{ARPPU}$'
            ],
            burn_in
        )


class ARPUModel(BaseABModel):
    """
    Conversion model with Bernoulli likelihood
    """
    def __init__(self, observed_A_conversion, observed_B_conversion,
                 observed_A_arppu, observed_B_arppu,
                 upper_prior_bound=1000,
                 std_coeff=2, auto_init=True):

        super(ARPUModel, self).__init__(
            'ARPU A/B model',
            auto_init
        )

        observed_A_arppu = np.array(observed_A_arppu)
        observed_B_arppu = np.array(observed_B_arppu)

        observed_A_conversion = np.array(observed_A_conversion)
        observed_B_conversion = np.array(observed_B_conversion)

        mu_conv_init_a = np.mean(observed_A_conversion)
        mu_conv_init_b = np.mean(observed_B_conversion)
        std_conv_init_a = np.std(observed_A_conversion) * std_coeff
        std_conv_init_b = np.std(observed_B_conversion) * std_coeff

        mu_arppu_init_a = np.mean(observed_A_arppu)
        mu_arppu_init_b = np.mean(observed_B_arppu)
        std_arppu_init_a = np.std(observed_A_arppu) * std_coeff
        std_arppu_init_b = np.std(observed_B_arppu) * std_coeff

        with self.model:

            # Priors for A group
            # ================================================================
            # Priors for ARPPU
            alpha0_a = pm.Uniform('$\\alpha_0^A$', lower=1,
                                  upper=upper_prior_bound)
            lam_a = pm.Exponential('$\\lambda_A$', lam=alpha0_a)
            A_arppu = pm.Gamma('$A_{ARPPU}$', mu=mu_arppu_init_a,
                               sd=std_arppu_init_a)

            # Priors for conversion
            conv_prob_a = pm.Gamma('$p_A$', mu=mu_conv_init_a,
                                   sd=std_conv_init_a)

            # Priors for B group
            # ================================================================
            # Priors for ARPPU
            alpha0_b = pm.Uniform('$\\alpha_0^B$', lower=1,
                                  upper=upper_prior_bound)
            lam_b = pm.Exponential('$\\lambda_B$', lam=alpha0_b)

            B_arppu = pm.Gamma('$B_{ARPPU}$', mu=mu_arppu_init_b,
                               sd=std_arppu_init_b)

            # Priors for conversion
            conv_prob_b = pm.Gamma('$p_B$', mu=mu_conv_init_b,
                                   sd=std_conv_init_b)

            # Likelihoods
            # ================================================================
            A_arppu_lhd = pm.Wald('A_arppu_lhd', mu=A_arppu, lam=lam_a,
                                  observed=observed_A_arppu)
            B_arppu_lhd = pm.Wald('B_arppu_lhd', mu=B_arppu, lam=lam_b,
                                  observed=observed_B_arppu)

            A_conversion_lhd = pm.Bernoulli('A_conversion_lhd', p=conv_prob_a,
                                            observed=observed_A_conversion)
            B_conversion_lhd = pm.Bernoulli('B_conversion_lhd', p=conv_prob_b,
                                            observed=observed_B_conversion)

            # Deterministic stats for a model
            # ================================================================
            A_arpu = pm.Deterministic('$A_{ARPU}$', A_arppu * conv_prob_a)
            B_arpu = pm.Deterministic('$B_{ARPU}$', B_arppu * conv_prob_b)
            # Difference between posterior expected values of model parameters
            delta_conv = pm.Deterministic('$\Delta_C$',
                                          conv_prob_b - conv_prob_a)
            delta_arppu = pm.Deterministic('$\Delta_{ARPPU}$',
                                           B_arppu - A_arppu)
            delta_arpu = pm.Deterministic('$\Delta_{ARPU}$', B_arpu - A_arpu)

            A_arppu_var = pm.Deterministic('$A_{ARPPU} var$', (A_arppu**3/lam_a))
            B_arppu_var = pm.Deterministic('$B_{ARPPU} var$', (B_arppu**3/lam_b))

            delta_std = pm.Deterministic('$\Delta_{std}$',
                                         np.sqrt(B_arppu_var) - np.sqrt(A_arppu_var))
            effect_size = pm.Deterministic(
                'effect size',
                delta_arppu / np.sqrt((A_arppu_var + B_arppu_var) / 2)
            )

    def plot_deltas(self, burn_in):
        return super(ARPUModel, self).plot_result(
            [
             '$\Delta_C$', '$\Delta_{ARPPU}$', '$\Delta_{ARPU}$',
             'effect_size', '$\Delta_{std}$'
            ],
            burn_in, ref_val=0
        )

    def plot_params(self, burn_in):
        return super(ARPUModel, self).plot_result(
            [
                '$p_A$', '$p_B$',
                '$A_{ARPPU}$', '$A_{ARPPU}$',
                '$A_{ARPU}$', '$A_{ARPU}$'
            ],
            burn_in
        )
