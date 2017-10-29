# -*- coding: utf-8 -*-

import numpy as np
import scipy

from scipy.stats import norm, invgamma, lognorm, beta, bernoulli

from .base import Distribution


__all__ = [
    'BernoulliModel',
    'LognormalModel',
    'ARPUModel',
    'NormalModel'
]


class BernoulliPrior(Distribution):
    def __init__(self, x=None, alpha=None, beta=None):
        if x is not None:
            self.mean = x.mean()
            self.var = x.var()
        elif alpha is not None and beta is not None:
            self.mean = alpha / (alpha + beta)
            self.var = self.mean * (1 - self.mean)

class LognormalPrior(Distribution):
    def __init__(self, x=None, mu=None, std=None):
        if x is not None:
            log_x = np.log(x)
            self.mean = log_x.mean()
            self.tau = 1 / log_x.var()
        elif mu is not None and std is not None:
            self.mean = mu
            self.tau = 1 / (std ** 2)


class BernoulliModel(Distribution):
    def __init__(self, x=None, prior=None, k=20, alpha=None, beta=None):
        if x is not None:
            self.alpha = x.sum() + 0.5
            self.beta = len(x) - x.sum() + 0.5
        elif alpha is not None and beta is not None:
            self.alpha = alpha + 0.5
            self.beta = beta + 0.5

        if prior:
            self.alpha += k * prior.mean
            self.beta += k * (1 - prior.mean)

    def _rvs(self, samples):
        return beta.rvs(self.alpha, self.beta, size=samples)

    def _mean(self):
        return self.alpha / (self.alpha + self.beta)

    def _sample_ppc(self, samples):
        return bernoulli.rvs(p=self.mean(), size=samples)


class NormalModel(Distribution):
    """
    """
    def __init__(self, x=None, mu=None, std=None, n=None):
        if x is not None:
            self.mu = x.mean()
            self.sigma = x.std() / np.sqrt(len(x))
        elif mu is not None and std is not None:
            self.mu = mu
            self.sigma = std / n

    def _rvs(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)

    def _mean(self):
        return self.mu

    def _sample_ppc(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)

class LognormalModel(Distribution):
    """
    """
    def __init__(self, x=None, prior=None, k=20, mu=None, std=None, n=None):

        if x is not None:
            log_x = np.log(x)
            n = len(log_x)
            var = log_x.var()
            mean = log_x.mean()

            if var > 0:
                tau = 1 / var
            else:
                tau = 0

        elif mu is not None and std is not None and n is not None:
            mean = mu
            if std > 0:
                tau = 1 / std ** 2
            else:
                tau = 0

        if prior is None:
            prior_mean = 0
            prior_tau = 0
            k = 0
        else:
            prior_mean = prior.mean
            prior_tau = prior.tau

        self.mu__mu = (k * prior_tau * prior_mean + n * tau * mean) / \
                                                    (k * prior_tau + n * tau)
        self.mu__sigma = 1 / (k * prior_tau + n * tau)

        self.var__alpha = (k + n) / 2
        self.var__beta = 0
        if prior_tau > 0:
            self.var__beta += k / 2 / prior_tau
        if tau > 0:
            self.var__beta += n / 2 / tau

    def _rvs(self, samples):
        mu = norm.rvs(self.mu__mu, self.mu__sigma, size=samples)
        var = invgamma.rvs(self.var__alpha, scale=self.var__beta, size=samples)
        return np.exp(mu + var / 2)

    def _mean(self):
        mu = self.mu__mu
        var = self.var__beta / (self.var__alpha - 1)
        return np.exp(mu + var / 2)

    def _sample_ppc(self, samples):
        mu = self.mu__mu
        var = self.var__beta / (self.var__alpha - 1)

        return np.random.lognormal(mean=mu, sigma=var ** (1 / 2), size=samples)


class ARPUPrior(Distribution):
    def __init__(self, x=None, mu=None, std=None, alpha=None, n=None):
        self.bernoulli_prior = BernoulliPrior(
                x=(x > 0) * 1 if x is not None else None,
                alpha=alpha,
                beta=n - alpha
            )
        self.ARPPU_prior = LognormalPrior(
                x=x[x > 0] if x is not None else None,
                mu=mu,
                std=std,
                alpha=alpha
            )

class ARPUModel(Distribution):
    """
    """
    def __init__(self, x=None, prior=None, k1=20, k2=20, mu=None, std=None, alpha=None, n=None):
        self.conversion_model = BernoulliModel(
                x=(x > 0) * 1 if x is not None else None,
                alpha=alpha,
                beta=n - alpha,
                prior=prior.bernoulli_prior if prior is not None else None,
                k=k1
            )
        self.ARPPU_model = LognormalModel(
                x=x[x > 0] if x is not None else None,
                mu=mu,
                std=std,
                alpha=alpha,
                n=n,
                prior=prior.ARPPU_prior if prior is not None else None,
                k=k2
            )

    def _rvs(self, samples):
        return self.conversion_model.rvs(samples) * self.ARPPU_model.rvs(samples)

    def _mean(self):
        return self.conversion_model.mean() * self.ARPPU_model.mean()

    def _sample_ppc(self, samples):
        return self.conversion_model.sample_ppc(samples) * \
                                    self.ARPPU_model.sample_ppc(samples)
