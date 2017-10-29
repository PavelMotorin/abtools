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
    def __init__(self, x=None):
        if x:
            self.mean = x.mean()
            self.var = x.var()

    def from_stats(self, alpha, beta):
        self.mean = alpha / (alpha + beta)
        self.var = self.mean * (1 - self.mean)
        return self

class LognormalPrior(Distribution):
    def __init__(self, x=None):
        if x:
            log_x = np.log(x)
            self.mean = log_x.mean()
            self.tau = 1 / log_x.var()

    def from_stats(self, mean, tau):
        self.mean = mean
        self.tau = tau
        return self

class BernoulliModel(Distribution):
    def __init__(self, x=None, prior=None, k=20):
        if x:
            self.alpha = x.sum() + 0.5
            self.beta = len(x) - x.sum() + 0.5

            if prior:
                self.alpha += k * prior.mean
                self.beta += k * (1 - prior.mean)

    def from_stats(self, alpha, beta, prior=None, k=20):
        self.alpha = alpha + 0.5
        self.beta = beta + 0.5

        if prior:
            self.alpha += k * prior.mean
            self.beta += k * (1 - prior.mean)

        return self

    def _rvs(self, samples):
        return beta.rvs(self.alpha, self.beta, size=samples)

    def _mean(self):
        return self.alpha / (self.alpha + self.beta)

    def _sample_ppc(self, samples):
        return bernoulli.rvs(p=self.mean(), size=samples)


class NormalModel(Distribution):
    """
    """
    def __init__(self, x=None):
        if x:
            self.mu = x.mean()
            self.sigma = x.std() / np.sqrt(len(x))

    def from_stats(self, mu, sigma, n):
        self.mu = mu
        self.sigma = sigma / n

        return self

    def _rvs(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)

    def _mean(self):
        return self.mu

    def _sample_ppc(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)

class LognormalModel(Distribution):
    """
    """
    def __init__(self, x, prior=None, k=20):
        log_x = np.log(x)

        n = len(log_x)

        if n > 0:
            mean = log_x.mean()
            if log_x.var() > 0:
                tau = 1 / log_x.var()
            else:
                tau = 0
        else:
            mean = 0
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

    def from_stats(self, mu, std, n, prior=None, k=20):
        if n > 0:
            mean = mu
            if std ** 2 > 0:
                tau = 1 / (std ** 2)
            else:
                tau = 0
        else:
            mean = 0
            tau = 0

        if not prior:
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
    def __init__(self, x):
        self.bernoulli_prior = BernoulliPrior((x > 0) * 1)
        self.ARPPU_prior = LognormalPrior(x[x > 0])

    def from_stats(self, mu, std, alpha, n):
        self.bernoulli_prior = BernoulliPrior().from_stats(alpha, n - alpha)
        self.ARPPU_prior = LognormalPrior().from_stats(mu, std, alpha)

class ARPUModel(Distribution):
    """
    """
    def __init__(self, x, prior=None, k1=20, k2=20):
        if prior:
            b_p = prior.bernoulli_prior
            l_p = prior.ARPPU_prior
            self.conversion_model = BernoulliModel((x > 0) * 1, prior=b_p, k=k1)
            self.ARPPU_model = LognormalModel(x[x > 0], prior=l_p, k=k2)
        else:
            self.conversion_model = BernoulliModel((x > 0) * 1, prior=None, k=k1)
            self.ARPPU_model = LognormalModel(x[x > 0], prior=None, k=k2)

    def from_stats(self, mu, std, alpha, n, prior=None, k1=20, k2=20):

        self.conversion_model = BernoulliModel(
                alpha,
                n - alpha,
                prior=prior.bernoulli_prior if prior is not None else None,
                k=k1
            )
        self.ARPPU_model = LognormalModel(
                mu, 
                std,
                alpha,
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
