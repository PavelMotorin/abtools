# -*- coding: utf-8 -*-

import numpy as np

from scipy.stats import norm, invgamma, beta, bernoulli

from .base import Distribution


class Bernoulli(Distribution):

    def __init__(self, x=None, alpha=None, beta=None, k=None):

        # set distribution's parameters if data passed in args
        if x is not None:
            x = np.array(x)
            self.alpha = x.sum()
            self.beta = len(x) - self.alpha

        # set distribution's parameters from passed variables
        elif alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta

        self.n = self.alpha + self.beta
        self.p = self.alpha / self.n
        self.k = self._set_k(k)

        super(Bernoulli, self).__init__()

    def _mean(self):
        return self.p

    def _var(self):
        return self.p * (1 - self.p)

    def _mean_rvs(self, samples):
        return beta.rvs(self.alpha, self.beta, size=samples)

    def _var_rvs(self, samples):
        return 1 - self._mean_rvs(samples)

    def _rvs(self, samples):
        return bernoulli.rvs(p=self.p, size=samples)

    def __rshift__(self, dist):
        if not isinstance(dist, Bernoulli):
            raise TypeError

        k1 = self.k if self.k is not None else self.n
        k2 = dist.k if dist.k is not None else dist.n

        alpha = k1 * self.p + k2 * dist.p
        beta = k1 * (1 - self.p) + k2 * (1 - dist.p)

        return Bernoulli(alpha=alpha, beta=beta)


class Normal(Distribution):
    """
    """
    def __init__(self, x=None, mu=None, std=None, n=None, k=None):
        if x is not None:
            self.mu = x.mean()
            self.std = x.std()
            self.sigma = x.std() / np.sqrt(len(x))
            self.var = self.sigma ** 2
            self.tau = 1 / self.var
        elif mu is not None and std is not None:
            self.mu = mu
            self.std = std
            self.sigma = std / np.sqrt(n)
            self.var = self.sigma ** 2
            self.tau = 1 / self.var

        self.n = n
        self.var__alpha = (self.n - 1) / 2
        self.var__beta = self.var__alpha / self.tau
        self.k = self._set_k(k)

        super(Normal, self).__init__()

    def _mean_rvs(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)

    def _mean(self):
        return self.mu

    def _rvs(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)

    def __rshift__(self, dist):
        if not isinstance(dist, Normal):
            raise TypeError

        k1 = self.k if self.k is not None else self.n
        k2 = dist.k if dist.k is not None else dist.n

        mu__mu = ((k2 * dist.tau * dist.mu + k1 * self.tau * self.mu) /
                  (k2 * dist.tau + k1 * self.tau))

        var__alpha = (k1 + k2) / 2
        var__beta = 0

        if dist.tau > 0:
            var__beta += k2 / 2 / dist.tau
        if self.tau > 0:
            var__beta += k1 / 2 / self.tau

        mu = mu__mu
        std = np.sqrt(var__beta / (var__alpha - 1))

        child = Normal(mu=mu, std=std, n=k1 + k2)
        child.var__alpha = var__alpha
        child.var__beta = var__beta

        return child


class Lognormal(Distribution):
    """
    """
    def __init__(self, x=None, mu=None, std=None, n=None, k=None):

        if x is not None:
            x = np.array(x)
            log_x = np.log(x)
            self.n = len(log_x)
            var = log_x.var()
            self.mu = log_x.mean()
            self.std = var ** (1/2)

            if var > 0:
                self.tau = 1 / var
            else:
                self.tau = 0

        elif mu is not None and std is not None and n is not None:
            self.mu = mu
            if std > 0:
                self.tau = 1 / std ** 2
            else:
                self.tau = 0
            self.n = n
            self.std = std

        self.mu__mu = self.mu
        self.mu__sigma = 1 / np.sqrt(self.n * self.tau)

        self.var__alpha = (self.n - 1) / 2
        self.var__beta = self.var__alpha / self.tau
        self.k = self._set_k(k)

        super(Lognormal, self).__init__()

    def _mean_rvs(self, samples):
        mu = norm.rvs(self.mu__mu, self.mu__sigma, size=samples)
        var = invgamma.rvs(self.var__alpha, scale=self.var__beta, size=samples)
        return np.exp(mu + var / 2)

    def _var_rvs(self, samples):
        mu = norm.rvs(self.mu__mu, self.mu__sigma, size=samples)
        var = invgamma.rvs(self.var__alpha, scale=self.var__beta, size=samples)
        return (np.exp(var) - 1) * np.exp(2 * mu + var)

    def _mean(self):
        mu = self.mu__mu
        var = self.var__beta / (self.var__alpha - 1)
        return np.exp(mu + var / 2)

    def _var(self):
        mu = self.mu__mu
        var = self.var__beta / (self.var__alpha - 1)
        return (np.exp(var) - 1) * np.exp(2 * mu + var)

    def _rvs(self, samples):
        mu = self.mu__mu
        var = self.var__beta / (self.var__alpha - 1)

        return np.random.lognormal(mean=mu, sigma=var ** (1 / 2), size=samples)

    def __rshift__(self, dist):
        if not isinstance(dist, Lognormal):
            raise TypeError

        k1 = self.k if self.k is not None else self.n
        k2 = dist.k if dist.k is not None else dist.n

        mu__mu = ((k2 * dist.tau * dist.mu + k1 * self.tau * self.mu) /
                  (k2 * dist.tau + k1 * self.tau))

        var__alpha = (k1 + k2) / 2
        var__beta = 0

        if dist.tau > 0:
            var__beta += k2 / 2 / dist.tau
        if self.tau > 0:
            var__beta += k1 / 2 / self.tau

        mu = mu__mu
        std = np.sqrt(var__beta / (var__alpha - 1))

        child = Lognormal(mu=mu, std=std, n=k1 + k2)
        child.var__alpha = var__alpha
        child.var__beta = var__beta

        return child
