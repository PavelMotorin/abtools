# -*- coding: utf-8 -*-

import numpy as np

from scipy.stats import norm, invgamma, beta, bernoulli

from .base import Distribution


class Bernoulli(Distribution):
    """
    In probability theory and statistics, the Bernoulli distribution, named
    after Swiss mathematician Jacob Bernoulli,[1] is the probability
    distribution of a random variable which takes the value 1 with probability
    p and the value 0 with probability
    q = 1 − p q=1-p — i.e., the probability distribution
    of any single experiment that asks a yes–no question; the question results
    in a boolean-valued outcome, a single bit of information whose value is
    success/yes/true/one with probability p and failure/no/false/zero with
    probability q. It can be used to represent a coin toss where 1 and 0 would
    represent "head" and "tail" (or vice versa), respectively. In particular,
    unfair coins would have p ≠ 0.5 {\displaystyle p\neq 0.5} p\neq 0.5.

    The Bernoulli distribution is a special case of the Binomial distribution
    where a single experiment/trial is conducted (n=1). It is also a special
    case of the two-point distribution, for which the outcome need not be a
    bit, i.e., the two possible outcomes need not be 0 and 1.

    [Source](https://en.wikipedia.org/wiki/Bernoulli_distribution)
    """
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
        """
        Performs transition from prior to posterior with weight `k`.
        """
        if not isinstance(dist, Bernoulli):
            raise TypeError

        k1 = self.k if self.k is not None else self.n
        k2 = dist.k if dist.k is not None else dist.n

        alpha = k1 * self.p + k2 * dist.p
        beta = k1 * (1 - self.p) + k2 * (1 - dist.p)

        return Bernoulli(alpha=alpha, beta=beta)


class Normal(Distribution):
    """
    In probability theory, the normal (or Gaussian) distribution is a very
    common continuous probability distribution. Normal distributions are
    important in statistics and are often used in the natural and social
    sciences to represent real-valued random variables whose distributions are
    not known. A random variable with a Gaussian distribution is said to
    be normally distributed and is called a normal deviate.

    The normal distribution is useful because of the central limit theorem.
    In its most general form, under some conditions (which include finite
    variance), it states that averages of samples of observations of random
    variables independently drawn from independent distributions converge
    in distribution to the normal, that is, become normally distributed when
    the number of observations is sufficiently large. Physical quantities that
    are expected to be the sum of many independent processes (such as
    measurement errors) often have distributions that are nearly normal.
    Moreover, many results and methods (such as propagation of uncertainty
    and least squares parameter fitting) can be derived analytically in
    explicit form when the relevant variables are normally distributed.

    [Source](https://en.wikipedia.org/wiki/Normal_distribution)

    Parameters
    ==========
    :x: array or pandas Series
        Observed data
    :mu: float
        μ is the mean or expectation of the distribution (and also
        its median and mode).
    :std: float
        σ is the standard deviation
    :n: int
        Sample size
    :k: int
        Number of observations when distribution used as prior
    """
    def __init__(self, x=None, mu=None, std=None, n=None, k=None, *args, **kwargs):
        if x is not None:
            self.mu = x.mean()
            self.std = x.std()
            if not (self.std > 0 and
                    not np.isinf(self.std) and
                    not np.isnan(self.std)):
                self.std = 0.99
            self.sigma = self.std / np.sqrt(len(x))
            self.var = self.sigma ** 2
            self.tau = 1 / self.var
        elif mu is not None and std is not None:
            self.mu = mu
            if std <= 0 or np.isinf(std) or np.isnan(std):
                std = 0.99
            self.std = std
            self.tau = 1 / std ** 2

            self.sigma = std / np.sqrt(n)
            self.var = self.sigma ** 2
            if self.var > 0:
                self.tau = 1 / self.var
            else:
                self.tau = 0

        self.n = max(n, 1)
        if self.tau > 0:
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
        """
        Performs transition from prior to posterior with weight `k`.
        """
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
    In probability theory, a log-normal (or lognormal) distribution is a
    continuous probability distribution of a random variable whose logarithm
    is normally distributed. Thus, if the random variable X is log-normally
    distributed, then Y = ln(X) has a normal distribution. Likewise, if Y has
    a normal distribution, then the exponential function of Y, X = exp(Y), has
    a log-normal distribution. A random variable which is log-normally
    distributed takes only positive real values.

    [Source](https://en.wikipedia.org/wiki/Log-normal_distribution)
    Parameters
    ==========
    :x: array or pandas Series
        Observed data
    :mu: float
        μ is the mean or expectation of the distribution (and also
        its median and mode).
    :std: float
        σ is the standard deviation
    :n: int
        Sample size
    :k: int
        Number of observations when distribution used as prior
    """
    def __init__(self, x=None, mu=None, std=None, n=None, k=None):

        self.empty = False
            
        if x is not None:
            x = np.array(x)
            log_x = np.log(x)
            self.n = len(log_x)
            std = log_x.std()
            if std <= 0 or np.isinf(std) or np.isnan(std):
                std = 0.33
            var = std ** 2
            self.mu = log_x.mean()
            self.std = var ** (1/2)
            self.tau = 1 / var

        elif mu is not None and std is not None and n is not None:
            if std <= 0 or np.isinf(std):
                std = 0.33
                self.empty = True
            if mu <= 0 or np.isinf(mu):
                mu = 0
                self.empty = True
            self.mu = mu
            self.std = std
            self.tau = 1 / std ** 2
            self.n = max(n, 5)

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
        mean = np.exp(mu + var / 2)
        return mean

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
        """
        Performs transition from prior to posterior with weight `k`.
        """
        if dist.empty:
            return self
        elif self.empty:
            return dist
        else:
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
