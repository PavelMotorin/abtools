# -*- coding: utf-8 -*-

import numpy as np
import scipy

from scipy.stats import norm, invgamma, lognorm, beta

from .base import BaseModel


__all__ = [
    'BernoulliModel',
    'LognormalModel',
    'ARPUModel',
    'NormalModel'
]


class BernoulliModel(BaseModel):
    
    """
    Model to fit the bernoulli distribution of random variates.
    """
    
    def __init__(self, x):
        self.alpha = x.sum() + 0.5
        self.beta = len(x) - self.alpha + 0.5

    def _rvs(self, samples):
        return beta.rvs(self.alpha, self.beta, size=samples)


class NormalModel(BaseModel):
    
    """
    Model to fit the normal distribution of random variates.
    """
    
    def __init__(self, x):
        self.mu = x.mean()
        self.sigma = x.std() / np.sqrt(len(x))

    def _rvs(self, samples):
        return norm.rvs(self.mu, self.sigma, size=samples)


class LognormalModel(BaseModel):
    
    """
    Model to fit the lognormal distribution of random variates.
    """
    
    def __init__(self, x):
        log_x = np.log(x)
        std = log_x.std()
        n = len(log_x)
        self.mu__mu = log_x.mean()
        self.mu__sigma = std / np.sqrt(n)
        self.var__alpha = (n - 1) / 2
        self.var__beta = self.var__alpha * std**2

    def _rvs(self, samples):
        mu = norm.rvs(self.mu__mu, self.mu__sigma, size=samples)
        var = invgamma.rvs(self.var__alpha, scale=self.var__beta, size=samples)
        return np.exp(mu + var / 2)

    def sample_ppc(self, samples):
        mu = np.mean(norm.rvs(self.mu__mu, self.mu__sigma, size=samples))
        var = np.mean(invgamma.rvs(self.var__alpha, scale=self.var__beta, size=samples))

        return np.random.lognormal(mean=mu, sigma=var ** (1 / 2), size=samples)


class ARPUModel(BaseModel):
     
    """
    Model to fit the combined lognormal and bernoulli distribution of random variates.
    """
    
    def __init__(self, x):
        self.conversion_model = BernoulliModel((x > 0) * 1)
        self.ARPPU_model = LognormalModel(x[x > 0])

    def _rvs(self, samples):
        return self.conversion_model.rvs(samples) * self.ARPPU_model.rvs(samples)
