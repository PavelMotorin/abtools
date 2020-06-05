# -*- coding: utf-8 -*-
import numpy as np

from .base import Distribution
from .distributions import Bernoulli, Lognormal


class BLModel(Distribution):

    def __init__(self, x=None, mu=None, std=None,
                 alpha=None, n=None, k_b=None, k_l=None):

        self.bernoulli = Bernoulli(
                x=(x > 0) * 1 if x is not None else None,
                alpha=alpha if alpha is not None else None,
                beta=n - alpha if n is not None and alpha is not None else None
            )
        self.lognormal = Lognormal(
                x=x[x > 0] if x is not None else None,
                mu=mu if mu is not None else None,
                std=std if std is not None else None,
                n=alpha if alpha is not None else None
            )

        super(BLModel, self).__init__()
        self._set_parents(self.bernoulli, self.lognormal)

        def prod(args):
            return np.prod(args, axis=0)

        self._set_parent_operation(prod, 'Product')

        self.bernoulli.k = self._set_k(k_b)
        self.lognormal.k = self._set_k(k_l)

    def __rshift__(self, dist):
        if not isinstance(dist, BLModel):
            raise TypeError

        new_b_model = self.bernoulli >> dist.bernoulli
        new_l_model = self.lognormal >> dist.lognormal

        new_bl = BLModel(
                mu=new_l_model.mu,
                std=new_l_model.std,
                alpha=new_b_model.alpha,
                n=new_b_model.n
            )

        return new_bl

    def __mul__(self, k):
        if not isinstance(k, list):
            raise TypeError
        self.bernoulli.k = self._set_k(k[0])
        self.lognormal.k = self._set_k(k[1])

        return self
