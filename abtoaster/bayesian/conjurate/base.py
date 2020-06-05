# -*- coding: utf-8 -*-
import numpy as np



class Distribution(object):
    """
    Base class for defining distributions.
    """
    def __init__(self, *args, **kwargs):

        self.__parents = list()
        self.__parent_operation = None
        self.__parent_name = 'Default'

    def mean(self):
        """
        Expected value of a distribution.

        Expected value $E(x)$ expressed through distribution's parameters.
        """
        return self._mean()

    def mean_rvs(self, samples=5000):
        """
        Draw samples of expected value from posterior  distribution.
        """
        return self._mean_rvs(samples)

    def _set_parents(self, *args):
        for p in args:
            self.__parents.append(p)

    def _get_parents(self):
        return self.__parents

    def _set_parent_operation(self, func, name):
        self.__parent_operation = func
        self.__parent_name = name

    def _mean(self):
        if self.__parents:
            return self.__parent_operation([i._mean()
                                            for i in self.__parents])
        else:
            raise NotImplementedError

    def _mean_rvs(self, samples):
        if self.__parents:
            return self.__parent_operation([i._mean_rvs(samples)
                                            for i in self.__parents])
        else:
            raise NotImplementedError

    def rvs(self, samples):
        """
        Draw samples from posterior predictive distribution.
        """
        return self._rvs(samples)

    def _rvs(self, samples):
        raise NotImplementedError

    def __mul__(self, k):
        self.k = self._set_k(k)
        return self

    def __sub__(self, dist):
        if not isinstance(dist, Distribution):
            raise TypeError

        child = Distribution()
        child._set_parents(self, dist)

        def subtract(args):
            return np.subtract(*args)

        child._set_parent_operation(subtract, 'Difference')

        return child

    def __lt__(self, dist):
        if not isinstance(dist, Distribution):
            raise TypeError

        child = Distribution()
        child._set_parents(self, dist)

        def lt(args):
            return (args[0] < args[1]) * 1

        child._set_parent_operation(lt, 'Less than')

        return child

    def __gt__(self, dist):
        if not isinstance(dist, Distribution):
            raise TypeError

        child = Distribution()
        child._set_parents(self, dist)

        def gt(args):
            return (args[0] > args[1]) * 1

        child._set_parent_operation(gt, 'Greater than')

        return child

    def _set_k(self, k):
        if isinstance(k, int) or k is None:
            return k
        elif isinstance(k, float):
            k = min(k, 1)
            return k * self.n
        else:
            raise ValueError('k must be 0-1 float or positive integer')
