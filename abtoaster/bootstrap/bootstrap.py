import numpy as np
import pandas as pd

from joblib import Parallel, delayed, cpu_count


class BootstrapDistribution(pd.Series):
    """"""
    def __init__(self, name=None, bootstrap_func=np.mean, n_samples=2000, sample_len=None, n_jobs=None):
        super(BootstrapDistribution, self).__init__(np.empty(n_samples), name=name)
        self.name = name if name is not None else "Bootstrap Distribution"
        self.func = bootstrap_func
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.sample_len = sample_len
        
    def _generate_indices(self, values, n_iter):
        """
        Generate bootstrap indices with (n_iter, n) shape
        where n_iter is number of bootstrap iterations and n
        """
        if self.sample_len is None:
            n = values.shape[0]
        else:
            n = self.sample_len
        ids = np.random.choice(n, size=(n_iter, n))
        return ids

    def _aggregate(self, values, agg_func, n_iter):
        """Apply agg_func to values array and create n_iter-len result distribution"""
        ids = self._generate_indices(values, n_iter)
        return agg_func(values[ids], axis=1)
    
    @property
    def distribution(self) -> pd.Series:
        """Bootstrap distribution as named pandas.Series"""
        return self.values
    
    def sample(self, arr, verbose=False):
        """Sample bootstrap distribution from given array applying aggregation function
        """
        if self.n_jobs is None:
            n_jobs = cpu_count()

        arr = np.array(arr)
        iter_per_job = self.n_samples // n_jobs
        if verbose: print(f'Starting sampling with {n_jobs} jobs and {iter_per_job} iterations per job.')
        res = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self._aggregate)(arr, self.func, iter_per_job) for i in range(n_jobs))
        res = np.hstack(res)
        if verbose: 
            print(f'Sampling for {self.name} has been done.')
            print(f'Drawn {self.n_samples} samples for "{self.func.__name__}"')
        super(BootstrapDistribution, self).__init__(res, name=self.name)
        return self
