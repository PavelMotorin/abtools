import pandas as pd
import numpy as np
import altair as alt

from itertools import product

from .bootstrap import BootstrapDistribution
from . import funcs


class BootstrapAB:
    """Bootstrap A/B test"""
    def __init__(self, name=None, bootstrap_func=None, hypothesis_func=None, verbose=1):
        self.name = name if name is not None else 'Measurement'
        self.bootstrap_func = np.mean if bootstrap_func is None else bootstrap_func
        self.hypothesis_func = funcs.diff if hypothesis_func is None else hypothesis_func
        self.verbose = verbose
        self._variant_dists = {}
        self._result = {}
        
    def _evaluate_pair(self, a_key:str, b_key:str):
        """Apply hypothesis comparison function to given variant keys (a and b)"""
        a = self._variant_dists[a_key]
        b = self._variant_dists[b_key]
        result = self.hypothesis_func(a, b)
        if a_key not in self._result:
            self._result[a_key] = {}
        self._result[a_key][b_key] = result
        return result

    def fit(self, data:pd.DataFrame, metric:str, variant_key:str, n_samples=2000, n_jobs=None) -> None:
        """
        Runs A/B test for given data.
        
        Use given metric as target for this test and variant_key to select variants from data.
        """
        variants = data[variant_key].unique()
        
        for variant in variants:
            b_dist = BootstrapDistribution(variant, self.bootstrap_func, n_samples, n_jobs)
            arr = data.loc[data[variant_key] == variant, metric]
            self._variant_dists[variant] = b_dist.sample(arr)

        for a_key, b_key in product(variants, variants):
            if a_key != b_key:
                self._evaluate_pair(a_key, b_key)

        return self

    def visualize(self, *args, **kwargs):
        """Plot all Bootstrap distributions of variants found in the A/B test data"""
        xbin = alt.Bin(maxbins=100)

        # Generating Data
        source = pd.DataFrame(self._variant_dists)
        names = list(source.columns)

        chart = alt.Chart(source).transform_fold(
            names, as_=['Variant', self.name]
        ).mark_area(
            opacity=0.5,
            interpolate='step'
        ).encode(
            alt.X(f'{self.name}:Q', bin=xbin),
            alt.Y('count()', stack=False),
            alt.Color('Variant:N')
        )
        return chart

        

