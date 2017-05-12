# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import norm
from ..utils import se


__all__ = [
    'permutation_test',
    'ztest',
    'ttest',
    'clt'
]


def permutation_test(a, b, iters=10000, alpha=0.05, print_stats=True,
                     plot=True, plt_kwds={}, hist_kwds={}):
    """
    Implementation of Fisher's permutation test.
    
    A Fisher's exact permutation test is a type of statistical significance 
    test in which the distribution of the test statistic under the 
    null hypothesis is obtained by calculating all possible values of the test 
    statistic under rearrangements of the labels on the observed data points. 
    In other words, the method by which treatments are allocated to subjects in
    an experimental design is mirrored in the analysis of that design. If the 
    labels are exchangeable under the null hypothesis, then the resulting tests 
    yield exact significance levels.
    
    Parameters
    ----------
    a, b : {list, ndarray}
        Observed data for two groups, where a - control group,  b - test group.
    iters : int
        Number of permutations.
    alpha : float
        Significance level.
    plot : bool
        Perform of not plotting of null distribution.
    plt_kwds : dict
        Matplotlib's figure parameters.
    hist_kwds: dict
        Matplotlib's histogram parameters.
        
    
    Returns
    -------
    p_value : float
        Two-sided p-value.
    sign : bool
        Rejected or not null hypotesis with given significance level.
    
    References
    ----------
    
    [1] - Fisher, R. A. (1935). The design of experiments. 1935. 
    Oliver and Boyd, Edinburgh.
    
    [2] - Ernst, M. D. (2004). Permutation methods: a basis for exact inference. 
    Statistical Science, 19(4), 676-685
    """
    a, b = np.array(a), np.array(b)
    
    n, p_value, null_dist = len(a), 0, []
    
    # compute absolute difference of group means
    diff_obs = b.mean() - a.mean()
    
    # pool groups together
    z = np.concatenate([a, b])
    
    # start permutations
    for j in tqdm(range(iters)):
        np.random.shuffle(z)
        null_dist.append(np.mean(z[:n]) - np.mean(z[n:]))
    
    null_dist = np.array(null_dist)
    
    # compute p-value
    p_value = np.mean(np.abs(diff_obs) < np.abs(null_dist))
    sign = (p_value <= alpha)
    CI = [np.percentile(null_dist, p*100) for p in [0 + alpha/2, 1 - alpha/2]]
    ci_label = '%d%% confidence interval' % ((1-alpha) * 100)
    
    if print_stats:
        print("Fisher's exact permutation test with %d permutations:" % iters)
        print("----------------------------------------------------------")
        print("\tObserved difference of means (E(B) - E(A)) = %.2f" % diff_obs)
        print("\tNull distribution's %s = [%.2f, %.2f]" % (ci_label, *CI))
        print("\tTwo-sided p-value =", p_value)
        print("\tNull hypotesis is %s" % ('not ' * (1 * (not sign)) + 'rejected'))
        
    # plot null distribution
    if plot:
        fig = plt.figure(**plt_kwds)
        plt.hist(null_dist, 50, histtype='step', 
                        color='r', label='Null distribution', 
                        **hist_kwds)
        
        plt.legend()
        plt.axvline(diff_obs, label='Observed difference')
        plt.axvline(CI[0], linestyle='--')
        plt.axvline(CI[1], linestyle='--', label=ci_label)
        
        plt.title("Fisher's permutation test")
        plt.legend(bbox_to_anchor=(1.05, .65))
        
    return p_value, sign

def ztest(a, b, alpha=0.05, print_stats=True, plot=True):
    """
    Fisher's z-test
    """
    def ci(x, alpha=0.05):
        x = np.array(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.norm.ppf(1-alpha/2)
        return m-h, m+h
    
    a, b = np.array(a), np.array(b)
    
    a_mean = a.mean()
    avar = a.var()
    na = a.size

    b_mean = b.mean()
    bvar = b.var()
    nb = b.size
    
    z = ((a_mean - b_mean)) / np.sqrt(avar/na + bvar/nb)
    z_critical = sp.stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - sp.stats.norm.cdf(abs(z)))
    sign = (p_value <= alpha)
    
    ci_a, ci_b = ci(a, alpha), ci(b, alpha)
    
    if print_stats:
        print("Fisher's z-test:")
        print("--------------------------")
        print("\tObserved difference of means (E(B) - E(A)) = %.2f" % (b.mean() - a.mean()))
        print('\tz-statistic = %.4f and z-critical = %.4f' % (z, z_critical))
        print("\tA group's mean %d%% confidence interval = [%.2f, %.2f]" % ((1-alpha)*100, *ci_a))
        print("\tB group's mean %d%% confidence interval = [%.2f, %.2f]" % ((1-alpha)*100, *ci_b))
        print("\tTwo-sided p-value =", p_value)
        print("\tNull hypotesis is %s" % ('not ' * (1 * (not sign)) + 'rejected'))
        
        if z > z_critical:
            result = 'E(A) > E(B)'
        elif z < -z_critical:
            result = 'E(A) < E(B)'
        else:
            result = 'E(A) = E(B)'
        print('\t' + result)
    
    if plot:
        pass
    
    return p_value, sign, result
                   

def ttest(a, b, alpha=0.05, print_stats=True, plot=True):
    """
    Student's unpaired t-test
    """
    def ci(x, alpha=0.05):
        x = np.array(x)
        n = len(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.t.ppf(1-alpha/2, df=n-1)
        return m-h, m+h
    
    a, b = np.array(a), np.array(b)
    
    a_mean = a.mean()
    avar = a.var(ddof=1)
    na = a.size
    adf = na - 1

    b_mean = b.mean()
    bvar = b.var(ddof=1)
    nb = b.size
    bdf = nb - 1

    t = (a_mean - b_mean) / np.sqrt(avar/na + bvar/nb)
    df = (avar/na + bvar/nb)**2 / (avar**2/(na**2*adf) + bvar**2/(nb**2*bdf))
    t_critical = sp.stats.t.ppf(1-alpha/2, df=df)
    p_value = 2 * sp.stats.t.cdf(-np.abs(t), df=df)
    sign = (p_value <= alpha)
    
    ci_a, ci_b = ci(a, alpha), ci(b, alpha)
    
    if print_stats:
        print("Student's unpaired t-test:")
        print("--------------------------")
        print("\tObserved difference of means (E(B) - E(A)) = %.2f" % (b.mean() - a.mean()))
        print('\tt-statistic = %.4f and t-critical = %.4f' % (t, t_critical))
        print("\tA group's mean %d%% confidence interval = [%.2f, %.2f]" % ((1-alpha)*100, *ci_a))
        print("\tB group's mean %d%% confidence interval = [%.2f, %.2f]" % ((1-alpha)*100, *ci_b))
        print("\tTwo-sided p-value =", p_value)
        print("\tNull hypotesis is %s" % ('not ' * (1 * (not sign)) + 'rejected'))
        
        if t > t_critical:
            result = 'E(A) > E(B)'
        elif t < -t_critical:
            result = 'E(A) < E(B)'
        else:
            result = 'E(A) = E(B)'
        print('\t' + result)
    
    if plot:
        pass
    
    return p_value, sign, result

def clt():
    pass