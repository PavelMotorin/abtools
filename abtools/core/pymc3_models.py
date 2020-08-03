import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


class BernoulliModelPymc3(object):
    """
    Bernoulli model with Bernoulli likelihood
    """
    estimated_var = 'p'
    

#     def __init__(self, x):
#         self.x_obs = np.array(x)
#         self.model = pm.Model()
    
    def __init__(self,x_obs):

        x_obs = np.array(x_obs)

        self.model = pm.Model()
        with self.model:
            p = pm.Uniform('p', 0, 1)
            x = pm.Bernoulli('x', p=p, observed = x_obs)

            
    def sample(self, n, step = 'default',init = 'auto'):
        
        if step ==  'default':
            with self.model:
                return pm.sample(n,
                    init = init)
        elif step == 'metropolis':
            with self.model:
                return pm.sample(n,
                    step=pm.Metropolis(),
                    init = init)
        elif step == 'hamiltonian_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.HamiltonianMC(),
                    init = init)
         elif step == 'sequential_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.SMC(),
                    init = init)
         else:
            print('Error: step parameter should be one of : (default, metropolis, hamiltonian_mc,sequential_mc)')
        
        
class LognormalModelPymc3(object):
    r"""
    Model with log-Normal likelihood.

    Model heavy-tail distributed data.
    x ~ logN(\mu, \tau)
    where \mu is Normal distributed and \tau is Gamma distributed according to
    conjurate priors for log-Normal distribution.

    This model is most stable to outliers and small data size.

    Parameters
    ----------

    Examples
    --------
    Simple usage example with artificial data:

    """
    estimated_var = 'm'

    def __init__(self, x):

        self.model = pm.Model()
        x_obs = np.array(x)

        m = x_obs.mean()
        v = x_obs.var()

        init_mu = np.log(m / np.sqrt(1 + v / (m ** 2)))
        init_tau = 1 / np.log(1 + v / (m ** 2))

        with self.model:

            tau = pm.Gamma('tau', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu = pm.Normal('mu', init_mu, init_tau ** (-2) * 2)

            x = pm.Lognormal('x', mu=mu, tau=tau, observed=x_obs)

            m = pm.Deterministic('m', np.exp(mu + 1/(2 * tau)))

            var = pm.Deterministic('var', (np.exp(1/tau - 1) *
                                           np.exp(2*mu - 1/tau)))

            
    
    def sample(self, n, step = 'default',init = 'auto'):
        
        if step ==  'default':
            with self.model:
                return pm.sample(n,
                    init = init)
        elif step == 'metropolis':
            with self.model:
                return pm.sample(n,
                    step=pm.Metropolis(),
                    init = init)
        elif step == 'hamiltonian_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.HamiltonianMC(),
                    init = init)
         elif step == 'sequential_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.SMC(),
                    init = init)
         else:
            print('Error: step parameter should be one of : (default, metropolis, hamiltonian_mc,sequential_mc)')
                
            
class WaldARPUModel(object):
    """
    
    """
    
    estimated_var = '$ARPU$'
    
    def __init__(self, a_obs):

        #a_obs_r = a['revenue']
        #a_obs_c = a['conversion']

        self.model = pm.Model()
        
        a_obs = np.array(a_obs)
        a_obs_r = np.array(a_obs[a_obs > 0])
        a_obs_c = np.array((a_obs > 0) * 1)
        
        x_min = np.min(a_obs_r)
        x_max = np.max(a_obs_r)

        with self.model:

            # priors
            lam_a = pm.Uniform('$\\lambda$', 0, x_max)
            mu_a = pm.Uniform('$\\mu$', x_min, x_max)

            p_a = pm.Uniform('$p$', 0, 1)
            
            # likelihoods
            a_r = pm.Wald('$R$', mu=mu_a, lam=lam_a, observed=a_obs_r)
            
            a_c = pm.Bernoulli('$C$', p=p_a, observed=a_obs_c)
            
            # deterministic stats
            a_arpu = pm.Deterministic('$ARPU$', mu_a * p_a)
           
            a_var = pm.Deterministic('$\\sigma^2$', mu_a ** 3 / lam_a)
            

    def sample(self, n, step = 'default',init = 'auto'):
        
        if step ==  'default':
            with self.model:
                return pm.sample(n,
                    init = init)
        elif step == 'metropolis':
            with self.model:
                return pm.sample(n,
                    step=pm.Metropolis(),
                    init = init)
        elif step == 'hamiltonian_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.HamiltonianMC(),
                    init = init)
         elif step == 'sequential_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.SMC(),
                    init = init)
         else:
            print('Error: step parameter should be one of : (default, metropolis, hamiltonian_mc,sequential_mc)')
                
                
class LognormalARPUModel(object):
    """
    Mixed A/B ARPU model with log-Normal likelihood for a revenue.

    ARPU model formalizes like follows ARPU = C * ARPPU,
    where C is conversion and ARPPU - expected value of revenue.
    In this model C has a Bernoulli likehood and Uniform prior for $p$, where
    $p$ is conversion probability.ARPPU has a log-Normal likelihood and also
    Uniform priors for $\mu$ and $\tau$.

    Parameters
    ----------

    data : dict
        Dictionary with named arrays of observed values. Must contains
        following keys:
        - A_rev, B_rev - non-zero revenue continuous observations
        - A_conv, B_conv - conversion Bernoulli [0, 1] observations

    Examples
    --------

    Simple usage example with artificial data:

    >>> from scipy.stats import bernoulli, lognorm
    >>> from abtools.bayesian import LognormalARPUABModel
    >>> a_conv = bernoulli.rvs(0.05, size=5000)
    >>> b_conv = bernoulli.rvs(0.06, size=5000)
    >>> a_rev = lognorm.rvs(1.03, size=1000)
    >>> b_rev = lognorm.rvs(1.05, size=1000)
    >>> a = {'revenue': a_rev, 'conversion': a_conv}
    >>> b = {'revenue': b_rev, 'conversion': b_conv}
    >>> model = LognormalARPUABModel(a, b)
    >>> model.fit()
    >>> model.summary()
    """
    
    estimated_var = '$ARPU$'
    
    def __init__(self, a_obs):
        """
        Build ARPU model for compartion of two groups
        """
        # get data from given dict
        #a_obs_r, b_obs_r = np.array(a['revenue']), np.array(b['revenue'])
        #a_obs_c, b_obs_c = np.array(a['conversion']), np.array(b['conversion'])
        self.model = pm.Model()
        a_obs = np.array(a_obs)
        a_obs_r = np.array(a_obs[a_obs > 0])
        a_obs_c = np.array((a_obs > 0) * 1)

        # pool groups statistics
        m = a_obs_r.mean()
        v = a_obs_r.var()
        # init values to make optimization more easy and speed up convergence
        init_mu = np.log(m / np.sqrt(1 + v / (m ** 2)))
        init_tau = 1 / np.log(1 + v / (m ** 2))

        with self.model:

            tau_a = pm.Gamma('$\\tau$', mu=init_tau, sd=init_tau ** (-2) * 2)
            mu_l_a = pm.Normal('$\mu_{ln}$', init_mu, init_tau ** (-2) * 2)

            a = pm.Lognormal('$R$', mu=mu_l_a, tau=tau_a, observed=a_obs_r)
            
            mu_a = pm.Deterministic('$\\mu$', np.exp(mu_l_a+1/(2 * tau_a)))
            
            a_var = pm.Deterministic(
                '$\\sigma^2$',
                (np.exp(1/tau_a - 1) * np.exp(2*mu_l_a - 1/tau_a))
            )

            p_a = pm.Uniform('$p$', 0, 1)
            
            a_c = pm.Bernoulli('$C$', p=p_a, observed=a_obs_c)
            
            a_arpu = pm.Deterministic('$ARPU$', mu_a * p_a)
            

    def sample(self, n, step = 'default',init = 'auto'):
        
        if step ==  'default':
            with self.model:
                return pm.sample(n,
                    init = init)
        elif step == 'metropolis':
            with self.model:
                return pm.sample(n,
                    step=pm.Metropolis(),
                    init = init)
        elif step == 'hamiltonian_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.HamiltonianMC(),
                    init = init)
         elif step == 'sequential_mc':
            with self.model:
                return pm.sample(n,
                    step=pm.SMC(),
                    init = init)
         else:
            print('Error: step parameter should be one of : (default, metropolis, hamiltonian_mc,sequential_mc)')
