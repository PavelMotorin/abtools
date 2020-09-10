import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


class BayesianABtest(object):
    
    def __init__(self, groups, models, n_samples=5000, group_names=None, alpha=0.05, step='default', init='auto'):
        print('ABtest for %d groups' % len(groups))
        self.alpha = alpha
        self.n_samples = n_samples
        if group_names:
          self.models = {
                'group%d' % (i + 1): models(group)
                for i, group in enumerate(groups)
            }
        else:
          self.models = {
              group_names[i]: models(group)
              for i, group in enumerate(groups)
          }
        self.traces = [
            self.models[name].sample(self.n_samples, step=step, init=init)
            for name in self.models
        ]
        
   
    def get_summary(self, only_estimated_var=True):
        sum_df = pd.DataFrame()
        for i, name in enumerate(self.models):
            with self.models[name].model:
                temp = pm.summary(self.traces[i])
                
            if only_estimated_var:
                temp = temp[temp.index == self.models[name].estimated_var]
                temp.index = ['%s of %s' % (temp.index[0], name)]
            else:
                temp.index = '%s of %s' % (temp.index, name)
                    
            sum_df = pd.concat([sum_df,temp])
        return sum_df
            
    
    def plot_posterior(self, is_diff=False, hdi_prob=0.95, kind='hist', only_estimated_var=True):
        if len(self.models)>1:
            if is_diff:
                for i, name in enumerate(self.models):
                    if i > 0:
                        with self.models[name].model:
                            print('plot_posterior for group1 - %s' % name)
                            pm.plot_posterior(self.traces[0][self.models[name].estimated_var] - self.traces[i][self.models[name].estimated_var], hdi_prob=hdi_prob, kind=kind)
            
            else:
                for i, name in enumerate(self.models):
                    with self.models[name].model:
                        print('plot_posterior for %s' % name)
                        if only_estimated_var:
                            pm.plot_posterior(self.traces[i], hdi_prob=hdi_prob, kind=kind, var_names=[self.models[name].estimated_var])
                        else:    
                            pm.plot_posterior(self.traces[i], hdi_prob=hdi_prob, kind=kind) 
                
        else:
            with self.models[list(self.models.keys())[0]].model:
                print('plot_posterior for group1')
                if only_estimated_var:
                    pm.plot_posterior(self.traces[0], hdi_prob=hdi_prob, kind=kind, var_names=[self.models[list(self.models.keys())[0]].estimated_var])
                else:
                    pm.plot_posterior(self.traces[0], hdi_prob=hdi_prob, kind=kind)
            

    def traceplot(self, only_estimated_var=True):
        for i, name in enumerate(self.models):
            with self.models[name].model:
                print('traceplot for model %s' % name)
                if only_estimated_var:
                    pm.traceplot(self.traces[i], var_names=[self.models[name].estimated_var])
                else:
                    pm.traceplot(self.traces[i])
                
    def forestplot(self, kind='forestplot', hdi_prob=0.95, only_estimated_var=True):
        with self.models[list(self.models.keys())[0]].model:
            if only_estimated_var:
                pm.forestplot(list(self.traces), model_names=list(self.models.keys()), kind=kind, hdi_prob=hdi_prob, var_names=[self.models[list(self.models.keys())[0]].estimated_var])
            else:
                pm.forestplot(list(self.traces), model_names=list(self.models.keys()), kind=kind, hdi_prob=hdi_prob)
                
    def energyplot(self, kind='kde'):
        for i, name in enumerate(self.models):
            with self.models[name].model:
                print('energyplot for model %s' % name)
                pm.energyplot(self.traces[i], kind=kind)
    
    def densityplot(self, only_estimated_var=True, hdi_prob=0.95):
        if only_estimated_var:
            with self.models[list(self.models.keys())[0]].model:
                pm.densityplot(list(self.traces), data_labels=list(self.models.keys()), hdi_prob=hdi_prob, var_names=[self.models[list(self.models.keys())[0]].estimated_var])
        else:
            with self.models[list(self.models.keys())[0]].model:
                pm.densityplot(list(self.traces), data_labels=list(self.models.keys()), hdi_prob=hdi_prob)
