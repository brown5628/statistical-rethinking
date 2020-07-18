# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm 
import scipy as sp
import arviz as az
from collections import OrderedDict
from theano import shared

# %%
trolley_df = pd.read_csv('/home/brown5628/projects/statistical-rethinking/data/Trolley.csv', sep=';')
trolley_df.head()

# %%
ax = (trolley_df.response
                .value_counts()
                .sort_index()
                .plot(kind='bar'))

ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)

# %%
ax = (trolley_df.response
                .value_counts()
                .sort_index()
                .cumsum()
                .div(trolley_df.shape[0])
                .plot(marker='o'))

ax.set_xlim(.9, 7.1)
ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("cumulative proportion", fontsize=14)

# %%
resp_lco = (trolley_df.response
                        .value_counts()
                        .sort_index()
                        .cumsum()
                        .iloc[:-1]
                        .div(trolley_df.shape[0])
                        .apply(lambda p: np.log(p / (1. - p))))

# %%
ax = resp_lco.plot(marker='o')

ax.set_xlim(.9, 7)
ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("log-cumulative-odds", fontsize=14)

# %%
with pm.Model() as m11_1:
    a = pm.Normal(
        'a', 0., 10.,
        transform=pm.distributions.transforms.ordered,
        shape=6, testval=np.arange(6)-2.5)

    resp_obs = pm.OrderedLogistic(
        'resp_obs', 0., a, 
        observed=trolley_df.response.values-1

    )

# %%
with m11_1:
    map_11_1 = pm.find_MAP()

# %%
map_11_1['a']

# %%
sp.special.expit(map_11_1['a'])

# %%
with m11_1:
    trace_11_1 = pm.sample(1000, tune=1000)

# %%


def ordered_logistic_proba(a):
    pa = sp.special.expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))

    return p_cum[1:] - p_cum[:-1]

# %%
ordered_logistic_proba(trace_11_1['a'].mean(axis=0))

# %%
(ordered_logistic_proba(trace_11_1['a'].mean(axis=0)) \
     * (1 + np.arange(7))).sum()

# %%
ordered_logistic_proba(trace_11_1['a'].mean(axis=0) - 0.5)

# %%
(ordered_logistic_proba(trace_11_1['a'].mean(axis=0) - 0.5) \
     * (1 + np.arange(7))).sum()

# %%
action = shared(trolley_df.action.values)
intention = shared(trolley_df.intention.values)
contact = shared(trolley_df.contact.values)

with pm.Model() as m11_2:
    a = pm.Normal(
        'a', 0., 10.,
        transform=pm.distributions.transforms.ordered,
        shape=6,
        testval=trace_11_1['a'].mean(axis=0)
    )

    bA = pm.Normal('bA', 0., 10.)
    bI = pm.Normal('bI', 0., 10.)
    bC = pm.Normal('bC', 0., 10.)
    phi = bA * action + bI * intention + bC * contact 

    resp_obs = pm.OrderedLogistic(
        'resp_obs', phi, a,
        observed=trolley_df.response.values -1
    )

# %%
with m11_2:
    map_11_2 = pm.find_MAP()

# %%
with pm.Model() as m11_3:
    a = pm.Normal(
        'a', 0., 10.,
        transform=pm.distributions.transforms.ordered,
        shape = 6,
        testval=trace_11_1['a'].mean(axis=0)
    )

    bA = pm.Normal('bA', 0., 10.)
    bI = pm.Normal('bI', 0., 10.)
    bC = pm.Normal('bC', 0., 10.)
    bAI = pm.Normal('bAI', 0., 10.)
    bCI = pm.Normal('bCI', 0., 10.)
    phi  = bA * action + bI * intention + bC * contact \
            + bAI * action * intention \
            + bCI * contact * intention

    resp_obs = pm.OrderedLogistic(
        'resp_obs', phi, a,
        observed=trolley_df.response - 1
    )

# %%
with m11_3:
    map_11_3 = pm.find_MAP()

# %%


def get_coefs(map_est):
    coefs = OrderedDict()
    
    for i, ai in enumerate(map_est['a']):
        coefs['a_{}'.format(i)] = ai
        
    coefs['bA'] = map_est.get('bA', np.nan)
    coefs['bI'] = map_est.get('bI', np.nan)
    coefs['bC'] = map_est.get('bC', np.nan)
    coefs['bAI'] = map_est.get('bAI', np.nan)
    coefs['bCI'] = map_est.get('bCI', np.nan)
        
    return coefs

# %%
(pd.DataFrame.from_dict(
    OrderedDict([
        ('m11_1', get_coefs(map_11_1)),
        ('m11_2', get_coefs(map_11_2)),
        ('m11_3', get_coefs(map_11_3))
    ]))
   .astype(np.float64)
   .round(2))

# %%
with m11_2:
    trace_11_2 = pm.sample(1000, tune=1000)

with m11_3:
    trace_11_3 = pm.sample(1000, tune=1000)

# %%
comp_df = pm.compare({m11_1:trace_11_1,
                      m11_2:trace_11_2,
                      m11_3:trace_11_3})

comp_df.loc[:,'model'] = pd.Series(['m11.1', 'm11.2', 'm11.3'])
comp_df = comp_df.set_index('model')
comp_df

# %%
