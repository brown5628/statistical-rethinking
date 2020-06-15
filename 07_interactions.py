# %%
import pymc3 as pm 
import numpy as np 
import pandas as pd 
from scipy import stats 
from scipy.interpolate import griddata 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 
import arviz as az  

# %%
d = pd.read_csv('data/rugged.csv', sep=';', header=0)

# %%
d['log_gdp'] = np.log(d.rgdppc_2000)
dd = d[np.isfinite(d['rgdppc_2000'])]
dA1 = dd[dd.cont_africa==1]
dA0 = dd[dd.cont_africa==0]

# %%
with pm.Model() as model_7_2:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)

    mu = pm.Deterministic('mu', a + bR * dA1['rugged'])
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dA1['rgdppc_2000']))
    trace_7_2 = pm.sample(1000, tune=1000)

# %%
varnames = ['~mu']
pm.traceplot(trace_7_2, varnames)

# %%
with pm.Model() as model_7_2_2: 
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10) 

    mu = pm.Deterministic('mu', a + bR * dA0['rugged'])
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dA0['rgdppc_2000']))
    trace_7_2_2 = pm.sample(1000, tune=1000)

# %%
pm.traceplot(trace_7_2_2, varnames)

# %%
mu_mean = trace_7_2['mu']
mu_hpd = pm.hpd(mu_mean)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,3))
ax1.plot(dA1['rugged'], np.log(dA1['rgdppc_2000']), 'C0o')
ax1.plot(dA1['rugged'], mu_mean.mean(0), 'C1')
az.plot_hpd(dA1['rugged'], mu_mean, ax=ax1)
ax1.set_title('Africa')
ax1.set_ylabel('log(rgdppc_2000')
ax1.set_xlabel('rugged')

mu_mean = trace_7_2_2['mu']

ax2.plot(dA0['rugged'], np.log(dA0['rgdppc_2000']), 'ko')
ax2.plot(dA0['rugged'], mu_mean.mean(0), 'C1')
ax2.set_title('not Africa')
ax2.set_ylabel('log(rgdppc_200)')
ax2.set_xlabel('rugged')
az.plot_hpd(dA0['rugged'], mu_mean, ax=ax2)

# %%
with pm.Model() as model_7_3:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper =10)
    mu=pm.Deterministic('mu', a+bR * dd.rugged)
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_3 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_7_4:
    a = pm.Normal('a', mu=8, sd=1000)
    bR = pm.Normal('bR', mu=0, sd=1)
    bA = pm.Normal('bA', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a+bR * dd.rugged + bA * dd.cont_africa) 
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_4 = pm.sample(1000, tune=1000)

# %%
comp_df = az.compare({'m7.3': trace_7_3, 'm7.4': trace_7_4})
comp_df

# %%
az.plot_compare(comp_df)

# %%
rugged_seq = np.arange(-1, 9, .25)

mu_pred_NotAfrica = np.zeros((len(rugged_seq), len(trace_7_4['bR'])))
mu_pred_Africa = np.zeros((len(rugged_seq), len(trace_7_4['bR'])))

for iSeq, seq in enumerate(rugged_seq):
    mu_pred_NotAfrica[iSeq] = trace_7_4['a'] + trace_7_4['bR'] * rugged_seq[iSeq] + trace_7_4['bA'] * 0
    mu_pred_Africa[iSeq] = trace_7_4['a'] + trace_7_4['bR'] * rugged_seq[iSeq] + trace_7_4['bA'] * 1

# %%
mu_mean_NotAfrica = mu_pred_NotAfrica.mean(1)
mu_mean_Africa = mu_pred_Africa.mean(1)

# %%
plt.plot(dA1['rugged'], np.log(dA1['rgdppc_2000']), 'C0o')
plt.plot(rugged_seq, mu_mean_Africa, 'C0')
az.plot_hpd(rugged_seq, mu_pred_Africa.T, credible_interval=.97, color='C0')
plt.plot(dA0['rugged'], np.log(dA0['rgdppc_2000']), 'ko')
az.plot_hpd(rugged_seq, mu_pred_NotAfrica.T, credible_interval=.97, color='k')
plt.annotate('not Africa', xy=(6, 9.5))
plt.annotate('Africa', xy=(6,6))
plt.ylabel('log(rgdppc_2000)')
plt.xlabel('rugged')

# %%
with pm.Model() as model_7_5:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    bA = pm.Normal('bA', mu=0, sd=1)
    bAR = pm.Normal('bAR', mu = 0, sd =1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    gamma = bR + bAR * dd.cont_africa
    mu = pm.Deterministic('mu', a + gamma * dd.rugged + bA * dd.cont_africa)
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_5 = pm.sample(1000, tune=1000)

# %%
comp_df = az.compare({'m7.3': trace_7_3,
                      'm7.4' : trace_7_4,
                      'm7.5' : trace_7_5})

comp_df

# %%
az.plot_compare(comp_df)


# start from code 7.7