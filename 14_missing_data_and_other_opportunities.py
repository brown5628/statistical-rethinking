# %%
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# %%
pancake = np.array([[1, 1], [1, 0], [0, 0]])
# simulate a pancake and return randomly ordered sides
pancakes = np.asarray(
    [np.random.permutation(pancake[np.random.choice(range(3))]) for i in range(10000)]
)
up = pancakes[:, 0]
down = pancakes[:, 1]
# compute proportion 1/1 (BB) out of all 1/1 and 1/0
num_11_10 = np.sum(up == 1)
num_11 = np.sum((up == 1) & (down == 1))
num_11 / num_11_10

# %%
d = pd.read_csv("data/WaffleDivorce.csv", ";")
d["log_population"] = np.log(d["Population"])

_, ax = plt.subplots(1, 2, figsize=(18, 5))

# points
ax[0].scatter(
    d["MedianAgeMarriage"],
    d["Divorce"],
    marker="o",
    facecolor="none",
    edgecolors="k",
    linewidth=1,
)
# standard errors
ax[0].errorbar(
    d["MedianAgeMarriage"],
    d["Divorce"],
    d["Divorce SE"].values,
    ls="none",
    color="k",
    linewidth=1,
)
ax[0].set_xlabel("Median age marriage")
ax[0].set_ylabel("Divorce rate")
ax[0].set_ylim(4, 15)

# points
ax[1].scatter(
    d["log_population"],
    d["Divorce"],
    marker="o",
    facecolor="none",
    edgecolors="k",
    linewidth=1,
)
# standard errors
ax[1].errorbar(
    d["log_population"],
    d["Divorce"],
    d["Divorce SE"].values,
    ls="none",
    color="k",
    linewidth=1,
)
ax[1].set_xlabel("log population")
ax[1].set_ylabel("Divorce rate")
ax[1].set_ylim(4, 15)

# %%
div_obs = d["Divorce"].values
div_sd = d["Divorce SE"].values
R = d["Marriage"].values
A = d["MedianAgeMarriage"].values
N = len(d)

with pm.Model() as m_14_1:
    sigma = pm.HalfCauchy("sigma", 2.5)
    a = pm.Normal("a", 0.0, 10.0)
    bA = pm.Normal("bA", 0.0, 10.0)
    bR = pm.Normal("bR", 0.0, 10.0)
    mu = a + bA * A + bR * R
    div_est = pm.Normal("div_est", mu, sigma, shape=N)
    obs = pm.Normal("div_obs", div_est, div_sd, observed=div_obs)
    # start value and additional kwarg for NUTS
    start = dict(div_est=div_obs)
    trace_14_1 = pm.sample(
        4000, tune=1000, start=start, nuts_kwargs=dict(target_accept=0.95)
    )

# %%
az.summary(trace_14_1, var_names=["div_est", "a", "bA", "bR", "sigma"], round_to=2)

# %%
div_obs = d["Divorce"].values
div_sd = d["Divorce SE"].values
mar_obs = d["Marriage"].values
mar_sd = d["Marriage SE"].values
A = d["MedianAgeMarriage"].values
N = len(d)

with pm.Model() as m_14_2:
    sigma = pm.HalfCauchy("sigma", 2.5)
    a = pm.Normal("a", 0.0, 10.0)
    bA = pm.Normal("bA", 0.0, 10.0)
    bR = pm.Normal("bR", 0.0, 10.0)
    mar_est = pm.Flat("mar_est", shape=N)
    mu = a + bA * A + bR * mar_est
    div_est = pm.Normal("div_est", mu, sigma, shape=N)
    obs1 = pm.Normal("div_obs", div_est, div_sd, observed=div_obs)
    obs2 = pm.Normal("mar_obs", mar_est, mar_sd, observed=mar_obs)
    # start value and additional kwarg for NUTS
    start = dict(div_est=div_obs, mar_est=mar_obs)
    trace_14_2 = pm.sample(
        4000, tune=1000, start=start, nuts_kwargs=dict(target_accept=0.95)
    )

# %%
az.plot_trace(trace_14_2, compact=True)

# %%
d = pd.read_csv("data/milk.csv", ";")
d.loc[:, "neocortex.prop"] = d["neocortex.perc"] / 100
d.loc[:, "logmass"] = np.log(d["mass"])

# %%
# prep data
kcal = d["kcal.per.g"].values.copy()
logmass = d["logmass"].values.copy()
# PyMC3 can handle missing value quite naturally.
neocortex = d["neocortex.prop"].values.copy()
mask = np.isfinite(neocortex)
neocortex[~mask] = -999
neocortex = np.ma.masked_values(neocortex, value=-999)

# fit model
with pm.Model() as m_14_3:
    sigma = pm.HalfCauchy("sigma", 1.0)
    sigma_N = pm.HalfCauchy("sigma_N", 1.0)
    nu = pm.Normal("nu", 0.5, 1.0)
    bN = pm.Normal("bN", 0.0, 10.0)
    bM = pm.Normal("bM", 0.0, 10.0)
    a = pm.Normal("a", 0.0, 100.0)
    neocortex_ = pm.Normal("neocortex", nu, sigma_N, observed=neocortex)
    mu = a + bN * neocortex_ + bM * logmass
    kcal_ = pm.Normal("kcal", mu, sigma, observed=kcal)
    trace_14_3 = pm.sample(5000, tune=5000)

# %%
# the missing value in pymc3 is automatically model as a node with *_missing as name
az.summary(
    trace_14_3,
    var_names=["neocortex_missing", "a", "bN", "bM", "nu", "sigma_N", "sigma"],
    round_to=2,
)

# %%
# prep data
neocortex = np.copy(d["neocortex.prop"].values)
mask = np.isfinite(neocortex)
kcal = np.copy(d["kcal.per.g"].values[mask])
logmass = np.copy(d["logmass"].values[mask])
neocortex = neocortex[mask]

# fit model
with pm.Model() as m_14_3cc:
    sigma = pm.HalfCauchy("sigma", 1.0)
    bN = pm.Normal("bN", 0.0, 10.0)
    bM = pm.Normal("bM", 0.0, 10.0)
    a = pm.Normal("a", 0.0, 100.0)
    mu = a + bN * neocortex + bM * logmass
    kcal_ = pm.Normal("kcal", mu, sigma, observed=kcal)
    trace_14_3cc = pm.sample(5000, tune=5000)

az.summary(trace_14_3cc, var_names=["a", "bN", "bM", "sigma"], round_to=2)

# %%
# prep data
kcal = d["kcal.per.g"].values.copy()
logmass = d["logmass"].values.copy()
neocortex = d["neocortex.prop"].values.copy()
mask = np.isfinite(neocortex)
neocortex[~mask] = -999
neocortex = np.ma.masked_values(neocortex, value=-999)

with pm.Model() as m_14_4:
    sigma = pm.HalfCauchy("sigma", 1.0)
    sigma_N = pm.HalfCauchy("sigma_N", 1.0)
    a_N = pm.Normal("a_N", 0.5, 1.0)
    betas = pm.Normal("bNbMgM", 0.0, 10.0, shape=3)  # bN, bM, and gM
    a = pm.Normal("a", 0.0, 100.0)

    nu = a_N + betas[2] * logmass
    neocortex_ = pm.Normal("neocortex", nu, sigma_N, observed=neocortex)

    mu = a + betas[0] * neocortex_ + betas[1] * logmass
    kcal_ = pm.Normal("kcal", mu, sigma, observed=kcal)

    trace_14_4 = pm.sample(5000, tune=5000)

az.summary(
    trace_14_4,
    var_names=["neocortex_missing", "a", "bNbMgM", "a_N", "sigma_N", "sigma"],
    round_to=2,
)
