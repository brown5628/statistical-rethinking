# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from scipy import stats
from scipy.special import expit as logistic

# %%
d = pd.read_csv("data/reedfrogs.csv", sep=",")
print(d.shape)
d.head(8)

# %%
tank = np.arange(d.shape[0])

with pm.Model() as m_12_1:
    a_tank = pm.Normal("a_tank", 0, 5, shape=d.shape[0])
    p = pm.math.invlogit(a_tank[tank])
    surv = pm.Binomial("surv", n=d.density, p=p, observed=d.surv)
    trace_12_1 = pm.sample(2000, tune=2000)

# %%
with pm.Model() as m_12_2:
    a = pm.Normal("a", 0.0, 1.0)
    sigma = pm.HalfCauchy("sigma", 1.0)
    a_tank = pm.Normal("a_tank", a, sigma, shape=d.shape[0])
    p = pm.math.invlogit(a_tank[tank])
    surv = pm.Binomial("surv", n=d.density, p=p, observed=d.surv)
    trace_12_2 = pm.sample(10000, tune=10000)

# %%
comp_df = az.compare({"m12_1": trace_12_1, "m12_2": trace_12_2})
comp_df

# %%
post = pm.trace_to_dataframe(trace_12_2, varnames=["a_tank"])

d.loc[:, "propsurv_est"] = pd.Series(
    logistic(post.median(axis=0).values), index=d.index
)

# %%
_, ax = plt.subplots(1, 1, figsize=(12, 5))
# display raw proportions surviving in each tank
ax.scatter(np.arange(1, 49), d.propsurv)
ax.scatter(np.arange(1, 49), d.propsurv_est, facecolors="none", edgecolors="k", lw=1)
ax.hlines(logistic(np.median(trace_12_2["a"], axis=0)), 0, 49, linestyles="--")
ax.vlines([16.5, 32.5], -0.05, 1.05, lw=0.5)
ax.text(8, 0, "small tanks", horizontalalignment="center")
ax.text(16 + 8, 0, "medium tanks", horizontalalignment="center")
ax.text(32 + 8, 0, "large tanks", horizontalalignment="center")
ax.set_xlabel("tank", fontsize=14)
ax.set_ylabel("proportion survival", fontsize=14)
ax.set_xlim(-1, 50)
ax.set_ylim(-0.05, 1.05)

# %%
_, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
# show first 100 populations in the posterior
xrange = np.linspace(-3, 4, 200)
postcurve = [
    stats.norm.pdf(xrange, loc=trace_12_2["a"][i], scale=trace_12_2["sigma"][i])
    for i in range(100)
]
ax[0].plot(xrange, np.asarray(postcurve).T, alpha=0.1, color="k")
ax[0].set_xlabel("log-odds survive")
ax[0].set_ylabel("Density")
# sample 8000 imaginary tanks from the posterior distribution
sim_tanks = np.random.normal(loc=trace_12_2["a"], scale=trace_12_2["sigma"])
# transform to probability and visualize
az.plot_kde(logistic(sim_tanks), ax=ax[1], plot_kwargs={"color": "k"})
ax[1].set_xlabel("probability survive")
ax[1].set_ylabel("Density")

# %%
a, sigma, nponds = 1.4, 1.5, 60
ni = np.repeat([5, 10, 25, 35], 15)

# %%
a_pond = np.random.normal(loc=a, scale=sigma, size=nponds)

# %%
dsim = pd.DataFrame(dict(pond=np.arange(nponds), ni=ni, true_a=a_pond))

# %%
dsim.loc[:, "si"] = np.random.binomial(dsim["ni"], logistic(dsim["true_a"]))

# %%
dsim.loc[:, "p_nopool"] = dsim.si / dsim.ni

# %%
with pm.Model() as m_12_3:
    a = pm.Normal("a", 0.0, 1.0)
    sigma = pm.HalfCauchy("sigma", 1.0)
    a_pond = pm.Normal("a_pond", a, sigma, shape=nponds)
    p = pm.math.invlogit(a_pond[dsim.pond.values])
    si = pm.Binomial("si", n=dsim.ni.values, p=p, observed=dsim.si)
    trace_12_3 = pm.sample(1000, tune=1000)

# %%
summary_12_3 = az.summary(trace_12_3, credible_interval=0.89, round_to=2)
summary_12_3.iloc[[0, 1, 2, -3, -2, -1], :]

# %%
estimated_a_pond = az.summary(trace_12_3, credible_interval=0.89, var_names=["a_pond"])
dsim.loc[:, "p_partpool"] = logistic(estimated_a_pond["mean"].values)

# %%
dsim.loc[:, "p_true"] = logistic(dsim["true_a"].values)

# %%
nopool_error = np.abs(dsim.p_nopool - dsim.p_true)
partpool_error = np.abs(dsim.p_partpool - dsim.p_true)

# %%
_, ax = plt.subplots(1, 1, figsize=(12, 5))
xrange = np.arange(60)
xrange_ = xrange.reshape((4, 15))
# display raw proportions surviving in each tank
ax.scatter(xrange + 1, nopool_error)
ax.scatter(xrange + 1, partpool_error, facecolors="none", edgecolors="k", lw=1)
ax.vlines(xrange_[1:, 0] + 0.5, -0.025, 0.35, lw=0.5)

textall = ["tiny (5)", "small (10)", "medium (25)", "large (30)"]
for isem in range(4):
    ax.hlines(
        nopool_error[xrange_[isem, :]].mean(),
        xrange_[isem, 0] + 1,
        xrange_[isem, -1] + 1,
        color="C0",
    )
    ax.hlines(
        partpool_error[xrange_[isem, :]].mean(),
        xrange_[isem, 0] + 1,
        xrange_[isem, -1] + 1,
        color="k",
        linestyles="--",
    )
    ax.text(xrange_[isem, 7] + 0.5, 0.32, textall[isem], horizontalalignment="center")
ax.set_xlabel("pond")
ax.set_ylabel("absolute error")
ax.set_xlim(-1, 62)
ax.set_ylim(-0.025, 0.34)

# %%
y1 = np.random.normal(10.0, 1.0, 10000)
y2 = 10.0 + np.random.normal(0.0, 1.0, 10000)
print(y1)
print(y2)

# %%
d = pd.read_csv("data/chimpanzees.csv", sep=";")
d.actor = (d.actor - 1).astype(int)
Nactor = len(d.actor.unique())

with pm.Model() as m_12_4:
    bp = pm.Normal("bp", 0, 10)
    bpC = pm.Normal("bpC", 0, 10)

    a = pm.Normal("a", 0, 10)
    sigma_actor = pm.HalfCauchy("sigma_actor", 1.0)
    a_actor = pm.Normal("a_actor", 0.0, sigma_actor, shape=Nactor)

    p = pm.math.invlogit(
        a + a_actor[d.actor.values] + (bp + bpC * d.condition) * d.prosoc_left
    )
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    trace_12_4 = pm.sample(5000, tune=5000)

# %%
total_a_actor = np.asarray(
    [trace_12_4["a"] + trace_12_4["a_actor"][:, i] for i in range(Nactor)]
)
total_a_actor.mean(axis=1).round(2)

# %%
d.block = (d.block - 1).astype(int)
Nblock = len(d.block.unique())

with pm.Model() as m_12_5:
    bp = pm.Normal("bp", 0, 10)
    bpC = pm.Normal("bpC", 0, 10)

    a = pm.Normal("a", 0, 10)
    sigma_actor = pm.HalfCauchy("sigma_actor", 1.0)
    a_actor = pm.Normal("a_actor", 0.0, sigma_actor, shape=Nactor)

    sigma_block = pm.HalfCauchy("sigma_block", 1.0)
    a_block = pm.Normal("a_block", 0.0, sigma_block, shape=Nblock)

    p = pm.math.invlogit(
        a
        + a_actor[d.actor.values]
        + a_block[d.block.values]
        + (bp + bpC * d.condition) * d.prosoc_left
    )
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    trace_12_5 = pm.sample(6000, tune=6000)

# %%
az.summary(trace_12_5, credible_interval=0.89, round_to=2)

# %%
az.plot_forest(trace_12_5)

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 5))
az.plot_kde(trace_12_5["sigma_actor"], ax=ax)
az.plot_kde(trace_12_5["sigma_block"], ax=ax, plot_kwargs={"color": "C1"})
ax.text(2, 0.75, "actor", color="C0")
ax.text(0.5, 2, "block", color="C1")
ax.set_xlabel("sigma")
ax.set_ylabel("density")
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-0.05, 3.05)

# %%
comp_df = az.compare({"m12_4": trace_12_4, "m12_5": trace_12_5})

comp_df

# %%
chimp = 2 - 1
sel_actor = np.where((d.actor == (chimp)).values)[0]
_, uni_cond = np.unique(
    d.loc[sel_actor, ["condition", "prosoc_left"]], axis=0, return_index=True,
)
d.loc[sel_actor[uni_cond], :].head()

# %%
ppc = pm.sample_posterior_predictive(trace=trace_12_4, samples=1000, model=m_12_4)

# %%
rt = ppc["pulled_left"][:, sel_actor]

pred_mean = np.zeros((1000, 4))
cond = d.condition.unique()
prosoc_l = d.prosoc_left.unique()
for i in range(len(rt)):
    tmp = []
    for cp in cond:
        for pl in prosoc_l:
            tmp.append(
                np.mean(
                    rt[i][
                        (d.prosoc_left[sel_actor].values == pl)
                        & (d.condition[sel_actor].values == cp)
                    ]
                )
            )
    pred_mean[i] = tmp

mp = pred_mean.mean(0)

_, ax = plt.subplots(1, 1, figsize=(5, 5))
az.plot_hpd(range(4), pred_mean, credible_interval=0.89, color="k")

ax.plot(mp, color="k")
chimps = (
    d.groupby(["actor", "prosoc_left", "condition"])
    .agg("mean")["pulled_left"]
    .values.reshape(7, -1)
)
ax.plot(chimps[chimp], "C0")

ax.set_ylim(0, 1.1)
ax.set_xlabel("prosoc_left/condition")
ax.set_ylabel("proportion pulled left")
plt.xticks(range(4), ("0/0", "1/0", "0/1", "1/1"))

# %%
post = pm.trace_to_dataframe(trace_12_4)
post.head()

# %%
az.plot_kde(trace_12_4["a_actor"][:, 5 - 1])

# %%


def p_link(prosoc_left, condition, actor, trace):
    logodds = (
        trace["a"]
        + trace["a_actor"][:, actor]
        + (trace["bp"] + trace["bpC"] * condition) * prosoc_left
    )
    return logistic(logodds)


# %%
prosoc_left = [0, 1, 0, 1]
condition = [0, 0, 1, 1]

pred_raw = np.asarray(
    [p_link(p_l, c_d, 2 - 1, trace_12_4) for p_l, c_d in zip(prosoc_left, condition)]
).T
pred_p = pred_raw.mean(axis=0)
pred_p_PI = az.hpd(pred_raw, credible_interval=0.89)

# %%
d_pred = pd.DataFrame(
    dict(prosoc_left=[0, 1, 0, 1], condition=[0, 0, 1, 1], actor=np.repeat(2 - 1, 4))
)

# %%
a_actor_zeros = np.zeros((1000, 7))

# %%


def p_link(prosoc_left, condition, actor_sim, trace):
    Nsim = actor_sim.shape[0] // trace.nchains
    trace = trace[:Nsim]
    logodds = (
        trace["a"]
        + np.mean(actor_sim, axis=1)
        + (trace["bp"] + trace["bpC"] * condition) * prosoc_left
    )
    return logistic(logodds)


pred_raw = np.asarray(
    [
        p_link(p_l, c_d, a_actor_zeros, trace_12_4)
        for p_l, c_d in zip(prosoc_left, condition)
    ]
).T
pred_p = pred_raw.mean(axis=0)
_, ax = plt.subplots(1, 1, figsize=(5, 5))

az.plot_hpd(range(4), pred_raw, credible_interval=0.89, color="k", smooth=False)
ax.plot(pred_p, color="k")

ax.set_ylim(0, 1.1)
ax.set_xlabel("prosoc_left/condition")
ax.set_ylabel("proportion pulled left")
plt.xticks(range(4), ("0/0", "1/0", "0/1", "1/1"))

# %%
sigma_actor = trace_12_4.get_values("sigma_actor")
a_actor_sims = np.random.normal(loc=0, scale=np.reshape(sigma_actor[:7000], (1000, 7)))

# %%
pred_raw = np.asarray(
    [
        p_link(p_l, c_d, a_actor_sims, trace_12_4)
        for p_l, c_d in zip(prosoc_left, condition)
    ]
).T
pred_p = pred_raw.mean(axis=0)

_, ax = plt.subplots(1, 1, figsize=(5, 5))
az.plot_hpd(range(4), pred_raw, credible_interval=0.89, color="k", smooth=False)
ax.plot(pred_p, color="k")

ax.set_ylim(0, 1.1)
ax.set_xlabel("prosoc_left/condition")
ax.set_ylabel("proportion pulled left")
plt.xticks(range(4), ("0/0", "1/0", "0/1", "1/1"))

# %%


def sim_actor(tr, i):
    sim_a_actor = np.random.randn() * tr["sigma_actor"][i]
    P = np.array([0, 1, 0, 1])
    C = np.array([0, 0, 1, 1])
    p = logistic(tr["a"][i] + sim_a_actor + (tr["bp"][i] + tr["bpC"][i] * C) * P)
    return p


sim_actor(trace_12_4, 0)

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 5))
for i in range(50):
    ax.plot(sim_actor(trace_12_4, i), color="k", alpha=0.15)

ax.set_ylim(0, 1.1)
ax.set_xticks(range(4), ("0/0", "1/0", "0/1", "1/1"))
ax.set_xlabel("prosoc_left/condition")
ax.set_ylabel("proportion pulled left")
plt.xticks(range(4), ("0/0", "1/0", "0/1", "1/1"))

# %%
dk = pd.read_csv("data/Kline.csv", sep=";")
dk.loc[:, "log_pop"] = np.log(dk.population)
Nsociety = dk.shape[0]
dk.loc[:, "society"] = np.arange(Nsociety)

with pm.Model() as m_12_6:
    sigma_society = pm.HalfCauchy("sigma_society", 1)
    a_society = (pm.Normal("a_society", 0, sigma_society, shape=Nsociety),)
    a = pm.Normal("a", 0, 10)
    bp = pm.Normal("bp", 0, 1)
    lam = pm.math.exp(a + a_society + bp * dk.log_pop)
    obs = pm.Poisson("total_tools", lam, observed=dk.total_tools)
    trace_12_6 = pm.sample(5000, tune=1000)

# %%
sigma_society = trace_12_6.get_values("sigma_society", combine=True)[:, None]
a_society_sims = np.random.normal(loc=0, scale=sigma_society)
log_pop_seq = np.linspace(6, 14, 30)
a_post = trace_12_6.get_values(varname="a", combine=True)[:, None]
bp_post = trace_12_6.get_values(varname="bp", combine=True)[:, None]
link_m12_6 = np.exp(a_post + a_society_sims + bp_post * log_pop_seq)

# %%
_, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.scatter(dk.log_pop, dk.total_tools)

axes.plot(log_pop_seq, np.median(link_m12_6, axis=0), "--", color="k")

for alpha in [0.67, 0.89, 0.97]:
    mu_hpd = pm.hpd(link_m12_6, alpha=alpha)
    az.plot_hpd(
        log_pop_seq,
        link_m12_6,
        credible_interval=alpha,
        color="k",
        fill_kwargs={"alpha": alpha * 0.25},
    )

axes.set_xlabel("log-population", fontsize=14)
axes.set_ylabel("total tools", fontsize=14)
axes.set_xlim(6.8, 12.8)
axes.set_ylim(10, 73)
