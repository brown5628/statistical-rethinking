# %%
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import arviz as az

# %%
d = pd.read_csv("data/rugged.csv", sep=";", header=0)

# %%
d["log_gdp"] = np.log(d.rgdppc_2000)
dd = d[np.isfinite(d["rgdppc_2000"])]
dA1 = dd[dd.cont_africa == 1]
dA0 = dd[dd.cont_africa == 0]

# %%
with pm.Model() as model_7_2:
    a = pm.Normal("a", mu=8, sd=100)
    bR = pm.Normal("bR", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=10)

    mu = pm.Deterministic("mu", a + bR * dA1["rugged"])
    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=np.log(dA1["rgdppc_2000"]))
    trace_7_2 = pm.sample(1000, tune=1000)

# %%
varnames = ["~mu"]
pm.traceplot(trace_7_2, varnames)

# %%
with pm.Model() as model_7_2_2:
    a = pm.Normal("a", mu=8, sd=100)
    bR = pm.Normal("bR", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=10)

    mu = pm.Deterministic("mu", a + bR * dA0["rugged"])
    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=np.log(dA0["rgdppc_2000"]))
    trace_7_2_2 = pm.sample(1000, tune=1000)

# %%
pm.traceplot(trace_7_2_2, varnames)

# %%
mu_mean = trace_7_2["mu"]
mu_hpd = pm.hpd(mu_mean)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
ax1.plot(dA1["rugged"], np.log(dA1["rgdppc_2000"]), "C0o")
ax1.plot(dA1["rugged"], mu_mean.mean(0), "C1")
az.plot_hpd(dA1["rugged"], mu_mean, ax=ax1)
ax1.set_title("Africa")
ax1.set_ylabel("log(rgdppc_2000")
ax1.set_xlabel("rugged")

mu_mean = trace_7_2_2["mu"]

ax2.plot(dA0["rugged"], np.log(dA0["rgdppc_2000"]), "ko")
ax2.plot(dA0["rugged"], mu_mean.mean(0), "C1")
ax2.set_title("not Africa")
ax2.set_ylabel("log(rgdppc_200)")
ax2.set_xlabel("rugged")
az.plot_hpd(dA0["rugged"], mu_mean, ax=ax2)

# %%
with pm.Model() as model_7_3:
    a = pm.Normal("a", mu=8, sd=100)
    bR = pm.Normal("bR", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    mu = pm.Deterministic("mu", a + bR * dd.rugged)
    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_3 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_7_4:
    a = pm.Normal("a", mu=8, sd=1000)
    bR = pm.Normal("bR", mu=0, sd=1)
    bA = pm.Normal("bA", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    mu = pm.Deterministic("mu", a + bR * dd.rugged + bA * dd.cont_africa)
    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_4 = pm.sample(1000, tune=1000)

# %%
comp_df = az.compare({"m7.3": trace_7_3, "m7.4": trace_7_4})
comp_df

# %%
az.plot_compare(comp_df)

# %%
rugged_seq = np.arange(-1, 9, 0.25)

mu_pred_NotAfrica = np.zeros((len(rugged_seq), len(trace_7_4["bR"])))
mu_pred_Africa = np.zeros((len(rugged_seq), len(trace_7_4["bR"])))

for iSeq, seq in enumerate(rugged_seq):
    mu_pred_NotAfrica[iSeq] = (
        trace_7_4["a"] + trace_7_4["bR"] * rugged_seq[iSeq] + trace_7_4["bA"] * 0
    )
    mu_pred_Africa[iSeq] = (
        trace_7_4["a"] + trace_7_4["bR"] * rugged_seq[iSeq] + trace_7_4["bA"] * 1
    )

# %%
mu_mean_NotAfrica = mu_pred_NotAfrica.mean(1)
mu_mean_Africa = mu_pred_Africa.mean(1)

# %%
plt.plot(dA1["rugged"], np.log(dA1["rgdppc_2000"]), "C0o")
plt.plot(rugged_seq, mu_mean_Africa, "C0")
az.plot_hpd(rugged_seq, mu_pred_Africa.T, credible_interval=0.97, color="C0")
plt.plot(dA0["rugged"], np.log(dA0["rgdppc_2000"]), "ko")
az.plot_hpd(rugged_seq, mu_pred_NotAfrica.T, credible_interval=0.97, color="k")
plt.annotate("not Africa", xy=(6, 9.5))
plt.annotate("Africa", xy=(6, 6))
plt.ylabel("log(rgdppc_2000)")
plt.xlabel("rugged")

# %%
with pm.Model() as model_7_5:
    a = pm.Normal("a", mu=8, sd=100)
    bR = pm.Normal("bR", mu=0, sd=1)
    bA = pm.Normal("bA", mu=0, sd=1)
    bAR = pm.Normal("bAR", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    gamma = bR + bAR * dd.cont_africa
    mu = pm.Deterministic("mu", a + gamma * dd.rugged + bA * dd.cont_africa)
    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_5 = pm.sample(1000, tune=1000)

# %%
comp_df = az.compare({"m7.3": trace_7_3, "m7.4": trace_7_4, "m7.5": trace_7_5})

comp_df

# %%
az.plot_compare(comp_df)

# %%
with pm.Model() as model_7_5_b:
    a = pm.Normal("a", mu=8, sd=100)
    bR = pm.Normal("bR", mu=0, sd=1)
    bA = pm.Normal("bA", mu=0, sd=1)
    bAR = pm.Normal("bAR", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    mu = pm.Deterministic(
        "mu",
        a + bR * dd.rugged + bAR * dd.rugged * dd.cont_africa + bA * dd.cont_africa,
    )
    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_5b = pm.sample(1000, tune=1000)

# %%
rugged_seq = np.arange(-1, 9, 0.25)

# compute mu over samples
mu_pred_NotAfrica = np.zeros((len(rugged_seq), len(trace_7_5b["bR"])))
mu_pred_Africa = np.zeros((len(rugged_seq), len(trace_7_5b["bR"])))
for iSeq, seq in enumerate(rugged_seq):
    mu_pred_NotAfrica[iSeq] = (
        trace_7_5b["a"]
        + trace_7_5b["bR"] * rugged_seq[iSeq]
        + trace_7_5b["bAR"] * rugged_seq[iSeq] * 0
        + trace_7_5b["bA"] * 0
    )
    mu_pred_Africa[iSeq] = (
        trace_7_5b["a"]
        + trace_7_5b["bR"] * rugged_seq[iSeq]
        + trace_7_5b["bAR"] * rugged_seq[iSeq] * 1
        + trace_7_5b["bA"] * 1
    )

mu_mean_NotAfrica = mu_pred_NotAfrica.mean(1)
mu_mean_Africa = mu_pred_Africa.mean(1)

# %%
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
ax1.plot(dA1["rugged"], np.log(dA1["rgdppc_2000"]), "C0o")
ax1.plot(rugged_seq, mu_mean_Africa, "C0")
az.plot_hpd(rugged_seq, mu_pred_Africa.T, credible_interval=0.97, color="C0", ax=ax1)

ax1.set_title("African Nations")
ax1.set_ylabel("log GDP year 2000", fontsize=14)
ax1.set_xlabel("Terrain Ruggedness Index", fontsize=14)

ax2.plot(dA0["rugged"], np.log(dA0["rgdppc_2000"]), "ko")
ax2.plot(rugged_seq, mu_mean_NotAfrica, "k")
az.plot_hpd(rugged_seq, mu_pred_NotAfrica.T, credible_interval=0.97, color="C1", ax=ax2)
ax2.set_title("Non-African Nations")
ax2.set_ylabel("log GDP year 2000", fontsize=14)
ax2.set_xlabel("Terrain Ruggedness Index", fontsize=14)

# %%
varnames = ["~mu"]
az.summary(trace_7_5b, varnames, credible_interval=0.89).round(3)

# %%
gamma_Africa = trace_7_5b["bR"] + trace_7_5b["bAR"] * 1
gamma_notAfrica = trace_7_5b["bR"]

# %%
print("Gamma within Africa: {:.2f}".format(gamma_Africa.mean()))
print("Gamma outside Africa: {:.2f}".format(gamma_notAfrica.mean()))

# %%
_, ax = plt.subplots()
ax.set_xlabel("gamma")
ax.set_ylabel("Density")
ax.set_ylim(top=5.25)
az.plot_kde(gamma_Africa)
az.plot_kde(gamma_notAfrica, plot_kwargs={"color": "k"})

# %%
diff = gamma_Africa - gamma_notAfrica
az.plot_kde(diff)
plt.hist(diff, bins=len(diff))

# %%
sum(diff[diff < 0]) / len(diff)

# %%
q_rugged = [0, 0]
q_rugged[0] = np.min(dd.rugged)
q_rugged[1] = np.max(dd.rugged)

# %%
mu_ruggedlo = np.zeros((2, len(trace_7_5b["bR"])))
mu_ruggedhi = np.zeros((2, len(trace_7_5b["bR"])))

for iAfri in range(0, 2):
    mu_ruggedlo[iAfri] = (
        trace_7_5b["a"]
        + trace_7_5b["bR"] * q_rugged[0]
        + trace_7_5b["bAR"] * q_rugged[0] * iAfri
        + trace_7_5b["bA"] * iAfri
    )
    mu_ruggedhi[iAfri] = (
        trace_7_5b["a"]
        + trace_7_5b["bR"] * q_rugged[1]
        + trace_7_5b["bAR"] * q_rugged[1] * iAfri
        + trace_7_5b["bA"] * iAfri
    )

# %%
mu_ruggedlo_mean = np.mean(mu_ruggedlo, axis=1)
mu_hpd_ruggedlo = pm.hpd(mu_ruggedlo.T, 0.03)
mu_ruggedhi_mean = np.mean(mu_ruggedhi, axis=1)
mu_hpd_ruggedhi = pm.hpd(mu_ruggedhi.T, 0.03)

# %%


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))  # outward by 5 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color("none")  # don't draw spine

    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


# %%

med_r = np.median(dd.rugged)

ox = [0.05 if x > med_r else -0.05 for x in dd.rugged]
idxk = [i for i, x in enumerate(ox) if x == -0.05]
idxb = [i for i, x in enumerate(ox) if x == 0.05]
cont_africa_ox = dd.cont_africa + ox
plt.plot(
    cont_africa_ox[dd.cont_africa.index[idxk]],
    np.log(dd.rgdppc_2000[dd.cont_africa.index[idxk]]),
    "ko",
)
plt.plot(
    cont_africa_ox[dd.cont_africa.index[idxb]],
    np.log(dd.rgdppc_2000[dd.cont_africa.index[idxb]]),
    "C0o",
)
plt.plot([0, 1], mu_ruggedlo_mean, "k--")
plt.plot([0, 1], mu_ruggedhi_mean, "C0")
plt.fill_between(
    [0, 1], mu_hpd_ruggedlo[:, 0], mu_hpd_ruggedlo[:, 1], color="k", alpha=0.2
)
plt.fill_between(
    [0, 1], mu_hpd_ruggedhi[:, 0], mu_hpd_ruggedhi[:, 1], color="b", alpha=0.2
)
plt.ylabel("log GDP year 2000", fontsize=14)
plt.xlabel("Continent", fontsize=14)
axes = plt.gca()
axes.set_xlim([-0.25, 1.25])
axes.set_ylim([5.8, 11.2])
axes.set_xticks([0, 1])
axes.set_xticklabels(["other", "Africa"], fontsize=12)
axes.set_facecolor("white")
adjust_spines(axes, ["left", "bottom"])
axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
axes.spines["bottom"].set_linewidth(0.5)
axes.spines["left"].set_linewidth(0.5)
axes.spines["bottom"].set_color("black")
axes.spines["left"].set_color("black")
plt.show()

# %%
d = pd.read_csv("data/tulips.csv", sep=";", header=0)

# %%
d.info()


# %%
d.head()

# %%
d.describe()

# %%
with pm.Model() as model_7_6:
    a = pm.Normal("a", mu=0, sd=100)
    bW = pm.Normal("bW", mu=100, sd=100)
    bS = pm.Normal("bS", mu=0, sd=100)
    sigma = pm.Uniform("sigma", lower=0, upper=100)
    mu = pm.Deterministic("mu", a + bW * d.water + bS * d.shade)
    blooms = pm.Normal("blooms", mu, sigma, observed=d.blooms)
    trace_7_6 = pm.sample(1000, tune=1000)

# %%
with pm.Model() as model_7_7:
    a = pm.Normal("a", mu=0, sd=100)
    bW = pm.Normal("bW", mu=0, sd=100)
    bS = pm.Normal("bS", mu=0, sd=100)
    bWS = pm.Normal("bWS", mu=0, sd=100)
    sigma = pm.Uniform("sigma", lower=0, upper=100)
    mu = pm.Deterministic(
        "mu", a + bW * d.water + bS * d.shade + bWS * d.water * d.shade
    )
    blooms = pm.Normal("blooms", mu, sigma, observed=d.blooms)
    trace_7_7 = pm.sample(1000, tune=1000)

# %%
map_7_6 = pm.find_MAP(model=model_7_6)
map_7_6

# %%
map_7_7 = pm.find_MAP(model=model_7_7)
model_7_7

# %%
map_7_6 = pm.find_MAP(model=model_7_6, method="Powell")
map_7_6

# %%
map_7_7 = pm.find_MAP(model=model_7_7, method="Powell")
map_7_7

# %%
az.summary(trace_7_6, var_names=["~mu"])["mean"]

# %%
az.summary(trace_7_7, var_names=["~mu"])["mean"]

# %%
comp_df = az.compare({"m7.6": trace_7_6, "m7.7": trace_7_7})
comp_df

# %%
d["shade_c"] = d.shade - np.mean(d.shade)
d["water_c"] = d.water - np.mean(d.water)

# %%
with pm.Model() as model_7_8:
    a = pm.Normal("a", mu=0, sd=100)
    bW = pm.Normal("bW", mu=0, sd=100)
    bS = pm.Normal("bS", mu=0, sd=100)
    sigma = pm.Uniform("sigma", lower=0, upper=100)
    mu = pm.Deterministic("mu", a + bW * d.water_c + bS * d.shade_c)
    blooms = pm.Normal("blooms", mu, sigma, observed=d.blooms)
    trace_7_8 = pm.sample(1000, tune=1000)
    start = {"a": np.mean(d.blooms), "bW": 0, "bS": 0, "sigma": np.std(d.blooms)}

# %%
map_7_8 = pm.find_MAP(model=model_7_8)
map_7_8

# %%
with pm.Model() as model_7_9:
    a = pm.Normal("a", mu=0, sd=100)
    bW = pm.Normal("bW", mu=0, sd=100)
    bS = pm.Normal("bS", mu=0, sd=100)
    bWS = pm.Normal("bWS", mu=0, sd=100)
    sigma = pm.Uniform("sigma", lower=0, upper=100)
    mu = pm.Deterministic(
        "mu", a + bW * d.water_c + bS * d.shade_c + bWS * d.water_c * d.shade_c
    )
    blooms = pm.Normal("blooms", mu, sigma, observed=d.blooms)
    trace_7_9 = pm.sample(1000, tune=1000)
    start = {
        "a": np.mean(d.blooms),
        "bW": 0,
        "bS": 0,
        "bWS": 0,
        "sigma": np.std(d.blooms),
    }

# %%
map_7_9 = pm.find_MAP(model=model_7_9)
map_7_9

# %%
map_7_7["a"] + map_7_7["bW"] * 2 + map_7_7["bS"] * 2 + map_7_7["bWS"] * 2 * 2

# %%
map_7_9["a"] + map_7_9["bW"] * 0 + map_7_9["bS"] * 0 + map_7_9["bWS"] * 0 * 0

# %%
varnames = ["a", "bW", "bS", "bWS", "sigma"]
az.summary(trace_7_9, varnames, credible_interval=0.89).round(3)

# %%
# No interaction
f, axs = plt.subplots(1, 3, sharey=True, figsize=(8, 3))
# Loop over values of water_c and plot predictions.
shade_seq = range(-1, 2, 1)

mu_w = np.zeros((len(shade_seq), len(trace_7_8["a"])))
for ax, w in zip(axs.flat, range(-1, 2, 1)):
    dt = d[d.water_c == w]
    ax.plot(dt.shade - np.mean(dt.shade), dt.blooms, "C0o")
    for x, iSeq in enumerate(shade_seq):
        mu_w[x] = trace_7_8["a"] + trace_7_8["bW"] * w + trace_7_8["bS"] * iSeq
    mu_mean_w = mu_w.mean(1)
    mu_hpd_w = pm.hpd(
        mu_w.T, credible_interval=0.03
    )  # 97% probability interval: 1-.97 = 0.03
    ax.plot(shade_seq, mu_mean_w, "k")
    ax.plot(shade_seq, mu_hpd_w.T[0], "k--")
    ax.plot(shade_seq, mu_hpd_w.T[1], "k--")
    ax.set_ylim(0, 362)
    ax.set_ylabel("blooms")
    ax.set_xlabel("shade (centerd)")
    ax.set_title("water_c = {:d}".format(w))
    ax.set_xticks(shade_seq)
    ax.set_yticks(range(0, 301, 100))

# Interaction
f, axs = plt.subplots(1, 3, sharey=True, figsize=(8, 3))
# Loop over values of water_c and plot predictions.
shade_seq = range(-1, 2, 1)

mu_w = np.zeros((len(shade_seq), len(trace_7_9["a"])))
for ax, w in zip(axs.flat, range(-1, 2, 1)):
    dt = d[d.water_c == w]
    ax.plot(dt.shade - np.mean(dt.shade), dt.blooms, "C0o")
    for x, iSeq in enumerate(shade_seq):
        mu_w[x] = (
            trace_7_9["a"]
            + trace_7_9["bW"] * w
            + trace_7_9["bS"] * iSeq
            + trace_7_9["bWS"] * w * iSeq
        )
    mu_mean_w = mu_w.mean(1)
    mu_hpd_w = az.hpd(
        mu_w.T, credible_interval=0.97
    )  # 97% probability interval: 1-.97 = 0.03
    ax.plot(shade_seq, mu_mean_w, "k")
    ax.plot(shade_seq, mu_hpd_w.T[0], "k--")
    ax.plot(shade_seq, mu_hpd_w.T[1], "k--")
    ax.set_ylim(0, 362)
    ax.set_ylabel("blooms")
    ax.set_xlabel("shade (centered)")
    ax.set_title("water_c = {:d}".format(w))
    ax.set_xticks(shade_seq)
    ax.set_yticks(range(0, 301, 100))

# %%
m_7_x = smf.ols("blooms ~ shade + water + shade * water", data=d).fit()

# %%
m_7_x = smf.ols("blooms ~ shade * water", data=d).fit()

# %%
m_7_x = smf.ols("blooms ~ shade * water - water", data=d).fit()

# %%
m_7_x = smf.ols("blooms ~ shade * water * bed", data=d).fit()

# %%
