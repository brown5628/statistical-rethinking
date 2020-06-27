# %%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import seaborn as sns
import arviz as az

# %%
seed = 1234567890
np.random.seed(seed)

# %%
num_weeks = int(1e5)
positions = np.empty(num_weeks, dtype=np.int64)
current = 9

for i in range(num_weeks):
    positions[i] = current

    proposal = current + np.random.choice([-1, 1])
    proposal %= 10

    prob_move = (proposal + 1) / (current + 1)
    current = proposal if np.random.uniform() < prob_move else current

# %%
_, (week_ax, island_ax) = plt.subplots(ncols=2, figsize=(16, 6))

week_ax.scatter(np.arange(100) + 1, positions[:100] + 1)

week_ax.set_xlabel("week")
week_ax.set_ylim(0, 11)
week_ax.set_ylabel("island")

island_ax.bar(np.arange(10) + 0.6, np.bincount(positions))

island_ax.set_xlim(0.4, 10.6)
island_ax.set_xlabel("island")
island_ax.set_ylabel("number of weeks")

# %%
rugged_df = (
    pd.read_csv("data/rugged.csv", sep=";")
    .assign(log_gdp=lambda df: np.log(df.rgdppc_2000))
    .dropna(subset=["log_gdp"])
)

# %%
with pm.Model() as m8_1_map:
    a = pm.Normal("a", 0.0, 100.0)
    bR = pm.Normal("bR", 0.0, 10.0)
    bA = pm.Normal("bA", 0.0, 10.0)
    bAR = pm.Normal("bAR", 0.0, 10.0)
    mu = (
        a
        + bR * rugged_df.rugged
        + bA * rugged_df.cont_africa
        + bAR * rugged_df.rugged * rugged_df.cont_africa
    )

    sigma = pm.Uniform("sigma", 0.0, 10.0)

    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=rugged_df.log_gdp)

# %%
with m8_1_map:
    map_8_1 = pm.find_MAP()

# %%
map_8_1

# %%
with pm.Model() as m8_1:
    a = pm.Normal("a", 0.0, 100.0)
    bR = pm.Normal("bR", 0.0, 10.0)
    bA = pm.Normal("bA", 0.0, 10.0)
    bAR = pm.Normal("bAR", 0.0, 10.0)
    mu = (
        a
        + bR * rugged_df.rugged
        + bA * rugged_df.cont_africa
        + bAR * rugged_df.rugged * rugged_df.cont_africa
    )

    sigma = pm.HalfCauchy("sigma", 2.0)

    log_gdp = pm.Normal("log_gdp", mu, sigma, observed=rugged_df.log_gdp)

# %%
with m8_1:
    trace_8_1 = pm.sample(1000, tune=1000)

# %%
pm.summary(trace_8_1).round(2)

# %%
with m8_1:
    trace_8_1_4_chains = pm.sample(1000, tune=1000)

# %%
az.summary(trace_8_1_4_chains, credible_interval=0.89).round(2)

# %%
trace_8_1_df = pm.trace_to_dataframe(trace_8_1)
trace_8_1_df.head()

# %%
def plot_corr(x, y, **kwargs):
    corrcoef = np.corrcoef(x, y)[0, 1]

    artist = AnchoredText("{:.2f}".format(corrcoef), loc=10)
    plt.gca().add_artist(artist)
    plt.grid(b=False)


trace_8_1_df = pm.trace_to_dataframe(trace_8_1_4_chains)
grid = (
    sns.PairGrid(
        trace_8_1_df,
        x_vars=["a", "bR", "bA", "bAR", "sigma"],
        y_vars=["a", "bR", "bA", "bAR", "sigma"],
        diag_sharey=False,
    )
    .map_diag(sns.kdeplot)
    .map_upper(plt.scatter, alpha=0.1)
    .map_lower(plot_corr)
)

# %%
m8_1.logp({varname: trace_8_1[varname].mean() for varname in trace_8_1.varnames})

# %%
az.waic(trace_8_1)

# %%
az.plot_trace(trace_8_1)

# %%
y = np.array([-1.0, 1.0])

with pm.Model() as m8_2:
    alpha = pm.Flat("alpha")
    sigma = pm.Bound(pm.Flat, lower=0.0)("sigma")

    y_obs = pm.Normal("y_obs", alpha, sigma, observed=y)

# %%
with m8_2:
    trace_8_2 = pm.sample(draws=2000, tune=2000)

# %%
az.plot_trace(trace_8_2)

# %%
az.summary(trace_8_2, credible_interval=0.89).round(2)

# %%
with pm.Model() as m8_3:
    alpha = pm.Normal("alpha", 1.0, 10.0)
    sigma = pm.HalfCauchy("sigma", 1.0)

    y_obs = pm.Normal("y_obs", alpha, sigma, observed=y)

# %%
with m8_3:
    trace_8_3 = pm.sample(1000, tune=1000)

# %%
az.summary(trace_8_3, credible_interval=0.89).round(2)

# %%
az.plot_trace(trace_8_3)

# %%
y = sp.stats.cauchy.rvs(0.0, 5.0, size=int(1e4))
mu = y.cumsum() / (1 + np.arange(int(1e4)))

# %%
plt.plot(mu)

# %%
y = np.random.normal(0.0, 1.0, size=100)

# %%
with pm.Model() as m8_4:
    a1 = pm.Flat("a1")
    a2 = pm.Flat("a2")
    sigma = pm.HalfCauchy("sigma", 1.0)

    y_obs = pm.Normal("y_obs", a1 + a2, sigma, observed=y)

# %%
with m8_4:
    trace_8_4 = pm.sample(1000, tune=1000)

# %%
az.summary(trace_8_4, credible_interval=0.89).round(2)

# %%
az.plot_trace(trace_8_4)

# %%
with pm.Model() as m8_5:
    a1 = pm.Normal("a1", 0.0, 10.0)
    a2 = pm.Normal("a2", 0.0, 10.0)
    sigma = pm.HalfCauchy("sigma", 1.0)

    y_obs = pm.Normal("y_obs", a1 + a2, sigma, observed=y)

# %%
with m8_5:
    trace_8_5 = pm.sample(1000, tune=1000)

# %%
az.summary(trace_8_5, credible_interval=0.89).round(2)

# %%
az.plot_trace(trace_8_5)
