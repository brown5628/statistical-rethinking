# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
from collections import OrderedDict
from theano import shared

# %%
trolley_df = pd.read_csv(
    "/home/brown5628/projects/statistical-rethinking/data/Trolley.csv", sep=";"
)
trolley_df.head()

# %%
ax = trolley_df.response.value_counts().sort_index().plot(kind="bar")

ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)

# %%
ax = (
    trolley_df.response.value_counts()
    .sort_index()
    .cumsum()
    .div(trolley_df.shape[0])
    .plot(marker="o")
)

ax.set_xlim(0.9, 7.1)
ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("cumulative proportion", fontsize=14)

# %%
resp_lco = (
    trolley_df.response.value_counts()
    .sort_index()
    .cumsum()
    .iloc[:-1]
    .div(trolley_df.shape[0])
    .apply(lambda p: np.log(p / (1.0 - p)))
)

# %%
ax = resp_lco.plot(marker="o")

ax.set_xlim(0.9, 7)
ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("log-cumulative-odds", fontsize=14)

# %%
with pm.Model() as m11_1:
    a = pm.Normal(
        "a",
        0.0,
        10.0,
        transform=pm.distributions.transforms.ordered,
        shape=6,
        testval=np.arange(6) - 2.5,
    )

    resp_obs = pm.OrderedLogistic(
        "resp_obs", 0.0, a, observed=trolley_df.response.values - 1
    )

# %%
with m11_1:
    map_11_1 = pm.find_MAP()

# %%
map_11_1["a"]

# %%
sp.special.expit(map_11_1["a"])

# %%
with m11_1:
    trace_11_1 = pm.sample(1000, tune=1000)

# %%


def ordered_logistic_proba(a):
    pa = sp.special.expit(a)
    p_cum = np.concatenate(([0.0], pa, [1.0]))

    return p_cum[1:] - p_cum[:-1]


# %%
ordered_logistic_proba(trace_11_1["a"].mean(axis=0))

# %%
(ordered_logistic_proba(trace_11_1["a"].mean(axis=0)) * (1 + np.arange(7))).sum()

# %%
ordered_logistic_proba(trace_11_1["a"].mean(axis=0) - 0.5)

# %%
(ordered_logistic_proba(trace_11_1["a"].mean(axis=0) - 0.5) * (1 + np.arange(7))).sum()

# %%
action = shared(trolley_df.action.values)
intention = shared(trolley_df.intention.values)
contact = shared(trolley_df.contact.values)

with pm.Model() as m11_2:
    a = pm.Normal(
        "a",
        0.0,
        10.0,
        transform=pm.distributions.transforms.ordered,
        shape=6,
        testval=trace_11_1["a"].mean(axis=0),
    )

    bA = pm.Normal("bA", 0.0, 10.0)
    bI = pm.Normal("bI", 0.0, 10.0)
    bC = pm.Normal("bC", 0.0, 10.0)
    phi = bA * action + bI * intention + bC * contact

    resp_obs = pm.OrderedLogistic(
        "resp_obs", phi, a, observed=trolley_df.response.values - 1
    )

# %%
with m11_2:
    map_11_2 = pm.find_MAP()

# %%
with pm.Model() as m11_3:
    a = pm.Normal(
        "a",
        0.0,
        10.0,
        transform=pm.distributions.transforms.ordered,
        shape=6,
        testval=trace_11_1["a"].mean(axis=0),
    )

    bA = pm.Normal("bA", 0.0, 10.0)
    bI = pm.Normal("bI", 0.0, 10.0)
    bC = pm.Normal("bC", 0.0, 10.0)
    bAI = pm.Normal("bAI", 0.0, 10.0)
    bCI = pm.Normal("bCI", 0.0, 10.0)
    phi = (
        bA * action
        + bI * intention
        + bC * contact
        + bAI * action * intention
        + bCI * contact * intention
    )

    resp_obs = pm.OrderedLogistic("resp_obs", phi, a, observed=trolley_df.response - 1)

# %%
with m11_3:
    map_11_3 = pm.find_MAP()

# %%


def get_coefs(map_est):
    coefs = OrderedDict()

    for i, ai in enumerate(map_est["a"]):
        coefs["a_{}".format(i)] = ai

    coefs["bA"] = map_est.get("bA", np.nan)
    coefs["bI"] = map_est.get("bI", np.nan)
    coefs["bC"] = map_est.get("bC", np.nan)
    coefs["bAI"] = map_est.get("bAI", np.nan)
    coefs["bCI"] = map_est.get("bCI", np.nan)

    return coefs


# %%
(
    pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("m11_1", get_coefs(map_11_1)),
                ("m11_2", get_coefs(map_11_2)),
                ("m11_3", get_coefs(map_11_3)),
            ]
        )
    )
    .astype(np.float64)
    .round(2)
)

# %%
with m11_2:
    trace_11_2 = pm.sample(1000, tune=1000)

with m11_3:
    trace_11_3 = pm.sample(1000, tune=1000)

# %%
comp_df = pm.compare({m11_1: trace_11_1, m11_2: trace_11_2, m11_3: trace_11_3})

comp_df.loc[:, "model"] = pd.Series(["m11.1", "m11.2", "m11.3"])
comp_df = comp_df.set_index("model")
comp_df

# %%
pp_df = pd.DataFrame(
    np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]]),
    columns=["action", "contact", "intention"],
)

# %%
pp_df

# %%
action.set_value(pp_df.action.values)
contact.set_value(pp_df.contact.values)
intention.set_value(pp_df.intention.values)

with m11_3:
    pp_trace_11_3 = pm.sample_ppc(trace_11_3, samples=1500)

# %%
PP_COLS = ["pp_{}".format(i) for i, _ in enumerate(pp_trace_11_3["resp_obs"])]

pp_df = pd.concat(
    (pp_df, pd.DataFrame(pp_trace_11_3["resp_obs"].T, columns=PP_COLS)), axis=1
)

# %%
pp_cum_df = (
    pd.melt(
        pp_df,
        id_vars=["action", "contact", "intention"],
        value_vars=PP_COLS,
        value_name="resp",
    )
    .groupby(["action", "contact", "intention", "resp"])
    .size()
    .div(1500)
    .rename("proba")
    .reset_index()
    .pivot_table(
        index=["action", "contact", "intention"], values="proba", columns="resp"
    )
    .cumsum(axis=1)
    .iloc[:, :-1]
)

# %%
pp_cum_df

# %%
for (plot_action, plot_contact), plot_df in pp_cum_df.groupby(
    level=["action", "contact"]
):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], plot_df, c="C0")
    ax.plot([0, 1], [0, 0], "--", c="C0")
    ax.plot([0, 1], [1, 1], "--", c="C0")

    ax.set_xlim(0, 1)
    ax.set_xlabel("intention")

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("probability")

    ax.set_title(
        "action = {action}, contact = {contact}".format(
            action=plot_action, contact=plot_contact
        )
    )

# %%
PROB_DRINK = 0.2
RATE_WORK = 1.0

N = 365


# %%
drink = np.random.binomial(1, PROB_DRINK, size=N)
y = (1 - drink) * np.random.poisson(RATE_WORK, size=N)

# %%
drink_zeros = drink.sum()
work_zeros = (y == 0).sum() - drink_zeros

# %%
bins = np.arange(y.max() + 1) - 0.5

plt.hist(y, bins=bins)
plt.bar(0.0, drink_zeros, width=1.0, bottom=work_zeros, color="C1", alpha=0.5)

plt.xticks(bins + 0.5)
plt.xlabel("manuscripts completed")

plt.ylabel("Frequency")

# %%
with pm.Model() as m11_4:
    ap = pm.Normal("ap", 0.0, 1.0)
    p = pm.math.sigmoid(ap)

    al = pm.Normal("al", 0.0, 10.0)
    lambda_ = pm.math.exp(al)

    y_obs = pm.ZeroInflatedPoisson("y_obs", 1.0 - p, lambda_, observed=y)

# %%
with m11_4:
    map_11_4 = pm.find_MAP()

# %%
map_11_4

# %%
sp.special.expit(map_11_4["ap"])

# %%
np.exp(map_11_4["al"])

# %%


def dzip(x, p, lambda_, log=True):
    like = p ** (x == 0) + (1 - p) * sp.stats.poisson.pmf(x, lambda_)

    return np.log(like) if log else like


# %%
PBAR = 0.5
THETA = 5.0


# %%
a = PBAR * THETA
b = (1 - PBAR) * THETA

# %%
p = np.linspace(0, 1, 100)

plt.plot(p, sp.stats.beta.pdf(p, a, b))

plt.xlim(0, 1)
plt.xlabel("probability")

plt.ylabel("Density")

# %%
admit_df = pd.read_csv("data/UCBadmit.csv", sep=";")

# %%
with pm.Model() as m11_5:
    a = pm.Normal("a", 0.0, 2.0)
    pbar = pm.Deterministic("pbar", pm.math.sigmoid(a))

    theta = pm.Exponential("theta", 1.0)

    admit_obs = pm.BetaBinomial(
        "admit_obs",
        pbar * theta,
        (1.0 - pbar) * theta,
        admit_df.applications.values,
        observed=admit_df.admit.values,
    )


# %%
with m11_5:
    trace_11_5 = pm.sample(1000, tune=1000)

# %%
pm.summary(trace_11_5).round(2)

# %%
np.percentile(trace_11_5["pbar"], [2.5, 50.0, 97.5])

# %%
pbar_hat = trace_11_5["pbar"].mean()
theta_hat = trace_11_5["theta"].mean()

p_plot = np.linspace(0, 1, 100)

plt.plot(
    p_plot,
    sp.stats.beta.pdf(p_plot, pbar_hat * theta_hat, (1.0 - pbar_hat) * theta_hat),
)
plt.plot(
    p_plot,
    sp.stats.beta.pdf(
        p_plot[:, np.newaxis],
        trace_11_5["pbar"][:100] * trace_11_5["theta"][:100],
        (1.0 - trace_11_5["pbar"][:100]) * trace_11_5["theta"][:100],
    ),
    c="C0",
    alpha=0.1,
)

plt.xlim(0.0, 1.0)
plt.xlabel("probability admit")

plt.ylim(0.0, 3.0)
plt.ylabel("Density")

# %%
with m11_5:
    pp_trace_11_5 = pm.sample_ppc(trace_11_5)

# %%
x_case = np.arange(admit_df.shape[0])

plt.scatter(
    x_case, pp_trace_11_5["admit_obs"].mean(axis=0) / admit_df.applications.values
)
plt.scatter(x_case, admit_df.admit / admit_df.applications)

high = (
    np.percentile(pp_trace_11_5["admit_obs"], 95, axis=0) / admit_df.applications.values
)
plt.scatter(x_case, high, marker="x", c="k")

low = (
    np.percentile(pp_trace_11_5["admit_obs"], 5, axis=0) / admit_df.applications.values
)
plt.scatter(x_case, low, marker="x", c="k")

# %%
mu = 3.0
theta = 1.0

x = np.linspace(0, 10, 100)
plt.plot(x, sp.stats.gamma.pdf(x, mu / theta, scale=theta))
