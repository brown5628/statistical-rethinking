# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.special import logsumexp
import theano

# %%
az.style.use("arviz-darkgrid")
az.rcParams["stats.credible_interval"] = 0.89
np.random.seed(0)

# %%
brains = pd.DataFrame.from_dict(
    {
        "species": [
            "afarensis",
            "africanus",
            "habilis",
            "boisei",
            "rudolfensis",
            "ergaster",
            "sapiens",
        ],
        "brain": [438, 452, 612, 521, 752, 871, 1350],
        "mass": [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5],
    }
)
d = brains

# %%
m_6_1 = smf.ols("brain ~ mass", data=d).fit()

# %%
1 - m_6_1.resid.var() / d.brain.var()

# %%
m_6_1.summary()

# %%
m_6_2 = smf.ols("brain ~ mass + I(mass**2)", data=d).fit()

# %%
m_6_3 = smf.ols("brain ~ mass + I(mass**2) + I(mass**3)", data=d).fit()
m_6_4 = smf.ols("brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4)", data=d).fit()
m_6_5 = smf.ols(
    "brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4) + I(mass**5)", data=d
).fit()
m_6_6 = smf.ols(
    "brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4) + I(mass**5) + I(mass**6)",
    data=d,
).fit()

# %%
m_6_7 = smf.ols("brain ~ 1", data=d).fit()

# %%
d_new = d.drop(d.index[-1])

# %%
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
ax1.scatter(d.mass, d.brain, alpha=0.8)
ax2.scatter(d.mass, d.brain, alpha=0.8)
for i in range(len(d)):
    d_new = d.drop(d.index[-i])
    m0 = smf.ols("brain ~ mass", d_new).fit()
    # need to calculate regression line
    # need to add intercept term explicitly
    x = sm.add_constant(d_new.mass)  # add constant to new data frame with mass
    x_pred = pd.DataFrame(
        {"mass": np.linspace(x.mass.min() - 10, x.mass.max() + 10, 50)}
    )  # create linspace dataframe
    x_pred2 = sm.add_constant(
        x_pred
    )  # add constant to newly created linspace dataframe
    y_pred = m0.predict(x_pred2)  # calculate predicted values
    ax1.plot(x_pred, y_pred, "gray", alpha=0.5)
    ax1.set_ylabel("body mass (kg)", fontsize=12)
    ax1.set_xlabel("brain volume (cc)", fontsize=12)
    ax1.set_title("Underfit model")

    # fifth order model
    m1 = smf.ols(
        "brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4) + I(mass**5)", data=d_new
    ).fit()
    x = sm.add_constant(d_new.mass)  # add constant to new data frame with mass
    x_pred = pd.DataFrame(
        {"mass": np.linspace(x.mass.min() - 10, x.mass.max() + 10, 200)}
    )  # create linspace dataframe
    x_pred2 = sm.add_constant(
        x_pred
    )  # add constant to newly created linspace dataframe
    y_pred = m1.predict(x_pred2)  # calculate predicted values from fitted model
    ax2.plot(x_pred, y_pred, "gray", alpha=0.5)
    ax2.set_xlim(32, 62)
    ax2.set_ylim(-250, 2200)
    ax2.set_ylabel("body mass (kg)", fontsize=12)
    ax2.set_xlabel("brain volume (cc)", fontsize=12)
    ax2.set_title("Overfit model")
plt.show()

# %%
p = (0.3, 0.7)
-sum(p * np.log(p))

# %%
-2 * m_6_1.llf

# %%
d["mass_s"] = d["mass"] - np.mean(d["mass"] / np.std(d["mass"]))

with pm.Model() as m_6_8:
    a = pm.Normal("a", mu=np.mean(d["brain"]), sd=10)
    b = pm.Normal("b", mu=0, sd=10)
    sigma = pm.Uniform("sigma", 0, np.std(d["brain"]) * 10)
    mu = pm.Deterministic("mu", a + b * d["mass_s"])
    brain = pm.Normal("brain", mu=mu, sd=sigma, observed=d["brain"])
    m_6_8 = pm.sample(2000, tune=5000)

# %%
theta = az.summary(m_6_8)["mean"][:3]

# %%
dev = -2 * sum(
    stats.norm.logpdf(d["brain"], loc=theta[0] + theta[1] * d["mass_s"], scale=theta[2])
)
dev

# %%


def sim_train_test(N=20, k=3, rho=[0.15, -0.4], b_sigma=100):

    n_dim = 1 + len(rho)
    if n_dim < k:
        n_dim = k
    Rho = np.diag(np.ones(n_dim))
    Rho[0, 1:3:1] = rho
    i_lower = np.tril_indices(n_dim, -1)
    Rho[i_lower] = Rho.T[i_lower]

    x_train = stats.multivariate_normal.rvs(cov=Rho, size=N)
    x_test = stats.multivariate_normal.rvs(cov=Rho, size=N)

    mm_train = np.ones((N, 1))

    np.concatenate([mm_train, x_train[:, 1:k]], axis=1)

    # Using pymc3

    with pm.Model() as m_sim:
        vec_V = pm.MvNormal(
            "vec_V",
            mu=0,
            cov=b_sigma * np.eye(n_dim),
            shape=(1, n_dim),
            testval=np.random.randn(1, n_dim) * 0.01,
        )
        mu = pm.Deterministic("mu", 0 + pm.math.dot(x_train, vec_V.T))
        y = pm.Normal("y", mu=mu, sd=1, observed=x_train[:, 0])

    with m_sim:
        trace_m_sim = pm.sample()

    vec = pm.summary(trace_m_sim)["mean"][:n_dim]
    vec = np.array([i for i in vec]).reshape(n_dim, -1)

    dev_train = -2 * sum(
        stats.norm.logpdf(x_train, loc=np.matmul(x_train, vec), scale=1)
    )

    mm_test = np.ones((N, 1))

    mm_test = np.concatenate([mm_test, x_test[:, 1 : k + 1]], axis=1)

    dev_test = -2 * sum(
        stats.norm.logpdf(x_test[:, 0], loc=np.matmul(mm_test, vec), scale=1)
    )

    return np.mean(dev_train), np.mean(dev_test)


# %%
n = 20
tries = 10
param = 6
r = np.zeros(shape=(param - 1, 4))

train = []
test = []

for j in range(2, param + 1):
    print(j)
    for i in range(1, tries + 1):
        tr, te = sim_train_test(N=n, k=param)
        train.append(tr), test.append(te)
    r[j - 2, :] = (
        np.mean(train),
        np.std(train, ddof=1),
        np.mean(test),
        np.std(test, ddof=1),
    )

# %%
num_param = np.arange(2, param + 1)

plt.figure(figsize=(10, 6))
plt.scatter(num_param, r[:, 0], color="C0")
plt.xticks(num_param)

for j in range(param - 1):
    plt.vlines(
        num_param[j],
        r[j, 0] - r[j, 1],
        r[j, 0] + r[j, 1],
        color="mediumblue",
        zorder=-1,
        alpha=0.80,
    )

plt.scatter(num_param + 0.1, r[:, 2], facecolors="none", edgecolors="k")

for j in range(param - 1):
    plt.vlines(
        num_param[j] + 0.1,
        r[j, 2] - r[j, 3],
        r[j, 2] + r[j, 3],
        color="k",
        zorder=-2,
        alpha=0.70,
    )

dist = 0.20
plt.text(num_param[1] - dist, r[1, 0] - dist, "in", color="C0", fontsize=13)
plt.text(num_param[1] + dist, r[1, 2] - dist, "out", color="k", fontsize=13)
plt.text(num_param[1] + dist, r[1, 2] + r[1, 3] - dist, "+1 SD", color="k", fontsize=10)
plt.text(num_param[1] + dist, r[1, 2] - r[1, 3] - dist, "+1 SD", color="k", fontsize=10)
plt.xlabel("Number of parameters", fontsize=14)
plt.ylabel("Deviance", fontsize=14)
plt.title("N = {}".format(n), fontsize=14)
plt.show()

# %%
data = pd.read_csv("data/cars.csv", sep=",")

# %%
with pm.Model() as m_6_15:
    a = pm.Normal("a", mu=0, sd=100)
    b = pm.Normal("b", mu=0, sd=10)
    sigma = pm.Uniform("sigma", 0, 30)
    mu = pm.Deterministic("mu", a + b * data["speed"])
    dist = pm.Normal("dist", mu=mu, sd=sigma, observed=data["dist"])
    m_6_15 = pm.sample(5000, tune=10000)

# %%
n_samples = 1000
n_cases = data.shape[0]
ll = np.zeros((n_cases, n_samples))

for s in range(0, n_samples):
    mu = m_6_15["a"][s] + m_6_15["b"][s] * data["speed"]
    p_ = stats.norm.logpdf(data["dist"], loc=mu, scale=m_6_15["sigma"][s])
    ll[:, s] = p_

# %%
n_cases = data.shape[0]
lppd = np.zeros((n_cases))
for a in range(1, n_cases):
    lppd[a,] = logsumexp(ll[a,]) - np.log(n_samples)

# %%
pWAIC = np.zeros((n_cases))
for i in range(1, n_cases):
    pWAIC[i,] = np.var(ll[i,])

# %%
-2 * (sum(lppd) - sum(pWAIC))

# %%
waic_vec = -2 * (lppd - pWAIC)
(n_cases * np.var(waic_vec)) ** 0.5

# %%
d = pd.read_csv("data/milk.csv", sep=";")
d["neocortex"] = d["neocortex.perc"] / 100
d.dropna(inplace=True)
d.shape

# %%
a_start = d["kcal.per.g"].mean()
sigma_start = d["kcal.per.g"].std()

# %%
mass_shared = theano.shared(np.log(d["mass"].values))
neocortex_shared = theano.shared(d["neocortex"].values)

with pm.Model() as m6_11:
    alpha = pm.Normal("alpha", mu=0, sd=10, testval=a_start)
    mu = alpha + 0 * neocortex_shared
    sigma = pm.HalfCauchy("sigma", beta=10, testval=sigma_start)
    kcal = pm.Normal("kcal", mu=mu, sd=sigma, observed=d["kcal.per.g"])
    trace_m6_11 = pm.sample(1000, tune=1000)

with pm.Model() as m6_12:
    alpha = pm.Normal("alpha", mu=0, sd=10, testval=a_start)
    beta = pm.Normal("beta", mu=0, sd=10)
    sigma = pm.HalfCauchy("sigma", beta=10, testval=sigma_start)
    mu = alpha + beta * neocortex_shared
    kcal = pm.Normal("kcal", mu=mu, sd=sigma, observed=d["kcal.per.g"])
    trace_m6_12 = pm.sample(5000, tune=15000)

with pm.Model() as m6_13:
    alpha = pm.Normal("alpha", mu=0, sd=10, testval=a_start)
    beta = pm.Normal("beta", mu=0, sd=10)
    sigma = pm.HalfCauchy("sigma", beta=10, testval=sigma_start)
    mu = alpha + beta * mass_shared
    kcal = pm.Normal("kcal", mu=mu, sd=sigma, observed=d["kcal.per.g"])
    trace_m6_13 = pm.sample(1000, tune=1000)

with pm.Model() as m6_14:
    alpha = pm.Normal("alpha", mu=0, sd=10, testval=a_start)
    beta = pm.Normal("beta", mu=0, sd=10, shape=2)
    sigma = pm.HalfCauchy("sigma", beta=10, testval=sigma_start)
    mu = alpha + beta[0] * mass_shared + beta[1] * neocortex_shared
    kcal = pm.Normal("kcal", mu=mu, sd=sigma, observed=d["kcal.per.g"])
    trace_m6_14 = pm.sample(5000, tune=15000)

# %%
az.waic(trace_m6_14, m6_14)

# %%
compare_df = az.compare(
    {
        "m6_11": trace_m6_11,
        "m6_12": trace_m6_12,
        "m6_13": trace_m6_13,
        "m6_14": trace_m6_14,
    },
    method="pseudo-BMA",
)
compare_df

# %%
az.plot_compare(compare_df)

# %%
diff = np.random.normal(loc=6.7, scale=7.26, size=100000)
sum(diff < 0) / 100000

# %%
coeftab = pd.DataFrame(
    {
        "m6_11": pm.summary(trace_m6_11)["mean"],
        "m6_12": pm.summary(trace_m6_12)["mean"],
        "m6_13": pm.summary(trace_m6_13)["mean"],
        "m6_14": pm.summary(trace_m6_14)["mean"],
    }
)
coeftab

# %%
traces = [trace_m6_11, trace_m6_12, trace_m6_13, trace_m6_14]
models = [m6_11, m6_12, m6_13, m6_14]

# %%
az.plot_forest(traces, figsize=(10, 5))

# %%
kcal_per_g = np.repeat(0, 30)
neocortex = np.linspace(0.5, 0.8, 30)
mass = np.repeat(4.5, 30)

# %%
mass_shared.set_value(np.log(mass))
neocortex_shared.set_value(neocortex)
post_pred = pm.sample_posterior_predictive(trace_m6_14, samples=10000, model=m6_14)

# %%
milk_ensemble = pm.sample_posterior_predictive_w(
    traces, 10000, models, weights=compare_df.weight.sort_index(ascending=True)
)

# %%
plt.figure(figsize=(8, 6))

plt.plot(neocortex, post_pred["kcal"].mean(0), ls="--", color="k")
az.plot_hpd(
    neocortex,
    post_pred["kcal"],
    fill_kwargs={"alpha": 0},
    plot_kwargs={"alpha": 1, "color": "k", "ls": "--"},
)

plt.plot(neocortex, milk_ensemble["kcal"].mean(0), color="C1")
az.plot_hpd(neocortex, milk_ensemble["kcal"])

plt.scatter(d["neocortex"], d["kcal.per.g"], facecolor="None", edgecolors="C0")

plt.ylim(0.3, 1)
plt.xlabel("neocortex")
plt.ylabel("kcal.per.g")
