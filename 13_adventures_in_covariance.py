# %%
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from theano import tensor as tt
import arviz as az
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# %%
a = 3.5
b = -1.0
sigma_a = 1.0
sigma_b = 0.5
rho = -0.7

# %%
Mu = [a, b]

# %%
cov_ab = sigma_a * sigma_b * rho
Sigma = np.array([[sigma_a ** 2, cov_ab], [cov_ab, sigma_b ** 2]])
Sigma

# %%
sigmas = [sigma_a, sigma_b]
Rho = np.matrix([[1, rho], [rho, 1]])

Sigma = np.diag(sigmas) * Rho * np.diag(sigmas)
Sigma

# %%
N_cafes = 20

# %%
np.random.seed(42)
vary_effects = np.random.multivariate_normal(mean=Mu, cov=Sigma, size=N_cafes)

# %%
a_cafe = vary_effects[:, 0]
b_cafe = vary_effects[:, 1]

# %%


def Gauss2d(mu, cov, ci, ax=None):
    """Copied from statsmodel"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    v_, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees

    for level in ci:
        v = 2 * np.sqrt(v_ * chi2.ppf(level, 2))  # get size corresponding to level
        ell = Ellipse(
            mu[:2],
            v[0],
            v[1],
            180 + angle,
            facecolor="None",
            edgecolor="k",
            alpha=(1 - level) * 0.5,
            lw=1.5,
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    return ax


# %%
_, ax = plt.subplots(1, 1, figsize=(5, 5))
Gauss2d(Mu, np.asarray(Sigma), [0.1, 0.3, 0.5, 0.8, 0.99], ax=ax)
ax.scatter(a_cafe, b_cafe)
ax.set_xlim(1.5, 6.1)
ax.set_ylim(-2, 0)
ax.set_xlabel("intercepts (a_cafe)")
ax.set_ylabel("slopes (b_cafe)")

# %%
N_visits = 10
afternoon = np.tile(
    [0, 1], N_visits * N_cafes // 2
)  # wrap with int() to suppress warnings
cafe_id = np.repeat(
    np.arange(0, N_cafes), N_visits
)  # 1-20 (minus 1 for python indexing)

mu = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
sigma = 0.5  # std dev within cafes
wait = np.random.normal(loc=mu, scale=sigma, size=N_visits * N_cafes)
d = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))

# %%
R = pm.LKJCorr.dist(n=2, eta=2).random(size=10000)
_, ax = plt.subplots(1, 1, figsize=(5, 5))
az.plot_kde(R)
ax.set_xlabel("correlation")
ax.set_ylabel("Density")

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 5))
textloc = [[0, 0.5], [0, 0.8], [0.5, 0.9]]
for eta, loc in zip([1, 2, 4], textloc):
    R = pm.LKJCorr.dist(n=2, eta=eta).random(size=10000)
    az.plot_kde(R)
    ax.text(loc[0], loc[1], "eta = %s" % (eta), horizontalalignment="center")

ax.set_ylim(0, 1.1)
ax.set_xlabel("correlation")
ax.set_ylabel("Density")

# %%
cafe_idx = d["cafe"].values
with pm.Model() as m_13_1:
    sd_dist = pm.HalfCauchy.dist(beta=2)
    packed_chol = pm.LKJCholeskyCov("chol_cov", eta=2, n=2, sd_dist=sd_dist)

    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
    cov = pm.math.dot(chol, chol.T)

    sigma_ab = pm.Deterministic("sigma_cafe", tt.sqrt(tt.diag(cov)))
    corr = tt.diag(sigma_ab ** -1).dot(cov.dot(tt.diag(sigma_ab ** -1)))
    r = pm.Deterministic("Rho", corr[np.triu_indices(2, k=1)])

    ab = pm.Normal("ab", mu=0, sd=10, shape=2)
    ab_cafe = pm.MvNormal("ab_cafe", mu=ab, chol=chol, shape=(N_cafes, 2))

    mu = ab_cafe[:, 0][cafe_idx] + ab_cafe[:, 1][cafe_idx] * d["afternoon"].values
    sd = pm.HalfCauchy("sigma", beta=2)
    wait = pm.Normal("wait", mu=mu, sd=sd, observed=d["wait"])
    trace_13_1 = pm.sample(5000, tune=2000)

# %%
az.plot_trace(
    trace_13_1,
    var_names=["ab", "Rho", "sigma", "sigma_cafe"],
    compact=True,
    lines=[
        ("ab", {}, Mu),
        ("Rho", {}, rho),
        ("sigma", {}, sigma),
        ("sigma_cafe", {}, sigmas),
    ],
)

# %%
post = pm.trace_to_dataframe(trace_13_1)

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 5))
R = pm.LKJCorr.dist(n=2, eta=2).random(size=10000)
az.plot_kde(R, plot_kwargs={"color": "k", "linestyle": "--"})
ax.text(0, 0.8, "prior", horizontalalignment="center")
az.plot_kde(trace_13_1["Rho"], plot_kwargs={"color": "C0"})
ax.text(-0.15, 1.5, "posterior", color="C0", horizontalalignment="center")
ax.set_ylim(-0.025, 2.5)
ax.set_xlabel("correlation")
ax.set_ylabel("Density")

# %%
a1b1 = d.groupby(["afternoon", "cafe"]).agg("mean").unstack(level=0).values
a1 = a1b1[:, 0]
b1 = a1b1[:, 1] - a1

# extract posterior means of partially pooled estimates
a2b2 = trace_13_1["ab_cafe"].mean(axis=0)
a2 = a2b2[:, 0]
b2 = a2b2[:, 1]

# plot both and connect with lines
_, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(a1, b1)
ax.scatter(a2, b2, facecolors="none", edgecolors="k", lw=1)

ax.plot([a1, a2], [b1, b2], "k-", alpha=0.5)
ax.set_xlabel("intercept")
ax.set_ylabel("slope")

# %%
Mu_est = trace_13_1["ab"].mean(axis=0)
chol_model = pm.expand_packed_triangular(
    2, trace_13_1["chol_cov"].mean(0), lower=True
).eval()
Sigma_est = np.dot(chol_model, chol_model.T)
# draw contours
_, ax = plt.subplots(1, 1, figsize=(5, 5))
Gauss2d(Mu_est, np.asarray(Sigma_est), [0.1, 0.3, 0.5, 0.8, 0.99], ax=ax)
ax.scatter(a1, b1)
ax.scatter(a2, b2, facecolors="none", edgecolors="k", lw=1)
ax.plot([a1, a2], [b1, b2], "k-", alpha=0.5)
ax.set_xlabel("intercept", fontsize=14)
ax.set_ylabel("slope", fontsize=14)
ax.set_xlim(1.5, 6.1)
ax.set_ylim(-2.5, 0)

# %%
wait_morning_1 = a1
wait_afternoon_1 = a1 + b1
wait_morning_2 = a2
wait_afternoon_2 = a2 + b2

# %%
d_ad = pd.read_csv("data/UCBadmit.csv", sep=";")
d_ad["male"] = (d_ad["applicant.gender"] == "male").astype(int)
d_ad["dept_id"] = pd.Categorical(d_ad["dept"]).codes

# %%
Dept_id = d_ad["dept_id"].values
Ndept = len(d_ad["dept_id"].unique())
with pm.Model() as m_13_2:
    a = pm.Normal("a", 0, 10)
    bm = pm.Normal("bm", 0, 1)
    sigma_dept = pm.HalfCauchy("sigma_dept", 2)
    a_dept = pm.Normal("a_dept", a, sigma_dept, shape=Ndept)
    p = pm.math.invlogit(a_dept[Dept_id] + bm * d_ad["male"])
    admit = pm.Binomial("admit", p=p, n=d_ad.applications, observed=d_ad.admit)

    trace_13_2 = pm.sample(4500, tune=500)

az.summary(trace_13_2, credible_interval=0.89, round_to=2)

# %%
with pm.Model() as m_13_3:
    a = pm.Normal("a", 0, 10)
    bm = pm.Normal("bm", 0, 1)

    sd_dist = pm.HalfCauchy.dist(beta=2)
    packed_chol = pm.LKJCholeskyCov("chol_cov", eta=2, n=2, sd_dist=sd_dist)

    # compute the covariance matrix
    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
    cov = pm.math.dot(chol, chol.T)

    # Extract the standard deviations and rho
    sigma_ab = pm.Deterministic("sigma_dept", tt.sqrt(tt.diag(cov)))
    corr = tt.diag(sigma_ab ** -1).dot(cov.dot(tt.diag(sigma_ab ** -1)))
    r = pm.Deterministic("Rho", corr[np.triu_indices(2, k=1)])

    mu = pm.MvNormal("ab_dept", mu=tt.stack([a, bm]), chol=chol, shape=(Ndept, 2))

    a_dept = pm.Deterministic("a_dept", mu[:, 0])
    bm_dept = pm.Deterministic("bm_dept", mu[:, 1])

    p = pm.math.invlogit(mu[Dept_id, 0] + mu[Dept_id, 1] * d_ad["male"])
    admit = pm.Binomial("admit", p=p, n=d_ad.applications, observed=d_ad.admit)

    trace_13_3 = pm.sample(5000, tune=1000)

# %%
az.plot_forest(trace_13_3, var_names=["bm_dept", "a_dept"], credible_interval=0.89)

# %%
with pm.Model() as m_13_4:
    a = pm.Normal("a", 0, 10)
    sigma_dept = pm.HalfCauchy("sigma_dept", 2)
    a_dept = pm.Normal("a_dept", a, sigma_dept, shape=Ndept)
    p = pm.math.invlogit(a_dept[Dept_id])
    admit = pm.Binomial("admit", p=p, n=d_ad.applications, observed=d_ad.admit)

    trace_13_4 = pm.sample(4500, tune=500)

comp_df = az.compare({"m13_2": trace_13_2, "m13_3": trace_13_3, "m13_4": trace_13_4})

comp_df

# %%
d = pd.read_csv("data/chimpanzees.csv", sep=";")

actor = (d["actor"] - 1).values
block = (d["block"] - 1).values
Nactor = len(np.unique(actor))
Nblock = len(np.unique(block))

with pm.Model() as model_13_6:
    sd_dist = pm.HalfCauchy.dist(beta=2)
    pchol1 = pm.LKJCholeskyCov("pchol_actor", eta=4, n=3, sd_dist=sd_dist)
    pchol2 = pm.LKJCholeskyCov("pchol_block", eta=4, n=3, sd_dist=sd_dist)
    chol1 = pm.expand_packed_triangular(3, pchol1, lower=True)
    chol2 = pm.expand_packed_triangular(3, pchol2, lower=True)

    Intercept = pm.Normal("intercept", 0.0, 1.0, shape=3)

    beta_actor = pm.MvNormal("beta_actor", mu=0.0, chol=chol1, shape=(Nactor, 3))
    beta_block = pm.MvNormal("beta_block", mu=0.0, chol=chol2, shape=(Nblock, 3))

    A = Intercept[0] + beta_actor[actor, 0] + beta_block[block, 0]
    BP = Intercept[1] + beta_actor[actor, 1] + beta_block[block, 1]
    BPC = Intercept[2] + beta_actor[actor, 2] + beta_block[block, 2]

    p = pm.math.invlogit(A + (BP + BPC * d["condition"]) * d["prosoc_left"])
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d["pulled_left"])

    trace_13_6 = pm.sample(5000, tune=1000)

# %%
with pm.Model() as model_13_6NC:
    sd_dist = pm.HalfCauchy.dist(beta=2)
    pchol1 = pm.LKJCholeskyCov("pchol_actor", eta=4, n=3, sd_dist=sd_dist)
    pchol2 = pm.LKJCholeskyCov("pchol_block", eta=4, n=3, sd_dist=sd_dist)
    chol1 = pm.expand_packed_triangular(3, pchol1, lower=True)
    chol2 = pm.expand_packed_triangular(3, pchol2, lower=True)

    Intercept = pm.Normal("intercept", 0.0, 1.0, shape=3)

    b1 = pm.Normal("b1", 0.0, 1.0, shape=(3, Nactor))
    b2 = pm.Normal("b2", 0.0, 1.0, shape=(3, Nblock))
    beta_actor = pm.math.dot(chol1, b1)
    beta_block = pm.math.dot(chol2, b2)

    A = Intercept[0] + beta_actor[0, actor] + beta_block[0, block]
    BP = Intercept[1] + beta_actor[1, actor] + beta_block[1, block]
    BPC = Intercept[2] + beta_actor[2, actor] + beta_block[2, block]

    p = pm.math.invlogit(A + (BP + BPC * d["condition"]) * d["prosoc_left"])
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d["pulled_left"])

    trace_13_6NC = pm.sample(5000, tune=1000)

# %%
# extract n_eff values for each model
neff_c = az.summary(trace_13_6)["ess_bulk"].values
neff_nc = az.summary(trace_13_6NC)["ess_bulk"].values
# plot distributions
_, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.boxplot([neff_c, neff_nc], labels=["m13.6", "m13.6NC"])
ax.set_xlabel("model", fontsize=14)
ax.set_ylabel("effective samples")

# %%


def unpack_sigma(pack_chol):
    idxs = np.tril_indices(3)
    chol_ = np.zeros((3, 3, pack_chol.shape[0]))
    chol_[idxs] = pack_chol.T
    chol = np.transpose(chol_, (2, 0, 1))
    cholt = np.transpose(chol, (0, 2, 1))
    sigma = np.matmul(chol, cholt)
    return np.sqrt(np.diagonal(sigma, axis1=1, axis2=2))


sigmadict = dict(
    Sigma_actor=unpack_sigma(trace_13_6NC.get_values("pchol_actor", combine=True)),
    Sigma_block=unpack_sigma(trace_13_6NC.get_values("pchol_block", combine=True)),
)
trace_13_6NC.add_values(sigmadict)
az.summary(trace_13_6NC, var_names=["Sigma_actor", "Sigma_block"], round_to=2)

# %%
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
        + a_actor[actor]
        + a_block[block]
        + (bp + bpC * d["condition"]) * d["prosoc_left"]
    )
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d["pulled_left"])

    trace_12_5 = pm.sample(6000, tune=1000)

# %%
comp_df = az.compare({"m13_6NC": trace_13_6NC, "m12_5": trace_12_5})

comp_df

# %%
Dmat = pd.read_csv("data/islandsDistMatrix.csv", sep=",", index_col=0)
Dmat.round(1)

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 5))
xrange = np.linspace(0, 4, 100)
ax.plot(xrange, np.exp(-1 * xrange), "k--")
ax.plot(xrange, np.exp(-1 * xrange ** 2), "k")
ax.set_xlabel("distance")
ax.set_ylabel("correlation")

# %%
dk = pd.read_csv("data/Kline2.csv", sep=",")
Nsociety = dk.shape[0]
dk.loc[:, "society"] = np.arange(Nsociety)
Dmat_ = Dmat.values
Dmatsq = np.power(Dmat_, 2)

# %%
with pm.Model() as m_13_7:
    etasq = pm.HalfCauchy("etasq", 1)
    rhosq = pm.HalfCauchy("rhosq", 1)
    Kij = etasq * (tt.exp(-rhosq * Dmatsq) + np.diag([0.01] * Nsociety))

    g = pm.MvNormal("g", mu=np.zeros(Nsociety), cov=Kij, shape=Nsociety)

    a = pm.Normal("a", 0, 10)
    bp = pm.Normal("bp", 0, 1)
    lam = pm.math.exp(a + g[dk.society.values] + bp * dk.logpop)
    obs = pm.Poisson("total_tools", lam, observed=dk.total_tools)
    trace_13_7 = pm.sample(1000, tune=1000)

# %%
az.plot_trace(trace_13_7, var_names=["g", "a", "bp", "etasq", "rhosq"], compact=True)

# %%
az.summary(trace_13_7, var_names=["g", "a", "bp", "etasq", "rhosq"], round_to=2)

# %%
post = pm.trace_to_dataframe(trace_13_7, varnames=["g", "a", "bp", "etasq", "rhosq"])
post_etasq = post["etasq"].values
post_rhosq = post["rhosq"].values

_, ax = plt.subplots(1, 1, figsize=(8, 5))
xrange = np.linspace(0, 10, 200)

ax.plot(
    xrange, np.median(post_etasq) * np.exp(-np.median(post_rhosq) * xrange ** 2), "k"
)
ax.plot(
    xrange,
    (post_etasq[:100][:, None] * np.exp(-post_rhosq[:100][:, None] * xrange ** 2)).T,
    "k",
    alpha=0.1,
)

ax.set_ylim(0, 1)
ax.set_xlabel("distance")
ax.set_ylabel("covariance")

# %%
Kij_post = np.median(post_etasq) * (
    np.exp(-np.median(post_rhosq) * Dmatsq) + np.diag([0.01] * Nsociety)
)

# %%
# convert to correlation matrix
sigma_post = np.sqrt(np.diag(Kij_post))
Rho = np.diag(sigma_post ** -1).dot(Kij_post.dot(np.diag(sigma_post ** -1)))
# add row/col names for convenience
Rho = pd.DataFrame(
    Rho,
    index=["Ml", "Ti", "SC", "Ya", "Fi", "Tr", "Ch", "Mn", "To", "Ha"],
    columns=["Ml", "Ti", "SC", "Ya", "Fi", "Tr", "Ch", "Mn", "To", "Ha"],
)

Rho.round(2)

# %%
# scale point size to logpop
logpop = np.copy(dk["logpop"].values)
logpop /= logpop.max()
psize = np.exp(logpop * 5.5)

_, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(dk["lon2"], dk["lat"], psize)
labels = dk["culture"].values
for i, itext in enumerate(labels):
    ax.text(dk["lon2"][i] + 1, dk["lat"][i] + 1, itext)
# overlay lines shaded by Rho
for i in range(10):
    for j in np.arange(i + 1, 10):
        ax.plot(
            [dk["lon2"][i], dk["lon2"][j]],
            [dk["lat"][i], dk["lat"][j]],
            "k-",
            alpha=Rho.iloc[i, j] ** 2,
            lw=2.5,
        )
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")

# %%
# compute posterior median relationship, ignoring distance
Nsamp, Nbin = 1000, 30
log_pop_seq = np.linspace(6, 14, Nbin)
a_post = trace_13_7.get_values(varname="a", combine=True)[:, None]
bp_post = trace_13_7.get_values(varname="bp", combine=True)[:, None]
lambda_post = np.exp(a_post + bp_post * log_pop_seq)

_, axes = plt.subplots(1, 1, figsize=(5, 5))
cred_interval = 0.8

# display posterior predictions
axes.plot(log_pop_seq, np.median(lambda_post, axis=0), "--", color="k")


az.plot_hpd(
    log_pop_seq,
    lambda_post,
    credible_interval=cred_interval,
    color="k",
    fill_kwargs={"alpha": cred_interval * 0.5},
)

# plot raw data and labels
axes.scatter(dk["logpop"], dk["total_tools"], psize)
labels = dk["culture"].values
for i, itext in enumerate(labels):
    axes.text(dk["logpop"][i] + 0.1, dk["total_tools"][i] - 2.5, itext)

# overlay correlations
for i in range(10):
    for j in np.arange(i + 1, 10):
        axes.plot(
            [dk["logpop"][i], dk["logpop"][j]],
            [dk["total_tools"][i], dk["total_tools"][j]],
            "k-",
            alpha=Rho.iloc[i, j] ** 2,
            lw=2.5,
        )

axes.set_xlabel("log-population")
axes.set_ylabel("total tools")
axes.set_xlim(6.8, 12.8)
axes.set_ylim(10, 73)
