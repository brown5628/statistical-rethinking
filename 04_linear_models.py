# %%
import arviz as az 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import pymc3 as pm 
import scipy.stats as stats 
import altair as alt
from scipy.interpolate import griddata 

# %%
pos = np.random.uniform(-1, 1, size=(16, 1000)).sum(0)
az.plot_kde(pos)
plt.xlabel("position")
plt.ylabel("Density")

# %%
pos = np.random.uniform(1, 1.1, size=(12, 10000)).prod(0)
az.plot_kde(pos)

# %%
log_big = np.log(np.random.uniform(1, 1.5, size=(12, 10000)).prod(0))
az.plot_kde(log_big)

# %%
w, n = 6, 9
p_grid = np.linspace(0, 1, 100)
posterior = stats.binom.pmf(k=w, n=n, p=p_grid) * stats.uniform.pdf(p_grid, 0, 1)
posterior = posterior / (posterior).sum()
plt.plot(p_grid, posterior)
plt.xlabel("p")
plt.ylabel("Density")

# %%
d= pd.read_csv("data/Howell1.csv", sep=";", header=0)
d.head()

# %%
az.summary(d.to_dict(orient="list"), kind="stats")

# %%
d.height

# %%
d2 = d[d.age >= 18]

# %%
d2['height'].plot(kind='hist')

# %%
alt.Chart(d2).mark_bar().encode(
    alt.X("height:Q", bin=True),
    y='count()'
)

# %%
x = np.linspace(100, 250, 100)
source = pd.DataFrame({
    'x': x,
    'f(x)': stats.norm.pdf(x, 178, 20)
})

alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
)

# %%
x = np.linspace(-10, 60, 100) 
source = pd.DataFrame({
    'x': x,
    'f(x)': stats.uniform.pdf(x, 0, 50)
})

alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
)

# %%
n_samples = 10000
sample_mu = stats.norm.rvs(loc=178, scale=20, size=n_samples)
sample_sigma = stats.uniform.rvs(loc=0, scale=50, size=n_samples)
prior_h = stats.norm.rvs(loc=sample_mu, scale=sample_sigma)
az.plot_kde(prior_h)
plt.xlabel("heights")
plt.yticks([])

# %%
sample_mu = stats.norm.rvs(loc=178, scale =100, size=n_samples)
prior_h = stats.norm.rvs(loc=sample_mu, scale=sample_sigma)
az.plot_kde(prior_h)
plt.xlabel("heights")
plt.yticks([])

# %%
post = np.mgrid[150:160:0.05, 7:9:0.05].reshape(2, -1).T 

likelihood = [
    sum(stats.norm.logpdf(d2.height, loc=post[:, 0][i], scale = post[:, 1][i]))
    for i in range(len(post))
]

post_prod = (
    likelihood
    + stats.norm.logpdf(post[:, 0], loc=178, scale=20)
    + stats.uniform.logpdf(post[:, 1], loc=0, scale =50)
)
post_prob = np.exp(post_prod - max(post_prod))

# %%
xi = np.linspace(post[:, 0].min(), post[:, 0].max(), 100)
yi = np.linspace(post[:, 1].min(), post[:, 1].max(), 100)
zi = griddata((post[:, 0], post[:, 1]), post_prob, (xi[None, :], yi[:, None]))

plt.contour(xi, yi, zi)

# %%
