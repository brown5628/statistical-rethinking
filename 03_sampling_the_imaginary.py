# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# %%
PrPV = 0.95
PrPM = 0.01
PrV = 0.001
PrP = PrPV * PrV + PrPM * (1 - PrV)
PrVP = PrPV * PrV / PrP
PrVP

# %%


def posterior_grid_approx(grid_points=100, success=6, tosses=9):
    """
    """
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform
    # prior = (p_grid >= .5).astype(int) # truncated
    # prior = np.exp(- 5 * abs(p_grid - .5)) # double exp

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior


# %%
p_grid, posterior = posterior_grid_approx(grid_points=100, success=6, tosses=9)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)

# %%
_, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.plot(samples, "o", alpha=0.2)
ax0.set_xlabel("sample number")
ax0.set_ylabel("proportion water (p)")
az.plot_kde(samples, ax=ax1)
ax1.set_xlabel("proportion water (p)")
ax1.set_ylabel("density")

# %%
sum(posterior[p_grid < 0.5])


# %%
sum(samples < 0.5) / 1e4

# %%
sum((samples > 0.5) & (samples < 0.75)) / 1e4

# %%
np.percentile(samples, 80)

# %%
np.percentile(samples, [10, 90])

# %%
p_grid, posterior = posterior_grid_approx(success=3, tosses=3)
plt.plot(p_grid, posterior)
plt.xlabel("proportion water (p)")
plt.ylabel("Density")

# %%
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
np.percentile(samples, [25, 75])

# %%
az.hpd(samples, credible_interval=0.5)

# %%
p_grid[posterior == max(posterior)]

# %%
stats.mode(samples)[0]

# %%
np.mean(samples), np.median(samples)

# %%
sum(posterior * abs(0.5 - p_grid))

# %%
loss = [sum(posterior * abs(p - p_grid)) for p in p_grid]
p_grid[loss == min(loss)]

# %%
stats.binom.pmf(range(3), n=2, p=0.7)

# %%
stats.binom.rvs(n=2, p=0.7, size=1)

# %%
stats.binom.rvs(n=2, p=0.7, size=10)

# %%
dummy_w = stats.binom.rvs(n=2, p=0.7, size=int(1e5))
[(dummy_w == i).mean() for i in range(3)]

# %%
dummy_w = stats.binom.rvs(n=9, p=0.7, size=int(1e5))
plt.hist(dummy_w, bins=50)
plt.xlabel("dummy water count")
plt.ylabel("Frequency")

# %%
p_grid, posterior = posterior_grid_approx(grid_points=100, success=6, tosses=9)
np.random.seed(100)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
