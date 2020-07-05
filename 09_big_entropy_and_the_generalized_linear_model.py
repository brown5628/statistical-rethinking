# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# %%
d = {
    "A": [0, 0, 10, 0, 0],
    "B": [0, 1, 8, 1, 0],
    "C": [0, 2, 6, 2, 0],
    "D": [1, 2, 4, 2, 1],
    "E": [2, 2, 2, 2, 2],
}
p = pd.DataFrame(data=d)

# %%
p_norm = p / p.sum(0)

# %%


def entropy(x):
    y = []
    for i in x:
        if i == 0:
            y.append(0)
        else:
            y.append(i * np.log(i))
    h = -sum(y)
    return h


H = p_norm.apply(entropy, axis=0)
H

# %%
ways = [1, 90, 1260, 37800, 113400]
logwayspp = np.log(ways) / 10
plt.plot(logwayspp, H, "o")
plt.plot([0.0, max(logwayspp)], [0.0, max(H)], "--k")
plt.ylabel("entropy", fontsize=14)
plt.xlabel("log(ways) per pebble")

# %%
p = [
    [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    [2 / 6, 1 / 6, 1 / 6, 2 / 6],
    [1 / 6, 2 / 6, 2 / 6, 1 / 6],
    [1 / 8, 4 / 8, 2 / 8, 1 / 8],
]

p_ev = [np.dot(i, [0, 1, 1, 2]) for i in p]
p_ev

# %%
p_ent = [entropy(i) for i in p]
p_ent

# %%
p = 0.7
A = [(1 - p) ** 2, p * (1 - p), (1 - p) * p, p ** 2]
A

# %%
-np.sum(A * np.log(A))

# %%
def sim_p(G=1.4):
    x123 = np.random.uniform(size=3)
    x4 = G * np.sum(x123) - x123[1] - x123[2] / (2 - G)
    x1234 = np.concatenate((x123, [x4]))
    z = np.sum(x1234)
    p = x1234 / z
    return -np.sum(p * np.log(p)), p


# %%
H = []
p = np.zeros((10 ** 5, 4))
for rep in range(10 ** 5):
    h, p_ = sim_p()
    H.append(h)
    p[rep] = p_

# %%
az.plot_kde(H)
plt.xlabel("Entropy")
plt.ylabel("Density")

# %%
np.max(H)

# %%
p[np.argmax(H)]

# %%
