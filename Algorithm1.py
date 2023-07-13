import numpy as np
from covariance_function import secd

# Parameters
tl = 0
tu = 1
nodes = 50
beta = 0.8
alpha = 0.5

w1 = np.linspace(tl, tu, nodes)


def cscd(w1, beta, alpha):
    c = np.zeros((len(w1), len(w1)))
    for i in range(0, len(w1)):
        for j in range(0, len(w1)):
            c[i, j] = secd(beta, alpha, w1[i], w1[j])
    return c


c = cscd(w1, beta, alpha)

print(c)


