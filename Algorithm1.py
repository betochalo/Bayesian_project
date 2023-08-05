import numpy as np
from covariance_function import secd


# Parameters
t1 = 0
tw = 600
N = 501
w = 20
t = np.linspace(t1, tw, w)

# Initial_values
u0 = np.array([0, 1000])
# u0 = u0.reshape((2, 1))
# Treatment vector
x = np.array([7.5, 1000])
# x = x.reshape((2, 1))
# Physical parameters
theta = np.array([200, 0.05, 100, 100])
# theta = theta.reshape((4, 1))
# Evaluation grid
tao = np.linspace(t1, tw, N)

# Function


def function(u0, x, theta):
    u12 = u0[0] + u0[1]
    x12 = x[0] + x[1]
    uax = (1/theta[0])*(2*x12*u12 + (1 + theta[1])*(theta[3]*x12 + theta[2]*u12) + 2*theta[2]*theta[3])
    f1 = (x[0]*(u0[1] + theta[1]*theta[3]) - u0[0]*(x[1] + theta[1]*theta[2]))/uax
    f2 = (x[1]*(u0[0] + theta[1]*theta[3]) - u0[1]*(x[0] + theta[1]*theta[3]))/uax
    f = np.array([f1, f2])
    return f


A = 0
f = function(u0, x, theta)
r = 2 + 1
tao1 = tao[0:r]
a1 = np.array([1, 2])
a2 = np.array([[1, 2], [2, 4]])
b = a1 * a2
print(tao1)
# for r in range(0, N-2):
    # tao[r] =







