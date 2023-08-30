import numpy as np
import covariance_function as cf
import math
from scipy import special as sp
from scipy.stats import multivariate_normal


# Nuevas funciones
def rr_t(u1, v1, w, alpha):
    u = np.tile(u1, (1, len(v1)))
    v = np.tile(v1.T, (len(u1), 1))
    a1 = np.exp(-np.power(u-v, 2) / (4*w**2))
    rrt = math.sqrt(math.pi) * (alpha**-1) * w * a1
    return rrt


def rq_t(u1, v1, w, alpha, a):
    u = np.tile(u1, (1, len(v1)))
    v = np.tile(v1.T, (len(u1), 1))
    a1 = sp.erf((v-u)/(2*w))
    a2 = sp.erf((u-a)/(2*w))
    rqt = (alpha**-1)*math.pi * (w**2) * a1 + (alpha**-1)*math.pi * (w**2) * a2
    return rqt


def qr_t(u1, v1, w, alpha, a):
    qrt = rq_t(v1, u1, w, alpha, a).T
    return qrt


def qq_0(u1, v1, w, alpha, a):
    u = np.tile(u1, (1, len(v1)))
    v = np.tile(v1.T, (len(u1), 1))
    a1 = (u-a) * sp.erf((u-a)/(2*w))
    a2 = (v-u) * sp.erf((v-u)/(2*w))
    a3 = (v-a) * sp.erf((v-a)/(2*w))
    a4 = np.exp(-np.power(u-a, 2) / (4*w**2))
    a5 = np.exp(-np.power(v-u, 2) / (4*w**2))
    a6 = np.exp(-np.power(v-a, 2) / (4*w**2))
    qqt = (alpha**-1)*math.pi * (w**2) * (a1 - a2 + a3) + 2*(alpha**-1)*math.sqrt(math.pi)*(w**3)*(a4 - a5 + a6 - 1)
    return qqt


def aux(u, du, r, b, a, cr, ode, pars):
    for i in range(0, b):
        x0 = np.dot(du[i, 0:r+1, :].T, a)
        x = u[i, 0:1, :].T + x0
        # x0 = x + np.sqrt(cr) * np.random.randn(i+1, len(x)).T
        cov = cr * np.identity(len(x))
        u[i, r+1, :] = np.reshape(multivariate_normal.rvs(np.reshape(x, (len(x),)), cov), (1, len(x)))
        # x1 = u[i, r+1:r+2, :].T
        du[i, r+1, :] = ode(u[i, r+1:r+2, :].T, pars)
    return u, du


# def c_t(tao, beta):
#     c0 = np.zeros((len(tao), len(tao)))
#     for i in range(0, len(tao)):
#         for j in range(0, len(tao)):
#             c0[i, j] = cf.secd(beta, tao[i, 0], tao[j, 0])
#
#     return c0
#
#
# def c_0(n, beta, t):
#     c0 = np.zeros((len(t), len(t)))
#     for i in range(0, len(t)):
#         for j in range(0, len(t)):
#             c0[i, j] = cf.sec(n, beta, t[i, 0], t[j, 0])
#
#     return c0
#
#
# def cc_0(tao, tao1, beta, r, n, a):
#     c0 = np.zeros((len(tao1), 1))
#     for i in range(0, len(tao1)):
#         c0[i, 0] = cf.qr(n, beta, tao1[i, 0], tao[r+1, 0], a)
#
#     return c0
#
#
# def cc_t(tao, tao1, n, beta, r):
#     c0 = np.zeros((len(tao1), 1))
#     for i in range(0, len(tao1)):
#         c0[i, 0] = cf.secd(n, beta, tao1[i, 0], tao[r + 1, 0])
#
#     return c0
#
#
# def cc_0n(alpha, beta, tao, t):
#     c0 = np.zeros((len(tao), len(t)))
#     for i in range(0, len(tao)):
#         for j in range(0, len(t)):
#             c0[i, j] = cf.sec(alpha, beta, tao[i, 0], t[j, 0])
#     return c0
#
#

