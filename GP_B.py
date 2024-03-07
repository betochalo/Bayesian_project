import numpy as np
from scipy.linalg import block_diag
from numba import njit
from scipy.stats import multivariate_normal
# from ODEs import lorenzode
import Utilities as oa


@njit
def gp_odes(a, b, n_d, n_k, la, alpha, ode, v0, pars, t):
    n = n_k
    m = len(v0)
    b1 = n_d
    tao = np.reshape(np.linspace(a, b, n), (1, n)).T
    # z = np.zeros((n-1, m))
    fk = ode(v0, pars)
    # u = np.tile(np.append(v0.T, z, axis=0), (b1, 1, 1))
    # Du = np.tile(np.append(ode(v0, pars), z, axis=0), (b1, 1, 1))
    # um = np.tile(np.tile(v0.T, (len(t), 1)), (b1, 1, 1))
    aa = np.zeros((1, 1))
    for r in range(0, n - 1):
        tao1 = np.reshape(tao[0: r + 1, 0], (1, r + 1)).T
        # Compute the inverse matrix (RRd + A)^{-1}
        a1 = oa.rr_t(tao1, tao1, la, alpha)

        a2 = a1 + aa

        br = np.linalg.inv(a2)
        # Compute the vector a_{r} and c_{r}
        a3 = oa.rq_t(tao1, np.array([[tao[r + 1, 0]]]), la, alpha, 0)
        ar = np.dot(br, a3)
        a4 = oa.qr_t(np.array([[tao[r + 1, 0]]]), tao1, la, alpha, 0)
        cr = oa.qq_0(np.array([[tao[r + 1, 0]]]), np.array([[tao[r + 1, 0]]]), la, alpha, 0) - np.dot(a4, ar)
        # Build A_{r+1}
        a5 = oa.rr_t(np.array([[tao[r + 1, 0]]]), np.array([[tao[r + 1, 0]]]), la, alpha)
        a6 = oa.rr_t(tao1, np.array([[tao[r + 1, 0]]]), la, alpha)
        cr1 = a5 - np.dot(a6.T, np.dot(br, a6))
        aa = block_diag(aa, cr1)

        # u, Du = oa.aux(u, Du, r, b1, ar, cr, ode, pars)

        mr = v0 + np.dot(fk.T, ar)
        cov = cr * np.identity(len(v0))
        ur = np.reshape(multivariate_normal.rvs(np.reshape(mr, (len(v0),)), cov), (1, len(v0)))
        fk1 = ode(ur.T, pars)
        fk = np.append(fk, fk1, axis=0)

    a7 = oa.rr_t(tao, tao, la, alpha)
    a8 = a7 + aa
    bn = np.linalg.inv(a8)
    a9 = oa.rq_t(tao, t, la, alpha, 0)
    arn = np.dot(bn, a9)
    uni = np.ones((len(t), 1))
    a10 = np.kron(uni, v0.T)
    a11 = np.dot(arn.T, fk)
    M = a10 + a11
    a12 = oa.qq_0(t, t, la, alpha, 0)
    a13 = oa.qr_t(t, tao, la, alpha, 0)
    C = a12 - np.dot(a13, arn)

    # u1 = M[:, 0]
    # u2 = M[:, 1]
    # u1_uni = u1 / np.linalg.norm(u1)
    # u2_uni = u2 / np.linalg.norm(u2)
    #
    # a14 = oa.qr_t(t, tao, la, alpha, 0)
    # a15 = np.dot(a14, bn)
    # a16 = np.dot(a15, fk)
    # x = 1

    return M, C
