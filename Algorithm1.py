import numpy as np
from scipy.linalg import block_diag
import covariance_function as cf
from scipy.stats import multivariate_normal
import Op_Aux as oa
import ODEs as ode
# from matplotlib import pyplot as plt

# Parameters
tao0 = 0
taoN = 20
N = 501
beta = 2*(taoN-tao0)/(N-1)
alpha = N
tao = np.reshape(np.linspace(tao0, taoN, N), (1, N)).T


# Initial_values
u0 = np.array([[-12, -5, 38]]).T
p = np.array([[10, 8/3, 28]]).T

# Treatment vector
# x0 = np.array([7.5, 1000])

# Physical parameters
# theta0 = np.array([200, 0.05, 100, 100])


# def function(u):
# #     f1 = u[0, 0]
# #     f2 = u[0, 0] - u[1, 0]
# #     f = np.array([[f1, f2]])
# #     return f


f_k = ode.lorenzode(u0, p)
A = np.zeros((1, 1))
for r in range(0, N-1):
    tao_r = np.reshape(tao[0: r+1, 0], (1, r+1)).T
    a1 = oa.rr_t(tao_r, tao_r, beta,  alpha)
    B_r = np.linalg.inv(a1+A)
    a2 = oa.rq_t(tao_r, np.array([[tao[r+1, 0]]]), beta, alpha, 0)
    a_r = np.dot(B_r, a2)
    a3 = oa.qr_t(tao_r, np.array([[tao[r+1, 0]]]), beta, alpha, 0)
    c_r = oa.qq_0(np.array([[tao[r+1, 0]]]), np.array([[tao[r+1, 0]]]), beta, alpha, 0) - np.dot(a3, np.dot(B_r, a2))
    a4 = oa.rr_t(np.array([[tao[r+1, 0]]]), np.array([[tao[r+1, 0]]]), beta, alpha)
    a5 = oa.rr_t(tao_r, np.array([[tao[r+1, 0]]]), beta, alpha)
    c_r1 = a4 - np.dot(a5.T, np.dot(B_r, a5))
    A = block_diag(A, c_r1)
    m_r = u0 + np.dot(f_k.T, a_r)
    cov = c_r * np.identity(len(u0))
    u_r1 = np.reshape(multivariate_normal.rvs(np.reshape(m_r, (len(u0),)), cov), (1, len(u0)))
    f_k1 = function(u_r1.T)
    f_k = np.append(f_k, f_k1, axis=0)



print(tao_r)
# for r in range(0, N-1):
#     tao_r = np.reshape(tao[0: r+1, 0], (1, r+1)).T
#     c0 = oa.c_t(tao_r, N, beta)
#     B_r = np.linalg.inv(c0 + A)
#     c1 = oa.cc_0(tao, tao_r, beta, r, N, 0)
#     a_r = np.dot(B_r, c1)
#     h = np.dot(c1.T, np.dot(B_r, c1))
#     C_r = np.array([[cf.qq(N, beta, tao_r[r, 0], tao_r[r, 0], 0)]]) - np.dot(c1.T, np.dot(B_r, c1))
#     c2 = oa.cc_t(tao, tao_r, N, beta, r)
#     C_r1 = np.array([[cf.secd(N, beta, tao[r+1, 0], tao[r+1, 0])]]) - np.dot(c2.T, np.dot(B_r, c2))
#     A = block_diag(A, C_r1)
#     m_r = u0 + np.dot(f_k.T, a_r)
#     cov = C_r * np.identity(len(u0))
#     u_r1 = np.reshape(multivariate_normal.rvs(np.reshape(m_r, (len(u0),)), cov), (1, len(u0)))
#     f_k1 = function(u_r1.T)
#     f_k = np.append(f_k, f_k1, axis=0)
#
# # 3 Compute
# t0 = 0
# tw = 100
# N1 = 51
# beta1 = 40 / N1
# t1 = np.linspace(t0, tw, N1)
# t = np.reshape(np.linspace(t0, tw, N1), (1, N1)).T
#
#
# aux = oa.c_t(tao, N, beta)
# B = np.linalg.inv(aux + A)
# aux1 = oa.cc_0n(N, beta, tao, t)
# A1 = np.dot(B, aux1).T
# aux2 = np.dot(A1, f_k)
# uni = np.ones((N1, 1))
# aux3 = np.kron(uni, u0.T)
# M = aux3 + aux2
# aux4 = np.dot(aux1.T, np.dot(B, aux1))
# aux5 = oa.c_0(N1, beta1, t)
# C = aux5 - aux4
# norma1 = np.linalg.norm(M[:, 0])
# e1 = M[:, 0] / norma1
# norma2 = np.linalg.norm(M[:, 1])
# e2 = M[:, 1] / norma2
# M1 = M[:, 0] * e1
# M2 = M[:, 1] * e2
#
# u11 = multivariate_normal.rvs(M1, C)
# u12 = multivariate_normal.rvs(M2, C)

# def es_definida_semipositiva(matriz):
#     autovalores = np.linalg.eigvals(matriz)
#     if all(autovalores >= 0):
#         return True
#     else:
#         return False
# matrix = es_definida_semipositiva(C)
