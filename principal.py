import numpy as np
# from scipy.linalg import block_diag
# import covariance_function as cf
# from scipy.stats import multivariate_normal
# import Op_Aux as oa
from ODEs import lorenzode, lotka_volterra
import GP_B as gp
from matplotlib import pyplot as plt

# Parameters
a = 0
b = 20
N = 1001
ND = 500
la = 2*(b-a)/(N-1)
alpha = N

# Initial_values
# u0 = np.array([[-12, -5, 38]]).T
u0 = np.array([[10, 10]]).T
# p = np.array([[10, 8/3, 28]]).T
p = np.array([[1.1, 0.4, 0.1, 0.4]]).T
ode = lorenzode
ode1 = lotka_volterra

t = np.reshape(np.linspace(a, b, ND), (1, ND)).T
t1 = np.linspace(a, b, ND)

M, C, tao, bn, fk = gp.gp_odes(a, b, ND, N, la, alpha, ode1, u0, p, t)

# samples = np.random.multivariate_normal(np.zeros(ND), C, size=(2, 2))
# Samples

u1 = M[:, 0]
u2 = M[:, 1]

# u3 = M[:, 2]


u1_uni = u1/np.linalg.norm(u1)
u2_uni = u2/np.linalg.norm(u2)



s1 = np.random.multivariate_normal(u1, C, size=100)
s2 = np.random.multivariate_normal(u2, C, size=100)
# u3_uni = u3/np.linalg.norm(u3)
plt.plot(t1, u1)
plt.plot(t1, u2)
for i in range(0, 100):
    plt.plot(t1, s1[i, :], color='black')
    plt.plot(t1, s2[i, :], color='blue')

plt.show()
# fig, axs = plt.subplots(2)
# fig.suptitle('Solutions of Lotka Volterra')
# axs[0].plot(t1, u1)
# axs[0].set(ylabel='$u_{1}(t)$')
# axs[1].plot(t1, u2)
# axs[1].set(ylabel='$u_{2}(t)$')
# axs[2].plot(t1, u3)
# axs[2].set(ylabel='$u_{3}(t)$')
plt.plot(u1, u2)
plt.show()
# ax = plt.axes(projection='3d')
# ax.plot3D(u1, u2, u3, 'gray')
# plt.show()
