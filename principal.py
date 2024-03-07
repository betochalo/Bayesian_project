import numpy as np
# from scipy.linalg import block_diag
# import covariance_function as cf
# from scipy.stats import multivariate_normal
# import Op_Aux as oa
from ODEs import lorenzode, lotka_volterra
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import GP_B as gp
from matplotlib import pyplot as plt

# Parameters
a = 0
b = 10
N = 1000
ND = 1000
la = 2*(b-a)/(N-1)
alpha = N

# Initial_values
# u0 = np.array([[-12, 15, 38]]).T
u0 = np.array([[10, 10]]).T
# p = np.array([[10, 8/3, 28]]).T
p = np.array([[1.0, 0.1, 0.075, 1.5]]).T
ode = lorenzode
ode1 = lotka_volterra

t = np.reshape(np.linspace(a, b, ND), (1, ND)).T
t1 = np.linspace(a, b, ND)

M, C = gp.gp_odes(a, b, ND, N, la, alpha, ode1, u0, p, t)

# samples = np.random.multivariate_normal(np.zeros(ND), C, size=(2, 2))
# Samples

u1 = M[:, 0]
u2 = M[:, 1]
# u3 = M[:, 2]

# x, y, z = lo()
x, y, t2 = lv()
# u1_uni = u1/np.linalg.norm(u1)
# u2_uni = u2/np.linalg.norm(u2)
#
#
#
# s1 = np.random.multivariate_normal(u1, C, size=100)
# s2 = np.random.multivariate_normal(u2, C, size=100)
# # u3_uni = u3/np.linalg.norm(u3)
# plt.plot(t1, u1)
# plt.plot(t1, u2)
# for i in range(0, 100):
#     plt.plot(t1, s1[i, :], color='black')
#     plt.plot(t1, s2[i, :], color='blue')
#
# plt.show()
# fig, axs = plt.subplots(2)
# fig.suptitle('Solutions of Lotka Volterra')
# axs[0].plot(t1, u1)
# axs[0].set(ylabel='$u_{1}(t)$')
# axs[1].plot(t1, u2)
# axs[1].set(ylabel='$u_{2}(t)$')
# axs[2].plot(t1, u3)
# axs[2].set(ylabel='$u_{3}(t)$')
# plt.plot(u1, u2)
# plt.show()
# ax = plt.axes(projection='3d')
# ax.plot3D(u1, u2, u3, 'gray')
# plt.show()
# Metrics
rmse_x = np.sqrt(mean_squared_error(x, u1))
rmse_y = np.sqrt(mean_squared_error(y, u2))
# rmse_z = np.sqrt(mean_squared_error(z, u2))
print("RMSE X:", rmse_x)
print("RMSE Y:", rmse_y)
# print("RMSE Z:", rmse_z)
# SMSE
mse_x = mean_squared_error(x, u1)
mse_y = mean_squared_error(y, u2)
# mse_z = mean_squared_error(z, u3)
var_x = np.var(x)
var_y = np.var(y)
# var_z = np.var(z)
smse_x = mse_x / var_x
smse_y = mse_y / var_y
# smse_z = mse_z / var_z
print("SMSE X:", smse_x)
print("SMSE Y:", smse_y)
# print("SMSE Z:", smse_z)
# SMLL
log_pred_x = np.abs(np.log(np.abs(u1)))
log_pred_y = np.abs(np.log(np.abs(u2)))
# log_pred_z = np.abs(np.log(np.abs(u3)))
log_x = np.abs(np.log(np.abs(x)))
log_y = np.abs(np.log(np.abs(y)))
# log_z = np.abs(np.log(np.abs(z)))
msll_x = mean_squared_log_error(log_x, log_pred_x)
msll_y = mean_squared_log_error(log_y, log_pred_y)
# msll_z = mean_squared_log_error(log_z, log_pred_z)
print("MSLL X:", msll_x)
print("MSLL Y:", msll_y)
# print("MSLL Z:", msll_z)
# Figures
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(u1, u2, u3, lw=0.5, label="PS+GP")
# ax.plot(x, y, z, lw=0.5, label="Numerical-RK45", color='r')
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Comparison of Two Lorenz Attractor Solutions")
# ax.legend()
# plt.show()
# fig.savefig("lorenz_attractor_comparison.png", dpi=300)
# Figures
plt.plot(t1, u1, label="Prey (PS+GP)")
plt.plot(t1, u2, label="Predator (PS+GP)")

plt.plot(t2, x, label="Prey (RK45)")
plt.plot(t2, y, label="Predator (RK45)")
plt.title('Comparison of two Predator-Prey solutions')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.savefig("LV_MODEL_comparison.png", dpi=300)
plt.show()
