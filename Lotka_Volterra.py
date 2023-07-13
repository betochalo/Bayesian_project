import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import math


def drift(x, y, t):
    alpha = 1.5
    beta = 0.4
    gamma = 1.5
    delta = 0.4
    return alpha * x - beta * x * y, -gamma * y + delta * x * y - 0.2 * y**2


def diffusion(x, y, t):
    sigma1 = 0.05
    sigma2 = 0.1
    return math.sqrt(sigma1 * x), math.sqrt(sigma2 * x)


def euler_maruyama(drift, diffusion, x0, y0, t0, t1, dt):
    num_steps = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, num_steps + 1)
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    x[0] = x0
    y[0] = y0

    for i in range(num_steps):
        dW_x = np.sqrt(dt) * np.random.normal(0, 1)
        dW_y = np.sqrt(dt) * np.random.normal(0, 1)
        drift1, drift2 = drift(x[i], y[i], t[i])
        diffusion1, diffusion2 = diffusion(x[i], y[i], t[i])
        # dx, dy = drift(x[i], y[i], t[i]), drift(y[i], x[i], t[i])
        x[i + 1] = x[i] + drift1 * dt + diffusion1 * dW_x
        y[i + 1] = y[i] + drift2 * dt + diffusion2 * dW_y

    return t, x, y


# Generate synthetic data for the Lotka-Volterra equations
t0 = 0
t1 = 5
dt = 0.01
x0 = 2
y0 = 1
t, x, y = euler_maruyama(drift, diffusion, x0, y0, t0, t1, dt)
t = t[~np.isnan(x)]
x = x[~np.isnan(x)]
y = y[~np.isnan(x)]
# Fit Gaussian Process to the SDE data
X = t.reshape(-1, 1)
Y = np.column_stack((x, y))
kernel = RBF(length_scale_bounds=(1e-6, np.inf))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X, Y)

# Predict using Gaussian Process
t_pred = np.linspace(t0, t1, len(t))
X_pred = t_pred.reshape(-1, 1)
Y_pred, Y_std = gp.predict(X_pred, return_std=True)
x_pred, y_pred = Y_pred[:, 0], Y_pred[:, 1]
x_std, y_std = Y_std[:, 0], Y_std[:, 1]

# Compute metrics
rmse_x = np.sqrt(mean_squared_error(x, x_pred))
rmse_y = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE X:", rmse_x)
print("RMSE Y:", rmse_y)
# SMSE
mse_x = mean_squared_error(x, x_pred)
mse_y = mean_squared_error(y, y_pred)
var_x = np.var(x)
var_y = np.var(y)
smse_x = mse_x / var_x
smse_y = mse_y / var_y
print("SMSE X:", smse_x)
print("SMSE Y:", smse_y)
# SMLL
log_pred_x = np.abs(np.log(np.abs(x_pred)))
log_pred_y = np.abs(np.log(np.abs(y_pred)))
log_x = np.abs(np.log(np.abs(x)))
log_y = np.abs(np.log(np.abs(y)))
msll_x = mean_squared_log_error(log_x, log_pred_x)
msll_y = mean_squared_log_error(log_y, log_pred_y)
print("MSLL X:", msll_x)
print("MSLL Y:", msll_y)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, x, 'b-', label='Prey (x)')
plt.plot(t, y, 'g-', label='Predator (y)')
plt.plot(t_pred, x_pred, 'r-', label='Prey prediction')
plt.plot(t_pred, y_pred, 'm-', label='Predator prediction')
plt.fill_between(t_pred.flatten(), (x_pred - x_std).flatten(), (x_pred + x_std).flatten(), color='tab:orange', alpha=0.3)
plt.fill_between(t_pred.flatten(), (y_pred - y_std).flatten(), (y_pred + y_std).flatten(), color='tab:orange', alpha=0.3)
plt.xlabel('t')
plt.ylabel('Population')
plt.title('Gaussian Process Regression for Lotka-Volterra Equations')
plt.legend()
plt.grid(True)
plt.show()
