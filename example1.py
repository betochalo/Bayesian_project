import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
"""
The Wright-Fisher model is a classic population genetics 
model that describes the evolution of allele 
frequencies in a population over time.
"""

# Define the SDE


def drift(x, t, N):
    return x * (1 - x) / N


def diffusion(x, t, N):
    return np.sqrt(np.maximum(x * (1 - x) / N, 0))

# Euler-Maruyama methods for SDE simulation


def euler_maruyama(drift, diffusion, x0, t0, t1, dt, N):
    num_steps = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, num_steps + 1)
    x = np.zeros(num_steps + 1)
    x[0] = x0

    for i in range(num_steps):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        x[i + 1] = x[i] + drift(x[i], t[i], N) * dt + diffusion(x[i], t[i], N) * dW

    return t, x

# Generate synthetic SDE data


t0 = 0
t1 = 1
dt = 0.01
x0 = 0.5
N = 100 # Population size
t, x = euler_maruyama(drift, diffusion, x0, t0, t1, dt, N)
# Remove NaN values
t = t[~np.isnan(x)]
x = x[~np.isnan(x)]
# Fit Gaussian Process to the SDE data
X = t.reshape(-1, 1)
y = x.reshape(-1, 1)
kernel = RBF(length_scale_bounds=(1e-6, np.inf))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)

# Predict using Gaussian Process
t_pred = np.linspace(t0, t1, 100)
X_pred = t_pred.reshape(-1, 1)
y_pred, y_std = gp.predict(X_pred, return_std=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, x, 'b-', label=r'SDE data: $dX_{t} = X_{t}(1-X_{t})dt + \sqrt{X_{t}(1-X_{t})}dW_{t}$')
plt.plot(t_pred, y_pred, 'r-', label='GP prediction', linewidth=0.9)
# plt.fill_between(t_pred.flatten(), (y_pred - y_std).flatten(), (y_pred + y_std).flatten(), color='tab:orange', alpha=0.5,
#                  label=r"95% confidence interval")
plt.fill_between(
    t_pred.ravel(),
    y_pred - 1.96 * y_std,
    y_pred + 1.96 * y_std,
    color="tab:orange",
    alpha=0.5,
    label=r" 95% confidence interval",
)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Gaussian Process Regression for the Wright-Fisher model')
plt.legend()
plt.grid(True)
plt.show()
