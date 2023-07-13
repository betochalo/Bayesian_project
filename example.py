import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel

# train_data = pd.read_csv('D:/Users/Roberth/Desktop/B_P/data/train.csv')
# test_data = pd.read_csv('D:/Users/Roberth/Desktop/B_P/data/test.csv')
# features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1',
#             '2ndFlrSF', 'GarageArea', '1stFlrSF', 'YearBuilt']
#
# X_train = train_data[features]
# y_train = train_data["SalePrice"]
# final_X_test = test_data[features]
#
# X_train = X_train.fillna(X_train.mean())
# final_X_test = final_X_test.fillna(final_X_test.mean())
#
# kernel = DotProduct() + WhiteKernel()
# regressor = GaussianProcessRegressor(kernel=kernel)
# regressor.fit(X_train, y_train)
#
# mean_prediction, std_prediction = regressor.predict(final_X_test, return_std=True)
#
#
# plt.plot(final_X_test, mean_prediction, label="Mean prediction")
# plt.show()

# x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]).reshape(-1, 1)
# y = np.squeeze(np.array([3.1, 4.4, 5.2, 6.8, 7.9, 8.6, 9.1, 9.5, 9.7, 9.8, 10.5, 11.6, 12.3, 12.9, 13.5, 14.6, 15.1, 16.4, 17.1]))
x = np.linspace(start=0, stop=10, num=1000).reshape(-1, 1)
y = np.squeeze(x**3 * np.cos(x))

# plt.plot(x, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
# plt.legend()
# plt.xlabel("$x$")
# plt.ylabel("$f(x)$")
# plt.title("True generative process")

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=8, replace=False)
x_train, y_train = x[training_indices], y[training_indices]

noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
gaussian_process.fit(x_train, y_train_noisy)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(x, return_std=True)

plt.plot(x, y, label=r"$f(x)=x^{3}\cos(x)$", linestyle="dotted")
plt.errorbar(
    x_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations"
)
plt.plot(x, mean_prediction, label="Mean prediction", color="red")
plt.fill_between(
    x.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian process regression on a noisy dataset")
plt.show()
