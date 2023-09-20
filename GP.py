import numpy as np
import matplotlib.pyplot as plt


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


class GaussianProcess:
    def __init__(self, X, y, noise_var=1e-4):
        self.X = X
        self.y = y
        self.noise_var = noise_var
        self.K = kernel(X, X) + noise_var * np.eye(len(X))
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_star):
        K_star = kernel(self.X, X_star)
        K_star_star = kernel(X_star, X_star)
        mean = np.dot(K_star.T, np.dot(self.K_inv, self.y))
        cov = K_star_star - np.dot(K_star.T, np.dot(self.K_inv, K_star))
        return mean, cov


