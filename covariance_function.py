import scipy
import numpy as np
import math

# np.linspace(0, 1, dt)

def RBF(xa, xb, sigma):
    sq_norm = -0.5*(1/sigma**2)*scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def secd(beta, alpha, t1, t2):
    c = math.sqrt(math.pi) * (alpha**-1) * beta \
        * math.exp((-(t1 - t2)**2)/(4*beta**2))
    return c


def sec(beta, t1, t2):
    c = math.exp((-(t1-t2))**2/(2*beta**2))
    return c


