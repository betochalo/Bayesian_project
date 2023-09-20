import scipy
import numpy as np
import math
from scipy import special as sp
# from scipy.integrate import quad

# np.linspace(0, 1, dt)

def RBF(xa, xb, sigma):
    sq_norm = -0.5*(1/sigma**2)*scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def secd(beta, t1, t2):
    c = math.sqrt(math.pi) * beta \
        * math.exp(-((t1 - t2)**2)/(4*beta**2))
    return c


def sec(alpha, beta, t1, t2):
    c = (alpha**-1) * math.exp((-(t1-t2)**2)/(2*beta**2))
    return c


def qr(alpha, beta, t1, t2, a):
    q = math.pi * (alpha**-1) * (beta**2) * (sp.erf((t1 - t2)/(2*beta)) + sp.erf((t2-a)/(2*beta)))
    return q


def qq(alpha, beta, t1, t2, a):
    a1 = (t1-a)*sp.erf((t1-a)/(2*beta)) - (t2-t1)*sp.erf((t2-t1)/(2*beta)) + (t2-a)*sp.erf((t2-a)/(2*beta))
    a2 = math.exp(-((t1-a)**2)/(4*beta**2)) - math.exp(-((t2-t1)**2)/(4*beta**2)) + math.exp(-((t2-a)**2)/(4*beta**2))-1
    q = math.pi * (alpha**-1) * (beta**2) * a1 + 2*math.sqrt(math.pi)*(alpha**-1)*(beta**3)*a2
    return q


