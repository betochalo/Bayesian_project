import numpy as np


# Funciones
# x[0] = S
# x[1] = W
# p[0] = alpha, p[1] = beta, p[2] = gamma, p[3] = delta

def lv(x, p):
    ds = x[0] * (p[0] - p[1] * x[1])
    dw = - x[1] * (p[2] - p[3] * x[0])

    return np.array([ds, dw])


# Parametros iniciales
x0 = np.array([5, 3])
p = np.array([2, 1, 4, 1])
x = lv(x0, p)
print(p)

