import numpy as np

# Aquí se va codificar varios ejemplos


def lorenzode(u, p):
    f1 = -p[0, 0] * u[0, 0] + p[0, 0] * u[1, 0]
    f2 = p[2, 0] * u[0, 0] - u[1, 0] - u[0, 0] * u[2, 0]
    f3 = -p[1, 0] * u[2, 0] + u[0, 0] * u[1, 0]
    f = np.array([[f1, f2, f3]])
    return f


def odeex1(u, p):
    r = u
    p = 0
    r[0, 0] = u[0, 0]
    r[1, 0] = u[0, 0] - u[1, 0]

    return r


def lotka_volterra(u, p):
    f1 = p[0, 0] * u[0, 0] - p[1, 0] * u[0, 0] * u[1, 0]
    f2 = p[2, 0] * u[0, 0] * u[1, 0] - p[3, 0] * u[1, 0]
    f = np.array([[f1, f2]])
    return f

