import numpy as np


a1 = 0

u0 = np.array([[1, 2]])

def function(u):
    f1 = u[0, 0] + u[0, 1]
    f2 = u[0, 0] - u[0, 1]
    f = np.array([[f1, f2]])
    return f
f = function(u0)
tao = np.reshape(np.linspace(0, 20, 100), (1, 100)).T
r = 2
tao1 = np.reshape(tao[0:r+1, 0], (r+1, 1))

print(u0)

