import numpy as np


# Integration Time and Sample Time
T = 2

dt = 0.01
truetime = np.arange(0, T+dt, dt)
Tindex = len(truetime)


Tdt = 0.2
samptime = np.arange(0, T+Tdt, Tdt)
TTT = len(samptime)

aux1 = samptime/dt
aux2 = np.ones(np.shape(samptime))
a = aux1 + aux2

itrue =np.int_(np.around(a))

# Ground truth generation by numerical integration

