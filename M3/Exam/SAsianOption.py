from __future__ import division # Force to return a float in division

import numpy as np
import pandas as pd
from IPython import embed
import math
from math import exp, sqrt, log

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters

S0 = 100.  # initial value
K = 100.  # strike price
T = 1.0  # maturity
r = 0.05  # risk-free interest rate (replaces mu in formulae)
sigma = 0.2  # volatility
M = 100  # number of time steps
I = 100  # number of paths to simulate

dt = T / M  # time step size

# Define S_plus and S_minus to apply antithetic variance reduction technique
# warning: must ensure S_plus and S_minus are different objects otherwise memory issues causes df's to be identical
shape = (M + 1, I)
S_plus = np.zeros((M + 1, I), dtype=np.float)  # fill S object with zeros
S_plus[0] = S0
S_minus = np.zeros((M + 1, I), dtype=np.float)
S_minus[0] = S0

# Define Average numpy array
A = np.zeros((M + 1, I), dtype=np.float)
A[0] = S0

# Define time index for dataframe
t_index = np.zeros((M + 1), dtype=np.float)  # fill t_index object with zeros
t_index[0] = 0

# Define column names, starting from sim1
colNames = ['sim' + str(x + 1) for x in range(I)]

# Set up the random number generator, in this case only RN is required
np.random.seed(1000)  # makes tne random numbers predictable (if commented diff RNs will be generated every time
rand1 = np.random.standard_normal(shape)

for i in xrange(1, M + 1, 1):
    S_plus[i] = S_plus[i - 1] * (1
                                 + r * dt
                                 + sigma * math.sqrt(dt) * rand1[i]
                                 + 0.5 * sigma ** 2 * (rand1[i] ** 2 - 1) * dt
                                 )

    S_minus[i] = S_minus[i - 1] * (1
                                   + r * dt
                                   + sigma * math.sqrt(dt) * (-rand1[i])
                                   + 0.5 * sigma ** 2 * ((-rand1[i]) ** 2 - 1) * dt
                                   )
    t_index[i] = t_index[i - 1] + dt
    A[i] = (i / (i + 1)) * A[i - 1] + S_plus[i] / (i + 1)


# Put numpy arrays into dataframes
df_S_plus = pd.DataFrame(data=S_plus, index=t_index, columns=colNames)
df_S_minus = pd.DataFrame(data=S_minus, index=t_index, columns=colNames)
df_A = pd.DataFrame(data=A, index=t_index, columns=colNames)

# Value of European Call Option using antithetic variance reduction technique

C_plus = math.exp(-r * T) * np.mean(np.maximum(S_plus[-1] - K, 0))
# mean equivalent to np.sum(np.maximum(S_plus[-1] - K, 0)) / I
C_minus = math.exp(-r * T) * np.mean(np.maximum(S_minus[-1] - K, 0))
C = (C_plus + C_minus) / 2.0  # antithetic correction
print 'EU C(T) = {0}'.format(C)

# Value of Asian Call Option using antithetic variance reduction technique

# Continuous sampling
AC_plus = math.exp(-r * T) * np.mean(np.maximum(df_S_plus.mean() - K, 0))
AC_minus = math.exp(-r * T) * np.mean(np.maximum(df_S_minus.mean() - K, 0))
AC = (AC_plus + AC_minus) / 2.0  # antithetic correction
print 'Asian C(T) = {0}'.format(AC)

# Time evolution of Asian Call value
b = np.maximum(df_A - K, 0)
df_AC = math.exp(-r * T) * b.mean(axis=1)


# # Discrete sampling
# sel = list(range(0, M + 1, 4))
# df_S_plus_d = df_S_plus.reset_index()
# df_S_plus_d = df_S_plus_d[df_S_plus_d.index.map(lambda x: x in sel)]
# df_S_plus_d = df_S_plus_d.drop('index', 1)


