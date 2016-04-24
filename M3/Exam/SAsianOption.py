from __future__ import division  # Force to return a float in division

import numpy as np
import pandas as pd
from IPython import embed
import math
from math import exp, sqrt, log

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import sys

# Parameters

S0 = 100.  # initial value
K = 100.  # strike price
T = 1.0  # maturity
r = 0.05  # risk-free interest rate (replaces mu in formulae)
sigma = 0.2  # volatility
M = 100  # number of time steps
I = 100  # number of paths to simulate

dt = T / M  # time step size
DF = math.exp(-r * T)  # discount factor (can assume constant since r constant)

# Define S_plus and S_minus to apply antithetic variance reduction technique
# warning: must ensure S_plus and S_minus are different objects otherwise memory issues causes df's to be identical
shape = (M + 1, I)
S_plus = np.zeros((M + 1, I), dtype=np.float)  # fill S object with zeros
S_plus[0] = S0
S_minus = np.zeros((M + 1, I), dtype=np.float)
S_minus[0] = S0

# Define time index for dataframe
t_index = np.zeros((M + 1), dtype=np.float)  # fill t_index object with zeros
t_index[0] = 0

# Define Continuous Average numpy array
A_c = np.zeros((M + 1, I), dtype=np.float)
A_c[0] = S0

# Define Discrete Average
A_d = A_c.copy()
k = 8  # sampling frequency
sel = list(range(0, M + 1, k))

# Set up the random number generator, in this case only RN is required
np.random.seed(1000)  # makes tne random numbers predictable (if commented diff RNs will be generated every time
rand1 = np.random.standard_normal(shape)

# Numpy array Loop
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

    # MC 'continuous' average (using updating rule)
    A_c[i] = (i / (i + 1)) * A_c[i - 1] + S_plus[i] / (i + 1)

    # MC 'discrete' average
    if i in sel:
        j = int(i/k)
        A_d[i] = (j / (j + 1)) * A_d[i - 1] + S_plus[i] / (j + 1)
    else:  # if not part of the sampling, then just copy previous value to have same dim as other arrays
        A_d[i] = A_d[i - 1]


# Define column names, starting from sim1
colNames = ['sim' + str(x + 1) for x in range(I)]

# Put numpy arrays into dataframes

df_S_plus = pd.DataFrame(data=S_plus, index=t_index, columns=colNames)
# Averages related to S_plus paths (those for S_minus are ommitted)
df_A_c = pd.DataFrame(data=A_c, index=t_index, columns=colNames)
df_A_d = pd.DataFrame(data=A_d, index=t_index, columns=colNames)

df_S_minus = pd.DataFrame(data=S_minus, index=t_index, columns=colNames)

# ---------------------------------------------------------------------------------
# EUROPEAN CALL - using antithetic variance reduction
# ---------------------------------------------------------------------------------

V_plus = DF * np.maximum(S_plus[-1] - K, 0)
# mean equivalent to np.sum(np.maximum(S_plus[-1] - K, 0)) / I
V_minus = DF * np.maximum(S_minus[-1] - K, 0)
V = (np.mean(V_plus) + np.mean(V_minus)) / 2.0  # antithetic correction
print 'EU V(T) = {0}'.format(V)
# # V vs S plot
# V_join = np.append(V_plus, V_minus)
# S_join = np.append(S_plus[-1], S_minus[-1])
# sns.regplot(x=S_join, y=V_join, fit_reg=False)
# sns.regplot(x=S_plus[-1], y=V_plus, fit_reg=False, color='red', marker='+')

# ---------------------------------------------------------------------------------
# ASIAN CALL - using antithetic variance reduction
# ---------------------------------------------------------------------------------

# -----------------------
# Continuous sampling
# -----------------------
# C_plus = DF * np.mean(np.maximum(df_S_plus.mean() - K, 0))
# C_minus = DF * np.mean(np.maximum(df_S_minus.mean() - K, 0))
# C = (C_plus + C_minus) / 2.0  # antithetic correction
C_plus = DF * np.maximum(A_c[-1] - K, 0)
C_minus = np.array(DF * (np.maximum(df_S_minus.mean() - K, 0)))
C_join = np.append(C_plus, C_minus)
C = (np.mean(C_plus) + np.mean(C_minus)) / 2.0  # antithetic correction
print 'Asian C(T) = {0}'.format(C)
# # V vs S plot
# sns.regplot(x=S_join, y=C_join, fit_reg=False)

# Evolution of Asian Call value with number of time steps (only for plotting)
b = np.maximum(df_A_c - K, 0)
df_C_c = DF * b.mean(axis=1)  # average across columns instead of rows
b = np.maximum(df_A_d - K, 0)
df_C_d = DF * b.mean(axis=1)  # average across columns instead of rows
# df_C_c.plot()
# df_C_d.plot()

# -----------------------
# Continuous vs Discrete Sampling
# -----------------------
df_S_plus.sim1.plot(label='S_plus_sim1')
df_A_c.sim1.plot(label='Cont. Avg.')
df_A_d.sim1.plot(label='Disc. Avg.')
plt.legend()



# -----------------------
# Discrete sampling
# -----------------------
# sel = list(range(0, M + 1, 4))
# df_S_plus_d = df_S_plus.reset_index()
# df_S_plus_d = df_S_plus_d[df_S_plus_d.index.map(lambda x: x in sel)]
# df_S_plus_d = df_S_plus_d.drop('index', 1)


