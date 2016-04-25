from __future__ import division  # Force to return a float in division

import numpy as np
import pandas as pd
import math
from math import exp, sqrt, log

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Debugging
from IPython import embed
import sys


def asian_option_simulator(S0, T, r, sigma, M, I, k):
    """
    :param S0: initial stock value
    :param T: maturity
    :param r: risk-free rate
    :param sigma: volatility
    :param M: no. of time steps
    :param I: no. of MC simulations (S paths)
    :param k: discrete sampling freq
    :return: dictionary with paths (S_join), cont. and disc. arithmetic avg's of S (A_c_join, A_d_join)
    and cont. geo avg. (G_c)
    """
    dt = T / M  # time step size

    # Define time index for dataframe
    t_index = np.zeros((M + 1), dtype=np.float)  # fill t_index object with zeros
    t_index[0] = 0

    # Define S_plus and S_minus to apply antithetic variance reduction technique
    # warning: must ensure S_plus and S_minus are different objects otherwise memory issues causes df's to be identical
    shape = (M + 1, I)
    S_plus = np.zeros((M + 1, I), dtype=np.float)  # fill S object with zeros
    S_plus[0] = S0
    S_minus = S_plus.copy()

    # Define continuous and discrete arithmetic average numpy arrays
    A_c_plus = np.zeros((M + 1, I), dtype=np.float)  # continuous
    A_c_plus[0] = S0
    A_c_minus = A_c_plus.copy()
    # ----
    A_d_plus = A_c_plus.copy()  # discrete
    A_d_minus = A_d_plus.copy()
    sel = list(range(0, M + 1, k))  # discrete indices based on sampling freq

    # Define continuous geometric average numpy arrays
    G_c_plus = A_c_plus.copy()
    G_c_plus[0] = math.log(S0)
    G_c_minus = G_c_plus.copy()

    # Define random number generator, in this case only one 'phi' required
    np.random.seed(1000)  # makes tne random numbers predictable (if commented diff will be generated every time)
    rand1 = np.random.standard_normal(shape)

    # Numpy array loop
    for i in xrange(1, M + 1, 1):

        t_index[i] = t_index[i - 1] + dt

        S_plus[i] = S_plus[i - 1] * (1 +
                                     r * dt +
                                     sigma * math.sqrt(dt) * rand1[i] +
                                     0.5 * sigma ** 2 * (rand1[i] ** 2 - 1) * dt
                                     )

        S_minus[i] = S_minus[i - 1] * (1 +
                                       r * dt +
                                       sigma * math.sqrt(dt) * (-rand1[i]) +
                                       0.5 * sigma ** 2 * ((-rand1[i]) ** 2 - 1) * dt
                                       )

        # MC 'continuous' average using updating rule (just for demonstration purposes since df.mean more efficient)
        A_c_plus[i] = (i / (i + 1)) * A_c_plus[i - 1] + S_plus[i] / (i + 1)
        A_c_minus[i] = (i / (i + 1)) * A_c_minus[i - 1] + S_minus[i] / (i + 1)
        G_c_plus[i] = (i / (i + 1)) * G_c_plus[i - 1] + np.log(S_plus[i]) / (i + 1)
        G_c_minus[i] = (i / (i + 1)) * G_c_minus[i - 1] + np.log(S_minus[i]) / (i + 1)

        # MC 'discrete' average
        if i in sel:
            j = int(i / k)
            A_d_plus[i] = (j / (j + 1)) * A_d_plus[i - 1] + S_plus[i] / (j + 1)
            A_d_minus[i] = (j / (j + 1)) * A_d_minus[i - 1] + S_minus[i] / (j + 1)
        else:  # if not part of the sampling, then just copy previous value to have same dim as other arrays
            A_d_plus[i] = A_d_plus[i - 1]
            A_d_minus[i] = A_d_minus[i - 1]

    # Join plus and minus stats (antithetic correction)
    S_join = np.concatenate((S_plus, S_minus), axis=1)
    A_c_join = np.concatenate((A_c_plus, A_c_minus), axis=1)
    A_d_join = np.concatenate((A_d_plus, A_d_minus), axis=1)
    G_c_join = np.concatenate((G_c_plus, G_c_minus), axis=1)
    G_c_join = np.exp(G_c_join)

    dic = {'S_join': S_join,
           'S_plus': S_plus,
           'S_minus': S_minus,
           'A_c_join': A_c_join,
           'A_d_join': A_d_join,
           'G_c_join': G_c_join,
           'DF': math.exp(-r * T) # discount factor (can assume constant since r constant)
           }
    return dic


dic = asian_option_simulator(S0=100., T=1.0, r=0.05, sigma=0.2, M=100, I=100, k=8)

S_join = dic['S_join']
S_plus = dic['S_plus']
A_c_join = dic['A_c_join']
A_d_join = dic['A_d_join']
G_c_join = dic['G_c_join']
DF = dic['DF']

# Additional parameters for option value calc
K = 100.  # strike price

# ---------------------------------------------------------------------------------
# EUROPEAN CALL - using antithetic variance reduction
# ---------------------------------------------------------------------------------

V_join = DF * np.maximum(S_join - K, 0)
V = np.mean(V_join[-1])
print 'EU V(T) = {0}'.format(V)

# V vs S plot
# sns.regplot(x=S_join, y=V_join, fit_reg=False)
V_plus = DF * np.maximum(S_plus[-1] - K, 0) # just for demonstration purposes
# sns.regplot(x=S_plus[-1], y=V_plus, fit_reg=False, color='yellow', marker='+')

# ---------------------------------------------------------------------------------
# ASIAN CALL - using antithetic variance reduction
# ---------------------------------------------------------------------------------

# ----------------------------------------
# ARITHMETIC
# ----------------------------------------
print '________ ARITHMETIC ________'

# -----------------------
# Continuous sampling
# -----------------------
C_c_join = DF * np.maximum(A_c_join - K, 0)
C_c = np.mean(C_c_join[-1])
print 'Asian C_c(T) = {0}'.format(C_c)
# # V vs S plot
# sns.regplot(x=S_join, y=C_c_join, fit_reg=False)

# -----------------------
# Discrete sampling
# -----------------------
C_d_join = DF * np.maximum(A_d_join - K, 0)
C_d = np.mean(C_d_join[-1])
print 'Asian C_d(T) = {0}'.format(C_d)

# -----------------------
# Continuous vs Discrete
# -----------------------

# # S plot for MC, cont. and discrete avg
# plt.plot(S_join[0:,0:1], label='S_sim1')
# plt.plot(A_c_join[0:,0:1], label='Cont. Avg.')
# plt.plot(A_d_join[0:,0:1], label='Disc. Avg.')
# plt.legend()

# Evolution of Asian Call value with number of time steps
c = C_c_join.mean(axis=1)
d = C_d_join.mean(axis=1)
# plt.plot(c)
# plt.plot(d)

# ----------------------------------------
# GEOMETRIC
# ----------------------------------------
print '________ GEOMETRIC ________'

# -----------------------
# Continuous sampling
# -----------------------
C_c_join = DF * np.maximum(G_c_join - K, 0)
C_c = np.mean(C_c_join[-1])
print 'Asian C_c(T) = {0}'.format(C_c)


