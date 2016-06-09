from __future__ import division  # Force to return a float in division

# Vectorisation
import numpy as np
import pandas as pd

# Math functions
import math
from math import exp, sqrt, log

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Debugging
from IPython import embed
import sys

#############################################################
#
#  This script does N MC simulations of the HJM model in
#  M4/Exam/docs/HJM Model - MC - Caplet v2.xlsm
#
#############################################################

# def asian_option_simulator(S0, K, T, r, sigma, M, I, k, mode='fixed'):

"""
:param S0: initial stock value
:param K: strike
:param T: maturity
:param r: risk-free rate
:param sigma: volatility
:param M: no. of time steps
:param I: no. of MC simulations (S paths)
:param k: discrete sampling freq
:param mode: fixed or floating strike used in payoff function (default is fixed)
:return: dictionary with paths (S_join), cont. and disc. arithmetic avg's of S (A_c_join, A_d_join)
and cont. geo avg. (G_c)
"""

df = pd.read_csv("input.csv", index_col=0)

T = 10.0  # t_max
M = 1000  # no. of time steps
I = 100  # no. of MC simulations (S paths)
n_tau = 51  # no. of tenors

# Derived params
dt = T / M  # time step size
shape_3D = (M + 1, I, n_tau)
# shape_2D = (M + 1, I)
# shape_3D = (M + 1, n_tau, I)
# shape_2D = (M + 1, n_tau)
# shape_MC = (M + 1)
shape_MC = (M + 1, I)

# Take drift, vol1, vol2, vol4 and r_t0_T values from CQF spreadsheet
mu = np.array(df.iloc[0, :])
vol1 = np.array(df.iloc[1, :])
vol2 = np.array(df.iloc[2, :])
vol3 = np.array(df.iloc[3, :])
S0 = np.array(df.iloc[4, :])

# Calculate the diffs for the tenor and df and append last value to get correct vector dimension
d_tau = np.diff(np.array(df.columns, dtype=np.float))
d_tau = np.append(d_tau, d_tau[-1])
d_S_plus = np.diff(S0)
d_S_plus = np.append(d_S_plus, d_S_plus[-1])

# Define matrix equivalents
# warning: must ensure X_plus and X_minus are different objects otherwise memory issues causes df's to be identical
# notation: [t, N, tau]

# Constant

# mu_m = np.zeros(shape_2D, dtype=np.float)
# vol1_m = np.zeros(shape_2D, dtype=np.float)
# vol2_m = np.zeros(shape_2D, dtype=np.float)
# vol3_m = np.zeros(shape_2D, dtype=np.float)
# d_tau_m = np.zeros(shape_2D, dtype=np.float)

mu_m = np.zeros(shape_3D, dtype=np.float)
vol1_m = np.zeros(shape_3D, dtype=np.float)
vol2_m = np.zeros(shape_3D, dtype=np.float)
vol3_m = np.zeros(shape_3D, dtype=np.float)
d_tau_m = np.zeros(shape_3D, dtype=np.float)

# Dynamic

# d_S_plus_m = np.zeros(shape_2D, dtype=np.float)
# t_index = np.zeros((M + 1), dtype=np.float)  # fill t_index object with zeros
# S_plus_m = np.zeros(shape_2D, dtype=np.float)
# A_c_plus_m = np.zeros(shape_2D, dtype=np.float)  # arithmentic continuous average

d_S_plus_m = np.zeros(shape_3D, dtype=np.float)
t_index = np.zeros((M + 1), dtype=np.float)  # fill t_index object with zeros
S_plus_m = np.zeros(shape_3D, dtype=np.float)
A_c_plus_m = np.zeros(shape_3D, dtype=np.float)  # arithmentic continuous average


# Initialise row=0 of matrices since loop starts at row=1

# mu_m[0] = mu
# vol1_m[0] = vol1
# vol2_m[0] = vol2
# vol3_m[0] = vol3
# d_tau_m[0] = d_tau
# d_S_plus_m[0] = d_S_plus
# t_index[0] = 0
# S_plus_m[0] = S0
# A_c_plus_m[0] = S0  # set first value

mu_m[0][:] = mu
vol1_m[0][:] = vol1
vol2_m[0][:] = vol2
vol3_m[0][:] = vol3
d_tau_m[0][:] = d_tau
d_S_plus_m[0][:] = d_S_plus
t_index[0] = 0
S_plus_m[0][:] = S0
A_c_plus_m[0][:] = S0  # set first value

# plus and minus for S and A to apply antithetic variance reduction technique
S_minus_m = S_plus_m.copy()
A_c_minus_m = A_c_plus_m.copy()
d_S_minus_m = d_S_plus_m.copy()

# Define random number generator, in this case 3 different (phi's) are required
np.random.seed(1000)  # makes tne random numbers predictable (if commented diff will be generated every time)

# rand1 = np.random.standard_normal(shape_MC)  # this creates a numpy array of RNs to feed S array for a given time step
# rand2 = np.random.standard_normal(shape_MC)
# rand3 = np.random.standard_normal(shape_MC)

# rand1_m = np.zeros(shape_3D, dtype=np.float)
# rand2_m = np.zeros(shape_3D, dtype=np.float)
# rand3_m = np.zeros(shape_3D, dtype=np.float)
# rand1_m[0][:] = np.random.standard_normal()
# rand2_m[0][:] = np.random.standard_normal()
# rand3_m[0][:] = np.random.standard_normal()

# rand_m = np.zeros((M + 1, 3, I), dtype=np.float)
# rand_m = np.random.standard_normal((M + 1, 3))

# Numpy array loop - start from 2nd row since we have set initial values
for i in xrange(1, M + 1, 1):

    # # Constant
    # mu_m[i] = mu
    # vol1_m[i] = vol1
    # vol2_m[i] = vol2
    # vol3_m[i] = vol3
    # d_tau_m[i] = d_tau

    # Constant
    mu_m[i][:] = mu
    vol1_m[i][:] = vol1
    vol2_m[i][:] = vol2
    vol3_m[i][:] = vol3
    d_tau_m[i][:] = d_tau

    # rand
    rand1 = np.random.standard_normal((I, 1))
    rand2 = np.random.standard_normal((I, 1))
    rand3 = np.random.standard_normal((I, 1))

    # # Dynamic
    # temp = np.diff(S_plus_m[i-1])
    # d_S_plus_m[i] = np.append(temp, temp[-1])

    # Dynamic
    temp = np.diff(S_plus_m[i-1][:])
    d_S_plus_m[i][:] = np.hstack((temp, temp[:, -1].reshape(I, 1)))
    temp = np.diff(S_minus_m[i-1][:])
    d_S_minus_m[i][:] = np.hstack((temp, temp[:, -1].reshape(I, 1)))

    # time step
    t_index[i] = t_index[i - 1] + dt

    # Generate S paths using antithetic reduction
    S_plus_m[i][:] = S_plus_m[i - 1][:] + \
                  mu_m[i][:] * dt +\
                  (vol1_m[i][:] * rand1 + vol2_m[i][:] * rand2 + vol3_m[i][:] * rand3) * math.sqrt(dt) + \
                  (d_S_plus_m[i][:] / d_tau_m[i][:]) * dt
    
    S_minus_m[i][:] = S_minus_m[i - 1][:] + \
                     mu_m[i][:] * dt + \
                     (vol1_m[i][:] * (-rand1) + vol2_m[i][:] * (-rand2) + vol3_m[i][:] * (-rand3)) * math.sqrt(dt) + \
                     (d_S_minus_m[i][:] / d_tau_m[i][:]) * dt
    
    # MC 'continuous' average using updating rule (just for demonstration purposes since df.mean more efficient)
    A_c_plus_m[i][:] = S_plus_m[i][:]
    # A_c_plus_m[i][:] = (i / (i + 1)) * A_c_plus_m[i - 1][:] + S_plus_m[i][:] / (i + 1)
    # A_c_plus[i] = (i / (i + 1)) * A_c_plus[i - 1] + S_plus[i] / (i + 1)
    # A_c_minus[i] = (i / (i + 1)) * A_c_minus[i - 1] + S_minus[i] / (i + 1)

############################################################################################

# Plotting
# notation: [t, N, tau]

# ------ Simulated Forward Curves -------

# # Today, 2 simulations
# plt.plot(S_plus_m[0, 0, :])  # notation: [t, N, tau]
# plt.plot(S_plus_m[0, 1, :])
# # 1Y, 2 simulations
# plt.plot(S_plus_m[101, 0, :])
# plt.plot(S_plus_m[101, 1, :])
# # 5Y, 2 simulations
# plt.plot(S_plus_m[501, 0, :])
# plt.plot(S_plus_m[501, 1, :])
#
# # ------ Projection of Forward Rate -------
#
# plt.plot(S_plus_m[:, 0, 0])  # today, notation: [t, N, tau]
# plt.plot(S_plus_m[:, 1, 0])  # today
# plt.plot(S_plus_m[:, 0, 11])  # 5Y, notation: [t, N, tau]
# plt.plot(S_plus_m[:, 1, 11])  # 5Y


############################################################################################

# Join plus and minus stats (antithetic technique)
S_join_m = np.concatenate((S_plus_m, S_minus_m), axis=1)  # shape (1001L, 4L, 51L)
# A_c_join = np.concatenate((A_c_plus, A_c_minus), axis=1)

############################################################################################


# ----------- PRICING
a = np.mean(S_plus_m, axis=1)
b = np.mean(S_join_m, axis=1)

# Rolling mean

x = np.zeros(I, dtype=np.float)
# y = np.zeros(shape_3D, dtype=np.float)

for i in xrange(1, I):
    x[i] = i
    # A_c_plus_m[1, i, 0] = (i / (i + 1)) * A_c_plus_m[1, i - 1, 0] + S_plus_m[1, i, 0] / (i + 1)  # for 1 tenor and 1 time only
    # A_c_plus_m[:, i, 0] = (i / (i + 1)) * A_c_plus_m[:, i - 1, 0] + S_plus_m[:, i, 0] / (i + 1)  # for 1 tenor only
    A_c_plus_m[:, i, :] = (i / (i + 1)) * A_c_plus_m[:, i - 1, :] + S_plus_m[:, i, :] / (i + 1)  # for all tenors and all times

sys.exit()

# plt.plot(x, A_c_plus_m[1, :, 0], 'o')
# df_plus = pd.DataFrame(index=x, data=A_c_plus_m[1, :, 0], columns=['V'])

# Calculate option value for Asian and EU (to have EU as benchmark)

# ----------------------------------------------------
# EUROPEAN CALL - using antithetic variance reduction
# ----------------------------------------------------
DF = math.exp(-r * T)
V_join = DF * np.maximum(S_join - K, 0)
V = np.mean(V_join[-1])
V_e = np.std(V_join[-1]) / I  # error std/sqrt(N)

V_plus = DF * np.maximum(S_plus[-1] - K, 0) # just for demonstration purposes

# --------------------------------------------------
# ASIAN CALL - using antithetic variance reduction
# --------------------------------------------------

if mode == 'float':  # convert between fixed and floating strike
    fac = -1.0
    K = S_join[-1]  # stock price at maturity S(T) - see https://en.wikipedia.org/wiki/Asian_option
else:
    fac = 1.0


# ----------------------------------------
# ARITHMETIC
# ----------------------------------------

# Continuous sampling
AC_c_join = DF * np.maximum(fac * A_c_join - fac * K, 0)
AC_c = np.mean(AC_c_join[-1])
AC_c_e = np.std(AC_c_join[-1]) / I  # error std/sqrt(N)

# --------------------------------------------------
# Variables into Dictionary
# --------------------------------------------------

t_index[-1] = 1.0  # ensures plots do not extend the x-axis to account for slight rounding error

dic = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'M': M, 'I': I, 'k': k,  # params
       't_index': t_index,
       'S_join': S_join,
       'A_c_join': A_c_join,
       'A_d_join': A_d_join,
       'G_c_join': G_c_join,
       'V_join': V_join,
       'V': V,
       'V_e': V_e,
       'AC_c_join': AC_c_join,
       'AC_d_join': AC_d_join,
       'GC_c_join': GC_c_join,
       'AC_c': AC_c,
       'AC_c_e': AC_c_e,
       'AC_d': AC_d,
       'AC_d_e': AC_d_e,
       'GC_c': GC_c,
       'GC_c_e': GC_c_e,
       'mode': mode
       }



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

dic = asian_option_simulator(S0=100., K=100., T=1.0, r=0.05, sigma=0.2, M=100, I=100, k=10, mode='fixed')

