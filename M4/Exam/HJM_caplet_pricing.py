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
I = 1  # no. of MC simulations (S paths)
n_tau = 51  # no. of tenors
# Take drift, vol1, vol2, vol4 and r_t0_T values from CQF spreadsheet
mu = np.array(df.iloc[0, :])  # drift
vol1 = np.array(df.iloc[1, :])
vol2 = np.array(df.iloc[2, :])
vol3 = np.array(df.iloc[3, :])
S0 = np.array(df.iloc[4, :])  # r_t0_T (forward curve at t=0 for different tenors)

# Derived params
dt = T / M  # time step size
shape = (M + 1, n_tau, I)

# Define time index for dataframe
t_index = np.zeros((M + 1), dtype=np.float)  # fill t_index object with zeros
t_index[0] = 0  # set first value

# Define X_plus and X_minus for S and A to apply antithetic variance reduction technique
# warning: must ensure X_plus and X_minus are different objects otherwise memory issues causes df's to be identical

# Define numpy array for the underlying S
S_plus = np.zeros(shape, dtype=np.float)  # fill S object with zeros
S_plus[0] = S0  # set first value
S_minus = S_plus.copy()

# Define numpy array for the rolling arithmetic average of S
A_c_plus = np.zeros(shape, dtype=np.float)  # continuous
A_c_plus[0] = S0  # set first value
A_c_minus = A_c_plus.copy()

# Define random number generator, in this case 3 different (phi's) are required
np.random.seed(1000)  # makes tne random numbers predictable (if commented diff will be generated every time)
rand1 = np.random.standard_normal(shape)  # this creates a numpy array of RNs to feed S array for a given time step
rand2 = np.random.standard_normal(shape)
rand3 = np.random.standard_normal(shape)

# Numpy array loop - start from 2nd row since we have set initial values
for i in xrange(1, M + 1, 1):

    t_index[i] = t_index[i - 1] + dt

    # Generate S paths using Milstein convention
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

# Join plus and minus stats (antithetic technique)
S_join = np.concatenate((S_plus, S_minus), axis=1)
A_c_join = np.concatenate((A_c_plus, A_c_minus), axis=1)
A_d_join = np.concatenate((A_d_plus, A_d_minus), axis=1)
G_c_join = np.concatenate((G_c_plus, G_c_minus), axis=1)
G_c_join = np.exp(G_c_join)

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

# Discrete sampling
AC_d_join = DF * np.maximum(fac * A_d_join - fac * K, 0)
AC_d = np.mean(AC_d_join[-1])
AC_d_e = np.std(AC_d_join[-1]) / I  # error std/sqrt(N)

# ----------------------------------------
# GEOMETRIC
# ----------------------------------------

# Continuous sampling
GC_c_join = DF * np.maximum(fac * G_c_join - fac * K, 0)
GC_c = np.mean(GC_c_join[-1])
GC_c_e = np.std(GC_c_join[-1]) / I  # error std/sqrt(N)

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

