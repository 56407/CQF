from __future__ import division  # Force to return a float in division

# Vectorisation
import numpy as np
import pandas as pd

# Math functions
import math
from math import exp, sqrt, log

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging
from IPython import embed
import sys

#############################################################
#
#  This script does I MC simulations of the HJM model in
#  M4/Exam/docs/HJM Model - MC - Caplet v2.xlsm
#
#############################################################

# def asian_option_simulator(S0, K, T, r, sigma, M, I, k, mode='fixed'):

"""
:param S0: initial forward rate values f(t=0, T)
:param K: strike
:param T: maturity
:param M: no. of time steps
:param I: no. of MC simulations (S paths)
:param mode: fixed or floating strike used in payoff function (default is fixed)
:return:
"""

# Define parameters values
T = 10.0  # t_max or maturity
M = 1000  # no. of time steps
I = 2  # no. of MC simulations (S paths)
n_tau = 50  # no. of tenors

# Derived params
dt = T / M  # time step size
shape_3D = (M + 1, I, n_tau + 1)
# shape_2D = (M + 1, I)

# Take drift, volatilities and f(t=0, T)  from CQF 'HJM Model - MC - Caplet v2.xlsm'
data = pd.read_csv("input.csv", index_col=0)
mu = np.array(data.iloc[0, :])  # drift
vol1 = np.array(data.iloc[1, :])  # volatility of highest eigenvalue e(1)
vol2 = np.array(data.iloc[2, :])  # volatility of 2nd highest eigenvalue e(2)
vol3 = np.array(data.iloc[3, :])  # volatility of 3rd highest eigenvalue e(3)
S0 = np.array(data.iloc[4, :])  # initial forward rate f(t=0, T)

# Calculate the dtau and dF for Musiela correction term dF/dtau in forward rate
# Must append last value again to get correct vector dimension - this means 25Y maturity uses a backward derivative
d_tau = np.diff(np.array(data.columns, dtype=np.float))  # convert from string to float to take diff
d_tau = np.append(d_tau, d_tau[-1])  # this is dtau
d_S_plus = np.diff(S0)
d_S_plus = np.append(d_S_plus, d_S_plus[-1])  # this is dF

# Define numpy arrays to account for I MC simulations
# 3D array shape is [t, I, tau] where t is time, I is number of simulations and tau the tenor

# Constant
mu_m = np.zeros(shape_3D, dtype=np.float)
vol1_m = np.zeros(shape_3D, dtype=np.float)
vol2_m = np.zeros(shape_3D, dtype=np.float)
vol3_m = np.zeros(shape_3D, dtype=np.float)
d_tau_m = np.zeros(shape_3D, dtype=np.float)

# Dynamic
t_index = np.zeros((M + 1), dtype=np.float)  # note this is a 2D array
d_S_plus_m = np.zeros(shape_3D, dtype=np.float)
S_plus_m = np.zeros(shape_3D, dtype=np.float)
# A_c_plus_m = np.zeros(shape_3D, dtype=np.float)  # arithmetic continuous average

# Set first value of arrays (row=0, t=0) since loop starts at row=1
mu_m[0][:] = mu
vol1_m[0][:] = vol1
vol2_m[0][:] = vol2
vol3_m[0][:] = vol3
d_tau_m[0][:] = d_tau
d_S_plus_m[0][:] = d_S_plus
t_index[0] = 0
S_plus_m[0][:] = S0
# A_c_plus_m[0][:] = S0  # same as S_plus_m

# Create 'minus' array for S and dS to apply antithetic variance reduction technique
# warning: must ensure plus and minus objects are set via copy() instead of '=' otherwise memory issues
S_minus_m = S_plus_m.copy()
# A_c_minus_m = A_c_plus_m.copy()
d_S_minus_m = d_S_plus_m.copy()

# Define random number generator seed - this makes tne RNs predictable, otherwise get different everytime
np.random.seed(1000)

# *** Below is only for testing purposes (to replicate excel results) ***
RNs = pd.read_csv("temp_RNs.csv")
rand1_m = np.array(RNs.iloc[:, 0])
rand1_m = np.insert(rand1_m, 0, rand1_m[0])
rand2_m = np.array(RNs.iloc[:, 1])
rand2_m = np.insert(rand2_m, 0, rand2_m[0])
rand3_m = np.array(RNs.iloc[:, 2])
rand3_m = np.insert(rand3_m, 0, rand3_m[0])

# Loop over time - start from 2nd row since we have set initial values
for i in xrange(1, M + 1, 1):

    # ----- CONSTANT -----

    # (although must set array to these values at each row)
    mu_m[i][:] = mu
    vol1_m[i][:] = vol1
    vol2_m[i][:] = vol2
    vol3_m[i][:] = vol3
    d_tau_m[i][:] = d_tau

    # # Get I RNs sampled from standard normal distribution
    # rand1 = np.random.standard_normal((I, 1))
    # rand2 = np.random.standard_normal((I, 1))
    # rand3 = np.random.standard_normal((I, 1))

    # *** Below is only for testing purposes (to replicate excel results) ***
    rand1 = np.array([[rand1_m[i]], [np.random.standard_normal()]])
    rand2 = np.array([[rand2_m[i]], [np.random.standard_normal()]])
    rand3 = np.array([[rand3_m[i]], [np.random.standard_normal()]])
    # rand1 = rand_m[i]
    # rand2 = rand2_m[i]
    # rand3 = rand3_m[i]

    # ----- DYNAMIC -----

    # Recalculate dF term (dtau is constant)
    temp = np.diff(S_plus_m[i-1][:])
    d_S_plus_m[i][:] = np.hstack((temp, temp[:, -1].reshape(I, 1)))
    temp = np.diff(S_minus_m[i-1][:])
    d_S_minus_m[i][:] = np.hstack((temp, temp[:, -1].reshape(I, 1)))

    # Time step
    t_index[i] = t_index[i - 1] + dt

    # Generate S paths using antithetic reduction
    S_plus_m[i][:] = S_plus_m[i - 1][:] + \
                     mu_m[i][:] * dt + \
                     (vol1_m[i][:] * rand1 + vol2_m[i][:] * rand2 + vol3_m[i][:] * rand3) * math.sqrt(dt) + \
                     (d_S_plus_m[i][:] / d_tau_m[i][:]) * dt

    S_minus_m[i][:] = S_minus_m[i - 1][:] + \
                      mu_m[i][:] * dt + \
                      (vol1_m[i][:] * (-rand1) + vol2_m[i][:] * (-rand2) + vol3_m[i][:] * (-rand3)) * math.sqrt(dt) + \
                      (d_S_minus_m[i][:] / d_tau_m[i][:]) * dt
    
    # # Arithmetic continuous average must be set equal to S to calculate the true values later
    # A_c_plus_m[i][:] = S_plus_m[i][:]
    # A_c_minus_m[i][:] = S_minus_m[i][:]

############################################################################################

# -------------------------
#       PLOTTING
# -------------------------
#  notation: [t, I, tau]

# #------ Simulated Forward Curves -------
#
# # t=0, simulation 1 and 2, tau=all
# plt.plot(S_plus_m[0, 0, :])
# plt.plot(S_plus_m[0, 1, :])
# # t=1Y, simulation 1 and 2, all tenors
# plt.plot(S_plus_m[101, 0, :])
# plt.plot(S_plus_m[101, 1, :])
# # t=5Y, simulation 1 and 2, all tenors
# plt.plot(S_plus_m[501, 0, :])
# plt.plot(S_plus_m[501, 1, :])
#
# # ------ Projection of Forward Rate -------
#
# # t=all, simulation 1 and 2, tau=5Y
# plt.plot(S_plus_m[:, 0, 0])
# plt.plot(S_plus_m[:, 1, 0])
# # t=all, simulation 1 and 2, tau=5Y
# plt.plot(S_plus_m[:, 0, 11])
# plt.plot(S_plus_m[:, 1, 11])


############################################################################################

# --------------------------------
#       ANTITHETIC TECHNIQUE
# --------------------------------

# Join plus and minus stats (antithetic technique)
S_join_m = np.concatenate((S_plus_m, S_minus_m), axis=1)  # shape (M + 1, 2 * I, n_tau)
# A_c_join_m = np.concatenate((S_plus_m, S_minus_m), axis=1)  # shape (M + 1, 2 * I, n_tau)
# A_c_join_m = np.concatenate((A_c_plus_m, A_c_minus_m), axis=1)  # shape (M + 1, 2 * I, n_tau)

# --------------------------------
#       EXPANDING MEAN
# --------------------------------

# Arithmetic continuous expanding mean
A_c_plus_m = S_plus_m.copy()
A_c_minus_m = S_minus_m.copy()
A_c_join_m = S_plus_m.copy()  # antithetic version, for now set to same no. of dims as S_plus to plot it against it
# A_c_join_m[:][0] = 0.5 * (A_c_plus_m[:][0] + A_c_minus_m[:][0])

for i in xrange(1, I):
    # loop over all simulations for all tenors and times
    A_c_plus_m[:, i, :] = (i / (i + 1)) * A_c_plus_m[:, i - 1, :] + S_plus_m[:, i, :] / (i + 1)
    A_c_minus_m[:, i, :] = (i / (i + 1)) * A_c_minus_m[:, i - 1, :] + S_minus_m[:, i, :] / (i + 1)
    A_c_join_m[:, i, :] = 0.5 * (A_c_plus_m[:, i, :] + A_c_minus_m[:, i, :])

# -------------------------
#       PLOTTING
# -------------------------

sys.exit()

# p = sns.pointplot(np.arange(I), A_c_plus_m[1, :, 0])
# # p.xaxis.set_visible(False)
# plt.locator_params(nbins=5, axis='x')
# x_values = [0, 20, 40, 60, 80, 100]
# p.xaxis.set_ticklabels(x_values)

sns.pointplot(np.arange(I), A_c_plus_m[1, :, 0])
sns.pointplot(np.arange(I), A_c_join_m[1, :, 0])


############################################################################################


# ----------- PRICING
a = np.mean(S_plus_m, axis=1)
b = np.mean(S_join_m, axis=1)

# Rolling mean


# y = np.zeros(shape_3D, dtype=np.float)



## ---------- Plotting



# max_xticks = 10
# xloc = plt.MaxNLocator(max_xticks)
# p.xaxis.set_major_locator(xloc)

plt.show()
# df_plus = pd.DataFrame(data=A_c_plus_m[1, :, 0], columns=['V'])

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

