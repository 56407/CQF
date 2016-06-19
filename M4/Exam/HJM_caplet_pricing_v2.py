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

# MC parameters
M = 50  # no. of time steps
I = 6000  # no. of MC simulations (S paths)
n_tau = 50  # no. of tenors
dt = 0.01  # time step size
shape_3D = (M + 1, I, n_tau + 1)

# Caplet parameters
my_t = 0.50  # future time (want expectation of 6M LIBOR)
my_tau = 0.50
my_DF = 0.996  # discount factor
my_notional = 100000.0  # notional
my_K = 0.035  # strike

# Take drift, volatilities and f(t=0, T)  from CQF 'HJM Model - MC - Caplet v2.xlsm'
data = pd.read_csv("params.csv", index_col=0)
mu = np.array(data.iloc[0, :], dtype=np.float)  # drift
vol1 = np.array(data.iloc[1, :], dtype=np.float)  # volatility of highest eigenvalue e(1)
vol2 = np.array(data.iloc[2, :], dtype=np.float)  # volatility of 2nd highest eigenvalue e(2)
vol3 = np.array(data.iloc[3, :], dtype=np.float)  # volatility of 3rd highest eigenvalue e(3)
S0 = np.array(data.iloc[4, :], dtype=np.float)  # initial forward rate f(t=0, T)

# Calculate Musiela correction term dF/dtau
# Must append last value again to get correct vector dimension - this means 25Y maturity uses a backward derivative
d_S_plus = np.diff(S0)
d_S_plus = np.append(d_S_plus, d_S_plus[-1])  # this is dF
tau = np.array(data.columns, dtype=np.float) # convert from string to float to take diff
d_tau = np.diff(tau)
d_tau = np.append(d_tau, d_tau[-1])  # this is dtau

# Create a 3D numpy array for the above variables to account for n MC simulations
# 3D array shape is [t, I, tau] where t is time, I is number of simulations and tau the tenor

# Constant
mu_m = np.zeros(shape_3D, dtype=np.float)
vol1_m = np.zeros(shape_3D, dtype=np.float)
vol2_m = np.zeros(shape_3D, dtype=np.float)
vol3_m = np.zeros(shape_3D, dtype=np.float)
d_tau_m = np.zeros(shape_3D, dtype=np.float)

# Dynamic
t_m = np.zeros((M + 1), dtype=np.float)  # note this is a 2D array
d_S_plus_m = np.zeros(shape_3D, dtype=np.float)
S_plus_m = np.zeros(shape_3D, dtype=np.float)
C_plus_m = np.zeros(shape_3D, dtype=np.float)

# Initialise arrays (row=0, t=0) since loop starts at row=1

mu_m[0][:] = mu
vol1_m[0][:] = vol1
vol2_m[0][:] = vol2
vol3_m[0][:] = vol3
d_tau_m[0][:] = d_tau
d_S_plus_m[0][:] = d_S_plus
t_m[0] = 0.0
S_plus_m[0][:] = S0
# Caplet payoff
C_plus_m[0][:] = my_DF * np.maximum((1.0 / my_tau) * (np.exp(S0 * my_tau) - 1.0) - my_K, 0) * my_tau * my_notional

# Create Antithetic for S_plus, d_S_plus and C_plus, called them 'minus'
# warning: must ensure plus and minus objects are set via copy() instead of '=' otherwise memory issues
d_S_minus_m = d_S_plus_m.copy()
S_minus_m = S_plus_m.copy()
C_minus_m = C_plus_m.copy()

# Define random number generator seed - this makes tne RNs predictable, otherwise get different everytime
np.random.seed(1000)

# Loop over time - start from 2nd row since we have set initial values
print 'Starting t loop....'

for i in xrange(1, M + 1, 1):
    
    if i % 10 == 0:
        print 'i=', i
        
    # ----- CONSTANT -----

    # (although must set array to these values at each row)
    mu_m[i][:] = mu
    vol1_m[i][:] = vol1
    vol2_m[i][:] = vol2
    vol3_m[i][:] = vol3
    d_tau_m[i][:] = d_tau

    # Get I RNs sampled from standard normal distribution
    rand1 = np.random.standard_normal((I, 1))
    rand2 = np.random.standard_normal((I, 1))
    rand3 = np.random.standard_normal((I, 1))

    # ----- DYNAMIC -----

    # Recalculate dF term (dtau is constant)
    temp = np.diff(S_plus_m[i-1][:])
    d_S_plus_m[i][:] = np.hstack((temp, temp[:, -1].reshape(I, 1)))
    temp = np.diff(S_minus_m[i-1][:])
    d_S_minus_m[i][:] = np.hstack((temp, temp[:, -1].reshape(I, 1)))

    # Time step
    t_m[i] = t_m[i - 1] + dt

    # Generate S paths using antithetic reduction
    S_plus_m[i][:] = S_plus_m[i - 1][:] + \
                     mu_m[i][:] * dt + \
                     (vol1_m[i][:] * rand1 + vol2_m[i][:] * rand2 + vol3_m[i][:] * rand3) * math.sqrt(dt) + \
                     (d_S_plus_m[i][:] / d_tau_m[i][:]) * dt

    S_minus_m[i][:] = S_minus_m[i - 1][:] + \
                      mu_m[i][:] * dt + \
                      (vol1_m[i][:] * (-rand1) + vol2_m[i][:] * (-rand2) + vol3_m[i][:] * (-rand3)) * math.sqrt(dt) + \
                      (d_S_minus_m[i][:] / d_tau_m[i][:]) * dt

    # Calculate payoff function for caplet
    C_plus_m[i][:] = my_DF * np.maximum((1.0 / d_tau_m[i][:]) * (np.exp(S_plus_m[i][:] * d_tau_m[i][:]) - 1.0) - my_K, 0) * d_tau_m[i][:] * my_notional
    C_minus_m[i][:] = my_DF * np.maximum((1.0 / d_tau_m[i][:]) * (np.exp(S_minus_m[i][:] * d_tau_m[i][:]) - 1.0) - my_K, 0) * d_tau_m[i][:] * my_notional


print 'Finished t loop....'

############################################################################################

# --------------------------------
#       ROLLING MEAN
# --------------------------------

# Join plus and minus stats (antithetic technique)
# S_join_m = np.concatenate((S_plus_m, S_minus_m), axis=1)  # shape (M + 1, 2 * I, n_tau)

# Arithmetic continuous running average for plus, minus and join
A_c_plus_m = C_plus_m.copy()
A_c_minus_m = C_minus_m.copy()
A_c_join_m = np.zeros(shape_3D, dtype=np.float)  # antithetic version, for now set to same no. of dims as S_plus to plot it against it
A_c_join_m[:, 0, :] = 0.5*(A_c_plus_m[:, 0, :] + A_c_minus_m[:, 0, :])

print 'Starting MC loop....'

for i in xrange(1, I):
    if i % 100 == 0:
        print 'i=', i
    # loop over all simulations for all tenors and times
    A_c_plus_m[:, i, :] = (i / (i + 1)) * A_c_plus_m[:, i - 1, :] + C_plus_m[:, i, :] / (i + 1)
    A_c_minus_m[:, i, :] = (i / (i + 1)) * A_c_minus_m[:, i - 1, :] + C_minus_m[:, i, :] / (i + 1)
    A_c_join_m[:, i, :] = 0.5 * (A_c_plus_m[:, i, :] + A_c_minus_m[:, i, :])

print 'End MC loop....'

# Locate index of desired t and tau to get caplet pricing for corresponding libor L(t, tau)
my_t_loc = np.where(np.round(t_m, 2) == np.round(my_t, 2))[0][0]
my_tau_loc = np.where(np.round(tau, 2) == np.round(my_tau, 2))[0][0]

# Convergence diagram
plt.plot(np.arange(I), A_c_plus_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Standard MC')
plt.plot(np.arange(I), A_c_minus_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Antithetic MC')
plt.plot(np.arange(I), A_c_join_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Average')
plt.xlabel('No. of MC simulations')
plt.ylabel('Caplet price')
plt.legend()

e_plus = A_c_plus_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
e_minus = A_c_minus_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
e_join = A_c_join_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)

print 'Caplet price is: \n' \
      'Std. MC ={0} ({3}) \n ' \
      'Antithetic MC ={1} ({4}) \n' \
      'Average ={2} ({5})'.format(A_c_plus_m[my_t_loc, -1, my_tau_loc],
                                  A_c_minus_m[my_t_loc, -1, my_tau_loc],
                                  A_c_join_m[my_t_loc, -1, my_tau_loc],
                                  e_plus,
                                  e_minus,
                                  e_join)
