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
#  This script does I MC simulations of the HJM model
#
#############################################################

# MC parameters
M = 500  # no. of time steps
I = 1000  # no. of MC simulations (S paths)
n_tau = 50  # no. of tenors
dt = 0.01  # time step size
shape_3D = (M + 1, I, n_tau + 1)
np.random.seed(1000)  # this makes RNs predictable

# Take drift, volatilities and f(t=0, T)  from CQF 'my HJM Model - MC - Caplet v2.xlsm'
data = pd.read_csv("params.csv", index_col=0)
mu = np.array(data.iloc[0, :], dtype=np.float)  # drift
vol1 = np.array(data.iloc[1, :], dtype=np.float)  # volatility of highest eigenvalue e(1)
vol2 = np.array(data.iloc[2, :], dtype=np.float)  # volatility of 2nd highest eigenvalue e(2)
vol3 = np.array(data.iloc[3, :], dtype=np.float)  # volatility of 3rd highest eigenvalue e(3)
S0 = np.array(data.iloc[4, :], dtype=np.float)  # initial forward rate f(t=0, T)
tau = np.array(data.columns, dtype=np.float)  # convert from string to float to take diff

# Calculate Musiela correction term dF/dtau
# Must append last value again to get correct vector dimension - this means 25Y maturity uses a backward derivative
d_S_plus = np.diff(S0)
d_S_plus = np.append(d_S_plus, d_S_plus[-1])  # this is dF
d_tau = np.diff(tau)
d_tau = np.append(d_tau, d_tau[-1])  # this is dtau

# Get pre-calculated discount factors
data2 = pd.read_csv("input.csv", index_col=0)
DF_m = np.zeros((M + 1), dtype=np.float)
i = 0
j = 50
for row in data2['DF_T1_T2']:
    DF_m[i:j] = row
    i = j
    j += 50

# Define numpy arrays for MC
t_m = np.zeros((M + 1), dtype=np.float)  # note this is a 2D array
S_plus_m = np.zeros(shape_3D, dtype=np.float)

# Initialise arrays (row=0, t=0) since loop starts at row=1
t_m[0] = 0.0
S_plus_m[0][:] = S0
S_minus_m = S_plus_m.copy()  # antithetic

# Loop over time - start from 2nd row since we have set initial values
print 'Starting t loop....'

for i in xrange(1, M + 1, 1):
    if i % 10 == 0:
        print 'i=', i
        
    # Get I RNs sampled from standard normal distribution
    rand1 = np.random.standard_normal((I, 1))
    rand2 = np.random.standard_normal((I, 1))
    rand3 = np.random.standard_normal((I, 1))

    # Recalculate dF for Musiela term
    d_S_plus = np.diff(S_plus_m[i-1][:])
    d_S_plus = np.hstack((d_S_plus, d_S_plus[:, -1].reshape(I, 1)))
    d_S_minus = np.diff(S_minus_m[i-1][:])
    d_S_minus = np.hstack((d_S_minus, d_S_minus[:, -1].reshape(I, 1)))

    # Time step
    t_m[i] = round(t_m[i - 1] + dt, 2)  # rounding used to correct float arithmetic

    # Generate S paths using antithetic reduction
    S_plus_m[i][:] = S_plus_m[i - 1][:] + \
                     mu * dt + \
                     (vol1 * rand1 + vol2 * rand2 + vol3 * rand3) * math.sqrt(dt) + \
                     (d_S_plus / d_tau) * dt
    S_minus_m[i][:] = S_minus_m[i - 1][:] + \
                     mu * dt + \
                     (vol1 * (-rand1) + vol2 * (-rand2) + vol3 * (-rand3)) * math.sqrt(dt) + \
                     (d_S_minus / d_tau) * dt

print 'Finished t loop....'


# Take only the relevant rates for the swap we want and store in a dataframe
# t = 0.5, 1.0, ..., 5.0 (index=50, 100, 150,...,500); I = all (index=:); tau = 0.5 (index=1)
col_names = ['Sim'+str(x) for x in xrange(1, I+1)]
df_plus = pd.DataFrame(index=data2.index, columns=col_names, dtype=np.float)  # ensure dtype is set to float otherwise np.exp doesn't work
i = 0
for index, row in df_plus.iterrows():
    df_plus.loc[index, :] = S_plus_m[int(i), :, 1]
    i += 50

# Convert to LIBOR
my_freq = 0.5  # payment frequency (day count fraction)
L_plus = (1.0 / my_freq) * (np.exp(df_plus * my_freq) - 1.0)

# Get swap payments
my_N = 1.0  # notional
my_K_plus = L_plus.iloc[0, :]  # fixed rate (set to L(t, 0, 0.5) to have zero initial cashflows
DF = data2['DF_T1_T2'].reshape(data2.shape[0], 1)  # Discount Factors (transpose to multiply by dataframe correctly)
payments_plus = my_N * my_freq * DF * (L_plus - my_K_plus)

# Get the M2M value of the swap (reverse cum sum of payments)
V_plus = payments_plus.sort_index(axis=0, ascending=False).cumsum()
V_plus = V_plus.shift(1).sort_index(ascending=True)
V_plus.iloc[-1, :] = 0  # set the value of the swap to zero at the end of the term

# Get Exposure
E_plus = np.maximum(V_plus, 0)

# # Plot all Exposure profiles for which the 0.5 tenor is less than or equal to zero
# E_plus.loc[:, E_plus.loc[0.5] <= 0].plot()

sys.exit()
############################################################################################

# # # Other parameters
# my_t = 0.50  # future time (want expectation of 6M LIBOR)
# my_tau = 0.50
# # my_DF = 0.996  # discount factor -- now calculated more accurately
# my_notional = 1.0  # notional
# my_K = 0.00637435465921353  # fixed rate (set to L(t, 0, 0.5) to have zero initial cashflows

# --------------------------------
#       ROLLING MEAN
# --------------------------------

# Join plus and minus stats (antithetic technique)
# S_join_m = np.concatenate((S_plus_m, S_minus_m), axis=1)  # shape (M + 1, 2 * I, n_tau)

# Arithmetic continuous running average for plus, minus and join
A_c_plus_m = E_plus_m.copy()
A_c_minus_m = E_minus_m.copy()
A_c_join_m = np.zeros(shape_3D, dtype=np.float)  # antithetic version, for now set to same no. of dims as S_plus to plot it against it
A_c_join_m[:, 0, :] = 0.5*(A_c_plus_m[:, 0, :] + A_c_minus_m[:, 0, :])

print 'Starting MC loop....'

for i in xrange(1, I):
    if i % 100 == 0:
        print 'i=', i
    # loop over all simulations for all tenors and times
    A_c_plus_m[:, i, :] = (i / (i + 1)) * A_c_plus_m[:, i - 1, :] + E_plus_m[:, i, :] / (i + 1)
    A_c_minus_m[:, i, :] = (i / (i + 1)) * A_c_minus_m[:, i - 1, :] + E_minus_m[:, i, :] / (i + 1)
    A_c_join_m[:, i, :] = 0.5 * (A_c_plus_m[:, i, :] + A_c_minus_m[:, i, :])

print 'End MC loop....'

# Locate index of desired t and tau to get corresponding LIBOR L(t, tau)
my_t_loc = np.where(np.round(t_m, 2) == np.round(my_t, 2))[0][0]
my_tau_loc = np.where(np.round(tau, 2) == np.round(my_tau, 2))[0][0]

# Convergence diagram
plt.plot(np.arange(I), A_c_plus_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Standard MC')
plt.plot(np.arange(I), A_c_minus_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Antithetic MC')
plt.plot(np.arange(I), A_c_join_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Average')
plt.xlabel('No. of MC simulations')
plt.ylabel('LIBOR')
plt.legend()

e_plus = A_c_plus_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
e_minus = A_c_minus_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
e_join = A_c_join_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)

print 'LIBOR is: \n' \
      'Std. MC ={0} ({3}) \n ' \
      'Antithetic MC ={1} ({4}) \n' \
      'Average ={2} ({5})'.format(A_c_plus_m[my_t_loc, -1, my_tau_loc],
                                  A_c_minus_m[my_t_loc, -1, my_tau_loc],
                                  A_c_join_m[my_t_loc, -1, my_tau_loc],
                                  e_plus,
                                  e_minus,
                                  e_join)
