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
#  and calculates the CVA for an IRS
#
#############################################################

# MC parameters
M = 500  # no. of time steps
I = 5  # no. of MC simulations (S paths)
n_tau = 50  # no. of tenors
dt = 0.01  # time step size
shape_3D = (M + 1, I, n_tau + 1)
np.random.seed(1000)  # this makes RNs predictable

# Read other pre-calculated data like DFs, PD, CDS, etc. necessary for CVA calculation
data2 = pd.read_csv("input.csv", index_col=0)

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

###################################################
#
#       CVA CALCULATION
#
###################################################

# Take only the relevant forward rates for the IRS we want (6M LIBOR expiring in 5Y) and store in dataframe
# t = 0.5, 1.0, ..., 5.0 (index=50, 100, 150,...,500); I = all (index=:); tau = 0.5 (index=1)
col_names = ['Sim' + str(x) for x in xrange(1, I + 1)]
f_plus = pd.DataFrame(index=data2.index, columns=col_names, dtype=np.float)  # ensure dtype is set to float otherwise np.exp doesn't work
i = 0
for index, row in f_plus.iterrows():
    f_plus.loc[index, :] = S_plus_m[int(i), :, 1]
    i += 50

# Convert to LIBOR
freq = 0.5  # payment frequency (day count fraction)
L_plus = (1.0 / freq) * (np.exp(f_plus * freq) - 1.0)

# Calculate Discount Factors matrix
ois = data2['OIS_spot']
DF = pd.DataFrame(index=data2.index, columns=list(data2.index))
DF.loc[0.0, :] = np.exp(-ois * (ois.index - 0.0))  # set first row to DF(0, T_i)

for index, row in DF.iterrows():
    if index == 0.0:  # skip first row as already set above
        continue
    x = DF.loc[0.0][row.index]/DF.loc[0.0, index]
    # x = np.exp(-ois * (row.index - index))
    x[x > 1] = 0  # set to zero cases when the DF start time > final time
    DF.loc[index, :] = x

np.fill_diagonal(DF.values, 0)  # set values along diagonal to zero to exclude them from the dot product sum below

# Get the MTM value of the swap i.e. as a function of time
N = 1.0  # notional
K_plus = L_plus.iloc[0, :]  # fixed rate set to L(t, 0, 0.5) to have zero initial cashflows (par swap)
V_plus = N * freq * DF.dot(L_plus - K_plus)  # note matrix multiplication using dot method

# Get Exposure profile
E_plus = np.maximum(V_plus, 0)

# # Plot all Exposure profiles for which the 0.5 tenor is less than or equal to zero
# E_plus.loc[:, E_plus.loc[0.5] <= 0].plot()
# # Plot all Exposure profiles for which the 0.0 tenor values are less than 0.5 values
# E_plus.loc[:, E_plus.loc[0.5] < E_plus.loc[1.0]].plot()

# Get Expected Exposure (EE)
x = np.ma.masked_where(E_plus == 0, E_plus)  # need to mask zero values to exclude them from the median
x = np.ma.median(x[:-1], axis=1)  # mean is not so good as big outliers in data
EE_plus = pd.DataFrame(index=E_plus.index[:-1], data=x, columns=['EE_plus'])

sys.exit()

# Calculate CVA by taking median (average) between tenors for the exposure and DFs (PD already ok)
E_plus_med = (E_plus + E_plus.shift(-1)) / 2.0
E_plus_med = E_plus_med.reset_index(drop=True).dropna()  # must reset index and drop NaN to multiply by below vars
DF2 = data2['DF_T1_T2']  # need to parse discount factors again to use shift and take median
DF2.iloc[0] = 1.0  # set this to one to take the median and not get nan
DF_med = (DF2 + DF2.shift(-1)) / 2.0
DF_med = DF_med.reset_index(drop=True).dropna()
PD = data2['PD'].shift(-1)
PD = PD.reset_index(drop=True).dropna()
RR = 0.4  # recovery rate
CVA = E_plus_med * DF_med * PD * (1 - RR)
CVA_cum = CVA.cumsum()

# Mention accruals missing
# Only include positive part becasue negative is a libability rather than an asset - so only consider +v2 M2M, this is exposure


sys.exit()
############################################################################################

# # Get the MTM value of the swap i.e. as a function of time (reverse cum sum of payments)
# V_plus = payments_plus.sort_index(axis=0, ascending=False).cumsum()
# V_plus = V_plus.shift(1).sort_index(ascending=True)
# V_plus.iloc[-1, :] = 0  # set the value of the swap to zero at the end of the term
#
# # Get Exposure profile
# E_plus = np.maximum(V_plus, 0)
#
# # # Plot all Exposure profiles for which the 0.5 tenor is less than or equal to zero
# # E_plus.loc[:, E_plus.loc[0.5] <= 0].plot()
# # # Plot all Exposure profiles for which the 0.0 tenor values are less than 0.5 values
# # E_plus.loc[:, E_plus.loc[0.5] < E_plus.loc[1.0]].plot()

# # --------------------------------
# #       ROLLING MEAN
# # --------------------------------
#
# # Join plus and minus stats (antithetic technique)
# # S_join_m = np.concatenate((S_plus_m, S_minus_m), axis=1)  # shape (M + 1, 2 * I, n_tau)
#
# # Arithmetic continuous running average for plus, minus and join
# A_c_plus_m = E_plus_m.copy()
# A_c_minus_m = E_minus_m.copy()
# A_c_join_m = np.zeros(shape_3D, dtype=np.float)  # antithetic version, for now set to same no. of dims as S_plus to plot it against it
# A_c_join_m[:, 0, :] = 0.5*(A_c_plus_m[:, 0, :] + A_c_minus_m[:, 0, :])
#
# print 'Starting MC loop....'
#
# for i in xrange(1, I):
#     if i % 100 == 0:
#         print 'i=', i
#     # loop over all simulations for all tenors and times
#     A_c_plus_m[:, i, :] = (i / (i + 1)) * A_c_plus_m[:, i - 1, :] + E_plus_m[:, i, :] / (i + 1)
#     A_c_minus_m[:, i, :] = (i / (i + 1)) * A_c_minus_m[:, i - 1, :] + E_minus_m[:, i, :] / (i + 1)
#     A_c_join_m[:, i, :] = 0.5 * (A_c_plus_m[:, i, :] + A_c_minus_m[:, i, :])
#
# print 'End MC loop....'
#
# # Locate index of desired t and tau to get corresponding LIBOR L(t, tau)
# my_t_loc = np.where(np.round(t_m, 2) == np.round(my_t, 2))[0][0]
# my_tau_loc = np.where(np.round(tau, 2) == np.round(my_tau, 2))[0][0]
#
# # Convergence diagram
# plt.plot(np.arange(I), A_c_plus_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Standard MC')
# plt.plot(np.arange(I), A_c_minus_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Antithetic MC')
# plt.plot(np.arange(I), A_c_join_m[my_t_loc, :, my_tau_loc], '-', ms=7, label='Average')
# plt.xlabel('No. of MC simulations')
# plt.ylabel('LIBOR')
# plt.legend()
#
# e_plus = A_c_plus_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
# e_minus = A_c_minus_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
# e_join = A_c_join_m[my_t_loc, :, my_tau_loc].std()/np.sqrt(I)
#
# print 'LIBOR is: \n' \
#       'Std. MC ={0} ({3}) \n ' \
#       'Antithetic MC ={1} ({4}) \n' \
#       'Average ={2} ({5})'.format(A_c_plus_m[my_t_loc, -1, my_tau_loc],
#                                   A_c_minus_m[my_t_loc, -1, my_tau_loc],
#                                   A_c_join_m[my_t_loc, -1, my_tau_loc],
#                                   e_plus,
#                                   e_minus,
#                                   e_join)
