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

# Read other pre-calculated data like DFs, PD, CDS, etc. necessary for CVA calculation
data2 = pd.read_csv("input.csv", index_col=0)

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

# Set fixed rate K - to L(t, 0, 0.5) to have zero initial cashflows (par swap)
K_plus = L_plus.iloc[0, :][0]  # getting scalar here but know same value across all simulations

# Mask the rates which don't contribute to the Exposure (when Libor is less than the agreed rate)
L_plus_masked = np.ma.masked_where(L_plus < K_plus, L_plus)

# Calculate Discount Factors for each simulation, ZCB stands for Zero Coupon Bond
ZCB_plus = 1.0 / (1 + freq * L_plus)
ZCB_plus.iloc[0, :] = 1.0  # set first row to 1 by definition
ZCB_plus = ZCB_plus.cumprod()  # take cumulative product (equivalent to 'integrating under f_plus')
ZCB_plus_mean = pd.Series(index=ZCB_plus.index, data=np.mean(ZCB_plus, axis=1))

# Not sure if commented lines below would be valid
# # For simplicity, take the mean of the DFs that contribute to the Exposure and calculate the DF matrix from this
# ZCB_plus_masked = np.ma.masked_array(ZCB_plus, mask=L_plus_masked.mask)
# ZCB_plus_masked_mean = pd.Series(index=ZCB_plus.index, data=np.mean(ZCB_plus_masked, axis=1))

DF = pd.DataFrame(index=ZCB_plus.index, columns=list(ZCB_plus.index))
# DF.loc[0.0, :] = ZCB_plus_masked_mean  # unsure if this would be valid (see above)
DF.loc[0.0, :] = ZCB_plus_mean  # set first row to DF(0, T_i)

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
V_plus = N * freq * DF.dot(L_plus - K_plus)  # note matrix multiplication using dot method

# Get Exposure profile (only include positive part because negative is a liability rather than an asset)
E_plus = np.maximum(V_plus, 0)

# # Plot all Exposure profiles for which the 0.5 tenor is less than or equal to zero
# E_plus.loc[:, E_plus.loc[0.5] <= 0].plot()
# # Plot all Exposure profiles for which the 0.0 tenor values are less than 0.5 values
# E_plus.loc[:, E_plus.loc[0.5] < E_plus.loc[1.0]].plot()

# Mask zero values to exclude them from the mean and median calculations
E_plus2 = np.array(E_plus, dtype=np.float32)  # need to do this line otherwise np.ma.mean throws error
E_plus_masked = np.ma.masked_where(E_plus2 == 0, E_plus2)

# Get Expected Exposure (EE) from median and mean of positive E (though mean is not so good as big outliers in data)
# must remove last row of E_plus as all elements zero & below will throw error
EE_plus_median = pd.Series(index=E_plus.index[:-1], data=np.ma.median(E_plus_masked[:-1], axis=1))
EE_plus_median.loc[5.0] = 0.0  # add 5.0 tenor row as we know it's zero EE
EE_plus_mean = pd.Series(index=E_plus.index[:-1], data=np.ma.mean(E_plus_masked[:-1], axis=1))
EE_plus_mean.loc[5.0] = 0.0  # add 5.0 tenor row as we know it's zero EE

# Get PFE from the Exposure 97.5 percentile
PFE_plus = pd.Series(index=E_plus.index, data=np.percentile(E_plus_masked, q=97.5, axis=1))

# # Plotting
# EE_plus_mean.plot(label="EE_mean")
# EE_plus_median.plot(label="EE_median")
# PFE_plus.plot(label="PFE")
# plt.legend()

# Interpolate EE and DF between tenors to calculate CVA
index_interpol = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5',
                  '4.5-5.0']

EE_plus_median_interpol = (EE_plus_median + EE_plus_median.shift(-1)) / 2.0
EE_plus_median_interpol = EE_plus_median_interpol.iloc[:-1]
EE_plus_median_interpol.index = index_interpol

DF_interpol = DF.loc[0.0, :].copy()  # for simplicity take only DF(t=0, T_i)
DF_interpol[0.0] = 1.0  # need to set this to 1 again to take the average
DF_interpol = (DF_interpol + DF_interpol.shift(-1))/2.0
DF_interpol = DF_interpol.iloc[:-1]
DF_interpol.index = index_interpol

PD_interpol = data2['PD'].iloc[1:]  # this is already interpolated
PD_interpol.index = index_interpol

# Calculate CVA
RR = 0.4  # recovery rate
CVA = (1 - RR) * EE_plus_median_interpol * DF_interpol * PD_interpol
CVA_total = CVA.sum()

# # Plotting
# CVA.plot.bar(width=1.0, alpha=0.5)
# Loss = pd.Series(index=index_interpol, data=(1-RR))
# df = pd.concat([Loss, EE_plus_median_interpol, DF_interpol, PD_interpol, CVA], axis=1,
#                keys=['1-RR', 'EE_interpol', 'DF_interpol', 'PD_bootstr', 'CVA'])
# df.plot(subplots=True, marker='o')
# # df.plot(subplots=True, kind='bar', alpha=0.5, width=1.0)

# NOTE: the above analysis doesn't account for accruals effects

############################################################################################
#
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
# A_c_join_m = np.zeros(shape_3D, dtype=np.float)
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
