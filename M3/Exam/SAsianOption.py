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


def asian_option_simulator(S0, K, T, r, sigma, M, I, k, mode='fixed'):
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
    return dic


def fill_in_df(df, dic, i):
    # i represents the row number in the df
    mode = dic['mode']
    df.loc[i, 'mode'] = mode
    df.loc[i, 'N_S'] = dic['I']
    df.loc[i, 'N_t'] = dic['M']
    df.loc[i, 'S0'] = dic['S0']
    if mode == 'fixed':
        df.loc[i, 'K'] = dic['K']
        if dic['S0'] > dic['K']:
            df.loc[i, 'Class'] = 'ITM'  # in the money
        elif dic['S0'] < dic['K']:
            df.loc[i, 'Class'] = 'OTM'  # out of the money
        else:
            df.loc[i, 'Class'] = 'ATM'  # at the money
    df.loc[i, 'r'] = dic['r']
    df.loc[i, 'sigma'] = dic['sigma']
    df.loc[i, 'T'] = dic['T']
    df.loc[i, 'k'] = dic['k']
    df.loc[i, 'diff_GC_c'] = '{0:.6f}'.format(dic['V'] - dic['GC_c'])
    df.loc[i, 'V'] = '{0:.6f} ({1:.6f})'.format(dic['V'], dic['V_e'])
    df.loc[i, 'AC_c'] = '{0:.6f} ({1:.6f})'.format(dic['AC_c'], dic['AC_c_e'])
    df.loc[i, 'AC_d'] = '{0:.6f} ({1:.6f})'.format(dic['AC_d'], dic['AC_d_e'])
    df.loc[i, 'GC_c'] = '{0:.6f} ({1:.6f})'.format(dic['GC_c'], dic['GC_c_e'])

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

dic = asian_option_simulator(S0=100., K=100., T=1.0, r=0.05, sigma=0.2, M=100, I=100, k=10, mode='fixed')

t_index = dic['t_index']
S_join = dic['S_join']
S_plus = S_join[:, 0:100]
A_c_join = dic['A_c_join']
A_d_join = dic['A_d_join']
G_c_join = dic['G_c_join']
V_join = dic['V_join']
V_plus = V_join[:, 0:100]
V = dic['V']
AC_c_join = dic['AC_c_join']
AC_d_join = dic['AC_d_join']
GC_c_join = dic['GC_c_join']
AC_c = dic['AC_c']
AC_d = dic['AC_d']
GC_c = dic['GC_c']

# --------------------------------------------------
# EUROPEAN CALL - using antithetic variance reduction
# --------------------------------------------------
print 'EU V(T) = {0}'.format(V)

# # V vs S plot without and with variance reduction
# sns.regplot(x=S_join[-1], y=V_join[-1], fit_reg=False) # with antithetic correction
# sns.regplot(x=S_plus[-1], y=V_plus[-1], fit_reg=False, color='yellow', marker='+') # without antithetic correction

# --------------------------------------------------
# ASIAN CALL - using antithetic variance reduction
# --------------------------------------------------

# ----------------------------------------
# ARITHMETIC
# ----------------------------------------

# -----------------------
# Continuous sampling
# -----------------------
print 'Asian AC_c(T) = {0}'.format(AC_c)
# # V vs S plot
# sns.regplot(x=S_join[-1], y=AC_c_join[-1], fit_reg=False) # with antithetic correction
# # sns.regplot(x=S_join[-1, 0:100], y=AC_c_join[-1, 0:100], fit_reg=False)  # without antithetic correction

# -----------------------
# Discrete sampling
# -----------------------
print 'Asian AC_d(T) = {0}'.format(AC_d)

# -----------------------
# Continuous vs Discrete plots
# -----------------------

# # S plot for MC, Arithmetic cont. and discrete averages
# plt.plot(S_join[0:,0:1], label='S_sim1')
# plt.plot(A_c_join[0:,0:1], label='Cont. Avg.')
# plt.plot(A_d_join[0:,0:1], label='Disc. Avg.')
# plt.legend()

# # Evolution of Asian Call value with number of time steps
# c = AC_c_join.mean(axis=1)
# d = AC_d_join.mean(axis=1)
# plt.plot(c)
# plt.plot(d)

# ----------------------------------------
# GEOMETRIC
# ----------------------------------------

# -----------------------
# Continuous sampling
# -----------------------
print 'Asian GC_c(T) = {0}'.format(GC_c)


