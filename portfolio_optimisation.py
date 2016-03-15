import numpy as np
import math
from IPython import embed


def get_useful_matrices(mu, sigma, R):

    # Vector of 1's
    I = np.matrix([[1],
                   [1],
                   [1],
                   [1]])

    # Std deviation matrix (Diagonal of std dev vector)
    S = np.diag(sigma)

    # Construct covariance matrix X
    X = S * R * S

    # Inverse of covariance matrix
    X_inv = np.matrix.getI(X)

    # A, B , C matrices
    I_T = np.matrix.transpose(I)
    mu_T = np.matrix.transpose(mu)
    A = I_T * X_inv * I
    B = I_T * X_inv * mu  # equivalent to np.matrix.transpose(mu) * X_inv * I
    C = mu_T * X_inv * mu

    dic = {'X': X,
           'X_inv': X_inv,
           'A': A,
           'B': B,
           'C': C,
           'I': I,
           'I_T': I_T,
           'mu': mu,
           'mu_T': mu_T,
           'sigma': sigma,
           'R': R}

    return dic


def N_risky_plus_RFA_portfolio(M, m, r):

    X = M['X']
    X_inv = M['X_inv']
    I = M['I']

    # Optimised portfolio allocations
    w = (m - r) * X_inv * (mu - r * I) / (np.matrix.transpose(mu - r * I) * X_inv * (mu - r * I))
    w_T = np.matrix.transpose(w)

    # Optimised risk
    s = math.sqrt(w_T * M['X'] * w)

    l = [w, s]
    print '--- Portfolio of N risky assets plus RFA ---\n' \
          'w_opt = {0}, s_opt = {1}'.format(l[0], l[1])

    return l

def tangency_portfolio(M, r):

    # Optimised portfolio allocations
    w = M['X_inv'] * (M['mu'] - r * M['I']) / (M['B'] - M['A'] * r)
    w_T = np.matrix.transpose(w)
    
    # Optimised risk
    s = math.sqrt(w_T * M['X'] * w)

    # Tangency return
    m = (M['C'] - M['B'] * r) / (M['B'] - M['A'] * r)

    l = [w, s, m]

    print '--- Tangency Portfolio ---\n' \
          'w_opt = {0}, s_opt = {1}, m = {2}'.format(l[0], l[1], l[2])

    return l


if __name__ == "__main__":

    # Expected returns 4D vector
    mu = np.matrix([[0.04],
                    [0.08],
                    [0.12],
                    [0.15]])

    # Std deviation (risk) 4D vector
    sigma = np.array([0.07, 0.12, 0.18, 0.26])
    # sigma = np.matrix([[0.07],
    #                    [0.12],

    # Correlation 4x4 matrix
    R = np.matrix([[1, 0.2, 0.5, 0.3],
                   [0.2, 1, 0.7, 0.4],
                   [0.5, 0.7, 1, 0.9],
                   [0.3, 0.4, 0.9, 1]])

    # Target return
    m = 0.1

    # Risk-free rate
    r = 0.03

    # Calculate portfolio allocations and risk
    M = get_useful_matrices(mu, sigma, R)
    P1 = N_risky_plus_RFA_portfolio(M, m, r)
    P2 = tangency_portfolio(M, r)

    # # -------------------------------------
    # # Lectures example - M2S2 page 78
    #
    # # Expected returns 4D vector
    # mu = np.matrix([[0.05],
    #                 [0.07],
    #                 [0.15],
    #                 [0.27]])
    #
    # # Std deviation (risk) 4D vector
    # sigma = np.array([0.07, 0.12, 0.30, 0.60])
    # # sigma = np.matrix([[0.07],
    # #                    [0.12],
    #
    # # Correlation 4x4 matrix
    # R = np.matrix([[1, 0.8, 0.5, 0.4],
    #                [0.8, 1, 0.7, 0.5],
    #                [0.5, 0.7, 1, 0.8],
    #                [0.4, 0.5, 0.8, 1]])
    #
    # # Target return
    # m = 0.1
    #
    # # Risk-free rate
    # r = 0.025
    #
    # # Calculate portfolio allocations and risk
    # M = get_useful_matrices(mu, sigma, R)
    # P1 = N_risky_plus_RFA_portfolio(M, m, r)
    # P2 = tangency_portfolio(M, m, r)


# RESULTS
#
# In [13]: %run portfolio_optimisation.py
# --- Portfolio of N risky assets plus RFA ---
# w_opt = [[ 0.39572412]
#  [ 1.05408235]
#  [-0.82682859]
#  [ 0.73127679]], s_opt = 0.132119656709
# <class 'numpy.matrixlib.defmatrix.matrix'>
# --- Tangency Portfolio ---
# w_opt = [[ 0.29220805]
#  [ 0.77834869]
#  [-0.61054144]
#  [ 0.53998469]], s_opt = 0.0975589452358, m = [[ 0.08168895]]

# Lectures example - M2S2 page 78
# --- Portfolio of N risky assets plus RFA ---
# w_opt = [[ 0.88735248]
#  [ 0.08126325]
#  [ 0.15484343]
#  [ 0.12164862]], s_opt = 0.160284145616
# <class 'numpy.matrixlib.defmatrix.matrix'>
# --- Tangency Portfolio ---
# w_opt = [[ 0.71267122]
#  [ 0.06526604]
#  [ 0.12436147]
#  [ 0.09770128]], s_opt = 0.128731140503, m = [[ 0.08523575]]
