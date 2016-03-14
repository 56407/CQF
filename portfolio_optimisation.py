import numpy as np
import math
from IPython import embed


def min_var_portfolio(mu, sigma, R, m, r):

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

    # Global minimum variance portfolio allocations
    w_g = (m - r) * X_inv * (mu - r * I) / (np.matrix.transpose(mu - r * I) * X_inv * (mu - r * I))

    # Global minimum std deviation
    w_g_T = np.matrix.transpose(w_g)
    sigma_g = math.sqrt(w_g_T * X * w_g)

    return [w_g, sigma_g]


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

    print min_var_portfolio(mu, sigma, R, m, r)

    # Result:
    # [matrix([[0.39572412],
    #          [1.05408235],
    #          [-0.82682859],
    #          [0.73127679]]), 0.13211965670857084]

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
    # print min_var_portfolio(mu, sigma, R, m, r)
    # Result:
    # [matrix([[0.88735248],
    #          [0.08126325],
    #          [0.15484343],
    #          [0.12164862]]), 0.16028414561637888]
