from datetime import datetime
import sys

import pandas as pd
import numpy as np

import math
from math import exp, sqrt, log

from IPython import embed

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# =================   OU PROCESS   =================

# MC params
np.random.seed(2000)  # set the seed
dt = 1  # time step
M = 1000  # no. of time steps

# Model params:
mu = 10
sigma = 0.3

Y_t1 = np.zeros((M + 1))
Y_t2 = np.zeros((M + 1))
Y_t3 = np.zeros((M + 1))

Y_t1[0] = -50.0
Y_t2[0] = 50.0
Y_t3[0] = 0.0

theta1 = 0.003
theta2 = 0.01
theta3 = 0.1

for i in xrange(1, M + 1, 1):
    Y_t1[i] = Y_t1[i-1] + theta1 * (mu - Y_t1[i-1]) * dt + sigma * math.sqrt(dt) * np.random.normal(0, 1)
    Y_t2[i] = Y_t2[i-1] + theta2 * (mu - Y_t2[i-1]) * dt + sigma * math.sqrt(dt) * np.random.normal(0, 1)
    Y_t3[i] = Y_t3[i-1] + theta3 * (mu - Y_t3[i-1]) * dt + sigma * math.sqrt(dt) * np.random.normal(0, 1)

Y_t1 = pd.Series(Y_t1, name='Y_t1')
Y_t2 = pd.Series(Y_t2, name='Y_t2')
Y_t3 = pd.Series(Y_t3, name='Y_t3')

# =================   MULTIVARIATE REGRESSION   =================

# Main refs used:
# http://www.ats.ucla.edu/stat/sca/finn/finn4.pdf
# CQF_January_2016_M4S11_Annotated.pdf
# http://statsmodels.sourceforge.net/stable/_modules/statsmodels/tsa/ar_model.html#AR

from statsmodels.tsa.tsatools import (lagmat, add_trend)  # helper functions to add lags and trends


def my_df_resid(exog=None, my_df=None):
    """
    Function to get degrees of freedom to calculate the scale of the OLS covariance matrix (see below)
    :param exog: exogenous variable used to get its dimension
    :param nobs: number of observations used to estimate parameters
    :param my_df: if already know the df, can also plug instead of using default
    :return:
    """
    if my_df is None:
        # rank = np.ndim(exog)
        nobs = exog.shape[0]
        rank = exog.shape[1]
        df = np.float(nobs - rank)
    else:
        df = my_df
    return df

def my_OLS(Y, X, df_resid=None):
    """
    Linear Regression implementation using Ordinary Least Squares (OLS) results
    :param Y: endogenous (dependent) variables
    :param X: exogenous (independent) variables
    :param df_resid: degrees of freedom
    :return: dictionary with regression results

    See ref: http://statsmodels.sourceforge.net/devel/_modules/statsmodels/regression/linear_model.html#OLS
    """

    # Get estimates for beta coefficients using result beta_hat = [(X'X)^-1]X'Y
    G = np.linalg.inv(np.dot(X.T, X))  # [(X'X)^-1] term, aka variance-covariance factor
    params = np.dot(G, np.dot(X.T, Y))  # beta_hat

    # Get estimates for epsilon residuals using result resid_hat = Y - X*beta_hat
    resid_hat = Y - np.dot(X, params)

    # Get t-statistics for the ADF using result tvalue = beta_hat / bse, where bse is the standard error of beta_hat
    # Note: must first estimate the standard error using result sqrt(diag[kron(G, ols_scale)]) where:
    # G: as above
    # ols_scale: the unbiased estimate of the residuals covariance (scaled by the residual degrees of freedom)
    # kron: kronecker product
    # diag: diagonal elements
    # See ref above or p.29 in  for more info

    nobs = len(resid_hat)  # number of observations

    # The residual degree of freedom, defined as the number of observations minus the rank of the regressor matrix
    if df_resid is None:  # if degrees of freedom not specified, then set to number of observations in resid_hat
        df_resid = my_df_resid(exog=X)
    else:
        df_resid = my_df_resid(my_df=df_resid)

    # Calculate other useful values to store in result dictionary
    ssr = np.dot(resid_hat, resid_hat.T)  # ee' term
    ols_scale = ssr / df_resid  # ee' term must be scaled by df_resid to obtain unbiased estimate
    cov_params = np.kron(G, ols_scale)  # covariance matrix of parameters
    bvar = np.diag(cov_params)  # entries on the diagonal of the covariance matrix  are the variances
    bse = np.sqrt(bvar)  # must take square root to get standard error
    tvalue = params / bse  # t-statistic for a given parameter estimate
    nobs2 = nobs / 2.0
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(1 / (2 * nobs2) * \
                                                      np.dot(np.transpose(Y - np.dot(X, params)),
                                                             (Y - np.dot(X, params)))) - nobs2 # log-likelihood function of OLS model
    df_model = X.shape[1]  # degrees of freedom of model
    aic = -2 * llf + 2 * df_model  # Akaike information criterion

    dic = {
        'X': X,
        'Y': Y,
        'params': params,
        'resid_hat': resid_hat,
        'nobs': nobs,
        'df_resid': df_resid,
        'ssr': ssr,
        'ols_scale': ols_scale,
        'cov_params': cov_params,
        'bse': bse,
        'tvalue': tvalue,
        'llf': llf,
        'df_model': df_model,
        'aic': aic
           }

    return dic


def my_AR(endog, maxlag, trend=None):
    """
    Autoregressive model implementation, aka AR(p)
    :param endog: the dependent variables
    :param maxlag: maximum number of lags to use
    :param trend: 'c': add constant, 'nc' or None: no constant
    :return: my_OLS dictionary

    See ref: http://statsmodels.sourceforge.net/stable/_modules/statsmodels/tsa/ar_model.html#AR
    """
    # Dependent data matrix
    Y = endog[maxlag:]  # has observations for the fit p lags removed
    # Explanatory data matrix
    X = lagmat(endog, maxlag, trim='both')
    if trend is not None:
        X = add_trend(X, prepend=True, trend=trend)  # prepends puts trend column at the beginning

    # Get degrees of freedom
    nobs = len(Y)  # number of observations
    k_ar = maxlag  # number of lags used, which affects number of observations
    k_trend = 0  # the number of trend terms included 'nc'=0, 'c'=1
    df_resid_AR = nobs - k_ar - k_trend  # degrees of freedom

    return my_OLS(Y, X, df_resid=df_resid_AR)


def my_adfuller(y, maxlag=None, autolag=None):
    """
    Augmented Dickey-Fuller test (it reduces to non-augmented version if maxlag=0: dY_t = phi*Y_{t-1} + eps_t)
    e.g. maxlag=1 model: dY_t = phi*Y_{t-1} + phi_1*dY_{t-1} + eps_t
    NOTE: for simplicity this implementation does not allow to add a constant or time-dependence term,
    also, only the
    :param y: time series which wants to be checked for stationarity
    :param maxlag: maximum lag to include
    :param autolag: set True if you want to use AIC criterion to choose optimal lag, set False or None if maxlag should be used
    :return: dictionary with OLS results
    """
    y = np.asarray(y)  # ensure it is in array form
    ydiff = np.diff(y)  # get the differences (dY_t term)
    ydall = lagmat(ydiff[:, None], maxlag, trim='both', original='in')  # lagged differences (dY_{t-k} terms)
    nobs = ydall.shape[0]  # number of observations
    ydall[:, 0] = y[-nobs - 1:-1]  # replace 0 ydiff with level of y (Y_{t-1} term)
    ydshort = ydiff[-nobs:]  # level up the dimensions of ydiff to match nobs

    Y = ydshort  # endogenous var
    X = ydall[:, :maxlag + 1]  # exogenous var

    result = my_OLS(Y, X, df_resid=None)  # do the usual regression using OLS to estimate parameters

    # Add a few other info to the results dictionary
    result['adfstat'] = result['tvalue'][0] # define adfstat as tvalue of phi coefficient
    result['maxlag'] = maxlag

    return result


def get_optimal_lag(y, maxlag):
    """
    Returns the optimal lag for an adfuller model applied to the series y
    :param y: array, the series to apply the adfuller model
    :param maxlag: the maximum lag to search for the lowest information criterion 'aic'
    :return: the minimum of the aic values, along with the corresponding lag
    """
    # Returns the results for the lag length that maximimizes the info criterion
    results = {}
    startlag = 0  # loop from 0 up to maxlag
    for lag in range(startlag, startlag + maxlag + 1):
        results[lag] = my_adfuller(y, maxlag=lag)
        # Cross-check results vs statsmodels result - warning: small difference observed
        py_result = adfuller(x=y, maxlag=lag, regression='nc', autolag=None, regresults=True)[3].resols
        print 'lag={0}, aic={1}, py_aic={2}'.format(lag, results[lag]['aic'], py_result.aic)
        # print 'lag={0}, llf={1}, py_llf={2}'.format(lag, results[lag]['llf'], py_result.llf)
    icbest, bestlag = min((v['aic'], k) for k, v in results.iteritems())

    return icbest, bestlag


# py_result[3].resols.llf
# py_result[3].resols.aic
# py_result[3].maxlag
# py_result[3].usedlag
# py_result[3].icbest
# Cross-check:
# lag
# py_result = adfuller(x=y, maxlag=lag, regression='nc', autolag=None, regresults=True)
# x = my_adfuller(y, maxlag=lag)
# py_result[3].resols.aic
# py_result[3].resols.llf
# x['aic']
# x['llf']

# In[23]: py_result = adfuller(x=y, maxlag=0, regression='nc', autolag='AIC', regresults=True)
# In[24]: get_optimal_lag(y, maxlag=0)
# Out[24]: (7.2308967346625259, 0)
# In[25]: py_result[3].icbest
# Out[25]: 7.2308967346625259
# In[26]: py_result[3].usedlag
# Out[26]: 0L

# def my_llf(ols_result):
#     """
#     Returns the value of the Gaussian log-likelihood function at params
#     :param ols_result: result from my_OLS function above
#     :return llf:
#     """
#     # nobs2 = ols_result['nobs'] / 2.0
#     # llf = -np.log(ssr) * nobs2  # concentrated likelihood
#     # llf -= (1 + np.log(np.pi / nobs2)) * nobs2  # with likelihood constant
#     # llf -= 0.5 * np.sum(np.log(sigma))
#
#     exog = ols_result['X']
#     endog = ols_result['Y']
#     llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(1 / (2 * nobs2) *\
#            np.dot(np.transpose(endog - np.dot(exog, params)), (endog - np.dot(exog, params)))) - nobs2
#     # llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(
#     #     1 / (2 * nobs2) * np.dot(np.transpose(my_result['Y'] - np.dot(my_result['X'], my_result['params'])),
#     #                              (my_result['Y'] - np.dot(my_result['X'], py_result[3].resols.params)))) - nobs2
#
#     return llf


# =================   COMPARE MY ADF VS STATSMODELS   =================

# My model
y = Y_t1.head(10)
my_result = my_adfuller(y, maxlag=3)

# Statsmodels
from statsmodels.tsa.stattools import adfuller
py_result = adfuller(x=y, maxlag=3, regression='nc', autolag=None, regresults=True)

sys.exit()

# # Cross-check for consistency
# my_result['adfstat'] = py_result[0] == py_result[3].resols.tvalues[0]
# my_result['bse'] == py_result[3].resols.bse
# my_result['ols_scale'] == py_result[3].resols.scale
# my_result['df_resid'] == py_result[3].resols.df_resid
# my_result['cov_params'] == py_result[3].resols.cov_params(scale=py_result[3].resols.scale)
# my_result['llf'] == py_result[3].resols.llf
# my_result['X'].shape[1] == py_result[3].resols.df_model  # rank of regressor matrix

# =================   COMPARE MY AR(p) VS STATSMODELS   =================

endog = Y_t1.head(10)
maxlag = 3

my_result = my_AR(endog=endog, maxlag=maxlag)

# Compare to statsmodels AR(p) ('cmle' - conditional maximum likelihood estimation is default method)
py_result = AR(np.array(endog)).fit(maxlag=maxlag, trend='nc')  # use only specified lags and remove constant

# Print fitted params and residuals of model, should be equivalent (or very close) to estimates above
print "\
AR.fit.params={0} \n MY params={1} \n\
AR.fit.resid={2} \n MY resid_hat={3} \n\
AR.fit.nobs={4} \n MY nobs={5} \n\
AR.fit.cov_params(scale=ols_scale)={6} \n MY cov_params={7} \n\
AR.fit.bse={8} \n MY bse={9} \n\
AR.fit.tvalues={10} \n MY tvalue={11} \n\
".format(
    py_result.params, my_result['params'],
    py_result.resid, np.array(my_result['resid_hat']),
    py_result.nobs, my_result['nobs'],
    py_result.cov_params(scale=my_result['ols_scale']), my_result['cov_params'],
    py_result.bse, my_result['bse'],
    py_result.tvalues, my_result['tvalue']
)





