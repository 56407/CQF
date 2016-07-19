import sys
from datetime import datetime

import pandas as pd
import numpy as np

import math
from math import exp, sqrt, log

# To compare my result to statsmodels result
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.tsatools import (lagmat, add_trend)  # helper functions to add lags and trends

# Debug
from IPython import embed

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# =================   MULTIVARIATE REGRESSION   =================

# Main refs used:
# http://www.ats.ucla.edu/stat/sca/finn/finn4.pdf
# CQF_January_2016_M4S11_Annotated.pdf
# http://statsmodels.sourceforge.net/stable/_modules/statsmodels/tsa/ar_model.html#AR



def my_OLS(Y, X):
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
    # Also calculate other useful values to store in result dictionary

    nobs = X.shape[0]
    rank = X.shape[1]
    df_resid = np.float(nobs - rank)
    ssr = np.dot(resid_hat, resid_hat.T)  # ee' term
    sigma = ssr / nobs  # aka 'sigma2' or 'scale' in statsmodels (MLE estimator for cov matrix)
    ols_scale = ssr / df_resid  # OLS estimator for cov matrix
    cov_params = np.kron(G, ols_scale)  # covariance matrix of parameters
    bvar = np.diag(cov_params)  # entries on the diagonal of the covariance matrix  are the variances
    bse = np.sqrt(bvar)  # must take square root to get standard error
    tvalue = params / bse  # t-statistic for a given parameter estimate
    nobs2 = nobs / 2.0
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(sigma) - nobs2  # log-likelihood function of OLS model
    df_model = rank  # degrees of freedom of model
    eigenvalues = np.roots(np.r_[1, -params])  # prepend 1 to -ve of params array to get characteristic equation
    roots = eigenvalues ** -1  # these are used to test for stability in is_stable method

    dic = {
        'X': X,
        'Y': Y,
        'params': params,
        'resid_hat': resid_hat,
        'nobs': nobs,
        'df_resid': df_resid,
        'ssr': ssr,
        'sigma': sigma,
        'ols_scale': ols_scale,
        'cov_params': cov_params,
        'bse': bse,
        'tvalue': tvalue,
        'llf': llf,
        'df_model': df_model,
        'roots': roots
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
    result = my_OLS(Y, X)
    # Akaike information criterion using statsmodel def for AR(p) (Lutkephol's definition)
    # note this is diff to adfuller's def
    result['aic'] = np.log(result['sigma']) + 2.0 * (1.0 + result['df_model']) / result['nobs']

    return result


def my_adfuller(y, maxlag=None, regression='c'):
    """
    Augmented Dickey-Fuller test (it reduces to non-augmented version if maxlag=0: dY_t = phi*Y_{t-1} + eps_t)
    e.g. maxlag=1 model: dY_t = phi*Y_{t-1} + phi_1*dY_{t-1} + eps_t
    NOTE: this implementation does not allow to add a time-dependence term
    :param y: time series which wants to be checked for stationarity
    :param maxlag: maximum lag to include
    :param regression: str {'c','nc'} Constant to include in regression
        * 'c' : constant only (default)
        * 'nc' : no constant, no trend
    :return: dictionary with OLS results
    """
    y = np.asarray(y)  # ensure it is in array form
    ydiff = np.diff(y)  # get the differences (dY_t term)
    ydall = lagmat(ydiff[:, None], maxlag, trim='both', original='in')  # lagged differences (dY_{t-k} terms)
    nobs = ydall.shape[0]  # number of observations
    ydall[:, 0] = y[-nobs - 1:-1]  # replace 0 ydiff with level of y (Y_{t-1} term)
    ydshort = ydiff[-nobs:]  # level up the dimensions of ydiff to match nobs

    Y = ydshort  # endogenous var
    if regression != 'nc':
        X = add_trend(ydall[:, :maxlag + 1], regression)  # exogenous var
    else:
        X = ydall[:, :maxlag + 1]  # exogenous var

    result = my_OLS(Y, X)  # do the usual regression using OLS to estimate parameters

    # Add a few other info to the results dictionary
    result['adfstat'] = result['tvalue'][0] # define adfstat as tvalue of phi coefficient
    result['maxlag'] = maxlag
    # Akaike information criterion using statsmodel def for adfuller - not this is different to the def in AR(p) model
    result['aic'] = -2 * result['llf'] + 2 * result['df_model']
    # result['aic'] = np.log(result['sigma']) + 2.0 * (1.0 + result['df_model']) / result['nobs']  # AR(p) def
    return result


def get_optimal_lag(y, maxlag, model):
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
        if model == 'adf':
            results[lag] = my_adfuller(y, maxlag=lag, regression='c')
        elif model == 'ar':
            results[lag] = my_AR(y, maxlag=lag)
        # print 'lag={0}, aic={1}'.format(lag, results[lag]['aic'])
        # # Cross-check results vs statsmodels result - warning: small difference observed
        # sm_result = adfuller(x=y, maxlag=lag, regression='c', autolag=None, regresults=True)[3].resols
        # print 'lag={0}, aic={1}, sm_aic={2}'.format(lag, results[lag]['aic'], sm_result.aic)
        # print 'lag={0}, nobs={1}, sm_nobs={2}'.format(lag, results[lag]['nobs'], sm_result.nobs)
        # print 'lag={0}, adfstat={1}, sm_adfstat={2}'.format(lag, results[lag]['adfstat'], sm_result.tvalues)
        # print 'lag={0}, llf={1}, sm_llf={2}'.format(lag, results[lag]['llf'], sm_result.llf)
    # Optimal lag is the one with lowest aic
    icbest, bestlag = min((v['aic'], k) for k, v in results.iteritems())
    # print 'bestlag={0}, icbest={1}'.format(bestlag, icbest)
    return bestlag, icbest

def is_stable(roots):
    """
    The roots of the AR process are the solution to
    (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0
    Stability requires that the roots in modulus lie outside the unit circle
    (or equivalently that modulus of eigenvalues are within unit circle)
    See ref: http://matthieustigler.github.io/Lectures/Lect2ARMA.pdf
    :param params: the coefficients of the VAR(p) or adfuller system
    :return: true/false depending on whether roots stable
    """
    if np.all(np.abs(roots) > 1):  # all must be True for it to pass test
        return True
    return False

if __name__ == "__main__":

    # Generate mean-reverting processes

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

    # =================   COMPARE MY OPTIMAL LAG VS STATSMODELS FOR ADF  =================
    sys.exit()
    y = Y_t3
    my_lag = 35

    # My result, try a maximum of lags equal to my_lag
    my_bestlag, my_icbest = get_optimal_lag(y, maxlag=my_lag, model='adf')

    # Statsmodel equivalent to optimise lag using aic criterion - will not use as some inconsistencies seen (see below)
    sm_bestlag = adfuller(x=y, maxlag=my_lag, regression='nc', autolag='AIC', regresults=True)[3].usedlag
    sm_icbest1 = adfuller(x=y, maxlag=sm_bestlag, regression='nc', autolag='AIC', regresults=True)[3].icbest
    sm_icbest2 = adfuller(x=y, maxlag=sm_bestlag, regression='nc', autolag=None, regresults=True)[3].resols.aic
    # sm_icbest1 and sm_icbest2 should be the same but are inconsitent

    # # Inconsistency observed in statsmodels for optimal lag selection using 'aic'
    # print 'sm_result[3].usedlag=', sm_result[3].usedlag  # should be == my_bestlag == my_result['maxlag'] but small diff observed
    # print 'sm_result[3].resols.df_model - 1=', sm_result[3].resols.df_model - 1  # equivalent to above
    # print 'sm_result[3].icbest=', sm_result[3].icbest  # should be  == my_icbest == my_result['aic']
    # print 'sm_result[3].resols.aic=', sm_result[3].resols.aic  # should be equivalent to above but for reason it isn't

    # =================   COMPARE MY ADF VS STATSMODELS   =================

    # Instead I use my optimised lag (my_bestlag), which gives consistent results with statsmodels too
    my_result = my_adfuller(y, maxlag=my_bestlag)
    sm_result = adfuller(x=y, maxlag=my_bestlag, regression='nc', autolag=None, regresults=True)
    print "my_result['maxlag']={0}, sm_result[3].usedlag={1}".format(my_result['maxlag'], sm_result[3].usedlag)
    print "my_result['aic']={0}, sm_result[3].resols.aic={1}".format(my_result['aic'], sm_result[3].resols.aic)

    # # Other cross-checks:
    # sm_result = adfuller(x=y, maxlag=lag, regression='nc', autolag=None, regresults=True)
    # my_result = my_adfuller(y, maxlag=lag)
    # sm_result[3].maxlag
    # sm_result[3].usedlag
    # sm_result[3].icbest
    # sm_result[3].resols.summary()
    # my_result['llf'] == sm_result[3].resols.llf
    # my_result['aic'] == sm_result[3].resols.aic
    # my_result['adfstat'] = sm_result[0] == sm_result[3].resols.tvalues[0]
    # my_result['bse'] == sm_result[3].resols.bse
    # my_result['ols_scale'] == sm_result[3].resols.scale
    # my_result['df_resid'] == sm_result[3].resols.df_resid
    # my_result['cov_params'] == sm_result[3].resols.cov_params(scale=sm_result[3].resols.scale)
    # my_result['llf'] == sm_result[3].resols.llf
    # my_result['X'].shape[1] == sm_result[3].resols.df_model  # rank of regressor matrix

    # =================   COMPARE MY OPTIMAL LAG VS STATSMODELS FOR AR(P)  =================

    y = Y_t3
    my_lag = 35

    # My result, try a maximum of lags equal to my_lag
    my_bestlag , my_icbest = get_optimal_lag(y, maxlag=my_lag, model='ar')

    # Statsmodel equivalent to optimise lag using aic criterion - will not use as some inconsistencies seen (see below)
    sm_bestlag = AR(np.array(y)).select_order(maxlag=my_lag, ic='aic', trend='nc', method='cmle')
    sm_icbest = AR(np.array(y)).fit(maxlag=sm_bestlag, trend='nc', method='cmle').aic
    # above line inconsistent since this isn't the lower aic found, e.f. maxlag=29 for Y_t3 gives aic=-2.45, which is lower
    # than sm_bestlag=7 with aic=-2.40

    # =================   COMPARE MY AR(P) VS STATSMODELS   =================

    # Instead I use my optimised lag (my_bestlag), which gives consistent results with statsmodels too

    my_result = my_AR(endog=y, maxlag=my_bestlag)
    sm_result = AR(np.array(y)).fit(maxlag=my_bestlag, trend='nc', method='cmle')

    # # Cross-checks
    # print "\
    # AR.fit.params={0} \n MY params={1} \n\
    # AR.fit.llf={2} \n MY llf={3} \n\
    # AR.fit.nobs={4} \n MY nobs={5} \n\
    # AR.fit.cov_params(scale=ols_scale)={6} \n MY cov_params={7} \n\
    # AR.fit.bse={8} \n MY bse={9} \n\
    # AR.fit.tvalues={10} \n MY tvalue={11} \n\
    # ".format(
    #     sm_result.params, my_result['params'],
    #     sm_result.llf, np.array(my_result['llf']),
    #     sm_result.nobs, my_result['nobs'],
    #     sm_result.cov_params(scale=my_result['ols_scale']), my_result['cov_params'],
    #     sm_result.bse, my_result['bse'],
    #     sm_result.tvalues, my_result['tvalue']
    # )

    # =================   COMPARE STABILITY CHECK VS STATSMODELS   =================

    # Cross-checks:
    # sm_result.roots == my_result['roots']
    # is_stable(my_result['roots']) == is_stable(sm_result.roots)
    print 'passes stability check:', is_stable(my_result['roots'])



