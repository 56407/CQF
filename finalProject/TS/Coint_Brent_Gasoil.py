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

# Y_t = pd.Series(index=range(M), data=Y_t)
Y_t1 = pd.Series(Y_t1, name='Y_t1')
Y_t2 = pd.Series(Y_t2, name='Y_t2')
Y_t3 = pd.Series(Y_t3, name='Y_t3')

# =================   REGRESSION   =================

# AR(p)
sys.exit()

from statsmodels.tsa.tsatools import (lagmat, add_trend)
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.ar_model import AR

maxlag = 3
# trend = 'c'
trend = None

endog = Y_t1.head(10)
nobs = len(endog)

# Dependent data matrix has observations for the fit p lags removed
Y = endog[maxlag:]

# Explanatory data matrix - first row of ones and
X = lagmat(endog, maxlag, trim='both')

# Add a constant or linear trend
if trend is not None:
    X = add_trend(X, prepend=True, trend=trend)  # put trend column at the beginning of X

# Get estimates for beta (coefficients) and epsilon (residuals)
X_prime = X.transpose()
beta_hat = np.dot(np.linalg.inv(np.dot(X_prime, X)), np.dot(X_prime, Y))
# resid_hat = Y.reshape(Y.shape[0], 1) - np.dot(X, beta_hat)
resid_hat = Y - np.dot(X, beta_hat)
print "beta_hat={0}\nresid_hat=\n{1}".format(beta_hat, resid_hat)


# Compare to statsmodels AR(p) model (using 'cmle' default method)
# fit = OLS(Y, X).fit()
fit = AR(np.array(endog)).fit(maxlag=maxlag, trend='nc')  # remove constant from model
# Print fitted params and residuals of model, should be equivalent (or very close) to estimates above
print "AR.fit.params={0}\nAR.fit.resid=\n{1}".format(fit.params, fit.resid)

# Build ADF test statistic
gamma = 1 - beta_hat

# Estimate Standard Error
Y_hat = np.dot(X, beta_hat) + resid_hat
n = len(X)
X_bar = X.mean()
se = math.sqrt(np.dot((Y - Y_hat), (Y - Y_hat)) / (n - 2)) / math.sqrt(np.dot((X - X_bar), (X - X_bar)))
