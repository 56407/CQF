from datetime import datetime
import pandas as pd
import numpy as np

# Stats models
import statsmodels.tsa.stattools as ts  # for ADF test and coint function
import statsmodels.api as sm  # for OLS

# Data
import Quandl as quandl

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)


# # Get quandl data from Steven continuous series, method: Roll on Last Trading Day, No Price Adjustment
# df_gasoil = quandl.get("SCF/ICE_G1_EN", authtoken="-uuuQ3n6PchShn5Pcu9T", trim_start = "2009-01-01", trim_end="2016-06-30")
# df_brent = quandl.get("SCF/ICE_B1_EN", authtoken="-uuuQ3n6PchShn5Pcu9T", trim_start = "2009-01-01", trim_end="2016-06-30")
#
# df_gasoil.to_csv("data/quandl_gasoil_010109_300616.csv")
# df_brent.to_csv("data/quandl_brent_010109_300616.csv")

