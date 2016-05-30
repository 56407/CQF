######################################################################
# 2015. Richard Diamond. Quieries to r.diamond@cqf.com               #
# Models are specified and validated but any use is at your own risk #
######################################################################

# Install packages (once)
install.packages("quantmod") # Includes object types xts, TTR and functions for time series analsysis.
install.packages("urca") # Includes the ur.df() function for the Dickey-Fuller Unit Root test.

# Load required library
library(quantmod)
library(zoo)
library(urca)

# Delete any existing data
rm(list = ls(all.names = TRUE))
setwd("~/Documents/Dev R")

# COINTEGRATION IN SPOT RATES (with AUTOCORRELATION PROFILE) // # COINTEGRATION IN FORWARD RATES (with AUTOCORRELATION PROFILE)

curve.zoo = read.zoo("spot_curve.csv", header=TRUE, sep=",", format = "%d/%m/%y") # %y Year (2 digit) // %Y Year (4 digit)
curve.zoo = curve.zoo[complete.cases(curve.zoo),] # remove empty lines, also  curve.zoo = curve.zoo[rowSums(is.na(curve.zoo)) == 0,]

# SELECT SUBSAMPLES .this
#head(curve.zoo) #tail(curve.zoo)
curve.this = window(curve.zoo, start=as.Date("2013-5-30"), end=as.Date("2015-5-30")) 
# Alternative method curve.this = curve.zoo[as.Date(c("2015-1-1", "2015-5-30"))] -- not working, needs correction in index


# PRELIMINARY JOHANSEN TEST (incorporates Johansen Procedure)

#RD: Running Johansen Test for the full matrix up to 25Y tenor -- a lot of cointegration, approx r<=40 
#RD: Let's pick up tenors 0.08, 25. Result: no Johansen cointegration
#RD: As we include more tenors, ONE or more cointegrating relationship transpires (one but non-unique)
curve2.this = curve.this[, colnames(curve.this) %in% c("X0.08","X1", "X3", "X6", "X10", "X20", "X25")] 

curve2.this = curve.this[, colnames(curve.this) %in% c("X0.08","X1","X2")] # cointegrated all data // not cointegrated 2013-05 to 2015-05 ***

curve2.this = curve.this[, colnames(curve.this) %in% c("X0.08","X1")] # cointegrated all data // not cointegrated 2013-05 to 2015-05 ?
curve2.this = curve.this[, colnames(curve.this) %in% c("X1","X2")] # not cointegrated all data // not cointegrated 2013-05 to 2015-05 ? 

curve2.this = curve.this[, colnames(curve.this) %in% c("X10", "X20", "X25")] # not cointegrated all data // cointegrated 2013-05 to 2015-05 ***

curve2.this = curve.this[, colnames(curve.this) %in% c("X10", "X20")] # N/A // cointegrated 2013-05 to 2015-05
curve2.this = curve.this[, colnames(curve.this) %in% c("X10", "X25")] # N/A // cointegrated 2013-05 to 2015-05 MODEL CHOICE for important tenors

curve2.this = curve.this[, colnames(curve.this) %in% c("X20", "X25")] # N/A // near-cointegrated 2013-05 to 2015-05



#RD: Can formally find lag p by fitting each series to AR process and calculating AIC/BIC
for(k in seq(2,5)) {
  
  print("######################################")
  print(paste("#              Lag = ",k,"               #",sep=""))
  print("######################################")
  
  # Run Johansen Maximum Eigen Statistic Test on prices with trend and lag of k
  print(summary(ca.jo(curve2.this, type="eigen", ecdet="trend", K=k)))
  
  # Run Johansen Trace Statistic Test on prices with trend and lag of k
  print(summary(ca.jo(curve2.this, type="trace", ecdet="trend", K=k)))

}

#VECM from Johansen Test and Procedure

johansen.test = ca.jo(curve2.this, ecdet = "const", type="eigen", K=2, spec="longrun")
cajools(johansen.test) # OLS regression of unrestricted VECM 
cajorls(johansen.test) # OLS regression of restricted VECM -- EC-term instead of differences Delta Y_t
