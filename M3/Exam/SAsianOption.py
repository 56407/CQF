import numpy as np
import pandas as pd

# Parameters

S0 = 100.  # initial value
K = 100.  # strike price
T = 1.0  # maturity
r = 0.05  # risk-free interest rate
sigma = 0.2  # volatility
M = 50  # number of time steps
dt = T / M  # time step size
I = 100  # number of paths to simulate

# df = pd.DataFrame(rn, index=list(len(rn)))

shape = (M + 1, I)
S = np.zeros((M + 1, I), dtype=np.float)

S[0] = S0
np.random.seed(10000)
rand1 = np.random.standard_normal(shape)

for t in xrange(1, M + 1, 1):
    S[t] = S[t - 1] * (
    np.exp((r - rj - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rand1[t]) + (np.exp(mu + delta * rand2[t]) - 1) *
    rand3[t])
