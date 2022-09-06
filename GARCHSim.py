import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy as sp



# Specify the model
model = "APARCH(1,1)"

# Specify the sample
ticker = "^GSPC"
start = "2011-12-31"
end = "2021-12-31"

# Specify the simulation days
simDays = 100000

# Download data
prices = yf.download(ticker,start,end)["Close"]

# Calculate returns
returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1

# Starting parameter values (sample μ and σ)
mean = np.average(returns)
median = np.median(returns)
vol = np.std(returns)
var = vol**2
skew = sp.stats.skew(returns)
kurt = sp.stats.kurtosis(returns)

def aparch_mle(params):

    # Specify model parameters
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    gamma = params[3]
    beta = params[4]
    delta = params[5]

    # Calculate long-run volatility
    long_run_var = omega/(1 - alpha * (1 - gamma)**delta - beta)
    long_run_vol = long_run_var**(1/delta)

    # Calculate realized and conditional volatility
    resid = returns - mu
    realized = abs(resid)
    conditional = np.zeros(len(returns))
    conditional[0] = long_run_vol
    for t in range(1,len(returns)):
        conditional[t] = (omega + alpha*(abs(resid[t-1]) - gamma*resid[t-1])**delta + beta*conditional[t-1]**delta)**(1/delta)

    # Calculate log-likelihood
    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realized**2/(2*conditional**2))
    log_likelihood = -np.sum(np.log(likelihood))
    return log_likelihood



# Maximize log-likelihood
bMu = (-1,1)
bOmega = (0,var)
b = (0,1)
bDelta = (0,10)
res = sp.optimize.minimize(aparch_mle, [median,0,0,0,0.75,2], bounds=(bMu,bOmega,b,b,b,bDelta), method="Nelder-Mead")
# Alpha + gamma should = around 0.2

# Retrieve optimal parameters
params = res.x
mu = res.x[0]
omega = res.x[1]
alpha = res.x[2]
gamma = res.x[3]
beta = res.x[4]
delta = res.x[5]
log_likelihood = -float(res.fun)

# Calculate realized and conditional volatility for optimal parameters
suitTest = alpha * (1 - gamma)**delta - beta
long_run_var = omega/(1 - alpha * (1 - gamma)**delta - beta)
long_run_vol = long_run_var**(1/delta)
resid = returns - mu
realized = abs(resid)
conditional = np.zeros(len(returns))
conditional[0] = long_run_vol
for t in range(1,len(returns)):
    conditional[t] = (omega + alpha*(abs(resid[t-1]) - gamma*resid[t-1])**delta + beta*conditional[t-1]**delta)**(1/delta)

# Print optimal parameters
print(ticker + " " + model + " model parameters:")
print("suit: " + str(round(suitTest,4)))
print("mu: " + str(round(mu,6)))
print("omega: " + str(round(omega,6)))
print("alpha: " + str(round(alpha,4)))
print("gamma: " + str(round(gamma,4)))
print("beta: " + str(round(beta,4)))
print("delta: " + str(round(delta,4)))
print("long-run vol: " + str(round(long_run_vol,4)))
print("log-likelihood: " + str(round(log_likelihood,4)))
print("")

# Visualize volatility
plt.figure(1)
plt.rc("xtick",labelsize=10)
plt.plot(prices.index[1:],realized,label="Realized")
plt.plot(prices.index[1:],conditional,label="Conditional")
plt.title(label=ticker + " Volatility")
plt.legend()
plt.show()



def random_num():
    randNum = 0
    while randNum == 0:
        randNum = np.random.rand()
    return randNum


# Get last realized period's needed data
lastRealReturn = returns[-1]
lastRealResid = lastRealReturn - mu
lastRealVol = realized[-1]
lastRealVar = (lastRealVol)**2
lastRealPrice = prices[-1]

# Assign first lookback values
lastReturn = lastRealReturn
lastResid = lastRealResid
lastVol = lastRealVol
lastVar = lastRealVar
lastPrice = lastRealPrice

# Start Monte Carlo simulation loop
#drift = mu - long_run_var/2
drift = mean - var/2
simVars = np.zeros(simDays)
realVols = np.zeros(simDays)
constVols = np.zeros(simDays)
simVols = np.zeros(simDays)
simZs = np.zeros(simDays)
simPrices = np.zeros(simDays)
simReturns = np.zeros(simDays)
for t in range(simDays):
    
    # Determine this period's var
    if lastReturn < mu:
        i = 1
    else:
        i = 0
    simVars[t] = omega + alpha*(abs(lastResid) - gamma*lastResid)**delta + beta*lastVol**delta

    # Assign other values
    realVols[t] = vol
    constVols[t] = long_run_vol

    simVols[t] = (simVars[t])**(1/delta)
    simZs[t] = sp.stats.norm.ppf(random_num())
    simPrices[t] = lastPrice * math.exp(drift + simVols[t]*simZs[t])
    simReturns[t] = simPrices[t]/lastPrice - 1

    # Reset last values
    lastReturn = simReturns[t]
    lastResid = lastReturn - mu
    lastVol = simVols[t]
    lastVar = simVars[t]
    lastPrice = simPrices[t]


# Test actual vs. expected values here
actualMu = np.average(simReturns)
actualSigma = np.std(simReturns)
actualSkew = sp.stats.skew(simReturns)
actualKurt = sp.stats.kurtosis(simReturns)

muError = abs((actualMu-mean)/mean) * 100
sigmaError = abs((actualSigma-vol)/vol) * 100
skewError = abs((actualSkew-skew)/skew) * 100
kurtError = abs((actualKurt-kurt)/kurt) * 100

if actualMu-mean > 0:
    muDir = "+"
else:
    muDir = "-"
if actualSigma-vol > 0:
    sigmaDir = "+"
else:
    sigmaDir = "-"
if actualSkew-skew > 0:
    skewDir = "+"
else:
    skewDir = "-"
if actualKurt-kurt > 0:
    kurtDir = "+"
else:
    kurtDir = "-"

GARCHVols = np.zeros(simDays)
for t in range(simDays):
    GARCHVols[t] = actualSigma

# Print Monte Carlo simulation
print(ticker + " " + model + " Monte Carlo Simulation (" + str(1) + " " + str(simDays) + "-Day Runs):")
print("   Actual   | Expected  | Error    | Commentary")
print("μ: " + f"{actualMu:.6f}" + " | " + f"{mean:.6f}" + "  | " + muDir + f"{muError:.2f}" + "%" + "   | Should be equal on average")
print("σ: " + f"{actualSigma:.6f}" + " | " + f"{vol:.6f}" + "  | " + sigmaDir + f"{sigmaError:.2f}" + "%" + "   | Should be equal on average")
print("S: " + f"{actualSkew:.6f}" + " | " + f"{skew:.6f}" + " | " + skewDir + f"{skewError:.2f}" + "%" + " | Should be wrong (vol clustering actually tends to push it up from the sim default of 0 for some reason)")
print("K: " + f"{actualKurt:.6f}" + " | " + f"{kurt:.6f}" + " | " + kurtDir + f"{kurtError:.2f}" + "%" + "  | Should be > 3 due to vol clustering but not high enough")

plt.figure(2)
plt.rc("xtick",labelsize=10)
plt.plot(range(simDays),simVols,label=model + " Conditional")
plt.plot(range(simDays),constVols,label=model + " LR Implied")
plt.plot(range(simDays),GARCHVols,label=model + " Realized")
plt.plot(range(simDays),realVols,label="Empirical Realized")
plt.title(label=ticker + " Volatility")
plt.legend()
plt.show()



