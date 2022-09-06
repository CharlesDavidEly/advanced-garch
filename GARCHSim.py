import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy as sp



# Model types:
# ARCH(p) ->        𝛿 = 2, μ = mean, γ = 0, β = 0
# GARCH(p,q) ->     𝛿 = 2, μ = mean, γ = 0
# GJR-GARCH(p,q) -> 𝛿 = 2
# APARCH(p,q) ->    NO ASSUMPTIONS

'''
'''
# Specify the model
modelType = "APARCH"
pGARCH = 1
qGARCH = 1

# Specify the sample
ticker = "^GSPC"
start = "2011-12-31"
end = "2021-12-31"

# Specify the simulation days
simDays = 100000
'''
'''

# Add check on pGARCH's value here (must be int >= 1)
if modelType == "ARCH":
    modelName = modelType + "(" + str(pGARCH) + ")"
    qGARCH = 0
else:
    modelName = modelType + "(" + str(pGARCH) + "," + str(qGARCH) + ")"
    # Add check on qGARCH's value here (must be int >= 1)



# Download data
prices = yf.download(ticker,start,end)["Close"]
priceNum = prices.shape[0]

# Calculate returns
returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1
returnNum = returns.shape[0]

# Starting parameter values (sample μ and σ)
mean = np.average(returns)
median = np.median(returns)
vol = np.std(returns)
var = vol**2
skew = sp.stats.skew(returns)
kurt = sp.stats.kurtosis(returns)


# Set up guesses

#def aparch_mle(params):
def aparch_mle(params,*args):

    # Specify model parameters
    omega = params[0]
    alpha = params[1]
    match args[0]:
        case "ARCH":
            beta = args[6]
            gamma = args[5]
            mu = args[4]
            delta = args[3]
        case "GARCH":
            beta = params[2]
            gamma = args[5]
            mu = args[4]
            delta = args[3]
        case "GJR-GARCH":
            beta = params[2]
            gamma = params[3]
            mu = params[4]
            delta = args[3]
        case "APARCH":
            # Change number here
            beta = params[2]
            gamma = params[3]
            mu = params[4]
            delta = params[5]

    # Calculate long-run volatility
    long_run_var = omega/(1 - alpha - beta)
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



# Handle bounds (later constraints)
bMu = (-1,1)
bOmega = (0,var)
b = (0,1)
bDelta = (0,10)


# Handle constantParams and initialGuesses
match modelType:
    case "ARCH":
        #                                         𝛿  μ   γ β                   
        constantParams = (modelType,pGARCH,qGARCH,2,mean,0,0)
        #                 ω  α
        initialGuesses = [0,0.5]
        paramBounds = (bOmega,b)
    case "GARCH":
        constantParams = (modelType,pGARCH,qGARCH,2,mean,0)
        initialGuesses = [0,0.2,0.7]
        paramBounds = (bOmega,b,b)
    case "GJR-GARCH":
        constantParams = (modelType,pGARCH,qGARCH,2)
        initialGuesses = [0,0.1,0.7,0.1,median]
        paramBounds = (bOmega,b,b,b,bMu)
    case "APARCH":
        constantParams = (modelType,pGARCH,qGARCH)
        initialGuesses = [0,0.1,0.7,0.1,median,2]
        paramBounds = (bOmega,b,b,b,bMu,bDelta)

# Maximize log-likelihood
res = sp.optimize.minimize(aparch_mle,initialGuesses,args=constantParams,bounds=paramBounds,method="Nelder-Mead",options={"disp":True})
# Alpha + gamma should = around 0.2


# Set default parameters
beta = 0
gamma = 0
mu = mean
delta = 2

# Retrieve optimal parameters
params = res.x
omega = res.x[0]
alpha = res.x[1]
if modelType != "ARCH":
    beta = res.x[2]
    if modelType != "GARCH":
        gamma = res.x[3]
        mu = res.x[4]
        if modelType != "GJR-GARCH":
            delta = res.x[5]
log_likelihood = -float(res.fun)
k = res.x.shape[0] + 1
bic = k*np.log(returnNum) - 2*log_likelihood
aic = 2*k - 2*log_likelihood

# Calculate realized and conditional volatility for optimal parameters
suitTest = alpha + beta
long_run_var = omega/(1 - alpha - beta)
long_run_vol = long_run_var**(1/delta)
resid = returns - mu
realized = abs(resid)
conditional = np.zeros(len(returns))
conditional[0] = long_run_vol
for t in range(1,len(returns)):
    conditional[t] = (omega + alpha*(abs(resid[t-1]) - gamma*resid[t-1])**delta + beta*conditional[t-1]**delta)**(1/delta)

# Print optimal parameters
print("")
print(ticker + " " + modelName + " model parameters:")
print("omega: " + str(round(omega,6)))
print("alpha: " + str(round(alpha,4)))
if modelType == "ARCH":
    print("---DEFAULTS---")
print("beta:  " + str(round(beta,4)))
if modelType == "GARCH":
    print("---DEFAULTS---")
print("gamma: " + str(round(gamma,4)))
print("mu:    " + str(round(mu,6)))
if modelType == "GJR-GARCH":
    print("---DEFAULTS---")
print("delta: " + str(round(delta,4)))

print("")
print(ticker + " " + modelName + " model results:")
print("suitability:    " + str(round(suitTest,4)))
print("long-run vol:   " + str(round(long_run_vol,4)))
print("log-likelihood: " + str(round(log_likelihood,4)))
print("BIC:            " + str(round(bic,4)))
print("AIC:            " + str(round(aic,4)))
print("")

# Visualize volatility
plt.figure(1)
plt.rc("xtick",labelsize=10)
plt.plot(prices.index[1:],realized,label="Empirical Realized")
plt.plot(prices.index[1:],conditional,label=modelName + " Conditional")
plt.title(label=ticker + " " + modelName + " Volatility")
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
print(ticker + " " + modelName + " Monte Carlo Simulation (" + str(1) + " " + str(simDays) + "-Day Runs):")
print("   Actual   | Expected  | Error    | Commentary")
print("μ: " + f"{actualMu:.6f}" + " | " + f"{mean:.6f}" + "  | " + muDir + f"{muError:.2f}" + "%" + "   | Should be equal on average")
print("σ: " + f"{actualSigma:.6f}" + " | " + f"{vol:.6f}" + "  | " + sigmaDir + f"{sigmaError:.2f}" + "%" + "   | Should be equal on average")
print("S: " + f"{actualSkew:.6f}" + " | " + f"{skew:.6f}" + " | " + skewDir + f"{skewError:.2f}" + "%" + " | Should be wrong (vol clustering actually tends to push it up from the sim default of 0)")
print("K: " + f"{actualKurt:.6f}" + " | " + f"{kurt:.6f}" + " | " + kurtDir + f"{kurtError:.2f}" + "%" + "  | Should be > 3 due to vol clustering but not high enough")

plt.figure(2)
plt.rc("xtick",labelsize=10)
plt.plot(range(simDays),simVols,label=modelName + " Conditional")
plt.plot(range(simDays),constVols,label=modelName + " LR Implied")
plt.plot(range(simDays),GARCHVols,label=modelName + " Realized")
plt.plot(range(simDays),realVols,label="Empirical Realized")
plt.title(label=ticker + " " + modelName + " Volatility")
plt.legend()
plt.show()



