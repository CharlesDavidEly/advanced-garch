import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy as sp



'''
'''
# Specify the sample
ticker = "^GSPC"
start = "2011-12-31"
end = "2021-12-31"

# Specify the models to test
doARCH = True
doGARCH = True
doGJRGARCH = True
doAPARCH = True

# Sort by LL, BIC, or AIC
sortBy = "BIC"

# Specify the simulation days
simDays = 1000000
'''
'''



# Model types:
# ARCH(p) ->        𝛿 = 2, μ = mean, γ = 0, β = 0
# GARCH(p,q) ->     𝛿 = 2, μ = mean, γ = 0
# GJR-GARCH(p,q) -> 𝛿 = 2
# APARCH(p,q) ->    NO ASSUMPTIONS
modelList = []
if doARCH:
    modelList.append("ARCH")
if doGARCH:
    modelList.append("GARCH")
if doGJRGARCH:
    modelList.append("GJR-GARCH")
if doAPARCH:
    modelList.append("APARCH")



def random_num():
    randNum = 0
    while randNum == 0:
        randNum = np.random.rand()
    return randNum

def uncond_vol(omega,alpha,beta,gamma,delta):
    #uncondVar = omega/(1 - alpha - beta)
    uncondVar = omega/(1 - alpha - gamma/2 - beta)
    uncondVol = uncondVar**(1/delta)
    return uncondVol

def cond_vol(omega,alpha,beta,gamma,delta,lastResid,lastCond):
    condVol = (omega + alpha*(abs(lastResid) - gamma*lastResid)**delta + beta*lastCond**delta)**(1/delta)
    return condVol

def aparch_mle(params,*args):

    # Isolate returns
    returns = args[0]
    # Specify model parameters
    omega = params[0]
    alpha = params[1]
    match args[1]:
        case "ARCH":
            beta = args[7]
            gamma = args[6]
            mu = args[5]
            delta = args[4]
        case "GARCH":
            beta = params[2]
            gamma = args[6]
            mu = args[5]
            delta = args[4]
        case "GJR-GARCH":
            beta = params[2]
            gamma = params[3]
            mu = params[4]
            delta = args[4]
        case "APARCH":
            beta = params[2]
            gamma = params[3]
            mu = params[4]
            delta = params[5]

    # Calculate unconditional volatility
    uncondVol = uncond_vol(omega,alpha,beta,gamma,delta)
    # Calculate realized and conditional volatility
    resids = returns - mu
    realVols = abs(resids)
    condVols = np.zeros(len(returns))
    condVols[0] = uncondVol
    for t in range(1,len(returns)):
        condVols[t] = cond_vol(omega,alpha,beta,gamma,delta,resids[t-1],condVols[t-1])

    # Calculate log-likelihood
    likelihood = 1/((2*np.pi)**(1/2)*condVols)*np.exp(-realVols**2/(2*condVols**2))
    log_likelihood = -np.sum(np.log(likelihood))
    return log_likelihood

def run_optimizer(tick,t0,tF,type,p,q,returns,initialGuesses):

    # Starting parameter values (sample μ and σ)
    returnNum = returns.shape[0]
    mean = np.average(returns)
    vol = np.std(returns)
    var = vol**2

    # Handle bounds (later constraints)
    bMu = (-1,1)
    bOmega = (0,var)
    b = (0,1)
    bDelta = (0,10)

    # Handle constantParams
    match type:
        case "ARCH":
            #                                  𝛿  μ   γ β                   
            constantParams = (returns,type,p,q,2,mean,0,0)
            #                ω    α
            paramBounds = (bOmega,b)
        case "GARCH":
            constantParams = (returns,type,p,q,2,mean,0)
            paramBounds = (bOmega,b,b)
        case "GJR-GARCH":
            constantParams = (returns,type,p,q,2)
            paramBounds = (bOmega,b,b,b,bMu)
        case "APARCH":
            constantParams = (returns,type,p,q)
            paramBounds = (bOmega,b,b,b,bMu,bDelta)

    # Maximize log-likelihood
    res = sp.optimize.minimize(aparch_mle,initialGuesses,args=constantParams,bounds=paramBounds,method="Nelder-Mead",options={"disp":False})

    # Set default parameters
    beta = 0
    gamma = 0
    mu = mean
    delta = 2

    # Retrieve optimal parameters
    params = res.x
    omega = params[0]
    alpha = params[1]
    if type != "ARCH":
        beta = params[2]
        if type != "GARCH":
            gamma = params[3]
            mu = params[4]
            if type != "GJR-GARCH":
                delta = params[5]
    log_likelihood = -float(res.fun)
    k = params.shape[0] + 1
    bic = k*np.log(returnNum) - 2*log_likelihood
    aic = 2*k - 2*log_likelihood

    return [[tick,t0,tF],[type,p,q],[initialGuesses],[omega,alpha,beta,gamma,mu,delta],[log_likelihood,bic,aic]]

def run_model(modelType,pGARCH,qGARCH):

    # Fix ARCH case
    if modelType == "ARCH":
        qGARCH = 0

    # Handle constantParams and initialGuesses
    match modelType:
        case "ARCH":
            omegaMin, omegaMax, omegaInc = 0, var, var
            alphaMin, alphaMax, alphaInc = 0.3, 0.7, 0.1
            betaMin, betaMax, betaInc = 0, 0, 0
            gammaMin, gammaMax, gammaInc = 0, 0, 0
            muMin, muMax, muInc = 0, 0, 0
            deltaMin, deltaMax, deltaInc = 0, 0, 0
        case "GARCH":
            omegaMin, omegaMax, omegaInc = 0, var, var
            alphaMin, alphaMax, alphaInc = 0, 0.4, 0.1
            betaMin, betaMax, betaInc = 0.5, 0.9, 0.1
            gammaMin, gammaMax, gammaInc = 0, 0, 0
            muMin, muMax, muInc = 0, 0, 0
            deltaMin, deltaMax, deltaInc = 0, 0, 0
        case "GJR-GARCH":
            omegaMin, omegaMax, omegaInc = 0, var, var
            alphaMin, alphaMax, alphaInc = 0, 0.2, 0.1
            betaMin, betaMax, betaInc = 0.5, 0.9, 0.1
            gammaMin, gammaMax, gammaInc = 0, 0.2, 0.1
            muMin, muMax, muInc = 0, mean, mean
            deltaMin, deltaMax, deltaInc = 0, 0, 0
        case "APARCH":
            omegaMin, omegaMax, omegaInc = 0, var, var
            alphaMin, alphaMax, alphaInc = 0, 0.2, 0.1
            betaMin, betaMax, betaInc = 0.5, 0.9, 0.1
            gammaMin, gammaMax, gammaInc = 0, 0.2, 0.1
            muMin, muMax, muInc = 0, mean, mean
            deltaMin, deltaMax, deltaInc = 1, 3, 1

    # Setup counts
    if omegaInc == 0:
        omegaCount = 1
    else:
        omegaCount = (omegaMax - omegaMin)/omegaInc + 1
        if omegaCount.is_integer():
            omegaCount = int(omegaCount)
        else:
            raise Exception("omegaCount is not integer (" + str(omegaCount) + ")")
    if alphaInc == 0:
        alphaCount = 1
    else:
        alphaCount = (alphaMax - alphaMin)/alphaInc + 1
        if alphaCount.is_integer():
            alphaCount = int(alphaCount)
        else:
            raise Exception("alphaCount is not integer (" + str(alphaCount) + ")")
    if betaInc == 0:
        betaCount = 1
    else:
        betaCount = (betaMax - betaMin)/betaInc + 1
        if betaCount.is_integer():
            betaCount = int(betaCount)
        else:
            raise Exception("betaCount is not integer (" + str(betaCount) + ")")
    if gammaInc == 0:
        gammaCount = 1
    else:
        gammaCount = (gammaMax - gammaMin)/gammaInc + 1
        if gammaCount.is_integer():
            gammaCount = int(gammaCount)
        else:
            raise Exception("gammaCount is not integer (" + str(gammaCount) + ")")
    if muInc == 0:
        muCount = 1
    else:
        muCount = (muMax - muMin)/muInc + 1
        if muCount.is_integer():
            muCount = int(muCount)
        else:
            raise Exception("muCount is not integer (" + str(muCount) + ")")
    if deltaInc == 0:
        deltaCount = 1
    else:
        deltaCount = (deltaMax - deltaMin)/deltaInc + 1
        if deltaCount.is_integer():
            deltaCount = int(deltaCount)
        else:
            raise Exception("deltaCount is not integer (" + str(deltaCount) + ")")

    """
    print("ω * α * β * γ * μ * 𝛿 = Total")
    print(str(omegaCount)+" * "+str(alphaCount)+" * "+str(betaCount)+" * "+str(gammaCount)+" * "+str(muCount)+" * "+str(deltaCount)+" = "+str(omegaCount*alphaCount*betaCount*gammaCount*muCount*deltaCount)+" initialGuesses")
    """

    # Initialize arrays
    initialCount = omegaCount * alphaCount * betaCount * gammaCount * muCount * deltaCount
    initialGuesses = [None] * initialCount
    myResults = [None] * initialCount

    # Handle loops
    match modelType:
        case "ARCH":
            for o in range(omegaCount):
                for a in range(alphaCount):
                        initialGuesses[o*alphaCount + a] = [omegaMin+omegaInc*(o),alphaMin+alphaInc*(a)]
        case "GARCH":
            for o in range(omegaCount):
                for a in range(alphaCount):
                    for b in range(betaCount):
                        initialGuesses[o*alphaCount*betaCount + a*betaCount + b] = [omegaMin+omegaInc*(o),alphaMin+alphaInc*(a),betaMin+betaInc*(b)]
        case "GJR-GARCH":
            for o in range(omegaCount):
                for a in range(alphaCount):
                    for b in range(betaCount):
                        for g in range(gammaCount):
                            for m in range(muCount):
                                initialGuesses[o*alphaCount*betaCount*gammaCount*muCount + a*betaCount*gammaCount*muCount + b*gammaCount*muCount + g*muCount + m] = [omegaMin+omegaInc*(o),alphaMin+alphaInc*(a),betaMin+betaInc*(b),gammaMin+gammaInc*(g),muMin+muInc*(m)]
        case "APARCH":
            for o in range(omegaCount):
                for a in range(alphaCount):
                    for b in range(betaCount):
                        for g in range(gammaCount):
                            for m in range(muCount):
                                for d in range(deltaCount):
                                    initialGuesses[o*alphaCount*betaCount*gammaCount*muCount*deltaCount + a*betaCount*gammaCount*muCount*deltaCount + b*gammaCount*muCount*deltaCount + g*muCount*deltaCount + m*deltaCount + d] = [omegaMin+omegaInc*(o),alphaMin+alphaInc*(a),betaMin+betaInc*(b),gammaMin+gammaInc*(g),muMin+muInc*(m),deltaMin+deltaInc*(d)]


    for i in range(initialCount):
        result = run_optimizer(ticker,start,end,modelType,pGARCH,qGARCH,returns,initialGuesses[i])
        myResults[i] = result

    return myResults

def run_simulation(prices,run,simDays):

    # Get returns
    returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1
    # Get needed data stats
    mean = np.average(returns)
    vol = np.std(returns)
    var = vol**2
    # Get param list
    params = run[3]
    # Get params
    omega = params[0]
    alpha = params[1]
    beta = params[2]
    gamma = params[3]
    mu = params[4]
    delta = params[5]

    # Calculate realized and conditional volatility
    #resids = returns - mu
    #realVols = abs(resids)
    # Calculate unconditional volatility
    uncondVol = uncond_vol(omega,alpha,beta,gamma,delta)

    # Get last realized period's needed data
    lastRealReturn = returns[-1]
    lastRealResid = lastRealReturn - mu
    lastRealVol = abs(lastRealResid)
    lastRealPrice = prices[-1]
    # Assign first lookback values
    lastReturn = lastRealReturn
    lastResid = lastRealResid
    lastVol = lastRealVol
    lastPrice = lastRealPrice
    # Set up Monte Carlo data structures
    drift = mean - var/2
    realVols = np.zeros(simDays)
    uncondVols = np.zeros(simDays)
    condVols = np.zeros(simDays)
    simZs = np.zeros(simDays)
    simPrices = np.zeros(simDays)
    simReturns = np.zeros(simDays)

    # Start Monte Carlo simulation loop
    for t in range(simDays):
        # Assign constant values
        realVols[t] = vol
        uncondVols[t] = uncondVol
        # Assign other values
        condVols[t] = cond_vol(omega,alpha,beta,gamma,delta,lastResid,lastVol)
        simZs[t] = sp.stats.norm.ppf(random_num())
        simPrices[t] = lastPrice * math.exp(drift + condVols[t]*simZs[t])
        simReturns[t] = simPrices[t]/lastPrice - 1
        # Reset last values
        lastReturn = simReturns[t]
        lastResid = lastReturn - mu
        lastVol = condVols[t]
        lastPrice = simPrices[t]

    # Calculate conditional GARCH sample vol
    sampleSimVol = np.std(simReturns)
    sampleSimVols = np.zeros(simDays)
    for t in range(simDays):
        sampleSimVols[t] = sampleSimVol

    return (run[0],run[1],simPrices,simReturns,(condVols,uncondVols,sampleSimVols,realVols))

def print_model_results(run):

    # Grab sim data
    modelType = str(run[1][0])
    modelName = modelType+"("+str(run[1][1])+","+str(run[1][2])+")"
    omega = run[3][0]
    alpha = run[3][1]
    beta = run[3][2]
    gamma = run[3][3]
    mu = run[3][4]
    delta = run[3][5]
    log_likelihood = run[4][0]
    bic = run[4][1]
    aic = run[4][2]

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

    # Print summary stats
    print(ticker + " " + modelName + " model results:")
    print("log-likelihood: " + str(round(log_likelihood,4)))
    print("BIC:            " + str(round(bic,4)))
    print("AIC:            " + str(round(aic,4)))
    print("")

# FIX ALL FUNCTIONS PAST THIS POINT
def generate_model_chart(run):

    # Grab variables
    modelType = str(run[1][0])
    modelName = modelType+"("+str(run[1][1])+","+str(run[1][2])+")"
    omega = run[3][0]
    alpha = run[3][1]
    beta = run[3][2]
    gamma = run[3][3]
    mu = run[3][4]
    delta = run[3][5]

    # Calculate realized and conditional volatility for optimal parameters
    uncondVol = uncond_vol(omega,alpha,beta,gamma,delta)
    resids = returns - mu
    realVols = abs(resids)
    condVols = np.zeros(len(returns))
    condVols[0] = uncondVol
    for t in range(1,len(returns)):
        condVols[t] = cond_vol(omega,alpha,beta,gamma,delta,resids[t-1],condVols[t-1])

    # Display chart
    plt.figure(1)
    plt.rc("xtick",labelsize=10)
    plt.plot(prices.index[1:],realVols,label="Empirical Realized")
    plt.plot(prices.index[1:],condVols,label=modelName + " Conditional")
    plt.title(label=ticker + " " + modelName + " Volatility")
    plt.legend()
    plt.show()

def print_simulation_results(returns,sim):

    #Calculate empirical stats
    mean = np.average(returns)
    vol = np.std(returns)
    skew = sp.stats.skew(returns)
    kurt = sp.stats.kurtosis(returns)

    # Test actual vs. expected values here
    actualMu = np.average(sim[3])
    actualSigma = np.std(sim[3])
    actualSkew = sp.stats.skew(sim[3])
    actualKurt = sp.stats.kurtosis(sim[3])

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

    # Print Monte Carlo simulation
    modelType = str(sim[1][0])
    modelName = modelType+"("+str(sim[1][1])+","+str(sim[1][2])+")"
    print("")
    print(ticker + " " + modelName + " Monte Carlo Simulation (" + str(1) + " " + str(simDays) + "-Day Runs):")
    print("   Actual   | Expected  | Error    | Commentary")
    print("μ: " + f"{actualMu:.6f}" + " | " + f"{mean:.6f}" + "  | " + muDir + f"{muError:.2f}" + "%" + "   | Should be equal on average")
    print("σ: " + f"{actualSigma:.6f}" + " | " + f"{vol:.6f}" + "  | " + sigmaDir + f"{sigmaError:.2f}" + "%" + "   | Should be equal on average")
    print("S: " + f"{actualSkew:.6f}" + " | " + f"{skew:.6f}" + " | " + skewDir + f"{skewError:.2f}" + "%" + " | Should be wrong (vol clustering actually tends to push it up from the sim default of 0)")
    print("K: " + f"{actualKurt:.6f}" + " | " + f"{kurt:.6f}" + " | " + kurtDir + f"{kurtError:.2f}" + "%" + "  | Should be > 3 due to vol clustering but not high enough")
    print("")

def generate_simulation_chart(sim):

    # Grab variables
    modelType = str(sim[1][0])
    modelName = modelType+"("+str(sim[1][1])+","+str(sim[1][2])+")"
    simVols = sim[4][0]
    uncondVols = sim[4][1]
    sampleSimVols = sim[4][2]
    realVols = sim[4][3]

    # Display chart
    plt.figure(2)
    plt.rc("xtick",labelsize=10)
    plt.plot(range(simDays),simVols,label=modelName + " Conditional")
    plt.plot(range(simDays),uncondVols,label=modelName + " Unconditional")
    plt.plot(range(simDays),sampleSimVols,label=modelName + " Realized")
    plt.plot(range(simDays),realVols,label="Empirical Realized")
    plt.title(label=ticker + " " + modelName + " Volatility")
    plt.legend()
    plt.show()





# Download data
prices = yf.download(ticker,start,end)["Close"]
returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1

# Get needed data stats
mean = np.average(returns)
median = np.median(returns)
vol = np.std(returns)
var = vol**2
skew = sp.stats.skew(returns)
kurt = sp.stats.kurtosis(returns)

# Get gross simulation data
grossList = []
for m in range(len(modelList)):
    run = run_model(modelList[m],1,1)
    grossList = grossList + run
initialCount = len(grossList)

# Check validCount
validCount = 0
for i in range(initialCount):
    #print(str(i)+": "+str(type(myResults[i][4][0]))+" "+str(type(myResults[i][4][1]))+" "+str(type(myResults[i][4][2])))
    if str(grossList[i][4][0]) != "nan" and str(grossList[i][4][0]) != "inf" and str(grossList[i][4][0]) != "-inf":
        validCount = validCount + 1

# Filter results
filteredResults = [None] * validCount
filteredCount = 0
for i in range(initialCount):
    if str(grossList[i][4][0]) != "nan" and str(grossList[i][4][0]) != "inf" and str(grossList[i][4][0]) != "-inf":
        filteredResults[filteredCount] = grossList[i]
        filteredCount = filteredCount + 1

# Sort
sortedResults = [None] * validCount
match sortBy:
    case "LL":
        accessNum = 0
        highBest = True
    case "BIC":
        accessNum = 1
        highBest = False
    case "AIC":
        accessNum = 2
        highBest = False
tempSort = [None] * validCount
for i in range(len(filteredResults)):
    tempSort[i] = i,filteredResults[i][4][accessNum]
tempSort = sorted(tempSort, key = lambda x: x[1], reverse=highBest)
for i in range(len(sortedResults)):
    sortedResults[i] = filteredResults[tempSort[i][0]]

# Print sorted results
"""
print(str(len(sortedResults))+"/"+str(initialCount)+" tests valid")
print("Sorted:")
print(sortBy)
for i in range(len(sortedResults)):
    print(str(i+1)+" ("+f"{sortedResults[i][4][accessNum]:.12f}"+"): "+str(sortedResults[i][0])+str(sortedResults[i][1])+"["+f"{sortedResults[i][3][0]:.4f}"+","+f"{sortedResults[i][3][1]:.4f}"+","+f"{sortedResults[i][3][2]:.4f}"+","+f"{sortedResults[i][3][3]:.4f}"+","+f"{sortedResults[i][3][4]:.4f}"+","+f"{sortedResults[i][3][5]:.4f}"+"]")
"""


# Determine optimal model/parameters
optimalRun = sortedResults[0]
print_model_results(optimalRun)
generate_model_chart(optimalRun)

# Run Monte Carlo simulation
mySim = run_simulation(prices,optimalRun,simDays)
print_simulation_results(returns,mySim)
generate_simulation_chart(mySim)