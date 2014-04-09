
import random
import numpy as np
from math import * 
import matplotlib.pyplot as plt


'''cumulative function of normal distribution
aproximation to speed up simulations'''
def norm_cdf(x):

    u = abs(x)
 
    Z = 1/(sqrt(2*pi))*exp(-u*u/2)
 
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
 
    t = 1/(1+0.2316419*u)
    t2 = t*t
    t4 = t2*t2
 
    P = 1-Z*(b1*t + b2*t2 + b3*t2*t + b4*t4 + b5*t4*t)
 
    if x<0:
        P = 1.0-P
 
    return P



'''black scholes pricing for call option'''
def OptionPrice(S0, K, r, sigma, T):
    
    log_moneyness = log(S0/K)
    sqrt_t = sqrt(T)
    
    d1 = (log_moneyness + (r + sigma**2 / 2) * T)/(sigma * sqrt_t)
    d2 = (log_moneyness + (r - sigma**2 / 2) * T) / (sigma * sqrt_t)

    return S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


'''parameters used for option pricing'''
T = 5.
K = 100
S0 = 100.
r = 0.01
sigma = 0.3

#NB : the step size match the risk margin period 
step_size = 1./12
nb_simu = 100000

'''historical rate and vol for underlying simulation'''
histo_r = 0.02
histo_sigma = 0.2

useCollat = True
#useCollat = False

initPrice = OptionPrice(S0,K,r,sigma, T)

currentspots = [S0]*nb_simu
previousOptionPrices = np.array([initPrice]*nb_simu)
currentOptionPrices = np.array([None]*nb_simu)

exposure_95 = list()
exposure_99 = list()
exposure_90 = list()
exposure_50 = list()

dates = np.arange(0,T,step_size)
for date in dates:
    for simu in range(nb_simu):
        currentspots[simu] = currentspots[simu]*random.lognormvariate( (histo_r-histo_sigma**2/2)*step_size , histo_sigma)
        currentOptionPrices[simu] =  OptionPrice(currentspots[simu], K, r, sigma, T-date)
    
    if useCollat:
        current_exposure = currentOptionPrices - previousOptionPrices
    else:
        current_exposure = currentOptionPrices
            
    quantile_option_99 = np.percentile(current_exposure,99)
    quantile_option_95 = np.percentile(current_exposure,95)
    quantile_option_90 = np.percentile(current_exposure,90)
    quantile_option_50 = np.percentile(current_exposure,50)
    
    exposure_99.append(quantile_option_99)
    exposure_95.append(quantile_option_95)
    exposure_90.append(quantile_option_90)
    exposure_50.append(quantile_option_50)
    
    '''switch array pointers to prevent full copy of array at each date'''
    temp = previousOptionPrices;
    previousOptionPrices = currentOptionPrices
    currentOptionPrices = temp


'''Display graphs'''
plt.figure("Exposure Profile - With Collateral, 1 month margin perdiod of risk")
plt.title("Exposure Profile - Without Collateral, 1 month margin perdiod of risk")
#plt.figure("Exposure Profile - Without Collateral")
#plt.title("Exposure Profile - Without Collateral")
plt.plot(dates, exposure_99, label = "99%")
plt.plot(dates, exposure_95, label = "95%")
plt.plot(dates, exposure_90, label = "90%")
plt.plot(dates, exposure_50, label = "50%")
plt.legend(loc='upper right')
        