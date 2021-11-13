import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import fsolve

import numpy_financial as npf

def replace_char(f,c=',',d=''):
	for ii, i in enumerate(f):
		f[ii]=i.replace(c,d)
	return f



def read_data(filename,index_col='Date'):
	# df = pd.read_csv(filename, index_col=index_col)
	df = pd.read_csv(filename) 
	return df

def normalize_return(df, col,log=True):
 	
 	f = df[col].to_numpy()
 	if type(f[1])=='str':
 		f = replace_char(f,',','')
 	f = f.astype(np.float)
 	result = np.ones(len(f))

 	if log==True:

 		result[1:] = np.log(f[1:]/f[0:-1])
 	else:
 		result[1:]= np.divide(f[1:],f[0:-1])
 	
 	
 	return result


def squared(f):
	f1=f*f
	return f1

def rollling_avg(f,window):
	f2=np.zeros(len(f))
	f2[0:window] = np.nan
	for i in range(window,len(f)):
		f2[i] = np.mean(f[i-window:i])
	return f2

def risk_measure(returns_squared,window,mu=0.94):
	returns_squared[np.isnan(returns_squared)==1] = 0
	f2=np.zeros(len(returns_squared))


	f2[window] = returns_squared[window]
	for i in range(window,len(returns_squared)):
		f2[i] = mu * f2[i-1] + (1-mu)*returns_squared[i] 
	return f2

def filtered(returns, risk_measure_):
	risk_measure_[risk_measure_==0]=1
	return np.divide(returns,np.sqrt(risk_measure_))


def percentile(returns,window_percentile,alpha):
	f2=np.zeros(len(returns))
	for i in range(window_percentile,len(returns)):
		f2[i] = np.percentile(returns[i-window_percentile:i],100*alpha)
	return f2


def val_risk(f, alpha=0.01):
	coef = np.abs(st.norm.ppf(alpha))
	return coef * np.sqrt (f)

def expected_shortfall(returns, val_at_risk):
	
	f = np.zeros(len(returns))
	f = returns[returns<-val_at_risk]
	return np.mean(f)


def bond_ytm(FV, rate, price, freq):
	rate = rate /freq
	coupon = FV * rate

	

def YTM_equation_discrete(Price0, r, c, FV, M, freq):

	if freq==0:
		steps = 1
		if M < steps:
			steps = M
		N = M
		c1 = 0
	else:
		steps = 1/freq
		c1 = c/freq
		N = int(np.floor(M/(1/freq)))
	if N<1:
		N = 1

	payment_series = np.repeat(c1 * FV, N + 1)  
	payment_series[0] = -1 * Price0 
	payment_series[N]  += FV

	if freq >0:
		r = r/freq
	sol = npf.npv(r, payment_series)

	return(sol)


def YTM_equation_continious(Price0, r, c, FV, M, freq):

	if freq==0:
		steps = 1
		if M < steps:
			steps = M
		N = M
		c1 = 0
	else:
		steps = 1/freq
		c1 = c/freq
		N = int(np.floor(M/(1/freq)))
	if N<1:
		N = 1

	payment_series = np.repeat(c1 * FV, N + 1)  
	payment_series[0] = -1 * Price0 
	payment_series[N]  += FV


	time_intervals = np.arange(0, M + steps, steps) 
	discount_factor = np.exp(-time_intervals*r)
	sol = np.sum(payment_series * discount_factor)
	return(sol)


def YTM_equation_portfolio(Price0, r, coupons, time_intervals):

	sol = np.sum(coupons * np.exp(-r*time_intervals)) - Price0
	return(sol)

def solve_YTO_portfolio(Price0, coupons, time_intervals):

	ytm = fsolve(lambda x : YTM_equation_portfolio(r = x, Price0 = Price0, coupons=coupons, time_intervals=time_intervals), 
	x0 = 0, 
	xtol=1.49012e-08)
	return ytm


def solve_YTO(Price0, c, FV, M, freq, cont=0):
	if cont == 0:

		ytm = fsolve(lambda x : YTM_equation_discrete(r = x, Price0 = Price0, FV = FV, c = c, M = M, freq= freq), 
		x0 = 0, 
		xtol=1.49012e-08)
	else:
		ytm = fsolve(lambda x : YTM_equation_continious(r = x, Price0 = Price0, FV = FV, c = c, M = M, freq= freq), 
		x0 = 0, 
		xtol=1.49012e-08)
	return ytm


def price_portfolio(r, coupons, time_intervals):

	sol = np.sum(coupons * np.exp(-r*time_intervals)) 
	return(sol)

def Mduration(r, coupons, time_intervals, FV):

	sol = coupons * np.exp(-r*time_intervals)
	print(sol)
	for i in range(0, len(sol)):
		sol[i] = time_intervals[i] * sol[i]
	print(sol)
	print(time_intervals)
	Mac_dur = np.sum(sol)/FV
	Mod_dur = Mac_dur / (1 + r)
	return Mac_dur, Mod_dur






