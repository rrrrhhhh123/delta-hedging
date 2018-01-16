# -*- coding: utf-8 -*-
"""
Test for the trade-off between Delta-Hedging and Transaction cost
测试过程：
1. 在Global variable 中设置参数
2. monte_carlo_parallel_2() 和 monte_carlo_parallel_3()
	2是根据书上的公式，3是根据Zak的论文上的公式
	根据Black-Scholes Model进行若干次Monte Carlo Stimulation, 生成若干条path， 包括T_left, Spot, delta(call), gamma, modified_delta, H_0, H_w供测试使用
3. delta_hedging_methods()
	Option为只读的tuple, 因为每次测试仅针对一单option
	Account对象用作模拟交易时记录各项数据，例如现金、交易费用、头寸等，close_account()返回P/L, 总交易费用, hedged value, Option pay-off
	然后对各种Hedging方法分别进行测试， 同时迭代各方法的参数
	返回一个dict, 记录各个方法的测试结果（以DataFrame的形式储存，以方便排序等操作）
4. Hedging_Method()
	测试对冲的框架:
	account.delta_hedging(para_0), 期初开仓
	for path in range(N): #测试第path条路径
		for step in {观测间隔， 例如每0.25天检查一次}:
			if method(当时的状态) == True:
				account.delta_hedging
	统计N次结果的均值和StD，并返回结果至3.中

"""
from scipy.stats import norm
import math
import numpy as np
from pandas import DataFrame
from collections import namedtuple
import time

# Global variable define
T_EXPIRATION = 30
DAILY_STEPS = 4
MU = 0
REALIZED_VOL = 0.3
IMPLIED_VOL = 0.3
SPOT_0 = 100
STRIKE = 100
INTEREST_RATE = 0.04
OPTION_POSITION = -1
OPTION_TYPE = 'call'
MC_PATHS_NUM = 100000

ONE_SIDE_COST = 0.0025  # lambda

# define the Option as an namedtuple, the Account as ana Object
Option = namedtuple('Option', ['price', 'strike', 'optionType'])

def close_option(option: namedtuple, spot_price: float) -> float:
	""" return the pay-off of the option """
	if option.optionType == 'call':
		return max(0, spot_price - option.strike)
	elif option.optionType == 'put':
		return max(0, option.strike - spot_price)

class Account:
	"""
	Account initiated with an option, for discretly delta-hedging the option
		option: the namedtuple

	Record:
		The cash account
		The transaction fee account
		The delta hedged P/L

	When closed:
		return tuple( cash P&L, the transaction cost, hedged_income)

	"""

	def __init__(self, init, option_num, option, one_side_cost):
		self._init = init
		self._cash = init
		self._trans_cost = 0.0

		self._one_side_cost = one_side_cost

		self._underlying_posi = 0.0
		self._option_posi = option_num

		self.option = option
		self._last_delta = 0

		self._cash -= self.option.price * self._option_posi

	def copy(self):
		""" deep copy the Account """
		return Account(self._init, self._option_posi, self.option, self._one_side_cost)

	def trade_underlying_to(self, target_posi, spot_price):
		# Re-balance the underlying position based on the current delta
		self._cash -= (target_posi - self._underlying_posi) * spot_price
		self._cash -= (abs(target_posi - self._underlying_posi)) * spot_price * self._one_side_cost
		self._trans_cost += (abs(target_posi - self._underlying_posi)) * spot_price * self._one_side_cost
		self._underlying_posi = target_posi

	def delta_hedging(self, delta, spot_price):
		""" 根据delta和option, 计算underlying的目标仓位，再根据现有仓位进行underlying的建仓 """
		target_posi = - delta * self._option_posi
		self._last_delta = delta
		self.trade_underlying_to(target_posi, spot_price)

	def get_option_posi(self):

		return self._option_posi

	def get_underlying_posi(self):

		return self._underlying_posi

	def get_last_delta(self):
		""" return the last delta for band solution """
		return self._last_delta

	def close_account(self, spot_price):
		""" return the P&L, transaction cost, delta_hedged, option_payoff as a tuple """
		# Close the underlying position
		self._cash += self._underlying_posi * spot_price
		self._cash -= abs(self._underlying_posi) * spot_price * self._one_side_cost
		self._trans_cost += abs(self._underlying_posi) * spot_price * self._one_side_cost

		# Unwind the option
		option_payoff = self._option_posi * ( close_option(self.option, spot_price) - self.option.price )
		self._cash += self._option_posi * close_option(self.option, spot_price)

		pnl = self._cash - self._init
		delta_hedged = pnl - option_payoff

		return pnl, self._trans_cost, delta_hedged, option_payoff

# the helper functions for the test proposal
def d_j(j, S, K, r, sigma, T):
	"""
	:param T: in year
	compute the d_1 and d_2 part in BS model pricing model for Call
	"""
	return (math.log(S / K) + (r + (-1) ** (j - 1) * 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def get_delta(S, K, r, sigma, T, optionType = 'call'):
	if optionType == 'call':
		return norm.cdf(d_j(1, S, K, r, sigma, T))
	elif optionType == 'put':
		return norm.cdf(d_j(1, S, K, r, sigma, T)) - 1

def vanilla_call_price(S, K, r, sigma, T):
	"""
	:param T: in year
	Get the vanilla call price
	"""
	return S * norm.cdf(d_j(1, S, K, r, sigma, T)) - K * math.exp(- r * T) * norm.cdf(d_j(2, S, K, r, sigma, T))

def vanilla_put_price(S, K, r, sigma, T):
	"""
	:param T: in year
	Get the vanilla put price
	"""
	return - S * norm.cdf(- d_j(1, S, K, r, sigma, T)) + K * math.exp(- r * T) * norm.cdf(- d_j(2, S, K, r, sigma, T))

# main test functions
def monte_carlo_parallel_2(T=30, daily_step=4, mu=0, realized_sigma=0.3, implied_sigma=0.3, S_0=100, K=100, r=0.04, optionType='call', path_num=10000, one_side_cost=0.0025, utility_gamma = 0.1):
	"""
	根据书上的公式。
	:param T: expiration， in days, should convert to years
	:param daily_step:
	:param mu, sigma, S_0, K, r: classical option parameters
	:param optionType: the European option type, 'call' or 'put'
	:param path_num: the number of stimulated Monte Carlo paths
	:return: MC_Paths, np.array, 2d array with following structure:
        Path[0] : 	T_left[1, steps] 			-> MC_Paths[0 * 7 + 0, :]
                    Spot_price[1, steps]		-> MC_Paths[0 * 7 + 1, :]
                    call_delta[1, steps]		-> MC_Paths[0 * 7 + 2, :]
                    gamma[1, steps]				-> MC_Paths[0 * 7 + 3, :]
                    modified_delta_[1, steps]	-> MC_Paths[0 * 7 + 4, :]
                    H_0[1, steps]				-> MC_Paths[0 * 7 + 5, :]
                    H_1[1, steps]				-> MC_Paths[0 * 7 + 6, :]
        ...
        ...
	"""
	steps = T * daily_step + 1  # 连头带尾, 所以要+1, S_0,..., S_T

	MC_Paths = np.empty(shape=(path_num * 7 * steps), dtype=float, order='C')
	T_left = np.linspace(T / 360, 0.000001, num=steps, endpoint=True, dtype=float)

	ts_year = np.linspace(0.000001, T / 360, num=steps, endpoint=True, dtype=float)

	W = np.concatenate((np.zeros((path_num, 1), dtype=float), np.random.randn(path_num, steps - 1)), axis=1)
	W = np.cumsum(W, axis=1) * np.sqrt(1 / daily_step / 360)
	X = (mu - 0.5 * realized_sigma ** 2) * ts_year + realized_sigma * W
	BS_Spots = S_0 * np.exp(X)

	col_idxs = np.tile(np.array(range(steps), dtype=int, order='C'), path_num).reshape(path_num, steps)
	rows = np.array(range(path_num), dtype=int, order='C').reshape(-1, 1) * 7 * steps
	T_left_idxs = (col_idxs + rows + 0 * steps).ravel()
	S_price_idxs = (col_idxs + rows + 1 * steps).ravel()
	delta_idxs = (col_idxs + rows + 2 * steps).ravel()
	gamma_idxs = (col_idxs + rows + 3 * steps).ravel()
	Modified_Delta_idxs = (col_idxs + rows + 4 * steps).ravel()
	H_0_idxs = (col_idxs + rows + 5 * steps).ravel()
	H_1_idxs = (col_idxs + rows + 6 * steps).ravel()

	MC_Paths.put(T_left_idxs, T_left)
	MC_Paths.put(S_price_idxs, BS_Spots)
	# the delta
	log_S_div_K = np.log(np.divide(MC_Paths[S_price_idxs], K))
	sig_mul_sqrT = np.multiply(np.sqrt(MC_Paths[T_left_idxs]), implied_sigma)
	r_add_sig2_mul_T = np.multiply((r + 0.5 * implied_sigma ** 2), MC_Paths[T_left_idxs])

	d_1s = np.divide(np.add(log_S_div_K, r_add_sig2_mul_T), sig_mul_sqrT)
	if optionType == 'call':
		MC_Paths.put(delta_idxs, norm.cdf(d_1s))
	else:
		MC_Paths.put(delta_idxs, np.subtract(norm.cdf(d_1s), 1))
	#put the gamma
	pdf_d_1 = norm.pdf(d_1s)
	sig_mul_S_mul_sqrT = np.multiply(implied_sigma, np.multiply(MC_Paths[S_price_idxs], np.power(MC_Paths[T_left_idxs], 0.5)))
	MC_Paths.put(gamma_idxs, np.divide(pdf_d_1, sig_mul_S_mul_sqrT))
	#Km, used for computing modified volatility
	K_m_1 = np.multiply(-5.76, np.divide(one_side_cost ** 0.78, np.power(MC_Paths[T_left_idxs], 0.02)))
	K_m_2 = np.power(np.divide(np.exp(np.multiply(-r, MC_Paths[T_left_idxs])), implied_sigma), 0.25)
	K_m_ = np.multiply(K_m_1, K_m_2)
	K_m_3 = np.power( np.multiply( np.square(MC_Paths[S_price_idxs]), np.abs(MC_Paths[gamma_idxs])), 0.15)
	K_m = np.multiply( np.multiply( K_m_, K_m_3), utility_gamma**0.15 )
	#get the modified delta
	Modified_Vol = np.multiply( implied_sigma, np.sqrt( np.subtract(1, K_m) ) )
	log_S_div_K = np.log(np.divide(MC_Paths[S_price_idxs], K))
	sig_mul_sqrT = np.multiply(np.sqrt(MC_Paths[T_left_idxs]), Modified_Vol)
	r_add_sig2_mul_T = np.multiply((np.add(r, np.multiply(0.5,np.square(Modified_Vol)))), MC_Paths[T_left_idxs])
	d_1s = np.divide(np.add(log_S_div_K, r_add_sig2_mul_T), sig_mul_sqrT)
	if optionType == 'call':
		MC_Paths.put(Modified_Delta_idxs, norm.cdf(d_1s))
	else:
		MC_Paths.put(Modified_Delta_idxs, np.subtract(norm.cdf(d_1s), 1))
	#H_0, given the utility gamma
	H_0_de = np.multiply( utility_gamma, np.multiply( MC_Paths[S_price_idxs], np.multiply( implied_sigma**2, MC_Paths[T_left_idxs])))
	H_0_nu = one_side_cost
	H_0 = np.divide(H_0_nu, H_0_de)
	MC_Paths.put(H_0_idxs, H_0)
	#H_1, given the utility gamma
	H_1_1 = np.multiply(1.12 * one_side_cost ** 0.31, np.power(MC_Paths[T_left_idxs], 0.05))
	H_1_2 = np.power(np.divide(np.exp(np.multiply(-r, MC_Paths[T_left_idxs])), implied_sigma), 0.25)
	H_1_3 = np.sqrt( np.divide( np.abs(MC_Paths[gamma_idxs]), utility_gamma) )

	MC_Paths.put(H_1_idxs, np.multiply( np.multiply(H_1_1, H_1_2), H_1_3 ))

	MC_Paths = MC_Paths.reshape(path_num * 7, steps)

	return MC_Paths

def monte_carlo_parallel_3(T=30, daily_step=4, mu=0, realized_sigma=0.3, implied_sigma=0.3, S_0=100, K=100, r=0.04, optionType='call', path_num=10000, one_side_cost=0.0025, utility_gamma = 0.1):
	"""
	根据文献Valeriy Zakamulin，Efficient analytic approximation of the optimal hedging strategy for a European call option with transaction costs中的公式
	:param
	:return: MC_Paths, np.array, 2d array with following structure:
        Path[0] : 	T_left[1, steps] 			-> MC_Paths[0 * 7 + 0, :]
                    Spot_price[1, steps]		-> MC_Paths[0 * 7 + 1, :]
                    call_delta[1, steps]		-> MC_Paths[0 * 7 + 2, :]
                    gamma[1, steps]				-> MC_Paths[0 * 7 + 3, :]
                    modified_delta_[1, steps] 	-> MC_Paths[0 * 7 + 4, :]
                    H_0[1, steps]				-> MC_Paths[0 * 7 + 5, :]
                    H_w[1, steps]				-> MC_Paths[0 * 7 + 6, :]
	"""

	steps = T * daily_step + 1  # 连头带尾, 所以要+1, S_0,..., S_T

	MC_Paths = np.empty(shape=(path_num * 7 * steps), dtype=float, order='C')
	T_left = np.linspace(T / 360, 0.000001, num=steps, endpoint=True, dtype=float)

	ts_year = np.linspace(0.000001, T / 360, num=steps, endpoint=True, dtype=float)

	W = np.concatenate((np.zeros((path_num, 1), dtype=float), np.random.randn(path_num, steps - 1)), axis=1)
	W = np.cumsum(W, axis=1) * np.sqrt(1 / daily_step / 360)
	X = (mu - 0.5 * realized_sigma ** 2) * ts_year + realized_sigma * W
	BS_Spots = S_0 * np.exp(X)

	col_idxs = np.tile(np.array(range(steps), dtype=int, order='C'), path_num).reshape(path_num, steps)
	rows = np.array(range(path_num), dtype=int, order='C').reshape(-1, 1) * 7 * steps
	T_left_idxs = (col_idxs + rows + 0 * steps).ravel()
	S_price_idxs = (col_idxs + rows + 1 * steps).ravel()
	delta_idxs = (col_idxs + rows + 2 * steps).ravel()
	gamma_idxs = (col_idxs + rows + 3 * steps).ravel()
	Modified_Delta_idxs = (col_idxs + rows + 4 * steps).ravel()
	H_0_idxs = (col_idxs + rows + 5 * steps).ravel()
	H_w_idxs = (col_idxs + rows + 6 * steps).ravel()

	MC_Paths.put(T_left_idxs, T_left)
	MC_Paths.put(S_price_idxs, BS_Spots)
	# the delta
	log_S_div_K = np.log(np.divide(MC_Paths[S_price_idxs], K))
	sig_mul_sqrT = np.multiply(np.sqrt(MC_Paths[T_left_idxs]), implied_sigma)
	r_add_sig2_mul_T = np.multiply((r + 0.5 * implied_sigma ** 2), MC_Paths[T_left_idxs])

	d_1s = np.divide(np.add(log_S_div_K, r_add_sig2_mul_T), sig_mul_sqrT)
	if optionType == 'call':
		MC_Paths.put(delta_idxs, norm.cdf(d_1s))
	else:
		MC_Paths.put(delta_idxs, np.subtract(norm.cdf(d_1s), 1))
	# the gamma
	pdf_d_1 = norm.pdf(d_1s)
	sig_mul_S_mul_sqrT = np.multiply(implied_sigma, np.multiply(MC_Paths[S_price_idxs], np.power(MC_Paths[T_left_idxs], 0.5)))
	MC_Paths.put(gamma_idxs, np.divide(pdf_d_1, sig_mul_S_mul_sqrT))
	#Km, used for computing modified volatility
	K_m_1 = np.multiply( -6.85, np.multiply( one_side_cost ** 0.78, np.power(MC_Paths[T_left_idxs], 0.1) ) )
	K_m_2 = np.multiply( implied_sigma ** 0.25, np.exp( np.multiply(MC_Paths[T_left_idxs], 0.2) ) )
	K_m_ = np.divide( K_m_1, K_m_2)
	K_m_3 = np.power( np.multiply( utility_gamma, np.multiply( np.square(MC_Paths[S_price_idxs]), np.abs(MC_Paths[gamma_idxs]))), 0.15)
	K_m = np.multiply( K_m_, K_m_3 )
	#get the modified delta
	Modified_Vol = np.multiply( implied_sigma, np.sqrt( np.subtract(1, K_m) ) )
	log_S_div_K = np.log(np.divide(MC_Paths[S_price_idxs], K))
	sig_mul_sqrT = np.multiply(np.sqrt(MC_Paths[T_left_idxs]), Modified_Vol)
	r_add_sig2_mul_T = np.multiply((np.add(r, np.multiply(0.5,np.square(Modified_Vol)))), MC_Paths[T_left_idxs])
	d_1s = np.divide(np.add(log_S_div_K, r_add_sig2_mul_T), sig_mul_sqrT)
	if optionType == 'call':
		MC_Paths.put(Modified_Delta_idxs, norm.cdf(d_1s))
	else:
		MC_Paths.put(Modified_Delta_idxs, np.subtract(norm.cdf(d_1s), 1))
	#H_0, given the utility gamma
	H_0_de = np.multiply( utility_gamma, np.multiply( MC_Paths[S_price_idxs], np.multiply( implied_sigma**2, MC_Paths[T_left_idxs])))
	H_0_nu = one_side_cost
	H_0 = np.divide(H_0_nu, H_0_de)
	MC_Paths.put(H_0_idxs, H_0)
	#H_w, given the utility gamma
	H_w_nu1 = np.multiply(1.18 * one_side_cost ** 0.31, np.power(MC_Paths[T_left_idxs], 0.1))
	H_w_de = np.multiply(implied_sigma**0.25, np.exp(np.multiply(0.15, MC_Paths[T_left_idxs])))
	H_w_nu2 = np.sqrt( np.divide( np.abs(MC_Paths[gamma_idxs]), utility_gamma) )

	MC_Paths.put(H_w_idxs, np.divide( np.multiply(H_w_nu1, H_w_nu2), H_w_de ))

	MC_Paths = MC_Paths.reshape(path_num * 7, steps)

	return MC_Paths

def Hedging_Method(_account: Account, MC_Paths: np.array, method='fixed_interval', *args) -> dict:
	"""
	期初进行Delta-Hedging, 之后离散地进行判断然后Re-Hedging， 其中Zak的方法是用Modified Delta进行hedging
	:param _account: Account Object, recording the trade and get the hedge result, must deepcopy this instance by the Account.copy() method each MC stimulation.
	:param MC_Paths: np.array(float), the stimulated Monte Carlo Paths under the Black-Scholes world, the shape is ((6 * N), steps), which means N MC paths.
	:param method: str, refers to the hedging method under the real world
	:param args: the args for the hedging method
	:return:
	"""

	# monte carlo的path数
	mc_path_num = int(MC_Paths.shape[0] / 7)
	# path的steps数
	steps = MC_Paths.shape[1]
	# 观察步长
	observ_steps = 4 * args[0] if method == 'fixed_interval' else 1
	# 结果列表
	pnl_list = []
	trans_cost_list = []
	delta_hedged_list = []
	option_payoff_list = []

	# 将一些全局变量赋值给局部变量
	utility_gamma = args[0]
	one_side_cost = ONE_SIDE_COST
	r = INTEREST_RATE

	for path_idx in range(mc_path_num):
		# 第 path_idx 次模拟测试
		account_cpy = _account.copy()  # 必须深拷贝Account对象

		# 不同指标的path位置
		T_left_path = path_idx * 7
		S_path = path_idx * 7 + 1
		delta_path = path_idx * 7 + 2
		gamma_path = path_idx * 7 + 3
		Mod_delta_path = path_idx * 7 + 4
		H_0_path = path_idx * 7 + 5
		H_w_path = path_idx * 7 + 6

		# 期初进行Delta Hedging
		if method == 'Zak_Band':
			account_cpy.delta_hedging(MC_Paths[Mod_delta_path, 0], MC_Paths[S_path, 0])
		else:
			account_cpy.delta_hedging(MC_Paths[delta_path, 0], MC_Paths[S_path, 0])

		# 根据观察步长和策略进行Re-Hedging
		for step in range(observ_steps, steps - 1, observ_steps):
			# 观察时刻的各个变量的值
			cur_T_left = MC_Paths[T_left_path, step]
			cur_spot = MC_Paths[S_path, step]
			cur_delta = MC_Paths[delta_path, step]
			cur_gamma = MC_Paths[gamma_path, step]
			mod_delta = MC_Paths[Mod_delta_path, step]
			H_0 = MC_Paths[H_0_path, step]
			H_w = MC_Paths[H_w_path, step]

			if method == 'fixed_interval':
				account_cpy.delta_hedging(cur_delta, cur_spot)

			elif method == 'WW_Band':
				half_band = ((3 / 2 * math.exp(- r * cur_T_left) * one_side_cost * cur_spot * cur_gamma ** 2) / utility_gamma) ** (1 / 3)  # args[0]: utility_gamma
				WW_band_up = cur_delta + half_band
				WW_band_down = cur_delta - half_band
				if account_cpy.get_last_delta() > WW_band_up:
					account_cpy.delta_hedging(WW_band_up, cur_spot)
				elif account_cpy.get_last_delta() < WW_band_down:
					account_cpy.delta_hedging(WW_band_down, cur_spot)

			elif method == 'Zak_Band':
				Zak_Band_up = mod_delta + H_0 + H_w
				Zak_Band_down = mod_delta - (H_0 + H_w)
				if account_cpy.get_last_delta() > Zak_Band_up:
					account_cpy.delta_hedging(Zak_Band_up, cur_spot)
				elif account_cpy.get_last_delta() < Zak_Band_down:
					account_cpy.delta_hedging(Zak_Band_down, cur_spot)

		# 行权日Close Account
		close_spot = MC_Paths[S_path, steps - 1]
		pnl, trans, delta_hed, opt_payoff = account_cpy.close_account(close_spot)

		# 得到的结果储存
		pnl_list.append(pnl)
		trans_cost_list.append(trans)
		delta_hedged_list.append(delta_hed)
		option_payoff_list.append(opt_payoff)

	# 计算各指标的mean
	mean_pnl = sum(pnl_list) / len(pnl_list)
	mean_cost = sum(trans_cost_list) / len(trans_cost_list)
	mean_dh = sum(delta_hedged_list) / len(delta_hedged_list)
	mean_pf = sum(option_payoff_list) / len(option_payoff_list)
	# 计算各指标的volatility
	vol_pnl = np.std(pnl_list)
	vol_cost = np.std(trans_cost_list)
	vol_dh = np.std(delta_hedged_list)
	vol_pf = np.std(option_payoff_list)

	return {'avg_PnL': mean_pnl, 'vola_PnL': vol_pnl, 'avg_trans_cost': mean_cost, 'vola_trans_cost': vol_cost,
			'avg_hedged': mean_dh, 'vola_hedged': vol_dh, 'avg_payoff': mean_pf, 'vola_payoff': vol_pf}

def delta_hedging_methods(T, daily_step, mu, realized_sigma, implied_sigma, S_0, K, r, optionPosi, optionType, path_num, one_side_cost):
	"""
	:param: pass
	:return: dict,
			dict['fixed_interval'] : DataFrame for different parameters
			dict['WW_Band'] : DataFrame
			dict['Zak_Band'] : DataFrame
			DataFrame(columns=[ 'para_names', 'avg_PnL', 'vola_PnL', 'avg_trans_cost', 'vola_trans_cost', 'avg_hedged', 'vol_hedged' ]
	"""
	MC_Paths = monte_carlo_parallel_3(T, daily_step, mu, realized_sigma, implied_sigma, S_0, K, r, optionType, path_num, one_side_cost, 0.1)  # np.array( (7 * N), steps )
	print( MC_Paths.shape )
	# construct the option and account that we want to delta-hedging with
	if optionType == 'call':
		opt_price = vanilla_call_price(S_0, K, r, implied_sigma, T / 360)
	else:  # put option
		opt_price = vanilla_put_price(S_0, K, r, implied_sigma, T / 360)
	option = Option(opt_price, K, optionType)
	account = Account(100, optionPosi, option, one_side_cost)

	###			Test different methods to delta_hedging
	return_results = {}
	print("Start to do the fixed interval re-hedging method...")

	##			Fixed Interval Method
	#				method parameter: interval -> 1 day , 5 day, 10 day, 15 day
	fixed_results = DataFrame(
		columns=['interval_days', 'avg_PnL', 'vola_PnL', 'avg_trans_cost', 'vola_trans_cost', 'avg_hedged',
				 'vola_hedged', 'avg_payoff', 'vola_payoff'])
	for interval in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
		result = Hedging_Method(account.copy(), MC_Paths, 'fixed_interval', interval)
		result['interval_days'] = interval
		fixed_results = fixed_results.append(result, ignore_index=True)
	return_results['fixed_interval'] = fixed_results

	##			Utility-based Method
	# 				method parameter: utility_gamma
	utility_gamma_range = [0.15, 0.5, 1, 5, 15]
	mc_paths = {}
	print("Start to generate the paths under utilities:{}".format(utility_gamma_range))
	for utility_gamma in utility_gamma_range:
		mc_paths[utility_gamma] = monte_carlo_parallel_3(T, daily_step, mu, realized_sigma, implied_sigma, S_0, K, r, optionType, path_num, one_side_cost, utility_gamma)

	time_ww_0 = time.time()
	# 				1. Whalley and Wilmott solution;
	print("Start to do the W-W band method...")
	ww_band_results = DataFrame(
		columns=['utility_gamma', 'avg_PnL', 'vola_PnL', 'avg_trans_cost', 'vola_trans_cost', 'avg_hedged',
				 'vola_hedged', 'avg_payoff', 'vola_payoff'])
	for utility_gamma in utility_gamma_range:
		result = Hedging_Method(account.copy(), mc_paths[utility_gamma], "WW_Band", utility_gamma)
		result['utility_gamma'] = utility_gamma
		ww_band_results = ww_band_results.append(result, ignore_index=True)
	return_results['WW_Band'] = ww_band_results
	time_ww_1 = time.time()
	print("WW band method spend: {0:.2f} seconds".format(time_ww_1 - time_ww_0))

	time_zak_0 = time.time()
	#				2. Zakamouline Solution
	print("Start to do the Zak band method...")
	zak_band_results = DataFrame(
		columns=['utility_gamma', 'avg_PnL', 'vola_PnL', 'avg_trans_cost', 'vola_trans_cost', 'avg_hedged',
				 'vola_hedged', 'avg_payoff', 'vola_payoff'])
	for utility_gamma in utility_gamma_range:
		result = Hedging_Method(account.copy(), mc_paths[utility_gamma], "Zak_Band", utility_gamma)
		result['utility_gamma'] = utility_gamma
		zak_band_results = zak_band_results.append(result, ignore_index=True)
	return_results['Zak_Band'] = zak_band_results
	time_zak_1 = time.time()
	print("Zak band method spend: {0:.2f} seconds".format(time_zak_1 - time_zak_0))
	##				Other Methods

	return return_results

if __name__ == '__main__':
	results = delta_hedging_methods(T_EXPIRATION, DAILY_STEPS, MU, REALIZED_VOL, IMPLIED_VOL, SPOT_0, STRIKE, INTEREST_RATE,
									OPTION_POSITION, OPTION_TYPE, MC_PATHS_NUM, ONE_SIDE_COST)
	print("Fixed Interval Method test:...")
	print(results['fixed_interval'])
	print("W-W Band Method test:...")
	print(results['WW_Band'])
	print("Zak Band Method test:...")
	print(results['Zak_Band'])




