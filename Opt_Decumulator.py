# _*_ coding: utf-8 _*_
"""
Created on Nov 08 17:23 2023

@author: Grefer
"""
import time
import numpy as np

annual_days = 243.0


# 回归累计
def Opt_Decumulator_Back(
        s0: float,
        sr: list,
        K: float,
        T_over: int,
        T_Days: int,
        Observ: list,
        r: float,
        q: float,
        sigma: float,
        H: float,
        N: int,
        nPath: int,
        cp: int
) -> float:

    T = T_Days / annual_days
    nStep = T_Days
    dt = 1 / annual_days
    sr = np.array(sr)
    Observ = np.array(Observ)
    le = len(Observ)
    if T_Days == 0:
        flag_N = np.ones(le,dtype=int)
        if cp == 1:
            flag_N[sr >= H] = 0
            flag_N[sr <= K] = N
            cashflow = (sr - K) * flag_N
        else:
            flag_N[sr <= H] = 0
            flag_N[sr >= K] = N
            cashflow = (K - sr) * flag_N

        price = np.sum(cashflow)

    else:
        flag_N = np.ones([nPath, le],dtype=int)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            flag_N[ss >= H] = 0
            flag_N[ss <= K] = N
            cashflow = (ss - K) * flag_N * discount_factor
        else:
            flag_N[ss <= H] = 0
            flag_N[ss >= K] = N
            cashflow = (K - ss) * flag_N * discount_factor

        price_ls = np.sum(cashflow, 1)
        price = np.mean(price_ls, 0)

    return price


# 区间内固定赔付回归累计
def Opt_Decumulator_Fix(
        s0: float,
        sr: list,
        K: float,
        fix: float,
        T_over: int,
        T_Days: int,
        Observ: list,
        r: float,
        q: float,
        sigma: float,
        H: float,
        N: int,
        nPath: int,
        cp: int
) -> float:

    T = T_Days / annual_days
    nStep = T_Days
    dt = 1 / annual_days
    sr = np.array(sr)
    Observ = np.array(Observ)
    le = len(Observ)
    if T_Days == 0:
        flag_N = np.ones(le,dtype=int)
        idx_N = np.zeros(le,dtype=bool)
        if cp == 1:
            idx_N[sr <= K] = 1
            flag_N[sr >= H] = 0
            flag_N[sr <= K] = N
            cashflow = fix * flag_N * ~idx_N + (sr - K) * flag_N * idx_N
        else:
            idx_N[sr >= K] = 1
            flag_N[sr <= H] = 0
            flag_N[sr >= K] = N
            cashflow = fix * flag_N * ~idx_N + (K - sr) * flag_N * idx_N

        price = np.sum(cashflow)

    else:
        flag_N = np.ones([nPath, le],dtype=int)
        idx_N = np.zeros((nPath, le),dtype=bool)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            idx_N[ss <= K] = 1
            flag_N[ss >= H] = 0
            flag_N[ss <= K] = N
            cashflow = (fix * flag_N * ~idx_N + (ss - K) * flag_N * idx_N) * discount_factor
        else:
            idx_N[ss >= K] = 1
            flag_N[ss <= H] = 0
            flag_N[ss >= K] = N
            cashflow = (fix * flag_N * ~idx_N + (K - ss) * flag_N * idx_N) * discount_factor

        price_ls = np.sum(cashflow, 1)
        price = np.mean(price_ls, 0)

    return price

# 增强回归累计
def Opt_EnDecumulator(
        s0: float,
        sr: list,
        K: float,
        T_over: int,
        T_Days: int,
        Observ: list,
        r: float,
        q: float,
        sigma: float,
        H: float,
        N: int,
        nPath: int,
        cp: int
) -> float:

    T = T_Days / annual_days
    nStep = T_Days
    dt = 1 / annual_days
    sr = np.array(sr)
    Observ = np.array(Observ)
    le = len(Observ)

    if T_Days == 0:
        condition_k = np.zeros(le,dtype=int)
        condition_ki = np.zeros(le, dtype=int)
        condition_ko = np.zeros(le, dtype=int)
        if cp == 1:
            condition_ko[sr >= H] = 1
            condition_k[(K < sr) & (sr < H)] = 1
            condition_ki[sr <= K] = N
            cashflow = (sr - H) * condition_ko + (sr - K) * condition_k + (sr - K) * condition_ki
        else:
            condition_ko[sr <= H] = 1
            condition_k[(H < sr) & (sr < K)] = 1
            condition_ki[sr >= K] = N
            cashflow = (H - sr) * condition_ko + (K - sr) * condition_k + (K - sr) * condition_ki

        price = np.sum(cashflow)

    else:
        condition_k = np.zeros([nPath, le],dtype=int)
        condition_ki = np.zeros([nPath, le], dtype=int)
        condition_ko = np.zeros([nPath, le], dtype=int)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            condition_ko[ss >= H] = 1
            condition_k[(K < ss) & (ss < H)] = 1
            condition_ki[ss <= K] = N
            cashflow = ((ss - H) * condition_ko + (ss - K) * condition_k + (ss - K) * condition_ki) * discount_factor
        else:
            condition_ko[ss <= H] = 1
            condition_k[(H < ss) & (ss < K)] = 1
            condition_ki[ss >= K] = N
            cashflow = ((H - ss) * condition_ko + (K - ss) * condition_k + (K - ss) * condition_ki) * discount_factor

        price_ls = np.sum(cashflow, 1)
        price = np.mean(price_ls, 0)

    return price


# 固定赔付增强回归累计
def Opt_EnDecumulator_Fix(
        s0: float,
        sr: list,
        K: float,
        fix: float,
        T_over: int,
        T_Days: int,
        Observ: list,
        r: float,
        q: float,
        sigma: float,
        H: float,
        N: int,
        nPath: int,
        cp: int
) -> float:

    T = T_Days / annual_days
    nStep = T_Days
    dt = 1 / annual_days
    sr = np.array(sr)
    Observ = np.array(Observ)
    le = len(Observ)

    if T_Days == 0:
        condition_k = np.zeros(le,dtype=int)
        condition_ki = np.zeros(le, dtype=int)
        condition_ko = np.zeros(le, dtype=int)
        if cp == 1:
            condition_ko[sr >= H] = 1
            condition_k[(K < sr) & (sr < H)] = 1
            condition_ki[sr <= K] = N
            cashflow = (sr - H) * condition_ko + fix * condition_k + (sr - K) * condition_ki
        else:
            condition_ko[sr <= H] = 1
            condition_k[(H < sr) & (sr < K)] = 1
            condition_ki[sr >= K] = N
            cashflow = (H - sr) * condition_ko + fix * condition_k + (K - sr) * condition_ki

        price = np.sum(cashflow)

    else:
        condition_k = np.zeros([nPath, le],dtype=int)
        condition_ki = np.zeros([nPath, le], dtype=int)
        condition_ko = np.zeros([nPath, le], dtype=int)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            condition_ko[ss >= H] = 1
            condition_k[(K < ss) & (ss < H)] = 1
            condition_ki[ss <= K] = N
            cashflow = ((ss - H) * condition_ko + fix * condition_k + (ss - K) * condition_ki) * discount_factor
        else:
            condition_ko[ss <= H] = 1
            condition_k[(H < ss) & (ss < K)] = 1
            condition_ki[ss >= K] = N
            cashflow = ((H - ss) * condition_ko + fix * condition_k + (K - ss) * condition_ki) * discount_factor

        price_ls = np.sum(cashflow, 1)
        price = np.mean(price_ls, 0)

    return price


def McGbmQ(
        s0: float,
        r: float,
        sigma: float,
        T: float,
        nPath: int,
        nStep: int
):
    np.random.seed(20)

    W = np.random.randn(nPath, nStep)
    h = T / nStep
    dlogS = (r - 0.5 * pow(sigma, 2)) * h + sigma * np.sqrt(h) * W
    s = s0 * np.exp(np.cumsum(dlogS, 1))
    return s


if __name__ == "__main__":
    start = time.perf_counter()
    price = Opt_EnDecumulator_Fix(100, [], 90,20, 0,20,  list(range(1, 21)), 0.03, 0.03, 0.18, 110, 2, 100000, 1)
    end = time.perf_counter()
    print(price)
    print('历时%.2f秒！' % (end - start))
