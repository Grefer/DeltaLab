# _*_ coding: utf-8 _*_
"""
Created on Nov 08 17:23 2023

@author: Grefer
基于NumPy的向量化期权定价
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
        flag_N = np.ones(le, dtype=int)
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
        flag_N = np.ones([nPath, le], dtype=int)
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
        flag_N = np.ones(le, dtype=int)
        idx_N = np.zeros(le, dtype=bool)
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
        flag_N = np.ones([nPath, le], dtype=int)
        idx_N = np.zeros((nPath, le), dtype=bool)
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
        condition_k = np.zeros(le, dtype=int)
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
        condition_k = np.zeros([nPath, le], dtype=int)
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
        condition_k = np.zeros(le, dtype=int)
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
        condition_k = np.zeros([nPath, le], dtype=int)
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


# 到期观察熔断保障价格累计（到期日结算）
def Opt_ASGQ_call_put(
        s0: float,
        sr: list,
        K: float,
        P: float,
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

    # 是否熔断提前结束
    if cp == 1 and any(sr >= H):
        idx = np.where(sr >= H)[0][0]
        price = (sr[idx] - K) * idx + (sr[idx] - P) * (le - idx)
        return price
    elif cp == -1 and any(sr <= H):
        idx = np.where(sr <= H)[0][0]
        price = (K - sr[idx]) * idx + (P - sr[idx]) * (le - idx)
        return price

    # 交易是否到期
    if T_Days == 0:

        if cp == 1:
            flag_N = sr[-1] <= K
            price = (sr[-1] - K) * le * (flag_N * N + ~flag_N * 1)
        else:
            flag_N = sr[-1] >= K
            price = (K - sr[-1]) * le * (flag_N * N + ~flag_N * 1)

    else:
        condition_ko = np.zeros([nPath, le], dtype=bool)
        flag_N = np.zeros([nPath, le], dtype=bool)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
            # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
            condition_ko[np.cumsum(ss >= H, axis=1) > 0] = 1
            flag_N[(ss[:, -1] <= K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
            cashflow = (((ss[:, -1] - P).reshape(nPath,1) * condition_ko + (ss[:, -1] - K).reshape(nPath,1) * ~condition_ko) +
                        (ss[:, -1] - K).reshape(nPath,1) * le * (flag_N * N + ~flag_N * 1 - 1)) * discount_factor
        else:
            condition_ko[np.cumsum(ss <= H, axis=1) > 0] = 1
            flag_N[(ss[:, -1] >= K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
            cashflow = (((P - ss[:, -1]).reshape(nPath,1) * condition_ko + (K - ss[:, -1]).reshape(nPath,1) * ~condition_ko) +
                        (K - ss[:, -1]).reshape(nPath,1) * le * (flag_N * N + ~flag_N * 1 - 1)) * discount_factor

        price_ls = np.sum(cashflow, 1)
        price = np.mean(price_ls, 0)

    return price

# 到期观察熔断保障价格累计（每日结算）
def Opt_ASGQ_EP(
        s0: float,
        sr: list,
        K: float,
        P: float,
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

    # 是否熔断提前结束
    if cp == 1 and any(sr >= H):
        idx = np.where(sr >= H)[0][0]
        price = np.sum((sr[:idx] - K)) + (sr[idx] - P) * (le - idx)
        return price
    elif cp == -1 and any(sr <= H):
        idx = np.where(sr <= H)[0][0]
        price = np.sum((K - sr[:idx])) + (P - sr[idx]) * (le - idx)
        return price

    # 交易是否到期
    if T_Days == 0:

        if cp == 1:
            flag_N = sr[-1] <= K
            price = np.sum((sr - K)) + (sr[-1] - K) * le * (flag_N * N + ~flag_N * 1 - 1)
        else:
            flag_N = sr[-1] >= K
            price = np.sum((K - sr)) + (K - sr[-1]) * le * (flag_N * N + ~flag_N * 1 - 1)


    else:
        condition_ko = np.zeros([nPath, le], dtype=bool)
        flag_N = np.zeros([nPath, le], dtype=bool)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
            # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
            condition_ko[np.cumsum(ss >= H, axis=1) > 0] = 1
            flag_N[(ss[:, -1] <= K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
            cashflow = (((ss - P) * condition_ko + (ss - K) * ~condition_ko) +
                        (ss[:, -1] - K).reshape(nPath, 1) * le * (flag_N * N + ~flag_N * 1 - 1)) * discount_factor
        else:
            condition_ko[np.cumsum(ss <= H, axis=1) > 0] = 1
            flag_N[(ss[:, -1] >= K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
            cashflow = (((P - ss) * condition_ko + (K - ss) * ~condition_ko) +
                        (K - ss[:, -1]) * le * (flag_N * N + ~flag_N * 1 - 1)) * discount_factor

        price_ls = np.sum(cashflow, 1)
        price = np.mean(price_ls, 0)

    return price

# 到期观察熔断后固定赔付累计（每日结算）
def Opt_ASGQ_EF(
        s0: float,
        sr: list,
        K: float,
        amount: float,
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

    # 是否熔断提前结束
    if cp == 1 and any(sr >= H):
        idx = np.where(sr >= H)[0][0]
        price = np.sum((sr[:idx] - K)) + amount * (le - idx)
        return price
    elif cp == -1 and any(sr <= H):
        idx = np.where(sr <= H)[0][0]
        price = np.sum((K - sr[:idx])) + amount * (le - idx)
        return price

    # 交易是否到期
    if T_Days == 0:

        if cp == 1:
            flag_N = sr[-1] <= K
            price = np.sum((sr - K)) + (sr[-1] - K) * le * (flag_N * N + ~flag_N * 1 - 1)
        else:
            flag_N = sr[-1] >= K
            price = np.sum((K - sr)) + (K - sr[-1]) * le * (flag_N * N + ~flag_N * 1 - 1)


    else:
        condition_ko = np.zeros([nPath, le], dtype=bool)
        flag_N = np.zeros([nPath, le], dtype=bool)
        discount_factor = np.tile(np.exp(-r * (np.maximum(Observ - T_over, 0)) * dt), (nPath, 1))
        S = McGbmQ(s0, r - q, sigma, T, nPath, nStep)
        ss = np.c_[np.tile(sr, (nPath, 1)), S]
        if cp == 1:
            # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
            # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
            condition_ko[np.cumsum(ss >= H, axis=1) > 0] = 1
            flag_N[(ss[:, -1] <= K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
            cashflow = ((amount * condition_ko + (ss - K) * ~condition_ko) +
                        (ss[:, -1] - K).reshape(nPath, 1) * le * (flag_N * N + ~flag_N * 1 - 1)) * discount_factor
        else:
            condition_ko[np.cumsum(ss <= H, axis=1) > 0] = 1
            flag_N[(ss[:, -1] >= K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
            cashflow = ((amount * condition_ko + (K - ss) * ~condition_ko) +
                        (K - ss[:, -1]) * le * (flag_N * N + ~flag_N * 1 - 1)) * discount_factor

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
    W1 = np.random.randn(nPath//2, nStep)
    W2 = np.random.randn(nPath//2, nStep)
    h = T/nStep
    dlogS1 = (r - 0.5 * pow(sigma, 2)) * h + sigma * np.sqrt(h) * W1
    dlogS2 = (r - 0.5 * pow(sigma, 2)) * h - sigma * np.sqrt(h) * W2
    s1 = s0 * np.exp(np.cumsum(dlogS1, 1))
    s2 = s0 * np.exp(np.cumsum(dlogS2, 1))
    s = np.r_[s1, s2]

    return s





if __name__ == "__main__":
    start = time.perf_counter()
    price = Opt_ASGQ_EF(100, [], 90, 20, 0, 20, list(range(1, 21)), 0.03, 0.03, 0.18, 110, 2, 100000, 1)
    end = time.perf_counter()
    print('price = %.2f' % price)
    print('历时%.2f秒！' % (end - start))
