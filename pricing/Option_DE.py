# _*_ coding: utf-8 _*_
"""
Created on Nov 08 17:23 2023

@author: Grefer
基于NumPy的向量化期权定价
"""
import time

import numpy as np

try:
    from .constants import ANNUAL_DAYS as annual_days
    from .mc_engine import McGbmQ
    from .option_base import OptionBase
except ImportError:
    from constants import ANNUAL_DAYS as annual_days
    from mc_engine import McGbmQ
    from option_base import OptionBase


class Option_DE(OptionBase):
    # 累计期权大类

    def __init__(self,
                 optiontype: str,
                 s0: float,
                 sr: list,
                 K: float,
                 T_over: int,
                 T_days: int,
                 observ: list,
                 sigma: float,
                 H: float,
                 N: int,
                 cp: int,
                 r: float = 0.03,
                 q: float = 0.03,
                 nPath: int = 100000,
                 **kwargs: float
                 ):
        self.optiontype = optiontype
        self.s0 = s0
        self.sr = sr
        self.K = K
        self.T_over = T_over
        self.T_days = T_days
        self.observ = observ
        self.r = r
        self.q = q
        self.sigma = sigma
        self.H = H
        self.N = N
        self.nPath = nPath
        self.cp = cp
        self.fix = kwargs.get('fix')
        self.P = kwargs.get('P')
        self.amount = kwargs.get('amount')

    # 定价整合函数
    def get_price(self):

        price = getattr(self, self.optiontype)()

        return price

    @property
    def _time_remaining(self):
        return self.T_days

    def _theta_overrides(self, dt):
        return {
            'sr': list(self.sr) + [self.s0],
            'T_over': self.T_over + dt,
            'T_days': self.T_days - dt,
        }

    def _decrement_time(self):
        self.T_days -= 1
        self.T_over += 1

    # 回归累计
    def Opt_Decumulator_Back(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)
        if self.T_days == 0:
            flag_N = np.ones(le, dtype=int)
            if self.cp == 1:
                flag_N[sr >= self.H] = 0
                flag_N[sr <= self.K] = self.N
                cashflow = (sr - self.K) * flag_N
            else:
                flag_N[sr <= self.H] = 0
                flag_N[sr >= self.K] = self.N
                cashflow = (self.K - sr) * flag_N

            price = np.sum(cashflow)

        else:
            flag_N = np.ones([self.nPath, le], dtype=int)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                flag_N[ss >= self.H] = 0
                flag_N[ss <= self.K] = self.N
                cashflow = (ss - self.K) * flag_N * discount_factor
            else:
                flag_N[ss <= self.H] = 0
                flag_N[ss >= self.K] = self.N
                cashflow = (self.K - ss) * flag_N * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 区间内固定赔付回归累计
    def Opt_Decumulator_Fix(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)
        if self.T_days == 0:
            flag_N = np.ones(le, dtype=int)
            idx_N = np.zeros(le, dtype=bool)
            if self.cp == 1:
                idx_N[sr <= self.K] = 1
                flag_N[sr >= self.H] = 0
                flag_N[sr <= self.K] = self.N
                cashflow = self.fix * flag_N * ~idx_N + (sr - self.K) * flag_N * idx_N
            else:
                idx_N[sr >= self.K] = 1
                flag_N[sr <= self.H] = 0
                flag_N[sr >= self.K] = self.N
                cashflow = self.fix * flag_N * ~idx_N + (self.K - sr) * flag_N * idx_N

            price = np.sum(cashflow)

        else:
            flag_N = np.ones([self.nPath, le], dtype=int)
            idx_N = np.zeros((self.nPath, le), dtype=bool)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                idx_N[ss <= self.K] = 1
                flag_N[ss >= self.H] = 0
                flag_N[ss <= self.K] = self.N
                cashflow = (self.fix * flag_N * ~idx_N + (ss - self.K) * flag_N * idx_N) * discount_factor
            else:
                idx_N[ss >= self.K] = 1
                flag_N[ss <= self.H] = 0
                flag_N[ss >= self.K] = self.N
                cashflow = (self.fix * flag_N * ~idx_N + (self.K - ss) * flag_N * idx_N) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 增强回归累计
    def Opt_EnDecumulator(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        if self.T_days == 0:
            condition_k = np.zeros(le, dtype=int)
            condition_ki = np.zeros(le, dtype=int)
            condition_ko = np.zeros(le, dtype=int)
            if self.cp == 1:
                condition_ko[sr >= self.H] = 1
                condition_k[(self.K < sr) & (sr < self.H)] = 1
                condition_ki[sr <= self.K] = self.N
                cashflow = (sr - self.H) * condition_ko + (sr - self.K) * condition_k + (sr - self.K) * condition_ki
            else:
                condition_ko[sr <= self.H] = 1
                condition_k[(self.H < sr) & (sr < self.K)] = 1
                condition_ki[sr >= self.K] = self.N
                cashflow = (self.H - sr) * condition_ko + (self.K - sr) * condition_k + (self.K - sr) * condition_ki

            price = np.sum(cashflow)

        else:
            condition_k = np.zeros([self.nPath, le], dtype=int)
            condition_ki = np.zeros([self.nPath, le], dtype=int)
            condition_ko = np.zeros([self.nPath, le], dtype=int)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ko[ss >= self.H] = 1
                condition_k[(self.K < ss) & (ss < self.H)] = 1
                condition_ki[ss <= self.K] = self.N
                cashflow = ((ss - self.H) * condition_ko + (ss - self.K) * condition_k + (
                        ss - self.K) * condition_ki) * discount_factor
            else:
                condition_ko[ss <= self.H] = 1
                condition_k[(self.H < ss) & (ss < self.K)] = 1
                condition_ki[ss >= self.K] = self.N
                cashflow = ((self.H - ss) * condition_ko + (self.K - ss) * condition_k + (
                        self.K - ss) * condition_ki) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 固定赔付增强回归累计
    def Opt_EnDecumulator_Fix(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        if self.T_days == 0:
            condition_k = np.zeros(le, dtype=int)
            condition_ki = np.zeros(le, dtype=int)
            condition_ko = np.zeros(le, dtype=int)
            if self.cp == 1:
                condition_ko[sr >= self.H] = 1
                condition_k[(self.K < sr) & (sr < self.H)] = 1
                condition_ki[sr <= self.K] = self.N
                cashflow = (sr - self.H) * condition_ko + self.fix * condition_k + (sr - self.K) * condition_ki
            else:
                condition_ko[sr <= self.H] = 1
                condition_k[(self.H < sr) & (sr < self.K)] = 1
                condition_ki[sr >= self.K] = self.N
                cashflow = (self.H - sr) * condition_ko + self.fix * condition_k + (self.K - sr) * condition_ki

            price = np.sum(cashflow)

        else:
            condition_k = np.zeros([self.nPath, le], dtype=int)
            condition_ki = np.zeros([self.nPath, le], dtype=int)
            condition_ko = np.zeros([self.nPath, le], dtype=int)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ko[ss >= self.H] = 1
                condition_k[(self.K < ss) & (ss < self.H)] = 1
                condition_ki[ss <= self.K] = self.N
                cashflow = ((ss - self.H) * condition_ko + self.fix * condition_k + (
                        ss - self.K) * condition_ki) * discount_factor
            else:
                condition_ko[ss <= self.H] = 1
                condition_k[(self.H < ss) & (ss < self.K)] = 1
                condition_ki[ss >= self.K] = self.N
                cashflow = ((self.H - ss) * condition_ko + self.fix * condition_k + (
                        self.K - ss) * condition_ki) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 到期观察熔断保障价格累计（到期日结算）
    def Opt_ASGQ_call_put(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            price = (sr[idx] - self.K) * idx + (sr[idx] - self.P) * (le - idx)

            return price

        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            price = (self.K - sr[idx]) * idx + (self.P - sr[idx]) * (le - idx)

            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                flag_N = sr[-1] <= self.K
                cashflow = (sr[-1] - self.K) * le * (flag_N * self.N + ~flag_N * 1)
            else:
                flag_N = sr[-1] >= self.K
                cashflow = (self.K - sr[-1]) * le * (flag_N * self.N + ~flag_N * 1)

            price = cashflow

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            flag_N = np.zeros([self.nPath, le], dtype=bool)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (((ss[:, -1] - self.P).reshape(self.nPath, 1) * condition_ko + (ss[:, -1] - self.K).reshape(
                    self.nPath, 1) * ~condition_ko)
                            + (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (((self.P - ss[:, -1]).reshape(self.nPath, 1) * condition_ko + (self.K - ss[:, -1]).reshape(
                    self.nPath, 1) * ~condition_ko)
                            + (self.K - ss[:, -1]).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 到期观察熔断保障价格累计（每日结算）
    def Opt_ASGQ_EP(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            price = np.sum((sr[:idx] - self.K)) + (sr[idx] - self.P) * (le - idx)

            return price

        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            price = np.sum((self.K - sr[:idx])) + (self.P - sr[idx]) * (le - idx)

            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                flag_N = sr[-1] <= self.K
                cashflow = np.sum((sr - self.K)) + (sr[-1] - self.K) * le * (flag_N * self.N + ~flag_N * 1 - 1)
            else:
                flag_N = sr[-1] >= self.K
                cashflow = np.sum((self.K - sr)) + (self.K - sr[-1]) * le * (flag_N * self.N + ~flag_N * 1 - 1)

            price = cashflow

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            flag_N = np.zeros([self.nPath, le], dtype=bool)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (((ss - self.P) * condition_ko + (ss - self.K) * ~condition_ko) +
                            (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (((self.P - ss) * condition_ko + (self.K - ss) * ~condition_ko) +
                            (self.K - ss[:, -1]) * le * (flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 到期观察熔断后固定赔付累计（每日结算）
    def Opt_ASGQ_EF(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            price = np.sum((sr[:idx] - self.K)) + self.amount * (le - idx)
            return price
        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            price = np.sum((self.K - sr[:idx])) + self.amount * (le - idx)
            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                flag_N = sr[-1] <= self.K
                price = np.sum((sr - self.K)) + (sr[-1] - self.K) * le * (flag_N * self.N + ~flag_N * 1 - 1)
            else:
                flag_N = sr[-1] >= self.K
                price = np.sum((self.K - sr)) + (self.K - sr[-1]) * le * (flag_N * self.N + ~flag_N * 1 - 1)

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            flag_N = np.zeros([self.nPath, le], dtype=bool)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = ((self.amount * condition_ko + (ss - self.K) * ~condition_ko) +
                            (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = ((self.amount * condition_ko + (self.K - ss) * ~condition_ko) +
                            (self.K - ss[:, -1]) * le * (flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 每日观察熔断后保障价格累计（每日结算）
    def Opt_ASGQ_DP(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            flag_N = sr <= self.K
            price = np.sum((sr[:idx] - self.K) * (flag_N[:idx] * self.N + ~flag_N[:idx] * 1)) + (sr[idx] - self.P) * (
                    le - idx)
            return price
        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            flag_N = sr >= self.K
            price = np.sum((self.K - sr[:idx]) * (flag_N[:idx] * self.N + ~flag_N[:idx] * 1)) + (self.P - sr[idx]) * (
                    le - idx)
            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                flag_N = sr <= self.K
                price = np.sum((sr - self.K) * (flag_N * self.N + ~flag_N * 1))
            else:
                flag_N = sr >= self.K
                price = np.sum((self.K - sr) * (flag_N * self.N + ~flag_N * 1))

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            flag_N = np.zeros([self.nPath, le], dtype=bool)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                flag_N[(ss <= self.K) * ~condition_ko] = 1
                cashflow = (((ss - self.P) * condition_ko + (ss - self.K) * ~condition_ko) * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                flag_N[(ss >= self.K) * ~condition_ko] = 1
                cashflow = (((self.P - ss) * condition_ko + (self.K - ss) * ~condition_ko) * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 每日观察熔断后固定赔付累计（每日结算）
    def Opt_ASGQ_DF(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            flag_N = sr <= self.K
            price = np.sum((sr[:idx] - self.K) * (flag_N[:idx] * self.N + ~flag_N[:idx] * 1)) + self.amount * (le - idx)
            return price
        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            flag_N = sr >= self.K
            price = np.sum((self.K - sr[:idx]) * (flag_N[:idx] * self.N + ~flag_N[:idx] * 1)) + self.amount * (le - idx)
            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                flag_N = sr <= self.K
                price = np.sum((sr - self.K) * (flag_N * self.N + ~flag_N * 1))
            else:
                flag_N = sr >= self.K
                price = np.sum((self.K - sr) * (flag_N * self.N + ~flag_N * 1))

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            flag_N = np.zeros([self.nPath, le], dtype=bool)
            discount_factor = np.tile(np.exp(-self.r * (np.maximum(observ - self.T_over, 0)) * dt), (self.nPath, 1))
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                flag_N[(ss <= self.K) * ~condition_ko] = 1
                cashflow = ((self.amount * condition_ko + (ss - self.K) * ~condition_ko) * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                flag_N[(ss >= self.K) * ~condition_ko] = 1
                cashflow = ((self.amount * condition_ko + (self.K - ss) * ~condition_ko) * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price


if __name__ == "__main__":
    start = time.perf_counter()
    option = Option_DE('Opt_ASGQ_DP', 100, [], 90, 0, 20, list(range(1, 21)), 0.18, 110, 2, 1, P=100)
    p = option.get_price()
    greeks_list = option.get_greeks()
    end = time.perf_counter()
    print('price = %.2f' % p)
    print('greeks = {}'.format(greeks_list))
    print('历时%.2f秒！' % (end - start))
