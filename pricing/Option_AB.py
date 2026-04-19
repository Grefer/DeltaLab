# _*_ coding: utf-8 _*_
"""
Created on 11月 22 16:19 2023 

@author: Grefer
"""

import numpy as np
import time

try:
    from .constants import ANNUAL_DAYS
    from .mc_engine import McGbmQ
    from .option_base import OptionBase
except ImportError:
    from constants import ANNUAL_DAYS
    from mc_engine import McGbmQ
    from option_base import OptionBase


class Option_AB(OptionBase):

    def __init__(self,
                 optiontype: str,
                 s0: float,
                 sr: list,
                 K: float,
                 KI: float,
                 T_days: int,
                 observ: list,
                 sigma: float,
                 pr: float,
                 pr_ki: float,
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
        self.KI = KI
        self.T_days = T_days
        self.observ = observ
        self.sigma = sigma
        self.pr = pr
        self.pr_ki = pr_ki
        self.cp = cp
        self.r = r
        self.q = q
        self.nPath = nPath

    def get_price(self):

        price = getattr(self, self.optiontype)()

        return price

    @property
    def _time_remaining(self):
        return self.T_days

    def _theta_overrides(self, dt):
        return {
            'sr': list(self.sr) + [self.s0],
            'T_days': self.T_days - dt,
        }

    def _decrement_time(self):
        self.T_days -= 1

    def Opt_Airbag(self):

        T = self.T_days / ANNUAL_DAYS
        nStep = self.T_days
        dt = 1 / ANNUAL_DAYS
        sr = np.array(self.sr)
        observ = np.array(self.observ) - 1

        if self.T_days == 0:
            if self.cp == 1:
                condition_ki = np.any(sr[observ] <= self.KI)
                price = self.pr * np.maximum(sr[-1] - self.K, 0) * ~condition_ki + self.pr_ki * (
                        sr[-1] - self.K) * condition_ki
            else:
                condition_ki = np.any(sr[observ] >= self.KI)
                price = self.pr * np.maximum(self.K - sr[-1], 0) * ~condition_ki + self.pr_ki * (
                        self.K - sr[-1]) * condition_ki

        elif self.T_days >= 0:
            discount_factor = np.exp(-self.r * self.T_days * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ki = np.any(ss[:, observ] <= self.KI, axis=1)
                cashflow = (self.pr * np.maximum(ss[:, -1] - self.K, 0) * ~condition_ki + self.pr_ki * (
                            ss[:, -1] - self.K) * condition_ki) * discount_factor
            else:
                condition_ki = np.any(ss[:, observ] >= self.KI, axis=1)
                cashflow = (self.pr * np.maximum(self.K - ss[:, -1], 0) * ~condition_ki + self.pr_ki * (
                            self.K - ss[:, -1]) * condition_ki) * discount_factor

            price = np.mean(cashflow, axis=0)

        return price


if __name__ == "__main__":
    start = time.perf_counter()
    option = Option_AB('Opt_Airbag', 100, [], 100, 90, 20, list(range(1, 21)), 0.18, 0.8, 1, 1)
    p = option.get_price()
    greeks_list = option.get_greeks()
    end = time.perf_counter()
    print('price = %.2f' % p)
    print('greeks = {}'.format(greeks_list))
    print('历时%.2f秒！' % (end - start))
