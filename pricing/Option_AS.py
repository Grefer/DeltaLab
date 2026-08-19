# _*_ coding: utf-8 _*_
"""
Created on 1月 29 13:59 2024 

@author: Grefer
"""
import time

import numpy as np

try:
    from .constants import ANNUAL_DAYS
    from .mc_engine import McGbmQ
    from .option_base import OptionBase
except ImportError:
    from constants import ANNUAL_DAYS
    from mc_engine import McGbmQ
    from option_base import OptionBase


class Option_AS(OptionBase):
    def __init__(self,
        optiontype: str,
        s0: float,
        sr: list,
        K: float,
        E: float,
        T: int,
        N: int,
        sigma: float,
        cp: int,
        minPay: float,
        maxPay: float,
        r: float = 0.03,
        q: float = 0.03,
        nPath: int = 100000,
        ** kwargs: float
        ):
        self.optiontype = optiontype
        self.s0 = s0
        self.sr = sr
        self.K = K
        self.E = E
        self.T = T
        self.N = N
        self.sigma = sigma
        self.cp = cp
        self.minPay = minPay
        self.maxPay = maxPay
        self.r = r
        self.q = q
        self.nPath = nPath
        self._apply_extra_options(kwargs)

    def _payoff_from_observations(self, observations):
        obs = np.asarray(observations, dtype=float)
        if obs.size == 0:
            raise ValueError("Asian option requires at least one observation")
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")

        window = obs[..., -self.N:]
        if self.cp == 1:
            match self.optiontype:
                case "Asian":
                    payoff = np.mean(window, axis=-1) - self.K
                case "EnhanceAsian":
                    payoff = np.mean(np.maximum(window, self.E), axis=-1) - self.K
                case _:
                    raise ValueError(f"Unsupported optiontype: {self.optiontype}")
        elif self.cp == -1:
            match self.optiontype:
                case "Asian":
                    payoff = self.K - np.mean(window, axis=-1)
                case "EnhanceAsian":
                    payoff = self.K - np.mean(np.minimum(window, self.E), axis=-1)
                case _:
                    raise ValueError(f"Unsupported optiontype: {self.optiontype}")
        else:
            raise ValueError(f"Unsupported cp: {self.cp} (需要 1=call 或 -1=put)")

        return np.minimum(np.maximum(payoff, self.minPay), self.maxPay)

    def get_price(self):
        dt = 1 / ANNUAL_DAYS
        nStep = self.T
        sr = np.array(self.sr)
        if self.T < 0:
            raise ValueError(f"T must be non-negative, got {self.T}")
        if self.T == 0:
            observations = np.r_[sr, float(self.s0)]
            return float(self._payoff_from_observations(observations))

        S = McGbmQ(self.s0, self.r - self.q, self.sigma, self.T * dt, self.nPath, nStep,
                   seed=self.mc_seed)
        ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
        cashflow = self._payoff_from_observations(ss) * np.exp(-self.r * self.T * dt)
        price = np.mean(cashflow, 0)
        return price

    @property
    def _time_remaining(self):
        return self.T

    def _theta_overrides(self, dt):
        return {
            'sr': list(self.sr) + [self.s0],
            'T': self.T - dt,
        }

    def _decrement_time(self):
        self.T -= 1



# if __name__ == "__main__":
#     start = time.perf_counter()
#     option = Option_AS('EnhanceAsian', 100, [], 100, 100, 22, 22, 0.15, 1, 0, float('inf'))
#     p = option.get_price()
#     greeks_list = option.get_greeks()
#     end = time.perf_counter()
#     print('price = %.2f' % p)
#     print('greeks = {}'.format(greeks_list))
#     print('历时%.2f秒！' % (end - start))







