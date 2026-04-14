# _*_ coding: utf-8 _*_
"""
Created on 12月 20 11:05 2023

@author: Grefer
"""
from scipy.stats import norm
from math import log, sqrt, exp

try:
    from .constants import ANNUAL_DAYS
    from .option_base import OptionBase
except ImportError:
    from constants import ANNUAL_DAYS
    from option_base import OptionBase


class Option_Vanilla(OptionBase):
    def __init__(self,
                 optiontype: str,
                 s0: float,
                 sr: list,
                 K: float,
                 T: int,
                 sigma: float,
                 cp: int,
                 r: float = 0.03,
                 q: float = 0.03,
                 exe_mode: str = "Eu",
                 **kwargs: float
                 ):
        self.optiontype = optiontype
        self.s0 = s0
        self.sr = sr
        self.K = K
        self.T = T
        self.sigma = sigma
        self.cp = cp
        self.r = r
        self.q = q
        self.exe_mode = exe_mode

    def get_price(self):
        if self.T <= 0:
            payoff = max(self.cp * (self.s0 - self.K), 0.0)
            return payoff
        if self.exe_mode == "Eu":
            t_years = self.T / ANNUAL_DAYS
            return blsprice(self.s0, self.K, self.r, self.q, t_years, self.sigma, self.cp)
        return None

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


def blsprice(s, k, r, g, t, sigma, cp):
    #
    d1 = (log(s / k) + (r - g + sigma ** 2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    c = s * exp(-g * t) * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2)
    p = k * exp(-r * t) * norm.cdf(-d2) - s * exp(-g * t) * norm.cdf(-d1)
    price = c if cp == 1 else p

    return price
