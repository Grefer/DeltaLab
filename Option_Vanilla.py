# _*_ coding: utf-8 _*_
"""
Created on 12月 20 11:05 2023 

@author: Grefer
"""
from scipy.stats import norm
from math import *


class Option_Vanilla(object):

    def __init__(self,
                 s: float,
                 k: float,
                 r: float,
                 g: float,
                 t: float,
                 sigma: float,
                 cp: bool,
                 exe_mode: str
                 ):
        self.s = s
        self.k = k
        self.r = r
        self.g = g
        self.t = t
        self.sigma = sigma
        self.cp = cp
        self.exe_mode = exe_mode

        def get_price(self):
            if self.exe_mode == "Eu":
                price = blsprice(self.s, self.k, self.r, self.g, self.t, self.sigma, self.cp)
                return price


def blsprice(s, k, r, g, t, sigma, cp):
    #
    d1 = (log(s / k) + (r - g + sigma ** 2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    c = s * exp(-g * t) * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2)
    p = k * exp(-r * t) * norm.cdf(-d2) - s * exp(-g * t) * norm.cdf(-d1)
    price = c * cp + p * ~cp

    return price
