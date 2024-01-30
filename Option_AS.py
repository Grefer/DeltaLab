# _*_ coding: utf-8 _*_
"""
Created on 1月 29 13:59 2024 

@author: Grefer
"""
import time

import numpy as np

annual_days = 243.0

def McGbmQ(
        s0: float,
        r: float,
        sigma: float,
        T: float,
        nPath: int,
        nStep: int
):
    np.random.seed(20)
    W1 = np.random.randn(nPath // 2, nStep)
    W2 = np.random.randn(nPath // 2, nStep)
    W = np.r_[W1, -W2]
    h = T / nStep
    dlogS = (r - 0.5 * pow(sigma, 2)) * h - sigma * np.sqrt(h) * W
    s = s0 * np.exp(np.cumsum(dlogS, 1))
    return s

class Option_AS(object):
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

    def get_price(self):
        dt = 1 / annual_days
        nStep = self.T
        sr = np.array(self.sr)
        if self.T > 0:
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, self.T * dt, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                match self.optiontype:
                    case "Asian":
                        cashflow = np.minimum( np.maximum(np.mean(ss[self.T-self.N:-1],1) - self.K, self.minPay), self.maxPay) * np.exp(-self.r * self.T * dt)
                    case "EnhanceAsian":
                        cashflow = np.minimum( np.maximum(np.mean( np.maximum(ss[self.T-self.N:-1], self.E),1) - self.K, self.minPay), self.maxPay) * np.exp(-self.r * self.T * dt)
            elif self.cp == -1:
                match self.optiontype:
                    case "Asian":
                        cashflow = np.minimum( np.maximum(self.K - np.mean(ss[self.T-self.N:-1],1), self.minPay), self.maxPay) * np.exp(-self.r * self.T * dt)
                    case "EnhanceAsian":
                        cashflow = np.minimum( np.maximum(self.K - np.mean(np.minimum(ss[self.T-self.N:-1], self.E),1), self.minPay), self.maxPay) * np.exp(-self.r * self.T * dt)

            price = np.mean(cashflow, 0)
            return price

    # Greeks计算函数
    def get_greeks(self):

        if self.T <= 0:
            greeks = [0, 0, 0, 0, 0]
        else:
            ds = min(20., self.s0 / 100.)
            dz = 1 / 100.
            dt = 1
            dr = 1 / 100.
            # delta
            price0 = self.get_price()
            self.s0 += ds
            price1 = self.get_price()
            self.s0 -= 2 * ds
            price2 = self.get_price()
            delta = (price1 - price2) / (2 * ds)
            # gamma
            self.s0 += 3 * ds
            price3 = self.get_price()
            self.s0 -= 4 * ds
            price4 = self.get_price()
            gamma = ((price3 - price0) - (price0 - price4)) / (4 * pow(ds, 2))
            # vega
            self.s0 += 2 * ds
            self.sigma += dz
            price5 = self.get_price()
            self.sigma -= 2 * dz
            price6 = self.get_price()
            vega = (price5 - price6) / (2 * dz)
            # theta
            self.sigma += dz
            self.sr = np.r_[self.sr, self.s0]
            self.T += dt
            self.T -= dt
            price7 = self.get_price()
            theta = (price7 - price0) / (dt / annual_days)
            # rho
            self.sr = np.delete(self.sr, -1)
            self.T += dt
            self.r += dr
            price8 = self.get_price()
            self.r -= 2 * dr
            price9 = self.get_price()
            rho = (price8 - price9) / 2

            greeks = [delta, gamma, vega, theta, rho]

        return greeks



# if __name__ == "__main__":
#     start = time.perf_counter()
#     option = Option_AS('EnhanceAsian', 100, [], 100, 100, 22, 22, 0.15, 1, 0, float('inf'))
#     p = option.get_price()
#     greeks_list = option.get_greeks()
#     end = time.perf_counter()
#     print('price = %.2f' % p)
#     print('greeks = {}'.format(greeks_list))
#     print('历时%.2f秒！' % (end - start))








