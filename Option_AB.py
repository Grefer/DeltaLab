# _*_ coding: utf-8 _*_
"""
Created on 11月 22 16:19 2023 

@author: Grefer
"""

import numpy as np
import time

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


class Option_AB(object):

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

    def get_greeks(self):

        if self.T_days <= 0:
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
            self.T_days -= dt
            price7 = self.get_price()
            theta = (price7 - price0) / (dt / annual_days)
            # rho
            self.sr = np.delete(self.sr, -1)
            self.T_days += dt
            self.r += dr
            price8 = self.get_price()
            self.r -= 2 * dr
            price9 = self.get_price()
            rho = (price8 - price9) / 2

            greeks = [delta, gamma, vega, theta, rho]

        return greeks

    def Opt_Airbag(self):

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ) - 1

        if self.T_days == 0:
            if self.cp == 1:
                condition_ki = np.any(sr[observ] <= self.KI)
                price = self.pr * np.max(self.K - sr[-1], 0) * ~condition_ki + self.pr_ki * (
                        sr[-1] - self.K) * condition_ki
            else:
                condition_ki = np.any(sr[observ] >= self.KI)
                price = self.pr * np.max(sr[-1] - self.K, 0) * ~condition_ki + self.pr_ki * (
                        self.K - sr[-1]) * condition_ki

        elif self.T_days >= 0:
            discount_factor = np.exp(-self.r * self.T_days * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ki = np.any(ss[:, observ] <= self.KI, axis=1)
                cashflow = (self.pr * np.maximum(ss[:, -1] - self.K, 0) * ~condition_ki + self.pr_ki * (
                            ss[:, -1] - self.K) * condition_ki) * discount_factor
            else:
                condition_ki = np.any(ss[observ] >= self.KI, axis=1)
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
