# _*_ coding: utf-8 _*_
"""
期权定价基类，提供通用的有限差分 Greeks 计算方法
"""

import copy
import numpy as np

try:
    from .constants import ANNUAL_DAYS
except ImportError:
    from constants import ANNUAL_DAYS


class OptionBase:
    """期权定价基类

    子类需实现：
        - get_price() -> float
        - _time_remaining (property) -> float/int : 剩余到期时间（交易日数）
        - _theta_overrides(dt) -> dict : theta bump 所需的参数覆盖
        - _decrement_time() : 时间推进一天
    """

    @property
    def _time_remaining(self):
        raise NotImplementedError

    def get_price(self):
        raise NotImplementedError

    def _theta_overrides(self, dt):
        raise NotImplementedError

    def _decrement_time(self):
        """推进一天的时间参数调整，由子类实现"""
        raise NotImplementedError

    def step_forward(self, new_price):
        """推进一个交易日：当前 s0 进入历史记录，更新为新价格，时间减一天"""
        if hasattr(self, 'sr'):
            self.sr = list(self.sr) + [self.s0]
        self.s0 = new_price
        self._decrement_time()

    def _bumped_copy(self, **overrides):
        """创建参数副本用于 bump-and-reprice，不修改原对象状态"""
        obj = copy.copy(self)
        for k, v in overrides.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _safe_price(val):
        """将 None 返回值转为 0.0"""
        return val if val is not None else 0.0

    def get_greeks(self):
        """有限差分法计算 Greeks: [delta, gamma, vega, theta, rho]"""

        if self._time_remaining <= 0:
            return [0, 0, 0, 0, 0]

        ds = min(20.0, self.s0 / 100.0)
        dz = 0.01
        dt = 1
        dr = 0.01

        price0 = self._safe_price(self.get_price())

        # delta（中心差分）
        price_up = self._safe_price(self._bumped_copy(s0=self.s0 + ds).get_price())
        price_dn = self._safe_price(self._bumped_copy(s0=self.s0 - ds).get_price())
        delta = (price_up - price_dn) / (2 * ds)

        # gamma（二阶中心差分，步长 2*ds）
        price_up2 = self._safe_price(self._bumped_copy(s0=self.s0 + 2 * ds).get_price())
        price_dn2 = self._safe_price(self._bumped_copy(s0=self.s0 - 2 * ds).get_price())
        gamma = (price_up2 - 2 * price0 + price_dn2) / (4 * ds ** 2)

        # vega
        price_vup = self._safe_price(self._bumped_copy(sigma=self.sigma + dz).get_price())
        price_vdn = self._safe_price(self._bumped_copy(sigma=self.sigma - dz).get_price())
        vega = (price_vup - price_vdn) / (2 * dz)

        # theta
        price_theta = self._safe_price(self._bumped_copy(**self._theta_overrides(dt)).get_price())
        theta = (price_theta - price0) / (dt / ANNUAL_DAYS)

        # rho
        price_rup = self._safe_price(self._bumped_copy(r=self.r + dr).get_price())
        price_rdn = self._safe_price(self._bumped_copy(r=self.r - dr).get_price())
        rho = (price_rup - price_rdn) / 2

        return [delta, gamma, vega, theta, rho]
