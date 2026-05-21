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

    MC Greeks 说明：
        对 MC 定价的子类（Option_AB / Option_AS / Option_DE），get_greeks
        在有限差分 bump-and-reprice 过程中会复用 self.mc_seed 以实现
        Common Random Numbers (CRN)：所有 bump 路径共享同一组随机数，
        以保证不同 bump 的定价差由真实参数敏感性产生，而不是 MC 噪声。
        解析定价的 Option_Vanilla 不受这条约束。
    """

    # 日内已流逝比例（0~1，单位=日）。bump copy 里用它把 T 方向的剩余时间
    # 按小数日衰减；默认 0 不影响现有行为。目前仅 Option_Vanilla 在 get_price
    # 里消费这个字段，MC 类的 T_days 需要整数 nStep，暂不支持 intraday 衰减。
    _intraday_elapsed = 0.0

    # MC 采样种子：子类实例可单独覆盖。None = 每次调用 McGbmQ 都用 OS 熵
    # 重新抽样；int = 确定性序列，bump 之间共享即为 CRN。默认 20 保持
    # 与 Review 前行为一致（所有 MC 期权共用一组路径）。
    mc_seed = 20

    # Greeks bump 时使用的路径数；None = 与 self.nPath 一致。某些 MC 期权
    # 路径很重，bump 五个方向（±ds, ±2ds, ±sigma, ±r, theta）可能明显变慢，
    # 这里允许外部覆盖为更小值（例如 max(nPath//4, 5000)），以换取性能。
    greeks_nPath = None

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
        # 跨日后重置日内 elapsed，避免 bump copy 继承残值
        self._intraday_elapsed = 0.0

    def knockout_event(self, prices, steps_per_day=1):
        """提前了结（敲出）检测钩子，供对冲回测在已知价格路径时截断存续期。

        默认返回 None（普通期权不会提前了结）。路径依赖且带敲出观察的结构
        （如雪球 Option_SNB）可重写：给定回测价格路径 prices（prices[0]=今日）
        与 steps_per_day，返回 (i_ko, settle_value)——i_ko 为 prices 中触发敲出
        的索引，settle_value 为敲出当日的结算价值（之后不再有现金流）。
        """
        return None

    def _bumped_copy(self, **overrides):
        """创建参数副本用于 bump-and-reprice，不修改原对象状态。

        使用 deepcopy 避免 sr / observ 等可变列表被父子对象共享：
        get_greeks 里 _theta_overrides 会对 sr 做 list(...) + [s0] 新建列表，
        但 gamma / delta bump 路径只改 s0，不新建 sr；若是浅拷贝，后续若有
        任何分支原地 append 到 self.sr，父子对象会相互污染。
        """
        obj = copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _safe_price(val):
        """将 None 返回值转为 0.0"""
        return val if val is not None else 0.0

    def get_greeks(self):
        """有限差分法计算 Greeks: [delta, gamma, vega, theta, rho]

        对 MC 期权（self 有 nPath 属性时），在 bump 期间临时将 self.nPath
        切换为 self.greeks_nPath（若已设置），并保持 self.mc_seed 不变，
        以实现 CRN：所有 bump 共享同一批随机数路径，bump 定价差仅反映
        参数敏感性而非 MC 噪声。Option_Vanilla 走解析路径，不受影响。
        """

        if self._time_remaining <= 0:
            return [0, 0, 0, 0, 0]

        ds = min(20.0, self.s0 / 100.0)
        dz = 0.01
        dt = 1
        dr = 0.01

        # ---- MC 期权：bump 前临时压路径数，bump 结束后恢复 ----
        _npath_saved = None
        if hasattr(self, "nPath") and getattr(self, "greeks_nPath", None):
            _npath_saved = self.nPath
            self.nPath = int(self.greeks_nPath)

        try:
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
        finally:
            if _npath_saved is not None:
                self.nPath = _npath_saved

        return [delta, gamma, vega, theta, rho]
