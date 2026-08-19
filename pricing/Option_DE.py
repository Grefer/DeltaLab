# _*_ coding: utf-8 _*_
"""
Created on Nov 08 17:23 2023

@author: Grefer
基于NumPy的向量化期权定价
"""
import time
from typing import Optional

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
    # 累计期权大类。13 个结构按三个正交维度区分，中文名见
    # ``deltalab_ui.constants.SUBTYPE_DISPLAY``：
    #   ① 触碰障碍 H 之后：敲出终止 / 敲出计零 / 敲出增强 /
    #      熔断保障（改按 P 结算）/ 熔断赔付（改付 amount）
    #   ② 区间 (K, H)：线性 (S−K) 或按 `fix` 固赔
    #   ③ 杠杆腿 (S ≤ K)：每日观察逐日乘 N，或到期观察一次性结算
    #
    # ``ASGQ_E*`` 与 ``ASGQ_D*`` 的差别**只有 ③**。这两组的注释此前
    # 写作「到期观察熔断」与「每日观察熔断」，是错的：七个 ASGQ 的熔断
    # 判定都是同一行 ``np.cumsum(ss >= H, axis=1) > 0``，逐日路径依赖、
    # 完全一致。E 只在到期日看 S_T ≤ K 并按累计天数一次性补 (N−1) 倍，
    # D 则逐日看 S ≤ K、当日乘 N。

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
                 fix: Optional[float] = None,
                 P: Optional[float] = None,
                 amount: Optional[float] = None,
                 **kwargs
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
        self.fix = fix
        self.P = P
        self.amount = amount
        self._apply_extra_options(kwargs)
        self._validate_required_params()
        self._validate_dimensions()

    # 各子类型必需的参数。缺了会在定价深处炸成
    # ``TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'``，
    # 看不出是哪个字段没填，所以在构造期挡下来并点名。
    #
    # **0 是合法取值**：区间赔付 0 = 标的落在区间内那几天不结算，熔断赔付
    # 0 = 熔断后不再有现金流，都是真实条款。只有 None（真的没传）才算缺。
    # GUI 的建构闭包一度写成 ``p["fix"] if p["fix"] else None``，把 0.0 当
    # falsy 转成了 None，于是填 0 直接报错——那是本条校验的误伤，不是它的本意。
    _REQUIRED_PARAMS = {
        "Opt_Decumulator_Fix":   ("fix",),
        "Opt_Decumulator_Fix_E": ("fix",),
        "Opt_EnDecumulator_Fix": ("fix",),
        "Opt_ASGQ_call_put":     ("P",),
        "Opt_ASGQ_EP":           ("P",),
        "Opt_ASGQ_DP":           ("P",),
        "Opt_ASGQ_EF":           ("amount",),
        "Opt_ASGQ_DF":           ("amount",),
        "Opt_ASGQ_EFF":          ("amount", "fix"),
        "Opt_ASGQ_DFF":          ("amount", "fix"),
    }

    _PARAM_LABELS = {
        "fix": "区间赔付", "P": "保障价格", "amount": "熔断赔付",
    }

    def _validate_dimensions(self):
        """已实现前缀 + 剩余期限必须正好铺满观察日。

        ``ss`` 是 ``sr``（已实现）与 ``S``（模拟 ``T_days`` 步）横向拼出来的，
        而各类掩码矩阵按 ``len(observ)`` 开。三者对不上就会在定价深处炸成
        ``IndexError: boolean index did not match indexed array``——只报出两个
        数字，看不出是哪个字段填错了。

        更糟的是 ``len(observ) == 1`` 那档：形状 ``(nPath,)`` 与 ``(nPath, 1)``
        能广播成 ``(nPath, nPath)``，**不报错**，价格直接放大 ``nPath`` 倍。

        这条不变量在所有状态转移下都守恒：``step_forward`` 给 sr 加一项、
        ``_decrement_time`` 给 T_days 减一天，``_theta_overrides`` 同理。
        """
        n_realized = len(self.sr) if self.sr is not None else 0
        n_observ = len(self.observ) if self.observ is not None else 0
        if n_realized != self.T_over:
            raise ValueError(
                f"已实现价格序列长度（{n_realized}）必须等于已过天数"
                f"（{self.T_over}）：sr 就是这 {self.T_over} 天的收盘价。")
        if n_realized + self.T_days != n_observ:
            raise ValueError(
                f"观察日数量对不上：已实现 {n_realized} 天 + 剩余 "
                f"{self.T_days} 天 = {n_realized + self.T_days}，"
                f"但 observ 有 {n_observ} 项。"
                f"「已过天数」不为 0 时必须同时提供这些天的已实现价格。")

    def _validate_required_params(self):
        missing = [
            name for name in self._REQUIRED_PARAMS.get(self.optiontype, ())
            if getattr(self, name) is None
        ]
        if missing:
            fields = "、".join(
                f"{self._PARAM_LABELS[n]}({n})" for n in missing)
            raise ValueError(
                f"结构 {self.optiontype} 必须提供 {fields}。"
                f"（0 是合法取值，只有真正省略这个参数才会报这个错。）")

    @staticmethod
    def _knockout_state(ss, breached, discount_factor):
        """熔断当日的价格与贴现因子，按路径取出、形状 (nPath, 1)。

        熔断当天直接结算，所以熔断日及之后那一整笔在**熔断当日**一次付清，
        应当用熔断当日的贴现因子折回 0 时刻，而不是各天用各天的因子。
        早退分支不贴现是对的——那种情形下熔断确实发生在今天。

        此前 MC 分支不是这么算的：``EP``/``DP`` 用的是**当天**价格、
        ``call_put`` 用的是**到期**价格，而 ``EF``/``DF``/``EFF``/``DFF``
        虽然金额是常数，却把它逐日铺开贴现——四种写法都与早退分支（熔断已经
        发生在真实历史里的情形）矛盾。同一条路径，只把熔断日从模拟段挪进
        已实现段，价格就会跳；而对冲回测每天都在往已实现段追加收盘价，标的
        一穿越障碍，当天估值就从一条路径跳到另一条。

        整段保持向量化：``np.argmax`` 一次取出所有路径的首个触发列，
        再用花式索引取价格与因子，不需要按路径循环。整行都没触发时
        ``argmax`` 返回 0，但那一行的 ``condition_ko`` 全为 False、
        ``n_ko`` 为 0，取到的值不会进入结果。

        前置条件：``discount_factor`` 必须是一维的（长度 = 观察日数）。
        """
        idx = np.argmax(breached, axis=1)
        rows = np.arange(ss.shape[0])
        return ss[rows, idx][:, None], discount_factor[idx][:, None]

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

    # 敲出终止累计：首次 S ≥ H 后当日及之后均停止累计
    def Opt_Decumulator(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)
        if self.T_days == 0:
            condition_ko = np.zeros(le, dtype=bool)
            flag_N = np.ones(le, dtype=int)
            if self.cp == 1:
                # 敲出当日及之后均不再累计（对应 MATLAB 中的 break）
                condition_ko[np.cumsum(sr >= self.H) > 0] = 1
                flag_N[sr <= self.K] = self.N
                cashflow = (sr - self.K) * flag_N * ~condition_ko
            else:
                condition_ko[np.cumsum(sr <= self.H) > 0] = 1
                flag_N[sr >= self.K] = self.N
                cashflow = (self.K - sr) * flag_N * ~condition_ko

            price = np.sum(cashflow)

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            flag_N = np.ones([self.nPath, le], dtype=int)
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                flag_N[ss <= self.K] = self.N
                cashflow = (ss - self.K) * flag_N * ~condition_ko * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                flag_N[ss >= self.K] = self.N
                cashflow = (self.K - ss) * flag_N * ~condition_ko * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 敲出计零累计：仅 S ≥ H 的当日计 0，之后仍继续观察
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
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

    # 敲出计零·区间固赔累计（K~H 区间按 fix 结算）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
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

    # 敲出计零·区间固赔·到期杠杆累计
    # （K~H 区间按 fix 结算；杠杆腿到期日观察、到期结算）
    def Opt_Decumulator_Fix_E(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)
        if self.T_days == 0:
            flag = np.ones(le)
            flag_ki = np.zeros(le)
            if self.cp == 1:
                flag[sr >= self.H] = 0
                flag[sr < self.K] = 0
                flag_ki[-1] = sr[-1] < self.K
                cashflow = self.fix * flag + (sr[-1] - self.K) * le * flag_ki * self.N
            else:
                flag[sr <= self.H] = 0
                flag[sr > self.K] = 0
                flag_ki[-1] = sr[-1] > self.K
                cashflow = self.fix * flag + (self.K - sr[-1]) * le * flag_ki * self.N

            price = np.sum(cashflow)

        else:
            flag = np.ones([self.nPath, le])
            flag_ki = np.zeros([self.nPath, le])
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                flag[ss >= self.H] = 0
                flag[ss < self.K] = 0
                flag_ki[:, -1] = ss[:, -1] < self.K
                cashflow = (self.fix * flag
                            + (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * flag_ki * self.N) * discount_factor
            else:
                # 注意：MATLAB 原版此处 cp==-1 分支与 cp==1 条件完全一致（疑似复制粘贴遗漏），
                # 会导致累沽（cp==-1）的 fix 永不赔付。此处按到期分支与 cp==1 对称修正。
                flag[ss <= self.H] = 0
                flag[ss > self.K] = 0
                flag_ki[:, -1] = ss[:, -1] > self.K
                cashflow = (self.fix * flag
                            + (self.K - ss[:, -1]).reshape(self.nPath, 1) * le * flag_ki * self.N) * discount_factor

            price_ls = np.sum(cashflow, 1)
            price = np.mean(price_ls, 0)

        return price

    # 敲出增强累计：S ≥ H 的当日仍付 (S − H)，保留敲出后的上行
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
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

    # 敲出增强·区间固赔累计（K~H 区间按 fix 结算）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
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

    # 熔断保障·到期结算累计：主项只用到期收盘价，
    # 是 13 个结构里唯一不逐日结算的一个（其余熔断族均每日结算）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                breached = ss >= self.H
                condition_ko[np.cumsum(breached, axis=1) > 0] = 1
                # 熔断即终止：这条路径的"终值"就是熔断日价格，熔断前后
                # 各日都据此结算（与早退分支的 (S_ko-K)*idx + (S_ko-P)*(le-idx) 同形）。
                s_ko, df_ko = self._knockout_state(ss, breached, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                s_term = np.where(condition_ko[:, -1:], s_ko, ss[:, -1:])
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                # 本结构到期日一次结算：熔断路径整条改在熔断当日结算，
                # 因此熔断前后各日一律按熔断当日的因子折回（与早退分支
                # (S_ko-K)*idx + (S_ko-P)*(le-idx) 完全同形）。
                ko_total = ((s_term - self.K) * (le - n_ko)
                            + (s_term - self.P) * n_ko) * df_ko
                # 本结构是**到期日一次结算**：期权按保证金逐日盯市估值，
                # 但现金流当作到期一次结清处理，所以整条腿按到期因子折现，
                # 而不是各天用各天的。逐日折会让跨越障碍的那一瞬间凭空跳一下
                # ——熔断的只有一天，换掉折现口径的却是全部 le 天。
                # 杠杆腿本来就只落在最后一列，改成标量后逐位不变。
                cashflow = ((s_term - self.K) * ~condition_ko
                            + (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)
                            ) * discount_factor[-1]
            else:
                breached = ss <= self.H
                condition_ko[np.cumsum(breached, axis=1) > 0] = 1
                s_ko, df_ko = self._knockout_state(ss, breached, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                s_term = np.where(condition_ko[:, -1:], s_ko, ss[:, -1:])
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                ko_total = ((self.K - s_term) * (le - n_ko)
                            + (self.P - s_term) * n_ko) * df_ko
                # 本结构是**到期日一次结算**：期权按保证金逐日盯市估值，
                # 但现金流当作到期一次结清处理，所以整条腿按到期因子折现，
                # 而不是各天用各天的。逐日折会让跨越障碍的那一瞬间凭空跳一下
                # ——熔断的只有一天，换掉折现口径的却是全部 le 天。
                # 杠杆腿本来就只落在最后一列，改成标量后逐位不变。
                cashflow = ((self.K - s_term) * ~condition_ko
                            + (self.K - ss[:, -1]).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)
                            ) * discount_factor[-1]

            price_ls = np.where(condition_ko[:, -1],
                                ko_total[:, 0], np.sum(cashflow, 1))
            price = np.mean(price_ls, 0)

        return price

    # 熔断保障·到期杠杆累计（每日结算；杠杆腿只在到期日观察）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                breached = ss >= self.H
                condition_ko[np.cumsum(breached, axis=1) > 0] = 1
                s_ko, df_ko = self._knockout_state(ss, breached, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                # 累计腿：每日结算，各天用各天的贴现因子。
                cashflow = ((ss - self.K) * ~condition_ko +
                            (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor
                # 熔断腿：整笔在熔断当日结算，按当日因子折回 0 时刻。
                ko_leg = (s_ko - self.P) * n_ko * df_ko
            else:
                breached = ss <= self.H
                condition_ko[np.cumsum(breached, axis=1) > 0] = 1
                s_ko, df_ko = self._knockout_state(ss, breached, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = ((self.K - ss) * ~condition_ko +
                            (self.K - ss[:, -1]).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor
                ko_leg = (self.P - s_ko) * n_ko * df_ko

            price_ls = np.sum(cashflow, 1) + ko_leg[:, 0]
            price = np.mean(price_ls, 0)

        return price

    # 熔断赔付·到期杠杆累计（每日结算；杠杆腿只在到期日观察）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss >= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：整笔在熔断当日结算，按当日因子折回 0 时刻。
                # 金额是常数 amount，与熔断后价格无关，但**结算日**同样要对齐——
                # 早退分支写的是 amount*(le-idx) 且不贴现（钱当天就到手）。
                ko_leg = self.amount * n_ko * df_ko
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (((ss - self.K) * ~condition_ko) +
                            (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss <= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：整笔在熔断当日结算，按当日因子折回 0 时刻。
                # 金额是常数 amount，与熔断后价格无关，但**结算日**同样要对齐——
                # 早退分支写的是 amount*(le-idx) 且不贴现（钱当天就到手）。
                ko_leg = self.amount * n_ko * df_ko
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (((self.K - ss) * ~condition_ko) +
                            (self.K - ss[:, -1]).reshape(self.nPath, 1) * le * (
                                    flag_N * self.N + ~flag_N * 1 - 1)) * discount_factor

            price_ls = np.sum(cashflow, 1) + ko_leg[:, 0]
            price = np.mean(price_ls, 0)

        return price

    # 熔断保障·每日杠杆累计（每日结算；杠杆腿逐日观察，S ≤ K 当日乘 N）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                breached = ss >= self.H
                condition_ko[np.cumsum(breached, axis=1) > 0] = 1
                s_ko, df_ko = self._knockout_state(ss, breached, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                flag_N[(ss <= self.K) * ~condition_ko] = 1
                cashflow = ((ss - self.K) * ~condition_ko * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor
                # 熔断日之后 flag_N 恒为 False，杠杆系数恒为 1，故不出现在熔断腿里。
                ko_leg = (s_ko - self.P) * n_ko * df_ko
            else:
                breached = ss <= self.H
                condition_ko[np.cumsum(breached, axis=1) > 0] = 1
                s_ko, df_ko = self._knockout_state(ss, breached, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                flag_N[(ss >= self.K) * ~condition_ko] = 1
                cashflow = ((self.K - ss) * ~condition_ko * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor
                ko_leg = (self.P - s_ko) * n_ko * df_ko

            price_ls = np.sum(cashflow, 1) + ko_leg[:, 0]
            price = np.mean(price_ls, 0)

        return price

    # 熔断赔付·每日杠杆累计（每日结算；杠杆腿逐日观察，S ≤ K 当日乘 N）
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
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                # 对逻辑值累加以实现找到第一个满足条件的值后，之后的条件全为True
                # 现实意义：对于路径依赖的熔断累计期权，熔断日及之后都采用保障价格结算
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss >= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：整笔在熔断当日结算，按当日因子折回 0 时刻。
                # 金额是常数 amount，与熔断后价格无关，但**结算日**同样要对齐——
                # 早退分支写的是 amount*(le-idx) 且不贴现（钱当天就到手）。
                ko_leg = self.amount * n_ko * df_ko
                flag_N[(ss <= self.K) * ~condition_ko] = 1
                cashflow = (((ss - self.K) * ~condition_ko) * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss <= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：整笔在熔断当日结算，按当日因子折回 0 时刻。
                # 金额是常数 amount，与熔断后价格无关，但**结算日**同样要对齐——
                # 早退分支写的是 amount*(le-idx) 且不贴现（钱当天就到手）。
                ko_leg = self.amount * n_ko * df_ko
                flag_N[(ss >= self.K) * ~condition_ko] = 1
                cashflow = (((self.K - ss) * ~condition_ko) * (
                        flag_N * self.N + ~flag_N * 1)) * discount_factor

            price_ls = np.sum(cashflow, 1) + ko_leg[:, 0]
            price = np.mean(price_ls, 0)

        return price


    # 熔断赔付·区间固赔·每日杠杆累计（每日结算）
    # fix    : 区间赔付——标的落在 K~H 区间时每日结算的金额
    # amount : 熔断赔付——熔断日起每日结算的金额
    # 两者曾经都叫「固定赔付」，读代码时分不出说的是哪一个。
    def Opt_ASGQ_DFF(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            flag_N = sr[:idx] <= self.K
            price = np.sum((sr[:idx] - self.K) * flag_N * self.N + self.fix * ~flag_N) + self.amount * (le - idx)
            return price
        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            flag_N = sr[:idx] >= self.K
            price = np.sum((self.K - sr[:idx]) * flag_N * self.N + self.fix * ~flag_N) + self.amount * (le - idx)
            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                flag_N = sr <= self.K
                price = np.sum((sr - self.K) * flag_N * self.N + self.fix * ~flag_N)
            else:
                flag_N = sr >= self.K
                price = np.sum((self.K - sr) * flag_N * self.N + self.fix * ~flag_N)

        else:
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss >= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：金额是常数 amount，但**结算日**同样要对齐——
                # 整笔在熔断当日付清，按当日因子折回 0 时刻。
                ko_leg = self.amount * n_ko * df_ko
                flag_N = (ss <= self.K) & ~condition_ko
                cashflow = ((ss - self.K) * flag_N * self.N
                            + self.fix * ~flag_N * ~condition_ko) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss <= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：金额是常数 amount，但**结算日**同样要对齐——
                # 整笔在熔断当日付清，按当日因子折回 0 时刻。
                ko_leg = self.amount * n_ko * df_ko
                flag_N = (ss >= self.K) & ~condition_ko
                cashflow = ((self.K - ss) * flag_N * self.N
                            + self.fix * ~flag_N * ~condition_ko) * discount_factor

            price_ls = np.sum(cashflow, 1) + ko_leg[:, 0]
            price = np.mean(price_ls, 0)

        return price

    # 熔断赔付·区间固赔·到期杠杆累计（每日结算）
    # fix    : 区间赔付——标的落在 K~H 区间时每日结算的金额
    # amount : 熔断赔付——熔断日起每日结算的金额
    # 杠杆腿到期日观察：到期收盘价穿越执行价时，按累计天数 le 结算 (N-1) 倍杠杆
    def Opt_ASGQ_EFF(self) -> float:

        T = self.T_days / annual_days
        nStep = self.T_days
        dt = 1 / annual_days
        sr = np.array(self.sr)
        observ = np.array(self.observ)
        le = len(observ)

        # 是否熔断提前结束
        if self.cp == 1 and any(sr >= self.H):
            idx = np.where(sr >= self.H)[0][0]
            flag_ko = np.cumsum(sr >= self.H) > 0
            band = (sr > self.K) & (sr < self.H)
            price = np.sum(((sr - self.K) * ~band + self.fix * band) * ~flag_ko) + self.amount * (le - idx)
            return price
        elif self.cp == -1 and any(sr <= self.H):
            idx = np.where(sr <= self.H)[0][0]
            flag_ko = np.cumsum(sr <= self.H) > 0
            band = (sr < self.K) & (sr > self.H)
            price = np.sum(((self.K - sr) * ~band + self.fix * band) * ~flag_ko) + self.amount * (le - idx)
            return price

        # 交易是否到期
        if self.T_days == 0:

            if self.cp == 1:
                band = (sr > self.K) & (sr < self.H)
                price = np.sum((sr - self.K) * ~band + self.fix * band)
                if sr[-1] <= self.K:
                    price = price + (sr[-1] - self.K) * le * (self.N - 1)
            else:
                band = (sr < self.K) & (sr > self.H)
                price = np.sum((self.K - sr) * ~band + self.fix * band)
                if sr[-1] >= self.K:
                    price = price + (self.K - sr[-1]) * le * (self.N - 1)

        else:
            flag_N = np.zeros([self.nPath, le], dtype=bool)
            condition_ko = np.zeros([self.nPath, le], dtype=bool)
            # 只沿观察日变化，对路径是常数：直接靠广播参与后面的逐元素
            # 乘法，不必物化成 nPath × le 的矩阵。nPath=1e5、le=243 时
            # 这一个 tile 就是 194 MB，而它每一行都一模一样。
            discount_factor = np.exp(
                -self.r * (np.maximum(observ - self.T_over, 0)) * dt)
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T, self.nPath, nStep,
                       seed=self.mc_seed, draw_steps=self.mc_draw_steps)
            ss = np.c_[np.tile(sr, (self.nPath, 1)), S]
            if self.cp == 1:
                condition_ko[np.cumsum(ss >= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss >= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：金额是常数 amount，但**结算日**同样要对齐——
                # 整笔在熔断当日付清，按当日因子折回 0 时刻。
                ko_leg = self.amount * n_ko * df_ko
                # 未熔断且到期收盘 <= K 的路径，到期日结算杠杆腿
                flag_N[(ss[:, -1] <= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                # 注意：MATLAB 原版杠杆腿误用剩余天数 T_Days（到期分支恒为 0），
                # 此处按代码注释「累计天数」及同族 Opt_ASGQ_EF/EP 的口径改用 le。
                cashflow = (self.fix * ((ss > self.K) & (ss < self.H)) * ~condition_ko
                            + (ss - self.K) * (ss <= self.K) * ~condition_ko
                            + (ss[:, -1] - self.K).reshape(self.nPath, 1) * le * (self.N - 1) * flag_N) * discount_factor
            else:
                condition_ko[np.cumsum(ss <= self.H, axis=1) > 0] = 1
                _, df_ko = self._knockout_state(
                    ss, ss <= self.H, discount_factor)
                n_ko = condition_ko.sum(axis=1, keepdims=True)
                # 熔断腿：金额是常数 amount，但**结算日**同样要对齐——
                # 整笔在熔断当日付清，按当日因子折回 0 时刻。
                ko_leg = self.amount * n_ko * df_ko
                flag_N[(ss[:, -1] >= self.K) & (np.all(condition_ko == 0, axis=1)), -1] = 1
                cashflow = (self.fix * ((ss < self.K) & (ss > self.H)) * ~condition_ko
                            + (self.K - ss) * (ss >= self.K) * ~condition_ko
                            + (self.K - ss[:, -1]).reshape(self.nPath, 1) * le * (self.N - 1) * flag_N) * discount_factor

            price_ls = np.sum(cashflow, 1) + ko_leg[:, 0]
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
