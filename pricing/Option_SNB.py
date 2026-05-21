# _*_ coding: utf-8 _*_
"""
雪球期权大类（由 MATLAB Option_SNB / Opt_Snowball 移植）

支持不同 type 的雪球期权价格及 Greeks 计算：
    - cp = -1  卖看跌，雪球（KI 向下、KO 向上）
    - cp =  1  卖看涨，反雪球（KI 向上、KO 向下）

敲出观察日两种输入方式（对应原 MATLAB 的两个版本）：
    - ko_observ : 敲出观察交易日序号序列（1-based，如 range(5, 61, 5)），自包含、无需日历
    - ko_date   : 敲出观察自然日日期序列（如 ["20220826", ...]），需交易日历换算成序号

计息方式（act）：
    - act = 1  交易日计息：纯序号运算，不依赖交易日历
    - act = 0  自然日计息：需交易日历把观察序号映射到日期，再按自然日数贴现/计息

MC 引擎复用 pricing.mc_engine.McGbmQ（对偶变量 + 固定 seed=20），与
Option_AB / Option_AS / Option_DE 保持一致；因此结果与原 MATLAB 在统计意义上
等价，但不会逐位相同（MATLAB 的 Mersenne-Twister 序列无法用 NumPy 复现）。

Greeks 计算直接复用 OptionBase.get_greeks（有限差分 bump-and-reprice），本类
仅实现定价及其所需的 _time_remaining / _theta_overrides / _decrement_time 钩子。

@author: Grefer
"""

import numpy as np

try:
    from .constants import ANNUAL_DAYS, CALENDAR_DAYS
    from .mc_engine import McGbmQ
    from .option_base import OptionBase
except ImportError:
    from constants import ANNUAL_DAYS, CALENDAR_DAYS
    from mc_engine import McGbmQ
    from option_base import OptionBase


class Option_SNB(OptionBase):
    """雪球期权定价类

    Parameters
    ----------
    optiontype : str
        结构名，目前支持 "Opt_Snowball"。
    s00 : float
        入场价（期初价格），用于票息与亏损的名义换算。
    s0 : float
        最新价（当前标的价格），MC 模拟起点。
    K : float
        行权价（敲入后承担亏损的参考价，通常 = s00）。
    KI : float
        敲入价。
    KO : float | sequence
        敲出价。标量为平敲；传与 ko_observ 等长的序列即为降敲（逐观察日 KO）。
    T : int
        剩余交易日数。
    sigma : float
        波动率。
    coupon : float
        未敲出票息率（年化）。
    coupon_ko : float
        敲出票息率（年化）。
    margin : float
        保证金比例。仅在 margin_call=False（不追保/有限损失）时作为最大亏损
        封顶比例，亏损以 -margin * s00 封顶。
    margin_call : bool, optional
        True = 追保雪球，敲入未敲出亏损不按保证金封顶；False = 不追保雪球，
        亏损限制在保证金内。未传时保留旧口径：margin == 0 视为追保，
        margin > 0 视为不追保。
    act : int
        计息方式：0 = 自然日计息，1 = 交易日计息。
    cp : int
        cp = -1 雪球（卖看跌），cp = 1 反雪球（卖看涨）。
    r, q : float
        无风险利率、分红率。
    sr : list, optional
        已实现价格序列（不含当前 s0）。默认空。
    startdate, enddate : str, optional
        起息日 / 到期日（"YYYYMMDD"）。act = 0 或使用 ko_date 时需要。
    ko_observ : list, optional
        敲出观察交易日序号序列（1-based）。与 ko_date 至少提供一个；
        只给 ko_date 时会通过交易日历换算出 ko_observ。
    ko_date : list, optional
        敲出观察自然日日期序列，与 ko_observ 元素一一对应（同一批观察日）。
        act = 0 时提供它即可直接得到敲出日期，省去查日历。
    datenow : str, optional
        估值日（"YYYYMMDD"）。act = 0 贴现/计息需要"今日"。未提供时：sr 为空
        取 startdate，否则用交易日历推算（startdate 后第 len(sr) 个交易日）。
    nPath : int
        MC 路径数（需为偶数），默认 100000。
    trading_calendar : list, optional
        显式交易日列表（"YYYYMMDD" 或 datetime），覆盖 [startdate, enddate]，
        calendar[0] = startdate。提供后完全不触发文件/接口查询。

    交易日历依赖说明
    ----------------
    act = 1 + ko_observ（序号）：完全自包含，无需任何日历。
    act = 0：仅当需要"序号 ↔ 日期"互转时才查日历，按离线优先解析
        （详见 trade_calendar：本地 data/tradingday.csv → akshare → Wind）。
        若直接同时提供 ko_observ（序号）、ko_date（日期）与 datenow（或 sr 为空），
        则 act = 0 也完全无需查日历/接口。
    """

    def __init__(self,
                 optiontype: str,
                 s00: float,
                 s0: float,
                 K: float,
                 KI: float,
                 KO: float,
                 T: int,
                 sigma: float,
                 coupon: float,
                 coupon_ko: float,
                 margin: float,
                 act: int,
                 cp: int,
                 r: float = 0.03,
                 q: float = 0.03,
                 sr: list = None,
                 startdate: str = None,
                 enddate: str = None,
                 ko_observ: list = None,
                 ko_date: list = None,
                 datenow: str = None,
                 nPath: int = 100000,
                 trading_calendar: list = None,
                 margin_call = None,
                 **kwargs: float
                 ):
        self.optiontype = optiontype
        self.s00 = s00
        self.s0 = s0
        self.K = K
        self.KI = KI
        self.KO = KO
        self.T = T
        self.sigma = sigma
        self.coupon = coupon
        self.coupon_ko = coupon_ko
        self.margin = float(margin)
        if self.margin < 0:
            raise ValueError(f"margin 输入有误：{margin}（需 >= 0）")
        if margin_call is None:
            self.margin_call = (self.margin == 0)
        else:
            self.margin_call = self._to_bool(margin_call)
        if not self.margin_call and self.margin <= 0:
            raise ValueError("不追保雪球必须设置 margin > 0 作为最大亏损封顶比例。")
        self.act = act
        self.cp = cp
        self.r = r
        self.q = q
        self.sr = list(sr) if sr is not None else []
        self.startdate = startdate
        self.enddate = enddate
        self.ko_observ = ko_observ
        self.ko_date = ko_date
        self.datenow = datenow
        self.nPath = nPath
        self.trading_calendar = trading_calendar

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("1", "true", "yes", "y", "追保", "margin_call"):
                return True
            if s in ("0", "false", "no", "n", "不追保", "no_margin_call"):
                return False
            raise ValueError(f"margin_call 输入有误：{value!r}")
        return bool(value)

    # ------------------------------------------------------------------
    #  定价入口与 OptionBase 钩子
    # ------------------------------------------------------------------
    def get_price(self):
        return getattr(self, self.optiontype)()

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

    # ------------------------------------------------------------------
    #  交易日历 / 敲出观察日解析
    # ------------------------------------------------------------------
    @staticmethod
    def _to_date64(d):
        """把 'YYYYMMDD' / datetime / date / datetime64 统一为 datetime64[D]"""
        if isinstance(d, np.datetime64):
            return d.astype('datetime64[D]')
        if isinstance(d, str):
            s = d.replace('-', '').replace('/', '')
            return np.datetime64(f"{s[0:4]}-{s[4:6]}-{s[6:8]}", 'D')
        return np.datetime64(d).astype('datetime64[D]')

    def _calendar_dates(self):
        """返回覆盖 [startdate, enddate] 的交易日 datetime64[D] 数组，
        calendar[0] = startdate。优先用显式 trading_calendar，否则交给
        trade_calendar 按"本地文件 → akshare → Wind"的离线优先顺序解析。"""
        if self.trading_calendar is not None:
            return np.array([self._to_date64(d) for d in self.trading_calendar])
        try:
            from .trade_calendar import trade_days
        except ImportError:
            from trade_calendar import trade_days
        return trade_days(self.startdate, self.enddate)

    def _resolve_ko_observ(self):
        """得到 1-based 的敲出观察交易日序号数组（指向 ss 的列）。"""
        if self.ko_observ is not None:
            return np.asarray(self.ko_observ, dtype=int)
        if self.ko_date is not None:
            cal = self._calendar_dates()
            ko_days = np.array([self._to_date64(d) for d in self.ko_date])
            pos = []
            for d in ko_days:
                hit = np.where(cal == d)[0]
                if hit.size == 0:
                    raise ValueError(f"敲出观察日 {d} 不在交易日历中，请检查 ko_date / 日历。")
                pos.append(hit[0])
            return np.asarray(pos, dtype=int) + 1  # 转 1-based 列序号
        raise ValueError("必须提供 ko_observ（交易日序号）或 ko_date（观察日期）之一。")

    def _resolve_ko_level(self, ko_observ):
        """返回敲出价：标量 KO 直接返回；序列 KO（降敲）须与 ko_observ 等长，
        逐观察日比较时按最后一轴广播。"""
        ko = np.asarray(self.KO, dtype=float)
        if ko.ndim == 0:
            return float(ko)
        if ko.size != len(ko_observ):
            raise ValueError(
                f"降敲 KO 序列长度 {ko.size} 与敲出观察日数 {len(ko_observ)} 不一致。")
        return ko

    # ------------------------------------------------------------------
    #  雪球 MC 定价
    # ------------------------------------------------------------------
    def Opt_Snowball(self):
        dT = 1.0 / ANNUAL_DAYS       # 交易日年化因子
        dt = 1.0 / CALENDAR_DAYS     # 自然日年化因子
        nPath = int(self.nPath)
        if nPath <= 0 or nPath % 2 != 0:
            raise ValueError(f"nPath 必须为正偶数，当前为 {self.nPath}。")
        T = int(self.T)
        if T < 0:
            raise ValueError(f"T 必须 >= 0，当前为 {self.T}。")
        sr = np.asarray(self.sr, dtype=float)
        nsr = sr.size
        T_all = nsr + T              # 产品全期交易日数（已实现 + 剩余）

        ko_observ = self._resolve_ko_observ()
        if ko_observ.ndim != 1 or ko_observ.size == 0:
            raise ValueError("ko_observ 必须是一维非空观察日序列。")
        if np.any(ko_observ < 1) or np.any(ko_observ > T_all):
            raise ValueError(
                f"ko_observ 必须落在 [1, {T_all}] 内，当前范围为 "
                f"[{int(np.min(ko_observ))}, {int(np.max(ko_observ))}]。")
        cols = ko_observ - 1         # 转 0-based 列索引
        ko_level = self._resolve_ko_level(ko_observ)   # 标量或逐观察日 KO（降敲）

        # 模拟资产价格序列：列 = [已实现 sr | 自 s0 模拟的未来 T 日]
        if T > 0:
            S = McGbmQ(self.s0, self.r - self.q, self.sigma, T * dT,
                       nPath, T, seed=self.mc_seed)
            ss = np.c_[np.tile(sr, (nPath, 1)), S] if nsr > 0 else S
        else:
            ss = np.tile(sr, (nPath, 1))

        # 敲入（全路径逐日监测）与敲出（仅在观察日，KO 可逐日不同=降敲）
        if self.cp == -1:
            ki_flag = np.any(ss <= self.KI, axis=1)
            ko = ss[:, cols] >= ko_level
        elif self.cp == 1:
            ki_flag = np.any(ss >= self.KI, axis=1)
            ko = ss[:, cols] <= ko_level
        else:
            raise ValueError(f"cp 输入有误：{self.cp}（需 -1 雪球 或 1 反雪球）")

        ko_flag = np.any(ko, axis=1)
        idx_first = np.argmax(ko, axis=1)        # 各路径首个敲出在 ko_observ 中的位置
        ko_col = ko_observ[idx_first]            # 首个敲出列序号（1-based，仅敲出路径有意义）

        # 到期亏损（敲入未敲出时承担）。追保模式不封顶；不追保模式按保证金封顶。
        end_price = ss[:, -1]
        if self.cp == -1:
            loss = np.minimum(end_price - self.K, 0.0)
        else:
            loss = np.minimum(self.K - end_price, 0.0)
        if not self.margin_call:
            loss = np.maximum(loss, -self.margin * self.s00)

        price_ls = np.zeros(nPath)
        no_ki = ~ki_flag
        loss_paths = ki_flag & (~ko_flag)

        if self.act == 1:
            # ---- 交易日计息（自包含，无需日历）----
            disc_full = np.exp(-self.r * T * dT)
            T_ko = ko_col                        # 起息到敲出的交易日数
            dT_ko = T_ko - nsr                   # 当前到敲出的交易日数（贴现期）
            # ① 未敲入未敲出：全期票息
            price_ls[no_ki & (~ko_flag)] = self.s00 * self.coupon * T_all * dT * disc_full
            # ②③ 敲出（无论是否敲入）：按持有期计敲出票息
            ko_pay = self.s00 * self.coupon_ko * T_ko * dT * np.exp(-self.r * dT_ko * dT)
            price_ls[ko_flag] = ko_pay[ko_flag]
            # ④ 敲入未敲出：承担到期亏损
            price_ls[loss_paths] = loss[loss_paths] * disc_full

        elif self.act == 0:
            # ---- 自然日计息：需把观察序号映射为日历日期。优先用直接输入的
            #      ko_date / datenow，缺啥才按需查一次日历（离线优先），
            #      若三者齐备则完全不触发日历/接口。----
            start64 = self._to_date64(self.startdate)
            end64 = self._to_date64(self.enddate)
            t_all = (end64 - start64).astype('timedelta64[D]').astype(int) + 1

            # 估值日（今日）
            if self.datenow is not None:
                now64 = self._to_date64(self.datenow)
            elif nsr == 0:
                now64 = start64
            else:
                now64 = self._calendar_dates()[min(T_all, nsr + 1) - 1]

            # 各路径首个敲出对应的日历日期
            if self.ko_date is not None:
                obs_dates = np.array([self._to_date64(d) for d in self.ko_date])
                ko_days = obs_dates[idx_first]   # ko_date 与 ko_observ 元素一一对应
            else:
                cal = self._calendar_dates()
                if cal.size < int(ko_observ.max()):
                    raise ValueError(
                        f"[startdate, enddate] 内交易日数 {cal.size} 少于最大敲出观察序号 "
                        f"{int(ko_observ.max())}；请核对 startdate/enddate/T 是否与实际交易日历一致，"
                        f"或直接提供 ko_date 以免查日历。")
                ko_days = cal[ko_col - 1]

            t_days = (end64 - now64).astype('timedelta64[D]').astype(int)
            disc_full = np.exp(-self.r * t_days * dt)
            t_ko = (ko_days - start64).astype('timedelta64[D]').astype(int) + 1
            dt_ko = np.maximum((ko_days - now64).astype('timedelta64[D]').astype(int), 0)
            # ① 未敲入未敲出：全期票息
            price_ls[no_ki & (~ko_flag)] = self.s00 * self.coupon * t_all * dt * disc_full
            # ②③ 敲出：按持有自然日计敲出票息
            ko_pay = self.s00 * self.coupon_ko * t_ko * dt * np.exp(-self.r * dt_ko * dt)
            price_ls[ko_flag] = ko_pay[ko_flag]
            # ④ 敲入未敲出：承担到期亏损
            price_ls[loss_paths] = loss[loss_paths] * disc_full

        else:
            raise ValueError(f"act 输入有误：{self.act}（需 0 自然日计息 或 1 交易日计息）")

        return float(np.mean(price_ls))

    # ------------------------------------------------------------------
    #  敲出提前了结检测（供对冲回测截断存续期，见 OptionBase.knockout_event）
    # ------------------------------------------------------------------
    def knockout_event(self, prices, steps_per_day=1):
        """沿已知价格路径 prices（prices[0]=今日收盘）逐观察日检查敲出。

        观察日序号为产品全期 1-based 绝对交易日序号；prices[0] 对应今日的绝对
        序号 = len(sr)。日频时观察日 v ↔ prices[v-len(sr)]；steps_per_day>1 时
        取该日收盘 bar = (v-len(sr))*spd。首次触发敲出返回 (i_ko, settle_value)：
        settle_value 为敲出当日结算票息，与定价口径一致——act=1 按交易日
        s00*coupon_ko*v/ANNUAL_DAYS，act=0 按起息日到敲出日的自然日数
        s00*coupon_ko*t_ko/CALENDAR_DAYS。未敲出返回 None。
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices) - 1
        spd = max(1, int(steps_per_day))
        start_off = len(self.sr)
        ko_observ = np.asarray(self._resolve_ko_observ(), dtype=int)
        ko_level = self._resolve_ko_level(ko_observ)     # 标量或逐观察日（降敲）
        ko_arr = np.broadcast_to(ko_level, (len(ko_observ),))

        if self.cp == -1:
            def breached(x, lv): return x >= lv
        elif self.cp == 1:
            def breached(x, lv): return x <= lv
        else:
            return None

        for k in np.argsort(ko_observ):
            v = int(ko_observ[k])
            j = (v - start_off) * spd            # 该观察日收盘在 prices 中的索引
            if j < 1 or j > n:
                continue                          # 已过去 或 超出回测窗口
            if breached(prices[j], ko_arr[k]):
                if self.act == 0:
                    start64 = self._to_date64(self.startdate)
                    if self.ko_date is not None:
                        ko_day = self._to_date64(self.ko_date[k])
                    else:
                        ko_day = self._calendar_dates()[v - 1]
                    t_ko = (ko_day - start64).astype('timedelta64[D]').astype(int) + 1
                    settle = self.s00 * self.coupon_ko * t_ko / CALENDAR_DAYS
                else:
                    settle = self.s00 * self.coupon_ko * v / ANNUAL_DAYS
                return j, float(settle)
        return None


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    # 交易日计息（自包含）：对应 MATLAB 例子，ko_observ = 5:5:60
    option = Option_SNB(
        "Opt_Snowball", s00=100, s0=100, K=100, KI=93, KO=103, T=60,
        sigma=0.23, coupon=0.37, coupon_ko=0.37, margin=0.2, act=1, cp=-1,
        r=0.03, q=0.03, sr=[], ko_observ=list(range(5, 61, 5)),
    )
    p = option.get_price()
    greeks = option.get_greeks()
    end = time.perf_counter()
    print('price  = %.4f' % p)
    print('greeks = [delta=%.4f, gamma=%.6f, vega=%.4f, theta=%.4f, rho=%.4f]' % tuple(greeks))
    print('历时 %.2f 秒' % (end - start))
