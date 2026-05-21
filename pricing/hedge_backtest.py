# _*_ coding: utf-8 _*_
"""
DeltaLab - 期权动态对冲回测框架

支持对任意继承自 OptionBase 的期权进行 Delta 对冲回测，
追踪每日盈亏分解、累计对冲误差和 Greeks 变动。
"""

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .constants import ANNUAL_DAYS
except ImportError:
    from constants import ANNUAL_DAYS


# ============================================================
#  对冲触发策略（HedgeStrategy）
# ============================================================
#
# 主循环在每个 bar 调用 strategy.should_hedge(ctx) -> bool 决定是否重新
# 计算目标持仓并交易。除 Day 0 建仓、Day n 到期平仓外，所有对冲决策都
# 交给 strategy。
#
# ctx 字段：
#   i            : 当前 bar 索引（1..n-1，Day 0 / Day n 主循环单独处理）
#   S            : 当前 bar 标的价
#   S_last       : 上一次触发对冲时的标的价（Day 0 或上次 hedge）
#   i_last       : 上一次触发对冲时的 bar 索引
#   dt_bar       : 单 bar 的年化时长
#   sigma_impl   : 期权隐含波动率（option.sigma）
#   log_ret_hist : 截至当前 bar 的对数收益序列（长度 = i），np.ndarray


class HedgeStrategy:
    """对冲触发策略基类。子类实现 should_hedge(ctx) -> bool。"""

    name = "base"

    def should_hedge(self, ctx):
        raise NotImplementedError


class FixedFreqStrategy(HedgeStrategy):
    """按固定 bar 间隔触发：每 hedge_freq 个 bar 调仓一次。"""

    name = "fixed_freq"

    def __init__(self, hedge_freq=1):
        self.hedge_freq = max(1, int(hedge_freq))

    def should_hedge(self, ctx):
        return (ctx["i"] % self.hedge_freq) == 0


class SigmaBandStrategy(HedgeStrategy):
    """
    x-sigma 带触发：当 |ln(S/S_last)| >= k * sigma_ref * sqrt(dt_since_last)
    时重建 Δ 对冲。sigma_ref 有两种来源：
      - 'implied'  : 使用 option.sigma
      - 'realized' : 过去 window_days 个交易日对数收益的年化 std（intraday
                     下自动换算为 window_days * spd 根 bar），
                     不足样本时回退到 option.sigma

    window 参数（旧名）已弃用，保留兼容：若调用方传入 window，按 bar 粒度
    理解并打印 DeprecationWarning；推荐使用 window_days（日单位）。
    """

    name = "sigma_band"

    def __init__(self, k=0.5, sigma_source="implied", window_days=None, window=None):
        self.k = float(k)
        self.sigma_source = str(sigma_source).lower()
        if self.sigma_source not in ("implied", "realized"):
            raise ValueError(f"未知 sigma_source: {sigma_source}")

        # 归一化参数：优先 window_days（日），fallback 到旧 window（bar）并告警
        if window_days is not None:
            self.window_days = max(2, int(window_days))
            self._legacy_window_bars = None
        elif window is not None:
            import warnings
            warnings.warn(
                "SigmaBandStrategy(window=...) 已弃用，请改用 window_days（单位=日）。"
                "传入的 window 按 bar 粒度直接解释。",
                DeprecationWarning, stacklevel=2,
            )
            self.window_days = max(2, int(window))  # 保留一个 days 估计值兜底
            self._legacy_window_bars = max(2, int(window))
        else:
            self.window_days = 20
            self._legacy_window_bars = None

        # 最近一次 should_hedge 返回 True 的 bar 索引；realized σ 估计时
        # 会剔除该 bar 对应的 log return，避免触发自身污染窗口样本。
        self._last_trigger_i = 0

    @property
    def window(self):
        """向后兼容属性：返回 window_days。"""
        return self.window_days

    def _window_bars(self, ctx):
        if self._legacy_window_bars is not None:
            return self._legacy_window_bars
        spd = int(ctx.get("steps_per_day", 1))
        return self.window_days * max(1, spd)

    def _sigma_ref(self, ctx):
        if self.sigma_source == "implied":
            return float(ctx["sigma_impl"])
        lr = ctx["log_ret_hist"]
        if lr is None:
            return float(ctx["sigma_impl"])
        win_bars = self._window_bars(ctx)

        # 剔除"触发当根 bar"的 log return：该 bar 的收益是触发 σ 带的原因，
        # 若回灌到窗口里会系统性抬高 σ 估计，进一步抑制下一次触发（偏差）。
        # log_ret_hist[k] = ln(S[k+1]/S[k])，所以触发 bar i 对应的 return 索引是 i-1。
        i_trig = int(self._last_trigger_i)
        mask = np.ones(len(lr), dtype=bool)
        if 0 < i_trig <= len(lr):
            mask[i_trig - 1] = False
        clean = lr[mask]
        if len(clean) < max(2, min(win_bars, 2)):
            return float(ctx["sigma_impl"])
        tail = clean[-win_bars:]
        if len(tail) < 2:
            return float(ctx["sigma_impl"])
        std = float(np.std(tail, ddof=1))
        if not np.isfinite(std) or std <= 0:
            return float(ctx["sigma_impl"])
        spd = int(ctx.get("steps_per_day", 1))
        return std * np.sqrt(ANNUAL_DAYS * max(1, spd))

    def should_hedge(self, ctx):
        sigma_ref = self._sigma_ref(ctx)
        if sigma_ref <= 0:
            return False
        bars_since = max(1, ctx["i"] - ctx["i_last"])
        # dt_since_last = bars_since * dt_bar（从上次触发至今的年化时长）
        dt_since = bars_since * ctx["dt_bar"]
        threshold = self.k * sigma_ref * np.sqrt(dt_since)
        s_last = ctx["S_last"]
        if s_last <= 0:
            return False
        move = abs(np.log(ctx["S"] / s_last))
        triggered = move >= threshold
        if triggered:
            self._last_trigger_i = int(ctx["i"])
        return triggered


# ============================================================
#  期权要素伸缩（用于真实行情回测）
# ============================================================
#
# 用户在 GUI 中通常基于一个"参考价" S_ref（即 option.s0）配置 strike /
# barrier 等价格量纲字段。当切换到真实行情时，真实起始价 S_real 与 S_ref
# 不一致，需要按比例 ratio = S_real / S_ref 把这些字段缩放到真实价格水平
# 上，否则期权结构会被破坏（例如 ATM call 变成深度虚值）。
#
# 仅缩放价格量纲 / cashflow 量纲字段；σ、r、q、期限、观察日索引、N、cp、
# 参与率等比例量保持不变。
#
# 字段白名单按期权类名维护。若未识别类名则只缩放 s0，并在 sr 已经写入历史
# 时按比例搬运（rebase 后历史价的相对位置保持不变）。

_PRICE_FIELDS_BY_CLS = {
    "Option_Vanilla": ("K",),
    "Option_AB":      ("K", "KI"),
    "Option_AS":      ("K", "E", "minPay", "maxPay"),
    "Option_DE":      ("K", "H", "P", "fix", "amount"),
    "Option_SNB":     ("s00", "K", "KI", "KO"),
}


def _rescale_option_to_real_s0(option, real_s0):
    """
    将期权对象的价格量纲要素从 option.s0 (S_ref) 缩放到 real_s0。

    返回 (scaled_option, info_dict)，其中 info_dict 记录原始值/缩放后值，
    便于 GUI / 日志展示。原对象不会被修改。

    Parameters
    ----------
    option : OptionBase
        用户配置的期权实例，option.s0 视为参考价 S_ref。
    real_s0 : float
        真实行情下的起始价格。

    Returns
    -------
    (OptionBase, dict)
    """
    s_ref = float(option.s0)
    if s_ref <= 0:
        raise ValueError(
            f"无法对参考价 s_ref={s_ref} 做伸缩处理；请在 GUI 中填入正的 s0 "
            f"作为期权要素的参考价水平。"
        )

    ratio = float(real_s0) / s_ref
    cls_name = type(option).__name__
    fields = _PRICE_FIELDS_BY_CLS.get(cls_name, ())

    opt = copy.deepcopy(option)

    info = {
        "cls": cls_name,
        "s_ref": s_ref,
        "s_real": float(real_s0),
        "ratio": ratio,
        "fields": {},   # name -> (old, new)
    }

    # s0 永远缩放
    info["fields"]["s0"] = (s_ref, float(real_s0))
    opt.s0 = float(real_s0)

    # 已经入场的历史价（sr）按比例搬运；保持相对位置
    if hasattr(opt, "sr") and opt.sr is not None and len(opt.sr) > 0:
        old_sr = list(opt.sr)
        opt.sr = [float(x) * ratio for x in old_sr]
        info["fields"]["sr"] = (old_sr, list(opt.sr))

    # 价格量纲字段
    for name in fields:
        if not hasattr(opt, name):
            continue
        old = getattr(opt, name)
        if old is None:
            continue
        if isinstance(old, (list, tuple, np.ndarray)):
            try:
                old_arr = np.asarray(old, dtype=float)
            except (TypeError, ValueError):
                continue
            if old_arr.size == 0 or not np.all(np.isfinite(old_arr)):
                continue
            new_arr = old_arr * ratio
            if isinstance(old, np.ndarray):
                new = new_arr
            else:
                new = new_arr.tolist()
            setattr(opt, name, new)
            info["fields"][name] = (old, new)
            continue
        # maxPay 用 float('inf') 占位时无需缩放
        try:
            old_f = float(old)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(old_f):
            continue
        new_f = old_f * ratio
        setattr(opt, name, new_f)
        info["fields"][name] = (old_f, new_f)

    return opt, info


def _format_rescale_info(info):
    """把 _rescale_option_to_real_s0 返回的 info 渲染成可读多行字符串。"""
    def _fmt(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=float).reshape(-1)
            if arr.size > 6:
                head = ", ".join(f"{x:.6f}" for x in arr[:3])
                tail = ", ".join(f"{x:.6f}" for x in arr[-2:])
                return f"[{head}, ..., {tail}]"
            return "[" + ", ".join(f"{x:.6f}" for x in arr) + "]"
        return f"{float(v):.6f}"

    lines = [
        f"[期权要素伸缩] {info['cls']}  S_ref={info['s_ref']:.4f} -> "
        f"S_real={info['s_real']:.4f}  ratio={info['ratio']:.6f}"
    ]
    for name, (old, new) in info["fields"].items():
        if name == "sr":
            n = len(old) if hasattr(old, "__len__") else 0
            lines.append(f"  - sr (历史 {n} 个) 已按比例搬运")
            continue
        lines.append(f"  - {name:<8s}: {_fmt(old):>14s} -> {_fmt(new)}")
    return "\n".join(lines)


# ---- 多线程工作函数 ----

def _run_single_path(args):
    """线程池工作函数：对单条路径执行回测

    注入 per-path mc_seed，避免所有路径共用同一批 MC 随机数、
    人为压窄多路径统计分布。base_seed=None 时走 OS 熵，完全独立。
    """
    (option_init, price_path, hedge_freq, tc_rate, position, quantity, multiplier,
     strategy, steps_per_day, slippage_bps, path_idx, base_seed) = args

    # 为本路径构造独立的 option 副本并注入 per-path seed
    opt_local = copy.deepcopy(option_init)
    if base_seed is None:
        opt_local.mc_seed = None
    else:
        opt_local.mc_seed = int(base_seed) + int(path_idx)

    bt = HedgeBacktest(
        opt_local, price_path,
        hedge_freq=hedge_freq, tc_rate=tc_rate,
        position=position, quantity=quantity, multiplier=multiplier,
        strategy=copy.deepcopy(strategy) if strategy is not None else None,
        steps_per_day=steps_per_day,
        slippage_bps=slippage_bps,
    )
    res = bt.run()
    return {
        'hedging_error': res['hedging_error'],
        'final_pnl': res['cumulative_pnl'][-1],
        'total_tc': res['total_tc'],
        'final_price': res['prices'][-1],
        'realized_vol': res['realized_vol'],
        'knocked_out': res.get('knocked_out', False),
        'ko_day': res.get('ko_day'),
    }


class HedgeBacktest:
    """
    期权动态对冲回测

    Parameters
    ----------
    option : OptionBase
        初始期权对象（t=0 时刻状态）
    prices : array-like
        标的资产每日价格序列，prices[0] = 期初价格，长度 = 总交易日数 + 1
    hedge_freq : int
        调仓频率（每隔几个交易日调仓），默认 1（每日调仓）
    tc_rate : float
        单边交易成本率，默认 0
    position : int
        期权头寸方向：1 = 卖出（short）期权，-1 = 买入（long）期权
    quantity : float
        交易数量（如 10 吨、1000 股等），用于将 Delta 转换为实际持仓。
        实际持仓 = position * delta * quantity。默认 1（等价于 1 份标的）
    multiplier : float
        合约乘数（每手对应的标的数量），用于取整到整数手。
        手数 = round(持仓 / multiplier)，实际持仓 = 手数 * multiplier。
        0 表示不取整（连续对冲）。默认 5，与 GUI 默认保持一致。
        例：quantity=10, multiplier=10 → 最多 1 手。

    Examples
    --------
    >>> from pricing import Option_AB, HedgeBacktest
    >>> opt = Option_AB('Opt_Airbag', 100, [], 100, 90, 20, list(range(1,21)),
    ...                 0.18, 0.8, 1, 1)
    >>> prices = HedgeBacktest.simulate_prices(100, 0.18, 20)
    >>> bt = HedgeBacktest(opt, prices, hedge_freq=1, tc_rate=0.001, position=1,
    ...                    quantity=10, multiplier=10)
    >>> results = bt.run()
    >>> bt.summary()
    >>> bt.plot()
    """

    def __init__(self, option, prices=None, hedge_freq=1, tc_rate=0.0, position=1,
                 quantity=1.0, multiplier=5,
                 path_source="gbm", external_path=None, detrend=False,
                 is_future=False, contract_multiplier=1.0,
                 strategy=None, steps_per_day=1, slippage_bps=0.0,
                 base_seed=20):
        """
        Parameters
        ----------
        path_source : str
            "gbm"（默认）：使用传入的 prices（外部生成，兼容历史行为）。
            "historical"：使用 external_path 作为标的价格路径。
        external_path : pd.Series | np.ndarray | None
            historical 模式下的标的价格序列，长度需 >= 回测所需步数 + 1。
        detrend : bool
            去趋势开关（本轮占位，暂不实现）。
        is_future : bool
            标的是否为期货。True 时按 `contract_multiplier` 取整到整数张，
            并在结果中给出理论 Δ 手数与离散化误差分项。
        contract_multiplier : float
            期货合约乘数（每张合约对应的标的数量），`is_future=True` 时生效。
        base_seed : int | None
            run_multi 多路径 MC 采样的基础种子。每条路径实际使用的 option.mc_seed
            = base_seed + path_idx，实现路径间独立采样的同时保持整体可复现。
            传 None 时所有路径走 OS 熵（完全随机）。默认 20 与旧行为兼容。
        """
        self.option_init = copy.deepcopy(option)
        self._path_source = str(path_source).lower()
        self._external_path = external_path
        self._detrend = bool(detrend)  # 占位，本 Phase 不使用
        self.is_future = bool(is_future)
        self.contract_multiplier = float(contract_multiplier)

        # ---- 价格来源分流 ----
        if self._path_source == "historical":
            if external_path is None:
                raise ValueError("path_source='historical' 时必须传 external_path")
            ext = np.asarray(
                external_path.values if hasattr(external_path, "values") else external_path,
                dtype=float,
            )
            if ext.ndim != 1:
                ext = ext.reshape(-1)
            if len(ext) < 2:
                raise ValueError(f"external_path 长度不足：需要 >= 2, 实际 {len(ext)}")
            self.prices = ext
        elif self._path_source == "gbm":
            if prices is None:
                raise ValueError("path_source='gbm' 时必须传 prices")
            self.prices = np.asarray(prices, dtype=float)
        else:
            raise ValueError(f"未知 path_source: {path_source}，仅支持 'gbm' | 'historical'")

        self.hedge_freq = max(1, int(hedge_freq))
        self.tc_rate = tc_rate
        self.position = position
        self.quantity = float(quantity)
        self.multiplier = float(multiplier)
        # 日内 intraday 支持：每个交易日切分成 steps_per_day 个 bar。
        # steps_per_day=1 时行为与旧代码一致。
        self.steps_per_day = max(1, int(steps_per_day))
        self.slippage_bps = float(slippage_bps)
        # strategy=None 时回退到旧的 FixedFreqStrategy(hedge_freq) 以保持兼容。
        self.strategy = strategy if strategy is not None else FixedFreqStrategy(self.hedge_freq)
        self.base_seed = base_seed
        self._results = None

        # ---- 价格序列长度校正：严格裁剪到期权剩余期限 ----
        #
        # 口径：Day 0 = 建仓日，Day T = 到期日，共 T+1 个价格点。
        # 如果外部价格序列（historical / from_wind / from_csv）长度 > T+1，
        # 必须裁剪；否则主循环会跑到到期日之后，此时 V[i]=0 与 V[T] 之间会
        # 产生一次伪 MtM 盈亏跳变，对冲头寸平仓分支（i==n）也会错位到非到
        # 期日上，整段盈亏分解失真。长度不足 T+1 时直接报错，避免静默截断
        # 期权期限。
        try:
            t_rem = int(option._time_remaining)
        except Exception:
            t_rem = None
        if t_rem is not None and t_rem > 0:
            # intraday 模式下 prices 每日有 steps_per_day 个 bar，总长 T*spd + 1。
            need = t_rem * self.steps_per_day + 1
            have = len(self.prices)
            if have < need:
                raise ValueError(
                    f"价格序列长度不足：期权剩余 {t_rem} 日 x steps_per_day="
                    f"{self.steps_per_day} 需要 {need} 个价格点，实际仅 {have} 个。"
                )
            if have > need:
                self.prices = self.prices[:need]

    def run(self):
        """执行回测，返回结果字典"""
        option = copy.deepcopy(self.option_init)
        S = self.prices
        n = len(S) - 1
        r = option.r
        spd = self.steps_per_day
        dt_bar = 1.0 / (ANNUAL_DAYS * spd)  # 单 bar 年化时长
        dt = 1.0 / ANNUAL_DAYS  # 日粒度（兼容旧字段）
        pos = self.position

        # ---- 敲出提前了结：路径已知时把存续期截断到敲出日 ----
        # 仅当期权实现了 knockout_event 且检测到敲出时生效（默认 None=不截断），
        # 普通期权行为完全不变。截断后该日即为新"到期日"，结算价值取敲出票息。
        ko_settle = None
        ko_event = None
        ko_fn = getattr(option, "knockout_event", None)
        if callable(ko_fn):
            ko_event = ko_fn(S, spd)
        if ko_event is not None:
            i_ko, ko_settle = ko_event
            S = S[:i_ko + 1]
            n = len(S) - 1

        # 存储数组
        V = np.zeros(n + 1)        # 期权理论价值
        delta = np.zeros(n + 1)    # Delta
        gamma = np.zeros(n + 1)    # Gamma
        vega = np.zeros(n + 1)     # Vega
        theta = np.zeros(n + 1)    # Theta
        rho = np.zeros(n + 1)      # Rho
        H = np.zeros(n + 1)        # 持有标的数量
        TC = np.zeros(n + 1)       # 当日交易成本

        # 期货模式下的理论/实际 Δ 手数记录
        delta_theo_qty = np.zeros(n + 1)  # 理论 Δ 手数（float，期货张数）
        delta_real_qty = np.zeros(n + 1)  # 实际 Δ 手数（int，期货张数）
        delta_disc_pnl = np.zeros(n + 1)  # 每日离散化误差盈亏

        # 取整辅助函数
        qty = self.quantity
        mult = self.multiplier
        is_fut = self.is_future
        cmult = self.contract_multiplier

        def _round_to_lots(target):
            """将目标持仓取整到 multiplier 的整数倍（向零取整）"""
            if mult <= 0:
                return target
            if target >= 0:
                return int(target / mult) * mult
            else:
                return -int(-target / mult) * mult

        def _compute_target(delta_val):
            """
            根据 Delta 计算目标持仓与期货理论/实际手数。

            Returns
            -------
            target_shares : float
                实际用于对冲的标的份额（H 存的就是这个量）
            theo_lots : float
                理论期货张数（仅 is_future=True 时有意义，否则为 0）
            real_lots : int
                实际期货张数（仅 is_future=True 时有意义，否则为 0）
            """
            raw = pos * delta_val * qty
            if is_fut and cmult > 0:
                # 期货：按合约张数取整（round to nearest）
                theo = raw / cmult
                real = int(np.rint(theo))
                return real * cmult, theo, real
            # 非期货：沿用原 multiplier 机制
            return _round_to_lots(raw), 0.0, 0

        # ---- Day 0: 建仓 ----
        V[0] = option.get_price() or 0.0
        greeks = option.get_greeks()
        delta[0], gamma[0], vega[0], theta[0], rho[0] = greeks

        H[0], delta_theo_qty[0], delta_real_qty[0] = _compute_target(delta[0])
        # Day 0 建仓同样含滑点：买入时成交价上浮、卖出时下浮
        sl_rate_d0 = self.slippage_bps * 1e-4
        if H[0] != 0:
            sign0 = 1.0 if H[0] > 0 else -1.0
            s_exec0 = S[0] * (1.0 + sign0 * sl_rate_d0)
            TC[0] = abs(H[0]) * s_exec0 * self.tc_rate + abs(H[0]) * S[0] * sl_rate_d0
        else:
            TC[0] = 0.0

        # 策略 context：记录最近一次触发对冲的 bar 索引 / 价格
        i_last = 0
        S_last = float(S[0])
        hedge_triggered = np.zeros(n + 1, dtype=bool)
        hedge_triggered[0] = True  # Day 0 建仓视为一次触发

        sl_rate = self.slippage_bps * 1e-4  # bps -> ratio

        # ---- Day 1 ~ n ----
        for i in range(1, n + 1):
            # 日粒度划分：i % spd == 0 时是"跨日收盘"，需要 step_forward；
            # 其他 bar 为日内 bar，仅用临时 bumped copy 重算 Δ，不污染 option 内部状态。
            crosses_day = (i % spd == 0)

            if crosses_day:
                option.step_forward(S[i])
                eval_opt = option
            else:
                # 日内：用临时副本评估 price / Δ，option 本体状态保持到当日收盘。
                # intraday_elapsed = (i % spd)/spd ∈ [1/spd, (spd-1)/spd]，
                # 让 Δ 在日内也按 bar 比例消耗 T，避免跨日 bar 单次跳变掉一天 Θ。
                elapsed = (i % spd) / spd
                eval_opt = option._bumped_copy(
                    s0=float(S[i]), _intraday_elapsed=elapsed
                )

            time_left = eval_opt._time_remaining

            # 计算价值与 Greeks
            if time_left > 0:
                val = eval_opt.get_price()
                V[i] = val if val is not None else 0.0
                greeks = eval_opt.get_greeks()
                delta[i], gamma[i], vega[i], theta[i], rho[i] = greeks
            elif time_left == 0:
                val = eval_opt.get_price()
                V[i] = val if val is not None else 0.0
                delta[i] = gamma[i] = vega[i] = theta[i] = rho[i] = 0.0
            else:
                V[i] = delta[i] = gamma[i] = vega[i] = theta[i] = rho[i] = 0.0

            # 敲出日（截断后的末日）：用敲出结算票息覆盖 MC 价值，Greeks 归零
            # （期权已了结，下方 i==n 分支会平掉对冲头寸）。
            if ko_settle is not None and i == n:
                V[i] = ko_settle
                delta[i] = gamma[i] = vega[i] = theta[i] = rho[i] = 0.0

            H[i] = H[i - 1]

            # 默认沿用上一期的理论/实际手数（若本 bar 未调仓）
            delta_theo_qty[i] = delta_theo_qty[i - 1]
            delta_real_qty[i] = delta_real_qty[i - 1]

            # 调仓逻辑
            # 滑点改变成交价：买入（trade>0）成交价上浮 sl_rate，卖出下浮 sl_rate。
            # 交易费率基于成交价收取；滑点部分 |trade|*S*sl_rate 已相当于 sign 化后
            # trade*(S_exec - S)，与交易费率各记一次、不重复。
            if i == n:
                # 到期：平仓所有标的头寸
                trade = -H[i]
                if trade != 0:
                    sign_trade = 1.0 if trade > 0 else -1.0
                    s_exec = S[i] * (1.0 + sign_trade * sl_rate)
                    cost = abs(trade) * s_exec * self.tc_rate + abs(trade) * S[i] * sl_rate
                else:
                    cost = 0.0
                TC[i] = cost
                H[i] = 0.0
                delta_theo_qty[i] = 0.0
                delta_real_qty[i] = 0
                hedge_triggered[i] = True
            else:
                # 组装策略 context
                ctx = {
                    "i": i,
                    "S": float(S[i]),
                    "S_last": S_last,
                    "i_last": i_last,
                    "dt_bar": dt_bar,
                    "steps_per_day": spd,
                    "sigma_impl": float(option.sigma),
                    # 对数收益历史（到当前 bar 为止），仅 realized σ 时用到
                    "log_ret_hist": np.log(S[1:i + 1] / S[0:i]) if i >= 1 else np.array([]),
                }
                if self.strategy.should_hedge(ctx):
                    target, theo, real = _compute_target(delta[i])
                    trade = target - H[i]
                    if trade != 0:
                        sign_trade = 1.0 if trade > 0 else -1.0
                        s_exec = S[i] * (1.0 + sign_trade * sl_rate)
                        cost = abs(trade) * s_exec * self.tc_rate + abs(trade) * S[i] * sl_rate
                    else:
                        cost = 0.0
                    TC[i] = cost
                    H[i] = target
                    delta_theo_qty[i] = theo
                    delta_real_qty[i] = real
                    hedge_triggered[i] = True
                    i_last = i
                    S_last = float(S[i])

        # ---- 盈亏分解 ----
        hedge_daily = np.zeros(n + 1)     # 标的头寸每日盈亏（不含 TC）
        option_daily = np.zeros(n + 1)    # 期权 MtM 每日盈亏（从对冲方视角）

        for i in range(1, n + 1):
            hedge_daily[i] = H[i - 1] * (S[i] - S[i - 1])
            option_daily[i] = -pos * (V[i] - V[i - 1]) * qty
            # 期货取整的离散化误差：(理论手数 - 实际手数) * 价格变动 * 合约乘数
            if is_fut:
                qty_gap = delta_theo_qty[i - 1] - float(delta_real_qty[i - 1])
                delta_disc_pnl[i] = qty_gap * (S[i] - S[i - 1]) * cmult

        net_daily = hedge_daily + option_daily - TC
        cum_pnl = np.cumsum(net_daily)

        # 对冲误差新口径：期权权利金收入 + 标的腿累计盈亏 - 累计 TC - 期末期权赔付
        hedging_error = pos * V[0] * qty + np.sum(hedge_daily) - np.sum(TC) - pos * V[-1] * qty

        # ---- 波动率分析 ----
        implied_vol = option.sigma  # 成交时隐含波动率

        # bar 级对数收益
        log_ret = np.log(S[1:] / S[:-1])  # 长度 n（若 intraday 则每 bar 一个）

        # intraday 情形下年化因子按 bar 数换算
        ann_factor = ANNUAL_DAYS * spd

        # 全区间已实现波动率（年化）
        realized_vol = np.std(log_ret, ddof=1) * np.sqrt(ann_factor) if n > 1 else 0.0

        # 滚动已实现波动率（窗口 = min(20, n)）
        win = min(20, n)
        rolling_realized = np.full(n + 1, np.nan)
        for i in range(win, n + 1):
            window_ret = log_ret[i - win:i]
            rolling_realized[i] = np.std(window_ret, ddof=1) * np.sqrt(ann_factor)
        # 前 win 个 bar 用累计已实现波动率填充
        for i in range(1, min(win, n + 1)):
            rolling_realized[i] = np.std(log_ret[:i], ddof=1) * np.sqrt(ann_factor) if i > 1 else 0.0
        rolling_realized[0] = 0.0

        # 波动率价差：隐含 - 已实现（正值 → 卖方有优势）
        vol_spread = implied_vol - realized_vol

        # 逐 bar 累计已实现波动率
        cumulative_realized = np.zeros(n + 1)
        for i in range(2, n + 1):
            cumulative_realized[i] = np.std(log_ret[:i], ddof=1) * np.sqrt(ann_factor)

        # intraday 下 n 是 bar 数，真正的日数 = n // spd。
        self._results = {
            'n_days': n // spd if spd > 0 else n,
            'n_bars': n,
            'steps_per_day': spd,
            'n_trade_days': n // spd if spd > 0 else n,
            # 敲出提前了结标记：knocked_out=True 时回测在 ko_day（交易日）截断，
            # ko_settle 为敲出当日结算票息（普通期权恒为 False/None）。
            'knocked_out': ko_settle is not None,
            'ko_day': (n // spd if spd > 0 else n) if ko_settle is not None else None,
            'ko_settle': ko_settle,
            'hedge_triggered': hedge_triggered,
            'strategy_name': getattr(self.strategy, 'name', 'unknown'),
            'prices': S,
            'opt_value': V,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'shares': H,
            'hedge_daily': hedge_daily,
            'option_daily': option_daily,
            'tc_paid': TC,
            'net_daily': net_daily,
            'cumulative_pnl': cum_pnl,
            'hedging_error': hedging_error,
            'total_tc': np.sum(TC),
            'implied_vol': implied_vol,
            'realized_vol': realized_vol,
            'rolling_realized': rolling_realized,
            'cumulative_realized': cumulative_realized,
            'vol_spread': vol_spread,
            # 期货取整相关（非期货场景下保持为 0 数组）
            'delta_theo_qty_series': delta_theo_qty,
            'delta_real_qty_series': delta_real_qty,
            'delta_discretization_pnl': np.cumsum(delta_disc_pnl),
            'delta_discretization_daily': delta_disc_pnl,
            'path_source': self._path_source,
            'is_future': self.is_future,
            'contract_multiplier': self.contract_multiplier,
            # 为方便 Phase 3 调用，暴露 total_pnl 标量（= cum_pnl[-1]）
            'total_pnl': float(cum_pnl[-1]) if n > 0 else 0.0,
        }
        return self._results

    def summary(self):
        """打印回测摘要"""
        if self._results is None:
            self.run()
        r = self._results
        n = r['n_days']

        lines = [
            "=" * 52,
            "          动态对冲回测结果摘要",
            "=" * 52,
            f"  回测天数          :  {n:>10d}   (Day 0=建仓, Day {n}=到期)",
            f"  调仓频率          :  每 {self.hedge_freq} 天",
            f"  交易成本率        :  {self.tc_rate * 100:.2f}%",
            "-" * 52,
            f"  标的初始价格      :  {r['prices'][0]:>12.4f}",
            f"  标的到期价格      :  {r['prices'][-1]:>12.4f}",
            f"  标的涨跌幅        :  {(r['prices'][-1] / r['prices'][0] - 1) * 100:>11.2f}%",
            "-" * 52,
            f"  期权初始价值      :  {r['opt_value'][0]:>12.4f}",
            f"  期权到期价值      :  {r['opt_value'][-1]:>12.4f}",
            "-" * 52,
            f"  标的对冲盈亏      :  {np.sum(r['hedge_daily']):>12.4f}",
            f"  期权 MtM 盈亏     :  {np.sum(r['option_daily']):>12.4f}",
            f"  累计交易成本      :  {r['total_tc']:>12.4f}",
            f"  对冲误差          :  {r['hedging_error']:>12.4f}",
            "-" * 52,
            "  【Greeks 统计】",
            f"  {'':15s}  {'初始值':>10s}  {'均值':>10s}  {'最大|值|':>10s}",
            f"  {'Delta':15s}  {r['delta'][0]:>10.4f}  {np.mean(r['delta']):>10.4f}  {np.max(np.abs(r['delta'])):>10.4f}",
            f"  {'Gamma':15s}  {r['gamma'][0]:>10.4f}  {np.mean(r['gamma']):>10.4f}  {np.max(np.abs(r['gamma'])):>10.4f}",
            f"  {'Vega':15s}  {r['vega'][0]:>10.4f}  {np.mean(r['vega']):>10.4f}  {np.max(np.abs(r['vega'])):>10.4f}",
            f"  {'Theta':15s}  {r['theta'][0]:>10.4f}  {np.mean(r['theta']):>10.4f}  {np.max(np.abs(r['theta'])):>10.4f}",
            f"  {'Rho':15s}  {r['rho'][0]:>10.4f}  {np.mean(r['rho']):>10.4f}  {np.max(np.abs(r['rho'])):>10.4f}",
            "-" * 52,
            f"  调仓次数          :  {int(np.sum(np.abs(np.diff(r['shares'])) > 1e-10)):>10d}",
            "=" * 52,
        ]
        print("\n".join(lines))
        return r

    def plot(self, figsize=(14, 10)):
        """绘制回测结果四宫格图"""
        if self._results is None:
            self.run()
        r = self._results
        days = np.arange(len(r['prices']))

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # (1) 标的价格走势
        ax = axes[0, 0]
        ax.plot(days, r['prices'], 'b-', linewidth=1.2)
        ax.set_title('标的价格路径', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('价格')
        ax.grid(True, alpha=0.3)

        # (2) Delta 与实际持仓
        # 注意：r['shares'] 已经把 position (+1 卖出 / -1 买入) 的方向吸收在里面，
        # 正负号即对冲方向；过去版本做过 shares/position 的归一化，但会使 short
        # 情形下曲线与标题/图例意图相反，这里直接展示 shares 的原始符号。
        ax = axes[0, 1]
        ax.plot(days, r['delta'], 'r-', label='Delta', linewidth=1.2)
        ax.plot(days, r['shares'], 'b--', label='实际持仓 (shares)',
                linewidth=1.0, alpha=0.7)
        ax.set_title('Delta 与对冲持仓', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('Delta / shares')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (3) 期权理论价值
        ax = axes[1, 0]
        ax.plot(days, r['opt_value'], 'g-', linewidth=1.2)
        ax.set_title('期权理论价值', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('价值')
        ax.grid(True, alpha=0.3)

        # (4) 累计对冲盈亏
        ax = axes[1, 1]
        cp = r['cumulative_pnl']
        ax.plot(days, cp, 'k-', linewidth=1.2, label='累计盈亏')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(days, 0, cp, where=cp >= 0, alpha=0.15, color='green')
        ax.fill_between(days, 0, cp, where=cp < 0, alpha=0.15, color='red')
        ax.set_title('累计对冲盈亏', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('盈亏')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig

    def to_dataframe(self):
        """将回测结果转为 pandas DataFrame"""
        if self._results is None:
            self.run()
        r = self._results
        import pandas as pd
        df = pd.DataFrame({
            '标的价格': r['prices'],
            '期权价值': r['opt_value'],
            'Delta': r['delta'],
            'Gamma': r['gamma'],
            'Vega': r['vega'],
            'Theta': r['theta'],
            'Rho': r['rho'],
            '持仓': r['shares'],
            '标的盈亏': r['hedge_daily'],
            '期权盈亏': r['option_daily'],
            '交易成本': r['tc_paid'],
            '每日净盈亏': r['net_daily'],
            '累计盈亏': r['cumulative_pnl'],
        })
        # 使用 Wind 日期作为索引（如有）
        if hasattr(self, '_wind_meta') and self._wind_meta is not None:
            df.index = self._wind_meta['dates']
            df.index.name = '日期'
        else:
            df.index.name = '交易日'
        return df

    @staticmethod
    def simulate_prices(s0, sigma, T_days, r=0.03, q=0.03, seed=42, steps_per_day=1):
        """
        生成 GBM 模拟价格路径，用于回测测试

        Parameters
        ----------
        s0 : float
            初始价格
        sigma : float
            波动率
        T_days : int
            总交易日数
        r, q : float
            无风险利率、分红率
        seed : int
            随机种子
        steps_per_day : int
            每日 bar 数；默认 1 与历史行为一致。

        Returns
        -------
        prices : ndarray, shape (T_days * steps_per_day + 1,)
        """
        spd = max(1, int(steps_per_day))
        rng = np.random.default_rng(seed)
        dt = 1.0 / (ANNUAL_DAYS * spd)
        n_steps = T_days * spd
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)
        log_returns = drift + vol * rng.standard_normal(n_steps)
        prices = np.zeros(n_steps + 1)
        prices[0] = s0
        prices[1:] = s0 * np.exp(np.cumsum(log_returns))
        return prices

    @staticmethod
    def simulate_multi_paths(s0, sigma, T_days, n_paths=100, r=0.03, q=0.03, seed=42,
                             steps_per_day=1):
        """
        批量生成多条价格路径，用于对冲效果的统计分析

        Returns
        -------
        paths : ndarray, shape (n_paths, T_days * steps_per_day + 1)
        """
        spd = max(1, int(steps_per_day))
        rng = np.random.default_rng(seed)
        dt = 1.0 / (ANNUAL_DAYS * spd)
        n_steps = T_days * spd
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)
        log_returns = drift + vol * rng.standard_normal((n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = s0
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
        return paths

    def run_multi(self, paths, progress_callback=None, max_workers=None,
                  base_seed=None):
        """
        对多条价格路径批量回测，返回详细统计（多线程并行）

        NumPy 运算会释放 GIL，ThreadPoolExecutor 可有效利用多核并行。

        Parameters
        ----------
        paths : ndarray, shape (n_paths, T_days + 1)
        progress_callback : callable(completed: int, total: int), optional
            每完成一条路径后调用，用于更新进度
        max_workers : int, optional
            并行线程数，默认 min(cpu_count, n_paths)
        base_seed : int | None, optional
            覆盖 self.base_seed 用于本次调用的 MC 种子基准；
            实际每路径 seed = base_seed + path_idx，保证 MC 采样在路径间
            不同但可复现。传 None 使用 self.base_seed。

        Returns
        -------
        dict with keys:
            n_paths        : int
            errors         : ndarray — 每条路径的对冲误差
            total_pnl      : ndarray — 每条路径的累计净盈亏 (cum_pnl[-1])
            total_tc       : ndarray — 每条路径的累计交易成本
            final_prices   : ndarray — 每条路径的到期标的价格
            implied_vol    : float   — 隐含波动率
            realized_vols  : ndarray — 每条路径的已实现波动率（年化）
            knocked_out    : ndarray — 每条路径是否提前敲出
            ko_days        : ndarray — 提前敲出路径的敲出日，否则 NaN
        """
        n = len(paths)
        errors = np.zeros(n)
        total_pnl = np.zeros(n)
        total_tc = np.zeros(n)
        final_prices = np.zeros(n)
        realized_vols = np.zeros(n)
        knocked_out = np.zeros(n, dtype=bool)
        ko_days = np.full(n, np.nan)
        implied_vol = self.option_init.sigma

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, n)

        # base_seed 解析：显式参数优先，否则用实例字段
        effective_seed = base_seed if base_seed is not None else self.base_seed

        task_args = [
            (self.option_init, paths[i], self.hedge_freq,
             self.tc_rate, self.position, self.quantity, self.multiplier,
             self.strategy, self.steps_per_day, self.slippage_bps,
             i, effective_seed)
            for i in range(n)
        ]

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_path, a): i
                       for i, a in enumerate(task_args)}
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()
                errors[idx] = res['hedging_error']
                total_pnl[idx] = res['final_pnl']
                total_tc[idx] = res['total_tc']
                final_prices[idx] = res['final_price']
                realized_vols[idx] = res['realized_vol']
                knocked_out[idx] = bool(res.get('knocked_out', False))
                if res.get('ko_day') is not None:
                    ko_days[idx] = float(res['ko_day'])
                completed += 1
                if progress_callback:
                    progress_callback(completed, n)

        return {
            'n_paths': n,
            'errors': errors,
            'total_pnl': total_pnl,
            'total_tc': total_tc,
            'final_prices': final_prices,
            'implied_vol': implied_vol,
            'realized_vols': realized_vols,
            'knocked_out': knocked_out,
            'ko_days': ko_days,
        }

    def plot_error_dist(self, errors, figsize=(10, 5)):
        """绘制对冲误差分布直方图"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(errors), color='red', linestyle='--',
                   label=f'均值={np.mean(errors):.4f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title(f'对冲误差分布 (n={len(errors)}, std={np.std(errors):.4f})', fontsize=12)
        ax.set_xlabel('对冲误差')
        ax.set_ylabel('频次')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    # ============================================================
    #  Wind 数据接入
    # ============================================================

    @classmethod
    def from_wind(cls, option, code, start_date, end_date,
                  hedge_freq=1, tc_rate=0.0, position=1, adjust="F",
                  quantity=1.0, multiplier=5,
                  strategy=None, steps_per_day=None, slippage_bps=0.0,
                  bar_size=None):
        """
        使用 Wind 历史行情创建回测实例

        Parameters
        ----------
        option : OptionBase
            期权对象（s0 会被替换为 start_date 当日收盘价）
        code : str
            Wind 标的代码，如 "000001.SH", "510050.SH"
        start_date : str
            回测起始日 "YYYY-MM-DD"（含当日，作为建仓日）；
            intraday 模式下也接受 "YYYY-MM-DD HH:MM:SS"。
        end_date : str
            回测结束日 "YYYY-MM-DD"（含当日）；intraday 同上。
        hedge_freq : int
            调仓频率（bar 数），默认 1
        tc_rate : float
            单边交易成本率
        position : int
            1=卖出期权, -1=买入期权
        adjust : str
            复权方式 "F"=前复权, "B"=后复权, ""=不复权
        bar_size : str | None
            None（默认）走 `w.wsd` 日频分支，行为与旧版一致；
            非 None 走 `w.wsi` intraday 分支，取值如 "1"/"5"/"15"/"30"/"60"。
        steps_per_day : int | None
            日频模式下强制为 1；intraday 模式下若未指定则按 A 股日盘 240
            分钟自动推导：`spd = 240 // int(bar_size)`（60min→4, 5min→48,
            1min→240）。显式传入时以传入值为准并打印告警。
            注意：240 分钟假设仅适用于 A 股 / 沪深 300 股指期货日盘；
            对含夜盘的商品/能源期货（.SHF/.DCE/.CZC/.INE/.GFE）会打印
            警告，建议手动传入 steps_per_day。

        Returns
        -------
        HedgeBacktest
            已设置好价格序列的回测实例

        Examples
        --------
        >>> from pricing import Option_AS, HedgeBacktest
        >>> opt = Option_AS('Asian', 0, [], 100, 100, 22, 22, 0.15, 1, 0, float('inf'))
        >>> bt = HedgeBacktest.from_wind(opt, '510050.SH', '2025-01-02', '2025-02-07')
        >>> bt.run()
        >>> bt.summary()
        """
        if bar_size in (None, "", "日频", "daily", "day", "d"):
            # -------- 日频分支（保持旧行为） --------
            try:
                from .wind_data import get_close_prices
            except ImportError:
                from wind_data import get_close_prices

            series = get_close_prices(code, start_date, end_date, adjust)
            prices = series.values

            if len(prices) < 2:
                raise ValueError(
                    f"价格数据不足: {code} {start_date}~{end_date} "
                    f"仅 {len(prices)} 条"
                )

            # 按真实起始价缩放期权要素
            real_s0 = float(prices[0])
            opt, rescale_info = _rescale_option_to_real_s0(option, real_s0)
            try:
                print(_format_rescale_info(rescale_info))
            except Exception:
                pass

            # 日频：若调用方传入 spd>1 则强制置 1 并告警
            spd_final = 1 if steps_per_day is None else int(steps_per_day)
            if spd_final != 1:
                print(f"[from_wind] 日频行情仅支持 steps_per_day=1，"
                      f"收到 {spd_final} 已置 1")
                spd_final = 1

            bt = cls(opt, prices, hedge_freq=hedge_freq, tc_rate=tc_rate,
                     position=position, quantity=quantity, multiplier=multiplier,
                     strategy=strategy, steps_per_day=spd_final,
                     slippage_bps=slippage_bps)
            used_len = len(bt.prices)
            bt._wind_meta = {
                'code': code,
                'start_date': start_date,
                'end_date': end_date,
                'dates': series.index[:used_len],
                'n_trade_days': used_len - 1,
                'bar_size': None,
            }
            bt._rescale_info = rescale_info
            return bt

        # -------- intraday 分支 --------
        try:
            from .wind_data import get_intraday_close
        except ImportError:
            from wind_data import get_intraday_close

        bar_size_str = str(bar_size).strip()
        # bar_size 合法性校验
        try:
            bar_min = int(bar_size_str)
        except ValueError:
            raise ValueError(
                f"非法 bar_size={bar_size!r}，需要数字字符串（分钟数），"
                f"例如 '1' / '5' / '60'。"
            )
        if bar_min <= 0:
            raise ValueError(f"bar_size 必须为正整数分钟数，收到 {bar_min}")

        series = get_intraday_close(code, start_date, end_date,
                                    bar_size=bar_size_str, adjust=adjust)
        prices = series.values
        if len(prices) < 2:
            raise ValueError(
                f"intraday 价格数据不足: {code} {start_date}~{end_date} "
                f"bar_size={bar_size_str}min 仅 {len(prices)} 条"
            )

        # 通过 Wind wss(exch_eng, sec_type) 查本地常量表，得到日内可交易分钟数；
        # 无法判定时兜底按 A 股/股指的 240 分钟推导，并打印警告。
        try:
            from .wind_data import get_trading_minutes_per_day
        except ImportError:
            from wind_data import get_trading_minutes_per_day

        total_min = get_trading_minutes_per_day(code)
        if total_min is not None:
            if total_min % bar_min != 0:
                print(f"[from_wind] 警告：{code} 日内 {total_min} 分钟无法被 "
                      f"bar_size={bar_min} 整除，spd 取整可能引入偏差。")
            auto_spd = max(1, total_min // bar_min)
        else:
            auto_spd = max(1, 240 // bar_min)
            print(f"[from_wind] 警告：无法从 Wind 获取 {code} 的交易分钟数，"
                  f"按 240 分钟（A 股/股指）推导 spd={auto_spd}。"
                  f"如该合约含夜盘，请显式传入 steps_per_day。")

        if steps_per_day is None:
            spd_final = auto_spd
        else:
            spd_final = max(1, int(steps_per_day))
            if spd_final != auto_spd:
                print(f"[from_wind] 用户显式传入 steps_per_day={spd_final}，"
                      f"覆盖自动推导值 {auto_spd} (bar_size={bar_min}min)")

        # 按真实起始价缩放期权要素
        real_s0 = float(prices[0])
        opt, rescale_info = _rescale_option_to_real_s0(option, real_s0)
        try:
            print(_format_rescale_info(rescale_info))
        except Exception:
            pass

        bt = cls(opt, prices, hedge_freq=hedge_freq, tc_rate=tc_rate,
                 position=position, quantity=quantity, multiplier=multiplier,
                 strategy=strategy, steps_per_day=spd_final,
                 slippage_bps=slippage_bps)
        used_len = len(bt.prices)
        used_index = series.index[:used_len]
        bt._wind_meta = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'dates': used_index,
            # intraday 下 n_trade_days 按 bar 数除以 spd 近似
            'n_trade_days': max(0, (used_len - 1) // max(1, spd_final)),
            'bar_size': bar_size_str,
        }
        bt._rescale_info = rescale_info
        return bt

    @classmethod
    def from_csv(cls, option, filepath, price_col='close',
                 date_col=None, hedge_freq=1, tc_rate=0.0, position=1,
                 quantity=1.0, multiplier=5,
                 strategy=None, steps_per_day=1, slippage_bps=0.0):
        """
        从 CSV 文件加载价格数据创建回测实例（无需 Wind 终端）

        Parameters
        ----------
        filepath : str
            CSV 文件路径
        price_col : str
            收盘价列名，默认 'close'
        date_col : str or None
            日期列名，None 则使用第一列

        Returns
        -------
        HedgeBacktest
        """
        import pandas as pd
        if date_col:
            df = pd.read_csv(filepath, parse_dates=[date_col], index_col=date_col)
        else:
            df = pd.read_csv(filepath, parse_dates=[0], index_col=0)

        if price_col not in df.columns:
            raise ValueError(f"列 '{price_col}' 不在 CSV 中，可用列: {list(df.columns)}")

        series = df[price_col].dropna()
        prices = series.values

        # 按真实起始价缩放期权要素，逻辑与 from_wind 一致。
        real_s0 = float(prices[0])
        opt, rescale_info = _rescale_option_to_real_s0(option, real_s0)
        try:
            print(_format_rescale_info(rescale_info))
        except Exception:
            pass

        if int(steps_per_day) != 1:
            print(f"[from_csv] CSV 日频数据仅支持 spd=1，steps_per_day={steps_per_day} 已置 1")
            steps_per_day = 1
        bt = cls(opt, prices, hedge_freq=hedge_freq, tc_rate=tc_rate, position=position,
                 quantity=quantity, multiplier=multiplier,
                 strategy=strategy, steps_per_day=steps_per_day,
                 slippage_bps=slippage_bps)
        used_len = len(bt.prices)
        used_index = series.index[:used_len]
        bt._wind_meta = {
            'code': filepath,
            'start_date': str(used_index[0].date()),
            'end_date': str(used_index[-1].date()),
            'dates': used_index,
            'n_trade_days': used_len - 1,
        }
        bt._rescale_info = rescale_info
        return bt


if __name__ == "__main__":
    # --- smoke test: 固定频率 vs x-sigma 带 ---
    try:
        from .Option_Vanilla import Option_Vanilla
    except ImportError:
        from Option_Vanilla import Option_Vanilla

    s0, sigma, T = 100.0, 0.20, 20
    opt = Option_Vanilla('European', s0, [], 100.0, T, sigma, 1, r=0.03, q=0.0)

    # intraday 路径：每日 4 个 bar，共 20 日 -> 80 bar
    spd = 4
    prices = HedgeBacktest.simulate_prices(
        s0, sigma, T, r=0.03, q=0.0, seed=7, steps_per_day=spd
    )

    bt_fixed = HedgeBacktest(
        opt, prices,
        hedge_freq=1, tc_rate=0.0005,
        position=1, quantity=1.0, multiplier=0,
        strategy=FixedFreqStrategy(hedge_freq=1),
        steps_per_day=spd, slippage_bps=2.0,
    )
    r_fixed = bt_fixed.run()
    n_trades_fixed = int(np.sum(r_fixed['hedge_triggered']))

    bt_band = HedgeBacktest(
        opt, prices,
        hedge_freq=1, tc_rate=0.0005,
        position=1, quantity=1.0, multiplier=0,
        strategy=SigmaBandStrategy(k=0.5, sigma_source='implied'),
        steps_per_day=spd, slippage_bps=2.0,
    )
    r_band = bt_band.run()
    n_trades_band = int(np.sum(r_band['hedge_triggered']))

    bt_band_rv = HedgeBacktest(
        opt, prices,
        hedge_freq=1, tc_rate=0.0005,
        position=1, quantity=1.0, multiplier=0,
        strategy=SigmaBandStrategy(k=0.5, sigma_source='realized', window_days=20),
        steps_per_day=spd, slippage_bps=2.0,
    )
    r_band_rv = bt_band_rv.run()
    n_trades_band_rv = int(np.sum(r_band_rv['hedge_triggered']))

    print("=" * 60)
    print(f"Smoke test  (T={T}d, spd={spd}, n_bars={len(prices) - 1})")
    print("=" * 60)
    print(f"{'strategy':<20} {'trades':>8} {'total_tc':>12} {'hedge_err':>14}")
    print(f"{'FixedFreq(freq=1)':<20} {n_trades_fixed:>8d} "
          f"{r_fixed['total_tc']:>12.4f} {r_fixed['hedging_error']:>14.4f}")
    print(f"{'SigmaBand(impl)':<20} {n_trades_band:>8d} "
          f"{r_band['total_tc']:>12.4f} {r_band['hedging_error']:>14.4f}")
    print(f"{'SigmaBand(real)':<20} {n_trades_band_rv:>8d} "
          f"{r_band_rv['total_tc']:>12.4f} {r_band_rv['hedging_error']:>14.4f}")
    print("=" * 60)

    # ---- 断言: realized σ 年化口径修正（spd=4） ----
    #
    # 若错误地用 sqrt(ANNUAL_DAYS) 年化 bar 级 std，σ 会被压低 sqrt(spd)=2 倍，
    # σ 带阈值同步压低，导致触发次数显著变多。
    # 正确口径下 realized σ ≈ implied σ，触发次数应与 implied 档接近。
    assert n_trades_band_rv <= n_trades_band * 2, (
        f"realized σ 触发次数过高 ({n_trades_band_rv})，疑似年化因子丢失 spd 分量"
    )
    # n_days 口径校验：intraday 下应等于交易日数 T，而非 bar 数 T*spd
    assert r_fixed['n_days'] == T, (
        f"n_days 应为交易日数 {T}，实际 {r_fixed['n_days']}（疑似混用 bar 数）"
    )
    assert r_fixed['n_bars'] == T * spd, (
        f"n_bars 应为 bar 数 {T*spd}，实际 {r_fixed['n_bars']}"
    )
    print(f"[assert] n_days={r_fixed['n_days']} n_bars={r_fixed['n_bars']}  OK")

    # ---- 断言: Fixed > SigmaBand，且 SigmaBand 两档在同量级 ----
    assert n_trades_fixed > n_trades_band, (
        f"Fixed({n_trades_fixed}) 应多于 SigmaBand-implied({n_trades_band})"
    )
    assert n_trades_fixed > n_trades_band_rv, (
        f"Fixed({n_trades_fixed}) 应多于 SigmaBand-realized({n_trades_band_rv})"
    )
    # 同量级：两档不应差异超过 5 倍
    lo = min(n_trades_band, n_trades_band_rv)
    hi = max(n_trades_band, n_trades_band_rv)
    assert lo > 0 and hi <= lo * 5, (
        f"SigmaBand implied({n_trades_band}) vs realized({n_trades_band_rv}) 差异过大"
    )
    print(f"[assert] trades  Fixed={n_trades_fixed}  Band(impl)={n_trades_band}  "
          f"Band(real)={n_trades_band_rv}  OK")

    # ---- 断言: T=1 日剩余近到期期权，intraday Δ 随 bar 递进应单调变化 ----
    #
    # 构造一支 T=1 日 ATM Call：spd=4 下同一日 4 根 bar，价格固定（排除 Δ 对 s0
    # 的依赖，只考察 T 衰减）。ATM Call Δ 随 t->0 朝 0.5 靠拢；r=q=0 的 ATM
    # 在 q=r 场景下 Δ ≈ exp(-rT)N(d1)，T 越小 N(d1)≈0.5+vol小量，Δ 单调下行。
    opt_near = Option_Vanilla('European', 100.0, [], 100.0, 1, 0.20, 1, r=0.0, q=0.0)
    # 日初（bar 0，elapsed=0）、日内第 2 / 第 3 bar
    d_bar0 = opt_near._bumped_copy(s0=100.0, _intraday_elapsed=0.0).get_greeks()[0]
    d_bar1 = opt_near._bumped_copy(s0=100.0, _intraday_elapsed=0.25).get_greeks()[0]
    d_bar2 = opt_near._bumped_copy(s0=100.0, _intraday_elapsed=0.75).get_greeks()[0]
    # r=q=0 的 ATM Call：Δ = N(d1)，d1 = 0.5 σ√t，t 减小 d1 趋于 0，Δ 向 0.5 收敛。
    # 初始 Δ > 0.5，t 变小后 Δ 应单调下降靠近 0.5。
    assert d_bar0 > d_bar1 > d_bar2, (
        f"intraday Δ 应随 elapsed 递增单调下降: {d_bar0:.6f} > {d_bar1:.6f} > {d_bar2:.6f}"
    )
    print(f"[assert] intraday Δ 递减  {d_bar0:.6f} -> {d_bar1:.6f} -> {d_bar2:.6f}  OK")
