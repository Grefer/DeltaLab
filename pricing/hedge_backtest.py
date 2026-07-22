# _*_ coding: utf-8 _*_
"""
DeltaLab - 期权动态对冲回测框架

支持对任意继承自 OptionBase 的期权进行 Delta 对冲回测，
追踪每日盈亏分解、累计对冲误差和 Greeks 变动。
"""

import copy
import os
import datetime as _datetime
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
#   timestamp    : 当前 bar 时间（仅真实 DatetimeIndex 行情）
#   next_timestamp: 下一根 bar 时间；末 bar 为 None
#   is_day_close : 当前 bar 是否为真实交易日的最后一根 bar
#   sigma_impl   : 期权隐含波动率（option.sigma）
#   log_ret_hist : Day 0 前独立预热收益 + 截至当前 bar 的路径内对数收益
#   log_ret_warmup_bars: log_ret_hist 前缀中预热收益的根数


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


class CloseToCloseStrategy(HedgeStrategy):
    """每个交易日收盘对冲一次（close-to-close）。"""

    name = "close_to_close"

    def should_hedge(self, ctx):
        # 真实时间索引下以交易日边界为准；无时间戳的 GBM /
        # ndarray 路径仍保留 steps_per_day 取模的历史兼容逻辑。
        if ctx.get("timestamp") is not None:
            return bool(ctx.get("is_day_close", False))
        return bool(ctx.get("crosses_day", False))


class FixedTimeStrategy(HedgeStrategy):
    """在每天指定时刻对冲；每个时刻每天最多触发一次。

    ``trading_sessions`` 是可选的显式交易时段配置，格式为
    ``[(start, end), ...]``；起止值可以是 ``datetime.time`` 或 ``HH:MM``。
    只有落在交易时段内（含端点）的请求时刻才参与触发和逐日完整性校验。
    请求顺序会在去重后保留，便于按“夜盘 -> 次日日盘”的交易日业务顺序
    展示；实际触发仍严格跟随行情时间戳，不依赖元组排列。
    跨午夜时段用 ``start > end`` 表示，例如 ``("21:00", "02:30")``。

    不提供交易时段（``None``）时保留历史严格行为：所有请求时刻都必须
    在每个回测交易日组中存在。调用方在取得具体品种的 session 元数据后，
    也可以通过 :meth:`set_trading_sessions` 再配置。
    """

    name = "fixed_times"

    def __init__(self, times, trading_sessions=None):
        if isinstance(times, str):
            times = [x.strip() for x in times.split(",") if x.strip()]
        parsed = []
        for value in times or []:
            parsed.append(self._parse_time(value, label="固定对冲时刻"))
        if not parsed:
            raise ValueError("fixed_times 至少需要一个 HH:MM 时刻")
        # 不能按普通墙钟升序排序：对含夜盘品种，同一交易日的业务顺序是
        # 前一自然日夜盘 -> 次日日盘，例如 23:00 -> 11:30 -> 15:00。
        # dict.fromkeys 在去重的同时保留用户输入顺序。
        self.requested_times = tuple(dict.fromkeys(parsed))
        self.trading_sessions = None
        self.effective_times = self.requested_times
        self.skipped_times = ()
        # ``times`` 是历史公开属性。现在明确代表实际参与触发的有效时刻，
        # 让既有校验、深拷贝和策略调用方无需分叉处理。
        self.times = self.effective_times
        self._triggered = set()
        if trading_sessions is not None:
            self.set_trading_sessions(trading_sessions)

    @staticmethod
    def _parse_time(value, *, label):
        if isinstance(value, _datetime.time):
            return value.replace(
                second=0, microsecond=0, tzinfo=None)
        try:
            return _datetime.datetime.strptime(str(value), "%H:%M").time()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label}格式错误: {value!r}，应为 HH:MM") from exc

    @staticmethod
    def _time_in_session(value, start, end):
        """按分钟判断时刻是否在 session 内；起止端点都算交易时间。"""
        if start <= end:
            return start <= value <= end
        # 跨午夜，例如 21:00 -> 02:30。
        return value >= start or value <= end

    def set_trading_sessions(self, trading_sessions):
        """配置显式交易时段并反推有效/跳过时刻，返回 ``self``。

        ``None`` 会撤销时段过滤并恢复严格校验；显式空列表表示该品种没有
        可匹配的交易时段，因此全部请求时刻均自动跳过。
        """
        if trading_sessions is None:
            sessions = None
        else:
            sessions = []
            for session in trading_sessions:
                if (not isinstance(session, (tuple, list))
                        or len(session) != 2):
                    raise ValueError(
                        "固定时刻交易时段应为 (start, end) 二元组")
                start = self._parse_time(
                    session[0], label="交易时段起点")
                end = self._parse_time(
                    session[1], label="交易时段终点")
                sessions.append((start, end))
            sessions = tuple(sessions)

        self.trading_sessions = sessions
        if sessions is None:
            effective = self.requested_times
        else:
            effective = tuple(
                value for value in self.requested_times
                if any(self._time_in_session(value, start, end)
                       for start, end in sessions)
            )
        effective_set = set(effective)
        self.effective_times = effective
        self.skipped_times = tuple(
            value for value in self.requested_times
            if value not in effective_set
        )
        self.times = self.effective_times
        self._triggered.clear()
        return self

    def should_hedge(self, ctx):
        # 已由显式 session 证明所有请求都处于非交易时段时，策略是合法
        # no-op；此时甚至不需要用时间戳来判定触发。
        if not self.effective_times:
            return False
        timestamp = ctx.get("timestamp")
        if timestamp is None:
            raise ValueError("fixed_times 策略需要带 DatetimeIndex 的历史分钟行情")
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        if not isinstance(timestamp, _datetime.datetime):
            raise ValueError("fixed_times 策略需要真实 datetime 时间戳")
        current_time = timestamp.time().replace(
            second=0, microsecond=0, tzinfo=None)
        if current_time not in self.effective_times:
            return False
        key = (timestamp.date(), current_time)
        if key in self._triggered:
            return False
        self._triggered.add(key)
        return True


_EVENING_SESSION_START = _datetime.time(18, 0)
_DAY_SESSION_START = _datetime.time(6, 0)


def _trading_day_groups(timestamps):
    """把 DatetimeIndex 映射为连续的交易日组。

    日盘品种按 calendar date 分组。若出现 18:00 以后的夜盘，该夜盘会
    开启新交易日，并一直延续到后续日盘结束；因此 21:00 -> 次日
    02:30 -> 09:00 -> 15:00 属于同一组。这里依据观测到的 session
    顺序而不是简单的日历加一，因而也能正确跨越周末。
    """
    if len(timestamps) == 0:
        return np.array([], dtype=int)

    groups = np.zeros(len(timestamps), dtype=int)
    group = 0
    first = timestamps[0]
    first_time = first.time().replace(tzinfo=None)
    observed_times = {
        ts.time().replace(tzinfo=None) for ts in timestamps
    }
    # 日频 DatetimeIndex 通常全部落在 00:00（也可能统一为其它固定时刻）。
    # 只有观测到多个日内时刻时，<06:00 才能作为“夜盘跨午夜尾段”的证据。
    daily_like_single_time = len(observed_times) == 1

    def _is_night_session_time(value):
        return (
            value >= _EVENING_SESSION_START
            or (value < _DAY_SESSION_START and not daily_like_single_time)
        )

    # 查询可能从夜盘跨午夜后的尾段开始，未包含前一晚 21:00 opener。
    # 00:00~06:00 仍应视为夜盘延续，才能与周末/假期后的下一日盘
    # 合并为同一交易日组。
    has_evening = _is_night_session_time(first_time)
    day_session_seen = _DAY_SESSION_START <= first_time < _EVENING_SESSION_START
    previous = first

    for i in range(1, len(timestamps)):
        current = timestamps[i]
        current_time = current.time().replace(tzinfo=None)
        previous_time = previous.time().replace(tzinfo=None)
        date_changed = current.date() != previous.date()
        starts_new_group = False

        if current_time >= _EVENING_SESSION_START:
            # 日盘 -> 夜盘是下一个交易日的开始。夜盘内部多根
            # bar 仍属于同一组。
            starts_new_group = previous_time < _EVENING_SESSION_START or date_changed
        elif has_evening:
            # 夜盘已开始但日盘尚未出现时，允许跨午夜、周末/休市日
            # 继续到下一个日盘。日盘已出现后再跨日则开启新组。
            starts_new_group = date_changed and day_session_seen
        else:
            starts_new_group = date_changed

        if starts_new_group:
            group += 1
            has_evening = _is_night_session_time(current_time)
            day_session_seen = (
                _DAY_SESSION_START <= current_time < _EVENING_SESSION_START
            )
        else:
            if _is_night_session_time(current_time):
                has_evening = True
            elif current_time >= _DAY_SESSION_START:
                day_session_seen = True

        groups[i] = group
        previous = current

    return groups


def _trading_day_close_indices(timestamps):
    """返回每个连续交易日组在当前样本中的最后一根 bar 下标。"""
    if len(timestamps) == 0:
        return np.array([], dtype=int)
    groups = _trading_day_groups(timestamps)
    return np.flatnonzero(
        np.r_[groups[:-1] != groups[1:], True]
    ).astype(int, copy=False)


def _expiry_terminal_index(timestamps, term_days, steps_per_day):
    """验证真实日内组完整性并返回第 ``term_days`` 个收盘下标。

    连续分组本身只能说明“这是样本里最后一根”，不能证明最后一组真的
    已收盘。这里用声明/推导的典型 bar 数和完整组的常见收盘时刻共同
    判定；期限内出现盘中残段时明确拒绝，避免强制到期。
    """
    close_indices = _trading_day_close_indices(timestamps)
    bounds = []
    start = 0
    for end_value in close_indices:
        end = int(end_value)
        bounds.append((start, end))
        start = end + 1

    actionable = [(start, end) for start, end in bounds if end > 0]
    if len(actionable) < term_days:
        raise ValueError(
            f"价格序列交易日组不足：期权剩余 {term_days} 日，"
            f"Day 0 后仅观测到 {len(actionable)} 个交易日组。"
        )

    inferred_steps = _infer_intraday_steps(timestamps)
    expected_steps = max(1, int(steps_per_day), int(inferred_steps))

    # 只用 bar 数达到典型水平的组估计正常收盘时刻；盘中残段不会
    # 反过来污染预期。平票时取更晚时刻，对日盘/含夜盘数据更稳健。
    close_time_counts = {}
    for group_start, group_end in bounds:
        if group_end - group_start + 1 < expected_steps:
            continue
        close_time = timestamps[group_end].time().replace(
            second=0, microsecond=0, tzinfo=None)
        close_time_counts[close_time] = close_time_counts.get(close_time, 0) + 1
    expected_close = None
    if close_time_counts:
        max_frequency = max(close_time_counts.values())
        expected_close = max(
            time_value for time_value, frequency in close_time_counts.items()
            if frequency == max_frequency
        )

    for ordinal, (group_start, group_end) in enumerate(
            actionable[:term_days], start=1):
        count = group_end - group_start + 1
        actual_close = timestamps[group_end].time().replace(
            second=0, microsecond=0, tzinfo=None)

        # 缺少中间 bar 不代表没有收盘：只要能由典型组确认末时刻就是
        # 正常收盘，仍可用于 close-to-close。若无法判断正常收盘时刻，
        # 才退回到典型 bar 数作为完整性证据。
        count_complete = count >= expected_steps
        time_complete = (
            expected_close is not None and actual_close == expected_close
        )
        if expected_close is not None:
            complete = time_complete
        else:
            complete = count_complete

        if not complete:
            first_ts = timestamps[group_start]
            last_ts = timestamps[group_end]
            expected_text = (
                expected_close.strftime("%H:%M")
                if expected_close is not None else "无法从完整组判定"
            )
            raise ValueError(
                f"价格序列第{ordinal}个纳入期限的交易日组不完整："
                f"[{first_ts.strftime('%Y-%m-%d %H:%M')} ~ "
                f"{last_ts.strftime('%Y-%m-%d %H:%M')}] 仅 {count} 根 bar，"
                f"典型/声明为 {expected_steps} 根，末时刻 "
                f"{actual_close.strftime('%H:%M')}，预期收盘 {expected_text}。"
            )

    return int(actionable[term_days - 1][1])


def _infer_intraday_steps(timestamps):
    """由真实时间索引推导典型每交易日 bar 数；日频返回 1。"""
    if len(timestamps) < 2:
        return 1
    try:
        groups = _trading_day_groups(timestamps)
    except (AttributeError, TypeError, ValueError):
        # 非 datetime 索引无法证明日内 session；调用方可回退到
        # 交易所元数据或显式 steps_per_day。
        return 1
    counts = np.bincount(groups)
    complete = counts[counts > 1]
    if complete.size == 0:
        return 1
    # 首尾交易日常因查询边界而不完整；众数比 max 对偶发
    # 缺 bar 更稳健，平票时取较大值。
    values, frequencies = np.unique(complete, return_counts=True)
    max_frequency = frequencies.max()
    return int(values[frequencies == max_frequency].max())


def _has_repeated_day_close_evidence(timestamps, typical_steps):
    """判断真实索引是否足以推翻偏大的交易分钟元数据。

    单个盘中残段的 bar 数也可能形成“众数”，所以只有至少两个交易日组
    同时达到真实索引的典型 bar 数、并稳定结束在正常日盘收盘区间时，才
    把它视为完整 session 的直接证据。这样商品期货真实的 23 根 15min
    bar 可以优先于旧/粗粒度元数据，同时上午 11:30 截止的单个残段仍会
    继续使用元数据的保守下限并被完整性检查拒绝。
    """
    try:
        typical = int(typical_steps)
    except (TypeError, ValueError, OverflowError):
        return False
    if typical <= 1 or len(timestamps) < typical * 2:
        return False

    try:
        groups = _trading_day_groups(timestamps)
    except (AttributeError, TypeError, ValueError):
        return False

    close_times = []
    for group_id in np.unique(groups):
        positions = np.flatnonzero(groups == group_id)
        if len(positions) != typical:
            continue
        close_time = timestamps[int(positions[-1])].time().replace(
            second=0, microsecond=0, tzinfo=None)
        # 当前交易分钟元数据表覆盖的沪深证券和境内期货都在 15:00
        # 或更晚结束日盘；11:30/14:30 等时刻不能作为完整日证据。
        if _datetime.time(15, 0) <= close_time < _EVENING_SESSION_START:
            close_times.append(close_time)

    if len(close_times) < 2:
        return False
    _, frequencies = np.unique(np.asarray(close_times, dtype=object),
                               return_counts=True)
    return bool(frequencies.max() >= 2)


def _validate_fixed_time_data(strategy, timestamps):
    """在进入定价循环前验证 fixed_times 所需的时间粒度。"""
    # 显式 session 可能证明所有请求时刻均不属于该品种的交易时间。
    # 这是合法的自动跳过场景，不应再要求分钟索引或伪造缺 Bar 错误。
    if not strategy.effective_times:
        return
    if timestamps is None:
        raise ValueError(
            "fixed_times 策略需要真实 pandas.DatetimeIndex 的日内行情；"
            "RangeIndex、整数索引或无时间戳路径不受支持。"
        )
    if _infer_intraday_steps(timestamps) <= 1:
        raise ValueError(
            "fixed_times 策略仅支持真实日内行情；当前 DatetimeIndex "
            "每交易日只有一根 bar（日频）。"
        )

    close_indices = _trading_day_close_indices(timestamps)
    problems = []
    checked_group = 0
    start = 0
    requested_times = set(strategy.effective_times)
    for end in close_indices:
        end = int(end)
        # 只有下标 0 的单点组只是 Day 0 建仓观测，不会进入策略循环，
        # 不要求它补齐当天早先的固定时刻。其余组都实际纳入回测。
        if end > 0:
            checked_group += 1
            group_timestamps = timestamps[start:end + 1]
            available = {
                ts.time().replace(second=0, microsecond=0, tzinfo=None)
                for ts in group_timestamps
            }
            missing = requested_times.difference(available)
            if missing:
                first_ts = group_timestamps[0]
                last_ts = group_timestamps[-1]
                missing_text = ",".join(
                    t.strftime("%H:%M") for t in sorted(missing)
                )
                sample = ",".join(
                    t.strftime("%H:%M") for t in sorted(available)[:12]
                )
                problems.append(
                    f"第{checked_group}个交易日组 "
                    f"[{first_ts.strftime('%Y-%m-%d %H:%M')} ~ "
                    f"{last_ts.strftime('%Y-%m-%d %H:%M')}] "
                    f"缺失 [{missing_text}]（可用: [{sample}]）"
                )
        start = end + 1

    if problems:
        requested = ",".join(
            t.strftime("%H:%M") for t in strategy.effective_times)
        raise ValueError(
            f"fixed_times 目标时刻 [{requested}] 未逐交易日组完整匹配；"
            + "；".join(problems)
            + "。"
        )


class PriceIntervalStrategy(HedgeStrategy):
    """相对上次实际对冲价，按绝对价格或相对百分比间隔触发。"""

    name = "price_interval"

    def __init__(self, interval, interval_type="absolute"):
        self.interval = float(interval)
        self.interval_type = str(interval_type).lower()
        if self.interval <= 0:
            raise ValueError("价格间隔必须大于 0")
        if self.interval_type not in ("absolute", "relative"):
            raise ValueError("interval_type 仅支持 'absolute' 或 'relative'")

    def should_hedge(self, ctx):
        current = float(ctx["S"])
        last = float(ctx["S_last"])
        if self.interval_type == "absolute":
            return abs(current - last) >= self.interval
        if last <= 0:
            return False
        return abs(current / last - 1.0) >= self.interval


class SigmaBandStrategy(HedgeStrategy):
    """
    x-sigma 带触发：当 |ln(S/S_last)| >= k * sigma_ref * sqrt(dt_since_last)
    时重建 Δ 对冲。sigma_ref 有两种来源：
      - 'implied'  : 使用 option.sigma
      - 'realized' : 过去 window_days 个交易日对数收益的年化 std（intraday
                     下自动换算为 window_days * spd 根 bar）。普通单次回测
                     为兼容旧行为可在样本不足时回退到 option.sigma；历史
                     择优会启用严格预热，样本不足时明确失败。

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
        strict = bool(ctx.get("strict_realized_sigma", False))

        def _fallback_or_raise(reason):
            if strict:
                raise ValueError(f"realized sigma 严格预热不可用：{reason}")
            return float(ctx["sigma_impl"])

        lr = ctx.get("log_ret_hist")
        if lr is None:
            return _fallback_or_raise("缺少对数收益历史")
        lr = np.asarray(lr, dtype=float)
        if lr.ndim != 1 or not np.all(np.isfinite(lr)):
            return _fallback_or_raise("对数收益历史必须是一维有限数值")
        win_bars = self._window_bars(ctx)

        # 剔除"触发当根 bar"的 log return：该 bar 的收益是触发 σ 带的原因，
        # 若回灌到窗口里会系统性抬高 σ 估计，进一步抑制下一次触发（偏差）。
        # log_ret_hist[k] = ln(S[k+1]/S[k])，所以触发 bar i 对应的 return 索引是 i-1。
        i_trig = int(self._last_trigger_i)
        warmup_bars = int(ctx.get("log_ret_warmup_bars", 0) or 0)
        if warmup_bars < 0 or warmup_bars > len(lr):
            return _fallback_or_raise("预热收益长度元数据无效")
        mask = np.ones(len(lr), dtype=bool)
        # 路径内第 i_trig 根 bar 对应的 return 位于预热前缀之后，索引为
        # warmup_bars + i_trig - 1。不能再沿用未带预热时的 i_trig - 1，
        # 否则会错误删除一条 Day 0 前收益并保留真正的触发收益。
        trigger_return_index = warmup_bars + i_trig - 1
        if i_trig > 0 and 0 <= trigger_return_index < len(lr):
            mask[trigger_return_index] = False
        clean = lr[mask]
        if len(clean) < win_bars and strict:
            return _fallback_or_raise(
                f"需要 {win_bars} 根收益，实际仅 {len(clean)} 根")
        tail = clean[-win_bars:]
        if len(tail) < 2:
            return _fallback_or_raise("有效收益少于 2 根")
        std = float(np.std(tail, ddof=1))
        if not np.isfinite(std) or std <= 0:
            return _fallback_or_raise("已实现波动率非正或非有限")
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


class HedgeBandStrategy(HedgeStrategy):
    """同一价格带宽的三种等价表达：绝对值、相对值或日波动 σ 倍数。"""

    name = "hedge_band"

    def __init__(self, band_type="relative", threshold=None, k=0.5,
                 sigma_source="implied", window_days=20):
        self.band_type = str(band_type).lower()
        if self.band_type not in ("absolute", "relative", "sigma"):
            raise ValueError("band_type 仅支持 'absolute'、'relative' 或 'sigma'")
        # threshold 始终使用 band_type 对应的本单位。k 仅为旧调用兼容参数。
        if threshold is None:
            threshold = k if self.band_type == "sigma" else 0.01
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("带宽阈值必须是大于 0 的有限数值")
        self.sigma_source = str(sigma_source).lower()
        self.window_days = window_days
        # absolute / relative 触发只需要价格，不应因无关的 sigma
        # 配置缺失或非法而失败。仅 sigma 模式实例化波动率估计器。
        self._sigma_strategy = None
        if self.band_type == "sigma":
            self._sigma_strategy = SigmaBandStrategy(
                k=self.threshold,
                sigma_source=self.sigma_source,
                window_days=self.window_days,
            )

    @staticmethod
    def convert_threshold(value, from_type, reference_price, sigma_annual):
        """返回同一带宽的 absolute / relative / sigma 三种等价值。"""
        value = float(value)
        price = float(reference_price)
        sigma = float(sigma_annual)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("带宽值必须是大于 0 的有限数值")
        if (not np.isfinite(price) or not np.isfinite(sigma)
                or price <= 0 or sigma <= 0):
            raise ValueError(
                "reference_price 与 sigma_annual 必须是大于 0 的有限数值")
        daily_relative_sigma = sigma / np.sqrt(ANNUAL_DAYS)
        kind = str(from_type).lower()
        if kind == "absolute":
            absolute = value
        elif kind == "relative":
            absolute = value * price
        elif kind == "sigma":
            absolute = value * price * daily_relative_sigma
        else:
            raise ValueError("from_type 仅支持 'absolute'、'relative' 或 'sigma'")
        relative = absolute / price
        return {
            "absolute": absolute,
            "relative": relative,
            "sigma": relative / daily_relative_sigma,
        }

    def equivalent_thresholds(self, ctx):
        # 该方法显式要求返回 sigma 等价值，因而会计算 sigma；
        # should_hedge 的 absolute / relative 路径不会调用它。
        sigma_strategy = self._sigma_strategy
        if sigma_strategy is None:
            sigma_strategy = SigmaBandStrategy(
                k=self.threshold,
                sigma_source=self.sigma_source,
                window_days=self.window_days,
            )
        sigma = sigma_strategy._sigma_ref(ctx)
        return self.convert_threshold(
            self.threshold, self.band_type, ctx["S_last"], sigma)

    def should_hedge(self, ctx):
        current, last = float(ctx["S"]), float(ctx["S_last"])
        if last <= 0:
            return False
        if self.band_type == "absolute":
            absolute_band = self.threshold
        elif self.band_type == "relative":
            absolute_band = self.threshold * last
        else:
            sigma = self._sigma_strategy._sigma_ref(ctx)
            if not np.isfinite(sigma) or sigma <= 0:
                return False
            absolute_band = (
                self.threshold * last * sigma / np.sqrt(ANNUAL_DAYS)
            )
        triggered = abs(current - last) >= absolute_band
        if triggered and self.band_type == "sigma":
            self._sigma_strategy._last_trigger_i = int(ctx["i"])
        return triggered


def _realized_sigma_window_days(strategy):
    """返回策略实际参与触发的 realized-sigma 日窗口；否则返回 0。

    ``HedgeBandStrategy`` 的 absolute / relative 分支即使保存了
    ``sigma_source='realized'`` 也不会在触发时读取 sigma，因此不能令历史
    数据门槛无故增加。该判断同时供回测引擎和历史分析层使用。
    """
    sigma_strategy = None
    if isinstance(strategy, SigmaBandStrategy):
        sigma_strategy = strategy
    elif (isinstance(strategy, HedgeBandStrategy)
          and strategy.band_type == "sigma"):
        sigma_strategy = strategy._sigma_strategy
    if (sigma_strategy is None
            or sigma_strategy.sigma_source != "realized"):
        return 0
    return max(2, int(sigma_strategy.window_days))


def _realized_sigma_window_bars(strategy, steps_per_day):
    """返回策略严格预热所需 bar 数；非 realized-sigma 策略返回 0。"""
    if not _realized_sigma_window_days(strategy):
        return 0
    if isinstance(strategy, SigmaBandStrategy):
        sigma_strategy = strategy
    else:
        sigma_strategy = strategy._sigma_strategy
    return int(sigma_strategy._window_bars({
        "steps_per_day": max(1, int(steps_per_day)),
    }))


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


def _rescale_strategy_to_real_s0(strategy, ratio):
    """按标的价格重定基比例复制并缩放策略中的绝对价格阈值。

    absolute 带宽带有价格量纲，必须与期权要素一起乘以
    ``real_s0 / reference_s0``；relative / sigma 以及时间类策略不变。
    无论策略类型如何都返回独立副本，绝不修改调用方对象。
    """
    if strategy is None:
        return None
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("策略重定基 ratio 必须是大于 0 的有限数值")

    scaled = copy.deepcopy(strategy)
    if (isinstance(scaled, HedgeBandStrategy)
            and scaled.band_type == "absolute"):
        scaled.threshold *= ratio
    elif (isinstance(scaled, PriceIntervalStrategy)
          and scaled.interval_type == "absolute"):
        scaled.interval *= ratio
    return scaled


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
     strategy, steps_per_day, slippage_bps, force_day_close_hedge,
     path_idx, base_seed) = args

    price_path = np.asarray(price_path, dtype=float)
    if price_path.ndim != 1:
        price_path = price_path.reshape(-1)
    if price_path.size < 2:
        raise ValueError(f"path {path_idx}: price path length must be >= 2")
    if not np.all(np.isfinite(price_path)) or np.any(price_path <= 0):
        raise ValueError(f"path {path_idx}: price path must contain positive finite prices")

    npath = getattr(option_init, "nPath", None)
    if npath is not None and (int(npath) <= 0 or int(npath) % 2 != 0):
        raise ValueError(f"path {path_idx}: nPath must be a positive even integer, got {npath}")

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
        force_day_close_hedge=force_day_close_hedge,
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
        旧版固定 bar 频率，仅在 ``strategy=None`` 时生效；为后端 API
        兼容保留。新调用应显式传入 ``strategy``。
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
                 base_seed=20, force_day_close_hedge=False,
                 sigma_warmup_log_returns=None,
                 strict_sigma_warmup=False):
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
        force_day_close_hedge : bool
            公共的每日收盘兜底开关。开启后，非到期交易日的最后一根 bar
            至少执行一次 Delta 对齐；若当前策略已在同一 bar 触发，引擎
            自动去重。到期或敲出末 bar 始终直接平仓，不先做兜底调仓。
        sigma_warmup_log_returns : array-like | None
            Day 0 前、与 ``external_path`` 独立的对数收益预热种子。它只进入
            realized-sigma 的滚动估计，不会改变期权期限、价格路径、PnL
            起止点或回测时间戳。
        strict_sigma_warmup : bool
            要求 realized-sigma 策略在首个可交易 bar 前已具备完整窗口。
            历史择优使用 True；不足或波动率无效时明确失败，不回退 implied。
        base_seed : int | None
            run_multi 多路径 MC 采样的基础种子。每条路径实际使用的 option.mc_seed
            = base_seed + path_idx，实现路径间独立采样的同时保持整体可复现。
            传 None 时所有路径走 OS 熵（完全随机）。默认 20 与旧行为兼容。
        """
        self.option_init = copy.deepcopy(option)
        self._path_source = str(path_source).lower()
        self._external_path = external_path
        source_index = getattr(external_path, "index", None)
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
            source_index = getattr(prices, "index", None)
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
        self.force_day_close_hedge = bool(force_day_close_hedge)
        # strategy=None 时回退到旧的 FixedFreqStrategy(hedge_freq) 以保持兼容。
        self.strategy = strategy if strategy is not None else FixedFreqStrategy(self.hedge_freq)
        if sigma_warmup_log_returns is None:
            warmup_returns = np.array([], dtype=float)
        else:
            warmup_returns = np.asarray(
                sigma_warmup_log_returns, dtype=float)
            if warmup_returns.ndim != 1:
                raise ValueError("sigma_warmup_log_returns 必须是一维数组")
            if not np.all(np.isfinite(warmup_returns)):
                raise ValueError(
                    "sigma_warmup_log_returns 必须全部为有限数值")
            warmup_returns = warmup_returns.copy()
        self.sigma_warmup_log_returns = warmup_returns
        self.strict_sigma_warmup = bool(strict_sigma_warmup)
        required_warmup_bars = _realized_sigma_window_bars(
            self.strategy, self.steps_per_day)
        if (self.strict_sigma_warmup and required_warmup_bars
                and len(warmup_returns) < required_warmup_bars):
            raise ValueError(
                "realized sigma 严格预热不足："
                f"需要 Day 0 前 {required_warmup_bars} 根收益，"
                f"实际仅 {len(warmup_returns)} 根")
        self.timestamps = None
        if source_index is not None:
            try:
                import pandas as pd
                # 只接受数据源本身就是 DatetimeIndex 的情形。不再把
                # RangeIndex / 任意整数索引当作 Unix ns 强制转时间，
                # 否则 fixed_times 会在 1970-01-01 上静默运行。
                if isinstance(source_index, pd.DatetimeIndex):
                    idx = source_index.copy()
                    if len(idx) == len(self.prices):
                        if idx.hasnans:
                            raise ValueError("DatetimeIndex 包含 NaT，无法判定交易日边界")
                        if not idx.is_monotonic_increasing:
                            raise ValueError("DatetimeIndex 必须按时间升序排列")
                        if idx.has_duplicates:
                            raise ValueError("DatetimeIndex 包含重复时间戳")
                        self.timestamps = idx
            except ImportError:
                self.timestamps = None
        self.base_seed = base_seed
        self._results = None

        # ---- 价格序列长度校正：严格裁剪到期权剩余期限 ----
        #
        # 无真实时间戳时，口径仍为 Day 0 建仓、之后 T*spd 根 bar 到期。
        # 真实 DatetimeIndex 则不能假设每组恰好 spd 根：从 i>0 的第一个
        # 可完成交易日组收盘开始计数，严格截到第 T 个组收盘。这样首日从
        # 盘中开始、夜盘或偶发缺 bar 时，都不会把下一组 bar 带到到期后并
        # 制造伪 MtM 盈亏。
        try:
            t_rem = int(option._time_remaining)
        except Exception:
            t_rem = None
        fixed_time_validated = False
        if t_rem is not None and t_rem > 0:
            if self.timestamps is not None:
                # fixed_times 先在原始第 T 个组边界内逐组检查目标时刻，
                # 这样缺某根目标 bar 时能给出比“组 bar 数不足”更精确的错误。
                if isinstance(self.strategy, FixedTimeStrategy):
                    raw_closes = _trading_day_close_indices(self.timestamps)
                    raw_actionable = raw_closes[raw_closes > 0]
                    if len(raw_actionable) >= t_rem:
                        provisional_end = int(raw_actionable[t_rem - 1])
                        _validate_fixed_time_data(
                            self.strategy,
                            self.timestamps[:provisional_end + 1],
                        )
                        fixed_time_validated = True
                terminal_index = _expiry_terminal_index(
                    self.timestamps, t_rem, self.steps_per_day)
                self.prices = self.prices[:terminal_index + 1]
                self.timestamps = self.timestamps[:terminal_index + 1]
            else:
                need = t_rem * self.steps_per_day + 1
                have = len(self.prices)
                if have < need:
                    raise ValueError(
                        f"价格序列长度不足：期权剩余 {t_rem} 日 x "
                        f"steps_per_day={self.steps_per_day} 需要 {need} 个"
                        f"价格点，实际仅 {have} 个。"
                    )
                if have > need:
                    self.prices = self.prices[:need]

        # fixed_times 的数据问题在构造阶段直接报错，避免完成费时
        # Greeks / MC 计算后才发现全程没有任何可触发时刻。
        if (isinstance(self.strategy, FixedTimeStrategy)
                and not fixed_time_validated):
            _validate_fixed_time_data(self.strategy, self.timestamps)

    def run(self):
        """执行回测，返回结果字典"""
        option = copy.deepcopy(self.option_init)
        strategy = copy.deepcopy(self.strategy)
        S = self.prices
        n = len(S) - 1
        r = option.r
        spd = self.steps_per_day
        dt_bar = 1.0 / (ANNUAL_DAYS * spd)  # 单 bar 年化时长
        dt = 1.0 / ANNUAL_DAYS  # 日粒度（兼容旧字段）
        pos = self.position
        warmup_log_returns = self.sigma_warmup_log_returns
        path_log_returns = np.log(S[1:] / S[:-1])

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

        run_timestamps = (
            self.timestamps[:n + 1] if self.timestamps is not None else None
        )
        trading_day_groups = (
            _trading_day_groups(run_timestamps)
            if run_timestamps is not None else None
        )

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
        # 分开记录原策略与公共收盘兜底的触发来源。Day 0 建仓和末 bar
        # 平仓是引擎端点，不属于任一来源；fallback 数组仅标记“原策略未
        # 触发、由兜底补充”的 bar，因而同一收盘 bar 天然不会重复计数。
        strategy_hedge_triggered = np.zeros(n + 1, dtype=bool)
        day_close_fallback_triggered = np.zeros(n + 1, dtype=bool)

        sl_rate = self.slippage_bps * 1e-4  # bps -> ratio
        bars_since_day_close = 0

        # ---- Day 1 ~ n ----
        # Day 0 始终是建仓观测点，本身不消耗期限（即使它恰好
        # 是某个收盘 bar）。只有 i>=1 的真实交易日末 bar 才会
        # step_forward；无时间戳路径则继续每 spd 根推进一天。
        for i in range(1, n + 1):
            timestamp = run_timestamps[i] if run_timestamps is not None else None
            next_timestamp = (
                run_timestamps[i + 1]
                if run_timestamps is not None and i < n else None
            )
            if trading_day_groups is not None:
                # 真实行情按交易日组的末 bar 推进一天，而不是用
                # bar 序号取模。末 bar 没有 next_timestamp；若期权仍剩
                # 一天，将它视为回测到期观测并完成最后一次时间推进。
                is_day_close = (
                    i == n or
                    trading_day_groups[i] != trading_day_groups[i + 1]
                )
                crosses_day = bool(is_day_close)
            else:
                # 无时间戳 GBM / ndarray 路径保留历史行为。
                crosses_day = (i % spd == 0)
                is_day_close = crosses_day

            # 防止时间索引与期限参数不一致时将 option 推到负剩余
            # 期限。交易日边界 is_day_close 仍保留在策略 context 中。
            advances_option_day = crosses_day and option._time_remaining > 0
            bars_since_day_close += 1

            if advances_option_day:
                option.step_forward(S[i])
                eval_opt = option
                bars_since_day_close = 0
            else:
                # 日内：用临时副本评估 price / Δ，option 本体状态保持到当日收盘。
                # intraday_elapsed ∈ [1/spd, (spd-1)/spd]，
                # 让 Δ 在日内也按 bar 比例消耗 T，避免跨日 bar 单次跳变掉一天 Θ。
                if trading_day_groups is None:
                    elapsed = (i % spd) / spd
                else:
                    elapsed = min(
                        bars_since_day_close / spd,
                        max(0.0, (spd - 1) / spd),
                    )
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
                    "crosses_day": crosses_day,
                    "is_day_close": is_day_close,
                    "timestamp": timestamp,
                    "next_timestamp": next_timestamp,
                    "sigma_impl": float(option.sigma),
                    # 对数收益历史（到当前 bar 为止），仅 realized σ 时用到
                    "log_ret_hist": np.concatenate((
                        warmup_log_returns,
                        path_log_returns[:i],
                    )),
                    "log_ret_warmup_bars": len(warmup_log_returns),
                    "strict_realized_sigma": self.strict_sigma_warmup,
                }
                # 原策略无论兜底是否开启都只评估一次，确保固定时刻等
                # 有状态策略按原有逻辑推进。公共规则只对非终止收盘 bar
                # 做 OR 合并；原策略已触发时不会再产生第二笔交易。
                strategy_triggered = bool(strategy.should_hedge(ctx))
                fallback_only = bool(
                    self.force_day_close_hedge
                    and is_day_close
                    and not strategy_triggered
                )
                if strategy_triggered or fallback_only:
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
                    strategy_hedge_triggered[i] = strategy_triggered
                    day_close_fallback_triggered[i] = fallback_only
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

        # 真实时间索引按实际完成的交易日组计数；无时间戳路径才使用
        # 固定 spd 换算。Day 0 若单独成组（收盘基准点）不计入期限。
        if run_timestamps is not None:
            completed_days = int(np.count_nonzero(
                _trading_day_close_indices(run_timestamps) > 0
            ))
        else:
            completed_days = n // spd if spd > 0 else n

        normalization_schema = "s0_x_multiplier_x_abs_quantity_v1"
        normalization_s0 = float(S[0])
        normalization_notional = (
            normalization_s0 * self.multiplier * abs(self.quantity))
        normalization_reasons = []
        if not np.isfinite(normalization_s0) or normalization_s0 <= 0:
            normalization_reasons.append("S0 必须为有限正数")
        if not np.isfinite(self.multiplier) or self.multiplier <= 0:
            normalization_reasons.append("multiplier 必须为有限正数且不能为 0")
        if not np.isfinite(self.quantity) or self.quantity == 0:
            normalization_reasons.append("quantity 必须为有限非零数")
        if (not np.isfinite(normalization_notional)
                or normalization_notional <= 0):
            normalization_reasons.append("归一化分母必须为有限正数")
        normalization_available = not normalization_reasons
        normalization_reason = "；".join(normalization_reasons)

        if isinstance(strategy, FixedTimeStrategy):
            fixed_time_requested_times = tuple(
                value.strftime("%H:%M")
                for value in strategy.requested_times)
            fixed_time_effective_times = tuple(
                value.strftime("%H:%M")
                for value in strategy.effective_times)
            fixed_time_skipped_times = tuple(
                value.strftime("%H:%M")
                for value in strategy.skipped_times)
            fixed_time_trading_sessions = (
                None if strategy.trading_sessions is None else tuple(
                    (start.strftime("%H:%M"), end.strftime("%H:%M"))
                    for start, end in strategy.trading_sessions
                )
            )
        else:
            fixed_time_requested_times = None
            fixed_time_effective_times = None
            fixed_time_skipped_times = None
            fixed_time_trading_sessions = None

        self._results = {
            'n_days': completed_days,
            'n_bars': n,
            'steps_per_day': spd,
            'n_trade_days': completed_days,
            # 敲出提前了结标记：knocked_out=True 时回测在 ko_day（交易日）截断，
            # ko_settle 为敲出当日结算票息（普通期权恒为 False/None）。
            'knocked_out': ko_settle is not None,
            'ko_day': completed_days if ko_settle is not None else None,
            'ko_settle': ko_settle,
            'hedge_triggered': hedge_triggered,
            'strategy_hedge_triggered': strategy_hedge_triggered,
            'day_close_fallback_triggered': day_close_fallback_triggered,
            'force_day_close_hedge': self.force_day_close_hedge,
            'strategy_name': getattr(strategy, 'name', 'unknown'),
            # fixed_times 请求/过滤结果。非固定时刻策略为 None；空 tuple
            # 表示显式 session 已证明没有有效目标，而非元数据缺失。
            'fixed_time_requested_times': fixed_time_requested_times,
            'fixed_time_effective_times': fixed_time_effective_times,
            'fixed_time_skipped_times': fixed_time_skipped_times,
            'fixed_time_trading_sessions': fixed_time_trading_sessions,
            'timestamps': run_timestamps,
            # 与 timestamps 一一对齐的交易日组编号；夜盘、跨午夜
            # 与后续日盘保持在同一组，便于分析层按交易日聚合 PnL。
            'trading_day_groups': (
                trading_day_groups.copy()
                if trading_day_groups is not None else None
            ),
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
            # 跨合约路径比较的固定口径元数据。分母严格使用
            # S0 * multiplier * abs(quantity)，绝不以 1 或期货
            # contract_multiplier 代替无效输入；单次回测金额 PnL 不变。
            'quantity': self.quantity,
            'multiplier': self.multiplier,
            'normalization_schema': normalization_schema,
            'normalization_s0': normalization_s0,
            'normalization_notional': normalization_notional,
            'normalization_available': normalization_available,
            'normalization_reason': normalization_reason,
            'normalization_invalid_reason': normalization_reason,
            'sigma_warmup_log_returns': warmup_log_returns.copy(),
            'sigma_warmup_bars': int(len(warmup_log_returns)),
            'strict_sigma_warmup': self.strict_sigma_warmup,
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

        strategy = self.strategy
        if isinstance(strategy, FixedFreqStrategy):
            strategy_text = f"固定 bar 频率（每 {strategy.hedge_freq} bar）"
        elif isinstance(strategy, CloseToCloseStrategy):
            strategy_text = "每日收盘（close-to-close）"
        elif isinstance(strategy, FixedTimeStrategy):
            requested = ",".join(
                t.strftime("%H:%M") for t in strategy.requested_times)
            effective = ",".join(
                t.strftime("%H:%M") for t in strategy.effective_times)
            skipped = ",".join(
                t.strftime("%H:%M") for t in strategy.skipped_times)
            if not strategy.skipped_times:
                strategy_text = f"每日固定时刻（{requested}）"
            elif strategy.effective_times:
                strategy_text = (
                    f"每日固定时刻（请求 {requested}；有效 {effective}；"
                    f"自动跳过非交易时刻 {skipped}）")
            else:
                strategy_text = (
                    f"每日固定时刻（请求 {requested}；全部为非交易时刻，"
                    "自动跳过）")
        elif isinstance(strategy, HedgeBandStrategy):
            unit = {
                "absolute": "绝对价格",
                "relative": "相对价格",
                "sigma": "日波动 σ",
            }.get(strategy.band_type, strategy.band_type)
            strategy_text = f"固定间隔（{unit}={strategy.threshold:g}）"
        elif isinstance(strategy, PriceIntervalStrategy):
            strategy_text = (
                f"旧版价格间隔（{strategy.interval_type}={strategy.interval:g}）"
            )
        elif isinstance(strategy, SigmaBandStrategy):
            strategy_text = f"旧版 σ 带宽（k={strategy.k:g}）"
        else:
            strategy_text = getattr(strategy, "name", type(strategy).__name__)

        lines = [
            "=" * 52,
            "          动态对冲回测结果摘要",
            "=" * 52,
            f"  回测天数          :  {n:>10d}   (Day 0=建仓, Day {n}=到期)",
            f"  对冲策略          :  {strategy_text}",
            f"  每日收盘兜底      :  {'开启' if self.force_day_close_hedge else '关闭':>10s}",
            f"  兜底补充触发      :  {int(np.count_nonzero(r['day_close_fallback_triggered'])):>10d}",
            f"  实际采样 bar/日   :  {self.steps_per_day:>10d}",
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
            failed_paths   : list[int] — 失败路径索引；失败路径对应数值列为 NaN
            path_errors    : dict[int, str] — 失败路径错误摘要
        """
        n = len(paths)
        errors = np.full(n, np.nan)
        total_pnl = np.full(n, np.nan)
        total_tc = np.full(n, np.nan)
        final_prices = np.full(n, np.nan)
        realized_vols = np.full(n, np.nan)
        knocked_out = np.zeros(n, dtype=bool)
        ko_days = np.full(n, np.nan)
        implied_vol = self.option_init.sigma
        failed_paths = []
        path_errors = {}

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, n)

        # base_seed 解析：显式参数优先，否则用实例字段
        effective_seed = base_seed if base_seed is not None else self.base_seed

        task_args = [
            (self.option_init, paths[i], self.hedge_freq,
             self.tc_rate, self.position, self.quantity, self.multiplier,
             self.strategy, self.steps_per_day, self.slippage_bps,
             self.force_day_close_hedge,
             i, effective_seed)
            for i in range(n)
        ]

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_path, a): i
                       for i, a in enumerate(task_args)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                except Exception as exc:
                    failed_paths.append(idx)
                    path_errors[idx] = f"{type(exc).__name__}: {exc}"
                else:
                    errors[idx] = res['hedging_error']
                    total_pnl[idx] = res['final_pnl']
                    total_tc[idx] = res['total_tc']
                    final_prices[idx] = res['final_price']
                    realized_vols[idx] = res['realized_vol']
                    knocked_out[idx] = bool(res.get('knocked_out', False))
                    if res.get('ko_day') is not None:
                        ko_days[idx] = float(res['ko_day'])
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, n)

        return {
            'n_paths': n,
            'force_day_close_hedge': self.force_day_close_hedge,
            'errors': errors,
            'total_pnl': total_pnl,
            'total_tc': total_tc,
            'final_prices': final_prices,
            'implied_vol': implied_vol,
            'realized_vols': realized_vols,
            'knocked_out': knocked_out,
            'ko_days': ko_days,
            'failed_paths': sorted(failed_paths),
            'path_errors': path_errors,
        }

    def plot_error_dist(self, errors, figsize=(10, 5)):
        """绘制对冲误差分布直方图"""
        fig, ax = plt.subplots(figsize=figsize)
        errors = np.asarray(errors, dtype=float)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            ax.set_title("对冲误差分布 (无成功路径)", fontsize=12)
            ax.set_xlabel('对冲误差')
            ax.set_ylabel('频数')
            return fig
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
                  bar_size=None, force_day_close_hedge=False):
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
            旧版固定 bar 频率，仅在 ``strategy=None`` 时生效；为后端 API
            兼容保留。
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
            日频模式下强制为 1；intraday 模式下若未指定，优先按返回的
            DatetimeIndex 统计典型交易日组 bar 数，并用 Wind 品种的连续
            session 元数据交叉校验。未知品种不套用其它市场的硬编码下限；
            显式传入时以传入值为准并打印告警。

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
            scaled_strategy = _rescale_strategy_to_real_s0(
                strategy, rescale_info['ratio'])
            # Wind 日频数据无法承载任何盘中时刻。即使调用方此前给策略
            # 配过 session，也恢复严格目标，继续给出原有“仅支持日内行情”
            # 错误，避免把日频误解释成合法的固定时刻 no-op。
            if isinstance(scaled_strategy, FixedTimeStrategy):
                scaled_strategy.set_trading_sessions(None)
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

            bt = cls(opt, series, hedge_freq=hedge_freq, tc_rate=tc_rate,
                     position=position, quantity=quantity, multiplier=multiplier,
                     strategy=scaled_strategy, steps_per_day=spd_final,
                     slippage_bps=slippage_bps,
                     force_day_close_hedge=force_day_close_hedge)
            used_len = len(bt.prices)
            bt._wind_meta = {
                'code': code,
                'start_date': start_date,
                'end_date': end_date,
                'dates': series.index[:used_len],
                'n_trade_days': used_len - 1,
                'bar_size': None,
            }
            bt._full_price_history = series.copy()
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

        # 按每段连续 session 独立 ceil(bar_size) 得到元数据预期 bar 数；
        # 不能用 ceil(总分钟/bar_size)，否则 60min 会漏掉休市前的尾 bar。
        try:
            from .wind_data import (
                get_trading_bars_per_day,
                get_trading_session_clock_ranges,
            )
        except ImportError:
            from wind_data import (
                get_trading_bars_per_day,
                get_trading_session_clock_ranges,
            )

        metadata_spd = get_trading_bars_per_day(code, bar_min)
        index_spd = _infer_intraday_steps(series.index)
        if index_spd > 1:
            # 单个或多个同样的盘中残段可能把较小 bar 数自证成众数。
            # 默认仍取索引与元数据的较大值；只有多日都稳定走到正常日盘
            # 收盘时，才允许更强的真实索引证据覆盖偏大的粗粒度元数据。
            repeated_close_evidence = _has_repeated_day_close_evidence(
                series.index, index_spd)
            if metadata_spd is None:
                auto_spd = index_spd
                print(f"[from_wind] 警告：未命中 {code} 的连续交易时段元数据；"
                      f"采用实际时间索引 steps_per_day={index_spd}，不套用 "
                      "240 分钟硬编码下限。")
            elif metadata_spd > index_spd and repeated_close_evidence:
                # 元数据表按品种大类维护，可能落后于交易时段调整，或无法
                # 精确表达被休市段切开的 session。多日真实收盘证据更强时
                # 不再让偏大的元数据把完整交易日误判为盘中残段。
                auto_spd = index_spd
            else:
                auto_spd = max(index_spd, metadata_spd)
            if metadata_spd is not None and metadata_spd != index_spd:
                print(f"[from_wind] 实际时间索引推导 steps_per_day={index_spd}，"
                      f"与连续 session 元数据推导值 {metadata_spd} 不同；"
                      f"自动模式采用{'多日真实收盘证据' if repeated_close_evidence else '保守下限'}"
                      f" {auto_spd}。")
        else:
            auto_spd = metadata_spd if metadata_spd is not None else index_spd
            if metadata_spd is None:
                print(f"[from_wind] 警告：实际索引无法推导日内 bar 数，且无法从 "
                      f"Wind 识别 {code} 的连续交易时段；仅采用索引值 "
                      f"steps_per_day={auto_spd}。")

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
        scaled_strategy = _rescale_strategy_to_real_s0(
            strategy, rescale_info['ratio'])
        # 只有取得明确的品种交易时段时才过滤固定时刻。未知分类继续保留
        # trading_sessions=None 的严格逐日匹配，不能把元数据缺失误当休市。
        if isinstance(scaled_strategy, FixedTimeStrategy):
            session_profile = get_trading_session_clock_ranges(code)
            if session_profile is not None:
                scaled_strategy.set_trading_sessions(session_profile)
        try:
            print(_format_rescale_info(rescale_info))
        except Exception:
            pass

        bt = cls(opt, series, hedge_freq=hedge_freq, tc_rate=tc_rate,
                 position=position, quantity=quantity, multiplier=multiplier,
                 strategy=scaled_strategy, steps_per_day=spd_final,
                 slippage_bps=slippage_bps,
                 force_day_close_hedge=force_day_close_hedge)
        used_len = len(bt.prices)
        used_index = series.index[:used_len]
        if isinstance(bt.strategy, FixedTimeStrategy):
            fixed_time_wind_meta = {
                'fixed_time_requested_times': tuple(
                    value.strftime("%H:%M")
                    for value in bt.strategy.requested_times),
                'fixed_time_effective_times': tuple(
                    value.strftime("%H:%M")
                    for value in bt.strategy.effective_times),
                'fixed_time_skipped_times': tuple(
                    value.strftime("%H:%M")
                    for value in bt.strategy.skipped_times),
            }
        else:
            fixed_time_wind_meta = {}
        bt._wind_meta = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'dates': used_index,
            # intraday 下 n_trade_days 按 bar 数除以 spd 近似
            'n_trade_days': max(0, (used_len - 1) // max(1, spd_final)),
            'bar_size': bar_size_str,
            **fixed_time_wind_meta,
        }
        bt._full_price_history = series.copy()
        bt._rescale_info = rescale_info
        return bt

    @classmethod
    def from_csv(cls, option, filepath, price_col='close',
                 date_col=None, hedge_freq=1, tc_rate=0.0, position=1,
                 quantity=1.0, multiplier=5,
                 strategy=None, steps_per_day=None, slippage_bps=0.0,
                 force_day_close_hedge=False):
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
        if len(series) < 2:
            raise ValueError(f"CSV 价格数据不足：需要 >= 2 条，实际 {len(series)} 条")

        # CSV 同时支持日频和真实日内 DatetimeIndex。日内数据
        # 按交易日组（含夜盘）的典型 bar 数推导 spd；显式传入
        # 非 1 值时会与推导值核对，避免时间衰减与行情粒度错位。
        is_datetime_index = isinstance(series.index, pd.DatetimeIndex)
        inferred_spd = _infer_intraday_steps(series.index) if is_datetime_index else 1
        if inferred_spd > 1:
            if steps_per_day is None or int(steps_per_day) == 1:
                steps_per_day = inferred_spd
            else:
                requested_spd = max(1, int(steps_per_day))
                if requested_spd != inferred_spd:
                    raise ValueError(
                        f"CSV 日内行情推导 steps_per_day={inferred_spd}，"
                        f"与显式传入值 {requested_spd} 不一致。"
                    )
                steps_per_day = requested_spd
        else:
            requested_spd = 1 if steps_per_day is None else int(steps_per_day)
            if requested_spd != 1:
                print(
                    f"[from_csv] CSV 日频数据仅支持 spd=1，"
                    f"steps_per_day={requested_spd} 已置 1"
                )
            steps_per_day = 1

        # 按真实起始价缩放期权要素，逻辑与 from_wind 一致。
        real_s0 = float(prices[0])
        opt, rescale_info = _rescale_option_to_real_s0(option, real_s0)
        scaled_strategy = _rescale_strategy_to_real_s0(
            strategy, rescale_info['ratio'])
        try:
            print(_format_rescale_info(rescale_info))
        except Exception:
            pass

        bt = cls(opt, series, hedge_freq=hedge_freq, tc_rate=tc_rate, position=position,
                 quantity=quantity, multiplier=multiplier,
                 strategy=scaled_strategy, steps_per_day=steps_per_day,
                 slippage_bps=slippage_bps,
                 force_day_close_hedge=force_day_close_hedge)
        used_len = len(bt.prices)
        used_index = series.index[:used_len]
        bt._wind_meta = {
            'code': filepath,
            'start_date': str(used_index[0].date()),
            'end_date': str(used_index[-1].date()),
            'dates': used_index,
            'n_trade_days': max(0, (used_len - 1) // max(1, int(steps_per_day))),
        }
        bt._full_price_history = series.copy()
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
