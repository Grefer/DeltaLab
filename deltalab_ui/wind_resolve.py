# _*_ coding: utf-8 _*_
"""Wind 取数区间、交易日历倒推与行情粒度解析。

这一层全是纯函数：输入是 GUI 收集出来的 state dict，输出是补齐了起止日期与
实际 BarSize 的 state dict，中间不碰任何 tkinter 控件。抽出来的直接动机是它
们本来就是 ``BacktestApp`` 上的一堆 ``@staticmethod``——挂在窗口类上只是历史
原因，测试也一直绕开实例直接按类名调用。

调用方仍写 ``BacktestApp._resolve_wind_bar_size(...)``：类里按同名
staticmethod 别名暴露，与 ``history_selection`` 那组的做法一致。
"""

import bisect
import datetime

import numpy as np

import history_selection
from pricing import FixedTimeStrategy
from pricing.constants import ANNUAL_DAYS

from deltalab_ui.constants import (
    WIND_AUTO_BAR_SIZE,
    WIND_BAR_SIZE_OPTIONS,
    _WIND_BAND_BAR_LABEL,
    _WIND_BAR_MINUTES,
    _WIND_DATE_BUFFER_DAYS,
    _WIND_FIXED_TIME_BAR_LABELS,
)


# 进程内交易日历缓存：("ok", [date...]) | ("error", 原因)。
_LOCAL_TRADING_CALENDAR = None


def _local_trading_calendar():
    """返回本地交易日历（升序 ``datetime.date`` 列表）与解析状态。

    单次回测的建仓日必须落在真实交易日上，``calendar_span_for_trading_days``
    那套“自然日 + 节假日缓冲”只能用来估计取数区间，不能用来定位某一天。

    解析结果（含失败原因）在进程内只算一次：日期提示挂在 StringVar trace 上，
    每次击键都会调用，不能每次都去读文件，更不能反复触发联网刷新。
    """
    global _LOCAL_TRADING_CALENDAR
    if _LOCAL_TRADING_CALENDAR is None:
        try:
            from pricing.trade_calendar import load_calendar
            days = load_calendar().astype("datetime64[D]").astype(object)
            _LOCAL_TRADING_CALENDAR = ("ok", list(days))
        except Exception as exc:      # 缺日历文件且联网刷新也失败
            _LOCAL_TRADING_CALENDAR = (
                "error", str(exc) or exc.__class__.__name__)
    return _LOCAL_TRADING_CALENDAR


def parse_wind_date(value, label):
    """严格解析 GUI 的 Wind 日期字段，避免错误延迟到联网请求后。"""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}不能为空。")
    try:
        parsed = datetime.date.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}格式应为 YYYY-MM-DD，当前为 {text!r}。") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{label}格式应为 YYYY-MM-DD，当前为 {text!r}。")
    return parsed


def maturity_days_from_params(params):
    """读取各期权结构统一意义下的剩余交易日数。"""
    value = params.get("T_days")
    if value is None:
        value = params.get("T")
    try:
        days = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("无法从期权参数解析剩余期限。") from exc
    if days <= 0:
        raise ValueError("期权剩余期限必须大于 0 个交易日。")
    return days


def history_realized_warmup_days(state):
    """返回历史候选在 Day 0 前必须具备的 realized-HV 预热日数。

    界面已不再提供 σ 来源选项，``_collect_history_state`` 恒写入
    ``implied``，因此当前返回值恒为 0。保留这条判断是因为回测引擎仍
    支持 realized，而分析层用的是 ``strict_sigma_warmup=True``——一旦
    有 realized 状态从别处（旧配置、后端直调）流进来却没预热区间，会
    直接抛错而不是降级，那时静默返回 0 比现在更难查。
    """
    if not bool(state.get("history_include_band", False)):
        return 0
    if str(state.get("sigma_source", "implied")) != "realized":
        return 0
    try:
        days = int(state.get("sigma_window", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("历史候选 HV 回看期必须是正整数。") from exc
    if days < 2:
        raise ValueError("历史候选 HV 回看期必须至少为 2 日。")
    return days


def history_auto_wind_start(asof, *, required_trade_days):
    """自动模式下的历史行情起始日。

    界面回填与真正发出的 Wind 请求共用这一份推算。分成两处各算一套的
    话，置灰框里显示的日期和实际取数区间会各走各的，而那个框看上去正
    是"本次从哪天开始取"。
    """
    span = calendar_span_for_trading_days(required_trade_days)
    return asof - datetime.timedelta(days=span)


def validate_sigma_input(value, label):
    """波动率必须是大于 0 的有限数值。

    σ=0 不是"没有波动"这么无害：Black-Scholes 的 d1/d2 要除以
    ``σ√T``，σ=0 时直接除零，Greeks 全部退化成 NaN 或 0；模拟路径也塌
    成一条确定性曲线，多路径统计的每条路径完全相同。回测照样跑完、界面
    照样出数，只是那些数字没有任何意义——所以必须在启动前就拦掉，而不是
    让用户对着一屏 0.0000 猜哪里不对。
    """
    if value is None:
        return
    try:
        sigma = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是大于 0 的数值。") from exc
    if sigma == 0:
        raise ValueError(f"{label}不能为 0，必须是大于 0 的数值。")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"{label}必须是大于 0 的有限数值。")


def calendar_span_for_trading_days(trading_days):
    """把交易日需求换成带节假日缓冲的自然日拉取跨度。"""
    try:
        days = int(trading_days)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Wind 交易日需求必须是正整数。") from exc
    if days <= 0:
        raise ValueError("Wind 交易日需求必须是正整数。")
    return int(np.ceil(days * 365.0 / ANNUAL_DAYS)) + _WIND_DATE_BUFFER_DAYS


_CALENDAR_FIX_HINT = (
    "请更新 data/tradingday.csv，或取消勾选「按数据截止日倒推」"
    "后手工指定建仓日。")


def trading_calendar_days(*, required=True):
    """返回升序交易日列表；``required=False`` 时日历不可用只返回 None。"""
    status, payload = _local_trading_calendar()
    if status == "ok":
        return payload
    if required:
        raise ValueError(
            f"无法读取交易日历（{payload}），不能把数据截止日倒推成"
            f"建仓日。{_CALENDAR_FIX_HINT}")
    return None


def latest_trading_day(reference):
    """返回不晚于 ``reference`` 的最近交易日。

    只用于预填与提示：日历缺失或尚未覆盖到 ``reference`` 时原样返回，
    不假装知道更近的交易日，也不为一个默认值去联网。
    """
    days = trading_calendar_days(required=False)
    if not days or reference < days[0] or reference > days[-1]:
        return reference
    return days[bisect.bisect_right(days, reference) - 1]


def entry_date_from_asof(asof, maturity_days):
    """把数据截止日按交易日历倒推 ``maturity_days`` 个交易日。

    单次回测把行情首日当作建仓日、之后严格 T 个交易日到期（见
    ``HedgeBacktest`` 的评估窗口裁剪），因此这里不能沿用历史页那套
    “自然日跨度 + 节假日缓冲”的取数区间估计：多取的缓冲会整体前移
    建仓日，让期权在截止日之前就到期，界面上却仍写着截止日。

    返回 ``(建仓日, 落到交易日的截止日)``：截止日本身可能是周末或节假日，
    必须先落到不晚于它的最近交易日，倒推出来的两端才都是真实交易日。
    """
    days = trading_calendar_days()
    if asof < days[0] or asof > days[-1]:
        raise ValueError(
            f"交易日历未覆盖数据截止日 {asof.isoformat()}（可用区间 "
            f"{days[0].isoformat()} ~ {days[-1].isoformat()}）。"
            f"{_CALENDAR_FIX_HINT}")
    end_index = bisect.bisect_right(days, asof) - 1
    if end_index < maturity_days:
        raise ValueError(
            f"交易日历在 {days[end_index].isoformat()} 之前不足 "
            f"{maturity_days} 个交易日，无法倒推建仓日。"
            f"{_CALENDAR_FIX_HINT}")
    return days[end_index - maturity_days], days[end_index]


def local_trading_sessions(wind_code):
    """只读本地可确定规则的交易时段；无代码或元数据未知时返回 None。

    GUI 主线程的参数联动会频繁调用，因此固定 ``allow_wind=False``，
    绝不为了解析一个下拉值去启动 Wind 终端或发 ``wss``。
    """
    if not str(wind_code or "").strip():
        return None
    from pricing.wind_data import get_trading_session_clock_ranges
    return get_trading_session_clock_ranges(
        str(wind_code).strip(), allow_wind=False)


def recommended_fixed_time_bar_size(fixed_times, *, sessions=None):
    """返回能稳健覆盖目标 HH:MM 的推荐分钟粒度。

    Wind 的 bar 标签约定随粒度而变（实测 510050.SH）：1 分钟用左标签
    （09:30~14:59），5/15/60 分钟用右标签（09:35/09:45/10:30~15:00）。
    因此 session 开盘那一刻只有 1 分钟粒度取得到——整除判据只看墙钟，
    会把 09:30 误判成 15 分钟可覆盖。
    """
    strategy = FixedTimeStrategy(fixed_times)
    if sessions is not None:
        strategy.set_trading_sessions(sessions)
    parsed = strategy.effective_times
    opens = {start for start, _end in (strategy.trading_sessions or ())}
    if any(value in opens for value in parsed):
        return _WIND_FIXED_TIME_BAR_LABELS[-1]
    minute_marks = tuple(t.hour * 60 + t.minute for t in parsed)
    for label in _WIND_FIXED_TIME_BAR_LABELS:
        bar_minutes = _WIND_BAR_MINUTES[label]
        if all(mark % bar_minutes == 0 for mark in minute_marks):
            return label
    return _WIND_FIXED_TIME_BAR_LABELS[-1]


def recommended_band_bar_size():
    """价格波动触发一律用最细粒度观察穿带。

    不按带宽分档：带宽宽窄不改变“bar 内穿带后回落看不见”这一机制，
    只改变漏采的绝对数量，因此没有哪一档粗粒度是无条件安全的；分档
    还会让同一个候选的评分依赖“本批次里最窄的候选是谁”。
    """
    return _WIND_BAND_BAR_LABEL


def resolve_wind_bar_size(
        requested, *, strategy_name=None, fixed_times="",
        include_fixed_times=False, include_band=False, wind_code=None):
    """把“自动（推荐）”解析成实际 Wind 日频或分钟 BarSize。"""
    requested = str(requested or WIND_AUTO_BAR_SIZE).strip()
    if requested not in WIND_BAR_SIZE_OPTIONS:
        raise ValueError(f"未知 Wind 行情采样粒度: {requested!r}")

    needs_fixed = bool(
        include_fixed_times or strategy_name == "fixed_times")
    needs_band = bool(include_band or strategy_name == "hedge_band")
    # 交易时段只解析一次，供开盘瞬间拦截和固定时刻推荐共用。
    sessions = (
        local_trading_sessions(wind_code)
        if needs_fixed else None)
    if requested != WIND_AUTO_BAR_SIZE:
        if requested == "日频" and needs_fixed:
            raise ValueError(
                "固定时刻策略需要分钟行情，收到日频。粒度已改为按策略"
                "自动推导，显式传入日频的调用方需要改传分钟粒度或"
                f"{WIND_AUTO_BAR_SIZE!r}。")
        return requested

    recommendations = []
    if needs_band:
        recommendations.append(
            recommended_band_bar_size())
    if needs_fixed:
        recommendations.append(
            recommended_fixed_time_bar_size(
                fixed_times, sessions=sessions))
    if not recommendations:
        return "日频"
    return min(
        recommendations,
        key=lambda label: _WIND_BAR_MINUTES[label],
    )


def resolve_single_wind_state(state, *, today=None):
    """解析单次回测的数据截止日、倒推建仓日和实际行情粒度。

    日期主控是截止日（默认最新已收盘交易日），建仓日按期权期限往前倒推——与
    策略优选一致：那边也是从分析截至日向前回放严格区间。往后推的旧口径
    会把建仓日钉在过去某天、再拿期限去凑结束日，用户想问的“现在这套
    参数在最近一个完整期限上表现如何”反而要自己反算日期。
    """
    resolved = dict(state)
    if resolved.get("source") != "wind":
        return resolved
    if not str(resolved.get("wind_code", "")).strip():
        raise ValueError("Wind 代码不能为空。")

    today = today or datetime.date.today()
    if not isinstance(today, datetime.date):
        today = parse_wind_date(today, "当前日期")
    asof = parse_wind_date(
        resolved.get("wind_end"), "Wind 数据截止日")
    if asof > today:
        raise ValueError("Wind 数据截止日不能晚于当前日期。")

    maturity_days = maturity_days_from_params(
        resolved.get("params", {}))
    # 缺少开关的调用方（旧 API / 测试替身）自带完整两端日期，保持显式区间。
    auto_start = bool(resolved.get("wind_auto_start", False))
    if auto_start:
        start, asof = entry_date_from_asof(
            asof, maturity_days)
    else:
        start = parse_wind_date(
            resolved.get("wind_start"), "Wind 建仓日")
        if start > today:
            raise ValueError("Wind 建仓日不能晚于当前日期。")
    if asof <= start:
        raise ValueError("Wind 数据截止日必须晚于建仓日。")

    requested = resolved.get(
        "wind_bar_size_requested",
        resolved.get("wind_bar_size", WIND_AUTO_BAR_SIZE),
    )
    actual_bar_size = resolve_wind_bar_size(
        requested,
        strategy_name=resolved.get("strategy_name"),
        fixed_times=resolved.get("fixed_times", ""),
        wind_code=resolved.get("wind_code"),
    )
    resolved.update({
        "wind_start": start.isoformat(),
        "wind_end": asof.isoformat(),
        "wind_auto_start": auto_start,
        "wind_bar_size_requested": str(requested),
        "wind_bar_size": actual_bar_size,
        "wind_date_mode": (
            "auto_entry_from_asof" if auto_start else "custom_entry_date"),
        "wind_required_trade_days": maturity_days + 1,
    })
    return resolved


def resolve_history_wind_state(state, *, today=None):
    """解析历史择优独立的截至日、所选周期范围和统一行情粒度。"""
    resolved = dict(state)
    if resolved.get("source") != "wind":
        return resolved
    if not str(resolved.get("wind_code", "")).strip():
        raise ValueError("Wind 代码不能为空。")

    today = today or datetime.date.today()
    if not isinstance(today, datetime.date):
        today = parse_wind_date(today, "当前日期")
    asof = parse_wind_date(
        resolved.get("history_wind_asof", resolved.get("wind_end")),
        "历史分析截至日",
    )
    if asof > today:
        raise ValueError("历史分析截至日不能晚于当前日期。")

    warmup_days = history_realized_warmup_days(resolved)
    history_lookbacks = history_selection.normalize_lookbacks(
        resolved.get("history_lookbacks"))
    max_lookback_days = max(history_lookbacks.values())
    # L 与 V 是评分/估计所需交易日；额外 1 日只作为首日建仓的
    # Day 0 收盘锚点，不计入历史评分区间。
    required_trade_days = max_lookback_days + warmup_days + 1
    auto_start = bool(resolved.get("history_wind_auto_start", True))
    if auto_start:
        start = history_auto_wind_start(
            asof, required_trade_days=required_trade_days)
    else:
        start = parse_wind_date(
            resolved.get("history_wind_start", resolved.get("wind_start")),
            "历史行情起始日",
        )
    if start >= asof:
        raise ValueError("历史行情起始日必须早于分析截至日。")

    requested = resolved.get(
        "history_wind_bar_size_requested",
        resolved.get("wind_bar_size_requested",
                     resolved.get("wind_bar_size", WIND_AUTO_BAR_SIZE)),
    )
    actual_bar_size = resolve_wind_bar_size(
        requested,
        fixed_times=resolved.get("fixed_times", ""),
        include_fixed_times=resolved.get(
            "history_include_fixed_times", False),
        include_band=resolved.get("history_include_band", False),
        wind_code=resolved.get("wind_code"),
    )
    if not auto_start:
        date_mode = "history_custom_range"
    elif max_lookback_days == int(ANNUAL_DAYS):
        date_mode = "history_auto_year_strict_interval"
    else:
        date_mode = "history_auto_selected_strict_interval"
    resolved.update({
        "wind_start": start.isoformat(),
        "wind_end": asof.isoformat(),
        "wind_bar_size": actual_bar_size,
        "history_wind_asof": asof.isoformat(),
        "history_wind_start": start.isoformat(),
        "history_wind_auto_start": auto_start,
        "history_wind_bar_size_requested": str(requested),
        "wind_bar_size_requested": str(requested),
        "wind_date_mode": date_mode,
        "wind_required_trade_days": required_trade_days,
        "wind_sigma_warmup_days": warmup_days,
        "history_lookbacks": history_lookbacks,
        "history_max_lookback_days": max_lookback_days,
    })
    return resolved


def gui_steps_per_day(source, simulate_value):
    """GUI 仅允许模拟路径选择采样密度；真实行情返回自动占位值 1。"""
    if source != "simulate":
        return 1
    value = int(simulate_value or 1)
    if value <= 0:
        raise ValueError("模拟采样 bar/日必须大于 0。")
    return value


def normalize_position(value):
    """把所有 GUI / worker 方向入口收敛为唯一的 ``±1`` 表示。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("头寸方向只允许 1（卖出）或 -1（买入）。")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "头寸方向只允许 1（卖出）或 -1（买入）。") from exc
    if not np.isfinite(normalized) or normalized not in (-1.0, 1.0):
        raise ValueError("头寸方向只允许 1（卖出）或 -1（买入）。")
    return int(normalized)
