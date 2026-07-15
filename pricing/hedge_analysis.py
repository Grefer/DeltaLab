# _*_ coding: utf-8 _*_
"""多对冲策略比较与历史窗口择优。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import ANNUAL_DAYS
from .hedge_backtest import (
    FixedTimeStrategy,
    HedgeBacktest,
    HedgeStrategy,
    _expiry_terminal_index,
    _infer_intraday_steps,
    _rescale_option_to_real_s0,
    _rescale_strategy_to_real_s0,
    _trading_day_close_indices,
    _trading_day_groups,
)


# 与整个定价项目使用同一套 243 交易日年化口径。月、季度均由年度交易日
# 推导，避免此处再维护一组互相矛盾的 21/63/252 魔法数字。
LOOKBACK_DAYS = {
    "week": 5,
    "month": int(round(ANNUAL_DAYS / 12)),
    "quarter": int(round(ANNUAL_DAYS / 4)),
    "year": int(ANNUAL_DAYS),
}


@dataclass(frozen=True)
class StrategyCase:
    """一个可被批量比较的命名策略。"""

    name: str
    strategy: HedgeStrategy
    metadata: dict = field(default_factory=dict)


def _positive_int(value, name):
    """把整数型配置规范化，并拒绝 bool、小数和非正数。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须为正整数，当前 {value!r}")
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须为正整数，当前 {value!r}") from exc
    if not np.isfinite(number) or number != result or result <= 0:
        raise ValueError(f"{name} 必须为正整数，当前 {value!r}")
    return result


def _max_drawdown(pnl):
    curve = np.cumsum(np.asarray(pnl, dtype=float))
    if not len(curve):
        return 0.0
    running_max = np.maximum.accumulate(np.r_[0.0, curve])[1:]
    return float(np.max(running_max - curve))


def _result_steps_per_day(results, explicit=None):
    """
    从回测结果自动读取并校验 ``steps_per_day``。

    旧结果可能没有这个字段；只有这种情况下才回退到显式参数（或 1）。一旦
    结果携带该字段，显式值与结果值、不同策略结果之间都必须一致。
    """
    explicit_spd = None if explicit is None else _positive_int(
        explicit, "steps_per_day"
    )
    found = {}
    for name, result in results.items():
        if "steps_per_day" in result and result["steps_per_day"] is not None:
            found[name] = _positive_int(
                result["steps_per_day"],
                f"results[{name!r}]['steps_per_day']",
            )

    unique = set(found.values())
    if len(unique) > 1:
        details = ", ".join(f"{name}={spd}" for name, spd in found.items())
        raise ValueError(f"策略结果的 steps_per_day 不一致: {details}")

    inferred = next(iter(unique)) if unique else None
    if explicit_spd is not None and inferred is not None and explicit_spd != inferred:
        raise ValueError(
            "显式 steps_per_day 与回测结果不一致: "
            f"explicit={explicit_spd}, result={inferred}"
        )
    return inferred or explicit_spd or 1


def _result_series(result, key):
    if key not in result:
        raise KeyError(f"回测结果缺少字段 {key!r}")
    values = np.asarray(result[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"回测结果 {key!r} 必须是一维数组")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"回测结果 {key!r} 含 NaN 或无穷值")
    return values


def _aggregate_result_by_day(result, steps_per_day):
    """
    把 bar 级 ``net_daily`` / ``tc_paid`` 聚合成真正的交易日序列。

    数组第 0 项是建仓点，其交易成本会计入首个损益交易日；第 1 项起才是
    相邻价格点之间的损益。若引擎提供 ``trading_day_groups``，优先按真实
    交易 session 分组，使夜盘、跨午夜与次日日盘属于同一交易日；否则按
    ``steps_per_day`` 个价格变动区间为一天。聚合前后净 PnL 和交易成本总额
    分别保持一致。
    """
    spd = _positive_int(steps_per_day, "steps_per_day")
    net = _result_series(result, "net_daily")
    tc = _result_series(result, "tc_paid")
    if len(net) != len(tc):
        raise ValueError(
            "回测结果 net_daily 与 tc_paid 长度不一致: "
            f"{len(net)} != {len(tc)}"
        )
    if len(net) == 0:
        return pd.DataFrame(columns=["net_pnl", "tc_paid"])

    # 没有价格变动区间时，仍保留建仓成本这一条观测。
    if len(net) == 1:
        return pd.DataFrame({"net_pnl": [net[0]], "tc_paid": [tc[0]]})

    n_intervals = len(net) - 1
    trading_groups = result.get("trading_day_groups")
    if trading_groups is not None:
        groups = np.asarray(trading_groups)
        if groups.ndim != 1 or len(groups) != len(net):
            raise ValueError(
                "回测结果 trading_day_groups 必须与 net_daily 等长的一维数组")
        day_no = groups[1:]
    else:
        day_no = np.arange(n_intervals, dtype=int) // spd
    frame = pd.DataFrame(
        {"net_pnl": net[1:], "tc_paid": tc[1:]}, index=day_no
    )
    daily = frame.groupby(level=0, sort=True).sum()

    # 建仓点不是独立损益日；把其成本/PnL 归入首个交易日，避免日频路径
    # T+1 个价格点被错误统计成 T+1 个交易日。
    daily.iloc[0, daily.columns.get_loc("net_pnl")] += net[0]
    daily.iloc[0, daily.columns.get_loc("tc_paid")] += tc[0]
    return daily


def _daily_metrics(daily):
    net = np.asarray(daily["net_pnl"], dtype=float)
    tc = np.asarray(daily["tc_paid"], dtype=float)
    if len(net):
        rms = float(np.sqrt(np.mean(np.square(net))))
        mean = float(np.mean(net))
        volatility = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
        avg_tc = float(np.mean(tc))
    else:
        rms = np.inf
        mean = 0.0
        volatility = 0.0
        avg_tc = np.inf
    return {
        "n_trade_days": int(len(net)),
        "daily_net_pnl_rms": rms,
        "mean_daily_pnl": mean,
        "pnl_volatility": volatility,
        "avg_daily_tc": avg_tc,
        "total_tc": float(np.sum(tc)) if len(tc) else 0.0,
        "total_net_pnl": float(np.sum(net)) if len(net) else 0.0,
        "max_drawdown": _max_drawdown(net),
        # net_pnl 已经扣除 tc_paid；用其相对 0 的 RMS 作为唯一 score，既衡量
        # 对冲波动，也让持续成本造成的负漂移只进入一次。
        "score": rms,
    }


def _rank_rows(rows):
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    # 完整窗口永远优先于不完整窗口；同一完整性内再按 score 和策略名稳定
    # 排序。这样 incomplete 结果仍可诊断，但不会挤掉正式推荐。
    ranking = ranking.sort_values(
        ["lookback_days", "complete_window", "score", "strategy"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = ranking.groupby("lookback", sort=False).cumcount() + 1
    return ranking


def compare_strategies(option, prices, cases, backtest_kwargs=None):
    """
    在完全相同的期权、行情和成本假设下运行多个策略。

    返回 ``(summary, results)``：summary 是按综合得分升序排列的 DataFrame，
    results 是 ``case.name -> HedgeBacktest.run()`` 的明细字典。

    所有 bar 级损益与成本会先按交易日聚合。``score`` 明确定义为每日净 PnL
    相对 0 的 RMS；由于净 PnL 已经扣除交易成本，不会再叠加一次成本。分数越
    低，表示对冲后的逐日偏离越小。
    """
    kwargs = dict(backtest_kwargs or {})
    rows, results = [], {}
    cases = list(cases)
    seen = set()
    for case in cases:
        if case.name in seen:
            raise ValueError(f"策略名称重复: {case.name}")
        seen.add(case.name)
        bt = HedgeBacktest(
            copy.deepcopy(option), prices,
            strategy=copy.deepcopy(case.strategy), **kwargs,
        )
        results[case.name] = bt.run()

    spd = _result_steps_per_day(results, kwargs.get("steps_per_day"))
    for case in cases:
        result = results[case.name]
        daily = _aggregate_result_by_day(result, spd)
        metrics = _daily_metrics(daily)
        rows.append({
            "strategy": case.name,
            "strategy_type": result["strategy_name"],
            "hedging_error": float(result["hedging_error"]),
            "trade_count": int(np.count_nonzero(result["hedge_triggered"])),
            **metrics,
            **{f"meta_{k}": v for k, v in case.metadata.items()},
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["score", "strategy"], kind="stable"
        ).reset_index(drop=True)
        summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary, results


def recommend_by_lookback(results, steps_per_day=None, lookbacks=None):
    """
    对已经完成的同路径回测做尾部历史区间诊断。

    这是兼容旧调用的轻量 API：它不会重跑期权，只能截取每个结果的尾部交易
    日。bar 级 ``net_daily`` / ``tc_paid`` 会先按交易日聚合；默认从每个结果
    的 ``steps_per_day`` 自动读取并校验一致。样本不足的行仍保留在 ranking
    供诊断，但 ``recommendations`` 只返回 ``complete_window=True`` 的冠军。

    若要在真实历史价格的多个可比期权窗口上重新回测，请使用
    :func:`recommend_by_rolling_history`。
    """
    windows = dict(LOOKBACK_DAYS if lookbacks is None else lookbacks)
    spd = _result_steps_per_day(results, steps_per_day)
    daily_results = {
        name: _aggregate_result_by_day(result, spd)
        for name, result in results.items()
    }

    rows = []
    for label, days_value in windows.items():
        days = _positive_int(days_value, f"lookbacks[{label!r}]")
        for name, daily_all in daily_results.items():
            used = min(days, len(daily_all))
            daily = daily_all.iloc[-used:] if used else daily_all.iloc[:0]
            metrics = _daily_metrics(daily)
            rows.append({
                "lookback": label,
                "lookback_days": days,
                "strategy": name,
                "days_used": used,
                # 兼容旧展示字段；它表示该日窗口在固定 spd 口径下对应的
                # 最大 bar 数，正式样本量请读取 days_used。
                "bars_used": used * spd,
                "complete_window": used >= days,
                "window_pnl": metrics["total_net_pnl"],
                "window_tc": metrics["total_tc"],
                "daily_net_pnl_rms": metrics["daily_net_pnl_rms"],
                "avg_daily_tc": metrics["avg_daily_tc"],
                "mean_daily_pnl": metrics["mean_daily_pnl"],
                "pnl_volatility": metrics["pnl_volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "score": metrics["score"],
            })

    ranking = _rank_rows(rows)
    if ranking.empty:
        return ranking.copy(), ranking
    recommendations = ranking[
        ranking["complete_window"] & (ranking["rank"] == 1)
    ].copy().reset_index(drop=True)
    return recommendations, ranking


def _history_series(prices):
    """保留 pandas 时间索引，同时严格校验历史价格。"""
    if isinstance(prices, pd.Series):
        history = prices.copy()
    else:
        values = np.asarray(prices, dtype=float)
        if values.ndim != 1:
            raise ValueError("prices 必须是一维历史价格序列")
        history = pd.Series(values)
    try:
        history = history.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices 含无法转换为数值的元素") from exc
    values = history.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("prices 至少需要两个价格点")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices 必须全部为有限正数")
    if isinstance(history.index, pd.DatetimeIndex):
        if history.index.hasnans:
            raise ValueError("prices 的 DatetimeIndex 含 NaT")
        if not history.index.is_monotonic_increasing:
            raise ValueError("prices 的 DatetimeIndex 必须单调递增")
        if history.index.has_duplicates:
            raise ValueError("prices 的 DatetimeIndex 不能重复")
    return history


def _history_trading_day_groups(history, steps_per_day):
    """返回每个历史价格点所属的连续交易日组。

    真实 ``DatetimeIndex`` 与回测引擎共享夜盘/跨午夜分组规则，因此缺 bar、
    临时多一根 bar 都不会把后续窗口整体错位。无时间戳数组无法识别真实
    session，只能保持兼容口径：第 0 点是建仓锚点，之后每 ``spd`` 根组成
    一个交易日。
    """
    if isinstance(history.index, pd.DatetimeIndex):
        return _trading_day_groups(history.index)

    spd = _positive_int(steps_per_day, "steps_per_day")
    groups = np.zeros(len(history), dtype=int)
    if len(history) > 1:
        groups[1:] = 1 + np.arange(len(history) - 1, dtype=int) // spd
    return groups


def _all_observed_groups_are_complete(timestamps, steps_per_day):
    """用回测引擎同一规则验证当前时间索引中的全部可参与交易日组。"""
    closes = _trading_day_close_indices(timestamps)
    term_days = int(np.count_nonzero(closes > 0))
    if term_days <= 0:
        return
    _expiry_terminal_index(timestamps, term_days, steps_per_day)


def _trim_incomplete_trailing_groups(history, steps_per_day):
    """剔除可明确识别的尾部盘中残组，返回 ``(history, errors)``。

    Wind/CSV 查询的结束时刻可能落在当天盘中。若完整前缀能通过引擎的
    收盘证据校验、而加入最后一组后失败，就把该尾组视为尚未完成的当前
    交易日，不让它挤占“最近一周/月”的一个历史终点。
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        return history, []

    trimmed = history
    errors = []
    while len(trimmed) >= 2:
        try:
            _all_observed_groups_are_complete(trimmed.index, steps_per_day)
            break
        except ValueError as full_error:
            closes = _trading_day_close_indices(trimmed.index)
            if len(closes) <= 1:
                break
            prefix = trimmed.iloc[:int(closes[-2]) + 1]
            try:
                _all_observed_groups_are_complete(prefix.index, steps_per_day)
            except ValueError:
                # 问题不只在尾组；交给下面端点级诊断，不静默删除中间历史。
                break
            errors.append(str(full_error))
            trimmed = prefix
    return trimmed, errors


def _rolling_endpoint_groups(n_groups, lookback_days, step_days):
    """列出最近观察期内的回测终点（始终包含区间首尾）。"""
    if n_groups <= 0:
        return []
    first = max(0, n_groups - lookback_days)
    latest = n_groups - 1
    endpoints = list(range(latest, first - 1, -step_days))
    if endpoints[-1] != first:
        endpoints.append(first)
    return sorted(endpoints)


def recommend_by_rolling_history(
    option,
    prices,
    cases,
    backtest_kwargs=None,
    *,
    lookbacks=None,
    step_days=5,
    steps_per_day=None,
):
    """
    在真实历史价格的多个可比窗口上重跑并推荐对冲策略。

    与 :func:`recommend_by_lookback` 只截取一次回测结果不同，本函数直接接收
    ``pd.Series``（保留 DatetimeIndex）或一维价格数组。它使用 ``option`` 的
    剩余期限作为每个可比回测窗口的固定存续期，在最近周/月/季/年的观察期
    内按 ``step_days`` 选择多个回测终点；每个终点向前取完整期权期限，所以
    策略起点可以早于观察期起点。全部 ``StrategyCase`` 运行完全相同的真实
    价格窗口，每轮会按真实起始价等比例伸缩期权的价格量纲要素。

    对带 ``DatetimeIndex`` 的日内行情，窗口边界按真实交易日组定位，而非
    ``位置 / steps_per_day``。起点使用前一交易日最后一根作为 Day 0 锚点，
    终点使用到期交易日最后一根，因此偶发缺/多 bar 不会错位，也不会把到期
    后 bar 混入窗口。

    参数 ``lookbacks`` 描述“最近多少交易日内的可用回测终点用于评估策略”，
    而不是临时改写期权期限。只有整个观察期内的抽样终点都具备足够的前置
    到期历史时才视为完整；否则仍给出已有窗口的诊断排名，但不正式推荐。

    Returns
    -------
    (recommendations, ranking, window_results)
        ``recommendations`` 仅含完整历史区间的冠军；``ranking`` 含完整与不完整
        结果；``window_results[lookback][window_id][strategy]`` 保存逐窗回测明细。
    """
    history = _history_series(prices)
    cases = list(cases)
    seen = set()
    for case in cases:
        if case.name in seen:
            raise ValueError(f"策略名称重复: {case.name}")
        seen.add(case.name)

    kwargs = dict(backtest_kwargs or {})
    kw_spd = kwargs.pop("steps_per_day", None)
    if steps_per_day is not None and kw_spd is not None:
        explicit_spd = _positive_int(steps_per_day, "steps_per_day")
        kwargs_spd = _positive_int(kw_spd, "backtest_kwargs['steps_per_day']")
        if explicit_spd != kwargs_spd:
            raise ValueError(
                "steps_per_day 与 backtest_kwargs['steps_per_day'] 不一致: "
                f"{explicit_spd} != {kwargs_spd}"
            )
    if steps_per_day is not None:
        spd_value = steps_per_day
    elif kw_spd is not None:
        spd_value = kw_spd
    elif isinstance(history.index, pd.DatetimeIndex):
        spd_value = _infer_intraday_steps(history.index)
    else:
        spd_value = 1
    spd = _positive_int(spd_value, "steps_per_day")
    step = _positive_int(step_days, "step_days")

    reserved = {"strategy", "path_source", "external_path", "prices"}
    conflict = sorted(reserved.intersection(kwargs))
    if conflict:
        raise ValueError(
            "backtest_kwargs 不应覆盖滚动历史接口管理的参数: "
            + ", ".join(conflict)
        )

    try:
        maturity_value = float(option._time_remaining)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("option 必须提供正整数交易日 _time_remaining") from exc
    maturity_days = _positive_int(maturity_value, "option._time_remaining")
    windows = dict(LOOKBACK_DAYS if lookbacks is None else lookbacks)
    history, trailing_group_errors = _trim_incomplete_trailing_groups(
        history, spd)
    history_groups = _history_trading_day_groups(history, spd)
    group_ids, group_first_positions = np.unique(
        history_groups, return_index=True
    )
    # _trading_day_groups 与兼容数组分组都保证组号连续且排序；仍显式取每组
    # 最后一根，避免窗口切片依赖“典型每交易日 bar 数”。
    group_last_positions = np.r_[
        group_first_positions[1:] - 1,
        len(history) - 1,
    ].astype(int)
    n_history_groups = len(group_ids)

    rows = []
    window_results = {}
    for label, days_value in windows.items():
        days = _positive_int(days_value, f"lookbacks[{label!r}]")
        endpoints = _rolling_endpoint_groups(n_history_groups, days, step)
        valid_endpoints = [end for end in endpoints if end >= maturity_days]
        # 完整观察期既需要足够的近期终点，也需要最早抽样终点之前仍有一段
        # 完整期权期限。后者正是旧实现令 T > lookback 时先天 0 窗口的问题：
        # 起点应允许落在 lookback 之前，而不是强迫整个生命周期塞进 tail。
        history_complete = (
            n_history_groups >= days + maturity_days
            and len(valid_endpoints) == len(endpoints)
        )

        label_results = {}
        collected = {case.name: [] for case in cases}
        case_failures = {case.name: 0 for case in cases}
        strategy_types = {}
        endpoint_failures = 0
        usable_endpoints = 0
        for window_no, end_group in enumerate(valid_endpoints, start=1):
            start_group = end_group - maturity_days
            # Day 0 只取起始组最后一根作为建仓锚点；随后包含恰好 T 个完整
            # 交易日，直至目标终点组最后一根。这样不会带入到期后的 bar。
            start_pos = int(group_last_positions[start_group])
            stop_pos = int(group_last_positions[end_group]) + 1
            path = history.iloc[start_pos:stop_pos]
            # 只有真实 DatetimeIndex 才交给 HedgeBacktest 保留时间戳。RangeIndex
            # 若直接传入会被 pandas 解释成 1970 年的纳秒时间戳，进而把整段
            # 数据错误聚合到同一个自然日。
            external_path = (
                path if isinstance(history.index, pd.DatetimeIndex)
                else path.to_numpy(dtype=float)
            )
            window_id = f"window_{window_no}"
            per_case = {}
            if isinstance(path.index, pd.DatetimeIndex):
                try:
                    _expiry_terminal_index(
                        path.index, maturity_days, spd)
                except ValueError as exc:
                    # 一个盘中残段不应让其它完整历史窗全部丢失。保留错误
                    # 诊断并跳过该公共端点；它不会计入任何策略的完整性。
                    endpoint_failures += 1
                    label_results[window_id] = {
                        "_window_error": str(exc),
                    }
                    continue
            usable_endpoints += 1
            for case in cases:
                scaled_option, rescale_info = _rescale_option_to_real_s0(
                    option, float(path.iloc[0])
                )
                scaled_strategy = _rescale_strategy_to_real_s0(
                    case.strategy, rescale_info["ratio"]
                )
                try:
                    result = HedgeBacktest(
                        scaled_option,
                        path_source="historical",
                        external_path=external_path,
                        strategy=scaled_strategy,
                        steps_per_day=spd,
                        **kwargs,
                    ).run()
                except ValueError as exc:
                    # 固定时刻可能只在个别窗口缺少目标 bar；该策略保留
                    # incomplete 诊断，但不能阻断其它策略的历史推荐。
                    if not isinstance(case.strategy, FixedTimeStrategy):
                        raise
                    case_failures[case.name] += 1
                    strategy_types[case.name] = getattr(
                        case.strategy, "name", "fixed_times")
                    per_case[case.name] = {
                        "error": str(exc),
                        "strategy_name": strategy_types[case.name],
                    }
                    continue
                if _positive_int(
                    result.get("steps_per_day"),
                    f"{label}/{window_id}/{case.name} steps_per_day",
                ) != spd:
                    raise ValueError("滚动回测返回了不一致的 steps_per_day")
                daily = _aggregate_result_by_day(result, spd)
                collected[case.name].append(daily)
                strategy_types[case.name] = result.get("strategy_name", "unknown")
                per_case[case.name] = result
            label_results[window_id] = per_case
        window_results[label] = label_results

        for case in cases:
            daily_windows = collected[case.name]
            if daily_windows:
                combined = pd.concat(daily_windows, ignore_index=True)
                metrics = _daily_metrics(combined)
                worst_drawdown = max(_max_drawdown(x["net_pnl"]) for x in daily_windows)
            else:
                combined = pd.DataFrame(columns=["net_pnl", "tc_paid"])
                metrics = _daily_metrics(combined)
                worst_drawdown = 0.0
            skipped = endpoint_failures + case_failures[case.name]
            complete = (
                history_complete
                and skipped == 0
                and usable_endpoints > 0
                and len(daily_windows) == usable_endpoints
            )
            rows.append({
                "lookback": label,
                "lookback_days": days,
                "strategy": case.name,
                "strategy_type": strategy_types.get(case.name, "unknown"),
                "evaluation_mode": "rolling_history",
                "maturity_days": maturity_days,
                "step_days": step,
                "history_days_available": min(days, n_history_groups),
                "eligible_endpoints": len(valid_endpoints),
                "rolling_windows": len(daily_windows),
                "skipped_endpoints": skipped,
                "trailing_partial_groups_dropped": len(trailing_group_errors),
                "days_used": metrics["n_trade_days"],
                "complete_window": complete,
                "window_pnl": metrics["total_net_pnl"],
                "window_tc": metrics["total_tc"],
                "daily_net_pnl_rms": metrics["daily_net_pnl_rms"],
                "avg_daily_tc": metrics["avg_daily_tc"],
                "mean_daily_pnl": metrics["mean_daily_pnl"],
                "pnl_volatility": metrics["pnl_volatility"],
                # 各窗口彼此是独立试验，不把窗口边界首尾相连；披露其中最坏
                # 的单窗口回撤。
                "max_drawdown": worst_drawdown,
                "score": metrics["score"],
                **{f"meta_{k}": v for k, v in case.metadata.items()},
            })

    ranking = _rank_rows(rows)
    if ranking.empty:
        return ranking.copy(), ranking, window_results
    recommendations = ranking[
        ranking["complete_window"] & (ranking["rank"] == 1)
    ].copy().reset_index(drop=True)
    return recommendations, ranking, window_results
