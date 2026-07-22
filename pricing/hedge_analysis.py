# _*_ coding: utf-8 _*_
"""多对冲策略比较与历史窗口择优。"""

from __future__ import annotations

import copy
from collections.abc import Mapping
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
    _realized_sigma_window_days,
    _rescale_option_to_real_s0,
    _rescale_strategy_to_real_s0,
    _trading_day_close_indices,
    _trading_day_groups,
)


# 与整个定价项目使用同一套 243 交易日年化口径。月、季度、半年均由年度交易日
# 推导，避免此处再维护一组互相矛盾的 21/63/252 魔法数字。
LOOKBACK_DAYS = {
    "week": 5,
    "month": int(round(ANNUAL_DAYS / 12)),
    "quarter": int(round(ANNUAL_DAYS / 4)),
    "half_year": int(round(ANNUAL_DAYS / 2)),
    "year": int(ANNUAL_DAYS),
}

# 历史择优按“终点样本预算”均匀覆盖各观察期。预算随证据周期增长，既让
# 短周期保留全部近期交易日，也让长周期拥有更充分但仍受控的历史样本。
# 这是推荐给 GUI/调用方显式传入 ``target_endpoints`` 的默认映射；函数层
# 仍保留 ``None``=fixed-step、整数=全部周期统一预算的 Python API 兼容。
HISTORY_TARGET_ENDPOINTS = {
    "week": 5,
    "month": 12,
    "quarter": 24,
    "half_year": 36,
    "year": 48,
}
HISTORY_SELECTION_METRIC = "mean_bounded_window_advantage_vs_c2c"


@dataclass(frozen=True)
class StrategyCase:
    """一个可被批量比较的命名策略。"""

    name: str
    strategy: HedgeStrategy
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContractHistoryPool:
    """同一期货品种按历史主力日期组织的具体合约行情池。"""

    product_code: str
    main_contract_by_date: pd.Series
    contract_prices: Mapping[str, pd.Series]
    main_contract_asof: str
    contract_load_errors: Mapping[str, str] = field(default_factory=dict)


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


def result_daily_frame(result, steps_per_day=None):
    """把单次回测结果转换为可直接展示的日级净 PnL 曲线数据。

    Parameters
    ----------
    result : dict
        :meth:`HedgeBacktest.run` 返回的单策略结果。
    steps_per_day : int, optional
        仅供旧结果缺少 ``steps_per_day`` 时回退使用；若结果自身已有该字段，
        显式值必须与其一致。

    Returns
    -------
    pandas.DataFrame
        索引沿用真实交易 session 分组（无时间戳旧结果则为顺序日组），列为
        ``net_pnl``、``tc_paid`` 和 ``cumulative_net_pnl``。最后一列等于前两列
        中 ``net_pnl`` 的逐日累加，可直接供 GUI 绘制累计净 PnL 曲线。
    """
    spd = _result_steps_per_day({"result": result}, steps_per_day)
    daily = _aggregate_result_by_day(result, spd).copy()
    daily["cumulative_net_pnl"] = daily["net_pnl"].cumsum()
    daily.index.name = "trade_day"
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


_HISTORY_WINDOW_CURVE_COLUMNS = (
    "daily_net_pnl",
    "daily_gross_pnl",
    "daily_tc",
    "cumulative_net_pnl",
    "cumulative_gross_pnl",
    "cumulative_tc",
)

_HISTORY_WINDOW_NORMALIZED_CURVE_COLUMNS = (
    "normalized_daily_gross_pnl",
    "normalized_daily_net_pnl",
    "normalized_daily_tc",
    "normalized_cumulative_gross_pnl",
    "normalized_cumulative_net_pnl",
    "normalized_cumulative_tc",
)

_HISTORY_WINDOW_SUMMARY_COLUMNS = (
    "lookback",
    "window_id",
    "history_contract_code",
    "history_endpoint_date",
    "strategy",
    "success",
    "strategy_type",
    "start_ts",
    "end_ts",
    "days_used",
    "score",
    "total_net_pnl",
    "total_gross_pnl",
    "total_tc",
    "max_drawdown",
    "normalization_schema",
    "normalization_quantity",
    "normalization_multiplier",
    "normalization_s0",
    "normalization_notional",
    "normalization_available",
    "normalization_reason",
    *_HISTORY_WINDOW_CURVE_COLUMNS,
    *_HISTORY_WINDOW_NORMALIZED_CURVE_COLUMNS,
    "failure_scope",
    "failure_reason",
)


def _history_result_time_bounds(result):
    """返回单窗结果的首尾真实时间；旧数组结果没有时间轴时返回 NaT。"""
    timestamps = result.get("timestamps")
    if timestamps is None:
        return pd.NaT, pd.NaT
    try:
        index = pd.DatetimeIndex(timestamps)
    except (TypeError, ValueError) as exc:
        raise ValueError("历史窗口结果 timestamps 无法解析为时间索引") from exc
    if index.empty:
        return pd.NaT, pd.NaT
    valid = index[~index.isna()]
    if valid.empty:
        return pd.NaT, pd.NaT
    return pd.Timestamp(valid[0]), pd.Timestamp(valid[-1])


def _history_failure_summary_row(
        lookback, window_id, strategy, strategy_type, scope, reason,
        start_ts=pd.NaT, end_ts=pd.NaT, history_contract_code="",
        history_endpoint_date=pd.NaT):
    """构造保留统一曲线列的失败行，每个空数组也都独立持有。"""
    return {
        "lookback": lookback,
        "window_id": window_id,
        "history_contract_code": str(history_contract_code or ""),
        "history_endpoint_date": history_endpoint_date,
        "strategy": strategy,
        "success": False,
        "strategy_type": strategy_type,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "days_used": 0,
        "score": np.nan,
        "total_net_pnl": np.nan,
        "total_gross_pnl": np.nan,
        "total_tc": np.nan,
        "max_drawdown": np.nan,
        "normalization_schema": "",
        "normalization_quantity": np.nan,
        "normalization_multiplier": np.nan,
        "normalization_s0": np.nan,
        "normalization_notional": np.nan,
        "normalization_available": False,
        "normalization_reason": "回测失败，无法归一化",
        **{
            column: np.array([], dtype=float)
            for column in (
                *_HISTORY_WINDOW_CURVE_COLUMNS,
                *_HISTORY_WINDOW_NORMALIZED_CURVE_COLUMNS,
            )
        },
        "failure_scope": scope,
        "failure_reason": str(reason),
    }


def _history_normalization_metadata(result):
    """校验回测快照中的跨合约归一化分母，失败时保持 fail closed。"""
    required = (
        "normalization_schema",
        "quantity",
        "multiplier",
        "normalization_s0",
        "normalization_notional",
        "normalization_available",
    )
    missing = [key for key in required if key not in result]

    def _number(key):
        try:
            return float(result.get(key, np.nan))
        except (TypeError, ValueError, OverflowError):
            return np.nan

    schema = str(result.get("normalization_schema", "") or "")
    quantity = _number("quantity")
    multiplier = _number("multiplier")
    s0 = _number("normalization_s0")
    notional = _number("normalization_notional")
    reason = str(
        result.get("normalization_reason")
        or result.get("normalization_invalid_reason")
        or "")
    available = bool(result.get("normalization_available", False))

    validation_errors = []
    if missing:
        validation_errors.append(
            "回测结果缺少归一化元数据: " + ", ".join(missing))
    if not schema:
        validation_errors.append("归一化 schema 为空")
    if not np.isfinite(s0) or s0 <= 0:
        validation_errors.append("S0 必须为有限正数")
    if not np.isfinite(multiplier) or multiplier <= 0:
        validation_errors.append("multiplier 必须为有限正数且不能为 0")
    if not np.isfinite(quantity) or quantity == 0:
        validation_errors.append("quantity 必须为有限非零数")
    expected = s0 * multiplier * abs(quantity)
    if not np.isfinite(notional) or notional <= 0:
        validation_errors.append("归一化分母必须为有限正数")
    elif (not np.isfinite(expected)
          or not np.isclose(notional, expected, rtol=1e-12, atol=0.0)):
        validation_errors.append(
            "归一化分母不等于 S0 * multiplier * abs(quantity)")
    if not available and not reason:
        validation_errors.append("回测结果标记归一化不可用")
    if validation_errors:
        available = False
        reason = "；".join(filter(None, (reason, *validation_errors)))

    return {
        "normalization_schema": schema,
        "normalization_quantity": quantity,
        "normalization_multiplier": multiplier,
        "normalization_s0": s0,
        "normalization_notional": notional,
        "normalization_available": available,
        "normalization_reason": reason,
    }


def _history_pair_normalization_metadata(result, baseline_result):
    """要求候选与同一样本 C2C 使用完全相同的归一化分母。"""
    metadata = _history_normalization_metadata(result)
    if baseline_result is None:
        metadata["normalization_available"] = False
        metadata["normalization_reason"] = (
            metadata["normalization_reason"]
            or "同一历史样本缺少唯一 close-to-close 基准，无法校验分母")
        return metadata
    baseline = _history_normalization_metadata(baseline_result)
    if not metadata["normalization_available"]:
        return metadata
    if not baseline["normalization_available"]:
        metadata["normalization_available"] = False
        metadata["normalization_reason"] = (
            "同一历史样本的 close-to-close 基准归一化不可用："
            + baseline["normalization_reason"])
        return metadata
    same_schema = (
        metadata["normalization_schema"]
        == baseline["normalization_schema"])
    same_notional = np.isclose(
        metadata["normalization_notional"],
        baseline["normalization_notional"],
        rtol=1e-12,
        atol=0.0,
    )
    if not (same_schema and same_notional):
        metadata["normalization_available"] = False
        metadata["normalization_reason"] = (
            "同一历史样本的候选与 close-to-close 基准归一化分母不一致")
    return metadata


def history_window_summary(window_results):
    """把滚动历史原始结果整理成逐窗口、逐策略的日级曲线摘要。

    ``window_results`` 使用 :func:`recommend_by_rolling_history` 的第三项返回
    值结构。成功结果先经 :func:`result_daily_frame` 按真实交易 session 聚合，
    再复用 :func:`_daily_metrics` 计算与历史排名一致的评分。毛 PnL 定义为
    ``净 PnL + 交易成本``，因此三组累计曲线可以直接用于同一窗口内的策略
    与 close-to-close 配对展示。

    策略级错误保留策略名并标记 ``failure_scope='strategy'``；公共
    ``_window_error`` 以策略为空的独立行保留，并标记为 ``'endpoint'``。
    函数不会修改输入映射或其中的数组。
    """
    if not isinstance(window_results, Mapping):
        raise TypeError("window_results 必须是按观察期组织的映射")

    rows = []
    for lookback, lookback_results in window_results.items():
        if not isinstance(lookback_results, Mapping):
            raise TypeError(
                f"window_results[{lookback!r}] 必须是窗口映射")
        for window_id, window in lookback_results.items():
            if not isinstance(window, Mapping):
                raise TypeError(
                    f"window_results[{lookback!r}][{window_id!r}] "
                    "必须是策略结果映射")

            # 固定时刻等单策略失败时，仍可从同窗其它成功策略取得时间边界，
            # 便于 GUI 使用 (lookback, window_id) 做基准配对与诊断定位。
            window_start, window_end = pd.NaT, pd.NaT
            for key, candidate in window.items():
                if (str(key).startswith("_window_")
                        or not isinstance(candidate, Mapping)
                        or "error" in candidate):
                    continue
                window_start, window_end = _history_result_time_bounds(candidate)
                if not (pd.isna(window_start) and pd.isna(window_end)):
                    break

            baseline_results = [
                candidate
                for key, candidate in window.items()
                if (not str(key).startswith("_window_")
                    and isinstance(candidate, Mapping)
                    and "error" not in candidate
                    and candidate.get("strategy_name") == "close_to_close")
            ]
            baseline_result = (
                baseline_results[0] if len(baseline_results) == 1 else None)

            if "_window_error" in window:
                error_meta = window.get("_window_error_meta", {})
                if not isinstance(error_meta, Mapping):
                    error_meta = {}
                rows.append(_history_failure_summary_row(
                    lookback,
                    window_id,
                    None,
                    None,
                    "endpoint",
                    window["_window_error"],
                    window_start,
                    window_end,
                    error_meta.get("history_contract_code", ""),
                    error_meta.get("history_endpoint_date", pd.NaT),
                ))

            for strategy, result in window.items():
                if str(strategy).startswith("_window_"):
                    continue
                if not isinstance(result, Mapping):
                    raise TypeError(
                        f"{lookback!r}/{window_id!r}/{strategy!r} "
                        "必须是回测结果映射")
                if "error" in result:
                    rows.append(_history_failure_summary_row(
                        lookback,
                        window_id,
                        strategy,
                        result.get("strategy_name"),
                        "strategy",
                        result["error"],
                        window_start,
                        window_end,
                        result.get("history_contract_code", ""),
                        result.get("history_endpoint_date", pd.NaT),
                    ))
                    continue

                daily = result_daily_frame(result)
                metrics = _daily_metrics(daily)
                daily_net = daily["net_pnl"].to_numpy(
                    dtype=float, copy=True)
                daily_tc = daily["tc_paid"].to_numpy(
                    dtype=float, copy=True)
                daily_gross = np.add(daily_net, daily_tc)
                start_ts, end_ts = _history_result_time_bounds(result)
                normalization = _history_pair_normalization_metadata(
                    result, baseline_result)
                if normalization["normalization_available"]:
                    denominator = normalization["normalization_notional"]
                    normalized_daily_net = daily_net / denominator
                    normalized_daily_gross = daily_gross / denominator
                    normalized_daily_tc = daily_tc / denominator
                else:
                    normalized_daily_net = np.array([], dtype=float)
                    normalized_daily_gross = np.array([], dtype=float)
                    normalized_daily_tc = np.array([], dtype=float)
                rows.append({
                    "lookback": lookback,
                    "window_id": window_id,
                    "history_contract_code": str(
                        result.get("history_contract_code", "") or ""),
                    "history_endpoint_date": result.get(
                        "history_endpoint_date", pd.NaT),
                    "strategy": strategy,
                    "success": True,
                    "strategy_type": result.get("strategy_name", "unknown"),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "days_used": metrics["n_trade_days"],
                    "score": metrics["score"],
                    "total_net_pnl": metrics["total_net_pnl"],
                    "total_gross_pnl": float(np.sum(daily_gross)),
                    "total_tc": metrics["total_tc"],
                    "max_drawdown": metrics["max_drawdown"],
                    **normalization,
                    "daily_net_pnl": daily_net,
                    "daily_gross_pnl": daily_gross,
                    "daily_tc": daily_tc,
                    "cumulative_net_pnl": np.cumsum(daily_net),
                    "cumulative_gross_pnl": np.cumsum(daily_gross),
                    "cumulative_tc": np.cumsum(daily_tc),
                    "normalized_daily_gross_pnl": normalized_daily_gross,
                    "normalized_daily_net_pnl": normalized_daily_net,
                    "normalized_daily_tc": normalized_daily_tc,
                    "normalized_cumulative_gross_pnl": np.cumsum(
                        normalized_daily_gross),
                    "normalized_cumulative_net_pnl": np.cumsum(
                        normalized_daily_net),
                    "normalized_cumulative_tc": np.cumsum(
                        normalized_daily_tc),
                    "failure_scope": "",
                    "failure_reason": "",
                })

    frame = pd.DataFrame(rows, columns=_HISTORY_WINDOW_SUMMARY_COLUMNS)
    # 显式固定为 object 列，防止空表或等长数组被 pandas 误推断成其它 dtype。
    for column in (
            *_HISTORY_WINDOW_CURVE_COLUMNS,
            *_HISTORY_WINDOW_NORMALIZED_CURVE_COLUMNS):
        frame[column] = frame[column].astype(object)
    return frame


def _rank_rows(rows):
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    # 完整窗口永远优先于不完整窗口；同一完整性内先按 score，再按总成本、
    # 策略名稳定排序。这样分数相同的推荐可复现，也自然偏向成本更低的策略。
    ranking = ranking.sort_values(
        [
            "lookback_days", "complete_window", "score", "window_tc",
            "strategy",
        ],
        ascending=[True, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = ranking.groupby("lookback", sort=False).cumcount() + 1
    return ranking


def _history_baseline_case(cases):
    """返回历史择优中唯一的 close-to-close 基准案例。"""
    baselines = [
        case for case in cases
        if getattr(case.strategy, "name", "") == "close_to_close"
    ]
    if not baselines:
        raise ValueError(
            "历史择优必须包含一个 close_to_close 策略作为固定基准")
    if len(baselines) > 1:
        names = ", ".join(case.name for case in baselines)
        raise ValueError(
            "历史择优只能包含一个 close_to_close 固定基准，"
            f"当前为: {names}")
    return baselines[0]


def _history_improvement_vs_c2c(score, baseline_score):
    """返回日净 PnL RMS 相对 C2C 的改善；正值表示候选更优。"""
    try:
        candidate = float(score)
        baseline = float(baseline_score)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    if (not np.isfinite(candidate) or not np.isfinite(baseline)
            or np.isclose(baseline, 0.0, rtol=1e-12, atol=1e-12)):
        return np.nan
    return float((baseline - candidate) / abs(baseline))


def _history_selection_advantage_vs_c2c(score, baseline_score):
    """返回有界逐窗 C2C 优势，供历史终点等权排名。

    RMS 均为非负数，使用 ``(baseline - candidate) / max(baseline,
    candidate)`` 可把结果限定在 ``[-1, 1]``，并对候选/基准对称。双方都为
    0 时明确视为持平；因此 C2C 完美对冲而候选仍有波动的窗口不会被 NaN
    静默排除。
    """
    try:
        candidate = float(score)
        baseline = float(baseline_score)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    if (not np.isfinite(candidate) or not np.isfinite(baseline)
            or candidate < 0.0 or baseline < 0.0):
        return np.nan
    scale = max(candidate, baseline)
    if np.isclose(scale, 0.0, rtol=1e-12, atol=1e-12):
        return 0.0
    return float((baseline - candidate) / scale)


def _rank_history_rows(rows):
    """按窗口等权的同窗 C2C 改善为滚动历史结果排序。"""
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    # 正式可比 > 同窗诊断 > 部分配对 > 无配对。正式/诊断组内，每个历史
    # 终点先独立换算为有界的相对 C2C RMS 优势，再以逐窗均值为主排序。这样
    # 不同价格量级的具体合约、以及因提前结算而长度不同的窗口都各占一票，
    # 不会由名义金额较大的窗口支配结论。覆盖率和窗口胜率只使用窗口级
    # 相对证据；完全持平时固定优先 C2C，避免为了没有改善的候选增加复杂度。
    # 合并金额 RMS 继续保留作诊断，但不参与名次。旧结果缺少新字段时仍
    # 回退到合并 RMS 改善，以便读取本次升级前的内存结果。
    recommendation = ranking["recommendation_eligible"].fillna(False).astype(bool)
    comparable = ranking["comparison_eligible"].fillna(False).astype(bool)
    paired = pd.to_numeric(
        ranking["paired_windows"], errors="coerce").fillna(0).gt(0)
    ranking["_comparison_tier"] = np.select(
        [recommendation, comparable, paired], [0, 1, 2], default=3)
    legacy_values = ranking.get(
        "improvement_vs_c2c",
        pd.Series(np.nan, index=ranking.index, dtype=float),
    )
    selection_values = ranking.get(
        "selection_improvement_vs_c2c",
        pd.Series(np.nan, index=ranking.index, dtype=float),
    )
    selection_metrics = ranking.get(
        "selection_metric",
        pd.Series("", index=ranking.index, dtype=object),
    ).fillna("").astype(str)
    selection_values = selection_values.where(
        selection_metrics.eq(HISTORY_SELECTION_METRIC))
    # 混合新旧内存行时逐行回退；不能因某一行带新列，就把其它旧行的
    # 有效改善统一变成 -inf。
    selection_values = selection_values.where(
        pd.notna(selection_values), legacy_values)
    ranking["_selection_sort"] = pd.to_numeric(
        selection_values, errors="coerce").fillna(-np.inf)
    win_rate_values = ranking.get(
        "window_win_rate_vs_c2c",
        pd.Series(np.nan, index=ranking.index, dtype=float),
    )
    ranking["_win_rate_sort"] = pd.to_numeric(
        win_rate_values, errors="coerce").fillna(-np.inf)
    coverage_values = ranking.get(
        "comparison_coverage",
        pd.Series(0.0, index=ranking.index, dtype=float),
    )
    ranking["_coverage_sort"] = pd.to_numeric(
        coverage_values, errors="coerce").fillna(0.0)
    ranking["_baseline_tiebreak"] = (
        ~ranking["strategy_type"].astype(str).eq("close_to_close")
    ).astype(int)
    ranking = ranking.sort_values(
        [
            "lookback_days", "_comparison_tier", "_coverage_sort",
            "_selection_sort", "_baseline_tiebreak", "_win_rate_sort",
            "strategy",
        ],
        ascending=[True, True, False, False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking = ranking.drop(columns=[
        "_comparison_tier", "_selection_sort", "_win_rate_sort",
        "_coverage_sort", "_baseline_tiebreak",
    ])
    ranking["rank"] = ranking.groupby("lookback", sort=False).cumcount() + 1
    return ranking


def _strategy_activity_metrics(result):
    """区分策略触发、再调仓与真实成交，保留旧触发计数口径。"""
    if "hedge_triggered" not in result:
        raise KeyError("回测结果缺少字段 'hedge_triggered'")
    triggered = np.asarray(result["hedge_triggered"])
    if triggered.ndim != 1:
        raise ValueError("回测结果 'hedge_triggered' 必须是一维数组")
    triggered = triggered.astype(bool)

    fallback = result.get("day_close_fallback_triggered")
    if fallback is None:
        # 旧版已保存结果没有来源拆分字段，按未启用兜底兼容。
        fallback = np.zeros(len(triggered), dtype=bool)
    else:
        fallback = np.asarray(fallback)
        if fallback.ndim != 1:
            raise ValueError(
                "回测结果 'day_close_fallback_triggered' 必须是一维数组")
        fallback = fallback.astype(bool)
        if len(fallback) != len(triggered):
            raise ValueError(
                "回测结果 day_close_fallback_triggered 与 hedge_triggered "
                f"长度必须一致: {len(fallback)}, {len(triggered)}"
            )

    shares = _result_series(result, "shares")
    prices = _result_series(result, "prices")
    if len(triggered) != len(shares) or len(shares) != len(prices):
        raise ValueError(
            "回测结果 hedge_triggered、shares 与 prices 长度必须一致: "
            f"{len(triggered)}, {len(shares)}, {len(prices)}"
        )

    trades = np.empty_like(shares)
    if len(shares):
        trades[0] = shares[0]
        trades[1:] = np.diff(shares)
    actual_trade_mask = np.abs(trades) > 1e-10
    effective_trades = np.where(actual_trade_mask, trades, 0.0)

    return {
        # 兼容旧展示：包括 Day 0 建仓触发与到期平仓触发，即使目标持仓未变。
        "trade_count": int(np.count_nonzero(triggered)),
        # 新口径：只统计策略在存续期内的触发，不含首尾两个引擎端点。
        "rehedge_count": int(np.count_nonzero(triggered[1:-1])),
        # 公共收盘规则补充的触发数；已由原策略在同一 bar 触发的不计入。
        "day_close_fallback_count": int(np.count_nonzero(fallback[1:-1])),
        # 真实成交以持仓确实发生变化为准，包括首仓和到期平仓。
        "actual_trade_count": int(np.count_nonzero(actual_trade_mask)),
        "turnover": float(np.sum(np.abs(effective_trades) * prices)),
    }


def summarize_strategy_result(
        result, display_name, metadata=None, steps_per_day=None):
    """把单次回测结果汇总为与策略对比排名兼容的一行指标。

    该函数不运行回测、也不修改 ``result`` 或 ``metadata``。日级 RMS、成本、
    净 PnL 与回撤使用 :func:`result_daily_frame` 相同的交易 session 聚合口径；
    触发与成交指标则直接从单次结果的持仓轨迹计算。

    Parameters
    ----------
    result : dict
        :meth:`HedgeBacktest.run` 返回的单策略结果。
    display_name : str
        排名表中展示的策略名，可包含参数说明。
    metadata : mapping, optional
        展示元数据。为避免覆盖正式指标，输出键统一添加 ``meta_`` 前缀。
    steps_per_day : int, optional
        仅供旧结果缺少 ``steps_per_day`` 时回退使用；若结果已有该字段，显式值
        必须与其一致。

    Returns
    -------
    dict
        可直接用于构造 :func:`compare_strategies` summary 的指标行；不含只有
        多策略排序后才能确定的 ``rank``。
    """
    spd = _result_steps_per_day({"result": result}, steps_per_day)
    daily = _aggregate_result_by_day(result, spd)
    metadata_copy = (
        {} if metadata is None else copy.deepcopy(dict(metadata))
    )
    return {
        "strategy": display_name,
        "strategy_type": result["strategy_name"],
        "hedging_error": float(result["hedging_error"]),
        **_strategy_activity_metrics(result),
        **_daily_metrics(daily),
        **{f"meta_{key}": value for key, value in metadata_copy.items()},
    }


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
        rows.append(summarize_strategy_result(
            results[case.name], case.name, case.metadata, spd,
        ))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["score", "total_tc", "strategy"], kind="stable"
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


def _target_count_endpoint_groups(
        n_groups, lookback_days, target_endpoints):
    """在最近观察期内均匀选择至多 ``target_endpoints`` 个终点。

    只要区间至少有两个交易日，首尾都会纳入。整数终点使用四舍五入后的
    等距位置；目标数不超过可用交易日数，因此不会产生重复终点。
    """
    if n_groups <= 0:
        return []
    available = min(n_groups, lookback_days)
    first = n_groups - available
    latest = n_groups - 1
    count = min(target_endpoints, available)
    if count <= 1:
        return [latest]
    span = latest - first
    offsets = [
        int(np.floor(index * span / (count - 1) + 0.5))
        for index in range(count)
    ]
    return [first + offset for offset in offsets]


def _target_endpoint_budgets(windows, target_endpoints):
    """把统一或逐观察期的终点预算规范化为 ``label -> 正整数``。

    ``None`` 由调用方保留给 fixed-step 模式，不会进入本函数。整数沿用旧
    Python API，统一应用到全部观察期；Mapping 则必须覆盖本次实际请求的
    每个观察期。允许 Mapping 包含未被本次 ``lookbacks`` 使用的额外键，
    这样默认五周期预算也能直接用于只分析其中部分周期的调用。
    """
    labels = list(windows)
    if isinstance(target_endpoints, Mapping):
        missing = [label for label in labels if label not in target_endpoints]
        if missing:
            formatted = ", ".join(repr(label) for label in missing)
            raise ValueError(
                "target_endpoints 映射缺少观察期预算: " + formatted)
        return {
            label: _positive_int(
                target_endpoints[label],
                f"target_endpoints[{label!r}]",
            )
            for label in labels
        }

    target = _positive_int(target_endpoints, "target_endpoints")
    return {label: target for label in labels}


def _endpoint_spacing_metadata(endpoints, maturity_days):
    """返回实际可回测终点的间距及相邻 T 窗口重叠比例。"""
    if len(endpoints) < 2:
        return None, None, None, None
    spacing = np.diff(np.asarray(endpoints, dtype=int))
    spacing_min = int(np.min(spacing))
    spacing_max = int(np.max(spacing))
    # 每窗含 T 个损益交易日；相邻终点相隔 s 日时重叠 max(T-s, 0) 日。
    overlap_min = max(0.0, (maturity_days - spacing_max) / maturity_days)
    overlap_max = max(0.0, (maturity_days - spacing_min) / maturity_days)
    return spacing_min, spacing_max, float(overlap_min), float(overlap_max)


def _max_realized_sigma_warmup_days(cases):
    """返回本轮候选中实际参与触发的最大 realized-sigma 日窗口。"""
    return max(
        (_realized_sigma_window_days(case.strategy) for case in cases),
        default=0,
    )


def _history_warmup_log_returns(
        history, group_last_positions, day0_group, warmup_days,
        steps_per_day, context):
    """从同一历史序列 Day 0 前提取独立 realized-sigma 预热收益。

    种子从 Day 0 前第 ``warmup_days`` 个交易日的收盘锚点开始，到 Day 0
    收盘结束，因而形成恰好这 V 个交易日内的收益；它不会被拼入实际回测
    ``external_path``。日内数据还要求至少 ``V * steps_per_day`` 根收益，
    防止缺 bar 的残缺预热被误当成完整窗口。
    """
    warmup_days = int(warmup_days)
    if warmup_days <= 0:
        return np.array([], dtype=float)
    anchor_group = int(day0_group) - warmup_days
    if anchor_group < 0:
        raise ValueError(
            f"{context} 的 Day 0 前不足完整 {warmup_days} 个预热交易日")
    anchor_pos = int(group_last_positions[anchor_group])
    day0_pos = int(group_last_positions[int(day0_group)])
    seed_prices = history.iloc[anchor_pos:day0_pos + 1]
    spd = _positive_int(steps_per_day, "steps_per_day")
    seed_groups = _history_trading_day_groups(seed_prices, spd)
    _, group_counts = np.unique(seed_groups, return_counts=True)
    actionable_counts = group_counts[1:]
    if (len(actionable_counts) != warmup_days
            or np.any(actionable_counts < spd)):
        observed = ",".join(str(int(value)) for value in actionable_counts)
        raise ValueError(
            f"{context} 的 realized sigma 预热交易日不完整："
            f"要求 {warmup_days} 日且每日至少 {spd} 根 bar，"
            f"实际各日为 [{observed}]")
    values = seed_prices.to_numpy(dtype=float)
    warmup_returns = np.log(values[1:] / values[:-1])
    required_bars = warmup_days * spd
    if len(warmup_returns) < required_bars:
        raise ValueError(
            f"{context} 的 realized sigma 预热不完整："
            f"需要 Day 0 前 {warmup_days} 日 / {required_bars} 根收益，"
            f"实际仅 {len(warmup_returns)} 根")
    if not np.all(np.isfinite(warmup_returns)):
        raise ValueError(f"{context} 的 realized sigma 预热含非有限收益")
    return warmup_returns


def recommend_by_rolling_history(
    option,
    prices,
    cases,
    backtest_kwargs=None,
    *,
    lookbacks=None,
    step_days=5,
    target_endpoints=None,
    steps_per_day=None,
):
    """
    在真实历史价格的多个可比窗口上重跑并推荐对冲策略。

    与 :func:`recommend_by_lookback` 只截取一次回测结果不同，本函数直接接收
    ``pd.Series``（保留 DatetimeIndex）或一维价格数组。它使用 ``option`` 的
    剩余期限作为每个可比回测窗口的固定存续期，在最近周/月/季/半年/年的观察期
    内选择多个回测终点；每个终点向前取完整期权期限，所以策略起点可以早于
    观察期起点。``target_endpoints`` 可传一个统一正整数，或传
    ``{lookback: budget}`` 映射为各观察期指定独立预算；非空时会按对应目标
    数量均匀覆盖观察期并纳入首尾，``None`` 则沿用 ``step_days`` 固定步长。
    全部 ``StrategyCase`` 运行相同的真实价格窗口，每轮会按真实起始价等比例
    伸缩期权的价格量纲要素。``cases``
    必须且只能包含一个 ``close_to_close`` 策略作为固定基准；所有候选分数
    都在其与该基准双方成功的严格同窗交集上重算。主排名先计算各窗口相对
    C2C 的有界 RMS 优势，再取逐窗均值，使每个历史终点等权；合并金额 RMS
    只保留作诊断。

    对带 ``DatetimeIndex`` 的日内行情，窗口边界按真实交易日组定位，而非
    ``位置 / steps_per_day``。起点使用前一交易日最后一根作为 Day 0 锚点，
    终点使用到期交易日最后一根，因此偶发缺/多 bar 不会错位，也不会把到期
    后 bar 混入窗口。

    参数 ``lookbacks`` 描述“最近多少交易日内的可用回测终点用于评估策略”，
    而不是临时改写期权期限。只有整个观察期内的抽样终点都具备足够的前置
    到期历史时才视为完整；否则仍给出已有窗口的同窗诊断，但不形成正式历史参考。

    Returns
    -------
    (recommendations, ranking, window_results)
        ``recommendations`` 仅含存在至少一个正式可比候选时的同窗冠军（可能
        是 C2C 本身）；``ranking`` 含正式、同窗诊断、部分配对与失败结果，并
        披露 ``baseline_score`` / ``improvement_vs_c2c`` /
        ``selection_improvement_vs_c2c`` / ``paired_windows``；
        ``window_results[lookback][window_id][strategy]`` 保存逐窗回测明细。
    """
    history = _history_series(prices)
    cases = list(cases)
    seen = set()
    for case in cases:
        if case.name in seen:
            raise ValueError(f"策略名称重复: {case.name}")
        seen.add(case.name)
    baseline_case = _history_baseline_case(cases)
    warmup_days = _max_realized_sigma_warmup_days(cases)

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
    if target_endpoints is None:
        step = _positive_int(step_days, "step_days")
        sampling_mode = "fixed_step"
    else:
        # target_count 模式与固定步长互斥；目标数量显式给出时直接取得
        # 优先权，并在结果中把 step_days 记为 NaN。
        step = None
        sampling_mode = "target_count"

    reserved = {
        "strategy", "path_source", "external_path", "prices",
        "sigma_warmup_log_returns", "strict_sigma_warmup",
    }
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
    target_budgets = (
        _target_endpoint_budgets(windows, target_endpoints)
        if sampling_mode == "target_count" else None
    )
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
        if sampling_mode == "target_count":
            target = target_budgets[label]
            endpoints = _target_count_endpoint_groups(
                n_history_groups, days, target)
            planned_endpoints = min(days, target)
        else:
            target = None
            endpoints = _rolling_endpoint_groups(
                n_history_groups, days, step)
            planned_endpoints = len(
                _rolling_endpoint_groups(days, days, step))
        valid_endpoints = [end for end in endpoints if end >= maturity_days]
        warmup_valid_endpoints = [
            end for end in valid_endpoints
            if end >= maturity_days + warmup_days
        ]
        (spacing_min, spacing_max,
         overlap_min, overlap_max) = _endpoint_spacing_metadata(
            valid_endpoints, maturity_days)
        # 完整观察期既需要足够的近期终点，也需要最早抽样终点之前仍有一段
        # 完整期权期限。后者正是旧实现令 T > lookback 时先天 0 窗口的问题：
        # 起点应允许落在 lookback 之前，而不是强迫整个生命周期塞进 tail。
        history_complete = (
            n_history_groups >= days + maturity_days + warmup_days
            and len(endpoints) == planned_endpoints
            and len(valid_endpoints) == planned_endpoints
            and len(warmup_valid_endpoints) == planned_endpoints
        )

        label_results = {}
        # 按 window_id 保存日级结果，后续所有指标都在候选与 C2C 的严格
        # 同窗交集上重算，不能再把各策略各自成功的窗口直接拼接比较。
        collected = {case.name: {} for case in cases}
        case_failures = {case.name: 0 for case in cases}
        case_failure_reasons = {case.name: [] for case in cases}
        strategy_types = {}
        endpoint_failures = 0
        endpoint_failure_reasons = []
        warmup_eligible_endpoint_count = 0
        for window_no, end_group in enumerate(valid_endpoints, start=1):
            start_group = end_group - maturity_days
            # Day 0 只取起始组最后一根作为建仓锚点；随后包含恰好 T 个完整
            # 交易日，直至目标终点组最后一根。这样不会带入到期后的 bar。
            start_pos = int(group_last_positions[start_group])
            stop_pos = int(group_last_positions[end_group]) + 1
            path = history.iloc[start_pos:stop_pos]
            try:
                warmup_log_returns = _history_warmup_log_returns(
                    history,
                    group_last_positions,
                    start_group,
                    warmup_days,
                    spd,
                    f"{label}/window_{window_no}",
                )
                warmup_error = ""
                warmup_eligible_endpoint_count += 1
            except ValueError as exc:
                warmup_log_returns = np.array([], dtype=float)
                warmup_error = str(exc)
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
                    endpoint_failure_reasons.append(str(exc))
                    label_results[window_id] = {
                        "_window_error": str(exc),
                    }
                    continue
            for case in cases:
                case_warmup_days = _realized_sigma_window_days(case.strategy)
                if case_warmup_days and warmup_error:
                    case_failures[case.name] += 1
                    case_failure_reasons[case.name].append(warmup_error)
                    strategy_types[case.name] = getattr(
                        case.strategy, "name", "sigma_band")
                    per_case[case.name] = {
                        "error": warmup_error,
                        "strategy_name": strategy_types[case.name],
                    }
                    continue
                scaled_option, rescale_info = _rescale_option_to_real_s0(
                    option, float(path.iloc[0])
                )
                scaled_strategy = _rescale_strategy_to_real_s0(
                    case.strategy, rescale_info["ratio"]
                )
                try:
                    warmup_kwargs = ({
                        "sigma_warmup_log_returns": warmup_log_returns,
                        "strict_sigma_warmup": True,
                    } if case_warmup_days else {})
                    result = HedgeBacktest(
                        scaled_option,
                        path_source="historical",
                        external_path=external_path,
                        strategy=scaled_strategy,
                        steps_per_day=spd,
                        **warmup_kwargs,
                        **kwargs,
                    ).run()
                except ValueError as exc:
                    # 固定时刻可能只在个别窗口缺少目标 bar；该策略保留
                    # incomplete 诊断，但不能阻断其它策略的历史推荐。
                    if (not isinstance(case.strategy, FixedTimeStrategy)
                            and not case_warmup_days):
                        raise
                    case_failures[case.name] += 1
                    case_failure_reasons[case.name].append(str(exc))
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
                collected[case.name][window_id] = daily
                strategy_types[case.name] = result.get("strategy_name", "unknown")
                per_case[case.name] = result
            label_results[window_id] = per_case
        window_results[label] = label_results
        history_complete = bool(
            history_complete
            and warmup_eligible_endpoint_count == planned_endpoints)

        baseline_by_window = collected[baseline_case.name]
        baseline_window_ids = list(baseline_by_window)
        baseline_windows = len(baseline_window_ids)
        baseline_complete = bool(
            history_complete
            and endpoint_failures == 0
            and baseline_windows == planned_endpoints
            and baseline_windows > 0
        )

        for case in cases:
            candidate_by_window = collected[case.name]
            paired_window_ids = [
                window_id for window_id in baseline_window_ids
                if window_id in candidate_by_window
            ]
            daily_windows = [
                candidate_by_window[window_id]
                for window_id in paired_window_ids
            ]
            paired_baseline_windows = [
                baseline_by_window[window_id]
                for window_id in paired_window_ids
            ]
            if daily_windows:
                combined = pd.concat(daily_windows, ignore_index=True)
                metrics = _daily_metrics(combined)
                worst_drawdown = max(_max_drawdown(x["net_pnl"]) for x in daily_windows)
            else:
                combined = pd.DataFrame(columns=["net_pnl", "tc_paid"])
                metrics = _daily_metrics(combined)
                worst_drawdown = 0.0
            if paired_baseline_windows:
                baseline_combined = pd.concat(
                    paired_baseline_windows, ignore_index=True)
                baseline_metrics = _daily_metrics(baseline_combined)
            else:
                baseline_combined = pd.DataFrame(
                    columns=["net_pnl", "tc_paid"])
                baseline_metrics = _daily_metrics(baseline_combined)

            score = metrics["score"]
            baseline_score = baseline_metrics["score"]
            if np.isfinite(score) and np.isfinite(baseline_score):
                score_delta = float(score - baseline_score)
            else:
                score_delta = np.nan
            if case.name == baseline_case.name and np.isfinite(score):
                improvement = 0.0
            else:
                improvement = _history_improvement_vs_c2c(
                    score, baseline_score)

            window_improvements = []
            window_selection_advantages = []
            window_wins = 0
            for candidate_daily, baseline_daily in zip(
                    daily_windows, paired_baseline_windows):
                candidate_window_score = _daily_metrics(
                    candidate_daily)["score"]
                baseline_window_score = _daily_metrics(
                    baseline_daily)["score"]
                if (np.isfinite(candidate_window_score)
                        and np.isfinite(baseline_window_score)
                        and candidate_window_score < baseline_window_score
                        and not np.isclose(
                            candidate_window_score, baseline_window_score,
                            rtol=1e-12, atol=1e-12)):
                    window_wins += 1
                window_improvement = _history_improvement_vs_c2c(
                    candidate_window_score, baseline_window_score)
                if np.isfinite(window_improvement):
                    window_improvements.append(window_improvement)
                selection_advantage = _history_selection_advantage_vs_c2c(
                    candidate_window_score, baseline_window_score)
                if np.isfinite(selection_advantage):
                    window_selection_advantages.append(selection_advantage)
            paired_windows = len(paired_window_ids)
            if case.name == baseline_case.name and paired_windows:
                window_win_rate = 0.0
                median_window_improvement = 0.0
                selection_advantage = 0.0
            else:
                window_win_rate = (
                    float(window_wins / paired_windows)
                    if paired_windows else np.nan)
                median_window_improvement = (
                    float(np.median(window_improvements))
                    if window_improvements else np.nan)
                selection_advantage = (
                    float(np.mean(window_selection_advantages))
                    if window_selection_advantages else np.nan)

            skipped = endpoint_failures + case_failures[case.name]
            if case_failure_reasons[case.name]:
                failure_scope = "strategy"
                failure_reason = case_failure_reasons[case.name][0]
            elif endpoint_failure_reasons:
                failure_scope = "endpoint"
                failure_reason = endpoint_failure_reasons[0]
            else:
                failure_scope = ""
                failure_reason = ""
            comparison_eligible = bool(
                baseline_windows > 0
                and paired_windows == baseline_windows
            )
            recommendation_eligible = bool(
                baseline_complete and comparison_eligible)
            complete = recommendation_eligible
            if recommendation_eligible:
                comparison_status = "formal"
            elif comparison_eligible:
                comparison_status = "diagnostic"
            elif paired_windows:
                comparison_status = "partial_pair"
            else:
                comparison_status = "no_pair"
            rows.append({
                "lookback": label,
                "lookback_days": days,
                "strategy": case.name,
                "strategy_type": strategy_types.get(case.name, "unknown"),
                "evaluation_mode": "rolling_history",
                "maturity_days": maturity_days,
                "sampling_mode": sampling_mode,
                "step_days": step if step is not None else np.nan,
                "target_endpoints": target if target is not None else np.nan,
                "planned_endpoints": planned_endpoints,
                "selected_endpoints": len(endpoints),
                "endpoint_spacing_min": spacing_min,
                "endpoint_spacing_max": spacing_max,
                "window_overlap_min_ratio": overlap_min,
                "window_overlap_max_ratio": overlap_max,
                "maturity_exceeds_lookback": maturity_days > days,
                "realized_sigma_warmup_days": warmup_days,
                "required_history_days": maturity_days + days + warmup_days,
                "available_history_days": n_history_groups,
                "history_days_available": min(days, n_history_groups),
                "eligible_endpoints": len(valid_endpoints),
                "warmup_eligible_endpoints": warmup_eligible_endpoint_count,
                "rolling_windows": paired_windows,
                "skipped_endpoints": skipped,
                # 保留首个真实失败原因供 GUI 在零有效窗口时准确诊断；
                # 完整逐窗错误仍位于 window_results 中。
                "failure_scope": failure_scope,
                "failure_reason": failure_reason,
                "trailing_partial_groups_dropped": len(trailing_group_errors),
                "days_used": metrics["n_trade_days"],
                "complete_window": complete,
                "history_complete": bool(history_complete),
                "baseline_complete": baseline_complete,
                "comparison_eligible": comparison_eligible,
                "recommendation_eligible": recommendation_eligible,
                "comparison_status": comparison_status,
                "baseline_strategy": baseline_case.name,
                "baseline_windows": baseline_windows,
                "paired_windows": paired_windows,
                "unpaired_windows": max(0, baseline_windows - paired_windows),
                "comparison_coverage": (
                    float(paired_windows / baseline_windows)
                    if baseline_windows else 0.0),
                "window_pnl": metrics["total_net_pnl"],
                "window_tc": metrics["total_tc"],
                "daily_net_pnl_rms": metrics["daily_net_pnl_rms"],
                "avg_daily_tc": metrics["avg_daily_tc"],
                "mean_daily_pnl": metrics["mean_daily_pnl"],
                "pnl_volatility": metrics["pnl_volatility"],
                # 各窗口分别运行且不把边界首尾相连；它们在统计上可能高度
                # 重叠，并非独立样本。这里仅披露其中最坏的单窗口回撤。
                "max_drawdown": worst_drawdown,
                "score": score,
                "baseline_score": baseline_score,
                "baseline_days_used": baseline_metrics["n_trade_days"],
                "score_delta_vs_c2c": score_delta,
                "improvement_vs_c2c": improvement,
                "window_win_rate_vs_c2c": window_win_rate,
                "median_window_improvement_vs_c2c": (
                    median_window_improvement),
                # 历史最优按终点窗口等权：每窗先计算有界同窗 C2C
                # 优势，再取均值；合并金额 RMS 不参与名次。
                "selection_metric": (
                    HISTORY_SELECTION_METRIC),
                "selection_improvement_vs_c2c": (
                    selection_advantage),
                "relative_comparison_windows": len(
                    window_selection_advantages),
                **{f"meta_{k}": v for k, v in case.metadata.items()},
            })

    ranking = _rank_history_rows(rows)
    if ranking.empty:
        return ranking.copy(), ranking, window_results
    recommendation_rows = []
    for _lookback, group in ranking.groupby("lookback", sort=False):
        eligible = group[group["recommendation_eligible"].astype(bool)]
        non_baseline = eligible[
            eligible["strategy_type"].astype(str) != "close_to_close"]
        # 只有至少一个候选与 C2C 完成正式同窗比较时，才产生该周期的
        # “历史最优参考”；若候选全失败，保留排名诊断但不把仅有基准包装
        # 成已经完成的策略择优。
        if eligible.empty or non_baseline.empty:
            continue
        recommendation_rows.append(
            eligible.sort_values("rank", kind="stable").iloc[0])
    recommendations = (
        pd.DataFrame(recommendation_rows).reset_index(drop=True)
        if recommendation_rows else ranking.iloc[:0].copy()
    )
    return recommendations, ranking, window_results


def _recommend_from_explicit_history_plans(
        option, cases, kwargs, spd, baseline_case, plans):
    """运行已经按具体合约边界切好的历史窗口计划。"""
    rows = []
    window_results = {}
    for plan in plans:
        label = plan["label"]
        label_results = {}
        collected = {case.name: {} for case in cases}
        case_failures = {case.name: 0 for case in cases}
        case_failure_reasons = {case.name: [] for case in cases}
        strategy_types = {}
        endpoint_failures = 0
        endpoint_failure_reasons = []
        contract_by_window = {}

        for entry in plan["entries"]:
            window_id = entry["window_id"]
            contract_code = str(entry.get("contract_code", ""))
            contract_by_window[window_id] = contract_code
            if entry.get("error"):
                reason = str(entry["error"])
                endpoint_failures += 1
                endpoint_failure_reasons.append(reason)
                label_results[window_id] = {
                    "_window_error": reason,
                    "_window_error_meta": {
                        "history_contract_code": contract_code,
                        "history_endpoint_date": entry.get("endpoint_date"),
                    },
                }
                continue

            path = entry["path"]
            external_path = path
            entry_spd = _positive_int(
                entry.get("steps_per_day", spd),
                f"{label}/{window_id} steps_per_day",
            )
            per_case = {}
            try:
                _expiry_terminal_index(
                    path.index, plan["maturity_days"], entry_spd)
            except ValueError as exc:
                endpoint_failures += 1
                endpoint_failure_reasons.append(str(exc))
                label_results[window_id] = {
                    "_window_error": str(exc),
                    "_window_error_meta": {
                        "history_contract_code": contract_code,
                        "history_endpoint_date": entry.get("endpoint_date"),
                    },
                }
                continue

            for case in cases:
                case_warmup_days = _realized_sigma_window_days(case.strategy)
                warmup_error = str(entry.get("warmup_error", "") or "")
                if case_warmup_days and warmup_error:
                    case_failures[case.name] += 1
                    case_failure_reasons[case.name].append(warmup_error)
                    strategy_types[case.name] = getattr(
                        case.strategy, "name", "sigma_band")
                    per_case[case.name] = {
                        "error": warmup_error,
                        "strategy_name": strategy_types[case.name],
                        "history_contract_code": contract_code,
                        "history_endpoint_date": entry.get("endpoint_date"),
                    }
                    continue
                scaled_option, rescale_info = _rescale_option_to_real_s0(
                    option, float(path.iloc[0]))
                scaled_strategy = _rescale_strategy_to_real_s0(
                    case.strategy, rescale_info["ratio"])
                # 品种历史池中的每个窗口都属于一个明确的具体合约；固定
                # 时刻策略必须使用该合约自己的交易时段，不能沿用产品代码
                # 或上一个历史合约的 session 判断。
                if (isinstance(scaled_strategy, FixedTimeStrategy)
                        and "trading_sessions" in entry):
                    scaled_strategy.set_trading_sessions(
                        entry.get("trading_sessions"))
                try:
                    warmup_kwargs = ({
                        "sigma_warmup_log_returns": entry.get(
                            "warmup_log_returns", np.array([], dtype=float)),
                        "strict_sigma_warmup": True,
                    } if case_warmup_days else {})
                    result = HedgeBacktest(
                        scaled_option,
                        path_source="historical",
                        external_path=external_path,
                        strategy=scaled_strategy,
                        steps_per_day=entry_spd,
                        **warmup_kwargs,
                        **kwargs,
                    ).run()
                except ValueError as exc:
                    if (not isinstance(case.strategy, FixedTimeStrategy)
                            and not case_warmup_days):
                        raise
                    case_failures[case.name] += 1
                    case_failure_reasons[case.name].append(str(exc))
                    strategy_types[case.name] = getattr(
                        case.strategy, "name", "fixed_times")
                    per_case[case.name] = {
                        "error": str(exc),
                        "strategy_name": strategy_types[case.name],
                        "history_contract_code": contract_code,
                        "history_endpoint_date": entry.get("endpoint_date"),
                    }
                    continue
                if _positive_int(
                    result.get("steps_per_day"),
                    f"{label}/{window_id}/{case.name} steps_per_day",
                ) != entry_spd:
                    raise ValueError("滚动回测返回了不一致的 steps_per_day")
                result["history_contract_code"] = contract_code
                result["history_endpoint_date"] = entry.get("endpoint_date")
                daily = _aggregate_result_by_day(result, entry_spd)
                collected[case.name][window_id] = daily
                strategy_types[case.name] = result.get(
                    "strategy_name", "unknown")
                per_case[case.name] = result
            label_results[window_id] = per_case
        window_results[label] = label_results

        baseline_by_window = collected[baseline_case.name]
        baseline_window_ids = list(baseline_by_window)
        baseline_windows = len(baseline_window_ids)
        baseline_complete = bool(
            plan["history_complete"]
            and endpoint_failures == 0
            and baseline_windows == plan["planned_endpoints"]
            and baseline_windows > 0
        )

        for case in cases:
            candidate_by_window = collected[case.name]
            paired_window_ids = [
                window_id for window_id in baseline_window_ids
                if window_id in candidate_by_window
            ]
            daily_windows = [
                candidate_by_window[window_id]
                for window_id in paired_window_ids
            ]
            paired_baseline_windows = [
                baseline_by_window[window_id]
                for window_id in paired_window_ids
            ]
            if daily_windows:
                combined = pd.concat(daily_windows, ignore_index=True)
                metrics = _daily_metrics(combined)
                worst_drawdown = max(
                    _max_drawdown(frame["net_pnl"])
                    for frame in daily_windows)
            else:
                combined = pd.DataFrame(columns=["net_pnl", "tc_paid"])
                metrics = _daily_metrics(combined)
                worst_drawdown = 0.0
            if paired_baseline_windows:
                baseline_combined = pd.concat(
                    paired_baseline_windows, ignore_index=True)
                baseline_metrics = _daily_metrics(baseline_combined)
            else:
                baseline_combined = pd.DataFrame(
                    columns=["net_pnl", "tc_paid"])
                baseline_metrics = _daily_metrics(baseline_combined)

            score = metrics["score"]
            baseline_score = baseline_metrics["score"]
            if np.isfinite(score) and np.isfinite(baseline_score):
                score_delta = float(score - baseline_score)
            else:
                score_delta = np.nan
            if case.name == baseline_case.name and np.isfinite(score):
                improvement = 0.0
            else:
                improvement = _history_improvement_vs_c2c(
                    score, baseline_score)

            window_improvements = []
            window_selection_advantages = []
            window_wins = 0
            for candidate_daily, baseline_daily in zip(
                    daily_windows, paired_baseline_windows):
                candidate_window_score = _daily_metrics(
                    candidate_daily)["score"]
                baseline_window_score = _daily_metrics(
                    baseline_daily)["score"]
                if (np.isfinite(candidate_window_score)
                        and np.isfinite(baseline_window_score)
                        and candidate_window_score < baseline_window_score
                        and not np.isclose(
                            candidate_window_score, baseline_window_score,
                            rtol=1e-12, atol=1e-12)):
                    window_wins += 1
                window_improvement = _history_improvement_vs_c2c(
                    candidate_window_score, baseline_window_score)
                if np.isfinite(window_improvement):
                    window_improvements.append(window_improvement)
                selection_advantage = _history_selection_advantage_vs_c2c(
                    candidate_window_score, baseline_window_score)
                if np.isfinite(selection_advantage):
                    window_selection_advantages.append(selection_advantage)
            paired_windows = len(paired_window_ids)
            if case.name == baseline_case.name and paired_windows:
                window_win_rate = 0.0
                median_window_improvement = 0.0
                selection_advantage = 0.0
            else:
                window_win_rate = (
                    float(window_wins / paired_windows)
                    if paired_windows else np.nan)
                median_window_improvement = (
                    float(np.median(window_improvements))
                    if window_improvements else np.nan)
                selection_advantage = (
                    float(np.mean(window_selection_advantages))
                    if window_selection_advantages else np.nan)

            skipped = endpoint_failures + case_failures[case.name]
            if case_failure_reasons[case.name]:
                failure_scope = "strategy"
                failure_reason = case_failure_reasons[case.name][0]
            elif endpoint_failure_reasons:
                failure_scope = "endpoint"
                failure_reason = endpoint_failure_reasons[0]
            else:
                failure_scope = ""
                failure_reason = ""
            comparison_eligible = bool(
                baseline_windows > 0
                and paired_windows == baseline_windows)
            recommendation_eligible = bool(
                baseline_complete and comparison_eligible)
            if recommendation_eligible:
                comparison_status = "formal"
            elif comparison_eligible:
                comparison_status = "diagnostic"
            elif paired_windows:
                comparison_status = "partial_pair"
            else:
                comparison_status = "no_pair"
            paired_contracts = {
                contract_by_window.get(window_id, "")
                for window_id in paired_window_ids
                if contract_by_window.get(window_id, "")
            }
            rows.append({
                **plan["row_metadata"],
                "lookback": label,
                "strategy": case.name,
                "strategy_type": strategy_types.get(case.name, "unknown"),
                "rolling_windows": paired_windows,
                "skipped_endpoints": skipped,
                "failure_scope": failure_scope,
                "failure_reason": failure_reason,
                "days_used": metrics["n_trade_days"],
                "complete_window": recommendation_eligible,
                "baseline_complete": baseline_complete,
                "comparison_eligible": comparison_eligible,
                "recommendation_eligible": recommendation_eligible,
                "comparison_status": comparison_status,
                "baseline_strategy": baseline_case.name,
                "baseline_windows": baseline_windows,
                "paired_windows": paired_windows,
                "unpaired_windows": max(
                    0, baseline_windows - paired_windows),
                "comparison_coverage": (
                    float(paired_windows / baseline_windows)
                    if baseline_windows else 0.0),
                "contract_count": len(paired_contracts),
                "window_pnl": metrics["total_net_pnl"],
                "window_tc": metrics["total_tc"],
                "daily_net_pnl_rms": metrics["daily_net_pnl_rms"],
                "avg_daily_tc": metrics["avg_daily_tc"],
                "mean_daily_pnl": metrics["mean_daily_pnl"],
                "pnl_volatility": metrics["pnl_volatility"],
                "max_drawdown": worst_drawdown,
                "score": score,
                "baseline_score": baseline_score,
                "baseline_days_used": baseline_metrics["n_trade_days"],
                "score_delta_vs_c2c": score_delta,
                "improvement_vs_c2c": improvement,
                "window_win_rate_vs_c2c": window_win_rate,
                "median_window_improvement_vs_c2c": (
                    median_window_improvement),
                # 品种池尤其不能直接比较不同具体合约的金额 RMS；每个
                # 历史终点在主排名中拥有相同权重。
                "selection_metric": (
                    HISTORY_SELECTION_METRIC),
                "selection_improvement_vs_c2c": (
                    selection_advantage),
                "relative_comparison_windows": len(
                    window_selection_advantages),
                "paired_contract_codes": tuple(sorted(paired_contracts)),
                **{f"meta_{key}": value
                   for key, value in case.metadata.items()},
            })

    ranking = _rank_history_rows(rows)
    if ranking.empty:
        return ranking.copy(), ranking, window_results
    recommendation_rows = []
    for _lookback, group in ranking.groupby("lookback", sort=False):
        eligible = group[group["recommendation_eligible"].astype(bool)]
        non_baseline = eligible[
            eligible["strategy_type"].astype(str) != "close_to_close"]
        if eligible.empty or non_baseline.empty:
            continue
        recommendation_rows.append(
            eligible.sort_values("rank", kind="stable").iloc[0])
    recommendations = (
        pd.DataFrame(recommendation_rows).reset_index(drop=True)
        if recommendation_rows else ranking.iloc[:0].copy())
    return recommendations, ranking, window_results


def recommend_by_contract_history_pool(
        option, pool, cases, backtest_kwargs=None, *, lookbacks=None,
        step_days=5, target_endpoints=None, steps_per_day=None):
    """在历史主力映射中按具体合约边界回测同品种期货样本池。

    每个终点使用当日的历史主力合约，并只向前切取该具体合约自身的完整
    ``T`` 日路径；窗口绝不会跨换月边界。周期仍表示从最近完整交易日向前
    选择终点的证据范围。``target_endpoints`` 与单序列接口一致，支持统一
    正整数或逐观察期 Mapping；``None`` 保留 fixed-step 兼容。策略按逐窗
    有界 C2C 优势的均值排名，每个终点等权，因此不同价格量级的具体合约
    不会以金额大小形成隐含权重。
    """
    if not isinstance(pool, ContractHistoryPool):
        raise TypeError("pool 必须是 ContractHistoryPool")
    cases = list(cases)
    seen = set()
    for case in cases:
        if case.name in seen:
            raise ValueError(f"策略名称重复: {case.name}")
        seen.add(case.name)
    baseline_case = _history_baseline_case(cases)
    warmup_days = _max_realized_sigma_warmup_days(cases)

    prices_by_contract = {}
    invalid_contracts = {
        str(code).strip().upper(): str(reason)
        for code, reason in dict(pool.contract_load_errors).items()
    }
    for code, prices in dict(pool.contract_prices).items():
        normalized_code = str(code).strip().upper()
        try:
            history = _history_series(prices)
            if not isinstance(history.index, pd.DatetimeIndex):
                raise ValueError("具体合约行情必须保留 DatetimeIndex")
        except (TypeError, ValueError) as exc:
            invalid_contracts[normalized_code] = str(exc)
            continue
        prices_by_contract[normalized_code] = history
    if not prices_by_contract:
        detail = next(iter(invalid_contracts.values()), "没有传入行情")
        raise ValueError(f"跨合约历史样本池没有可用的具体合约行情：{detail}")
    kwargs = dict(backtest_kwargs or {})
    kw_spd = kwargs.pop("steps_per_day", None)
    if steps_per_day is not None and kw_spd is not None:
        explicit_spd = _positive_int(steps_per_day, "steps_per_day")
        kwargs_spd = _positive_int(
            kw_spd, "backtest_kwargs['steps_per_day']")
        if explicit_spd != kwargs_spd:
            raise ValueError(
                "steps_per_day 与 backtest_kwargs['steps_per_day'] 不一致: "
                f"{explicit_spd} != {kwargs_spd}")
    explicit_spd = (
        steps_per_day if steps_per_day is not None else kw_spd)
    default_spd = _positive_int(
        explicit_spd if explicit_spd is not None else 1,
        "steps_per_day",
    )
    if target_endpoints is None:
        step = _positive_int(step_days, "step_days")
        sampling_mode = "fixed_step"
    else:
        step = None
        sampling_mode = "target_count"
    reserved = {
        "strategy", "path_source", "external_path", "prices",
        "sigma_warmup_log_returns", "strict_sigma_warmup",
    }
    conflict = sorted(reserved.intersection(kwargs))
    if conflict:
        raise ValueError(
            "backtest_kwargs 不应覆盖滚动历史接口管理的参数: "
            + ", ".join(conflict))
    try:
        maturity_value = float(option._time_remaining)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("option 必须提供正整数交易日 _time_remaining") from exc
    maturity_days = _positive_int(
        maturity_value, "option._time_remaining")

    mapping = pool.main_contract_by_date.copy()
    if not isinstance(mapping, pd.Series):
        raise TypeError("main_contract_by_date 必须是 pandas Series")
    mapping.index = pd.DatetimeIndex(pd.to_datetime(mapping.index)).normalize()
    if mapping.index.tz is not None:
        mapping.index = mapping.index.tz_localize(None)
    if mapping.index.hasnans:
        raise ValueError("主力合约日期映射含 NaT")
    mapping = mapping[~mapping.index.duplicated(keep="last")].sort_index()
    mapping = mapping.dropna().map(lambda value: str(value).strip().upper())
    mapping = mapping[mapping.ne("")]
    if mapping.empty:
        raise ValueError("跨合约历史样本池没有主力合约日期映射")
    from .wind_data import (
        classify_wind_history_code,
        get_trading_session_clock_ranges,
    )
    product_info = classify_wind_history_code(pool.product_code)
    if product_info["mode"] != "product_pool":
        raise ValueError(
            "ContractHistoryPool.product_code 必须是期货品种代码，"
            "例如 P.DCE")
    for contract_code in set(mapping).union(prices_by_contract):
        contract_info = classify_wind_history_code(contract_code)
        if not (
                contract_info.get("is_futures_contract")
                and contract_info.get("product") == product_info["product"]
                and contract_info.get("exchange") == product_info["exchange"]):
            raise ValueError(
                f"历史合约 {contract_code!r} 与品种 "
                f"{product_info['code']} 不一致")
    declared_asof = str(pool.main_contract_asof).strip().upper()
    if declared_asof != str(mapping.iloc[-1]).strip().upper():
        raise ValueError(
            "main_contract_asof 必须等于主力映射截止日的具体合约："
            f"{declared_asof!r} != {mapping.iloc[-1]!r}")

    contract_details = {}
    trailing_partial_groups = 0
    contract_steps = {}
    for code, raw_history in prices_by_contract.items():
        contract_spd = _positive_int(
            explicit_spd if explicit_spd is not None
            else _infer_intraday_steps(raw_history.index),
            f"{code} steps_per_day",
        )
        history, trailing_errors = _trim_incomplete_trailing_groups(
            raw_history, contract_spd)
        trailing_partial_groups += len(trailing_errors)
        groups = _history_trading_day_groups(history, contract_spd)
        group_ids, first_positions = np.unique(groups, return_index=True)
        if not len(group_ids):
            continue
        last_positions = np.r_[
            first_positions[1:] - 1,
            len(history) - 1,
        ].astype(int)
        close_dates = pd.DatetimeIndex(
            history.index[last_positions]).normalize()
        if close_dates.tz is not None:
            close_dates = close_dates.tz_localize(None)
        raw_observed_dates = pd.DatetimeIndex(raw_history.index).normalize()
        if raw_observed_dates.tz is not None:
            raw_observed_dates = raw_observed_dates.tz_localize(None)
        contract_details[code] = {
            "history": history,
            "last_positions": last_positions,
            "close_dates": close_dates,
            "steps_per_day": contract_spd,
            "latest_observed_date": raw_observed_dates[-1],
            "trailing_group_errors": tuple(trailing_errors),
            "trading_sessions": get_trading_session_clock_ranges(code),
            "date_to_group": {
                date: group_no
                for group_no, date in enumerate(close_dates)
            },
        }
        contract_steps[code] = contract_spd

    # Wind 可能已返回分析当日的主力映射，但当日分钟行情仍处于盘中。
    # 与单序列历史的尾部残组规则一致，仅剔除“位于该合约最新完整行情日之后”
    # 的连续尾端映射；若整个合约缺失或中间断档，则保留为显式失败，不能
    # 静默把分析截止点回退到更早的历史合约。
    raw_mapping_days = len(mapping)
    mapping_trailing_days_dropped = 0
    while len(mapping):
        endpoint_date = mapping.index[-1]
        contract_code = mapping.iloc[-1]
        details = contract_details.get(contract_code)
        if details is None or not len(details["close_dates"]):
            break
        latest_complete_date = details["close_dates"][-1]
        if endpoint_date in details["date_to_group"]:
            break
        # 只有该合约确实包含并剔除了尾部盘中残组时才自动回退。行情只到
        # 主力期之前、整个当前合约缺失或中间断档都必须保留为显式诊断。
        if not details["trailing_group_errors"]:
            break
        if (endpoint_date <= latest_complete_date
                or endpoint_date > details["latest_observed_date"]):
            break
        mapping = mapping.iloc[:-1]
        mapping_trailing_days_dropped += 1
    if mapping.empty:
        raise ValueError("跨合约历史样本池在截止日前没有完整交易日")

    windows = dict(LOOKBACK_DAYS if lookbacks is None else lookbacks)
    target_budgets = (
        _target_endpoint_budgets(windows, target_endpoints)
        if sampling_mode == "target_count" else None
    )
    plans = []
    n_mapping_days = len(mapping)
    for label, days_value in windows.items():
        days = _positive_int(days_value, f"lookbacks[{label!r}]")
        if sampling_mode == "target_count":
            target = target_budgets[label]
            endpoint_positions = _target_count_endpoint_groups(
                n_mapping_days, days, target)
            planned_endpoints = min(days, target)
        else:
            target = None
            endpoint_positions = _rolling_endpoint_groups(
                n_mapping_days, days, step)
            planned_endpoints = len(
                _rolling_endpoint_groups(days, days, step))

        entries = []
        valid_positions = []
        warmup_valid_positions = []
        valid_contracts = set()
        for window_no, mapping_position in enumerate(
                endpoint_positions, start=1):
            endpoint_date = mapping.index[mapping_position]
            contract_code = mapping.iloc[mapping_position]
            details = contract_details.get(contract_code)
            entry = {
                "window_id": f"window_{window_no}",
                "endpoint_date": endpoint_date,
                "endpoint_position": mapping_position,
                "contract_code": contract_code,
                "steps_per_day": (
                    details["steps_per_day"] if details is not None
                    else default_spd),
                "trading_sessions": (
                    details["trading_sessions"]
                    if details is not None else None),
            }
            if details is None:
                detail = invalid_contracts.get(contract_code)
                entry["error"] = (
                    f"具体合约 {contract_code} 的历史行情不可用：{detail}"
                    if detail else f"缺少具体合约 {contract_code} 的历史行情"
                )
                entries.append(entry)
                continue
            end_group = details["date_to_group"].get(endpoint_date)
            if end_group is None:
                entry["error"] = (
                    f"具体合约 {contract_code} 缺少主力日 "
                    f"{endpoint_date.date()} 的完整行情")
                entries.append(entry)
                continue
            if end_group < maturity_days:
                entry["error"] = (
                    f"具体合约 {contract_code} 在 {endpoint_date.date()} 前"
                    f"不足完整 {maturity_days} 个交易日")
                entries.append(entry)
                continue
            start_group = end_group - maturity_days
            start_pos = int(details["last_positions"][start_group])
            stop_pos = int(details["last_positions"][end_group]) + 1
            entry["path"] = details["history"].iloc[start_pos:stop_pos]
            try:
                entry["warmup_log_returns"] = _history_warmup_log_returns(
                    details["history"],
                    details["last_positions"],
                    start_group,
                    warmup_days,
                    details["steps_per_day"],
                    (f"具体合约 {contract_code} 在 "
                     f"{endpoint_date.date()}"),
                )
                entry["warmup_error"] = ""
            except ValueError as exc:
                entry["warmup_log_returns"] = np.array([], dtype=float)
                entry["warmup_error"] = (
                    f"{exc}；预热与 {maturity_days} 日回测路径必须全部来自"
                    f"同一具体合约 {contract_code}，严禁跨合约补齐")
            entries.append(entry)
            valid_positions.append(mapping_position)
            if not entry["warmup_error"]:
                warmup_valid_positions.append(mapping_position)
            valid_contracts.add(contract_code)

        (spacing_min, spacing_max,
         overlap_min, overlap_max) = _endpoint_spacing_metadata(
            valid_positions, maturity_days)
        history_complete = bool(
            n_mapping_days >= days
            and len(endpoint_positions) == planned_endpoints
            and len(valid_positions) == planned_endpoints
            and len(warmup_valid_positions) == planned_endpoints)
        plans.append({
            "label": label,
            "maturity_days": maturity_days,
            "planned_endpoints": planned_endpoints,
            "history_complete": history_complete,
            "entries": entries,
            "row_metadata": {
                "lookback_days": days,
                "evaluation_mode": "contract_pool_history",
                "history_mode": "product_contract_pool",
                "product_code": str(pool.product_code).upper(),
                "main_contract_asof": str(pool.main_contract_asof).upper(),
                "effective_asof_date": mapping.index[-1],
                "effective_main_contract": str(mapping.iloc[-1]).upper(),
                "sampling_mode": sampling_mode,
                "step_days": step if step is not None else np.nan,
                "target_endpoints": (
                    target if target is not None else np.nan),
                "planned_endpoints": planned_endpoints,
                "selected_endpoints": len(endpoint_positions),
                "endpoint_spacing_min": spacing_min,
                "endpoint_spacing_max": spacing_max,
                "window_overlap_min_ratio": overlap_min,
                "window_overlap_max_ratio": overlap_max,
                "maturity_days": maturity_days,
                "maturity_exceeds_lookback": maturity_days > days,
                "realized_sigma_warmup_days": warmup_days,
                "required_history_days": maturity_days + days + warmup_days,
                "available_history_days": n_mapping_days,
                "raw_mapping_days": raw_mapping_days,
                "history_days_available": min(days, n_mapping_days),
                "eligible_endpoints": len(valid_positions),
                "warmup_eligible_endpoints": len(warmup_valid_positions),
                "history_complete": history_complete,
                "pool_contracts_available": len(contract_details),
                "pool_contracts_invalid": len(invalid_contracts),
                "eligible_contracts": len(valid_contracts),
                "steps_per_day_min": min(contract_steps.values()),
                "steps_per_day_max": max(contract_steps.values()),
                "mapping_trailing_days_dropped": (
                    mapping_trailing_days_dropped),
                "trailing_partial_groups_dropped": trailing_partial_groups,
            },
        })
    return _recommend_from_explicit_history_plans(
        option, cases, kwargs, default_spd, baseline_case, plans)
