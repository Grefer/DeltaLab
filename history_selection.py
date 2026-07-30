# _*_ coding: utf-8 _*_
"""策略优选（历史择优）的纯逻辑层。

这里集中放置候选空间构造、任务入参校验、排名展示模型与图表模型。全部函数
都不接触 tkinter，也不读写 ``BacktestApp`` 实例状态，因此可以脱离 GUI 单独
测试。``gui_app.BacktestApp`` 只保留同名薄封装作为界面调用入口。
"""

from collections.abc import Mapping

import numpy as np

from pricing import (
    CloseToCloseStrategy, FixedTimeStrategy, HedgeBandStrategy, StrategyCase,
)
from pricing.hedge_analysis import (
    HISTORY_SELECTION_METRIC,
    LOOKBACK_DAYS,
    STRICT_LOOKBACK_SELECTION_METRIC,
)
from pricing.hedge_backtest import _infer_intraday_steps

# 历史周期的唯一 GUI 顺序与中文标签。每个周期都表示截至分析日、严格
# 连续回放的最近 L 个交易日；交易日长度继续由后端常量维护。
HISTORY_PERIOD_DEFS = (
    ("week", "近周"),
    ("month", "近月"),
    ("quarter", "近季"),
    ("half_year", "近半年"),
    ("year", "近年"),
)

# 历史择优的固定间隔候选统一用“日波动 σ 倍数”表达和执行；用户当前的
# absolute / relative / sigma 输入只在参考点换算成 σ 后加入。这样同一组候选
# 可跨标的复用，也不会把三种局部等价表达当成三套策略重复搜索。
DEFAULT_BAND_CANDIDATE_SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0)
# 按同一交易日的业务顺序排列：前一自然日夜盘 -> 次日日盘。
DEFAULT_FIXED_TIMES = "23:00,11:30,15:00"
MAX_BAND_CANDIDATES = 10
# 图表同时显示的候选上限。正确性由「加入前预演共同分段交集」单独保证，
# 这个数只管可读性：默认候选是 5 档带宽 + 可选当前带宽 + 固定时刻，8 正好
# 让一次运行的全部候选都能同屏对比。实测 7 条曲线（含基准）颜色互不相同、
# 图例两行放得下，且能看出「0.75σ 在中段回撤后拉开差距、2σ 基本贴着基准」
# ——这类判断在只画 3 条时是看不到的。再往上标记会从第 11 个开始重复。
MAX_HISTORY_CHART_CANDIDATES = 8


def normalize_lookbacks(selection=None):
    """按固定顺序把周期勾选规范化为 ``lookback -> 交易日``。"""
    known = {key for key, _label in HISTORY_PERIOD_DEFS}
    if selection is None:
        selected = known
        unknown = set()
    elif isinstance(selection, Mapping):
        unknown = set(selection).difference(known)
        selected = {
            key for key, enabled in selection.items() if bool(enabled)
        }
    else:
        if isinstance(selection, str):
            selection = (selection,)
        selected = set(selection)
        unknown = selected.difference(known)
    if unknown:
        raise ValueError(
            "未知历史分析周期: " + ", ".join(sorted(map(str, unknown))))
    ordered = {
        key: int(LOOKBACK_DAYS[key])
        for key, _label in HISTORY_PERIOD_DEFS if key in selected
    }
    if not ordered:
        raise ValueError("请至少选择一个历史分析周期。")
    return ordered


def validate_source(gs):
    """策略优选只能基于用户提供的 CSV 或 Wind 真实行情。"""
    source = gs.get("source")
    if source not in ("csv", "wind"):
        raise ValueError(
            "策略优选必须使用 CSV 或 Wind 真实历史行情；模拟路径不可用。"
            "如需比较模拟结果，请分别运行回测并保留到结果池。")


def validate_fixed_time_granularity(bt):
    """历史候选启动前只验证真实时间索引与日内粒度。"""
    timestamps = getattr(bt, "timestamps", None)
    if timestamps is None:
        raise ValueError(
            "固定时刻策略需要真实 pandas.DatetimeIndex 的日内行情。")
    try:
        import pandas as pd
        index = pd.DatetimeIndex(timestamps)
    except Exception as exc:
        raise ValueError("行情时间戳无法解析为 DatetimeIndex。") from exc
    if _infer_intraday_steps(index) <= 1:
        raise ValueError(
            "固定时刻策略仅支持真实日内行情；当前 DatetimeIndex "
            "每交易日只有一根 bar（日频）。")


def validate_payload(
        recommendations, ranking, window_results):
    """确认成功结果确实包含至少一个基于真实历史的可评估代理段。"""
    if recommendations is None:
        raise ValueError("历史择优未返回历史参考结果表。")
    if ranking is None or getattr(ranking, "empty", True):
        raise ValueError("历史择优未返回诊断排名。")
    if not isinstance(window_results, dict) or not window_results:
        raise ValueError("历史择优未返回严格区间代理段明细。")

    try:
        import pandas as pd
        rolling_windows = pd.to_numeric(
            ranking["rolling_windows"], errors="coerce").to_numpy(dtype=float)
        scores = pd.to_numeric(
            ranking["score"], errors="coerce").to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("历史择优排名缺少有效样本或评分字段。") from exc
    if not np.any((rolling_windows > 0) & np.isfinite(scores)):
        # fixed_times 的逐代理段失败已由历史分析层保留原始原因。若全部
        # 代理段都因目标时刻缺失而失败，应把真实数据问题直接反馈给
        # 用户，而不是误报成历史区间长度不足。
        # 两组失败原因必须在同一个 try 里取：它们共用 failure_scopes /
        # failure_reasons。旧快照或第三方调用缺这两列时，第一个 try 会在
        # 赋值前就抛 KeyError，随后引用同名变量拿到的是 UnboundLocalError
        # ——它绕过下面的 except，最终以 traceback 弹窗替掉了本该给出的
        # “历史区间不足”提示。
        fixed_failures = []
        endpoint_failures = []
        try:
            strategy_types = ranking["strategy_type"].astype(str)
            failure_scopes = ranking["failure_scope"].astype(str)
            failure_reasons = ranking["failure_reason"].fillna("").astype(str)
            has_reason = failure_reasons.str.strip().ne("")
            fixed_failures = failure_reasons[
                strategy_types.eq("fixed_times")
                & failure_scopes.eq("strategy")
                & has_reason
            ]
            endpoint_failures = failure_reasons[
                failure_scopes.eq("endpoint") & has_reason
            ]
        except (KeyError, AttributeError, TypeError, ValueError):
            fixed_failures = []
            endpoint_failures = []
        if len(fixed_failures):
            first_failure = str(fixed_failures.iloc[0])
            raise ValueError(
                "固定时刻策略没有形成任何可评估代理段；"
                f"首个失败原因：{first_failure} "
                "请确认行情粒度以及各交易日的 Bar 时间戳与设定时刻一致。"
            )
        if len(endpoint_failures):
            first_failure = str(endpoint_failures.iloc[0])
            raise ValueError(
                "历史具体合约池没有形成任何可评估代理段；"
                f"首个失败原因：{first_failure}"
            )
        raise ValueError(
            "真实历史长度不足，尚未形成任何可评估严格区间；"
            "请扩大 CSV/Wind 历史区间后重试。")


def parse_band_candidate_sigmas(raw_values):
    """解析策略优选的日波动率倍数候选，并按显示精度去重排序。"""
    if raw_values is None:
        raw_values = DEFAULT_BAND_CANDIDATE_SIGMAS
    if isinstance(raw_values, str):
        normalized = raw_values.translate(str.maketrans({
            "，": ",", "；": ",", ";": ",", "\n": ",",
        }))
        tokens = [token.strip() for token in normalized.split(",")]
        tokens = [token for token in tokens if token]
    else:
        try:
            tokens = list(raw_values)
        except TypeError as exc:
            raise ValueError("策略优选带宽候选必须是逗号分隔数值。") from exc

    candidates_by_key = {}
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无效的策略优选波动率倍数候选: {token!r}") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError("策略优选波动率倍数候选必须全部是大于 0 的有限数值。")
        # 名称也使用 .10g，先用同一规范键去重可从源头保证 case.name 唯一。
        key = f"{value:.10g}"
        candidates_by_key[key] = float(key)
    if len(candidates_by_key) > MAX_BAND_CANDIDATES:
        raise ValueError(
            f"策略优选固定间隔候选最多 {MAX_BAND_CANDIDATES} 档。")
    return tuple(sorted(candidates_by_key.values()))


def band_cases(gs):
    """把历史页的 σ 档位与可选当前带宽归一化为候选案例。"""
    band_type = gs.get("interval_type", "absolute")
    threshold = float(gs.get("price_interval", 1.0))
    force_day_close_hedge = bool(
        gs.get("force_day_close_hedge", False))
    params = gs.get("params", {})
    reference_price = float(params["s0"])
    sigma_annual = float(params["sigma"])
    include_current = bool(
        gs.get("history_include_current_band", True))
    current_equivalents = None
    current_key = None
    if include_current:
        current_equivalents = HedgeBandStrategy.convert_threshold(
            threshold, band_type, reference_price, sigma_annual)
        current_key = f"{current_equivalents['sigma']:.10g}"

    preset_values = parse_band_candidate_sigmas(
        gs.get("band_candidate_sigmas"))
    preset_keys = {f"{value:.10g}" for value in preset_values}
    candidates_by_key = {
        f"{value:.10g}": value for value in preset_values
    }
    # 当前单次回测带宽只在历史页明确勾选时参与。
    if current_key is not None:
        candidates_by_key[current_key] = float(current_key)
    if len(candidates_by_key) > MAX_BAND_CANDIDATES:
        raise ValueError(
            f"策略优选固定间隔候选最多 {MAX_BAND_CANDIDATES} 档"
            "（包含勾选加入的当前带宽）。")

    cases = []
    for candidate_sigma in sorted(candidates_by_key.values()):
        key = f"{candidate_sigma:.10g}"
        is_current = current_key is not None and key == current_key
        equivalents = HedgeBandStrategy.convert_threshold(
            candidate_sigma, "sigma", reference_price, sigma_annual)
        origin = (
            "当前输入 / 常用候选" if is_current and key in preset_keys
            else "当前输入换算" if is_current else "常用候选"
        )
        marker = "·当前" if is_current else ""
        description = (
            f"{origin}；期初等价绝对 {equivalents['absolute']:.6g} / "
            f"相对 {equivalents['relative']:.4%} / "
            f"{candidate_sigma:.6g} 倍波动率；收盘保底"
            f"{'开启' if force_day_close_hedge else '关闭'}"
        )
        cases.append(StrategyCase(
            f"固定间隔({key}σ{marker})",
            HedgeBandStrategy(
                band_type="sigma",
                threshold=candidate_sigma,
                sigma_source=gs.get("sigma_source", "implied"),
                window_days=gs.get("sigma_window", 20),
            ),
            {
                "description": description,
                "strategy_name": "hedge_band",
                "candidate_origin": origin,
                "candidate_sigma": candidate_sigma,
                "equivalent_absolute": equivalents["absolute"],
                "equivalent_relative": equivalents["relative"],
                "input_band_type": band_type,
                "input_threshold": threshold,
                "is_current_band": is_current,
                "sigma_source": gs.get("sigma_source", "implied"),
                "sigma_window": gs.get("sigma_window", 20),
                "force_day_close_hedge": force_day_close_hedge,
            },
        ))
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("策略优选固定间隔候选名称重复。")
    return cases


def strategy_cases(gs, base_bt):
    """根据历史页独立开关生成候选策略集。"""
    # close-to-close 始终作为历史图表和严格区间改善的固定基准，不允许
    # 通过旧状态或直接 API 调用移除。
    cases = [StrategyCase(
        "每日收盘", CloseToCloseStrategy(),
        {
            "description": (
                "按真实交易日的最后一个采样点调仓；固定比较基准；收盘保底"
                f"{'开启' if gs.get('force_day_close_hedge', False) else '关闭'}"
            ),
            "strategy_name": "close_to_close",
            "is_history_baseline": True,
            "force_day_close_hedge": bool(
                gs.get("force_day_close_hedge", False)),
        },
    )]
    if gs.get("history_include_band", True):
        cases.extend(band_cases(gs))
    skipped = []
    if gs.get("history_include_fixed_times", True):
        try:
            if gs.get("source") == "simulate":
                raise ValueError("模拟路径没有真实时刻")
            if (gs.get("source") == "wind" and
                    gs.get("wind_bar_size", "日频") == "日频"):
                raise ValueError("Wind 日频没有日内时刻")
            fixed = FixedTimeStrategy(
                gs.get("fixed_times", DEFAULT_FIXED_TIMES))
            if gs.get("source") == "wind":
                from pricing.wind_data import (
                    get_trading_session_clock_ranges,
                )
                fixed.set_trading_sessions(
                    get_trading_session_clock_ranges(
                        gs.get("wind_code", "")))
            # 此处只确认候选具备真实日内时间轴。具体目标时刻是否在
            # 每个交易日出现，必须由严格历史区间中的代理段逐一判断；
            # base_bt 只提供粒度上下文，不能据此全局剔除。
            validate_fixed_time_granularity(base_bt)
            requested_label = ",".join(
                t.strftime("%H:%M") for t in fixed.requested_times)
            effective_label = ",".join(
                t.strftime("%H:%M") for t in fixed.effective_times)
            skipped_label = ",".join(
                t.strftime("%H:%M") for t in fixed.skipped_times)
            execution_text = (
                f"每日 {effective_label} 调仓"
                if effective_label else "所选时刻均不在该品种交易时段")
            if skipped_label:
                execution_text += f"；自动跳过非交易时刻 {skipped_label}"
            cases.append(StrategyCase(
                f"固定时刻({requested_label})", fixed,
                {
                    "description": (
                        f"{execution_text}；收盘保底"
                        f"{'开启' if gs.get('force_day_close_hedge', False) else '关闭'}"
                    ),
                    "strategy_name": "fixed_times",
                    "fixed_times": requested_label,
                    "fixed_times_requested": requested_label,
                    "fixed_times_effective": effective_label,
                    "fixed_times_skipped": skipped_label,
                    "force_day_close_hedge": bool(
                        gs.get("force_day_close_hedge", False)),
                },
            ))
        except (TypeError, ValueError) as exc:
            skipped.append(f"固定时刻策略未参与：{exc}")
    if len(cases) == 1:
        reason = f"（{'；'.join(skipped)}）" if skipped else ""
        raise ValueError(
            "策略优选只有每日收盘基准，没有可执行的候选策略"
            f"{reason}")
    return cases, skipped


def finite_value(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def safe_int(value, default=0):
    number = finite_value(value)
    if number is None or not number.is_integer():
        return int(default)
    return int(number)


def safe_bool(value, default=False):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(default)


def ranking_baseline(summary):
    """从对比摘要中稳定选择显式标记的 close-to-close 基准。"""
    if summary is None or getattr(summary, "empty", True):
        return None
    explicit = []
    close_rows = []
    for _index, row in summary.iterrows():
        item = row.to_dict()
        if item.get("strategy_type") != "close_to_close":
            continue
        close_rows.append(item)
        if safe_bool(
                item.get("meta_is_comparison_baseline"), False):
            explicit.append(item)

    candidates = explicit or close_rows
    if not candidates:
        return None

    def _stable_key(item):
        result_id = item.get("meta_result_id")
        if isinstance(result_id, str) and result_id:
            return 0, result_id
        rank = finite_value(item.get("rank"))
        return 1, np.inf if rank is None else rank, str(
            item.get("strategy", ""))

    return min(candidates, key=_stable_key)


def row_improvement(row, baseline=None):
    """读取历史择优主指标；旧结果回退到合并 RMS 改善。"""
    if not row:
        return None
    # 新版严格区间与旧版逐窗等权结果都有各自正式的 selection 指标；
    # 二者都优先读取该指标，但只有新版常量可以驱动“严格 L 日”文案。
    if row_uses_recognized_metric(row):
        return finite_value(
            row.get("selection_improvement_vs_c2c"))
    value = finite_value(
        row.get("improvement_vs_c2c"))
    if value is not None:
        return value
    if str(row.get("strategy_type", "")) == "close_to_close":
        return 0.0 if finite_value(
            row.get("score")) is not None else None
    candidate_score = finite_value(row.get("score"))
    baseline_score = finite_value(
        row.get("baseline_score"))
    if baseline_score is None and baseline:
        baseline_score = finite_value(
            baseline.get("score"))
    if (candidate_score is None or baseline_score is None
            or np.isclose(baseline_score, 0.0, rtol=1e-12, atol=1e-12)):
        return None
    return (baseline_score - candidate_score) / abs(baseline_score)


def row_uses_recognized_metric(row):
    """是否携带当前 GUI 能解释的新版或旧版选择指标。"""
    return bool(
        row
        and str(row.get("selection_metric", "")) in {
            HISTORY_SELECTION_METRIC,
            STRICT_LOOKBACK_SELECTION_METRIC,
        }
        and "selection_improvement_vs_c2c" in row)


def row_uses_strict_metric(row):
    """只有严格连续 L 日选择指标才允许展示 strict 口径文案。"""
    return bool(
        row
        and str(row.get("selection_metric", ""))
        == STRICT_LOOKBACK_SELECTION_METRIC
        and "selection_improvement_vs_c2c" in row)


def row_uses_window_equal_metric(row):
    """识别升级前的逐历史终点等权选择指标。"""
    return bool(
        row
        and str(row.get("selection_metric", "")) == HISTORY_SELECTION_METRIC
        and "selection_improvement_vs_c2c" in row)


def ranking_flags(ranking):
    """一次性判定排名表的口径与候选集合，供展示层统一取用。

    表头、说明文案与图表配色此前各自重复推导同一批列判断；集中在这里可
    保证同一份排名不会在三处得出不同口径。
    """
    empty = ranking is None or getattr(ranking, "empty", True)
    has_selection = bool(
        not empty
        and "selection_metric" in ranking
        and "selection_improvement_vs_c2c" in ranking)
    if has_selection:
        metrics = ranking["selection_metric"].fillna("").astype(str)
        uses_strict_metric = bool(
            metrics.eq(STRICT_LOOKBACK_SELECTION_METRIC).all())
        uses_window_equal_metric = bool(
            metrics.eq(HISTORY_SELECTION_METRIC).all())
    else:
        uses_strict_metric = False
        uses_window_equal_metric = False
    uses_product_pool = bool(
        not empty
        and "history_mode" in ranking
        and ranking["history_mode"].fillna("").astype(str).eq(
            "product_contract_pool").any())
    candidate_names = []
    if not empty and {"strategy", "strategy_type"}.issubset(ranking.columns):
        candidate_names = sorted(set(
            ranking.loc[
                ~ranking["strategy_type"].astype(str).eq("close_to_close"),
                "strategy",
            ].dropna().astype(str)
        ))
    return {
        "uses_strict_metric": uses_strict_metric,
        "uses_window_equal_metric": uses_window_equal_metric,
        "uses_product_pool": uses_product_pool,
        "candidate_names": candidate_names,
    }


def contract_codes_text(values, *, limit=6):
    """把历史评分实际参与合约压缩成适合详情栏的文本。"""
    if isinstance(values, str):
        values = (values,)
    try:
        codes = tuple(dict.fromkeys(
            str(code).strip() for code in values
            if str(code).strip() and str(code).strip() != "nan"))
    except TypeError:
        return ""
    shown = codes[:max(1, int(limit))]
    result = "、".join(shown)
    if len(codes) > len(shown):
        result += f" 等{len(codes)}个"
    return result


def recommendation_rows(
        recommendations, ranking, lookbacks=None):
    """按固定 每日收盘 基准整理本次已选周期的历史参考展示模型。"""
    selected = normalize_lookbacks(lookbacks)
    period_defs = tuple(
        (key, label) for key, label in HISTORY_PERIOD_DEFS
        if key in selected)

    def _subset(frame, key):
        if frame is None or getattr(frame, "empty", True):
            return None
        selected = frame[frame["lookback"] == key]
        return selected if not selected.empty else None

    def _sampling_context(item):
        item = item or {}
        segment_count = item.get(
            "segment_count", item.get("proxy_segments"))
        return {
            "maturity_days": item.get("maturity_days"),
            "evaluation_mode": item.get("evaluation_mode"),
            "evidence_days": safe_int(
                item.get("evidence_days"), 0),
            "days_used": safe_int(
                item.get("days_used"), 0),
            "segment_count": (
                safe_int(segment_count, 0)
                if segment_count is not None else None),
            "expiry_segments": safe_int(
                item.get("expiry_segments"), 0),
            "mtm_segments": safe_int(
                item.get("mtm_segments"), 0),
            "terminal_mode": item.get("terminal_mode"),
            "realized_sigma_warmup_days": (
                safe_int(
                    item.get("realized_sigma_warmup_days"), 0)),
            "warmup_eligible_endpoints": (
                safe_int(
                    item.get("warmup_eligible_endpoints"), 0)),
            "sampling_mode": item.get("sampling_mode", "fixed_step"),
            "step_days": item.get("step_days"),
            "required_history_days": safe_int(
                item.get("required_history_days"), 0),
            "available_history_days": safe_int(
                item.get("available_history_days"), 0),
            "history_complete": safe_bool(
                item.get("history_complete"), False),
            "history_mode": item.get("history_mode"),
            "product_code": item.get("product_code"),
            "main_contract_asof": item.get("main_contract_asof"),
            "effective_asof_date": item.get("effective_asof_date"),
            "effective_main_contract": item.get(
                "effective_main_contract"),
            "pool_contracts_available": safe_int(
                item.get("pool_contracts_available"), 0),
            "pool_contracts_invalid": safe_int(
                item.get("pool_contracts_invalid"), 0),
            "mapping_trailing_days_dropped": (
                safe_int(
                    item.get("mapping_trailing_days_dropped"), 0)),
            "selection_metric": item.get("selection_metric"),
            "uses_strict_metric": (
                row_uses_strict_metric(item)),
            "uses_window_equal_metric": (
                row_uses_window_equal_metric(item)),
            "relative_comparison_windows": (
                safe_int(
                    item.get("relative_comparison_windows"), 0)
                if "relative_comparison_windows" in item else None),
            "paired_contract_codes": item.get(
                "paired_contract_codes", ()),
        }

    rows = []
    for key, display in period_defs:
        rec_group = _subset(recommendations, key)
        rank_group = _subset(ranking, key)
        formal_row = (
            rec_group.iloc[0].to_dict()
            if rec_group is not None else None)
        baseline = ranking_baseline(rank_group)
        context_row = baseline
        if context_row is None and rank_group is not None:
            context_row = rank_group.iloc[0].to_dict()

        comparable_candidates = []
        if rank_group is not None:
            for _index, rank_row in rank_group.iterrows():
                item = rank_row.to_dict()
                if str(item.get("strategy_type", "")) == "close_to_close":
                    continue
                # 新结果显式携带 comparison_eligible；旧快照仅在完整行
                # 上回退为可比，避免把部分回测分段结果包装成诊断冠军。
                if "comparison_eligible" in item:
                    comparable = safe_bool(
                        item.get("comparison_eligible"), False)
                else:
                    comparable = safe_bool(
                        item.get("complete_window"), False)
                if comparable:
                    comparable_candidates.append(item)

        has_comparable_candidate = bool(comparable_candidates)
        if formal_row is not None:
            leader = formal_row
        elif has_comparable_candidate and rank_group is not None:
            comparable_names = {
                str(item.get("strategy", ""))
                for item in comparable_candidates
            }
            eligible_group = rank_group[
                rank_group["strategy"].astype(str).isin(comparable_names)
                | rank_group["strategy_type"].astype(str).eq(
                    "close_to_close")
            ]
            leader = eligible_group.sort_values(
                ["rank", "score", "strategy"],
                kind="stable").iloc[0].to_dict()
        else:
            leader = baseline

        leader_score = (
            finite_value(leader.get("score"))
            if leader is not None else None)
        leader_effective = (
            safe_int(
                leader.get("rolling_windows"), 0)
            if leader is not None else 0)
        if leader_score is None or leader_effective <= 0:
            leader = None

        if leader is None:
            context = _sampling_context(context_row)
            rows.append({
                "lookback": key, "period": display, "strategy": "—",
                "strategy_label": "—", "score": None,
                "baseline_score": None, "improvement_vs_c2c": None,
                "selection_improvement_vs_c2c": None,
                "incremental_pnl": None, "incremental_sharpe": None,
                "incremental_tc": None, "max_drawdown": None,
                "gap_ratio": None, "window_win_rate_vs_c2c": None,
                "paired": 0, "baseline_windows": 0,
                "effective": 0,
                "eligible": safe_int(
                    (context_row or {}).get("eligible_endpoints"), 0),
                "skipped": safe_int(
                    (context_row or {}).get("skipped_endpoints"), 0),
                "available": safe_int(
                    (context_row or {}).get(
                        "history_days_available", 0), 0),
                "requested": safe_int(
                    (context_row or {}).get("lookback_days", 0), 0),
                "status": "无可评估回测分段", "formal": False,
                "best_is_baseline": False,
                "has_comparable_candidate": False,
                "trailing_dropped": safe_int(
                    (context_row or {}).get(
                        "trailing_partial_groups_dropped"), 0),
                **context,
            })
            continue

        formal = formal_row is not None
        strategy = str(leader.get("strategy", "—"))
        best_is_baseline = (
            str(leader.get("strategy_type", "")) == "close_to_close")
        baseline_score = finite_value(
            leader.get("baseline_score"))
        if baseline_score is None and baseline:
            baseline_score = finite_value(
                baseline.get("score"))
        improvement = row_improvement(
            leader, baseline)
        uses_strict_metric = (
            row_uses_strict_metric(leader))
        paired = safe_int(
            leader.get("paired_windows", leader.get("rolling_windows")), 0)
        baseline_windows = safe_int(
            leader.get("baseline_windows", paired), paired)
        win_rate = finite_value(
            leader.get("window_win_rate_vs_c2c"))
        if not has_comparable_candidate:
            strategy_label = "仅基准（无可比候选）"
            status = "无可比候选"
        elif best_is_baseline:
            strategy_label = "每日收盘（基准最优）"
            if not formal:
                strategy_label = f"诊断：{strategy_label}"
            if uses_strict_metric:
                status = "数据完整" if formal else "数据不足（仅参考）"
            else:
                status = "旧版数据完整" if formal else "旧版数据不足（仅参考）"
        else:
            strategy_label = strategy if formal else f"诊断领先：{strategy}"
            if uses_strict_metric:
                status = "数据完整" if formal else "数据不足（仅参考）"
            else:
                status = "旧版数据完整" if formal else "旧版数据不足（仅参考）"
        rows.append({
            "lookback": key,
            "period": display,
            "strategy": strategy,
            "strategy_label": strategy_label,
            "score": leader_score,
            "baseline_score": baseline_score,
            "improvement_vs_c2c": improvement,
            "selection_improvement_vs_c2c": (
                improvement if uses_strict_metric else None),
            # 两个排名口径的增量值，供周期结论表并排展示。
            "incremental_pnl": finite_value(
                leader.get("incremental_pnl_vs_c2c")),
            "incremental_sharpe": finite_value(
                leader.get("incremental_sharpe_vs_c2c")),
            "incremental_tc": finite_value(
                leader.get("incremental_tc_vs_c2c")),
            "max_drawdown": finite_value(leader.get("max_drawdown")),
            # 旧展示模型字段保留同值软兼容；其语义已固定为较 每日收盘 改善。
            "gap_ratio": improvement,
            "window_win_rate_vs_c2c": win_rate,
            "paired": paired,
            "baseline_windows": baseline_windows,
            "effective": paired or leader_effective,
            "eligible": safe_int(
                leader.get("eligible_endpoints"), 0),
            "skipped": safe_int(
                leader.get("skipped_endpoints"), 0),
            "available": safe_int(leader.get(
                "history_days_available", leader.get("days_used")), 0),
            "requested": safe_int(
                leader.get("lookback_days"), 0),
            "status": status,
            "formal": formal,
            "best_is_baseline": best_is_baseline,
            "has_comparable_candidate": has_comparable_candidate,
            "trailing_dropped": safe_int(
                leader.get("trailing_partial_groups_dropped"), 0),
            **_sampling_context(leader),
        })
    return rows


def chart_pairs(summary, lookback, strategy):
    """按 ``(lookback, window_id)`` 配对候选与固定 每日收盘 基准。"""
    import pandas as pd

    if summary is None or getattr(summary, "empty", True) or not strategy:
        return pd.DataFrame()
    required = {
        "lookback", "window_id", "strategy", "strategy_type", "success",
    }
    if not required.issubset(summary.columns):
        return pd.DataFrame()
    rows = summary[summary["lookback"].astype(str) == str(lookback)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows = rows[rows["success"].fillna(False).astype(bool)]
    baseline = rows[rows["strategy_type"].astype(str) == "close_to_close"]
    candidate = rows[rows["strategy"].astype(str) == str(strategy)]
    if baseline.empty or candidate.empty:
        return pd.DataFrame()
    baseline = baseline.drop_duplicates("window_id", keep="first")
    candidate = candidate.drop_duplicates("window_id", keep="first")
    pairs = candidate.merge(
        baseline, on=["lookback", "window_id"], how="inner",
        suffixes=("_candidate", "_baseline"), validate="one_to_one",
    )
    if pairs.empty:
        return pairs
    # 只保留段内交易日数完全一致的共同回测分段；图表不得静默截断。
    def _daily_length(value):
        try:
            return len(np.asarray(value, dtype=float).reshape(-1))
        except (TypeError, ValueError):
            return -1

    candidate_daily = pairs.get("daily_net_pnl_candidate")
    baseline_daily = pairs.get("daily_net_pnl_baseline")
    if candidate_daily is None or baseline_daily is None:
        return pairs.iloc[:0].copy()
    same_length = np.fromiter((
        _daily_length(candidate_value) > 0
        and _daily_length(candidate_value) == _daily_length(baseline_value)
        for candidate_value, baseline_value in zip(
            candidate_daily, baseline_daily)
    ), dtype=bool, count=len(pairs))
    pairs = pairs.loc[same_length].copy()
    if pairs.empty:
        return pairs
    start_values = pairs.get("start_ts_candidate")
    if start_values is None:
        start_values = pairs.get("start_ts_baseline")
    end_values = pairs.get("end_ts_candidate")
    if end_values is None:
        end_values = pairs.get("end_ts_baseline")
    pairs = pairs.assign(
        _chart_start_ts=pd.to_datetime(start_values, errors="coerce"),
        _chart_end_ts=pd.to_datetime(end_values, errors="coerce"))
    return pairs.sort_values(
        ["_chart_start_ts", "_chart_end_ts", "window_id"], kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def chart_array(row, column, role):
    """安全读取 summary 单元格中的一维曲线数组。"""
    value = row.get(f"{column}_{role}")
    if value is None:
        return np.array([], dtype=float)
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return (
        array if array.size and np.any(np.isfinite(array))
        else np.array([], dtype=float)
    )


def chart_band(arrays):
    """按回测分段内交易日对齐累计曲线并返回中位数与 P25/P75。

    提前敲出/结算后的累计值保持不变直至最长回测分段，避免后半段只剩
    未敲出回测分段而产生幸存者偏差。调用方只向这里传累计曲线。
    """
    arrays = [np.asarray(array, dtype=float).reshape(-1)
              for array in arrays if len(array)]
    if not arrays:
        return None
    width = max(len(array) for array in arrays)
    matrix = np.full((len(arrays), width), np.nan, dtype=float)
    for row_no, array in enumerate(arrays):
        matrix[row_no, :len(array)] = array
        if len(array) < width and np.isfinite(array[-1]):
            matrix[row_no, len(array):] = array[-1]
    valid = np.any(np.isfinite(matrix), axis=0)
    if not np.any(valid):
        return None
    matrix = matrix[:, valid]
    return {
        "x": np.flatnonzero(valid).astype(float) + 1.0,
        "median": np.nanmedian(matrix, axis=0),
        "p25": np.nanpercentile(matrix, 25, axis=0),
        "p75": np.nanpercentile(matrix, 75, axis=0),
        "window_count": len(arrays),
        "show_interval": len(arrays) >= 2,
    }


def chart_model(
        summary, lookback, strategy, mode="single", metric="net",
        window_id=None, normalized=False):
    """生成历史图表的纯数据模型，不依赖 Tk，便于配对与空态测试。"""
    raw_metric_columns = {
        "net": ("cumulative_net_pnl", "累计净损益"),
        "gross": ("cumulative_gross_pnl", "累计成本前损益"),
        "tc": ("cumulative_tc", "累计成本"),
    }
    normalized_metric_columns = {
        "net": ("normalized_cumulative_net_pnl", "累计净损益 / 期初名义金额"),
        "gross": (
            "normalized_cumulative_gross_pnl",
            "累计成本前损益 / 期初名义金额",
        ),
        "tc": ("normalized_cumulative_tc", "累计成本 / 期初名义金额"),
    }
    metric_columns = (
        normalized_metric_columns if normalized else raw_metric_columns)
    import pandas as pd
    mode = str(mode or "single")
    metric = str(metric or "net")
    if mode not in {"single", "typical"}:
        return {"state": "empty", "message": "未知历史图表模式。", "mode": mode}
    if metric not in metric_columns:
        return {"state": "empty", "message": "未知历史图表指标。", "mode": mode}
    if not strategy:
        return {"state": "empty", "message": "请先选择一条历史策略。", "mode": mode}

    pairs = chart_pairs(
        summary, lookback, strategy)
    if pairs.empty:
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": "所选周期没有可与每日收盘基准配对的成功回测分段。",
            "window_options": [],
        }
    window_options = []
    for sample_no, (_index, row) in enumerate(pairs.iterrows(), start=1):
        end_ts = row.get("_chart_end_ts")
        date_text = (
            end_ts.strftime("%Y-%m-%d")
            if end_ts is not None and not getattr(end_ts, "isnat", False)
            and not pd.isna(end_ts)
            else "日期未知"
        )
        window_options.append({
            "window_id": str(row["window_id"]),
            "end_ts": end_ts,
            "label": f"第 {sample_no} 段 · {date_text}",
        })
    base = {
        "state": "ok", "mode": mode, "metric": metric,
        "lookback": str(lookback), "strategy": str(strategy),
        "window_options": window_options,
        "uses_normalized_notional": bool(normalized),
    }

    curve_column, metric_label = metric_columns[metric]
    selected_is_baseline = bool(
        pairs["strategy_type_candidate"].astype(str)
        .eq("close_to_close").all())
    roles = [("baseline", "每日收盘（固定基准）")]
    if not selected_is_baseline:
        roles.append(("candidate", str(strategy)))

    if mode == "single":
        option_ids = [item["window_id"] for item in window_options]
        selected_window = (
            str(window_id) if window_id is not None
            and str(window_id) in option_ids else option_ids[-1])
        row = pairs[pairs["window_id"].astype(str) == selected_window].iloc[0]
        series = []
        for role, label in roles:
            values = chart_array(
                row, curve_column, role)
            if not len(values):
                return {
                    **base, "state": "empty",
                    "message": f"配对回测分段缺少{metric_label}曲线。",
                    "selected_window_id": selected_window,
                }
            series.append({
                "role": role, "label": label,
                "x": np.arange(1, len(values) + 1, dtype=float),
                "y": values,
            })
        if len({len(item["y"]) for item in series}) != 1:
            return {
                **base, "state": "empty",
                "message": "配对回测分段内交易日数不一致，无法绘图。",
                "selected_window_id": selected_window,
            }
        return {
            **base, "selected_window_id": selected_window,
            "series": series, "metric_label": metric_label,
            "title": f"单回测分段路径 · {metric_label}",
        }

    bands = []
    for role, label in roles:
        arrays = [
            chart_array(row, curve_column, role)
            for _index, row in pairs.iterrows()
        ]
        band = chart_band(arrays)
        if band is None:
            return {
                **base, "state": "empty",
                "message": f"配对回测分段缺少{metric_label}曲线。",
            }
        bands.append({"role": role, "label": label, **band})
    return {
        **base, "bands": bands, "metric_label": metric_label,
        "title": (
            f"多回测分段中位路径 · {metric_label}"
            "（中位数；P25/P75 为描述区间，非置信区间）"
        ),
    }


def multi_chart_model(
        summary, lookback, strategies, mode="single", metric="net",
        window_id=None, primary_strategy=None, pairs_cache=None):
    """生成固定 每日收盘 基准加多个候选的严格配对回测分段图表模型。

    ``pairs_cache`` 是可选的 ``{(lookback, strategy): pairs}`` 字典。候选逐个
    试探与每次控件变化都会对同一批策略反复求同一份配对，传入跨调用复用的
    缓存可把重复 merge 降到每个策略一次。缓存只对同一个 ``summary`` 有效，
    调用方需要在结果页重建时丢弃它。
    """
    import pandas as pd

    if isinstance(strategies, str):
        strategies = [strategies]
    try:
        strategies = list(dict.fromkeys(
            str(value).strip() for value in (strategies or ())
            if str(value).strip()))
    except TypeError:
        strategies = []
    mode = str(mode or "single")
    metric = str(metric or "net")
    if mode not in {"full", "single", "typical"}:
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": "未知历史图表模式。", "window_options": [],
            "strategies": strategies,
        }
    if metric not in {"net", "gross", "tc"}:
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": "未知历史图表指标。", "window_options": [],
            "strategies": strategies,
        }
    required = {
        "lookback", "window_id", "strategy", "strategy_type", "success",
    }
    if (summary is None or getattr(summary, "empty", True)
            or not required.issubset(summary.columns)):
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": "所选周期没有可绘制的历史回测分段。",
            "window_options": [], "strategies": strategies,
        }

    period_rows = summary[
        summary["lookback"].astype(str) == str(lookback)].copy()
    if period_rows.empty:
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": "所选周期没有可绘制的历史回测分段。",
            "window_options": [], "strategies": strategies,
        }
    successful = period_rows[
        period_rows["success"].fillna(False).astype(bool)]
    baseline_rows = successful[
        successful["strategy_type"].astype(str).eq("close_to_close")]
    if baseline_rows.empty:
        return {
            "state": "empty", "mode": str(mode), "metric": str(metric),
            "message": "所选周期缺少每日收盘基准。",
            "window_options": [], "strategies": strategies,
        }
    baseline_strategy = str(baseline_rows.iloc[0]["strategy"])
    baseline_strategies = set(
        baseline_rows["strategy"].dropna().astype(str))
    candidates = [
        value for value in strategies if value not in baseline_strategies]
    primary_strategy = str(primary_strategy or "").strip()
    if primary_strategy not in candidates:
        primary_strategy = candidates[0] if candidates else baseline_strategy
    comparison_strategies = candidates or [baseline_strategy]

    contract_codes = set()
    if "history_contract_code" in successful.columns:
        contract_codes = {
            str(value).strip()
            for value in successful["history_contract_code"].dropna()
            if str(value).strip() and str(value).strip() != "nan"
        }
    uses_normalized_notional = (
        mode in {"full", "typical"} and len(contract_codes) > 1)

    if pairs_cache is None:
        pairs_cache = {}
    pairs_by_strategy = {}
    for strategy in comparison_strategies:
        key = (str(lookback), strategy)
        if key not in pairs_cache:
            pairs_cache[key] = chart_pairs(summary, lookback, strategy)
        # 下游只在过滤前 copy()，缓存帧本身不会被就地修改。
        pairs_by_strategy[strategy] = pairs_cache[key]
    missing = [
        strategy for strategy, pairs in pairs_by_strategy.items()
        if pairs.empty]
    if missing:
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": (
                "以下策略没有可与 每日收盘 严格配对的回测分段："
                + "、".join(missing)),
            "window_options": [], "strategies": candidates,
            "baseline_strategy": baseline_strategy,
        }

    raw_metric_columns = {
        "net": "cumulative_net_pnl",
        "gross": "cumulative_gross_pnl",
        "tc": "cumulative_tc",
    }
    normalized_metric_columns = {
        "net": "normalized_cumulative_net_pnl",
        "gross": "normalized_cumulative_gross_pnl",
        "tc": "normalized_cumulative_tc",
    }
    raw_daily_metric_columns = {
        "net": "daily_net_pnl",
        "gross": "daily_gross_pnl",
        "tc": "daily_tc",
    }
    normalized_daily_metric_columns = {
        "net": "normalized_daily_net_pnl",
        "gross": "normalized_daily_gross_pnl",
        "tc": "normalized_daily_tc",
    }
    metric_columns = (
        (normalized_daily_metric_columns
         if uses_normalized_notional else raw_daily_metric_columns)
        if mode == "full" else
        (normalized_metric_columns
         if uses_normalized_notional else raw_metric_columns)
    )
    required_curve_column = metric_columns[metric]
    if (uses_normalized_notional
            and required_curve_column not in successful.columns):
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": (
                "跨合约历史路径缺少安全归一化数据；"
                "请改用单回测分段路径查看原始金额。"),
            "window_options": [], "strategies": candidates,
            "baseline_strategy": baseline_strategy,
            "uses_normalized_notional": True,
        }

    def _valid_window_ids(pairs):
        valid_ids = set()
        curve_column = metric_columns[metric]
        for _index, row in pairs.iterrows():
            candidate_curve = chart_array(
                row, curve_column, "candidate")
            baseline_curve = chart_array(
                row, curve_column, "baseline")
            if (not len(candidate_curve)
                    or len(candidate_curve) != len(baseline_curve)
                    or not np.all(np.isfinite(candidate_curve))
                    or not np.all(np.isfinite(baseline_curve))):
                continue
            valid_ids.add(str(row["window_id"]))
        return valid_ids

    valid_ids_by_strategy = {
        strategy: _valid_window_ids(pairs)
        for strategy, pairs in pairs_by_strategy.items()
    }
    common_ids = None
    for window_ids in valid_ids_by_strategy.values():
        common_ids = (
            window_ids if common_ids is None
            else common_ids.intersection(window_ids))
    common_ids = common_ids or set()
    first_pairs = next(iter(pairs_by_strategy.values()))
    ordered_ids = [
        str(value) for value in first_pairs["window_id"]
        if str(value) in common_ids]
    if not ordered_ids:
        return {
            "state": "empty", "mode": mode, "metric": metric,
            "message": (
                "跨合约历史路径没有可安全归一化的共同回测分段；"
                "请改用单回测分段路径查看原始金额。"
                if uses_normalized_notional else
                "所选候选之间没有共同的严格配对回测分段。"),
            "window_options": [], "strategies": candidates,
            "baseline_strategy": baseline_strategy,
            "uses_normalized_notional": uses_normalized_notional,
        }

    selected_period = summary["lookback"].astype(str).eq(str(lookback))
    selected_windows = summary["window_id"].astype(str).isin(ordered_ids)
    filtered = summary.loc[~selected_period | selected_windows].copy()
    option_pairs = first_pairs[
        first_pairs["window_id"].astype(str).isin(ordered_ids)]
    option_pairs = option_pairs.set_index(
        option_pairs["window_id"].astype(str), drop=False)
    window_options = []
    for sample_no, common_id in enumerate(ordered_ids, start=1):
        row = option_pairs.loc[common_id]
        if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        end_ts = row.get("_chart_end_ts")
        date_text = (
            end_ts.strftime("%Y-%m-%d")
            if end_ts is not None and not pd.isna(end_ts)
            else "日期未知")
        contract_code = str(
            row.get("history_contract_code_candidate", "") or ""
        ).strip()
        contract_text = f" · {contract_code}" if contract_code else ""
        window_options.append({
            "window_id": common_id,
            "end_ts": end_ts,
            "label": f"第 {sample_no} 段 · {date_text}{contract_text}",
        })

    base = {
        "state": "ok", "mode": mode, "metric": metric,
        "lookback": str(lookback), "strategies": candidates,
        "baseline_strategy": baseline_strategy,
        "primary_strategy": primary_strategy,
        "window_options": window_options,
        "uses_normalized_notional": uses_normalized_notional,
        "common_window_count": len(ordered_ids),
        "individual_window_counts": {
            strategy: len(window_ids)
            for strategy, window_ids in valid_ids_by_strategy.items()
        },
        "sample_scope_differs": any(
            len(window_ids) != len(ordered_ids)
            for window_ids in valid_ids_by_strategy.values()),
    }
    if mode == "full":
        curve_column = metric_columns[metric]

        def _ordered_pair_rows(strategy):
            pairs = pairs_by_strategy[strategy].copy()
            pairs = pairs[
                pairs["window_id"].astype(str).isin(ordered_ids)]
            rows_by_id = {
                str(row["window_id"]): row
                for _index, row in pairs.iterrows()
            }
            return [rows_by_id[common_id] for common_id in ordered_ids]

        first_rows = _ordered_pair_rows(comparison_strategies[0])
        baseline_parts = [
            chart_array(
                row, curve_column, "baseline")
            for row in first_rows
        ]
        if (not baseline_parts
                or any(not len(part) for part in baseline_parts)):
            return {
                **base, "state": "empty",
                "message": "共同回测分段缺少每日收盘 日级曲线。",
            }
        segment_lengths = [len(part) for part in baseline_parts]

        def _full_series(parts, role, label, strategy):
            if (len(parts) != len(segment_lengths)
                    or any(
                        not len(part)
                        or len(part) != expected_length
                        or not np.all(np.isfinite(part))
                        for part, expected_length in zip(
                            parts, segment_lengths))):
                return None
            daily = np.concatenate(parts)
            return {
                "role": role, "label": label, "strategy": strategy,
                "x": np.arange(1, len(daily) + 1, dtype=float),
                "y": np.cumsum(daily),
            }

        baseline_item = _full_series(
            baseline_parts, "baseline", "每日收盘（固定基准）",
            baseline_strategy)
        if baseline_item is None:
            return {
                **base, "state": "empty",
                "message": "共同回测分段内 每日收盘 日级曲线不完整；未补零。",
            }
        series = [baseline_item]
        for strategy in candidates:
            candidate_parts = [
                chart_array(
                    row, curve_column, "candidate")
                for row in _ordered_pair_rows(strategy)
            ]
            item = _full_series(
                candidate_parts, "candidate", strategy, strategy)
            if item is None:
                return {
                    **base, "state": "empty",
                    "message": (
                        f"{strategy} 的共同回测分段日级曲线不完整；未补零。"),
                }
            series.append(item)

        common_day_count = int(sum(segment_lengths))
        expected_day_count = int(
            LOOKBACK_DAYS.get(str(lookback), common_day_count))
        complete_evidence = common_day_count == expected_day_count
        raw_labels = {
            "net": "累计净损益",
            "gross": "累计成本前损益",
            "tc": "累计成本",
        }
        normalized_labels = {
            "net": "累计净损益 / 各回测分段期初名义金额",
            "gross": "累计成本前损益 / 各回测分段期初名义金额",
            "tc": "累计成本 / 各回测分段期初名义金额",
        }
        metric_label = (
            normalized_labels if uses_normalized_notional
            else raw_labels)[metric]
        # 段数写进标题：这条曲线是把 N 段首尾接起来画的，和排名指标同一
        # 个口径（各段日损益 concat 后统计）。不写出来容易被当成一笔交易。
        segment_note = (
            f"，{len(ordered_ids)} 段接续" if len(ordered_ids) > 1 else "")
        title_prefix = (
            f"完整回放区间（{len(ordered_ids)} 段接续）"
            if complete_evidence and len(ordered_ids) > 1 else
            "完整回放区间"
            if complete_evidence else
            f"共同可比区间（{common_day_count}/{expected_day_count} 日"
            f"{segment_note}）"
        )
        return {
            **base,
            "series": series,
            "metric_label": metric_label,
            "title": f"{title_prefix} · {metric_label}",
            "common_day_count": common_day_count,
            "expected_day_count": expected_day_count,
            "complete_evidence": complete_evidence,
            "segment_lengths": tuple(segment_lengths),
            "segment_boundaries": tuple(
                int(value)
                for value in np.cumsum(segment_lengths)[:-1]),
        }

    first_strategy = comparison_strategies[0]
    first_model = chart_model(
        filtered, lookback, first_strategy, mode=mode, metric=metric,
        window_id=window_id, normalized=uses_normalized_notional)
    if first_model.get("state") != "ok":
        return first_model
    collection_key = "series" if mode == "single" else "bands"
    baseline_item = next(
        (item for item in first_model[collection_key]
         if item.get("role") == "baseline"), None)
    if baseline_item is None:
        return {
            **base, "state": "empty",
            "message": "共同回测分段缺少每日收盘 曲线。",
        }
    collection = [{**baseline_item, "strategy": baseline_strategy}]
    for strategy in candidates:
        model = chart_model(
            filtered, lookback, strategy, mode=mode, metric=metric,
            window_id=first_model.get("selected_window_id", window_id),
            normalized=uses_normalized_notional)
        if model.get("state") != "ok":
            return {
                **base, "state": "empty",
                "message": model.get(
                    "message", f"{strategy} 缺少共同回测分段曲线。"),
            }
        candidate_item = next(
            (item for item in model[collection_key]
             if item.get("role") == "candidate"), None)
        if candidate_item is not None:
            collection.append({
                **candidate_item, "strategy": strategy,
                "label": strategy,
            })
    title_prefix = (
        "单回测分段路径" if mode == "single" else "多回测分段中位路径")
    result = {
        **base, collection_key: collection,
        "metric_label": first_model.get("metric_label", "累计损益"),
        "title": (
            f"{title_prefix} · {first_model.get('metric_label', '累计损益')}"
            + ("（中位数与 P25/P75）" if mode == "typical" else "")),
    }
    if mode == "single":
        result["selected_window_id"] = first_model.get(
            "selected_window_id")
    return result
