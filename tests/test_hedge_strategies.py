from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from pricing import (
    CloseToCloseStrategy,
    ContractHistoryPool,
    FixedFreqStrategy,
    FixedTimeStrategy,
    HedgeBacktest,
    HedgeBandStrategy,
    HISTORY_TARGET_ENDPOINTS,
    LOOKBACK_DAYS,
    Option_Vanilla,
    PriceIntervalStrategy,
    SigmaBandStrategy,
    StrategyCase,
    compare_strategies,
    history_replay_index,
    history_window_summary,
    result_daily_frame,
    summarize_strategy_result,
    recommend_by_lookback,
    recommend_by_rolling_history,
    recommend_by_contract_history_pool,
)
from pricing.constants import ANNUAL_DAYS
from pricing.hedge_analysis import _strict_lookback_segment_lengths
from pricing.hedge_backtest import _infer_intraday_steps, _trading_day_groups


def _segment_end_offsets(lengths):
    """把段长还原成「各段终点距区间末端的偏移」，0 = 末端那一天。"""
    offsets, accumulated = [], 0
    for segment in reversed(lengths):
        offsets.append(accumulated)
        accumulated += segment
    return offsets


def test_strict_segments_put_the_remainder_at_the_oldest_end():
    """残段必须落在最老的一端，最近一段是完整到期交易。

    区间是从截止日往回数 L 天定的；若段边界改从起点正推，最相关的近端
    会被切成半截 MTM 交易——近年 243 日就会让「最近一段」只有 1 天。
    """
    assert _strict_lookback_segment_lengths(243, 22) == (1,) + (22,) * 11
    assert _strict_lookback_segment_lengths(61, 22) == (17, 22, 22)
    assert _strict_lookback_segment_lengths(122, 22) == (12,) + (22,) * 5
    # 整除时没有残段。
    assert _strict_lookback_segment_lengths(44, 22) == (22, 22)
    # 不足一个期限时只有一段，不该凭空造出零长段。
    assert _strict_lookback_segment_lengths(20, 22) == (20,)


def test_strict_segment_boundaries_near_the_end_do_not_move_with_lookback():
    """L 抖动只准改变最老那一段，近端边界一根都不能动。

    自然日口径下「近年」会在 241~243 之间浮动。若近端边界跟着 L 移动，
    同一批候选的名次会互换——排名对边界位置是敏感的。
    """
    offsets = {
        days: _segment_end_offsets(_strict_lookback_segment_lengths(days, 22))
        for days in (241, 242, 243)
    }
    shared = min(len(value) for value in offsets.values())
    assert offsets[241][:shared] == offsets[242][:shared] == \
        offsets[243][:shared]
    # 多出来的那一段挂在最老端，且它就是残段。
    assert offsets[243][shared:] == [242]
    assert _strict_lookback_segment_lengths(243, 22)[0] == 1


def _option(days=4):
    return Option_Vanilla(
        "Vanilla", s0=100.0, sr=[], K=100.0, T=days,
        sigma=0.2, cp=1, r=0.0, q=0.0,
    )


def test_close_to_close_triggers_at_each_day_end():
    prices = np.array([100, 101, 102, 101, 103, 104, 102, 103, 105], dtype=float)
    result = HedgeBacktest(
        _option(), prices, steps_per_day=2, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()
    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 2, 4, 6, 8]


def _three_day_intraday_prices():
    """Day 0 收盘建仓锚点 + 3 个完整的两 bar 交易日。"""
    index = pd.DatetimeIndex([
        "2026-01-02 15:00",
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 11:30", "2026-01-06 15:00",
        "2026-01-07 11:30", "2026-01-07 15:00",
    ])
    return pd.Series(
        [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2],
        index=index,
    )


def test_day_close_fallback_defaults_to_on_and_can_be_disabled():
    """收盘保底默认开启；显式关闭时退回纯策略触发。

    实务上收盘总要对冲一次，所以默认开；但它仍是可选项，关掉后不应留下
    任何兜底调仓。带宽取得极宽，策略自身不会触发，两者差异全部来自兜底。
    """
    prices = _three_day_intraday_prices()
    default_result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=HedgeBandStrategy("absolute", threshold=1000.0),
    ).run()
    disabled_result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=HedgeBandStrategy("absolute", threshold=1000.0),
        force_day_close_hedge=False,
    ).run()

    assert disabled_result["force_day_close_hedge"] is False
    assert np.flatnonzero(disabled_result["hedge_triggered"]).tolist() == [0, 6]
    assert not disabled_result["day_close_fallback_triggered"].any()

    assert default_result["force_day_close_hedge"] is True
    assert default_result["day_close_fallback_triggered"].any()
    assert (np.flatnonzero(default_result["hedge_triggered"]).size
            > np.flatnonzero(disabled_result["hedge_triggered"]).size)


def test_day_close_fallback_hedges_band_strategy_when_band_was_not_crossed():
    result = HedgeBacktest(
        _option(days=3), _three_day_intraday_prices(),
        steps_per_day=2, multiplier=0,
        strategy=HedgeBandStrategy("absolute", threshold=1000.0),
        force_day_close_hedge=True,
    ).run()

    # 最后一根是到期平仓，不计作收盘兜底。
    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 2, 4, 6]
    assert np.flatnonzero(
        result["day_close_fallback_triggered"]).tolist() == [2, 4]
    assert not result["strategy_hedge_triggered"][1:].any()
    assert result["force_day_close_hedge"] is True


def test_day_close_fallback_becomes_new_anchor_for_fixed_interval_strategy():
    prices = _three_day_intraday_prices().copy()
    prices.iloc[:] = [100.0, 100.8, 100.8, 101.2, 101.2, 101.4, 101.4]
    result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=HedgeBandStrategy("absolute", threshold=1.0),
        force_day_close_hedge=True,
    ).run()

    # Day 1 收盘兜底后上次实际对冲价已是 100.8。
    # Day 2 盘中的 101.2 只距新锚点 0.4，不应继续相对
    # Day 0 的 100 计算成 1.2 并误触发固定间隔策略。
    assert result["day_close_fallback_triggered"][2]
    assert not result["strategy_hedge_triggered"][3]
    assert result["day_close_fallback_triggered"][4]


def test_day_close_fallback_skips_fixed_time_strategy_already_triggered_at_close():
    prices = _three_day_intraday_prices()
    without_fallback = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=FixedTimeStrategy(["15:00"]),
        force_day_close_hedge=False,
    ).run()
    with_fallback = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=FixedTimeStrategy(["15:00"]),
        force_day_close_hedge=True,
    ).run()

    assert np.flatnonzero(with_fallback["hedge_triggered"]).tolist() == [0, 2, 4, 6]
    assert with_fallback["strategy_hedge_triggered"][2]
    assert with_fallback["strategy_hedge_triggered"][4]
    assert not with_fallback["day_close_fallback_triggered"].any()
    # 同一收盘 bar 不得产生第二次调仓或重复计费。
    np.testing.assert_allclose(
        with_fallback["shares"], without_fallback["shares"])
    np.testing.assert_allclose(
        with_fallback["tc_paid"], without_fallback["tc_paid"])


def test_day_close_fallback_complements_fixed_times_without_close_time():
    result = HedgeBacktest(
        _option(days=3), _three_day_intraday_prices(),
        steps_per_day=2, multiplier=0,
        strategy=FixedTimeStrategy(["11:30"]),
        force_day_close_hedge=True,
    ).run()

    assert np.flatnonzero(result["hedge_triggered"]).tolist() == list(range(7))
    assert np.flatnonzero(
        result["strategy_hedge_triggered"]).tolist() == [1, 3, 5]
    assert np.flatnonzero(
        result["day_close_fallback_triggered"]).tolist() == [2, 4]


def test_day_close_fallback_does_not_duplicate_close_to_close_strategy():
    prices = _three_day_intraday_prices()
    without_fallback = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=CloseToCloseStrategy(), force_day_close_hedge=False,
    ).run()
    with_fallback = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=CloseToCloseStrategy(), force_day_close_hedge=True,
    ).run()

    assert np.flatnonzero(with_fallback["hedge_triggered"]).tolist() == [0, 2, 4, 6]
    assert with_fallback["strategy_hedge_triggered"][2]
    assert with_fallback["strategy_hedge_triggered"][4]
    assert not with_fallback["day_close_fallback_triggered"].any()
    np.testing.assert_allclose(
        with_fallback["shares"], without_fallback["shares"])
    np.testing.assert_allclose(
        with_fallback["tc_paid"], without_fallback["tc_paid"])


def test_expiry_bar_closes_position_directly_without_day_close_fallback():
    result = HedgeBacktest(
        _option(days=3), _three_day_intraday_prices(),
        steps_per_day=2, multiplier=0,
        strategy=HedgeBandStrategy("absolute", threshold=1000.0),
        force_day_close_hedge=True,
    ).run()

    terminal = len(result["prices"]) - 1
    assert result["hedge_triggered"][terminal]
    assert not result["strategy_hedge_triggered"][terminal]
    assert not result["day_close_fallback_triggered"][terminal]
    assert result["shares"][terminal] == 0.0


def test_day_close_fallback_uses_end_of_night_session_trading_day_group():
    index = pd.DatetimeIndex([
        "2026-01-08 15:00",  # Day 0 建仓锚点
        "2026-01-08 21:00", "2026-01-09 02:30",
        "2026-01-09 09:00", "2026-01-09 15:00",
        "2026-01-09 21:00", "2026-01-10 02:30",
        "2026-01-12 09:00", "2026-01-12 15:00",
        "2026-01-12 21:00", "2026-01-13 02:30",
        "2026-01-13 09:00", "2026-01-13 15:00",
    ])
    prices = pd.Series(
        np.linspace(100.0, 101.2, len(index)), index=index)
    result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=4, multiplier=0,
        strategy=HedgeBandStrategy("absolute", threshold=1000.0),
        force_day_close_hedge=True,
    ).run()

    assert np.flatnonzero(
        result["day_close_fallback_triggered"]).tolist() == [4, 8]
    assert result["timestamps"][4] == pd.Timestamp("2026-01-09 15:00")
    assert result["timestamps"][8] == pd.Timestamp("2026-01-12 15:00")
    # 跨午夜的 02:30 仍在同一交易日组内，不是兜底收盘。
    assert not result["day_close_fallback_triggered"][2]
    assert not result["day_close_fallback_triggered"][6]
    assert not result["day_close_fallback_triggered"][10]


def test_legacy_fixed_frequency_remains_backend_api_compatibility_only():
    prices = np.array([100, 101, 102, 103, 104], dtype=float)
    bt = HedgeBacktest(
        _option(), prices, hedge_freq=2, multiplier=0, strategy=None,
        force_day_close_hedge=False)
    assert isinstance(bt.strategy, FixedFreqStrategy)
    result = bt.run()
    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 2, 4]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (FixedFreqStrategy(2), "固定 bar 频率（每 2 bar）"),
        (CloseToCloseStrategy(), "每日收盘（close-to-close）"),
        (HedgeBandStrategy("relative", 0.01), "固定间隔（相对价格=0.01）"),
    ],
)
def test_backend_summary_reports_active_strategy_instead_of_legacy_frequency(
        capsys, strategy, expected):
    prices = np.array([100, 101, 102, 103, 104], dtype=float)
    bt = HedgeBacktest(
        _option(), prices, hedge_freq=99, multiplier=0, strategy=strategy)
    bt.summary()
    output = capsys.readouterr().out
    assert expected in output
    assert "每 99 天" not in output
    assert "实际采样 bar/日" in output


def test_close_to_close_uses_actual_intraday_day_end():
    """真实分钟序列从首日盘中开始时，也必须落在每日最后一根 bar。"""
    idx = pd.DatetimeIndex([
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 11:30", "2026-01-06 15:00",
        "2026-01-07 11:30", "2026-01-07 15:00",
        "2026-01-08 11:30",
    ])
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106], index=idx)
    result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=2, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()
    # 第 3 个交易日组在 2026-01-07 15:00 收盘即到期，不得再纳入
    # 下一组的 2026-01-08 11:30。
    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 1, 3, 5]
    assert result["timestamps"][-1] == pd.Timestamp("2026-01-07 15:00")
    assert result["n_days"] == 3


def test_intraday_expiry_stops_at_tth_group_close_without_post_expiry_pnl():
    idx = pd.DatetimeIndex([
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 11:30", "2026-01-06 15:00",
        # 下一交易日故意放极端价格；若被错误纳入会产生巨大伪 PnL。
        "2026-01-07 11:30", "2026-01-07 15:00",
    ])
    full = pd.Series([100, 101, 102, 103, 1000, 1001], index=idx)
    exact = full.iloc[:4]

    full_result = HedgeBacktest(
        _option(days=2), full, steps_per_day=2, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()
    exact_result = HedgeBacktest(
        _option(days=2), exact, steps_per_day=2, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()

    assert full_result["timestamps"][-1] == pd.Timestamp("2026-01-06 15:00")
    assert full_result["n_bars"] == 3
    assert full_result["n_days"] == 2
    assert full_result["prices"].tolist() == exact_result["prices"].tolist()
    assert full_result["net_daily"] == pytest.approx(exact_result["net_daily"])


def test_intraday_expiry_rejects_partial_terminal_trading_day_group():
    idx = pd.DatetimeIndex([
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 11:30", "2026-01-06 15:00",
        "2026-01-07 11:30",  # 第 3 组尚未到 15:00 收盘
    ])
    prices = pd.Series([100, 101, 102, 103, 104], index=idx)

    with pytest.raises(ValueError, match=r"第3个.*交易日组不完整.*11:30"):
        HedgeBacktest(
            _option(days=3), prices, steps_per_day=2, multiplier=0,
            strategy=CloseToCloseStrategy(),
        )


def test_fixed_times_uses_real_datetime_index():
    idx = pd.DatetimeIndex([
        "2026-01-02 15:00", "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 11:30", "2026-01-06 15:00", "2026-01-07 11:30",
        "2026-01-07 15:00", "2026-01-08 11:30", "2026-01-08 15:00",
    ])
    prices = pd.Series([100, 101, 102, 101, 103, 104, 102, 103, 105], index=idx)
    result = HedgeBacktest(
        _option(), prices, steps_per_day=2, multiplier=0,
        strategy=FixedTimeStrategy(["11:30"]), force_day_close_hedge=False,
    ).run()
    # 末 bar 始终是到期平仓，因此固定时刻触发为 1/3/5/7，另含 0/8。
    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 1, 3, 5, 7, 8]
    assert result["timestamps"] is not None


def test_fixed_times_explicit_day_sessions_skip_night_target():
    idx = pd.DatetimeIndex([
        "2026-01-02 15:00",
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 11:30", "2026-01-06 15:00",
    ])
    prices = pd.Series([100, 101, 102, 103, 104], index=idx)
    strategy = FixedTimeStrategy(
        ["23:00", "11:30", "15:00"],
        trading_sessions=[
            ("09:00", "10:15"),
            ("10:30", "11:30"),
            ("13:30", "15:00"),
        ],
    )

    result = HedgeBacktest(
        _option(days=2), prices, steps_per_day=2, multiplier=0,
        strategy=strategy,
    ).run()

    assert tuple(t.strftime("%H:%M") for t in strategy.requested_times) == (
        "23:00", "11:30", "15:00")
    assert tuple(t.strftime("%H:%M") for t in strategy.effective_times) == (
        "11:30", "15:00")
    assert tuple(t.strftime("%H:%M") for t in strategy.skipped_times) == (
        "23:00",)
    assert result["fixed_time_requested_times"] == (
        "23:00", "11:30", "15:00")
    assert result["fixed_time_effective_times"] == ("11:30", "15:00")
    assert result["fixed_time_skipped_times"] == ("23:00",)
    assert result["fixed_time_trading_sessions"] == (
        ("09:00", "10:15"),
        ("10:30", "11:30"),
        ("13:30", "15:00"),
    )
    assert np.flatnonzero(result["strategy_hedge_triggered"]).tolist() == [1, 2, 3]


def test_fixed_times_session_endpoints_and_cross_midnight_are_inclusive():
    strategy = FixedTimeStrategy(
        ["20:59", "21:00", "23:00", "02:30", "02:31"],
        trading_sessions=[("21:00", "02:30")],
    )

    assert tuple(t.strftime("%H:%M") for t in strategy.effective_times) == (
        "21:00", "23:00", "02:30")
    assert tuple(t.strftime("%H:%M") for t in strategy.skipped_times) == (
        "20:59", "02:31")


def test_fixed_times_active_session_target_missing_still_fails():
    idx = pd.DatetimeIndex([
        "2026-01-02 15:00",
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 15:00",  # 有效交易时段内的 11:30 缺 Bar
    ])
    prices = pd.Series([100, 101, 102, 103], index=idx)
    strategy = FixedTimeStrategy(
        ["11:30", "23:00"],
        trading_sessions=[("09:00", "11:30"), ("13:30", "15:00")],
    )

    with pytest.raises(
            ValueError,
            match=r"11:30 在 1/2 个交易日组中缺失（如 2026-01-06）"):
        HedgeBacktest(
            _option(days=2), prices, steps_per_day=2, multiplier=0,
            strategy=strategy,
        )


def test_fixed_times_all_targets_skipped_is_valid_noop_without_timestamps(capsys):
    strategy = FixedTimeStrategy(
        ["23:00"], trading_sessions=[("09:00", "15:00")])
    result = HedgeBacktest(
        _option(), np.array([100, 101, 102, 103, 104], dtype=float),
        multiplier=0, strategy=strategy, force_day_close_hedge=False,
    ).run()

    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 4]
    assert not result["strategy_hedge_triggered"].any()
    assert result["fixed_time_requested_times"] == ("23:00",)
    assert result["fixed_time_effective_times"] == ()
    assert result["fixed_time_skipped_times"] == ("23:00",)

    HedgeBacktest(
        _option(), np.array([100, 101, 102, 103, 104], dtype=float),
        multiplier=0, strategy=strategy,
    ).summary()
    assert "全部为非交易时刻，自动跳过" in capsys.readouterr().out


def test_fixed_times_set_trading_sessions_none_restores_strict_targets():
    strategy = FixedTimeStrategy(
        ["11:30", "23:00"], trading_sessions=[("09:00", "15:00")])
    assert tuple(t.strftime("%H:%M") for t in strategy.times) == ("11:30",)

    returned = strategy.set_trading_sessions(None)

    assert returned is strategy
    assert strategy.trading_sessions is None
    assert tuple(t.strftime("%H:%M") for t in strategy.times) == (
        "11:30", "23:00")
    assert strategy.skipped_times == ()


@pytest.mark.parametrize(
    "index",
    [pd.date_range("2026-01-01", periods=5, freq="B"), pd.RangeIndex(5)],
)
def test_fixed_times_rejects_non_intraday_index(index):
    prices = pd.Series([100, 101, 102, 103, 104], index=index)
    with pytest.raises(ValueError, match="日内|intraday|时刻|DatetimeIndex"):
        HedgeBacktest(
            _option(), prices, multiplier=0,
            strategy=FixedTimeStrategy(["11:30"]),
        ).run()


def test_fixed_times_requires_every_requested_time():
    idx = pd.DatetimeIndex([
        "2026-01-05 10:00", "2026-01-05 11:30",
        "2026-01-06 10:00", "2026-01-06 11:30",
        "2026-01-07 10:00",
    ])
    prices = pd.Series([100, 101, 102, 103, 104], index=idx)
    with pytest.raises(ValueError, match="未能逐交易日组匹配"):
        HedgeBacktest(
            _option(days=2), prices, steps_per_day=2, multiplier=0,
            strategy=FixedTimeStrategy(["11:30", "15:00"]),
        )


def test_fixed_times_requires_targets_in_each_included_trading_day_group():
    idx = pd.DatetimeIndex([
        "2026-01-02 15:00",  # 单点 Day 0 基准组，不参与固定时刻校验
        "2026-01-05 10:00", "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 10:00", "2026-01-06 15:00",  # 仅本组缺 11:30
        "2026-01-07 10:00", "2026-01-07 11:30", "2026-01-07 15:00",
    ])
    prices = pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)
    with pytest.raises(
            ValueError,
            match=r"11:30 在 1/2 个交易日组中缺失（如 2026-01-06）"):
        HedgeBacktest(
            _option(days=2), prices, steps_per_day=3, multiplier=0,
            strategy=FixedTimeStrategy(["11:30", "15:00"]),
        )


def test_night_session_and_next_day_session_share_trading_day_group():
    idx = pd.DatetimeIndex([
        "2026-01-09 21:00",  # Friday night
        "2026-01-10 02:30",  # after midnight
        "2026-01-12 09:00",  # Monday day session
        "2026-01-12 15:00",  # same trading-day close
        "2026-01-12 21:00",  # next trading day starts
    ])
    assert _trading_day_groups(idx).tolist() == [0, 0, 0, 0, 1]


def test_fixed_time_default_business_order_matches_trading_day_sequence():
    idx = pd.DatetimeIndex([
        "2026-01-08 15:00",  # Day 0 建仓锚点
        "2026-01-08 23:00",  # 属于 2026-01-09 交易日
        "2026-01-09 11:30", "2026-01-09 15:00",
        "2026-01-09 23:00",  # 属于下一个交易日
        "2026-01-12 11:30", "2026-01-12 15:00",
    ])
    strategy = FixedTimeStrategy(
        "23:00,11:30,15:00",
        trading_sessions=[
            ("21:00", "23:00"),
            ("09:00", "10:15"),
            ("10:30", "11:30"),
            ("13:30", "15:00"),
        ],
    )
    result = HedgeBacktest(
        _option(days=2),
        pd.Series(np.linspace(100.0, 101.0, len(idx)), index=idx),
        steps_per_day=3,
        multiplier=0,
        strategy=strategy,
    ).run()

    assert result["fixed_time_requested_times"] == (
        "23:00", "11:30", "15:00")
    assert result["trading_day_groups"].tolist() == [0, 1, 1, 1, 2, 2, 2]
    assert np.flatnonzero(
        result["strategy_hedge_triggered"]).tolist() == [1, 2, 3, 4, 5]


def test_night_session_tail_without_evening_opener_stays_with_next_day_session():
    idx = pd.DatetimeIndex([
        "2026-01-10 02:30",  # 周五夜盘的周六凌晨尾段
        "2026-01-12 09:00", "2026-01-12 11:30", "2026-01-12 15:00",
    ])
    assert _trading_day_groups(idx).tolist() == [0, 0, 0, 0]


def test_midnight_daily_datetime_index_keeps_calendar_day_groups():
    idx = pd.date_range("2026-01-05", periods=4, freq="B")
    assert _trading_day_groups(idx).tolist() == [0, 1, 2, 3]


def test_intraday_step_inference_falls_back_for_non_datetime_index():
    assert _infer_intraday_steps(pd.RangeIndex(10)) == 1


def test_price_interval_absolute_and_relative():
    absolute = PriceIntervalStrategy(2.0, "absolute")
    relative = PriceIntervalStrategy(0.02, "relative")
    base = {"S_last": 100.0}
    assert not absolute.should_hedge({**base, "S": 101.99})
    assert absolute.should_hedge({**base, "S": 102.0})
    assert not relative.should_hedge({**base, "S": 101.99})
    assert relative.should_hedge({**base, "S": 102.0})


def test_unified_hedge_band_supports_all_threshold_types():
    base = {"S_last": 100.0, "i": 1, "i_last": 0, "dt_bar": 1 / ANNUAL_DAYS,
            "sigma_impl": 0.2, "log_ret_hist": np.array([]), "steps_per_day": 1}
    assert HedgeBandStrategy("absolute", threshold=2).should_hedge({**base, "S": 102})
    assert HedgeBandStrategy("relative", threshold=0.02).should_hedge({**base, "S": 102})
    assert HedgeBandStrategy("sigma", k=0.5).should_hedge({**base, "S": 102})

    # S=100、年化波动率 20% 时，1% 价格带等价于绝对 1 元和约 0.7794 日 σ（243 日口径）。
    eq = HedgeBandStrategy.convert_threshold(0.01, "relative", 100, 0.2)
    assert eq["absolute"] == pytest.approx(1.0)
    assert eq["relative"] == pytest.approx(0.01)
    assert eq["sigma"] == pytest.approx(0.01 / (0.2 / np.sqrt(ANNUAL_DAYS)))

    # 固定绝对/相对带宽不应因为 sigma=0 而失效。
    zero_sigma = {**base, "sigma_impl": 0.0, "S": 102.0}
    assert HedgeBandStrategy("absolute", threshold=2).should_hedge(zero_sigma)
    assert HedgeBandStrategy("relative", threshold=0.02).should_hedge(zero_sigma)


def test_strict_realized_sigma_uses_independent_warmup_without_moving_path():
    path = np.array([100.0, 101.0, 100.5])
    warmup = np.array([0.010, -0.006, 0.008, -0.004])
    result = HedgeBacktest(
        _option(days=2), path,
        strategy=SigmaBandStrategy(
            k=0.5, sigma_source="realized", window_days=4),
        steps_per_day=1,
        sigma_warmup_log_returns=warmup,
        strict_sigma_warmup=True,
    ).run()

    # 预热收益只作为 sigma seed；实际 T 日价格/PnL 区间仍是原始 3 点。
    np.testing.assert_array_equal(result["prices"], path)
    np.testing.assert_array_equal(result["sigma_warmup_log_returns"], warmup)
    assert result["sigma_warmup_bars"] == 4
    assert result["n_bars"] == 2
    assert result["n_trade_days"] == 2
    assert len(result["net_daily"]) == len(path)

    with pytest.raises(ValueError, match="严格预热不足.*需要 Day 0 前 4 根"):
        HedgeBacktest(
            _option(days=2), path,
            strategy=SigmaBandStrategy(
                k=0.5, sigma_source="realized", window_days=4),
            sigma_warmup_log_returns=warmup[:-1],
            strict_sigma_warmup=True,
        )


def test_realized_sigma_trigger_return_exclusion_accounts_for_warmup_prefix():
    strategy = SigmaBandStrategy(
        k=0.5, sigma_source="realized", window_days=3)
    strategy._last_trigger_i = 2
    warmup = np.array([0.01, 0.02, -0.01])
    path_returns = np.array([0.03, 0.20, -0.02])
    combined = np.r_[warmup, path_returns]
    actual = strategy._sigma_ref({
        "sigma_impl": 0.2,
        "log_ret_hist": combined,
        "log_ret_warmup_bars": len(warmup),
        "steps_per_day": 1,
        "strict_realized_sigma": True,
    })

    # 触发 bar i=2 对应 combined[warmup_bars + i - 1]，即 0.20；
    # 若仍按旧索引 i-1 删除，会错误删掉 warmup[1]。
    clean = np.delete(combined, len(warmup) + 2 - 1)
    expected = np.std(clean[-3:], ddof=1) * np.sqrt(ANNUAL_DAYS)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("strategy", "value_attr"),
    [
        (HedgeBandStrategy("absolute", threshold=1.0), "threshold"),
        (PriceIntervalStrategy(1.0, "absolute"), "interval"),
    ],
)
def test_csv_rebase_scales_absolute_strategy_without_mutating_caller(
        tmp_path, strategy, value_attr):
    path = tmp_path / "daily.csv"
    pd.DataFrame(
        {"date": pd.date_range("2026-01-05", periods=3, freq="B"),
         "close": [200.0, 202.0, 204.0]},
    ).to_csv(path, index=False)

    bt = HedgeBacktest.from_csv(
        _option(days=2), path, date_col="date", strategy=strategy,
        multiplier=0,
    )

    # 参考价 100 -> 真实首价 200，绝对 1 元带宽应同步变成 2 元；
    # 原策略对象仍保持 1 元，供其他回测复用。
    assert getattr(bt.strategy, value_attr) == pytest.approx(2.0)
    assert getattr(strategy, value_attr) == pytest.approx(1.0)


def _wind_60min_session_series(
        day_times, *, night_times=(), after_midnight_times=(), periods=3):
    """按交易日构造 Wind 以 session 尾时刻标记的 60min bar。"""
    trade_dates = pd.date_range("2026-03-02", periods=periods, freq="B")
    timestamps = []
    for trade_date in trade_dates:
        night_date = trade_date - pd.offsets.BDay(1)
        timestamps.extend(
            pd.Timestamp(f"{night_date.date()} {clock}")
            for clock in night_times
        )
        timestamps.extend(
            pd.Timestamp(f"{trade_date.date()} {clock}")
            for clock in after_midnight_times
        )
        timestamps.extend(
            pd.Timestamp(f"{trade_date.date()} {clock}")
            for clock in day_times
        )
    index = pd.DatetimeIndex(timestamps)
    return pd.Series(
        100.0 + np.arange(len(index), dtype=float) * 0.01,
        index=index,
    )


def _au_60min_full_sessions(periods=3):
    # 夜盘 330 分钟 -> 6 根；商品日盘三个连续段 -> 2+1+2 根。
    return _wind_60min_session_series(
        ("10:00", "11:00", "11:30", "14:00", "15:00"),
        night_times=("22:00", "23:00"),
        after_midnight_times=("00:00", "01:00", "02:00", "02:30"),
        periods=periods,
    )


def test_wind_au_60min_accepts_real_eleven_bar_sessions(monkeypatch):
    import pricing.wind_data as wind_data

    series = _au_60min_full_sessions()
    assert len(series) == 3 * 11
    assert _infer_intraday_steps(series.index) == 11
    monkeypatch.setattr(
        wind_data, "get_intraday_close",
        lambda *args, **kwargs: series,
    )
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day", lambda code, bar_size: 11,
    )

    bt = HedgeBacktest.from_wind(
        _option(days=2), "AU.SHF", "2026-03-02", "2026-03-04",
        bar_size="60", steps_per_day=None, multiplier=0,
    )

    assert bt.steps_per_day == 11
    assert len(bt.prices) == 2 * 11


def test_wind_au_60min_rejects_missing_final_tail_bar(
        monkeypatch):
    import pricing.wind_data as wind_data

    series = _au_60min_full_sessions(periods=1).iloc[:-1]
    assert len(series) == 10
    assert series.index[-1].strftime("%H:%M") == "14:00"
    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day", lambda code, bar_size: 11)

    with pytest.raises(ValueError, match=r"交易日组不完整|典型/声明为 11"):
        HedgeBacktest.from_wind(
            _option(days=1), "AU.SHF", "2026-03-02", "2026-03-02",
            bar_size="60", steps_per_day=None, multiplier=0,
        )


def test_wind_gfex_60min_uses_five_day_session_bars(monkeypatch):
    import pricing.wind_data as wind_data

    series = _wind_60min_session_series(
        ("10:00", "11:00", "11:30", "14:00", "15:00"), periods=2)
    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day", lambda code, bar_size: 5)

    bt = HedgeBacktest.from_wind(
        _option(days=1), "LC2609.GFE", "2026-03-02", "2026-03-03",
        bar_size="60", steps_per_day=None, multiplier=0,
    )

    assert bt.steps_per_day == 5
    assert len(bt.prices) == 5


def test_wind_intraday_fixed_times_skip_confirmed_non_trading_target(
        monkeypatch):
    import pricing.wind_data as wind_data

    series = _wind_60min_session_series(
        ("10:00", "11:00", "11:30", "14:00", "15:00"), periods=2)
    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day", lambda code, bar_size: 5)
    monkeypatch.setattr(
        wind_data, "get_trading_session_clock_ranges",
        lambda code: (
            (pd.Timestamp("09:30").time(), pd.Timestamp("11:30").time()),
            (pd.Timestamp("13:00").time(), pd.Timestamp("15:00").time()),
        ),
    )

    bt = HedgeBacktest.from_wind(
        _option(days=1), "510050.SH", "2026-03-02", "2026-03-03",
        bar_size="60", steps_per_day=None, multiplier=0,
        strategy=FixedTimeStrategy(["11:30", "15:00", "23:00"]),
    )
    result = bt.run()

    assert bt.strategy.requested_times != bt.strategy.effective_times
    assert tuple(t.strftime("%H:%M") for t in bt.strategy.effective_times) == (
        "11:30", "15:00")
    assert tuple(t.strftime("%H:%M") for t in bt.strategy.skipped_times) == (
        "23:00",)
    assert bt._wind_meta["fixed_time_requested_times"] == (
        "11:30", "15:00", "23:00")
    assert bt._wind_meta["fixed_time_effective_times"] == ("11:30", "15:00")
    assert bt._wind_meta["fixed_time_skipped_times"] == ("23:00",)
    assert result["fixed_time_skipped_times"] == ("23:00",)


def test_wind_daily_fixed_times_remain_invalid_even_with_preconfigured_session(
        monkeypatch):
    import pricing.wind_data as wind_data

    series = pd.Series(
        [100.0, 101.0], index=pd.bdate_range("2026-03-02", periods=2))
    monkeypatch.setattr(
        wind_data, "get_close_prices", lambda *args, **kwargs: series)
    strategy = FixedTimeStrategy(
        ["23:00"], trading_sessions=[("09:30", "15:00")])
    assert strategy.effective_times == ()

    with pytest.raises(ValueError, match=r"仅支持真实日内行情|每交易日只有一根"):
        HedgeBacktest.from_wind(
            _option(days=1), "510050.SH", "2026-03-02", "2026-03-03",
            multiplier=0, strategy=strategy,
        )


def test_wind_gfex_60min_rejects_four_bar_partial_day(monkeypatch):
    import pricing.wind_data as wind_data

    series = _wind_60min_session_series(
        ("10:00", "11:00", "11:30", "14:00", "15:00"), periods=1,
    ).iloc[:-1]
    assert series.index[-1].strftime("%H:%M") == "14:00"
    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day", lambda code, bar_size: 5)

    with pytest.raises(ValueError, match=r"交易日组不完整|典型/声明为 5"):
        HedgeBacktest.from_wind(
            _option(days=1), "LC2609.GFE", "2026-03-02", "2026-03-02",
            bar_size="60", steps_per_day=None, multiplier=0,
        )


def test_wind_intraday_rejects_single_partial_session_that_is_shorter_than_metadata(
        monkeypatch):
    import pricing.wind_data as wind_data

    series = pd.Series(
        [100.0, 100.1, 100.2],
        index=pd.DatetimeIndex([
            "2026-07-15 09:30", "2026-07-15 10:30", "2026-07-15 11:30",
        ]),
    )
    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day", lambda code, bar_size: 4)

    with pytest.raises(ValueError, match=r"交易日组不完整|典型/声明为 4"):
        HedgeBacktest.from_wind(
            _option(days=1), "510050.SH", "2026-07-15", "2026-07-15",
            bar_size="60", steps_per_day=None, multiplier=0,
        )


def _dce_15min_full_sessions():
    """三个真实形态的 DCE 交易日：8 根夜盘 + 15 根日盘。"""
    trade_dates = pd.date_range("2026-03-02", periods=3, freq="B")
    night_times = (
        "21:15", "21:30", "21:45", "22:00",
        "22:15", "22:30", "22:45", "23:00",
    )
    day_times = (
        "09:15", "09:30", "09:45", "10:00", "10:15",
        "10:45", "11:00", "11:15", "11:30",
        "13:45", "14:00", "14:15", "14:30", "14:45", "15:00",
    )
    timestamps = []
    for trade_date in trade_dates:
        night_date = trade_date - pd.offsets.BDay(1)
        timestamps.extend(
            pd.Timestamp(f"{night_date.date()} {clock}")
            for clock in night_times
        )
        timestamps.extend(
            pd.Timestamp(f"{trade_date.date()} {clock}")
            for clock in day_times
        )
    index = pd.DatetimeIndex(timestamps)
    return pd.Series(
        100.0 + np.arange(len(index), dtype=float) * 0.01,
        index=index,
    )


def test_commodity_trading_minutes_exclude_morning_recess():
    import pricing.wind_data as wind_data

    assert wind_data._TRADING_MINUTES_TABLE[("DCE", "农产品")] == 345
    assert wind_data._TRADING_MINUTES_TABLE[("SHFE", "贵金属")] == 555
    assert wind_data._TRADING_MINUTES_TABLE[("GFEX", "有色")] == 225
    assert wind_data._TRADING_MINUTES_TABLE[("CFFEX", "国债类")] == 255
    assert wind_data._SYMBOL_OVERRIDES["LU"] == 345


@pytest.mark.parametrize(
    ("sessions", "bar_size", "expected"),
    [
        ((75, 60, 90), 30, 8),
        ((75, 60, 90), 60, 5),
        ((120, 75, 60, 90), 30, 12),
        ((120, 75, 60, 90), 60, 7),
        ((240, 75, 60, 90), 30, 16),
        ((240, 75, 60, 90), 60, 9),
        ((330, 75, 60, 90), 30, 19),
        ((330, 75, 60, 90), 60, 11),
        ((120, 120), 60, 4),
        ((120, 135), 60, 5),
    ],
)
def test_trading_bar_count_ceils_each_continuous_session(
        monkeypatch, sessions, bar_size, expected):
    import pricing.wind_data as wind_data

    monkeypatch.setattr(
        wind_data, "_get_trading_session_minutes", lambda _code: sessions)
    wind_data.get_trading_bars_per_day.cache_clear()

    assert wind_data.get_trading_bars_per_day(
        f"TEST{sum(sessions)}_{bar_size}", bar_size) == expected


def test_wind_classification_aliases_and_prefixes_cover_p_and_t_contracts():
    import pricing.wind_data as wind_data

    assert wind_data._TRADING_SESSION_MINUTES_TABLE[
        ("DCE", "油脂油料")] == (120, 75, 60, 90)
    assert wind_data._TRADING_SESSION_MINUTES_TABLE[
        ("CFFEX", "利率类")] == (120, 135)
    assert wind_data.get_trading_bars_per_day("P2609.DCE", 60) == 7
    assert wind_data.get_trading_bars_per_day("T2609.CFE", 60) == 5


def test_wind_dce_15min_full_sessions_override_stale_24_bar_metadata(
        monkeypatch, capsys):
    import pricing.wind_data as wind_data

    series = _dce_15min_full_sessions()
    assert len(series) == 3 * 23
    assert _infer_intraday_steps(series.index) == 23

    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    # 模拟升级前把商品日盘误算为 240 分钟的旧元数据：360/15=24。
    # 多个真实交易日都稳定在 15:00 收盘时，应采用索引的 23 根。
    monkeypatch.setattr(
        wind_data, "get_trading_bars_per_day",
        lambda code, bar_size: 24)

    bt = HedgeBacktest.from_wind(
        _option(days=2), "P2609.DCE", "2026-03-02", "2026-03-04",
        bar_size="15", steps_per_day=None, multiplier=0,
        strategy=FixedTimeStrategy(["11:30", "15:00"]),
    )

    assert bt.steps_per_day == 23
    assert len(bt.prices) == 2 * 23
    assert bt.timestamps[-1].strftime("%H:%M") == "15:00"
    output = capsys.readouterr().out
    assert "多日真实收盘证据 23" in output


def test_compare_and_recommend_multiple_strategies():
    prices = np.array([100, 101, 103, 102, 105], dtype=float)
    cases = [
        StrategyCase("daily", CloseToCloseStrategy()),
        StrategyCase("move_2", PriceIntervalStrategy(2.0, "absolute")),
    ]
    summary, results = compare_strategies(
        _option(), prices, cases,
        {"steps_per_day": 1, "multiplier": 0, "tc_rate": 0.001},
    )
    assert set(summary["strategy"]) == {"daily", "move_2"}
    assert summary["rank"].tolist() == [1, 2]
    recommendations, ranking = recommend_by_lookback(
        results, lookbacks={"week": 3},
    )
    assert len(recommendations) == 1
    assert set(ranking["strategy"]) == {"daily", "move_2"}
    assert recommendations.iloc[0]["rank"] == 1


def test_comparison_distinguishes_triggers_rehedges_and_actual_trades():
    prices = np.array([100, 101, 103, 102, 105], dtype=float)
    summary, results = compare_strategies(
        _option(), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"steps_per_day": 1, "multiplier": 0, "tc_rate": 0.001},
    )
    row = summary.iloc[0]
    result = results["daily"]
    triggered = np.asarray(result["hedge_triggered"], dtype=bool)
    shares = np.asarray(result["shares"], dtype=float)
    trades = np.r_[shares[0], np.diff(shares)]

    # 旧 trade_count 保持“首仓 + 策略触发 + 末端平仓”的兼容定义。
    assert row["trade_count"] == np.count_nonzero(triggered)
    # 新 rehedge_count 只统计存续期内的策略触发。
    assert row["rehedge_count"] == np.count_nonzero(triggered[1:-1])
    assert row["actual_trade_count"] == np.count_nonzero(
        np.abs(trades) > 1e-10)
    assert row["turnover"] == pytest.approx(
        np.sum(np.where(np.abs(trades) > 1e-10, np.abs(trades), 0.0)
               * result["prices"])
    )


def test_result_daily_frame_exposes_cumulative_net_pnl():
    result = {
        "net_daily": np.array([-1.0, 2.0, 3.0, -2.0, 1.0]),
        "tc_paid": np.array([1.0, 0.2, 0.3, 0.4, 0.1]),
        "steps_per_day": 2,
    }
    daily = result_daily_frame(result)

    assert daily.index.name == "trade_day"
    assert daily.columns.tolist() == [
        "net_pnl", "tc_paid", "cumulative_net_pnl",
    ]
    assert daily["net_pnl"].tolist() == pytest.approx([4.0, -1.0])
    assert daily["tc_paid"].tolist() == pytest.approx([1.5, 0.5])
    assert daily["cumulative_net_pnl"].tolist() == pytest.approx([4.0, 3.0])


def test_history_window_summary_exposes_independent_daily_curves_and_metrics():
    result = {
        "net_daily": np.array([-1.0, 2.0, 3.0, -2.0, 1.0]),
        "tc_paid": np.array([1.0, 0.2, 0.3, 0.4, 0.1]),
        "steps_per_day": 2,
        "strategy_name": "close_to_close",
        "timestamps": pd.DatetimeIndex([
            "2026-01-02 15:00",
            "2026-01-05 11:30", "2026-01-05 15:00",
            "2026-01-06 11:30", "2026-01-06 15:00",
        ]),
    }
    original_net = result["net_daily"].copy()
    original_tc = result["tc_paid"].copy()
    windows = {"week": {"window_1": {"daily": result}}}

    summary = history_window_summary(windows)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["lookback"] == "week"
    assert row["window_id"] == "window_1"
    assert row["strategy"] == "daily"
    assert row["success"] is True or row["success"] == np.bool_(True)
    assert row["strategy_type"] == "close_to_close"
    assert row["start_ts"] == pd.Timestamp("2026-01-02 15:00")
    assert row["end_ts"] == pd.Timestamp("2026-01-06 15:00")
    assert row["days_used"] == 2
    assert row["score"] == pytest.approx(np.sqrt(8.5))
    assert row["total_net_pnl"] == pytest.approx(3.0)
    assert row["total_tc"] == pytest.approx(2.0)
    assert row["total_gross_pnl"] == pytest.approx(5.0)
    assert row["max_drawdown"] == pytest.approx(1.0)
    np.testing.assert_allclose(row["daily_net_pnl"], [4.0, -1.0])
    np.testing.assert_allclose(row["daily_tc"], [1.5, 0.5])
    np.testing.assert_allclose(row["daily_gross_pnl"], [5.5, -0.5])
    np.testing.assert_allclose(row["cumulative_net_pnl"], [4.0, 3.0])
    np.testing.assert_allclose(row["cumulative_tc"], [1.5, 2.0])
    np.testing.assert_allclose(row["cumulative_gross_pnl"], [5.5, 5.0])
    np.testing.assert_allclose(
        row["cumulative_net_pnl"],
        row["cumulative_gross_pnl"] - row["cumulative_tc"],
    )
    assert row["failure_scope"] == ""
    assert row["failure_reason"] == ""

    for column in (
            "daily_net_pnl", "daily_gross_pnl", "daily_tc",
            "cumulative_net_pnl", "cumulative_gross_pnl", "cumulative_tc"):
        assert isinstance(row[column], np.ndarray)
        assert row[column].dtype == np.dtype(float)
        assert not np.shares_memory(row[column], result["net_daily"])
        assert not np.shares_memory(row[column], result["tc_paid"])
    curve_arrays = [
        row[column] for column in (
            "daily_net_pnl", "daily_gross_pnl", "daily_tc",
            "cumulative_net_pnl", "cumulative_gross_pnl", "cumulative_tc")
    ]
    assert all(
        not np.shares_memory(left, right)
        for position, left in enumerate(curve_arrays)
        for right in curve_arrays[position + 1:]
    )

    row["daily_net_pnl"][0] = 999.0
    np.testing.assert_array_equal(result["net_daily"], original_net)
    np.testing.assert_array_equal(result["tc_paid"], original_tc)


def test_history_window_summary_preserves_strategy_and_endpoint_failures():
    windows = {
        "month": {
            "window_1": {
                "fixed": {
                    "error": "第 2 个交易日缺失 [11:30]",
                    "strategy_name": "fixed_times",
                },
            },
            "window_2": {
                "_window_error": "历史窗口没有完整到期交易日",
            },
        },
    }

    summary = history_window_summary(windows)

    assert len(summary) == 2
    fixed = summary[summary["strategy"] == "fixed"].iloc[0]
    assert not fixed["success"]
    assert fixed["strategy_type"] == "fixed_times"
    assert fixed["failure_scope"] == "strategy"
    assert fixed["failure_reason"] == "第 2 个交易日缺失 [11:30]"
    assert fixed["days_used"] == 0
    assert np.isnan(fixed["score"])
    assert all(
        isinstance(fixed[column], np.ndarray) and fixed[column].size == 0
        for column in (
            "daily_net_pnl", "daily_gross_pnl", "daily_tc",
            "cumulative_net_pnl", "cumulative_gross_pnl", "cumulative_tc")
    )

    endpoint = summary[summary["failure_scope"] == "endpoint"].iloc[0]
    assert not endpoint["success"]
    assert pd.isna(endpoint["strategy"])
    assert pd.isna(endpoint["strategy_type"])
    assert endpoint["window_id"] == "window_2"
    assert endpoint["failure_reason"] == "历史窗口没有完整到期交易日"


def _normalizable_history_result(
        strategy_name, *, s0=100.0, multiplier=5.0, quantity=2.0,
        position=None, normalization_available=True, normalization_reason=""):
    notional = s0 * multiplier * quantity
    result = {
        "net_daily": np.array([0.0, -10.0, 20.0]),
        "tc_paid": np.array([0.0, 1.0, 2.0]),
        "steps_per_day": 1,
        "strategy_name": strategy_name,
        "timestamps": pd.bdate_range("2026-01-05", periods=3),
        "quantity": quantity,
        "multiplier": multiplier,
        "normalization_schema": "s0_x_multiplier_x_abs_quantity_v1",
        "normalization_s0": s0,
        "normalization_notional": notional,
        "normalization_available": normalization_available,
        "normalization_reason": normalization_reason,
    }
    if position is not None:
        result["position"] = position
    return result


def test_history_window_summary_normalizes_signed_curves_by_initial_notional():
    baseline = _normalizable_history_result("close_to_close")
    candidate = _normalizable_history_result("fixed_freq")
    summary = history_window_summary({
        "month": {"window_1": {"c2c": baseline, "candidate": candidate}},
    }).set_index("strategy")

    row = summary.loc["candidate"]
    assert row["normalization_available"]
    assert row["normalization_notional"] == pytest.approx(1000.0)
    assert row["normalization_s0"] == pytest.approx(100.0)
    assert row["normalization_multiplier"] == pytest.approx(5.0)
    assert row["normalization_quantity"] == pytest.approx(2.0)
    np.testing.assert_allclose(
        row["normalized_daily_net_pnl"], [-0.01, 0.02])
    np.testing.assert_allclose(
        row["normalized_daily_gross_pnl"], [-0.009, 0.022])
    np.testing.assert_allclose(
        row["normalized_daily_tc"], [0.001, 0.002])
    np.testing.assert_allclose(
        row["normalized_cumulative_net_pnl"], [-0.01, 0.01])
    np.testing.assert_allclose(
        row["normalized_cumulative_gross_pnl"], [-0.009, 0.013])
    np.testing.assert_allclose(
        row["normalized_cumulative_tc"], [0.001, 0.003])


def test_history_window_summary_rejects_mismatched_pair_denominators():
    baseline = _normalizable_history_result("close_to_close")
    candidate = _normalizable_history_result("fixed_freq", s0=200.0)
    summary = history_window_summary({
        "month": {"window_1": {"c2c": baseline, "candidate": candidate}},
    }).set_index("strategy")

    assert summary.loc["c2c", "normalization_available"]
    candidate_row = summary.loc["candidate"]
    assert not candidate_row["normalization_available"]
    assert "分母不一致" in candidate_row["normalization_reason"]
    for column in (
            "normalized_daily_gross_pnl", "normalized_daily_net_pnl",
            "normalized_daily_tc", "normalized_cumulative_gross_pnl",
            "normalized_cumulative_net_pnl", "normalized_cumulative_tc"):
        assert candidate_row[column].size == 0


def test_history_window_summary_carries_position_and_rejects_cross_direction():
    baseline = _normalizable_history_result(
        "close_to_close", position=1)
    same_side = _normalizable_history_result(
        "hedge_band", position=1)
    summary = history_window_summary({
        "month": {
            "window_1": {
                "c2c": baseline,
                "candidate": same_side,
            },
        },
    })
    assert summary["position"].eq(1).all()

    opposite_side = _normalizable_history_result(
        "hedge_band", position=-1)
    with pytest.raises(ValueError, match="禁止跨方向"):
        history_window_summary({
            "month": {
                "window_1": {
                    "c2c": baseline,
                    "candidate": opposite_side,
                },
            },
        })


def test_history_window_summary_keeps_legacy_missing_position_readable():
    summary = history_window_summary({
        "month": {
            "window_1": {
                "c2c": _normalizable_history_result("close_to_close"),
                "candidate": _normalizable_history_result("hedge_band"),
            },
        },
    })

    assert summary["position"].isna().all()


def test_lookback_rejects_mixed_known_positions_but_allows_legacy_missing():
    legacy = {
        "a": _fake_intraday_result([0.0, 1.0, -1.0], steps_per_day=1),
        "b": _fake_intraday_result([0.0, 2.0, -2.0], steps_per_day=1),
    }
    _recommendations, legacy_ranking = recommend_by_lookback(
        legacy, lookbacks={"sample": 2})
    assert legacy_ranking["position"].isna().all()

    mixed = {
        "short": {**legacy["a"], "position": 1},
        "long": {**legacy["b"], "position": -1},
    }
    with pytest.raises(ValueError, match="禁止跨方向"):
        recommend_by_lookback(mixed, lookbacks={"sample": 2})


def test_backtest_normalization_zero_multiplier_fails_closed():
    result = HedgeBacktest(
        _option(days=2), np.array([100.0, 101.0, 102.0]),
        strategy=CloseToCloseStrategy(),
        multiplier=0.0,
        quantity=2.0,
        is_future=True,
        contract_multiplier=1000.0,
    ).run()

    assert not result["normalization_available"]
    assert result["normalization_notional"] == pytest.approx(0.0)
    assert result["normalization_reason"]
    assert result["normalization_invalid_reason"] == (
        result["normalization_reason"])
    # contract_multiplier 不能被拿来替换无效的 multiplier。
    assert result["normalization_notional"] != 100.0 * 1000.0 * 2.0


@pytest.mark.parametrize("quantity", [0.0, -2.0, np.nan, np.inf])
def test_backtest_rejects_non_positive_or_non_finite_quantity(quantity):
    with pytest.raises(ValueError, match="quantity 必须为有限正数"):
        HedgeBacktest(
            _option(days=2), np.array([100.0, 101.0, 102.0]),
            strategy=CloseToCloseStrategy(),
            multiplier=5.0,
            quantity=quantity,
        )


@pytest.mark.parametrize("s0", [0.0, np.nan, np.inf])
def test_backtest_normalization_invalid_s0_fails_closed(s0):
    with np.errstate(all="ignore"):
        result = HedgeBacktest(
            _option(days=2), np.array([s0, 101.0, 102.0]),
            strategy=CloseToCloseStrategy(),
            multiplier=5.0,
            quantity=2.0,
        ).run()

    assert not result["normalization_available"]
    assert "S0 必须为有限正数" in result["normalization_reason"]


def test_backtest_normalization_snapshot_uses_positive_quantity_formula():
    result = HedgeBacktest(
        _option(days=2), np.array([100.0, 101.0, 102.0]),
        strategy=CloseToCloseStrategy(),
        multiplier=5.0,
        quantity=2.0,
    ).run()

    assert result["quantity"] == pytest.approx(2.0)
    assert result["multiplier"] == pytest.approx(5.0)
    assert result["normalization_s0"] == pytest.approx(100.0)
    assert result["normalization_notional"] == pytest.approx(1000.0)
    assert result["normalization_available"]
    assert result["normalization_reason"] == ""
    assert result["normalization_schema"] == (
        "s0_x_multiplier_x_abs_quantity_v1")


def test_summarize_strategy_result_returns_comparison_compatible_metrics():
    result = {
        "net_daily": np.array([-1.0, 2.0, 3.0, -2.0, 1.0]),
        "tc_paid": np.array([1.0, 0.2, 0.3, 0.4, 0.1]),
        "steps_per_day": 2,
        "hedge_triggered": np.array([True, True, False, True, True]),
        "shares": np.array([1.0, 1.0, 2.0, 2.0, 0.0]),
        "prices": np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        "strategy_name": "hedge_band",
        "position": -1,
        "hedging_error": 3.0,
    }
    metadata = {"description": "绝对间隔=2", "nested": {"items": [1]}}

    row = summarize_strategy_result(result, "固定间隔(绝对=2)", metadata)

    assert row["strategy"] == "固定间隔(绝对=2)"
    assert row["strategy_type"] == "hedge_band"
    assert row["position"] == -1
    assert row["hedging_error"] == pytest.approx(3.0)
    assert row["n_trade_days"] == 2
    assert row["daily_net_pnl_rms"] == pytest.approx(np.sqrt(8.5))
    assert row["score"] == pytest.approx(np.sqrt(8.5))
    assert row["mean_daily_pnl"] == pytest.approx(1.5)
    assert row["pnl_volatility"] == pytest.approx(np.std([4.0, -1.0], ddof=1))
    assert row["avg_daily_tc"] == pytest.approx(1.0)
    assert row["total_tc"] == pytest.approx(2.0)
    assert row["total_net_pnl"] == pytest.approx(3.0)
    assert row["max_drawdown"] == pytest.approx(1.0)
    assert row["trade_count"] == 4
    assert row["rehedge_count"] == 2
    assert row["actual_trade_count"] == 3
    assert row["turnover"] == pytest.approx(410.0)
    assert row["meta_description"] == "绝对间隔=2"
    assert row["meta_nested"] == {"items": [1]}
    assert "rank" not in row

    # 输出元数据与调用方解耦；修改摘要不能反向污染输入。
    row["meta_nested"]["items"].append(2)
    assert metadata["nested"] == {"items": [1]}


def test_compare_strategies_reuses_single_result_summary_helper(monkeypatch):
    import pricing.hedge_analysis as hedge_analysis

    original = hedge_analysis.summarize_strategy_result
    calls = []

    def spy(result, display_name, metadata=None, steps_per_day=None):
        calls.append((display_name, metadata, steps_per_day))
        return original(result, display_name, metadata, steps_per_day)

    monkeypatch.setattr(hedge_analysis, "summarize_strategy_result", spy)
    cases = [
        StrategyCase(
            "daily", CloseToCloseStrategy(), {"description": "每日收盘"}),
        StrategyCase(
            "band", HedgeBandStrategy("absolute", 2.0),
            {"description": "绝对间隔=2"},
        ),
    ]
    summary, _ = compare_strategies(
        _option(), np.array([100, 101, 103, 102, 105], dtype=float), cases,
        {"steps_per_day": 1, "multiplier": 0, "tc_rate": 0.001},
    )

    assert [name for name, _metadata, _spd in calls] == ["daily", "band"]
    assert all(spd == 1 for _name, _metadata, spd in calls)
    assert set(summary["meta_description"]) == {"每日收盘", "绝对间隔=2"}


def test_equal_scores_rank_by_total_cost_then_strategy_name(monkeypatch):
    import pricing.hedge_analysis as hedge_analysis

    costs = {
        "close_to_close": 2.0,
        "fixed_freq": 1.0,
        "hedge_band": 1.0,
    }

    class FakeBacktest:
        def __init__(self, option, prices, strategy, **kwargs):
            self.strategy = strategy

        def run(self):
            strategy_name = self.strategy.name
            cost = costs[strategy_name]
            return {
                "steps_per_day": 1,
                "net_daily": np.array([0.0, 1.0]),
                "tc_paid": np.array([0.0, cost]),
                "hedge_triggered": np.array([True, True]),
                "shares": np.array([1.0, 0.0]),
                "prices": np.array([100.0, 101.0]),
                "strategy_name": strategy_name,
                "hedging_error": 1.0,
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)
    summary, _ = compare_strategies(
        _option(), np.array([100.0, 101.0]),
        [
            StrategyCase("z_costly", CloseToCloseStrategy()),
            StrategyCase("b_low", HedgeBandStrategy("absolute", 1.0)),
            StrategyCase("a_low", FixedFreqStrategy(1)),
        ],
    )

    assert summary["strategy"].tolist() == ["a_low", "b_low", "z_costly"]
    assert summary["total_tc"].tolist() == pytest.approx([1.0, 1.0, 2.0])


def test_equal_lookback_scores_rank_by_window_cost_then_strategy_name():
    results = {
        "z_costly": {
            **_fake_intraday_result([0.0, 1.0], steps_per_day=1),
            "tc_paid": np.array([0.0, 2.0]),
        },
        "b_low": {
            **_fake_intraday_result([0.0, 1.0], steps_per_day=1),
            "tc_paid": np.array([0.0, 1.0]),
        },
        "a_low": {
            **_fake_intraday_result([0.0, 1.0], steps_per_day=1),
            "tc_paid": np.array([0.0, 1.0]),
        },
    }
    recommendations, ranking = recommend_by_lookback(
        results, lookbacks={"day": 1},
    )

    assert ranking["strategy"].tolist() == ["a_low", "b_low", "z_costly"]
    assert recommendations["strategy"].tolist() == ["a_low"]


def _fake_intraday_result(net, steps_per_day=2):
    net = np.asarray(net, dtype=float)
    return {
        "net_daily": net,
        "tc_paid": np.zeros_like(net),
        "steps_per_day": steps_per_day,
    }


def test_recommendation_aggregates_bars_to_days_and_requires_complete_window():
    complete = {
        "a": _fake_intraday_result([0, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]),
        "b": _fake_intraday_result([0, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4]),
    }
    recs, ranking = recommend_by_lookback(complete, lookbacks={"week": 5})
    assert len(recs) == 1
    assert recs.iloc[0]["strategy"] == "a"
    assert ranking["days_used"].tolist() == [5, 5]
    assert ranking["bars_used"].tolist() == [10, 10]
    assert ranking.iloc[0]["daily_net_pnl_rms"] == pytest.approx(4.0)

    incomplete = {
        "a": _fake_intraday_result([0, 1, 1, 1, 1, 1, 1, 1, 1]),
        "b": _fake_intraday_result([0, 2, 2, 2, 2, 2, 2, 2, 2]),
    }
    no_recs, incomplete_ranking = recommend_by_lookback(
        incomplete, lookbacks={"week": 5})
    assert no_recs.empty
    assert not incomplete_ranking["complete_window"].any()


def test_history_periods_and_default_endpoint_budgets():
    assert LOOKBACK_DAYS == {
        "week": 5,
        "month": 20,
        "quarter": 61,
        "half_year": int(round(ANNUAL_DAYS / 2)),
        "year": int(ANNUAL_DAYS),
    }
    assert LOOKBACK_DAYS["month"] == int(round(ANNUAL_DAYS / 12))
    assert LOOKBACK_DAYS["quarter"] == int(round(ANNUAL_DAYS / 4))
    assert LOOKBACK_DAYS["half_year"] == int(round(ANNUAL_DAYS / 2))
    assert HISTORY_TARGET_ENDPOINTS == {
        "week": 5,
        "month": 12,
        "quarter": 24,
        "half_year": 36,
        "year": 48,
    }


def _history_window_indices(window_results, lookback, strategy):
    """按 window_id 顺序取出某策略实际运行的时间索引。"""
    return [
        pd.DatetimeIndex(window[strategy]["timestamps"])
        for window in window_results[lookback].values()
    ]


def _variable_return_history(periods):
    returns = np.resize(
        np.array([0.010, -0.004, 0.007, -0.006, 0.003]),
        periods - 1,
    )
    values = 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
    return pd.Series(
        values,
        index=pd.bdate_range("2026-01-05", periods=periods),
    )


def test_rolling_history_realized_sigma_uses_max_v_without_future_leakage():
    maturity = 2
    lookback = 5
    max_v = 4
    prices = _variable_return_history(maturity + lookback + max_v)
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase(
            "rv_short",
            SigmaBandStrategy(
                k=0.5, sigma_source="realized", window_days=2)),
        StrategyCase(
            "rv_long",
            HedgeBandStrategy(
                "sigma", threshold=0.5,
                sigma_source="realized", window_days=max_v)),
    ]
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=maturity), prices, cases,
        {"multiplier": 5.0, "quantity": 2.0, "tc_rate": 0.0},
        lookbacks={"week": lookback}, target_endpoints=lookback,
    )

    assert len(recommendations) == 1
    assert ranking["realized_sigma_warmup_days"].eq(max_v).all()
    assert ranking["required_history_days"].eq(
        lookback + max_v).all()
    assert ranking["required_price_groups"].eq(
        lookback + max_v + 1).all()
    assert ranking["history_complete"].all()
    assert ranking["segment_count"].eq(3).all()
    assert ranking["expiry_segments"].eq(2).all()
    assert ranking["mtm_segments"].eq(1).all()
    assert ranking["warmup_eligible_endpoints"].eq(3).all()
    assert ranking["days_used"].eq(lookback).all()
    for window in windows["week"].values():
        for strategy_name in ("rv_short", "rv_long"):
            result = window[strategy_name]
            day0 = pd.Timestamp(result["timestamps"][0])
            day0_position = prices.index.get_loc(day0)
            expected_prices = prices.iloc[
                day0_position - max_v:day0_position + 1]
            expected_warmup = np.log(
                expected_prices.to_numpy()[1:]
                / expected_prices.to_numpy()[:-1])
            # 种子只来自 Day 0 及以前；路径仍从 Day 0 开始并恰好覆盖 T 日。
            np.testing.assert_allclose(
                result["sigma_warmup_log_returns"], expected_warmup)
            assert result["sigma_warmup_bars"] == max_v
            assert result["timestamps"][0] == day0
            assert result["timestamps"][-1] == prices.index[
                day0_position + result["evaluation_days"]]
            assert len(result["prices"]) == result["evaluation_days"] + 1


def test_rolling_history_realized_sigma_formal_boundary_is_l_plus_v_plus_anchor():
    maturity, lookback, warmup = 2, 5, 4
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase(
            "rv",
            SigmaBandStrategy(
                k=0.5, sigma_source="realized", window_days=warmup)),
    ]
    outputs = {}
    required = lookback + warmup
    required_price_groups = required + 1
    for available in (required_price_groups - 1, required_price_groups):
        outputs[available] = recommend_by_rolling_history(
            _option(days=maturity), _variable_return_history(available), cases,
            {"multiplier": 5.0, "quantity": 1.0, "tc_rate": 0.0},
            lookbacks={"week": lookback}, target_endpoints=lookback,
        )

    incomplete_recommendations, incomplete_ranking, incomplete_windows = (
        outputs[required_price_groups - 1])
    assert incomplete_recommendations.empty
    assert incomplete_ranking["required_history_days"].eq(required).all()
    assert incomplete_ranking["required_price_groups"].eq(
        required_price_groups).all()
    assert not incomplete_ranking["history_complete"].any()
    assert not incomplete_ranking["recommendation_eligible"].any()
    assert any(
        "预热" in window.get("rv", {}).get("error", "")
        for window in incomplete_windows["week"].values())

    complete_recommendations, complete_ranking, _ = outputs[
        required_price_groups]
    assert len(complete_recommendations) == 1
    assert complete_ranking["history_complete"].all()
    assert complete_ranking["recommendation_eligible"].all()


def test_non_realized_band_modes_do_not_add_warmup_requirement():
    maturity, lookback = 2, 5
    prices = _variable_return_history(maturity + lookback)
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase(
            "absolute",
            HedgeBandStrategy(
                "absolute", threshold=1.0,
                sigma_source="realized", window_days=99)),
        StrategyCase(
            "implied_sigma",
            SigmaBandStrategy(
                k=0.5, sigma_source="implied", window_days=99)),
    ]
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=maturity), prices, cases,
        {"multiplier": 5.0, "tc_rate": 0.0},
        lookbacks={"week": lookback}, target_endpoints=lookback,
    )

    assert len(recommendations) == 1
    assert ranking["realized_sigma_warmup_days"].eq(0).all()
    assert ranking["required_history_days"].eq(lookback).all()
    assert ranking["required_price_groups"].eq(lookback + 1).all()
    assert ranking["history_complete"].all()
    for window in windows["week"].values():
        assert window["absolute"]["sigma_warmup_bars"] == 0
        assert window["implied_sigma"]["sigma_warmup_bars"] == 0


def test_strict_lookbacks_cover_each_evidence_day_once():
    maturity = 2
    prices = pd.Series(
        100.0 + np.sin(np.arange(int(ANNUAL_DAYS) + maturity) / 17.0),
        index=pd.date_range(
            "2025-01-02", periods=int(ANNUAL_DAYS) + maturity, freq="B"),
    )

    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=maturity), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        target_endpoints=HISTORY_TARGET_ENDPOINTS,
    )

    assert recommendations.empty
    expected_segments = {
        "week": (3, 2, 1),
        "month": (10, 10, 0),
        "quarter": (31, 30, 1),
        "half_year": (61, 61, 0),
        "year": (122, 121, 1),
    }
    for lookback, (
            segment_count, expiry_segments, mtm_segments,
    ) in expected_segments.items():
        row = ranking[ranking["lookback"] == lookback].iloc[0]
        lookback_days = LOOKBACK_DAYS[lookback]
        assert row["evaluation_mode"] == "strict_lookback"
        assert row["sampling_mode"] == "strict_contiguous"
        assert row["target_endpoints"] == HISTORY_TARGET_ENDPOINTS[lookback]
        assert row["legacy_target_endpoints_ignored"]
        assert np.isnan(row["step_days"])
        assert row["evidence_days"] == lookback_days
        assert row["segment_count"] == segment_count
        assert row["expiry_segments"] == expiry_segments
        assert row["mtm_segments"] == mtm_segments
        assert row["planned_endpoints"] == segment_count
        assert row["selected_endpoints"] == segment_count
        assert row["rolling_windows"] == segment_count
        assert row["required_history_days"] == lookback_days
        assert row["required_price_groups"] == lookback_days + 1
        assert row["days_used"] == lookback_days
        assert row["history_complete"]
        assert row["selection_metric"] == (
            "strict_lookback_daily_rms_advantage_vs_c2c")
        # 数据健康时，参与相对评分的段数必须等于配对段数——它是段数，
        # 不是是/否标志。展示层拿它跟配对段数比来决定要不要提示“参与评分
        # N/M 段”，两者单位一旦不同，提示就会恒亮。
        assert row["relative_comparison_windows"] == row["paired_windows"]

        paths = _history_window_indices(windows, lookback, "daily")
        assert len(paths) == segment_count
        assert paths[0][0] == prices.index[-lookback_days - 1]
        assert paths[-1][-1] == prices.index[-1]
        assert sum(len(path) - 1 for path in paths) == lookback_days
        assert all(
            left[-1] == right[0]
            for left, right in zip(paths, paths[1:])
        )


def test_legacy_target_endpoint_api_is_accepted_but_does_not_change_segments():
    prices = pd.Series(
        np.linspace(100.0, 105.0, 32),
        index=pd.bdate_range("2026-01-05", periods=32),
    )
    cases = [StrategyCase("daily", CloseToCloseStrategy())]

    _, scalar_ranking, scalar_windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"short": 5, "long": 20}, target_endpoints=4,
    )
    assert scalar_ranking["target_endpoints"].tolist() == [4, 4]
    assert {key: len(value) for key, value in scalar_windows.items()} == {
        "short": 3,
        "long": 10,
    }
    assert scalar_ranking["legacy_target_endpoints_ignored"].all()

    _, mapped_ranking, mapped_windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"short": 5, "long": 20},
        target_endpoints={"short": 3, "long": 7, "unused": 99},
    )
    assert mapped_ranking["target_endpoints"].tolist() == [3, 7]
    assert {key: len(value) for key, value in mapped_windows.items()} == {
        "short": 3,
        "long": 10,
    }

    _, missing_ranking, missing_windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"short": 5, "long": 20},
        target_endpoints={"short": 3},
    )
    assert missing_ranking.loc[
        missing_ranking["lookback"].eq("short"), "target_endpoints"
    ].iloc[0] == 3
    assert np.isnan(missing_ranking.loc[
        missing_ranking["lookback"].eq("long"), "target_endpoints"
    ].iloc[0])
    assert {key: len(value) for key, value in missing_windows.items()} == {
        "short": 3,
        "long": 10,
    }


def test_strict_evidence_bounds_do_not_depend_on_option_maturity():
    prices = pd.Series(
        100.0 + np.sin(np.arange(80) / 11.0),
        index=pd.date_range("2025-08-01", periods=80, freq="B"),
    )
    outputs = {}
    for maturity in (2, 22):
        _recommendations, ranking, windows = recommend_by_rolling_history(
            _option(days=maturity), prices,
            [StrategyCase("daily", CloseToCloseStrategy())],
            {"multiplier": 0, "tc_rate": 0.0},
            lookbacks={"month": 20}, target_endpoints=12,
        )
        paths = _history_window_indices(windows, "month", "daily")
        outputs[maturity] = {
            "row": ranking.iloc[0],
            "starts": pd.DatetimeIndex([path[0] for path in paths]),
            "ends": pd.DatetimeIndex([path[-1] for path in paths]),
            "lengths": [len(path) for path in paths],
        }

    assert outputs[2]["starts"][0] == outputs[22]["starts"][0]
    assert outputs[2]["ends"][-1] == outputs[22]["ends"][-1]
    assert outputs[2]["starts"][0] == prices.index[-21]
    assert outputs[2]["ends"][-1] == prices.index[-1]
    assert set(outputs[2]["lengths"]) == {3}
    assert outputs[22]["lengths"] == [21]
    assert len(outputs[2]["lengths"]) == 10
    assert sum(length - 1 for length in outputs[2]["lengths"]) == 20
    assert sum(length - 1 for length in outputs[22]["lengths"]) == 20

    short_row = outputs[2]["row"]
    long_row = outputs[22]["row"]
    assert not short_row["maturity_exceeds_lookback"]
    assert long_row["maturity_exceeds_lookback"]
    assert short_row["segment_count"] == 10
    assert short_row["terminal_mode"] == "expiry"
    assert long_row["segment_count"] == 1
    assert long_row["terminal_mode"] == "mark_to_market"
    assert short_row["window_overlap_max_ratio"] == pytest.approx(0.0)
    assert long_row["window_overlap_max_ratio"] == pytest.approx(0.0)


def test_strict_history_completeness_boundary_requires_l_plus_day0_anchor():
    maturity = 5
    lookback_days = 20
    cases = [
        StrategyCase("daily", CloseToCloseStrategy()),
        StrategyCase("frequent", FixedFreqStrategy(1)),
    ]

    outputs = {}
    for available_days in (lookback_days, lookback_days + 1):
        prices = pd.Series(
            np.linspace(100.0, 104.0, available_days),
            index=pd.date_range(
                "2026-01-05", periods=available_days, freq="B"),
        )
        outputs[available_days] = recommend_by_rolling_history(
            _option(days=maturity), prices, cases,
            {"multiplier": 0, "tc_rate": 0.0},
            lookbacks={"month": lookback_days}, target_endpoints=12,
        )

    incomplete_recommendations, incomplete_ranking, incomplete_windows = (
        outputs[lookback_days])
    assert incomplete_recommendations.empty
    assert incomplete_ranking["planned_endpoints"].eq(4).all()
    assert incomplete_ranking["selected_endpoints"].eq(0).all()
    assert incomplete_ranking["eligible_endpoints"].eq(0).all()
    assert incomplete_ranking["rolling_windows"].eq(0).all()
    assert not incomplete_ranking["history_complete"].any()
    assert not incomplete_ranking["baseline_complete"].any()
    assert not incomplete_ranking["recommendation_eligible"].any()
    assert incomplete_ranking["comparison_status"].eq("no_pair").all()
    assert "_window_error" in incomplete_windows["month"]["segment_1"]

    complete_recommendations, complete_ranking, complete_windows = (
        outputs[lookback_days + 1])
    assert len(complete_recommendations) == 1
    assert complete_ranking["planned_endpoints"].eq(4).all()
    assert complete_ranking["eligible_endpoints"].eq(4).all()
    assert complete_ranking["rolling_windows"].eq(4).all()
    assert complete_ranking["days_used"].eq(lookback_days).all()
    assert complete_ranking["history_complete"].all()
    assert complete_ranking["baseline_complete"].all()
    assert complete_ranking["recommendation_eligible"].all()
    assert complete_ranking["comparison_status"].eq("formal").all()
    assert len(complete_windows["month"]) == 4


def test_target_sampling_all_strategies_share_each_period_path():
    prices = pd.Series(
        100.0 + np.sin(np.arange(22) / 5.0),
        index=pd.date_range("2026-03-02", periods=22, freq="B"),
    )
    strategy_names = ("daily", "frequent", "band")
    _recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [
            StrategyCase("daily", CloseToCloseStrategy()),
            StrategyCase("frequent", FixedFreqStrategy(1)),
            StrategyCase(
                "band", HedgeBandStrategy("relative", threshold=0.01)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"month": 20}, target_endpoints=12,
    )

    assert len(windows["month"]) == 10
    for window in windows["month"].values():
        indices = [
            pd.DatetimeIndex(window[name]["timestamps"])
            for name in strategy_names
        ]
        assert indices[0].equals(indices[1])
        assert indices[0].equals(indices[2])
    assert ranking["planned_endpoints"].eq(10).all()
    assert ranking["baseline_windows"].eq(10).all()
    assert ranking["paired_windows"].eq(10).all()
    assert ranking["days_used"].eq(20).all()
    assert ranking["comparison_coverage"].eq(1.0).all()


def test_legacy_fixed_step_is_ignored_by_strict_contiguous_mode():
    prices = pd.Series(
        np.linspace(100.0, 104.0, 30),
        index=pd.date_range("2026-04-01", periods=30, freq="B"),
    )
    kwargs = {
        "backtest_kwargs": {"multiplier": 0, "tc_rate": 0.0},
        "lookbacks": {"month": 20},
    }
    _, default_ranking, default_windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        **kwargs,
    )
    _, explicit_ranking, explicit_windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        step_days=5, **kwargs,
    )

    default_paths = _history_window_indices(
        default_windows, "month", "daily")
    explicit_paths = _history_window_indices(
        explicit_windows, "month", "daily")
    default_endpoints = pd.DatetimeIndex([path[-1] for path in default_paths])
    explicit_endpoints = pd.DatetimeIndex(
        [path[-1] for path in explicit_paths])
    expected_positions = np.asarray([11, 13, 15, 17, 19, 21, 23, 25, 27, 29])

    assert default_endpoints.equals(explicit_endpoints)
    np.testing.assert_array_equal(
        prices.index.get_indexer(default_endpoints), expected_positions)
    for ranking in (default_ranking, explicit_ranking):
        row = ranking.iloc[0]
        assert row["sampling_mode"] == "strict_contiguous"
        assert row["legacy_step_days_ignored"]
        assert np.isnan(row["step_days"])
        assert np.isnan(row["target_endpoints"])
        assert row["planned_endpoints"] == 10
        assert row["selected_endpoints"] == 10
        assert row["days_used"] == 20
        assert np.isnan(row["endpoint_spacing_min"])


def test_strict_lookback_without_day0_anchor_is_diagnostic():
    maturity = 5
    prices = pd.Series(
        np.linspace(100.0, 101.0, maturity),
        index=pd.date_range("2026-06-01", periods=maturity, freq="B"),
    )

    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=maturity), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, target_endpoints=12,
    )

    assert recommendations.empty
    row = ranking.iloc[0]
    assert row["planned_endpoints"] == 1
    assert row["selected_endpoints"] == 0
    assert row["eligible_endpoints"] == 0
    assert row["rolling_windows"] == 0
    assert row["required_history_days"] == 5
    assert row["required_price_groups"] == 6
    assert row["available_history_days"] == 4
    assert not row["history_complete"]
    assert not row["recommendation_eligible"]
    assert row["comparison_status"] == "no_pair"
    assert "_window_error" in windows["week"]["segment_1"]


def test_strict_lookback_early_termination_stays_incomplete_without_zero_fill(
        monkeypatch):
    import pricing.hedge_analysis as hedge_analysis

    class EarlyTerminationBacktest:
        def __init__(
                self, option, path_source=None, external_path=None,
                strategy=None, steps_per_day=1, evaluation_days=None,
                **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day
            self.evaluation_days = evaluation_days

        def run(self):
            # 模拟所有策略共享的 Day 2 提前敲出；剩余三日不得补零后冒充 L=5。
            used = min(2, self.evaluation_days)
            path = self.external_path.iloc[:used + 1]
            net = np.zeros(len(path), dtype=float)
            net[1:] = 1.0
            return {
                "net_daily": net,
                "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "evaluation_days": self.evaluation_days,
                "strategy_name": self.strategy.name,
                "prices": path.to_numpy(dtype=float),
                "timestamps": path.index,
                "knocked_out": True,
                "ko_day": used,
            }

    monkeypatch.setattr(
        hedge_analysis, "HedgeBacktest", EarlyTerminationBacktest)
    prices = pd.Series(
        np.linspace(100.0, 106.0, 7),
        index=pd.bdate_range("2026-01-05", periods=7),
    )
    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=5), prices,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5},
    )

    assert recommendations.empty
    assert ranking["comparison_eligible"].all()
    assert not ranking["baseline_complete"].any()
    assert not ranking["recommendation_eligible"].any()
    assert ranking["comparison_status"].eq("diagnostic").all()
    assert ranking["days_used"].eq(2).all()
    assert ranking["observed_days"].eq(2).all()


def _install_deterministic_history_backtest(monkeypatch, pnl_by_strategy):
    """按策略类型返回常数日 PnL，隔离历史择优的配对与排名规则。"""
    import pricing.hedge_analysis as hedge_analysis

    class FakeBacktest:
        def __init__(
                self, option, path_source=None, external_path=None,
                strategy=None, steps_per_day=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day

        def run(self):
            strategy_name = self.strategy.name
            daily_pnl = float(pnl_by_strategy[strategy_name])
            net = np.zeros(len(self.external_path), dtype=float)
            net[1:] = daily_pnl
            return {
                "net_daily": net,
                "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": strategy_name,
                "prices": np.asarray(self.external_path, dtype=float),
                "timestamps": (
                    self.external_path.index
                    if isinstance(self.external_path, pd.Series) else None),
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)


def _two_contract_history_pool(*, include_first=True):
    mapping_dates = pd.bdate_range("2026-01-05", periods=8)
    mapping = pd.Series(
        ["P2601.DCE"] * 4 + ["P2605.DCE"] * 4,
        index=mapping_dates,
    )
    first_dates = pd.bdate_range(end=mapping_dates[3], periods=7)
    second_dates = pd.bdate_range(end=mapping_dates[-1], periods=7)
    contract_prices = {
        "P2605.DCE": pd.Series(
            np.linspace(200.0, 206.0, len(second_dates)),
            index=second_dates,
        ),
    }
    if include_first:
        contract_prices["P2601.DCE"] = pd.Series(
            np.linspace(100.0, 106.0, len(first_dates)),
            index=first_dates,
        )
    return ContractHistoryPool(
        product_code="P.DCE",
        main_contract_by_date=mapping,
        contract_prices=contract_prices,
        main_contract_asof="P2605.DCE",
    )


def _direction_history_run(mode, position, tc_rate):
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase(
            "band", HedgeBandStrategy("absolute", threshold=1.25)),
    ]
    kwargs = {
        "position": position,
        "quantity": 2.0,
        "multiplier": 0.0,
        "tc_rate": tc_rate,
    }
    if mode == "rolling":
        prices = pd.Series(
            [100.0, 101.0, 99.5, 102.0, 101.0, 103.0, 102.0, 104.0],
            index=pd.bdate_range("2026-01-05", periods=8),
        )
        return recommend_by_rolling_history(
            _option(days=2),
            prices,
            cases,
            kwargs,
            lookbacks={"sample": 5},
            target_endpoints=3,
        )
    if mode == "contract_pool":
        return recommend_by_contract_history_pool(
            _option(days=2),
            _two_contract_history_pool(),
            cases,
            kwargs,
            lookbacks={"sample": 8},
            target_endpoints=4,
        )
    raise AssertionError(f"未知 history mode: {mode}")


def _successful_direction_windows(window_results):
    return {
        (lookback, window_id, strategy): result
        for lookback, windows in window_results.items()
        for window_id, window in windows.items()
        for strategy, result in window.items()
        if (not str(strategy).startswith("_window_")
            and isinstance(result, dict)
            and "error" not in result)
    }


@pytest.mark.parametrize("mode", ["rolling", "contract_pool"])
def test_history_batch_rejects_backtest_result_with_opposite_position(
        monkeypatch, mode):
    import pricing.hedge_analysis as hedge_analysis

    class MismatchedPositionBacktest:
        def __init__(
                self, option, path_source=None, external_path=None,
                strategy=None, steps_per_day=1, position=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day
            self.position = position

        def run(self):
            values = np.asarray(self.external_path, dtype=float)
            net = np.zeros(len(values), dtype=float)
            return {
                "net_daily": net,
                "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": self.strategy.name,
                "prices": values,
                "timestamps": getattr(self.external_path, "index", None),
                "position": (
                    -self.position
                    if self.strategy.name == "hedge_band"
                    else self.position
                ),
            }

    monkeypatch.setattr(
        hedge_analysis, "HedgeBacktest", MismatchedPositionBacktest)
    with pytest.raises(ValueError, match="禁止跨方向比较"):
        _direction_history_run(mode, position=1, tc_rate=0.0)


@pytest.mark.parametrize("mode", ["rolling", "contract_pool"])
def test_history_zero_cost_long_short_are_mirrors_with_same_rms_ranking(mode):
    short_rec, short_rank, short_windows = _direction_history_run(
        mode, position=1, tc_rate=0.0)
    long_rec, long_rank, long_windows = _direction_history_run(
        mode, position=-1, tc_rate=0.0)

    assert short_rank["position"].eq(1).all()
    assert long_rank["position"].eq(-1).all()
    assert short_rank["strategy"].tolist() == long_rank["strategy"].tolist()
    np.testing.assert_allclose(
        short_rank["score"], long_rank["score"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        short_rank["baseline_score"],
        long_rank["baseline_score"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert short_rec["strategy"].tolist() == long_rec["strategy"].tolist()

    short_results = _successful_direction_windows(short_windows)
    long_results = _successful_direction_windows(long_windows)
    assert short_results.keys() == long_results.keys()
    for key in short_results:
        short = short_results[key]
        long = long_results[key]
        assert short["position"] == 1
        assert short["position_label"] == "short"
        assert long["position"] == -1
        assert long["position_label"] == "long"
        np.testing.assert_allclose(long["shares"], -short["shares"])
        np.testing.assert_allclose(
            long["hedge_daily"], -short["hedge_daily"])
        np.testing.assert_allclose(
            long["option_daily"], -short["option_daily"])
        np.testing.assert_allclose(long["net_daily"], -short["net_daily"])
        np.testing.assert_allclose(
            long["portfolio_gamma"], -short["portfolio_gamma"])
        assert not np.any(short["tc_paid"])
        assert not np.any(long["tc_paid"])


@pytest.mark.parametrize("mode", ["rolling", "contract_pool"])
def test_history_costs_keep_gross_mirrored_and_charge_both_directions(mode):
    _short_rec, short_rank, short_windows = _direction_history_run(
        mode, position=1, tc_rate=0.001)
    _long_rec, long_rank, long_windows = _direction_history_run(
        mode, position=-1, tc_rate=0.001)

    assert short_rank["position"].eq(1).all()
    assert long_rank["position"].eq(-1).all()
    short_results = _successful_direction_windows(short_windows)
    long_results = _successful_direction_windows(long_windows)
    assert short_results.keys() == long_results.keys()
    for key in short_results:
        short = short_results[key]
        long = long_results[key]
        short_tc = np.asarray(short["tc_paid"], dtype=float)
        long_tc = np.asarray(long["tc_paid"], dtype=float)
        short_net = np.asarray(short["net_daily"], dtype=float)
        long_net = np.asarray(long["net_daily"], dtype=float)
        short_gross = short_net + short_tc
        long_gross = long_net + long_tc
        np.testing.assert_allclose(long_tc, short_tc)
        np.testing.assert_allclose(long_gross, -short_gross)
        np.testing.assert_allclose(long_net + short_net, -2.0 * short_tc)
        assert np.sum(short_tc) > 0.0


def test_contract_pool_fixed_time_uses_each_concrete_contract_session(
        monkeypatch):
    import pricing.hedge_analysis as hedge_analysis
    import pricing.wind_data as wind_data

    pool = _two_contract_history_pool()
    day_sessions = ((datetime.time(9, 0), datetime.time(11, 30)),
                    (datetime.time(13, 30), datetime.time(15, 0)))
    night_sessions = ((datetime.time(21, 0), datetime.time(23, 0)),
                      *day_sessions)
    profiles = {
        "P2601.DCE": day_sessions,
        "P2605.DCE": night_sessions,
    }
    monkeypatch.setattr(
        wind_data, "get_trading_session_clock_ranges",
        lambda code, **_kwargs: profiles[str(code).upper()],
    )
    observed = {"day": [], "night": []}

    class FakeBacktest:
        def __init__(
                self, option, path_source=None, external_path=None,
                strategy=None, steps_per_day=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day
            if isinstance(strategy, FixedTimeStrategy):
                bucket = (
                    "day" if float(external_path.iloc[0]) < 150 else "night")
                observed[bucket].append((
                    tuple(t.strftime("%H:%M")
                          for t in strategy.effective_times),
                    tuple(t.strftime("%H:%M")
                          for t in strategy.skipped_times),
                ))

        def run(self):
            net = np.zeros(len(self.external_path), dtype=float)
            net[1:] = 1.0
            return {
                "net_daily": net,
                "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": self.strategy.name,
                "prices": np.asarray(self.external_path, dtype=float),
                "timestamps": self.external_path.index,
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)

    recommend_by_contract_history_pool(
        _option(days=1), pool,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase(
                "fixed", FixedTimeStrategy(["21:00", "15:00"])),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 8}, target_endpoints=4,
    )

    assert observed["day"]
    assert observed["night"]
    assert set(observed["day"]) == {(('15:00',), ('21:00',))}
    assert set(observed["night"]) == {(('21:00', '15:00'), ())}


def test_contract_pool_realized_sigma_requires_each_segment_warmup_in_contract():
    pool = _two_contract_history_pool()
    maturity, warmup = 2, 3
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase(
            "rv",
            SigmaBandStrategy(
                k=0.5, sigma_source="realized", window_days=warmup)),
    ]
    recommendations, ranking, windows = recommend_by_contract_history_pool(
        _option(days=maturity), pool, cases,
        {"multiplier": 5.0, "quantity": 1.0, "tc_rate": 0.0},
        lookbacks={"sample": 8}, target_endpoints=4,
    )

    assert recommendations.empty
    assert ranking["realized_sigma_warmup_days"].eq(warmup).all()
    assert ranking["required_history_days"].eq(8 + warmup).all()
    assert ranking["required_price_groups"].eq(8 + warmup + 1).all()
    assert ranking["segment_count"].eq(4).all()
    assert ranking["eligible_endpoints"].eq(4).all()
    assert ranking["warmup_eligible_endpoints"].eq(2).all()
    assert not ranking["history_complete"].any()

    failed = 0
    succeeded = 0
    for window in windows["sample"].values():
        # C2C/非 realized 策略仍可在全部 T 日样本上运行。
        assert "error" not in window["c2c"]
        rv = window["rv"]
        if "error" in rv:
            failed += 1
            assert "同一具体合约" in rv["error"]
            assert "严禁跨合约" in rv["error"]
        else:
            succeeded += 1
            assert rv["sigma_warmup_bars"] == warmup
            assert rv["history_contract_code"] == (
                window["c2c"]["history_contract_code"])
            assert len(rv["prices"]) == maturity + 1
    assert (failed, succeeded) == (2, 2)


def test_contract_pool_windows_follow_endpoint_main_without_crossing_roll(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
        "fixed_freq": 1.0,
    })
    pool = _two_contract_history_pool()

    recommendations, ranking, windows = recommend_by_contract_history_pool(
        _option(days=2), pool,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 8}, target_endpoints=4,
    )

    assert recommendations["strategy"].tolist() == ["candidate"]
    assert ranking["history_mode"].eq("product_contract_pool").all()
    assert ranking["product_code"].eq("P.DCE").all()
    assert ranking["main_contract_asof"].eq("P2605.DCE").all()
    assert ranking["effective_main_contract"].eq("P2605.DCE").all()
    assert ranking["contract_count"].eq(2).all()
    assert ranking["eligible_contracts"].eq(2).all()
    assert ranking["recommendation_eligible"].all()

    for window in windows["sample"].values():
        result = window["c2c"]
        contract_code = result["history_contract_code"]
        endpoint = pd.Timestamp(result["history_endpoint_date"])
        timestamps = pd.DatetimeIndex(result["timestamps"])
        prices = np.asarray(result["prices"], dtype=float)
        assert timestamps[-1].normalize() == endpoint.normalize()
        if contract_code == "P2601.DCE":
            assert np.max(prices) < 150.0
        else:
            assert contract_code == "P2605.DCE"
            assert np.min(prices) > 150.0

    summary = history_window_summary(windows)
    assert set(summary["history_contract_code"]) == {
        "P2601.DCE", "P2605.DCE",
    }
    assert summary["history_endpoint_date"].notna().all()


def test_contract_pool_rolls_split_strict_evidence_and_short_runs_start_mtm():
    mapping_dates = pd.bdate_range("2026-01-05", periods=8)
    mapping = pd.Series(
        ["P2601.DCE"] * 3 + ["P2605.DCE"] * 5,
        index=mapping_dates,
    )
    first_dates = pd.bdate_range(end=mapping_dates[2], periods=6)
    second_dates = pd.bdate_range(end=mapping_dates[-1], periods=8)
    pool = ContractHistoryPool(
        product_code="P.DCE",
        main_contract_by_date=mapping,
        contract_prices={
            "P2601.DCE": pd.Series(
                np.linspace(100.0, 105.0, len(first_dates)),
                index=first_dates,
            ),
            "P2605.DCE": pd.Series(
                np.linspace(200.0, 207.0, len(second_dates)),
                index=second_dates,
            ),
        },
        main_contract_asof="P2605.DCE",
    )

    recommendations, ranking, windows = recommend_by_contract_history_pool(
        _option(days=2), pool,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 8}, target_endpoints=99,
    )

    assert len(recommendations) == 1
    assert ranking["recommendation_eligible"].all()
    assert ranking["days_used"].eq(8).all()
    assert ranking["segment_count"].eq(5).all()
    assert ranking["expiry_segments"].eq(3).all()
    assert ranking["mtm_segments"].eq(2).all()
    assert ranking["terminal_mode"].eq("mixed").all()
    assert ranking["contract_run_count"].eq(2).all()
    # 每条主力连续段内部都从末端对齐：残段落在该段最老的一头，最近那
    # 一笔是完整到期交易。P2601 跑了 3 日 -> (1, 2)，P2605 跑了 5 日 ->
    # (1, 2, 2)。
    assert ranking["segment_lengths"].apply(
        lambda value: value == (1, 2, 1, 2, 2)).all()

    observed_evidence_dates = []
    observed_segments = []
    for window in windows["sample"].values():
        result = window["c2c"]
        timestamps = pd.DatetimeIndex(result["timestamps"])
        observed_evidence_dates.extend(timestamps[1:].normalize())
        observed_segments.append((
            result["history_contract_code"],
            result["evaluation_days"],
            result["terminal_mode"],
        ))
        assert len(result["prices"]) == result["evaluation_days"] + 1
    assert pd.DatetimeIndex(observed_evidence_dates).equals(mapping_dates)
    assert observed_segments == [
        ("P2601.DCE", 1, "mark_to_market"),
        ("P2601.DCE", 2, "expiry"),
        ("P2605.DCE", 1, "mark_to_market"),
        ("P2605.DCE", 2, "expiry"),
        ("P2605.DCE", 2, "expiry"),
    ]


def test_contract_pool_accepts_independent_endpoint_budget_mapping(monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
        "fixed_freq": 1.0,
    })
    pool = _two_contract_history_pool()

    _, ranking, windows = recommend_by_contract_history_pool(
        _option(days=2), pool,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"near": 4, "far": 8},
        target_endpoints={"near": 2, "far": 4},
    )

    budgets = (
        ranking[["lookback", "target_endpoints"]]
        .drop_duplicates()
        .set_index("lookback")["target_endpoints"]
        .to_dict()
    )
    assert budgets == {"near": 2, "far": 4}
    assert {key: len(value) for key, value in windows.items()} == {
        "near": 2,
        "far": 4,
    }
    assert ranking["sampling_mode"].eq("strict_contiguous").all()
    assert ranking["legacy_target_endpoints_ignored"].all()
    assert ranking["days_used"].tolist() == [4, 4, 8, 8]


@pytest.mark.parametrize("contract_scales", [(1.0, 100.0), (100.0, 1.0)])
def test_contract_pool_ranking_equal_weights_windows_across_price_scales(
        monkeypatch, contract_scales):
    """反转历史合约金额量级不能改变逐窗相对 C2C 的推荐。"""
    import pricing.hedge_analysis as hedge_analysis

    pool = _two_contract_history_pool()
    contract_codes = ("P2601.DCE", "P2605.DCE")
    scale_by_contract = dict(zip(contract_codes, contract_scales))
    for code, prices in pool.contract_prices.items():
        prices.name = code

    # A 在第一个合约改善 50%、第二个合约恶化 10%；B 则分别恶化
    # 10%、改善 5%。按有界逐窗优势，A 的均值约为 20.45%，B 约为
    # -2.05%。若把不同合约的金额 PnL 直接拼接，名义价格更高的一侧会
    # 错误主导推荐。
    ratios = {
        "close_to_close": {
            "P2601.DCE": 1.0, "P2605.DCE": 1.0,
        },
        "fixed_freq": {
            "P2601.DCE": 0.5, "P2605.DCE": 1.1,
        },
        "hedge_band": {
            "P2601.DCE": 1.1, "P2605.DCE": 0.95,
        },
    }

    class FakeBacktest:
        def __init__(
                self, option, path_source=None, external_path=None,
                strategy=None, steps_per_day=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day

        def run(self):
            code = str(self.external_path.name)
            strategy_name = self.strategy.name
            daily_pnl = (
                scale_by_contract[code] * ratios[strategy_name][code])
            net = np.zeros(len(self.external_path), dtype=float)
            net[1:] = daily_pnl
            return {
                "net_daily": net,
                "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": strategy_name,
                "prices": np.asarray(self.external_path, dtype=float),
                "timestamps": self.external_path.index,
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)
    recommendations, ranking, _ = recommend_by_contract_history_pool(
        _option(days=1), pool,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase("typical_leader", FixedFreqStrategy(1)),
            StrategyCase(
                "high_price_leader",
                HedgeBandStrategy("absolute", threshold=1.0)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 8}, target_endpoints=4,
    )

    assert recommendations["strategy"].tolist() == ["typical_leader"]
    assert ranking["strategy"].tolist() == [
        "typical_leader", "c2c", "high_price_leader",
    ]
    by_name = ranking.set_index("strategy")
    assert by_name.loc[
        "typical_leader", "selection_improvement_vs_c2c"] == pytest.approx(
            (0.50 - 1.0 / 11.0) / 2.0)
    assert by_name.loc[
        "high_price_leader",
        "selection_improvement_vs_c2c"] == pytest.approx(
            (-1.0 / 11.0 + 0.05) / 2.0)
    assert by_name["selection_metric"].eq(
        "strict_lookback_daily_rms_advantage_vs_c2c").all()
    assert by_name["relative_comparison_windows"].eq(8).all()
    assert by_name["paired_contract_codes"].apply(
        lambda value: value == contract_codes).all()


def test_contract_pool_missing_contract_degrades_without_aborting_valid_windows(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
        "fixed_freq": 1.0,
    })
    pool = _two_contract_history_pool(include_first=False)

    recommendations, ranking, windows = recommend_by_contract_history_pool(
        _option(days=2), pool,
        [
            StrategyCase("c2c", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 8}, target_endpoints=4,
    )

    assert recommendations.empty
    assert ranking["rolling_windows"].eq(2).all()
    assert ranking["eligible_endpoints"].eq(2).all()
    assert not ranking["history_complete"].any()
    assert not ranking["recommendation_eligible"].any()
    assert ranking["comparison_status"].eq("diagnostic").all()
    assert ranking["failure_scope"].eq("endpoint").all()
    assert ranking["failure_reason"].str.contains("P2601.DCE").all()
    assert sum("_window_error" in item for item in windows["sample"].values()) == 2


def test_contract_pool_does_not_hide_current_main_with_only_pre_main_prices(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
    })
    mapping_dates = pd.bdate_range("2026-02-02", periods=6)
    mapping = pd.Series(
        ["P2601.DCE"] * 3 + ["P2605.DCE"] * 3,
        index=mapping_dates,
    )
    first_dates = pd.bdate_range(end=mapping_dates[2], periods=5)
    # 新主力只有成为主力以前的旧数据；这不是“当日盘中残组”，不得回退
    # 成以前一合约为截止点的正式结果。
    second_dates = pd.bdate_range(end=mapping_dates[2], periods=5)
    pool = ContractHistoryPool(
        "P.DCE", mapping,
        {
            "P2601.DCE": pd.Series(
                np.linspace(100, 104, 5), index=first_dates),
            "P2605.DCE": pd.Series(
                np.linspace(200, 204, 5), index=second_dates),
        },
        "P2605.DCE",
    )

    recommendations, ranking, _ = recommend_by_contract_history_pool(
        _option(days=1), pool,
        [StrategyCase("c2c", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 6}, target_endpoints=6,
    )

    assert recommendations.empty
    assert ranking["mapping_trailing_days_dropped"].eq(0).all()
    assert ranking["main_contract_asof"].eq("P2605.DCE").all()
    assert ranking["effective_main_contract"].eq("P2605.DCE").all()
    assert not ranking["history_complete"].any()
    assert "P2605.DCE" in ranking.iloc[0]["failure_reason"]


def test_contract_pool_drops_only_proven_intraday_trailing_partial_day(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
    })
    mapping_dates = pd.bdate_range("2026-03-02", periods=6)
    full_dates = pd.bdate_range("2026-02-27", periods=6)
    timestamps = [
        pd.Timestamp(f"{date.date()} {clock}")
        for date in full_dates
        for clock in ("11:30", "15:00")
    ]
    timestamps.append(pd.Timestamp(f"{mapping_dates[-1].date()} 11:30"))
    prices = pd.Series(
        100.0 + np.arange(len(timestamps)) * 0.1,
        index=pd.DatetimeIndex(timestamps),
    )
    pool = ContractHistoryPool(
        "P.DCE",
        pd.Series("P2609.DCE", index=mapping_dates),
        {"P2609.DCE": prices},
        "P2609.DCE",
    )

    recommendations, ranking, _ = recommend_by_contract_history_pool(
        _option(days=1), pool,
        [StrategyCase("c2c", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, target_endpoints=5,
    )

    assert recommendations.empty
    assert ranking["mapping_trailing_days_dropped"].eq(1).all()
    assert ranking["effective_asof_date"].eq(mapping_dates[-2]).all()
    assert ranking["history_complete"].all()
    assert ranking["recommendation_eligible"].all()


def test_contract_pool_uses_each_contracts_own_intraday_steps(monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
    })
    mapping_dates = pd.bdate_range("2026-04-06", periods=4)
    daily_dates = pd.bdate_range(end=mapping_dates[1], periods=3)
    intraday_dates = pd.bdate_range(end=mapping_dates[-1], periods=3)
    intraday_index = pd.DatetimeIndex([
        pd.Timestamp(f"{date.date()} {clock}")
        for date in intraday_dates
        for clock in ("11:30", "15:00")
    ])
    pool = ContractHistoryPool(
        "P.DCE",
        pd.Series(
            ["P2605.DCE"] * 2 + ["P2609.DCE"] * 2,
            index=mapping_dates,
        ),
        {
            "P2605.DCE": pd.Series(
                [100.0, 101.0, 102.0], index=daily_dates),
            "P2609.DCE": pd.Series(
                200.0 + np.arange(len(intraday_index)) * 0.1,
                index=intraday_index,
            ),
        },
        "P2609.DCE",
    )

    recommendations, ranking, windows = recommend_by_contract_history_pool(
        _option(days=1), pool,
        [StrategyCase("c2c", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 4}, target_endpoints=4,
    )

    assert recommendations.empty
    assert ranking["history_complete"].all()
    assert ranking["steps_per_day_min"].eq(1).all()
    assert ranking["steps_per_day_max"].eq(2).all()
    result_steps = {
        window["c2c"]["history_contract_code"]:
            window["c2c"]["steps_per_day"]
        for window in windows["sample"].values()
    }
    assert result_steps == {"P2605.DCE": 1, "P2609.DCE": 2}


def test_rolling_history_requires_exactly_one_close_to_close_baseline():
    prices = pd.Series(
        np.linspace(100.0, 105.0, 6),
        index=pd.date_range("2026-01-05", periods=6, freq="B"),
    )
    kwargs = {
        "backtest_kwargs": {"multiplier": 0, "tc_rate": 0.0},
        "lookbacks": {"week": 3},
        "step_days": 1,
    }

    with pytest.raises(ValueError, match="必须包含一个 close_to_close"):
        recommend_by_rolling_history(
            _option(days=2), prices,
            [StrategyCase("candidate", FixedFreqStrategy(1))],
            **kwargs,
        )

    with pytest.raises(ValueError, match="只能包含一个 close_to_close"):
        recommend_by_rolling_history(
            _option(days=2), prices,
            [
                StrategyCase("c2c_a", CloseToCloseStrategy()),
                StrategyCase("c2c_b", CloseToCloseStrategy()),
            ],
            **kwargs,
        )


def test_rolling_history_ranks_candidates_by_same_window_c2c_improvement(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
        "fixed_freq": 1.0,
        "hedge_band": 8.0,
    })
    prices = pd.Series(
        np.linspace(100.0, 106.0, 7),
        index=pd.date_range("2026-01-05", periods=7, freq="B"),
    )
    cases = [
        StrategyCase("c2c_reference", CloseToCloseStrategy()),
        StrategyCase("earns_less", FixedFreqStrategy(1)),
        StrategyCase(
            "earns_more", HedgeBandStrategy("absolute", threshold=1.0)),
    ]

    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 4}, step_days=1,
    )

    # 排名口径是“日内额外调仓多赚了多少”：每天稳赚 8 元的候选优于基线的
    # 4 元，而每天只赚 1 元的候选劣于基线——即便后者的 RMS 更小。旧口径
    # 按 RMS 排会得到完全相反的顺序，那正是它奖励“把收益也一起抹平”的表现。
    assert recommendations["strategy"].tolist() == ["earns_more"]
    assert ranking["strategy"].tolist() == [
        "earns_more", "c2c_reference", "earns_less",
    ]
    assert ranking["recommendation_eligible"].all()
    assert ranking["comparison_eligible"].all()
    assert ranking["comparison_status"].eq("formal").all()

    baseline = ranking[ranking["strategy"] == "c2c_reference"].iloc[0]
    better = ranking[ranking["strategy"] == "earns_less"].iloc[0]
    worse = ranking[ranking["strategy"] == "earns_more"].iloc[0]
    assert baseline["score"] == pytest.approx(4.0)
    assert baseline["baseline_score"] == pytest.approx(4.0)
    assert baseline["improvement_vs_c2c"] == pytest.approx(0.0)
    assert better["score"] == pytest.approx(1.0)
    assert better["baseline_score"] == pytest.approx(4.0)
    assert better["score_delta_vs_c2c"] == pytest.approx(-3.0)
    assert better["improvement_vs_c2c"] == pytest.approx(0.75)
    assert better["window_win_rate_vs_c2c"] == pytest.approx(1.0)
    assert better["median_window_improvement_vs_c2c"] == pytest.approx(0.75)
    assert worse["score"] == pytest.approx(8.0)
    assert worse["baseline_score"] == pytest.approx(4.0)
    assert worse["score_delta_vs_c2c"] == pytest.approx(4.0)
    assert worse["improvement_vs_c2c"] == pytest.approx(-1.0)
    assert worse["window_win_rate_vs_c2c"] == pytest.approx(0.0)
    assert worse["median_window_improvement_vs_c2c"] == pytest.approx(-1.0)


def test_strict_ranking_uses_combined_l_day_rms_not_equal_segment_votes(
        monkeypatch):
    import pricing.hedge_analysis as hedge_analysis

    prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0],
        index=pd.date_range("2026-06-01", periods=4, freq="B"),
    )
    zero_baseline_endpoint = prices.index[-3]

    class FakeBacktest:
        def __init__(
                self, option, path_source=None, external_path=None,
                strategy=None, steps_per_day=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day

        def run(self):
            is_zero_window = (
                self.external_path.index[-1] == zero_baseline_endpoint)
            if self.strategy.name == "close_to_close":
                daily_pnl = 0.0 if is_zero_window else 10.0
            elif is_zero_window:
                daily_pnl = 100.0
            else:
                daily_pnl = 5.0
            net = np.asarray([0.0, daily_pnl])
            return {
                "net_daily": net,
                "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": self.strategy.name,
                "prices": np.asarray(self.external_path, dtype=float),
                "timestamps": self.external_path.index,
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)
    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=1), prices,
        [
            StrategyCase("c2c_reference", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"sample": 3}, target_endpoints=3,
    )

    candidate = ranking[ranking["strategy"] == "candidate"].iloc[0]
    baseline_rms = np.sqrt((0.0 ** 2 + 10.0 ** 2 + 10.0 ** 2) / 3.0)
    candidate_rms = np.sqrt((100.0 ** 2 + 5.0 ** 2 + 5.0 ** 2) / 3.0)
    expected_advantage = (
        (baseline_rms - candidate_rms) / candidate_rms)
    assert candidate["selection_improvement_vs_c2c"] == pytest.approx(
        expected_advantage)
    assert candidate["selection_metric"] == (
        "strict_lookback_daily_rms_advantage_vs_c2c")
    # 参与相对评分的段数，不是是/否标志：三段里有一段基线 RMS 为 0，
    # 改善率算不出来（_history_improvement_vs_c2c 遇到零基线返回 nan），
    # 所以只有 2 段进了相对口径。
    assert candidate["relative_comparison_windows"] == 2
    assert candidate["median_window_improvement_vs_c2c"] == pytest.approx(0.5)
    assert candidate["window_win_rate_vs_c2c"] == pytest.approx(2.0 / 3.0)
    # 合并 L 日 RMS 仍按上面的公式计算并保留为诊断值，但名次已改由增量
    # 收益决定：candidate 三段合计 110 元、基线 20 元，因此它排第一——
    # 即使它的 RMS 更差。这正是“收益优先、波动其次”的口径变更。
    assert candidate["incremental_pnl_vs_c2c"] == pytest.approx(90.0)
    assert ranking.iloc[0]["strategy"] == "candidate"
    assert recommendations["strategy"].tolist() == ["candidate"]


def test_history_ranking_falls_back_per_row_for_legacy_results():
    import pricing.hedge_analysis as hedge_analysis

    common = {
        "lookback": "week", "lookback_days": 5,
        "recommendation_eligible": True, "comparison_eligible": True,
        "paired_windows": 3, "comparison_coverage": 1.0,
        "window_win_rate_vs_c2c": 0.0,
    }
    ranking = hedge_analysis._rank_history_rows([
        {
            **common, "strategy": "c2c", "strategy_type": "close_to_close",
            "selection_improvement_vs_c2c": 0.0,
            "improvement_vs_c2c": 0.0,
        },
        {
            **common, "strategy": "legacy_candidate",
            "strategy_type": "hedge_band", "improvement_vs_c2c": 0.2,
        },
    ])

    assert ranking["strategy"].tolist() == ["legacy_candidate", "c2c"]

    # 只有新选择字段而没有旧改善列时也不能因默认值的提前求值而 KeyError。
    selection_only = hedge_analysis._rank_history_rows([{
        **common, "strategy": "new_candidate", "strategy_type": "hedge_band",
        "selection_improvement_vs_c2c": 0.1,
        "selection_metric": "mean_bounded_window_advantage_vs_c2c",
    }])
    assert selection_only.iloc[0]["rank"] == 1


def test_rolling_history_zero_c2c_score_never_produces_infinite_improvement(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 0.0,
        "fixed_freq": 1.0,
    })
    prices = pd.Series(
        np.linspace(100.0, 105.0, 6),
        index=pd.date_range("2026-01-05", periods=6, freq="B"),
    )

    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=2), prices,
        [
            StrategyCase("c2c_reference", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 3}, step_days=1,
    )

    candidate = ranking[ranking["strategy"] == "candidate"].iloc[0]
    assert candidate["baseline_score"] == pytest.approx(0.0)
    assert np.isnan(candidate["improvement_vs_c2c"])
    assert np.isnan(candidate["median_window_improvement_vs_c2c"])
    assert candidate["selection_improvement_vs_c2c"] == pytest.approx(-1.0)
    # 基线在该窗口完美对冲（日损益恒为 0），候选有正的日损益，因此按
    # 增量收益口径候选排第一；RMS 口径下则相反。两种口径都不得溢出。
    assert recommendations["strategy"].tolist() == ["candidate"]
    for column in (
            "improvement_vs_c2c", "median_window_improvement_vs_c2c",
            "selection_improvement_vs_c2c",
            "incremental_pnl_vs_c2c", "incremental_sharpe_vs_c2c"):
        values = pd.to_numeric(ranking[column], errors="coerce").to_numpy()
        assert not np.isinf(values).any()


def test_incomplete_strict_history_has_no_pair_or_synthetic_leader(
        monkeypatch):
    _install_deterministic_history_backtest(monkeypatch, {
        "close_to_close": 4.0,
        "fixed_freq": 1.0,
    })
    prices = pd.Series(
        np.linspace(100.0, 103.0, 4),
        index=pd.date_range("2026-01-05", periods=4, freq="B"),
    )

    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=2), prices,
        [
            StrategyCase("c2c_reference", CloseToCloseStrategy()),
            StrategyCase("candidate", FixedFreqStrategy(1)),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1,
    )

    assert recommendations.empty
    assert ranking["strategy"].tolist()[0] == "c2c_reference"
    assert not ranking["comparison_eligible"].any()
    assert not ranking["recommendation_eligible"].any()
    assert ranking["comparison_status"].eq("no_pair").all()
    assert ranking["days_used"].eq(0).all()


def test_rolling_history_recommendation_uses_multiple_windows():
    prices = pd.Series(
        [100, 101, 99, 102, 101, 103, 102, 104],
        index=pd.date_range("2026-01-05", periods=8, freq="B"),
    )
    cases = [
        StrategyCase("daily", CloseToCloseStrategy()),
        StrategyCase("move_2", HedgeBandStrategy("absolute", threshold=2)),
    ]
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1,
    )
    assert len(recommendations) == 1
    assert set(ranking["strategy"]) == {"daily", "move_2"}
    assert ranking["complete_window"].all()
    assert len(windows["week"]) >= 2


def test_rolling_history_uses_real_day_groups_with_irregular_intraday_bars():
    dates = pd.date_range("2026-02-02", periods=7, freq="B")
    session_times = [
        ["11:30", "15:00"],
        ["15:00"],                         # 缺一根 bar
        ["11:30", "13:00", "15:00"],    # 多一根 bar
        ["11:30", "15:00"],
        ["15:00"],
        ["11:30", "15:00"],
        ["11:30", "13:00", "15:00"],
    ]
    timestamps = [
        pd.Timestamp(f"{date.date()} {time}")
        for date, times in zip(dates, session_times)
        for time in times
    ]
    prices = pd.Series(
        100.0 + np.arange(len(timestamps)) * 0.1,
        index=pd.DatetimeIndex(timestamps),
    )
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"recent": 4}, step_days=1,
    )

    # 该用例只验证窗口切片；单独的 C2C 只是固定基准，不构成策略择优。
    assert recommendations.empty
    assert ranking["strategy"].tolist() == ["daily"]
    assert ranking.iloc[0]["complete_window"]
    assert ranking.iloc[0]["comparison_status"] == "formal"
    assert ranking.iloc[0]["rolling_windows"] == 2
    assert ranking.iloc[0]["days_used"] == 4
    assert len(windows["recent"]) == 2

    expected_end_dates = (dates[-3], dates[-1])
    path_lengths = []
    for result, expected_end in zip(
            (item["daily"] for item in windows["recent"].values()),
            expected_end_dates):
        result_groups = result["trading_day_groups"]
        assert len(np.unique(result_groups)) == 3  # Day 0 锚点 + T=2
        assert result["timestamps"][-1].date() == expected_end.date()
        assert result["timestamps"][-1].time().strftime("%H:%M") == "15:00"
        path_lengths.append(len(result["prices"]))
    # 缺/多 bar 保留在相应真实 session 中，后续窗口不会按位置整体错位。
    assert len(set(path_lengths)) > 1


def test_rolling_history_drops_latest_partial_group_and_keeps_complete_windows():
    dates = pd.date_range("2026-04-01", periods=10, freq="B")
    timestamps = [
        pd.Timestamp(f"{date.date()} {time}")
        for date in dates[:9]
        for time in ("11:30", "15:00")
    ] + [pd.Timestamp(f"{dates[9].date()} 11:30")]
    prices = pd.Series(
        100.0 + np.arange(len(timestamps)) * 0.1,
        index=pd.DatetimeIndex(timestamps),
    )

    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1,
    )

    # 该用例只验证残缺尾组处理；没有候选时不把基准包装成推荐结果。
    assert recommendations.empty
    assert ranking["strategy"].tolist() == ["daily"]
    assert ranking.iloc[0]["complete_window"]
    assert ranking.iloc[0]["rolling_windows"] == 3
    assert ranking.iloc[0]["days_used"] == 5
    assert ranking.iloc[0]["trailing_partial_groups_dropped"] == 1
    assert ranking.iloc[0]["skipped_endpoints"] == 0
    assert len(windows["week"]) == 3
    assert all(
        item["daily"]["timestamps"][-1].strftime("%H:%M") == "15:00"
        for item in windows["week"].values()
    )


def test_rolling_fixed_time_missing_bar_does_not_abort_other_strategies():
    dates = pd.date_range("2026-05-04", periods=9, freq="B")
    timestamps = [
        pd.Timestamp(f"{date.date()} {time}")
        for i, date in enumerate(dates)
        for time in (("15:00",) if i == 6 else ("11:30", "15:00"))
    ]
    prices = pd.Series(
        100.0 + np.arange(len(timestamps)) * 0.05,
        index=pd.DatetimeIndex(timestamps),
    )
    cases = [
        StrategyCase("daily", CloseToCloseStrategy()),
        StrategyCase("fixed", FixedTimeStrategy(["11:30", "15:00"])),
    ]

    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1,
    )

    daily_row = ranking[ranking["strategy"] == "daily"].iloc[0]
    fixed_row = ranking[ranking["strategy"] == "fixed"].iloc[0]
    assert daily_row["complete_window"]
    assert daily_row["comparison_status"] == "formal"
    assert not fixed_row["complete_window"]
    assert fixed_row["skipped_endpoints"] > 0
    assert 0 < fixed_row["paired_windows"] < fixed_row["baseline_windows"]
    assert fixed_row["unpaired_windows"] > 0
    assert 0.0 < fixed_row["comparison_coverage"] < 1.0
    assert not fixed_row["comparison_eligible"]
    assert not fixed_row["recommendation_eligible"]
    assert fixed_row["comparison_status"] == "partial_pair"
    successful_fixed_windows = [
        window for window in windows["week"].values()
        if "fixed" in window and "error" not in window["fixed"]
    ]
    paired_c2c_daily = pd.concat([
        result_daily_frame(window["daily"])
        for window in successful_fixed_windows
    ], ignore_index=True)
    expected_paired_baseline_score = float(np.sqrt(np.mean(np.square(
        paired_c2c_daily["net_pnl"].to_numpy(dtype=float)))))
    assert fixed_row["baseline_score"] == pytest.approx(
        expected_paired_baseline_score)
    assert fixed_row["baseline_days_used"] == fixed_row["days_used"]
    # 部分同窗只能留作排错，既不是正式比较，也不是完整历史的诊断资格。
    assert recommendations.empty


def test_rolling_fixed_time_preserves_preconfigured_closed_session_skip():
    dates = pd.date_range("2026-05-04", periods=9, freq="B")
    timestamps = [
        pd.Timestamp(f"{date.date()} {time}")
        for date in dates
        for time in ("11:30", "15:00")
    ]
    prices = pd.Series(
        100.0 + np.arange(len(timestamps)) * 0.05,
        index=pd.DatetimeIndex(timestamps),
    )
    fixed = FixedTimeStrategy(["21:00", "15:00"])
    fixed.set_trading_sessions((
        (datetime.time(9, 0), datetime.time(11, 30)),
        (datetime.time(13, 30), datetime.time(15, 0)),
    ))

    _recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [
            StrategyCase("daily", CloseToCloseStrategy()),
            StrategyCase("fixed", fixed),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1,
    )

    row = ranking[ranking["strategy"] == "fixed"].iloc[0]
    assert row["rolling_windows"] == 3
    assert row["days_used"] == 5
    assert row["paired_windows"] == row["baseline_windows"]
    assert row["comparison_eligible"]
    for window in windows["week"].values():
        result = window["fixed"]
        assert "error" not in result
        assert result["fixed_time_effective_times"] == ("15:00",)
        assert result["fixed_time_skipped_times"] == ("21:00",)


def test_rolling_fixed_time_all_windows_keep_real_failure_reason():
    dates = pd.date_range("2026-05-04", periods=9, freq="B")
    timestamps = [
        pd.Timestamp(f"{date.date()} {time}")
        for i, date in enumerate(dates)
        # 前两日证明数据是日内行情；近期窗口保持完整 bar 数和 15:00
        # 收盘，但刻意用 10:00 替代目标 11:30。
        for time in (("11:30", "15:00") if i < 2 else ("10:00", "15:00"))
    ]
    prices = pd.Series(
        100.0 + np.arange(len(timestamps)) * 0.05,
        index=pd.DatetimeIndex(timestamps),
    )

    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [
            StrategyCase("daily", CloseToCloseStrategy()),
            StrategyCase("fixed", FixedTimeStrategy(["11:30", "15:00"])),
        ],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1, steps_per_day=2,
    )

    row = ranking[ranking["strategy"] == "fixed"].iloc[0]
    assert recommendations.empty
    assert row["eligible_endpoints"] == 3
    assert row["rolling_windows"] == 0
    assert row["skipped_endpoints"] == 3
    assert row["failure_scope"] == "strategy"
    assert "11:30" in row["failure_reason"]
    assert "未能逐交易日组匹配" in row["failure_reason"]
    assert all(
        "_window_error" not in item
        and "11:30" in item["fixed"]["error"]
        and "未能逐交易日组匹配" in item["fixed"]["error"]
        for item in windows["week"].values()
    )


def test_short_lookbacks_use_only_strict_l_day_mtm_paths():
    prices = pd.Series(
        100 + np.sin(np.arange(50) / 3),
        index=pd.date_range("2026-01-05", periods=50, freq="B"),
    )
    cases = [StrategyCase("daily", CloseToCloseStrategy())]
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=22), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5, "month": 20},
    )

    assert recommendations.empty
    assert set(ranking["lookback"]) == {"week", "month"}
    assert set(ranking["strategy"]) == {"daily"}
    assert ranking["complete_window"].all()
    assert len(windows["week"]) == 1
    assert len(windows["month"]) == 1
    first_week = next(iter(windows["week"].values()))["daily"]
    first_month = next(iter(windows["month"].values()))["daily"]
    assert first_week["timestamps"][0] == prices.index[-6]
    assert first_week["timestamps"][-1] == prices.index[-1]
    assert first_week["evaluation_days"] == 5
    assert first_week["remaining_days_at_end"] == 17
    assert first_week["terminal_mode"] == "mark_to_market"
    assert first_month["timestamps"][0] == prices.index[-21]
    assert first_month["evaluation_days"] == 20
    assert first_month["remaining_days_at_end"] == 2
    assert ranking.set_index("lookback")["days_used"].to_dict() == {
        "week": 5,
        "month": 20,
    }


def test_default_rolling_history_covers_all_five_horizons_when_complete():
    prices = pd.Series(
        100.0 + np.sin(np.arange(300) / 15.0),
        index=pd.date_range("2025-01-02", periods=300, freq="B"),
    )
    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=22), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
    )

    assert recommendations.empty
    assert set(ranking["lookback"]) == {
        "week", "month", "quarter", "half_year", "year",
    }
    assert set(ranking["strategy"]) == {"daily"}
    assert ranking["complete_window"].all()


def test_long_maturity_short_lookback_is_formal_mtm_when_l_is_covered():
    prices = pd.Series(
        np.linspace(100, 103, 24),
        index=pd.date_range("2026-01-05", periods=24, freq="B"),
    )
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=22), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5},
    )

    assert recommendations.empty
    assert ranking.iloc[0]["complete_window"]
    assert ranking.iloc[0]["rolling_windows"] == 1
    assert ranking.iloc[0]["eligible_endpoints"] == 1
    assert ranking.iloc[0]["days_used"] == 5
    assert ranking.iloc[0]["terminal_mode"] == "mark_to_market"
    assert len(windows["week"]) == 1
    result = windows["week"]["segment_1"]["daily"]
    assert len(result["prices"]) == 6
    assert result["remaining_days_at_end"] == 17


def test_rolling_history_rebases_absolute_band_for_each_price_window():
    prices = pd.Series(
        [150.0, 160.0, 200.0, 203.0, 204.0],
        index=pd.date_range("2026-03-02", periods=5, freq="B"),
    )
    cases = [
        StrategyCase("daily", CloseToCloseStrategy()),
        StrategyCase(
            "absolute",
            HedgeBandStrategy("absolute", threshold=2.0),
        ),
        StrategyCase(
            "relative",
            HedgeBandStrategy("relative", threshold=0.02),
        ),
    ]
    recommendations, ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        # 本例验证的是带宽随窗口重新基准化，收盘兜底会混入额外触发。
        {"multiplier": 0, "tc_rate": 0.0, "force_day_close_hedge": False},
        lookbacks={"latest": 2}, step_days=1,
    )

    assert len(recommendations) == 1
    assert ranking["complete_window"].all()
    only_window = next(iter(windows["latest"].values()))
    # 原期权 s0=100 时绝对 2 元等价于 2%；窗口从 200 开始后应自动搬运为
    # 4 元，因此 200 -> 203 不触发，与相对 2% 的行为完全一致。
    assert only_window["absolute"]["hedge_triggered"].tolist() == [True, False, True]
    assert np.array_equal(
        only_window["absolute"]["hedge_triggered"],
        only_window["relative"]["hedge_triggered"],
    )


def test_rolling_history_replay_reproduces_each_segment_bar_by_bar():
    """重放配方必须逐 bar 复现原分段结果，否则展示页会与排名口径脱节。"""
    prices = _variable_return_history(2 + 5 + 4)
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase(
            "rv", HedgeBandStrategy(
                "sigma", threshold=0.5,
                sigma_source="realized", window_days=4)),
    ]
    kwargs = {"multiplier": 5.0, "quantity": 2.0, "tc_rate": 0.0}
    _rec, _ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices, cases, kwargs, lookbacks={"week": 5})

    index = history_replay_index(windows)
    assert set(index) == {"week"}
    assert set(index["week"]) == set(windows["week"])
    for window_id, spec in index["week"].items():
        assert set(spec.strategy_names()) == {"c2c", "rv"}
        for name in spec.strategy_names():
            backtest = spec.replay(name)
            original = windows["week"][window_id][name]
            assert np.allclose(
                backtest._results["cumulative_pnl"],
                original["cumulative_pnl"])
            assert np.array_equal(
                backtest._results["hedge_triggered"],
                original["hedge_triggered"])
            # 展示页按逐 bar 明细渲染，长度必须与原结果一致。
            assert len(backtest.to_dataframe()) == len(original["prices"])


def test_history_replay_spec_records_segment_metadata_and_rejects_unknown():
    prices = _variable_return_history(2 + 5 + 4)
    cases = [StrategyCase("c2c", CloseToCloseStrategy())]
    _rec, _ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 5.0, "quantity": 2.0, "tc_rate": 0.0},
        lookbacks={"week": 5})

    spec = history_replay_index(windows)["week"]["segment_1"]
    assert spec.lookback == "week"
    assert spec.window_id == "segment_1"
    assert spec.metadata["history_mode"] == "rolling_history"
    assert spec.metadata["segment_no"] == 1
    assert spec.metadata["evidence_days"] == 5
    with pytest.raises(KeyError, match="没有候选"):
        spec.replay("不存在的候选")


def test_contract_pool_replay_binds_each_segment_to_its_own_contract():
    pool = _two_contract_history_pool()
    cases = [
        StrategyCase("c2c", CloseToCloseStrategy()),
        StrategyCase("band", HedgeBandStrategy("absolute", threshold=1.25)),
    ]
    _rec, _ranking, windows = recommend_by_contract_history_pool(
        _option(days=2), pool, cases,
        {"position": 1, "quantity": 2.0, "multiplier": 0.0, "tc_rate": 0.0},
        lookbacks={"sample": 8})

    index = history_replay_index(windows)
    specs = index["sample"]
    assert specs
    codes = {
        spec.metadata["contract_code"] for spec in specs.values()}
    assert codes == {"P2601.DCE", "P2605.DCE"}
    for window_id, spec in specs.items():
        assert spec.metadata["history_mode"] == "product_contract_pool"
        for name in spec.strategy_names():
            backtest = spec.replay(name)
            original = windows["sample"][window_id][name]
            assert np.allclose(
                backtest._results["cumulative_pnl"],
                original["cumulative_pnl"])
            # 每段行情只来自该分段自己的具体合约，绝不跨合约拼接。
            assert np.allclose(
                backtest._results["prices"][0],
                original["prices"][0])


def test_history_replay_index_skips_windows_without_successful_candidate():
    prices = pd.Series(
        [100.0, 101.0, 99.5, 102.0, 101.0, 103.0, 102.0, 104.0],
        index=pd.bdate_range("2026-01-05", periods=8))
    _rec, _ranking, windows = recommend_by_rolling_history(
        _option(days=2), prices,
        [StrategyCase("c2c", CloseToCloseStrategy())],
        {"multiplier": 0.0, "tc_rate": 0.0}, lookbacks={"sample": 5})
    windows["sample"]["broken"] = {"_window_error": "构造失败"}

    index = history_replay_index(windows)

    assert "broken" not in index["sample"]
    assert set(index["sample"]).issubset(set(windows["sample"]))


def _intraday_15min_series(days, seed=11):
    """A 股 15 分钟右标签序列：每日 16 根 bar（09:45 ~ 15:00）。"""
    clocks = []
    for start, total in ((9 * 60 + 30, 120), (13 * 60, 120)):
        done = 0
        while done < total:
            done += 15
            mark = start + done
            clocks.append(f"{mark // 60:02d}:{mark % 60:02d}")
    dates = pd.bdate_range("2026-01-05", periods=days)
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {c}")
                            for d in dates for c in clocks])
    rng = np.random.default_rng(seed)
    steps = 0.2 * np.sqrt(1.0 / (ANNUAL_DAYS * len(clocks))) * (
        rng.standard_normal(len(idx) - 1))
    values = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(steps))))
    return pd.Series(values, index=idx), len(clocks)


def test_greeks_are_only_repriced_on_hedging_bars():
    """Greeks 只在调仓 bar 重算，其余 bar 沿用上一根。

    触发判定只看价格，Delta 也只在调仓时被 _compute_target 读取，因此
    非调仓 bar 上重算 Greeks 纯属为展示付费——1 分钟粒度下这是回测的
    主要开销。
    """
    prices, spd = _intraday_15min_series(days=4)
    result = HedgeBacktest(
        _option(days=4), prices, steps_per_day=spd, multiplier=0,
        strategy=FixedTimeStrategy("14:45"),
    ).run()

    triggered = np.asarray(result["hedge_triggered"], dtype=bool)
    assert triggered.sum() < len(triggered), "场景里必须存在非调仓 bar"
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        values = np.asarray(result[name], dtype=float)
        for i in range(1, len(values)):
            if not triggered[i]:
                assert values[i] == values[i - 1], (
                    f"{name} 在非调仓 bar {i} 上被重算了")


def test_hedging_bars_carry_true_greeks():
    """调仓 bar 上的 Greeks 必须是该 bar 的真值，而非沿用值。

    参照组用 ``FixedFreqStrategy(1)``：它每根 bar 都触发，因此走全量
    ``get_greeks`` 路径。期权状态只由交易日边界推进、与是否调仓无关，
    所以它给出的就是逐 bar 真值。
    """
    prices, spd = _intraday_15min_series(days=3)
    lazy_result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=spd, multiplier=0,
        strategy=FixedTimeStrategy("14:45")).run()
    full_result = HedgeBacktest(
        _option(days=3), prices, steps_per_day=spd, multiplier=0,
        strategy=FixedFreqStrategy(hedge_freq=1)).run()

    triggered = np.flatnonzero(
        np.asarray(lazy_result["hedge_triggered"], dtype=bool))
    assert len(triggered) >= 3, "参照场景至少要有几次调仓才有意义"
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        lazy = np.asarray(lazy_result[name], dtype=float)[triggered]
        full = np.asarray(full_result[name], dtype=float)[triggered]
        assert np.array_equal(lazy, full), f"{name} 在调仓 bar 上不是真值"

    # 反证懒算确实生效：非调仓 bar 上与全量值不同。
    non_trigger = np.setdiff1d(
        np.arange(1, len(lazy_result["delta"])), triggered)
    assert not np.array_equal(
        np.asarray(lazy_result["delta"], dtype=float)[non_trigger],
        np.asarray(full_result["delta"], dtype=float)[non_trigger])


def test_lazy_greeks_do_not_change_pnl_delta_or_trades():
    """懒算只影响展示用的高阶 Greeks，绝不能碰盈亏与调仓。"""
    prices, spd = _intraday_15min_series(days=4)
    result = HedgeBacktest(
        _option(days=4), prices, steps_per_day=spd, multiplier=0,
        strategy=HedgeBandStrategy(band_type="relative", threshold=0.004),
        tc_rate=0.0005,
    ).run()

    # 盈亏分解恒等式在懒算路径下仍然成立。
    net = np.asarray(result["net_daily"], dtype=float)
    rebuilt = (np.asarray(result["hedge_daily"], dtype=float)
               + np.asarray(result["option_daily"], dtype=float)
               - np.asarray(result["tc_paid"], dtype=float))
    assert np.allclose(net, rebuilt, rtol=0, atol=0)
    assert np.allclose(
        np.asarray(result["cumulative_pnl"], dtype=float),
        np.cumsum(net), rtol=0, atol=0)
    # Delta 与实际持仓仍然逐 bar 对应（multiplier=0 即不取整）。
    assert np.allclose(
        np.asarray(result["shares"], dtype=float)[result["hedge_triggered"]],
        np.asarray(result["delta"], dtype=float)[result["hedge_triggered"]])


def _a_share_1min_days(n_days, drop_from_first=0, drop_from_last=0):
    """A 股 1 分钟序列（09:30~11:30 + 13:00~15:00，每日 242 根）。

    Wind 实测每日 bar 数在 240~242 之间浮动——没有成交的分钟不返回 bar，
    尾盘无成交时末根停在 14:59。这里用参数复刻这两种缺失位置。
    """
    clocks = []
    for start, total in ((9 * 60 + 30, 120), (13 * 60, 120)):
        for step in range(total + 1):
            mark = start + step
            clocks.append(f"{mark // 60:02d}:{mark % 60:02d}")
    dates = pd.bdate_range("2026-03-02", periods=n_days)
    stamps = []
    for i, date in enumerate(dates):
        day_clocks = clocks
        if i == 0 and drop_from_first:
            day_clocks = clocks[:-drop_from_first]
        elif i == n_days - 1 and drop_from_last:
            day_clocks = clocks[:-drop_from_last]
        stamps.extend(pd.Timestamp(f"{date.date()} {c}") for c in day_clocks)
    index = pd.DatetimeIndex(stamps)
    return pd.Series(
        100.0 + np.arange(len(index), dtype=float) * 0.001, index=index)


def _last_day_close(prices):
    last_date = prices.index[-1].date()
    return prices.index[prices.index.date == last_date][-1].strftime("%H:%M")


@pytest.mark.parametrize("dropped", [1, 10, 30])
def test_historical_group_sparsity_never_blocks_backtest(dropped):
    """中间交易日组早已收盘，尾盘无成交不该阻断回测。

    Wind 分钟行情按成交返回 bar，510050 这类高流动性 ETF 也有约四分之一
    的交易日末根停在 14:59。历史组的稀疏是数据特性，不是盘中残段。
    """
    prices = _a_share_1min_days(3, drop_from_first=dropped)
    first_date = prices.index[0].date()
    first_day = prices.index[prices.index.date == first_date]
    assert len(first_day) == 242 - dropped

    result = HedgeBacktest(
        _option(days=2), prices, steps_per_day=242, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()
    assert result["n_days"] == 2


def test_terminal_group_missing_one_tail_bar_counts_as_closed():
    """样本末尾缺最后一根（末 14:59）仍是已收盘的交易日。"""
    prices = _a_share_1min_days(2, drop_from_last=1)
    assert _last_day_close(prices) == "14:59"

    result = HedgeBacktest(
        _option(days=2), prices, steps_per_day=242, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()
    assert result["n_days"] == 2


def test_terminal_group_still_mid_session_is_rejected():
    """样本末尾停在 14:50 可能是“今天还没收盘”，必须继续拒绝。"""
    prices = _a_share_1min_days(2, drop_from_last=10)
    assert _last_day_close(prices) == "14:50"

    with pytest.raises(ValueError, match=r"交易日组不完整"):
        HedgeBacktest(
            _option(days=2), prices, steps_per_day=242, multiplier=0,
            strategy=CloseToCloseStrategy(),
        )


def test_close_gap_minutes_handles_overnight_session_close():
    """夜盘收盘（02:30）与末 bar 相减不得因跨午夜得出错误间隔。"""
    from pricing.hedge_backtest import _close_gap_minutes
    assert _close_gap_minutes(
        datetime.time(2, 29), datetime.time(2, 30)) == 1
    assert _close_gap_minutes(
        datetime.time(14, 59), datetime.time(15, 0)) == 1
    assert _close_gap_minutes(
        datetime.time(14, 0), datetime.time(15, 0)) == 60
    # 末时刻晚于典型收盘属于异常数据，应得到远超容差的间隔。
    assert _close_gap_minutes(
        datetime.time(15, 1), datetime.time(15, 0)) > 60


_A_SHARE_CLOCK_SESSIONS = (
    (datetime.time(9, 30), datetime.time(11, 30)),
    (datetime.time(13, 0), datetime.time(15, 0)),
)


def _fixed_strategy(times, sessions=_A_SHARE_CLOCK_SESSIONS):
    strategy = FixedTimeStrategy(times)
    if sessions is not None:
        strategy.set_trading_sessions(sessions)
    return strategy


def test_fixed_time_prefers_exact_bar_over_backfill():
    """目标时刻本身有 bar 时必须在那一根触发，不能提前。"""
    strategy = _fixed_strategy("15:00")
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:58"),
           "next_timestamp": pd.Timestamp("2026-03-02 14:59")}
    assert strategy.should_hedge(ctx) is False
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:59"),
           "next_timestamp": pd.Timestamp("2026-03-02 15:00")}
    assert strategy.should_hedge(ctx) is False
    ctx = {"timestamp": pd.Timestamp("2026-03-02 15:00"),
           "next_timestamp": pd.Timestamp("2026-03-03 09:30")}
    assert strategy.should_hedge(ctx) is True


def test_fixed_time_backfills_to_last_bar_before_target():
    """目标时刻没有 bar 时，用之前最近一根承接（实务上就是最后一笔）。"""
    strategy = _fixed_strategy("15:00")
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:58"),
           "next_timestamp": pd.Timestamp("2026-03-02 14:59")}
    assert strategy.should_hedge(ctx) is False
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:59"),
           "next_timestamp": pd.Timestamp("2026-03-03 09:30")}
    assert strategy.should_hedge(ctx) is True


def test_fixed_time_backfill_respects_lunch_break():
    """11:30 缺失时由 11:29 承接，而不是等到下午第一根。"""
    strategy = _fixed_strategy("11:30")
    ctx = {"timestamp": pd.Timestamp("2026-03-02 11:29"),
           "next_timestamp": pd.Timestamp("2026-03-02 13:00")}
    assert strategy.should_hedge(ctx) is True


def test_fixed_time_backfill_never_crosses_into_another_session():
    """日盘末根不得顶替夜盘时刻——两者相距 8 小时且不同 session。"""
    sessions = (
        (datetime.time(21, 0), datetime.time(23, 0)),
        (datetime.time(9, 0), datetime.time(15, 0)),
    )
    strategy = _fixed_strategy("23:00", sessions=sessions)
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:59"),
           "next_timestamp": pd.Timestamp("2026-03-03 09:00")}
    assert strategy.should_hedge(ctx) is False
    # 夜盘自己的末根仍然可以承接 23:00。
    ctx = {"timestamp": pd.Timestamp("2026-03-02 22:59"),
           "next_timestamp": pd.Timestamp("2026-03-03 09:00")}
    assert strategy.should_hedge(ctx) is True


def test_fixed_time_backfill_has_a_distance_ceiling():
    """回退距离有上限，避免用几小时前的价格冒充目标时刻。"""
    strategy = _fixed_strategy("15:00", sessions=None)
    assert strategy.can_serve(datetime.time(14, 59), datetime.time(15, 0))
    assert strategy.can_serve(datetime.time(14, 0), datetime.time(15, 0))
    assert not strategy.can_serve(datetime.time(13, 59), datetime.time(15, 0))
    # 不能用目标之后的 bar 回补——那是 look-ahead。
    assert not strategy.can_serve(datetime.time(15, 1), datetime.time(15, 0))


def test_fixed_time_backfill_triggers_each_target_once_per_day():
    strategy = _fixed_strategy("15:00")
    first = {"timestamp": pd.Timestamp("2026-03-02 14:59"),
             "next_timestamp": pd.Timestamp("2026-03-03 09:30")}
    assert strategy.should_hedge(first) is True
    assert strategy.should_hedge(first) is False
    # 次日重新计数。
    second = {"timestamp": pd.Timestamp("2026-03-03 14:59"),
              "next_timestamp": pd.Timestamp("2026-03-04 09:30")}
    assert strategy.should_hedge(second) is True


def test_fixed_time_trade_count_is_invariant_to_granularity():
    """同一段行情下，调仓次数不应随采样粒度改变。

    这正是让择优可以统一用 1 分钟的前提：固定时刻候选不会因为粒度变细
    而丢失调仓。
    """
    fine = _a_share_1min_days(4)
    # 降采样成 15 分钟右标签网格（09:45、10:00 …… 15:00）。
    coarse_clocks = set()
    for start, total in ((9 * 60 + 30, 120), (13 * 60, 120)):
        for step in range(15, total + 1, 15):
            mark = start + step
            coarse_clocks.add(f"{mark // 60:02d}:{mark % 60:02d}")
    coarse = fine[[t.strftime("%H:%M") in coarse_clocks for t in fine.index]]

    counts = []
    for series, spd in ((fine, 242), (coarse, 16)):
        result = HedgeBacktest(
            _option(days=3), series, steps_per_day=spd, multiplier=0,
            strategy=_fixed_strategy("11:30,15:00"),
        ).run()
        counts.append(int(np.sum(result["hedge_triggered"])))
    assert counts[0] == counts[1]


def test_adjacent_targets_do_not_steal_each_others_bar():
    """相邻目标不得互相抢占同一根 bar。

    15:00 缺失时由 14:59 承接，而 14:59 本身也是目标——若先匹配的那个把
    bar 独占，另一个目标当天就凭空丢失了。
    """
    strategy = _fixed_strategy("14:59,15:00")
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:59"),
           "next_timestamp": pd.Timestamp("2026-03-03 09:30")}
    assert strategy.should_hedge(ctx) is True
    # 两个目标都应记为当日已消费，次日才重新计数。
    assert strategy._triggered == {
        (datetime.date(2026, 3, 2), datetime.time(14, 59)),
        (datetime.date(2026, 3, 2), datetime.time(15, 0)),
    }


def test_later_target_still_waits_for_a_closer_bar():
    """有更接近目标的 bar 时不得提前消费该目标。"""
    strategy = _fixed_strategy("14:30,15:00")
    # 14:29 可以承接 14:30，但 15:00 后面还有 14:59，应继续等待。
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:29"),
           "next_timestamp": pd.Timestamp("2026-03-02 14:59")}
    assert strategy.should_hedge(ctx) is True
    assert (datetime.date(2026, 3, 2), datetime.time(15, 0)) \
        not in strategy._triggered
    ctx = {"timestamp": pd.Timestamp("2026-03-02 14:59"),
           "next_timestamp": pd.Timestamp("2026-03-03 09:30")}
    assert strategy.should_hedge(ctx) is True


def _a_share_1min_days_missing_midday(n_days, drop_per_day=2):
    """每日都缺几根盘中 bar：根数不到声明值，但末根仍是正常收盘 15:00。"""
    clocks = []
    for start, total in ((9 * 60 + 30, 120), (13 * 60, 120)):
        for step in range(total + 1):
            mark = start + step
            clocks.append(f"{mark // 60:02d}:{mark % 60:02d}")
    dates = pd.bdate_range("2026-03-02", periods=n_days)
    stamps = []
    for i, date in enumerate(dates):
        # 去掉的是盘中 bar，不动首尾，保证末根仍为 15:00。
        day = [c for k, c in enumerate(clocks)
               if not (0 < k < len(clocks) - 1
                       and k % 97 < drop_per_day)]
        stamps.extend(pd.Timestamp(f"{date.date()} {c}") for c in day)
    index = pd.DatetimeIndex(stamps)
    return pd.Series(
        100.0 + np.arange(len(index), dtype=float) * 0.001, index=index)


def test_short_window_without_any_typical_length_day_still_validates():
    """短窗口里可能没有任何一天达到全样本的典型 bar 数。

    Wind 分钟行情按成交返回 bar，每日根数在 240~242 浮动。若用“bar 数达标”
    当门槛去推断正常收盘时刻，最近 5 个交易日很可能一天都不达标，
    expected_close 退化成 None，进而落到过严的根数判定上，把一整个正常的
    “近周”窗口判死——这正是策略优选里近周空白的成因。
    """
    prices = _a_share_1min_days_missing_midday(5)
    per_day = {}
    for stamp in prices.index:
        per_day[stamp.date()] = per_day.get(stamp.date(), 0) + 1
    assert max(per_day.values()) < 242, "构造前提：没有一天达到声明的 242 根"
    last_date = prices.index[-1].date()
    assert prices.index[prices.index.date == last_date][-1].strftime(
        "%H:%M") == "15:00"

    result = HedgeBacktest(
        _option(days=4), prices, steps_per_day=242, multiplier=0,
        strategy=CloseToCloseStrategy(),
    ).run()
    assert result["n_days"] == 4


def test_single_group_sample_still_falls_back_to_bar_count():
    """只有一个交易日组时无从统计正常收盘，仍按声明根数判定。"""
    prices = _a_share_1min_days(1, drop_from_last=30)
    with pytest.raises(ValueError, match=r"交易日组不完整"):
        HedgeBacktest(
            _option(days=1), prices, steps_per_day=242, multiplier=0,
            strategy=CloseToCloseStrategy(),
        )


def test_ranking_objective_switches_between_profit_and_sharpe(monkeypatch):
    """两个排名口径都基于“相对日内不动的增量”，但侧重不同。

    构造一个高增量、高波动的候选和一个低增量、极稳的候选：收益口径应选
    前者，性价比口径应选后者。
    """
    import pricing.hedge_analysis as hedge_analysis

    prices = pd.Series(
        np.linspace(100.0, 106.0, 7),
        index=pd.date_range("2026-01-05", periods=7, freq="B"),
    )

    class FakeBacktest:
        def __init__(self, option, path_source=None, external_path=None,
                     strategy=None, steps_per_day=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day

        def run(self):
            n = len(self.external_path)
            rng = np.random.default_rng(7)
            if self.strategy.name == "close_to_close":
                net = np.zeros(n)
            elif self.strategy.name == "fixed_freq":
                # 增量大但忽高忽低。
                net = np.zeros(n)
                net[1:] = 10.0 + rng.normal(0.0, 40.0, n - 1)
            else:
                # 增量小但几乎不波动。
                net = np.zeros(n)
                net[1:] = 3.0 + rng.normal(0.0, 0.5, n - 1)
            return {
                "net_daily": net, "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": self.strategy.name,
                "prices": np.asarray(self.external_path, dtype=float),
                "timestamps": self.external_path.index,
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)
    cases = [
        StrategyCase("基线", CloseToCloseStrategy()),
        StrategyCase("高增量高波动", FixedFreqStrategy(1)),
        StrategyCase("低增量极稳", HedgeBandStrategy("absolute", threshold=1.0)),
    ]
    common = dict(lookbacks={"week": 4}, step_days=1)

    by_pnl = recommend_by_rolling_history(
        _option(days=2), prices, cases, {"multiplier": 0, "tc_rate": 0.0},
        objective="incremental_pnl", **common)[1]
    by_sharpe = recommend_by_rolling_history(
        _option(days=2), prices, cases, {"multiplier": 0, "tc_rate": 0.0},
        objective="incremental_sharpe", **common)[1]

    assert by_pnl.iloc[0]["strategy"] == "高增量高波动"
    assert by_sharpe.iloc[0]["strategy"] == "低增量极稳"
    assert by_pnl.iloc[0]["selection_objective"] == "incremental_pnl"
    assert by_sharpe.iloc[0]["selection_objective"] == "incremental_sharpe"


def test_unknown_ranking_objective_is_rejected():
    with pytest.raises(ValueError, match="未知排名口径"):
        recommend_by_rolling_history(
            _option(days=2),
            pd.Series(np.linspace(100.0, 104.0, 5),
                      index=pd.date_range("2026-01-05", periods=5, freq="B")),
            [StrategyCase("c2c", CloseToCloseStrategy())],
            {"multiplier": 0, "tc_rate": 0.0},
            lookbacks={"week": 3}, objective="max_profit")


def test_rerank_history_switches_objective_without_rerunning(monkeypatch):
    """切换排名依据只重排已有结果，不得重跑回测。

    两个口径所需的指标在回测时已一并算出；一年 1 分钟行情重跑一次要几十
    秒，换个看法不该让人再等一遍。
    """
    import pricing.hedge_analysis as hedge_analysis
    from pricing import rerank_history

    runs = {"count": 0}
    prices = pd.Series(
        np.linspace(100.0, 106.0, 7),
        index=pd.date_range("2026-01-05", periods=7, freq="B"))

    class FakeBacktest:
        def __init__(self, option, path_source=None, external_path=None,
                     strategy=None, steps_per_day=1, **kwargs):
            self.external_path = external_path
            self.strategy = strategy
            self.steps_per_day = steps_per_day

        def run(self):
            runs["count"] += 1
            n = len(self.external_path)
            rng = np.random.default_rng(11)
            net = np.zeros(n)
            if self.strategy.name == "fixed_freq":
                net[1:] = 10.0 + rng.normal(0.0, 40.0, n - 1)
            elif self.strategy.name != "close_to_close":
                net[1:] = 3.0 + rng.normal(0.0, 0.5, n - 1)
            return {
                "net_daily": net, "tc_paid": np.zeros_like(net),
                "steps_per_day": self.steps_per_day,
                "strategy_name": self.strategy.name,
                "prices": np.asarray(self.external_path, dtype=float),
                "timestamps": self.external_path.index,
            }

    monkeypatch.setattr(hedge_analysis, "HedgeBacktest", FakeBacktest)
    cases = [
        StrategyCase("基线", CloseToCloseStrategy()),
        StrategyCase("高增量高波动", FixedFreqStrategy(1)),
        StrategyCase("低增量极稳", HedgeBandStrategy("absolute", threshold=1.0)),
    ]
    _rec, ranking, _win = recommend_by_rolling_history(
        _option(days=2), prices, cases, {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 4}, step_days=1, objective="incremental_pnl")
    runs_after_backtest = runs["count"]
    assert ranking.iloc[0]["strategy"] == "高增量高波动"

    rec2, reranked = rerank_history(ranking, "incremental_sharpe")

    # 关键：重排期间一次回测都不应发生。
    assert runs["count"] == runs_after_backtest
    assert reranked.iloc[0]["strategy"] == "低增量极稳"
    assert reranked.iloc[0]["selection_objective"] == "incremental_sharpe"
    assert rec2["strategy"].tolist() == ["低增量极稳"]
    # 重排不改变任何指标值，只改变名次。
    for name in ("高增量高波动", "低增量极稳"):
        before = ranking[ranking["strategy"] == name].iloc[0]
        after = reranked[reranked["strategy"] == name].iloc[0]
        assert after["incremental_pnl_vs_c2c"] == before["incremental_pnl_vs_c2c"]
        assert after["window_pnl"] == before["window_pnl"]


def test_rerank_history_tolerates_empty_ranking():
    from pricing import rerank_history
    rec, ranking = rerank_history(pd.DataFrame(), "incremental_pnl")
    assert rec.empty and ranking.empty
