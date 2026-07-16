from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing import (
    CloseToCloseStrategy,
    FixedFreqStrategy,
    FixedTimeStrategy,
    HedgeBacktest,
    HedgeBandStrategy,
    LOOKBACK_DAYS,
    Option_Vanilla,
    PriceIntervalStrategy,
    StrategyCase,
    compare_strategies,
    result_daily_frame,
    summarize_strategy_result,
    recommend_by_lookback,
    recommend_by_rolling_history,
)
from pricing.constants import ANNUAL_DAYS
from pricing.hedge_backtest import _infer_intraday_steps, _trading_day_groups


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


def test_legacy_fixed_frequency_remains_backend_api_compatibility_only():
    prices = np.array([100, 101, 102, 103, 104], dtype=float)
    bt = HedgeBacktest(
        _option(), prices, hedge_freq=2, multiplier=0, strategy=None)
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
        strategy=FixedTimeStrategy(["11:30"]),
    ).run()
    # 末 bar 始终是到期平仓，因此固定时刻触发为 1/3/5/7，另含 0/8。
    assert np.flatnonzero(result["hedge_triggered"]).tolist() == [0, 1, 3, 5, 7, 8]
    assert result["timestamps"] is not None


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
    with pytest.raises(ValueError, match="未全部匹配|缺失"):
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
    with pytest.raises(ValueError, match=r"2026-01-06.*缺失 \[11:30\]"):
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


def test_wind_intraday_uses_ceil_metadata_for_tail_bar(
        monkeypatch, capsys):
    import pricing.wind_data as wind_data

    dates = pd.date_range("2026-03-02", periods=3, freq="B")
    bar_times = [
        "09:00", "09:30", "10:00", "10:30", "11:00",
        "11:30", "13:30", "14:00", "14:30", "15:00",
    ]
    idx = pd.DatetimeIndex([
        f"{date.date()} {bar_time}"
        for date in dates for bar_time in bar_times
    ])
    series = pd.Series(
        100.0 + np.arange(len(idx), dtype=float) * 0.1,
        index=idx,
    )
    monkeypatch.setattr(
        wind_data, "get_intraday_close",
        lambda *args, **kwargs: series,
    )
    # 570 / 60 需向上取整为 10，Wind 索引也包含第 10 根尾 bar。
    monkeypatch.setattr(
        wind_data, "get_trading_minutes_per_day", lambda code: 570,
    )

    bt = HedgeBacktest.from_wind(
        _option(days=2), "AU.SHF", "2026-03-02", "2026-03-04",
        bar_size="60", steps_per_day=None, multiplier=0,
    )

    assert bt.steps_per_day == 10
    assert len(bt.prices) == 20
    output = capsys.readouterr().out
    assert "570 分钟无法被 bar_size=60 整除" in output


def test_wind_intraday_rejects_missing_tail_bar_for_non_divisible_session(
        monkeypatch):
    import pricing.wind_data as wind_data

    bar_times = [
        "09:00", "09:30", "10:00", "10:30", "11:00",
        "11:30", "13:30", "14:00", "14:30",
    ]
    series = pd.Series(
        100.0 + np.arange(len(bar_times), dtype=float) * 0.1,
        index=pd.DatetimeIndex([
            f"2026-03-02 {bar_time}" for bar_time in bar_times
        ]),
    )
    monkeypatch.setattr(
        wind_data, "get_intraday_close", lambda *args, **kwargs: series)
    monkeypatch.setattr(
        wind_data, "get_trading_minutes_per_day", lambda code: 570)

    with pytest.raises(ValueError, match=r"交易日组不完整|典型/声明为 10"):
        HedgeBacktest.from_wind(
            _option(days=1), "AU.SHF", "2026-03-02", "2026-03-02",
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
        wind_data, "get_trading_minutes_per_day", lambda code: 240)

    with pytest.raises(ValueError, match=r"交易日组不完整|典型/声明为 4"):
        HedgeBacktest.from_wind(
            _option(days=1), "510050.SH", "2026-07-15", "2026-07-15",
            bar_size="60", steps_per_day=None, multiplier=0,
        )


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


def test_summarize_strategy_result_returns_comparison_compatible_metrics():
    result = {
        "net_daily": np.array([-1.0, 2.0, 3.0, -2.0, 1.0]),
        "tc_paid": np.array([1.0, 0.2, 0.3, 0.4, 0.1]),
        "steps_per_day": 2,
        "hedge_triggered": np.array([True, True, False, True, True]),
        "shares": np.array([1.0, 1.0, 2.0, 2.0, 0.0]),
        "prices": np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        "strategy_name": "hedge_band",
        "hedging_error": 3.0,
    }
    metadata = {"description": "绝对间隔=2", "nested": {"items": [1]}}

    row = summarize_strategy_result(result, "固定间隔(绝对=2)", metadata)

    assert row["strategy"] == "固定间隔(绝对=2)"
    assert row["strategy_type"] == "hedge_band"
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


def test_lookback_year_uses_project_annual_days():
    assert LOOKBACK_DAYS["year"] == int(ANNUAL_DAYS)


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

    assert len(recommendations) == 1
    assert ranking.iloc[0]["complete_window"]
    assert ranking.iloc[0]["rolling_windows"] == 4
    assert ranking.iloc[0]["days_used"] == 8
    assert len(windows["recent"]) == 4

    expected_end_dates = dates[-4:]
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

    assert len(recommendations) == 1
    assert ranking.iloc[0]["complete_window"]
    assert ranking.iloc[0]["rolling_windows"] == 5
    assert ranking.iloc[0]["trailing_partial_groups_dropped"] == 1
    assert ranking.iloc[0]["skipped_endpoints"] == 0
    assert len(windows["week"]) == 5
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

    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=2), prices, cases,
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"week": 5}, step_days=1,
    )

    daily_row = ranking[ranking["strategy"] == "daily"].iloc[0]
    fixed_row = ranking[ranking["strategy"] == "fixed"].iloc[0]
    assert daily_row["complete_window"]
    assert not fixed_row["complete_window"]
    assert fixed_row["skipped_endpoints"] > 0
    assert recommendations["strategy"].tolist() == ["daily"]


def test_short_lookbacks_use_endpoints_even_when_maturity_is_longer():
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

    assert set(recommendations["lookback"]) == {"week", "month"}
    assert ranking["complete_window"].all()
    assert len(windows["week"]) == 2
    assert len(windows["month"]) == 5
    # 最早的周回测起点早于“最近一周”观察期起点。
    first_week = next(iter(windows["week"].values()))["daily"]
    assert first_week["timestamps"][0] < prices.index[-5]


def test_default_rolling_history_recommends_all_four_horizons_when_complete():
    prices = pd.Series(
        100.0 + np.sin(np.arange(300) / 15.0),
        index=pd.date_range("2025-01-02", periods=300, freq="B"),
    )
    recommendations, ranking, _ = recommend_by_rolling_history(
        _option(days=22), prices,
        [StrategyCase("daily", CloseToCloseStrategy())],
        {"multiplier": 0, "tc_rate": 0.0},
    )

    assert set(recommendations["lookback"]) == {
        "week", "month", "quarter", "year",
    }
    assert ranking["complete_window"].all()


def test_long_maturity_short_lookback_keeps_partial_diagnostics():
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
    assert not ranking.iloc[0]["complete_window"]
    assert ranking.iloc[0]["rolling_windows"] == 1
    assert ranking.iloc[0]["eligible_endpoints"] == 1
    assert len(windows["week"]) == 1


def test_rolling_history_rebases_absolute_band_for_each_price_window():
    prices = pd.Series(
        [150.0, 160.0, 200.0, 203.0, 204.0],
        index=pd.date_range("2026-03-02", periods=5, freq="B"),
    )
    cases = [
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
        {"multiplier": 0, "tc_rate": 0.0},
        lookbacks={"latest": 1}, step_days=1,
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
