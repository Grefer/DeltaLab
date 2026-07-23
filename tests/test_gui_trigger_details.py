from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gui_app import BacktestApp
from pricing.hedge_backtest import HedgeBacktest


def _raw_detail_frame():
    index = pd.DatetimeIndex([
        "2026-07-20 21:00",
        "2026-07-20 23:00",
        "2026-07-21 09:00",
        "2026-07-21 11:30",
        "2026-07-21 13:30",
        "2026-07-21 15:00",
    ], name="日期")
    return pd.DataFrame({
        "标的价格": np.arange(100.0, 106.0),
        "持仓": np.ones(6),
        "每日净盈亏": np.arange(6.0),
    }, index=index)


def _detail_backtest(*, knocked_out=False):
    frame = _raw_detail_frame()
    result = {
        "hedge_triggered": np.array(
            [True, False, True, True, False, True]),
        "strategy_hedge_triggered": np.array(
            [False, False, True, False, False, False]),
        "day_close_fallback_triggered": np.array(
            [False, False, False, True, False, False]),
        "knocked_out": knocked_out,
        "steps_per_day": 3,
    }
    return SimpleNamespace(
        _results=result,
        to_dataframe=lambda: frame.copy(),
    )


def test_trigger_detail_filters_rows_and_preserves_lifecycle_sources():
    detail, positions = BacktestApp._hedge_trigger_detail_frame(
        _detail_backtest())

    assert positions.tolist() == [0, 2, 3, 5]
    assert detail.index.tolist() == _raw_detail_frame().index[
        [0, 2, 3, 5]].tolist()
    assert detail["触发来源"].tolist() == [
        "初始建仓", "策略触发", "收盘保底", "到期平仓"]
    # “触发”不是“成交”：目标持仓没有变化的触发记录仍会保留。
    assert detail["持仓"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_trigger_detail_labels_knockout_terminal_close():
    detail, _ = BacktestApp._hedge_trigger_detail_frame(
        _detail_backtest(knocked_out=True))

    assert detail["触发来源"].iloc[-1] == "敲出平仓"


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"hedge_triggered": np.array([True, False])},
        {
            "hedge_triggered": np.ones(6, dtype=bool),
            "strategy_hedge_triggered": np.ones(5, dtype=bool),
        },
        {
            "hedge_triggered": np.ones(6, dtype=bool),
            "day_close_fallback_triggered": np.ones((2, 3), dtype=bool),
        },
    ],
)
def test_trigger_detail_rejects_missing_or_misaligned_markers(result):
    bt = SimpleNamespace(
        _results=result,
        to_dataframe=lambda: _raw_detail_frame(),
    )

    with pytest.raises(ValueError, match="缺少|长度不一致"):
        BacktestApp._hedge_trigger_detail_frame(bt)


def test_detail_index_formatter_keeps_intraday_time():
    value = pd.Timestamp("2026-07-21 11:30")

    assert BacktestApp._format_detail_index(
        value, include_time=True) == "2026-07-21 11:30"
    assert BacktestApp._format_detail_index(
        value, include_time=False) == "2026-07-21"


def _dataframe_result(timestamps):
    values = np.arange(3.0)
    return {
        "prices": values + 100.0,
        "opt_value": values,
        "delta": values,
        "gamma": values,
        "vega": values,
        "theta": values,
        "rho": values,
        "shares": values,
        "hedge_daily": values,
        "option_daily": values,
        "tc_paid": values,
        "net_daily": values,
        "cumulative_pnl": values,
        "timestamps": timestamps,
    }


def test_to_dataframe_prefers_run_timestamps_after_early_termination():
    run_timestamps = pd.date_range(
        "2026-07-01", periods=3, freq="D")
    bt = object.__new__(HedgeBacktest)
    bt._results = _dataframe_result(run_timestamps)
    # 数据源元信息仍可能保留完整到期窗口；敲出后的明细必须使用 run 结果。
    bt._wind_meta = {
        "dates": pd.date_range("2026-07-01", periods=6, freq="D"),
    }

    detail = bt.to_dataframe()

    assert detail.index.equals(run_timestamps.rename("日期"))


def test_to_dataframe_trims_legacy_metadata_index_to_result_length():
    bt = object.__new__(HedgeBacktest)
    bt._results = _dataframe_result(None)
    bt._wind_meta = {
        "dates": pd.date_range("2026-07-01", periods=6, freq="D"),
    }

    detail = bt.to_dataframe()

    assert detail.index.equals(
        bt._wind_meta["dates"][:3].rename("日期"))
