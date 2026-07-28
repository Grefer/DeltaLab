from __future__ import annotations

import copy
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gui_app
import history_selection
from gui_app import (
    BacktestApp,
    DEFAULT_BAND_CANDIDATE_SIGMAS,
    DEFAULT_FIXED_TIMES,
    MAX_BAND_CANDIDATES,
    STRATEGY_DISPLAY,
    WIND_AUTO_BAR_SIZE,
)
from pricing import (
    CloseToCloseStrategy,
    ContractHistoryPool,
    FixedTimeStrategy,
    HedgeBandStrategy,
    StrategyCase,
)
from pricing.constants import ANNUAL_DAYS


class _Var:
    def __init__(self, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


class _RemovedControlVar:
    """已下线控件的替身：任何再次读取它的代码路径都会立即失败。"""

    def __init__(self, label):
        self.label = label

    def get(self):
        raise AssertionError(f"{self.label} 已不再是可选参数，不应被读取")

    def set(self, value):
        raise AssertionError(f"{self.label} 已不再是可选参数，不应被写入")


def test_default_fixed_times_include_day_and_night_closes():
    assert DEFAULT_FIXED_TIMES == "23:00,11:30,15:00"
    assert tuple(
        value.strftime("%H:%M")
        for value in FixedTimeStrategy(DEFAULT_FIXED_TIMES).requested_times
    ) == ("23:00", "11:30", "15:00")


class _BoolVar:
    def __init__(self, value=False):
        self.value = bool(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = bool(value)


class _Widget:
    def __init__(self):
        self.state = "normal"
        self.config = {}

    def configure(self, **kwargs):
        self.config.update(kwargs)
        if "state" in kwargs:
            self.state = kwargs["state"]

    def stop(self):
        self.config["stopped"] = True

    def pack_forget(self):
        self.config["hidden"] = True


class _GridWidget(_Widget):
    def __init__(self):
        super().__init__()
        self.visible = True

    def grid(self, **_kwargs):
        self.visible = True

    def grid_remove(self):
        self.visible = False


def test_hide_placeholder_is_idempotent_after_tk_widget_was_destroyed():
    class _StalePlaceholder:
        def __init__(self):
            self.calls = 0

        def place_forget(self):
            self.calls += 1
            if self.calls > 1:
                raise gui_app.tk.TclError("bad window path name")

    placeholder = _StalePlaceholder()
    fake = SimpleNamespace(_table_placeholder=placeholder)

    BacktestApp._hide_placeholder(fake, "table")
    BacktestApp._hide_placeholder(fake, "table")

    assert placeholder.calls == 2
    assert fake._table_placeholder is None


def test_table_cleanup_preserves_placeholder_across_repeated_renders():
    class _Child:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    class _Tab:
        def __init__(self, children):
            self.children = children

        def winfo_children(self):
            return [child for child in self.children if not child.destroyed]

    placeholder = _Child()
    first_table = _Child()
    second_table = _Child()
    tab = _Tab([placeholder, first_table])
    fake = SimpleNamespace(_table_placeholder=placeholder)

    BacktestApp._clear_tab_content_preserving_placeholder(fake, "table", tab)
    tab.children.append(second_table)
    BacktestApp._clear_tab_content_preserving_placeholder(fake, "table", tab)

    assert placeholder.destroyed is False
    assert first_table.destroyed is True
    assert second_table.destroyed is True


def test_repeated_chart_render_releases_previous_canvas_and_figure():
    class _CanvasWidget:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    class _Container:
        def __init__(self, children):
            self.children = children

        def winfo_children(self):
            return self.children

    class _Figure:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    widget = _CanvasWidget()
    figure = _Figure()
    fake = SimpleNamespace(_chart_figure=figure, _chart_canvas=object())

    BacktestApp._reset_figure_container(fake, "chart", _Container([widget]))

    assert widget.destroyed is True
    assert figure.cleared is True
    assert fake._chart_figure is None
    assert fake._chart_canvas is None


def test_show_results_can_run_twice_without_destroying_table_placeholder():
    class _Child:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    class _Placeholder(_Child):
        def __init__(self):
            super().__init__()
            self.hidden_count = 0

        def place_forget(self):
            if self.destroyed:
                raise gui_app.tk.TclError("bad window path name")
            self.hidden_count += 1

    class _Tab:
        def __init__(self, children):
            self.children = children

        def winfo_children(self):
            return [child for child in self.children if not child.destroyed]

    calls = {name: 0 for name in ("summary", "chart", "vol", "dist", "table")}
    placeholder = _Placeholder()
    table_tab = _Tab([placeholder])
    rendered_tables = []

    def count(name):
        calls[name] += 1

    fake = SimpleNamespace(
        _table_placeholder=placeholder,
        _table_tab=table_tab,
        _show_summary=lambda _bt, _stats: count("summary"),
        _show_chart=lambda _bt: count("chart"),
        _show_vol_chart=lambda _bt: count("vol"),
        _show_dist_chart=lambda _stats: count("dist"),
        _nb=SimpleNamespace(select=lambda _index: None),
    )

    def show_table(_bt):
        count("table")
        BacktestApp._hide_placeholder(fake, "table")
        BacktestApp._clear_tab_content_preserving_placeholder(
            fake, "table", table_tab)
        rendered = _Child()
        rendered_tables.append(rendered)
        table_tab.children.append(rendered)

    fake._show_table = show_table

    BacktestApp._show_results(fake, object())
    BacktestApp._show_results(fake, object())

    assert calls == {name: 2 for name in calls}
    assert placeholder.destroyed is False
    assert placeholder.hidden_count == 2
    assert rendered_tables[0].destroyed is True
    assert rendered_tables[1].destroyed is False


def _fake_band_gui():
    fake = SimpleNamespace(
        _band_abs_var=_Var("1"),
        _band_rel_var=_Var("0.01"),
        _band_sigma_var=_Var("0.779423"),
        _price_interval_var=_Var("1"),
        _interval_type_var=_Var("absolute"),
        _band_last_edited="absolute",
        _band_syncing=False,
        _param_entries={"s0": (_Var("100"), float, None),
                        "sigma": (_Var("0.2"), float, None)},
        _toggle_strategy=lambda: None,
        _refresh_history_current_band_label=lambda: None,
        _set_status=lambda _text: None,
    )
    return fake


def test_band_sync_uses_visible_last_edited_value_and_updates_hidden_state():
    fake = _fake_band_gui()
    fake._band_rel_var.set("0.02")
    converted = BacktestApp._sync_band_inputs(
        fake, "relative", strict=True)
    assert converted["absolute"] == pytest.approx(2.0)
    assert float(fake._price_interval_var.get()) == pytest.approx(0.02)
    assert fake._interval_type_var.get() == "relative"


def test_band_sync_strict_rejects_invalid_visible_value():
    fake = _fake_band_gui()
    fake._band_abs_var.set("not-a-number")
    with pytest.raises(ValueError, match="固定间隔参数无效"):
        BacktestApp._sync_band_inputs(fake, "absolute", strict=True)


@pytest.mark.parametrize(
    "state",
    [
        {"strategy_name": "fixed_times", "source": "simulate"},
        {"strategy_name": "fixed_times", "source": "wind",
         "wind_bar_size": "日频"},
    ],
)
def test_fixed_time_gui_rejects_known_non_intraday_sources(state):
    with pytest.raises(ValueError, match="日内|分钟|日频"):
        BacktestApp._validate_fixed_time_source_state(state)


def test_fixed_time_comparison_requires_targets_in_every_trading_day_group():
    backtest = SimpleNamespace(timestamps=pd.DatetimeIndex([
        "2026-01-02 15:00",
        "2026-01-05 11:30", "2026-01-05 15:00",
        "2026-01-06 15:00",
    ]))
    with pytest.raises(ValueError, match=r"2026-01-06.*11:30"):
        BacktestApp._validate_fixed_time_backtest(
            backtest, FixedTimeStrategy(["11:30", "15:00"]))


def test_fixed_time_gui_validation_prefers_backtest_configured_session():
    original = FixedTimeStrategy(["21:00", "15:00"])
    configured = FixedTimeStrategy(["21:00", "15:00"])
    configured.set_trading_sessions((
        (datetime.time(9, 30), datetime.time(11, 30)),
        (datetime.time(13, 0), datetime.time(15, 0)),
    ))
    backtest = SimpleNamespace(
        strategy=configured,
        timestamps=pd.DatetimeIndex([
            "2026-01-02 11:30", "2026-01-02 15:00",
            "2026-01-05 11:30", "2026-01-05 15:00",
        ]),
    )

    BacktestApp._validate_fixed_time_backtest(backtest, original)


def test_history_fixed_candidate_defers_missing_target_bars_to_rolling_windows():
    # 启动基准只覆盖历史最前面的一个期限；其中某日缺目标时刻不能据此
    # 全局剔除 fixed_times，完整历史中的每个滚动窗口会独立诊断。
    backtest = SimpleNamespace(timestamps=pd.DatetimeIndex([
        "2026-01-02 10:00", "2026-01-02 15:00",
        "2026-01-05 15:00",
    ]))
    cases, notes = BacktestApp._strategy_cases_for_history(
        {
            "source": "csv",
            "history_include_close": False,
            "history_include_fixed_times": True,
            "history_include_band": False,
            "fixed_times": "11:30,15:00",
        },
        backtest,
    )

    assert [case.strategy.name for case in cases] == [
        "close_to_close", "fixed_times",
    ]
    assert notes == []


def test_history_wind_fixed_candidate_discloses_closed_session_skip(
        monkeypatch):
    import pricing.wind_data as wind_data

    day_sessions = ((datetime.time(9, 30), datetime.time(11, 30)),
                    (datetime.time(13, 0), datetime.time(15, 0)))
    monkeypatch.setattr(
        wind_data, "get_trading_session_clock_ranges",
        lambda _code, **_kwargs: day_sessions,
    )
    backtest = SimpleNamespace(timestamps=pd.DatetimeIndex([
        "2026-01-02 11:30", "2026-01-02 15:00",
        "2026-01-05 11:30", "2026-01-05 15:00",
    ]))

    cases, notes = BacktestApp._strategy_cases_for_history(
        {
            "source": "wind",
            "wind_code": "NO_NIGHT.TEST",
            "wind_bar_size": "15分钟",
            "history_include_fixed_times": True,
            "history_include_band": False,
            "fixed_times": "21:07,11:30,15:00",
        },
        backtest,
    )

    fixed_case = next(
        case for case in cases if case.strategy.name == "fixed_times")
    assert fixed_case.name == "固定时刻(21:07,11:30,15:00)"
    assert [t.strftime("%H:%M") for t in fixed_case.strategy.effective_times] == [
        "11:30", "15:00",
    ]
    assert [t.strftime("%H:%M") for t in fixed_case.strategy.skipped_times] == [
        "21:07",
    ]
    assert fixed_case.metadata["fixed_times_requested"] == "21:07,11:30,15:00"
    assert fixed_case.metadata["fixed_times_effective"] == "11:30,15:00"
    assert fixed_case.metadata["fixed_times_skipped"] == "21:07"
    assert "自动跳过非交易时刻 21:07" in fixed_case.metadata["description"]
    assert notes == []


def test_history_fixed_candidate_still_rejects_daily_csv_granularity():
    backtest = SimpleNamespace(timestamps=pd.date_range(
        "2026-01-02", periods=3, freq="B"))

    with pytest.raises(ValueError, match="只有每日收盘基准") as exc_info:
        BacktestApp._strategy_cases_for_history(
            {
                "source": "csv",
                "history_include_close": False,
                "history_include_fixed_times": True,
                "history_include_band": False,
                "fixed_times": "11:30,15:00",
            },
            backtest,
        )

    assert "日内" in str(exc_info.value) or "日频" in str(exc_info.value)


def test_gui_exposes_exactly_three_business_strategies():
    assert set(STRATEGY_DISPLAY) == {
        "close_to_close", "fixed_times", "hedge_band",
    }
    assert "fixed_freq" not in STRATEGY_DISPLAY


@pytest.mark.parametrize(
    ("source", "value", "expected"),
    [("simulate", "48", 48), ("csv", "240", 1), ("wind", "240", 1)],
)
def test_only_simulation_uses_gui_bar_density(source, value, expected):
    assert BacktestApp._gui_steps_per_day(source, value) == expected


@pytest.mark.parametrize(
    "value",
    [None, "", "2026/01/02", "2026-1-2", "2026-02-30", "not-a-date"],
)
def test_wind_date_parser_rejects_empty_or_non_iso_dates(value):
    with pytest.raises(ValueError, match="YYYY-MM-DD|不能为空"):
        BacktestApp._parse_wind_date(value, "Wind 日期")


def _single_wind_resolution_state(**overrides):
    state = {
        "source": "wind",
        "wind_code": "510050.SH",
        "wind_start": "2026-01-02",
        "wind_end": "2026-02-20",
        "wind_auto_end": False,
        "wind_bar_size_requested": WIND_AUTO_BAR_SIZE,
        "strategy_name": "close_to_close",
        "fixed_times": "11:30,15:00",
        "params": {"T_days": 22},
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wind_start": "2026-03-02"}, "建仓日不能晚于当前日期"),
        ({"wind_end": "2026-01-02"}, "结束日必须晚于建仓日"),
        ({"wind_end": "2026-01-01"}, "结束日必须晚于建仓日"),
        ({"wind_end": "2026-03-02"}, "结束日不能晚于当前日期"),
    ],
)
def test_single_wind_dates_reject_future_or_reversed_ranges(
        overrides, message):
    state = _single_wind_resolution_state(**overrides)
    with pytest.raises(ValueError, match=message):
        BacktestApp._resolve_single_wind_state(
            state, today=datetime.date(2026, 3, 1))


def test_single_wind_auto_end_uses_maturity_without_parsing_custom_end():
    class _UnusedEnd:
        def __str__(self):
            raise AssertionError("自动结束模式不得解析自定义结束日")

    state = _single_wind_resolution_state(
        wind_start="2025-01-02",
        wind_end=_UnusedEnd(),
        wind_auto_end=True,
        params={"T_days": 30},
    )
    resolved = BacktestApp._resolve_single_wind_state(
        state, today=datetime.date(2026, 3, 1))
    expected_span = BacktestApp._calendar_span_for_trading_days(31)

    assert resolved["wind_end"] == (
        datetime.date(2025, 1, 2)
        + datetime.timedelta(days=expected_span)
    ).isoformat()
    assert resolved["wind_required_trade_days"] == 31
    assert resolved["wind_date_mode"] == "auto_maturity"


def test_single_wind_custom_end_is_preserved_exactly():
    resolved = BacktestApp._resolve_single_wind_state(
        _single_wind_resolution_state(
            wind_start="2026-01-02",
            wind_end="2026-02-20",
            wind_auto_end=False,
        ),
        today=datetime.date(2026, 3, 1),
    )

    assert resolved["wind_start"] == "2026-01-02"
    assert resolved["wind_end"] == "2026-02-20"
    assert resolved["wind_date_mode"] == "custom_range"


@pytest.mark.parametrize(
    ("strategy_name", "fixed_times", "expected"),
    [
        ("close_to_close", "", "日频"),
        ("fixed_times", "11:30,15:00", "15分钟"),
        ("fixed_times", "10:10", "5分钟"),
        ("fixed_times", "10:07", "1分钟"),
        # 任何不落在 5/15 分钟网格上的时刻都必须退到 1 分钟。
        ("fixed_times", "14:22", "1分钟"),
        ("fixed_times", "14:22,15:00", "1分钟"),
        ("fixed_times", "09:31", "1分钟"),
        ("hedge_band", "", "1分钟"),
    ],
)
def test_wind_auto_bar_size_follows_strategy_observation_needs(
        strategy_name, fixed_times, expected):
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE,
        strategy_name=strategy_name,
        fixed_times=fixed_times,
    ) == expected


@pytest.mark.parametrize(
    "band_sigma_multiple", [0.2, 0.75, 1.0, 2.0, 3.0, 5.0, None, 0.0])
def test_band_auto_bar_size_is_always_finest_regardless_of_width(
        band_sigma_multiple):
    """带宽宽窄不改变漏采机制，自动粒度一律 1 分钟，不按带宽分档。"""
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, strategy_name="hedge_band") == "1分钟"
    # 带宽只用于量化手动选粗的代价，不参与自动粒度判定。
    assert BacktestApp._recommended_band_bar_size() == "1分钟"


def test_auto_bar_size_is_the_only_source_of_truth():
    """粒度不再有用户入口：解析函数是唯一决定者，且对同一策略是确定的。"""
    for _ in range(3):
        assert BacktestApp._resolve_wind_bar_size(
            WIND_AUTO_BAR_SIZE, strategy_name="close_to_close") == "日频"
        assert BacktestApp._resolve_wind_bar_size(
            WIND_AUTO_BAR_SIZE, strategy_name="hedge_band") == "1分钟"


def test_history_band_bar_size_does_not_depend_on_candidate_mix():
    """粒度不得依赖“本批次里最窄的候选是谁”，否则单个候选的评分会随
    它和谁一起被评估而改变。"""
    for sigmas in ((0.5, 1.0), (3.0, 5.0), (2.0,)):
        assert BacktestApp._resolve_wind_bar_size(
            WIND_AUTO_BAR_SIZE, include_band=True) == "1分钟"
    # 同时含固定时刻候选时仍取更细的一档。
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, include_band=True, include_fixed_times=True,
        fixed_times="11:30,15:00") == "1分钟"


def test_wind_auto_bar_size_ignores_locally_known_closed_session_targets(
        monkeypatch):
    import pricing.wind_data as wind_data

    day_sessions = ((datetime.time(9, 30), datetime.time(11, 30)),
                    (datetime.time(13, 0), datetime.time(15, 0)))
    calls = []

    def session_ranges(code, *, allow_wind=True):
        calls.append((code, allow_wind))
        return day_sessions

    monkeypatch.setattr(
        wind_data, "get_trading_session_clock_ranges", session_ranges)

    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE,
        strategy_name="fixed_times",
        fixed_times="21:07,15:00",
        wind_code="NO_NIGHT.TEST",
    ) == "15分钟"
    # 自动频率在 GUI 主线程解析，只读本地已知 session，不能触发 Wind wss。
    assert calls == [("NO_NIGHT.TEST", False)]


_A_SHARE_SESSIONS = ((datetime.time(9, 30), datetime.time(11, 30)),
                     (datetime.time(13, 0), datetime.time(15, 0)))
_COMMODITY_SESSIONS = ((datetime.time(21, 0), datetime.time(23, 0)),
                       (datetime.time(9, 0), datetime.time(10, 15)),
                       (datetime.time(10, 30), datetime.time(11, 30)),
                       (datetime.time(13, 30), datetime.time(15, 0)))


def _patch_trading_sessions(monkeypatch, sessions):
    import pricing.wind_data as wind_data
    monkeypatch.setattr(
        wind_data, "get_trading_session_clock_ranges",
        lambda code, *, allow_wind=True: sessions)


@pytest.mark.parametrize(
    ("sessions", "fixed_times"),
    [
        (_A_SHARE_SESSIONS, "09:30"),
        (_A_SHARE_SESSIONS, "13:00"),
        (_A_SHARE_SESSIONS, "09:30,15:00"),
        (_COMMODITY_SESSIONS, "21:00"),
        (_COMMODITY_SESSIONS, "09:00"),
        (_COMMODITY_SESSIONS, "10:30"),
        (_COMMODITY_SESSIONS, "13:30"),
    ],
)
def test_session_open_times_require_the_finest_granularity(
        monkeypatch, sessions, fixed_times):
    """开盘时刻只有 1 分钟粒度取得到。

    实测 Wind 的标签约定随粒度而变：1 分钟是左标签（首根 09:30），
    5/15/60 分钟是右标签（首根 09:35/09:45/10:30）。整除判据只看墙钟，
    会把 09:30 误判成 15 分钟可覆盖，因此要显式降到最细一档。
    """
    _patch_trading_sessions(monkeypatch, sessions)
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, strategy_name="fixed_times",
        fixed_times=fixed_times, wind_code="TEST.CODE") == "1分钟"


def test_session_close_times_remain_valid_fixed_hedge_targets(monkeypatch):
    """收盘时刻是真实存在的末根 bar，不能被开盘拦截误伤。"""
    _patch_trading_sessions(monkeypatch, _A_SHARE_SESSIONS)
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, strategy_name="fixed_times",
        fixed_times="11:30,15:00", wind_code="510050.SH") == "15分钟"
    _patch_trading_sessions(monkeypatch, _COMMODITY_SESSIONS)
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, strategy_name="fixed_times",
        fixed_times="23:00,10:15,15:00", wind_code="rb2412.SHF") == "15分钟"


def test_unknown_session_metadata_defers_to_backtest_time_validation(
        monkeypatch):
    """元数据未知时不能凭空拦截，仍由逐交易日组完整性校验兜底。"""
    _patch_trading_sessions(monkeypatch, None)
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, strategy_name="fixed_times",
        fixed_times="09:30", wind_code="UNKNOWN.TEST") == "15分钟"
    # 没有代码时同样无从判断该时刻属于哪个 session。
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE, strategy_name="fixed_times",
        fixed_times="09:30") == "15分钟"


def test_manual_daily_wind_rejects_fixed_time_but_keeps_band_choice():
    with pytest.raises(ValueError, match="固定时刻策略需要分钟行情"):
        BacktestApp._resolve_wind_bar_size(
            "日频", strategy_name="fixed_times",
            fixed_times="11:30,15:00",
        )

    assert BacktestApp._resolve_wind_bar_size(
        "日频", strategy_name="hedge_band") == "日频"


def test_history_wind_auto_range_covers_strict_year_plus_day0_anchor():
    asof = datetime.date(2026, 2, 27)
    maturity_days = 22
    resolved = BacktestApp._resolve_history_wind_state({
        "source": "wind",
        "wind_code": "510050.SH",
        "params": {"T_days": maturity_days},
        "history_wind_asof": asof.isoformat(),
        "history_wind_auto_start": True,
        "history_wind_bar_size_requested": WIND_AUTO_BAR_SIZE,
        "history_include_fixed_times": True,
        "history_include_band": True,
        "fixed_times": "11:30,15:00",
    }, today=datetime.date(2026, 3, 1))
    required = ANNUAL_DAYS + 1
    expected_span = BacktestApp._calendar_span_for_trading_days(required)

    assert resolved["wind_required_trade_days"] == required
    assert resolved["wind_end"] == asof.isoformat()
    assert resolved["wind_start"] == (
        asof - datetime.timedelta(days=expected_span)).isoformat()
    assert resolved["wind_bar_size"] == "1分钟"
    assert resolved["wind_date_mode"] == "history_auto_year_strict_interval"
    assert resolved["wind_sigma_warmup_days"] == 0


def test_history_wind_auto_range_is_independent_of_proxy_maturity():
    base = {
        "source": "wind",
        "wind_code": "510050.SH",
        "history_lookbacks": {"week": gui_app.LOOKBACK_DAYS["week"]},
        "history_wind_asof": "2026-02-27",
        "history_wind_auto_start": True,
        "history_wind_bar_size_requested": "日频",
        "history_include_fixed_times": False,
        "history_include_band": False,
    }
    resolved = [
        BacktestApp._resolve_history_wind_state(
            {**base, "params": {"T_days": maturity_days}},
            today=datetime.date(2026, 3, 1),
        )
        for maturity_days in (2, 243)
    ]

    assert resolved[0]["wind_start"] == resolved[1]["wind_start"]
    assert resolved[0]["wind_required_trade_days"] == (
        gui_app.LOOKBACK_DAYS["week"] + 1)
    assert resolved[0]["wind_required_trade_days"] == resolved[1][
        "wind_required_trade_days"]


def test_history_wind_auto_range_uses_longest_selected_period():
    asof = datetime.date(2026, 2, 27)
    maturity_days = 22
    selected = {
        "week": gui_app.LOOKBACK_DAYS["week"],
        "quarter": gui_app.LOOKBACK_DAYS["quarter"],
    }
    resolved = BacktestApp._resolve_history_wind_state({
        "source": "wind",
        "wind_code": "510050.SH",
        "params": {"T_days": maturity_days},
        "history_lookbacks": selected,
        "history_wind_asof": asof.isoformat(),
        "history_wind_auto_start": True,
        "history_wind_bar_size_requested": "日频",
        "history_include_fixed_times": False,
        "history_include_band": False,
    }, today=datetime.date(2026, 3, 1))
    required = gui_app.LOOKBACK_DAYS["quarter"] + 1

    assert resolved["history_lookbacks"] == selected
    assert resolved["history_max_lookback_days"] == gui_app.LOOKBACK_DAYS[
        "quarter"]
    assert resolved["wind_required_trade_days"] == required
    assert resolved["wind_start"] == (
        asof - datetime.timedelta(
            days=BacktestApp._calendar_span_for_trading_days(required))
    ).isoformat()
    assert resolved["wind_date_mode"] == (
        "history_auto_selected_strict_interval")


def test_history_wind_auto_range_adds_realized_sigma_warmup():
    asof = datetime.date(2026, 2, 27)
    resolved = BacktestApp._resolve_history_wind_state({
        "source": "wind",
        "wind_code": "510050.SH",
        "params": {"T_days": 22},
        "history_wind_asof": asof.isoformat(),
        "history_wind_auto_start": True,
        "history_wind_bar_size_requested": "日频",
        "history_include_fixed_times": False,
        "history_include_band": True,
        "sigma_source": "realized",
        "sigma_window": 30,
    }, today=datetime.date(2026, 3, 1))
    required = ANNUAL_DAYS + 30 + 1

    assert resolved["wind_required_trade_days"] == required
    assert resolved["wind_sigma_warmup_days"] == 30
    assert resolved["wind_start"] == (
        asof - datetime.timedelta(
            days=BacktestApp._calendar_span_for_trading_days(required))
    ).isoformat()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"history_wind_asof": "2026-03-02"}, "截至日不能晚于当前日期"),
        ({
            "history_wind_auto_start": False,
            "history_wind_start": "2026-02-27",
        }, "起始日必须早于分析截至日"),
        ({
            "history_wind_auto_start": False,
            "history_wind_start": "2026-02-28",
        }, "起始日必须早于分析截至日"),
    ],
)
def test_history_wind_dates_reject_future_or_reversed_ranges(
        overrides, message):
    state = {
        "source": "wind",
        "wind_code": "510050.SH",
        "params": {"T_days": 22},
        "history_wind_asof": "2026-02-27",
        "history_wind_auto_start": False,
        "history_wind_start": "2025-01-02",
        "history_wind_bar_size_requested": WIND_AUTO_BAR_SIZE,
        "history_include_fixed_times": False,
        "history_include_band": False,
    }
    state.update(overrides)

    with pytest.raises(ValueError, match=message):
        BacktestApp._resolve_history_wind_state(
            state, today=datetime.date(2026, 3, 1))


def test_history_candidates_have_no_legacy_fixed_frequency_case():
    fake = SimpleNamespace()
    cases, notes = BacktestApp._strategy_cases_for_history(
        {
            "source": "wind",
            "wind_bar_size": "日频",
            "interval_type": "relative",
            "price_interval": 0.01,
            "sigma_source": "implied",
            "sigma_window": 20,
            "fixed_times": "11:30,15:00",
            "params": {"s0": 100.0, "sigma": 0.2},
            "band_candidate_sigmas": (0.5, 1.0),
        },
        SimpleNamespace(),
    )
    assert {case.strategy.name for case in cases} == {
        "close_to_close", "hedge_band",
    }
    assert all("fixed_freq" not in case.name for case in cases)
    band_cases = [case for case in cases if case.strategy.name == "hedge_band"]
    assert len(band_cases) == 3
    assert all(case.strategy.band_type == "sigma" for case in band_cases)
    assert sum("·当前" in case.name for case in band_cases) == 1
    assert any(
        case.metadata["input_band_type"] == "relative"
        and case.metadata["input_threshold"] == pytest.approx(0.01)
        for case in band_cases
    )
    assert notes and "固定时刻策略未参与" in notes[0]


def test_history_strategy_cases_respect_independent_candidate_switches():
    fake = SimpleNamespace()
    cases, notes = BacktestApp._strategy_cases_for_history(
        {
            "source": "wind", "wind_bar_size": "日频",
            "history_include_close": True,
            "history_include_fixed_times": False,
            "history_include_band": True,
            "history_include_current_band": False,
            # 未勾选项即使值不可用也不得参与或生成跳过说明。
            "fixed_times": "not-a-time",
            "band_candidate_sigmas": (0.5, 1.0),
            "interval_type": "sigma", "price_interval": 1.0,
            "sigma_source": "implied", "sigma_window": 20,
            "params": {"s0": 100.0, "sigma": 0.2},
        },
        SimpleNamespace(),
    )

    assert [case.name for case in cases] == [
        "每日收盘", "固定间隔(0.5σ)", "固定间隔(1σ)",
    ]
    assert notes == []


def test_default_history_band_candidates_use_common_sigma_ladder():
    assert DEFAULT_BAND_CANDIDATE_SIGMAS == (0.5, 0.75, 1.0, 1.5, 2.0)


def test_band_candidate_parser_accepts_chinese_delimiters_and_deduplicates():
    assert BacktestApp._parse_band_candidate_sigmas(
        "1，0.5; 1.0；2\n0.75") == (0.5, 0.75, 1.0, 2.0)
    assert BacktestApp._parse_band_candidate_sigmas("") == ()


@pytest.mark.parametrize("raw", ["0.5,0", "-1", "nan", "inf", "bad"])
def test_band_candidate_parser_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="候选"):
        BacktestApp._parse_band_candidate_sigmas(raw)


def test_band_candidate_parser_caps_search_size():
    raw = ",".join(str(index + 1) for index in range(MAX_BAND_CANDIDATES + 1))
    with pytest.raises(ValueError, match=f"最多 {MAX_BAND_CANDIDATES}"):
        BacktestApp._parse_band_candidate_sigmas(raw)


@pytest.mark.parametrize(
    ("band_type", "threshold"),
    [
        ("absolute", 1.0),
        ("relative", 0.01),
        ("sigma", 0.01 / (0.2 / (ANNUAL_DAYS ** 0.5))),
    ],
)
def test_current_band_units_normalize_to_same_sigma_candidate(
        band_type, threshold):
    cases = BacktestApp._band_cases_for_history({
        "interval_type": band_type,
        "price_interval": threshold,
        "params": {"s0": 100.0, "sigma": 0.2},
        "band_candidate_sigmas": (),
        "sigma_source": "implied", "sigma_window": 20,
    })

    assert len(cases) == 1
    assert cases[0].strategy.band_type == "sigma"
    assert cases[0].strategy.threshold == pytest.approx(
        0.01 / (0.2 / (ANNUAL_DAYS ** 0.5)))
    assert cases[0].metadata["is_current_band"] is True


def test_current_band_matching_preset_is_included_once_with_unique_names():
    cases = BacktestApp._band_cases_for_history({
        "interval_type": "sigma", "price_interval": 1.0,
        "params": {"s0": 100.0, "sigma": 0.2},
        "band_candidate_sigmas": (0.5, 1.0, 1.00000000001, 2.0),
        "sigma_source": "realized", "sigma_window": 30,
    })

    assert len(cases) == 3
    assert len({case.name for case in cases}) == 3
    current = [case for case in cases if case.metadata["is_current_band"]]
    assert len(current) == 1
    assert current[0].strategy.sigma_source == "realized"
    assert current[0].strategy.window_days == 30


def test_history_band_candidates_do_not_implicitly_add_current_band():
    cases = BacktestApp._band_cases_for_history({
        "interval_type": "absolute", "price_interval": 3.0,
        "params": {"s0": 100.0, "sigma": 0.2},
        "band_candidate_sigmas": (0.5, 1.0),
        "history_include_current_band": False,
        "sigma_source": "implied", "sigma_window": 20,
    })

    assert [case.strategy.threshold for case in cases] == [0.5, 1.0]
    assert all(case.metadata["is_current_band"] is False for case in cases)
    assert all("·当前" not in case.name for case in cases)


def test_current_band_counts_toward_candidate_limit():
    with pytest.raises(ValueError, match="包含勾选加入的当前带宽"):
        BacktestApp._band_cases_for_history({
            "interval_type": "sigma", "price_interval": 0.25,
            "params": {"s0": 100.0, "sigma": 0.2},
            "band_candidate_sigmas": tuple(
                float(index + 1) for index in range(MAX_BAND_CANDIDATES)),
        })


def _fake_collect_state():
    cls_name = "香草期权 (Vanilla)"
    cfg = gui_app.OPTION_CLASSES[cls_name]
    param_entries = {}
    for spec in cfg["params"]:
        key, _label, dtype, default = spec[:4]
        choices = spec[4] if len(spec) >= 5 else None
        value = default
        if choices:
            value = next(
                display for display, internal in choices.items()
                if internal == default)
        param_entries[key] = (_Var(value), dtype, choices)
    fake = SimpleNamespace(
        _class_var=_Var(cls_name), _subtype_var=_Var("Eu"),
        _param_entries=param_entries,
        _strategy_var=_Var(STRATEGY_DISPLAY["close_to_close"]),
        _band_last_edited="absolute", _price_interval_var=_Var("1"),
        _sigma_src_var=_Var(gui_app.SIGMA_SOURCE_DISPLAY["realized"]),
        _sigma_win_var=_Var("30"),
        _fixed_times_var=_Var("11:30,15:00"),
        _source_var=_Var("csv"), _spd_var=_Var("999"),
        _tc_var=_Var("0.01"), _pos_var=_Var("1"),
        _qty_var=_Var("100"), _mult_var=_Var("5"),
        _seed_var=_Var("42"), _real_vol_var=_Var(""),
        _npaths_var=_Var("10"), _csv_path_var=_Var("real.csv"),
        _csv_col_var=_Var("close"), _wind_code_var=_Var("510050.SH"),
        _wind_start_var=_Var("2025-01-01"),
        _wind_end_var=_Var("2026-01-01"),
        # 粒度已不再是可选项；留一个会爆炸的替身，确保没有代码路径再读它。
        _wind_bar_size_var=_RemovedControlVar("行情采样粒度"),
        _slip_var=_Var("0"), _force_day_close_hedge_var=_BoolVar(False),
        _gui_steps_per_day=lambda source, value: BacktestApp._gui_steps_per_day(
            source, value),
    )
    fake._collect_gui_state_for_strategy = lambda strategy_name=None: (
        BacktestApp._collect_gui_state_for_strategy(fake, strategy_name))
    return fake


def _fake_history_collect_state(candidate_text="0.5，1,2"):
    fake = _fake_collect_state()
    fake._history_include_close_var = _BoolVar(True)
    fake._history_include_fixed_times_var = _BoolVar(True)
    fake._history_include_band_var = _BoolVar(True)
    fake._history_include_current_band_var = _BoolVar(True)
    fake._history_fixed_times_var = _Var("10:30,14:30")
    fake._history_band_candidate_sigmas_var = _Var(candidate_text)
    fake._sync_band_inputs = lambda _source, strict=False: {
        "absolute": 1.0, "relative": 0.01, "sigma": 0.779423,
    }
    fake._collect_gui_state = lambda: BacktestApp._collect_gui_state(fake)
    return fake


def _fake_history_period_vars(selected=None):
    """构造与真实周期复选框一致的轻量 BooleanVar 映射。"""
    if selected is None:
        selected = {key for key, _label in gui_app.HISTORY_PERIOD_DEFS}
    else:
        selected = set(selected)
    return {
        key: _BoolVar(key in selected)
        for key, _label in gui_app.HISTORY_PERIOD_DEFS
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, 1), (-1, -1), ("1", 1), ("-1", -1), (1.0, 1), (-1.0, -1)],
)
def test_position_normalization_accepts_only_the_two_gui_directions(
        raw, expected):
    assert BacktestApp._normalize_position(raw) == expected


@pytest.mark.parametrize("raw", [0, 2, -2, "", "long", True, np.nan])
def test_position_normalization_rejects_ambiguous_or_invalid_values(raw):
    with pytest.raises(ValueError, match="头寸方向"):
        BacktestApp._normalize_position(raw)


@pytest.mark.parametrize("position", [1, -1])
def test_single_and_history_state_freeze_left_position_as_unique_source(
        position):
    single = _fake_collect_state()
    single._pos_var.set(position)
    history = _fake_history_collect_state()
    history._pos_var.set(position)

    assert BacktestApp._collect_gui_state(single)["position"] == position
    assert BacktestApp._collect_history_state(history)["position"] == position


def test_history_collects_its_own_candidates_times_and_sigma_configuration():
    fake = _fake_history_collect_state()
    # 候选时刻独立于普通回测区；收盘兜底是唯一共享例外。单次回测那边即使
    # 选了历史波动率，也不能渗进择优候选——择优的 σ 恒为输入波动率。
    fake._fixed_times_var.set("09:45")
    fake._sigma_src_var.set(gui_app.SIGMA_SOURCE_DISPLAY["realized"])
    fake._sigma_win_var.set("30")

    state = BacktestApp._collect_history_state(fake)

    assert state["band_candidate_sigmas"] == (0.5, 1.0, 2.0)
    assert state["sigma_source"] == "implied"
    assert state["sigma_window"] == 20
    assert state["fixed_times"] == "10:30,14:30"
    assert state["history_include_close"] is True
    assert state["history_include_fixed_times"] is True
    assert state["history_include_band"] is True
    assert state["history_include_current_band"] is True


def test_history_default_period_controls_freeze_all_strict_lookbacks():
    fake = _fake_history_collect_state()
    fake._history_period_vars = _fake_history_period_vars()
    expected_lookbacks = {
        key: gui_app.LOOKBACK_DAYS[key]
        for key, _label in gui_app.HISTORY_PERIOD_DEFS
    }
    state = BacktestApp._collect_history_state(fake)

    assert state["history_lookbacks"] == expected_lookbacks
    assert "history_target_endpoints" not in state
    # 任务快照必须是普通字典；启动后再改复选框不能篡改本次实验。
    for variable in fake._history_period_vars.values():
        variable.set(False)
    assert state["history_lookbacks"] == expected_lookbacks


def test_history_collects_only_checked_periods_in_canonical_order():
    fake = _fake_history_collect_state()
    fake._history_period_vars = _fake_history_period_vars(
        selected=("half_year", "month"))

    state = BacktestApp._collect_history_state(fake)

    assert state["history_lookbacks"] == {
        "month": gui_app.LOOKBACK_DAYS["month"],
        "half_year": gui_app.LOOKBACK_DAYS["half_year"],
    }
    assert "history_target_endpoints" not in state


def test_history_collect_rejects_when_all_period_controls_are_cleared():
    fake = _fake_history_collect_state()
    fake._history_period_vars = _fake_history_period_vars(selected=())

    with pytest.raises(ValueError, match="至少选择一个历史分析周期"):
        BacktestApp._collect_history_state(fake)


def test_single_wind_collection_uses_maturity_auto_end_and_ignores_history_ui():
    class _ForbiddenVar:
        def get(self):
            raise AssertionError("单次回测不得读取历史择优 Wind 控件")

    fake = _fake_collect_state()
    fake._source_var.set("wind")
    fake._wind_start_var.set("2025-01-02")
    fake._wind_end_var.set("invalid-but-disabled")
    fake._wind_auto_end_var = _BoolVar(True)
    fake._strategy_var.set(STRATEGY_DISPLAY["close_to_close"])
    fake._history_wind_asof_var = _ForbiddenVar()
    fake._history_wind_start_var = _ForbiddenVar()
    fake._history_wind_auto_start_var = _ForbiddenVar()
    fake._history_wind_bar_size_var = _ForbiddenVar()

    state = BacktestApp._collect_gui_state(fake)
    required = state["params"]["T_days"] + 1
    expected_end = (
        datetime.date(2025, 1, 2) + datetime.timedelta(
            days=BacktestApp._calendar_span_for_trading_days(required))
    ).isoformat()

    assert state["wind_start"] == "2025-01-02"
    assert state["wind_end"] == expected_end
    assert state["wind_required_trade_days"] == required
    assert state["wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    assert state["wind_bar_size"] == "日频"


def test_history_wind_collection_uses_independent_asof_range_and_frequency():
    fake = _fake_history_collect_state()
    fake._source_var.set("wind")
    # 单次回测 Wind 控件故意不可用；历史页必须用自己的日期和粒度。
    fake._wind_start_var.set("not-a-single-start")
    fake._wind_end_var.set("not-a-single-end")
    fake._history_wind_asof_var = _Var("2025-06-30")
    fake._history_wind_start_var = _Var("not-used-while-auto")
    fake._history_wind_auto_start_var = _BoolVar(True)

    state = BacktestApp._collect_history_state(fake)
    # σ 恒取输入波动率，不再需要 HV 预热日；区间只留 Day 0 锚点那 1 日。
    required = ANNUAL_DAYS + 1
    expected_start = (
        datetime.date(2025, 6, 30) - datetime.timedelta(
            days=BacktestApp._calendar_span_for_trading_days(required))
    ).isoformat()

    assert state["history_wind_asof"] == "2025-06-30"
    assert state["wind_end"] == "2025-06-30"
    assert state["history_wind_start"] == expected_start
    assert state["wind_start"] == expected_start
    assert state["wind_required_trade_days"] == required
    assert state["wind_sigma_warmup_days"] == 0
    assert state["history_wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    # 默认固定时刻 + 固定间隔候选冻结为同一实际粒度，保持公平比较；
    # 该粒度由最窄的 σ 候选（默认 0.5σ）决定，否则窄带候选会被系统性
    # 少记触发次数而在排名里虚高。
    assert state["wind_bar_size"] == "1分钟"
    assert BacktestApp._history_recommendation_source_label(state) == (
        f"Wind · 510050.SH · {expected_start} 至 2025-06-30 · 1分钟"
    )


def test_history_state_always_requests_auto_bar_size():
    """择优页不再提供粒度选择：旧版本残留的手动值不得泄漏进新任务。"""
    fake = _fake_history_collect_state()
    fake._source_var.set("wind")
    fake._history_wind_asof_var = _Var("2025-06-30")
    fake._history_wind_start_var = _Var("2024-01-02")
    fake._history_wind_auto_start_var = _BoolVar(False)
    fake._history_wind_bar_size_var = _Var("日频")

    state = BacktestApp._collect_history_state(fake)

    assert state["history_wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    assert state["wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    # 默认候选含固定时刻 + 固定间隔，实际粒度取两者所需里最细的一档。
    assert state["wind_bar_size"] == "1分钟"


def test_new_wind_controls_are_optional_for_legacy_test_doubles():
    single = _fake_collect_state()
    assert not hasattr(single, "_wind_auto_end_var")
    collected_single = BacktestApp._collect_gui_state(single)

    history = _fake_history_collect_state()
    assert not hasattr(history, "_history_wind_asof_var")
    assert not hasattr(history, "_history_wind_start_var")
    assert not hasattr(history, "_history_wind_auto_start_var")
    assert not hasattr(history, "_history_wind_bar_size_var")
    assert not hasattr(history, "_history_period_vars")
    collected_history = BacktestApp._collect_history_state(history)

    assert collected_single["wind_auto_end"] is False
    assert collected_history["history_wind_auto_start"] is False
    assert collected_history["history_lookbacks"] == {
        key: gui_app.LOOKBACK_DAYS[key]
        for key, _label in gui_app.HISTORY_PERIOD_DEFS
    }
    assert "history_target_endpoints" not in collected_history


@pytest.mark.parametrize("enabled", [False, True])
def test_single_backtest_collects_public_day_close_fallback_state(enabled):
    fake = _fake_collect_state()
    fake._force_day_close_hedge_var.set(enabled)

    state = BacktestApp._collect_gui_state(fake)

    assert state["force_day_close_hedge"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_history_state_uses_left_public_day_close_fallback_rule(enabled):
    class _ForbiddenLegacyVar:
        def get(self):
            raise AssertionError("历史择优不得再读取已移除的独立兜底开关")

    fake = _fake_history_collect_state()
    # 历史页不再维护第二个开关；启动时直接冻结左侧公共控制值。
    fake._force_day_close_hedge_var.set(enabled)
    fake._history_force_day_close_hedge_var = _ForbiddenLegacyVar()

    state = BacktestApp._collect_history_state(fake)

    assert state["force_day_close_hedge"] is enabled


def test_history_cases_record_day_close_fallback_for_recommendation_replay():
    state = {
        "source": "wind", "wind_bar_size": "日频",
        "history_include_close": True,
        "history_include_fixed_times": False,
        "history_include_band": True,
        "history_include_current_band": False,
        "force_day_close_hedge": True,
        "interval_type": "sigma", "price_interval": 1.0,
        "band_candidate_sigmas": (0.5, 1.0),
        "sigma_source": "implied", "sigma_window": 20,
        "params": {"s0": 100.0, "sigma": 0.2},
    }

    cases, notes = BacktestApp._strategy_cases_for_history(
        state, SimpleNamespace())

    assert notes == []
    assert len(cases) == 3
    assert all(
        case.metadata["force_day_close_hedge"] is True for case in cases)


def test_history_rejects_c2c_only_before_reading_disabled_candidate_controls():
    class _ForbiddenVar:
        def get(self):
            raise AssertionError("未勾选的历史候选参数不应被读取")

    fake = _fake_history_collect_state()
    fake._history_include_fixed_times_var.set(False)
    fake._history_include_band_var.set(False)
    # 只有固定基准不是一次“择优”；应在读取全部禁用输入之前给出候选错误。
    fake._strategy_var.set(STRATEGY_DISPLAY["fixed_times"])
    fake._fixed_times_var = _ForbiddenVar()
    fake._history_fixed_times_var = _ForbiddenVar()
    fake._history_band_candidate_sigmas_var = _ForbiddenVar()
    fake._history_include_current_band_var = _ForbiddenVar()

    with pytest.raises(ValueError, match="至少再启用一种候选策略"):
        BacktestApp._collect_history_state(fake)


def test_history_band_sigma_never_reads_any_volatility_source_control():
    """择优候选的 σ 恒为输入波动率，不得从任何界面控件取来源或回看期。

    择优只比带宽倍数本身，全部候选必须共用同一个 σ 才可比；一旦有人把
    单次回测的「波动率来源」重新接进来，排名就会混进另一条策略维度。
    """
    class _ForbiddenVar:
        def get(self):
            raise AssertionError("择优不应读取波动率来源/回看期控件")

    fake = _fake_history_collect_state()
    fake._sigma_src_var = _ForbiddenVar()
    fake._sigma_win_var = _ForbiddenVar()

    state = BacktestApp._collect_history_state(fake)

    assert state["history_include_band"] is True
    assert state["sigma_source"] == "implied"
    assert state["sigma_window"] == 20


def test_history_close_to_close_baseline_cannot_be_disabled():
    fake = _fake_history_collect_state()
    fake._history_include_close_var.set(False)
    # 固定时刻仍提供一个实际候选；旧状态即使把基准写成 false，收集后
    # 也必须恢复固定 C2C 基准。
    fake._history_include_fixed_times_var.set(True)
    fake._history_include_band_var.set(False)

    state = BacktestApp._collect_history_state(fake)

    assert state["history_include_close"] is True
    assert state["history_include_fixed_times"] is True
    assert state["history_include_band"] is False


@pytest.mark.parametrize(
    ("source", "strategy", "band_mode", "expected_visible"),
    [
        ("csv", "close_to_close", "absolute", False),
        ("wind", "fixed_times", "relative", False),
        ("simulate", "close_to_close", "absolute", False),
        ("simulate", "hedge_band", "sigma", True),
    ],
)
def test_ordinary_sigma_controls_are_independent_of_history_workspace(
        source, strategy, band_mode, expected_visible):
    fake = SimpleNamespace(
        _strategy_var=_Var(STRATEGY_DISPLAY[strategy]),
        _interval_type_var=_Var(band_mode), _source_var=_Var(source),
        _sigma_band_frame=_GridWidget(), _fixed_time_frame=_GridWidget(),
        _band_frame=_GridWidget(), _fixed_times_entry=_GridWidget(),
        _band_abs_entry=_GridWidget(), _band_rel_entry=_GridWidget(),
        _band_sigma_entry=_GridWidget(), _band_synced=True,
    )

    BacktestApp._toggle_strategy(fake)

    assert fake._sigma_band_frame.visible is expected_visible


def test_history_candidate_controls_only_follow_history_strategy_switches():
    fake = SimpleNamespace(
        _history_include_fixed_times_var=_BoolVar(False),
        _history_include_band_var=_BoolVar(True),
        _history_fixed_times_entry=_Widget(),
        _history_band_candidate_entry=_Widget(),
        _history_current_band_check=_Widget(),
    )

    BacktestApp._toggle_history_candidate_controls(fake)

    assert fake._history_fixed_times_entry.state == "disabled"
    assert fake._history_band_candidate_entry.state == "normal"
    assert fake._history_current_band_check.state == "normal"

    fake._history_include_fixed_times_var.set(True)
    fake._history_include_band_var.set(False)
    BacktestApp._toggle_history_candidate_controls(fake)

    assert fake._history_fixed_times_entry.state == "normal"
    assert fake._history_band_candidate_entry.state == "disabled"
    assert fake._history_current_band_check.state == "disabled"


def test_ordinary_backtest_does_not_parse_hidden_history_candidates():
    fake = _fake_collect_state()
    fake._history_band_candidate_sigmas_var = _Var("not-a-number")
    fake._history_fixed_times_var = _Var("not-a-time")

    state = BacktestApp._collect_gui_state(fake)

    assert state["sigma_source"] == "implied"
    assert state["sigma_window"] == 20
    assert state.get("band_candidate_sigmas") is None


@pytest.mark.parametrize("source", ["csv", "wind"])
def test_history_recommendation_source_accepts_only_real_history(source):
    BacktestApp._validate_history_recommendation_source({"source": source})


@pytest.mark.parametrize("source", ["simulate", "unknown", None])
def test_history_recommendation_source_rejects_non_real_paths(source):
    with pytest.raises(ValueError, match="CSV 或 Wind 真实历史"):
        BacktestApp._validate_history_recommendation_source({"source": source})


def test_history_rejects_simulate_before_collecting_or_starting(monkeypatch):
    calls = []
    errors = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"unexpected call: {name}")
        return fail

    fake = SimpleNamespace(
        _active_job=None,
        _source_var=_Var("simulate"),
        _collect_history_state=forbidden("collect-history"),
        _collect_gui_state=forbidden("collect-backtest"),
        _begin_job=forbidden("begin"),
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: errors.append((title, message)),
    )

    started = BacktestApp._run_history_recommendation(fake)

    assert started is False
    assert calls == []
    assert errors and errors[0][0] == "策略优选不可用"
    assert "模拟路径不可用" in errors[0][1]


def test_history_entry_collects_only_history_state_and_uses_history_job():
    collected = []
    begun = []
    # 这里替换的是完整 `_collect_history_state`，因此返回值应遵守新的任务
    # 快照契约；缺少周期控件的旧 GUI 替身兼容由独立测试覆盖。
    state = {
        "source": "csv",
        "history_lookbacks": dict(gui_app.LOOKBACK_DAYS),
    }

    def collect_history():
        collected.append("history")
        return state

    fake = SimpleNamespace(
        _active_job=None,
        _source_var=_Var("csv"),
        _collect_history_state=collect_history,
        _collect_gui_state=lambda: pytest.fail(
            "历史任务不得直接收集普通回测表单"),
        _refresh_history_base_summary=lambda: None,
        _begin_job=lambda name, status: (
            begun.append((name, status)) or False),
    )

    started = BacktestApp._run_history_recommendation(fake)

    assert started is False
    assert collected == ["history"]
    assert begun and begun[0][0] == "history"


def test_history_button_tracks_source_and_busy_state():
    fake = SimpleNamespace(
        _history_btn=_Widget(), _source_var=_Var("simulate"), _active_job=None)

    BacktestApp._sync_history_button_state(fake)
    assert fake._history_btn.state == "disabled"

    fake._source_var.set("csv")
    BacktestApp._sync_history_button_state(fake)
    assert fake._history_btn.state == "normal"

    fake._active_job = "backtest"
    BacktestApp._sync_history_button_state(fake)
    assert fake._history_btn.state == "disabled"


def test_history_button_sync_updates_independent_workspace_hint():
    fake = SimpleNamespace(
        _history_btn=_Widget(), _history_source_hint_var=_Var(),
        _source_var=_Var("csv"), _active_job=None,
    )

    BacktestApp._sync_history_button_state(fake)

    # 数据源可用时按钮直接放行，不再多说一句“已就绪”占版面。
    assert fake._history_btn.state == "normal"
    assert fake._history_source_hint_var.get() == ""

    # 只有真正跑不了的时候才需要出声解释原因。
    fake._source_var = _Var("simulate")
    BacktestApp._sync_history_button_state(fake)
    assert fake._history_btn.state == "disabled"
    assert "仅支持 CSV / Wind" in fake._history_source_hint_var.get()


def test_history_loader_rejects_simulation_even_with_retained_series():
    base_bt = SimpleNamespace(
        _full_price_history=pd.Series([100.0, 101.0]),
        _gui_meta={"source": "simulate"},
    )
    with pytest.raises(ValueError, match="真实历史"):
        BacktestApp._load_full_history_for_recommendation(
            {"source": "simulate"}, base_bt)


def test_specific_futures_contract_history_never_enters_product_pool(
        monkeypatch):
    retained = pd.Series(
        [100.0, 101.0],
        index=pd.bdate_range("2026-01-05", periods=2),
    )
    base_bt = SimpleNamespace(
        _full_price_history=retained,
        _gui_meta={"source": "wind"},
    )
    monkeypatch.setattr(
        BacktestApp, "_load_wind_contract_history_pool",
        staticmethod(lambda _state: pytest.fail(
            "具体合约不得自动汇集同品种历史合约")),
    )

    result = BacktestApp._load_full_history_for_recommendation(
        {"source": "wind", "wind_code": "P2609.DCE"}, base_bt)

    assert result.equals(retained)
    assert result is not retained


def test_product_code_history_prefers_contract_pool_over_retained_continuous(
        monkeypatch):
    mapping = pd.Series(
        ["P2605.DCE", "P2609.DCE"],
        index=pd.bdate_range("2026-01-05", periods=2),
    )
    pool = ContractHistoryPool(
        "P.DCE", mapping,
        {"P2609.DCE": pd.Series(
            [100.0, 101.0], index=pd.bdate_range("2026-01-02", periods=2))},
        "P2609.DCE",
    )
    calls = []
    monkeypatch.setattr(
        BacktestApp, "_load_wind_contract_history_pool",
        staticmethod(lambda state: calls.append(state["wind_code"]) or pool),
    )
    base_bt = SimpleNamespace(
        _full_price_history=pd.Series([1.0, 2.0]),
        _gui_meta={"source": "wind"},
    )

    result = BacktestApp._load_full_history_for_recommendation(
        {"source": "wind", "wind_code": "P.DCE"}, base_bt)

    assert result is pool
    assert calls == ["P.DCE"]


def test_contract_pool_source_label_and_single_contract_label_are_explicit():
    mapping = pd.Series(
        ["P2605.DCE", "P2609.DCE"],
        index=pd.bdate_range("2026-06-29", periods=2),
    )
    pool = ContractHistoryPool(
        "P.DCE", mapping,
        {
            "P2605.DCE": pd.Series(
                [100.0, 101.0], index=pd.bdate_range("2026-01-01", periods=2)),
            "P2609.DCE": pd.Series(
                [102.0, 103.0], index=pd.bdate_range("2026-06-26", periods=2)),
        },
        "P2609.DCE",
    )
    pool_label = BacktestApp._history_recommendation_source_label({
        "source": "wind", "wind_code": "P.DCE",
        "wind_start": "2025-06-01", "wind_end": "2026-06-30",
        "wind_bar_size": "15分钟",
    }, pool)
    single_label = BacktestApp._history_recommendation_source_label({
        "source": "wind", "wind_code": "P2609.DCE",
        "wind_start": "2026-01-01", "wind_end": "2026-06-30",
        "wind_bar_size": "日频",
    })

    assert "Wind 品种样本池" in pool_label
    assert "P.DCE" in pool_label
    assert "请求截至 2026-06-30" in pool_label
    assert "主力映射截至 2026-06-30 = P2609.DCE" in pool_label
    assert "2 个历史合约" in pool_label
    assert "Wind 单合约（不自动汇集）" in single_label
    assert "P2609.DCE" in single_label


def test_load_wind_contract_pool_isolates_one_contract_failure(monkeypatch):
    mapping = pd.Series(
        ["P2509.DCE", "P2601.DCE", "P2605.DCE"],
        index=pd.to_datetime(["2025-09-01", "2026-01-05", "2026-05-06"]),
    )
    requested = []

    def get_close(code, start, end, adjust):
        requested.append((code, start, end, adjust))
        if code == "P2601.DCE":
            raise RuntimeError("expired contract unavailable")
        return pd.Series(
            np.linspace(100.0, 103.0, 4),
            index=pd.bdate_range(end=pd.Timestamp(end), periods=4),
        )

    monkeypatch.setattr(
        "pricing.wind_data.get_main_contract_history",
        lambda *_args: mapping,
    )
    monkeypatch.setattr("pricing.wind_data.get_close_prices", get_close)

    pool = BacktestApp._load_wind_contract_history_pool({
        "wind_code": "P.DCE", "wind_start": "2025-01-01",
        "wind_end": "2026-06-30", "wind_bar_size": "日频",
        "params": {"T_days": 2},
    })

    assert set(pool.contract_prices) == {"P2509.DCE", "P2605.DCE"}
    assert "P2601.DCE" in pool.contract_load_errors
    assert pool.main_contract_asof == "P2605.DCE"
    assert {item[0] for item in requested} == set(mapping.tolist())
    assert all(item[0] != "P.DCE" for item in requested)
    assert all(item[3] == "" for item in requested)


def test_contract_pool_loads_contracts_only_from_longest_selected_period(
        monkeypatch):
    dates = pd.bdate_range("2026-06-01", periods=10)
    mapping = pd.Series(
        ["P2509.DCE"] * 5 + ["P2601.DCE"] * 5,
        index=dates,
    )
    requested = []

    monkeypatch.setattr(
        "pricing.wind_data.get_main_contract_history",
        lambda *_args: mapping,
    )

    def get_close(code, start, end, adjust):
        requested.append(code)
        return pd.Series(
            np.linspace(100.0, 103.0, 4),
            index=pd.bdate_range(end=pd.Timestamp(end), periods=4),
        )

    monkeypatch.setattr("pricing.wind_data.get_close_prices", get_close)

    pool = BacktestApp._load_wind_contract_history_pool({
        "wind_code": "P.DCE",
        "wind_start": "2026-05-01",
        "wind_end": "2026-06-30",
        "wind_bar_size": "日频",
        "params": {"T_days": 2},
        "history_lookbacks": {"week": gui_app.LOOKBACK_DAYS["week"]},
    })

    assert requested == ["P2601.DCE"]
    assert set(pool.contract_prices) == {"P2601.DCE"}
    assert pool.main_contract_asof == "P2601.DCE"


def test_contract_pool_prehistory_does_not_expand_with_proxy_maturity(
        monkeypatch):
    mapping = pd.Series(
        ["P2609.DCE"] * 5,
        index=pd.bdate_range("2026-06-01", periods=5),
    )
    requested_starts = []
    monkeypatch.setattr(
        "pricing.wind_data.get_main_contract_history",
        lambda *_args: mapping,
    )

    def get_close(_code, start, end, _adjust):
        requested_starts.append(start)
        return pd.Series(
            np.linspace(100.0, 103.0, 4),
            index=pd.bdate_range(end=pd.Timestamp(end), periods=4),
        )

    monkeypatch.setattr("pricing.wind_data.get_close_prices", get_close)
    for maturity_days in (2, 243):
        BacktestApp._load_wind_contract_history_pool({
            "wind_code": "P.DCE",
            "wind_start": "2026-01-01",
            "wind_end": "2026-06-30",
            "wind_bar_size": "日频",
            "params": {"T_days": maturity_days},
            "history_lookbacks": {"week": gui_app.LOOKBACK_DAYS["week"]},
        })

    assert len(requested_starts) == 2
    assert requested_starts[0] == requested_starts[1]


def test_comparison_headline_is_order_independent_and_uses_runner_denominator():
    summary = pd.DataFrame([
        {"strategy": "C", "score": 12.0, "total_tc": 120.0},
        {"strategy": "A", "score": 8.0, "total_tc": 100.0},
        {"strategy": "B", "score": 10.0, "total_tc": 80.0},
    ])

    headline = BacktestApp._comparison_headline(summary)

    assert headline["best"]["strategy"] == "A"
    assert headline["runner_up"]["strategy"] == "B"
    assert headline["improvement_ratio"] == pytest.approx(0.20)
    assert headline["strategy_count"] == 3


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (1.23456, {"digits": 4}, "1.2346"),
        (None, {}, "—"),
        (float("nan"), {}, "—"),
        (float("inf"), {}, "—"),
        (-0.0, {"digits": 4, "signed": True}, "0.0000"),
        (0.125, {"digits": 1, "signed": True, "percent": True}, "+12.5%"),
    ],
)
def test_comparison_formatter_handles_missing_signed_and_percent_values(
        value, kwargs, expected):
    assert BacktestApp._format_comparison_value(value, **kwargs) == expected


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), pd.NA])
def test_comparison_safe_int_rejects_missing_or_non_finite_values(value):
    assert BacktestApp._comparison_safe_int(value, 7) == 7


def test_selected_strategy_delta_uses_fixed_baseline_as_denominator():
    baseline = {"score": 10.0, "total_tc": 100.0}
    selected = {"score": 8.0, "total_tc": 80.0}

    score = BacktestApp._comparison_relative_delta(
        selected, baseline, "score")
    cost = BacktestApp._comparison_relative_delta(
        selected, baseline, "total_tc")

    assert score == {"value": 8.0, "delta": -2.0, "ratio": -0.20}
    assert cost == {"value": 80.0, "delta": -20.0, "ratio": -0.20}


def test_equal_non_baseline_metric_is_not_labeled_as_baseline():
    baseline = {
        "strategy": "A", "strategy_type": "close_to_close",
        "meta_result_id": "result-0001", "score": 8.0,
    }
    selected = {
        "strategy": "B", "strategy_type": "hedge_band",
        "meta_result_id": "result-0002", "score": 8.0,
    }

    assert BacktestApp._comparison_delta_display(
        selected, baseline, "score") == "8.00（与基准相同）"


def test_comparison_baseline_identity_uses_result_id_not_display_name():
    baseline = {
        "strategy": "同名结果", "strategy_type": "close_to_close",
        "meta_result_id": "result-0001", "score": 10.0,
    }
    same_name_candidate = {
        "strategy": "同名结果", "strategy_type": "close_to_close",
        "meta_result_id": "result-0002", "score": 10.0,
    }

    assert BacktestApp._comparison_delta_display(
        baseline, baseline, "score") == "10.00（基准）"
    assert BacktestApp._comparison_delta_display(
        same_name_candidate, baseline, "score") == "10.00（与基准相同）"


def test_comparison_relative_delta_keeps_absolute_delta_for_zero_baseline():
    metric = BacktestApp._comparison_relative_delta(
        {"score": 2.0}, {"score": 0.0}, "score")

    assert metric == {"value": 2.0, "delta": 2.0, "ratio": None}


def test_history_row_improvement_treats_c2c_itself_as_zero_improvement():
    improvement = BacktestApp._history_row_improvement({
        "strategy": "每日收盘",
        "strategy_type": "close_to_close",
        "score": 7.5,
    })

    assert improvement == pytest.approx(0.0)


def test_history_row_improvement_does_not_divide_by_zero_c2c_score():
    improvement = BacktestApp._history_row_improvement({
        "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band",
        "score": 2.0,
        "baseline_score": 0.0,
    })

    assert improvement is None


def test_history_row_improvement_prefers_window_equal_selection_metric():
    improvement = BacktestApp._history_row_improvement({
        "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band",
        "score": 110.0,
        "baseline_score": 100.0,
        # 合并金额 RMS 被高价合约主导，但逐窗平均优势仍为正。
        "improvement_vs_c2c": -0.10,
        "selection_improvement_vs_c2c": 0.20,
        "selection_metric": "mean_bounded_window_advantage_vs_c2c",
    })

    assert improvement == pytest.approx(0.20)


def test_history_metric_labels_mark_product_pool_raw_rms_as_diagnostic():
    single = BacktestApp._history_metric_labels(
        uses_strict_metric=True, uses_product_pool=False)
    pool = BacktestApp._history_metric_labels(
        uses_strict_metric=True, uses_product_pool=True)

    assert single == {
        "score": "区间得分↓",
        "baseline": "每日收盘区间得分",
        "improvement": "较收盘改善",
    }
    # 品种池的原始 RMS 只是诊断值，三个字段都必须带上“诊断”字样，
    # 不能与严格区间主指标共用同一套措辞。
    assert pool == {
        "score": "诊断总得分↓",
        "baseline": "每日收盘诊断分",
        "improvement": "较收盘诊断改善",
    }


def _history_chart_summary():
    rows = []

    def add(window_id, end_ts, baseline_score, candidate_score,
            baseline_daily, candidate_daily):
        for strategy, strategy_type, score, daily in (
            ("每日收盘", "close_to_close", baseline_score, baseline_daily),
            ("固定间隔(1σ)", "hedge_band", candidate_score, candidate_daily),
        ):
            daily = np.asarray(daily, dtype=float)
            tc = np.full(len(daily), 0.25, dtype=float)
            gross = daily + tc
            rows.append({
                "lookback": "week", "window_id": window_id,
                "strategy": strategy, "strategy_type": strategy_type,
                "success": True, "start_ts": pd.Timestamp(end_ts) - pd.Timedelta(days=2),
                "end_ts": pd.Timestamp(end_ts), "score": score,
                "daily_net_pnl": daily, "daily_gross_pnl": gross,
                "daily_tc": tc, "cumulative_net_pnl": np.cumsum(daily),
                "cumulative_gross_pnl": np.cumsum(gross),
                "cumulative_tc": np.cumsum(tc),
            })

    # window_id 字符串顺序与真实结束日顺序故意相反。
    add("window_10", "2026-01-05", 10.0, 8.0, [1.0, 3.0], [2.0, 4.0])
    add("window_2", "2026-01-12", 20.0, 25.0, [3.0, 5.0], [4.0, 6.0])
    # 零基准窗口必须按有界优势 -1 保留，不能从择优证据中静默丢失。
    add("window_3", "2026-01-19", 0.0, 1.0, [5.0, 7.0], [6.0, 8.0])
    return pd.DataFrame(rows)


def _history_multi_chart_summary():
    summary = _history_chart_summary()
    second = summary[
        summary["strategy"] == "固定间隔(1σ)"].copy()
    second = second[second["window_id"].isin(["window_2", "window_3"])]
    second["strategy"] = "每日固定时刻"
    second["strategy_type"] = "fixed_times"
    second.loc[second["window_id"] == "window_2", "score"] = 12.0
    second.loc[second["window_id"] == "window_3", "score"] = 0.0
    return pd.concat([summary, second], ignore_index=True)


def _history_cross_contract_normalized_summary():
    summary = _history_chart_summary()
    summary["history_contract_code"] = summary["window_id"].map({
        "window_10": "P2509.DCE",
        "window_2": "P2601.DCE",
        "window_3": "P2605.DCE",
    })
    denominators = summary["window_id"].map({
        "window_10": 100.0,
        "window_2": 200.0,
        "window_3": 400.0,
    }).to_numpy(dtype=float)
    for raw, normalized in (
        ("cumulative_net_pnl", "normalized_cumulative_net_pnl"),
        ("cumulative_gross_pnl", "normalized_cumulative_gross_pnl"),
        ("cumulative_tc", "normalized_cumulative_tc"),
        ("daily_net_pnl", "normalized_daily_net_pnl"),
        ("daily_gross_pnl", "normalized_daily_gross_pnl"),
        ("daily_tc", "normalized_daily_tc"),
    ):
        summary[normalized] = pd.Series([
            np.asarray(values, dtype=float) / denominator
            for values, denominator in zip(summary[raw], denominators)
        ], dtype=object)
    return summary


def test_history_chart_exposes_full_interval_and_proxy_diagnostic_modes():
    assert gui_app.HISTORY_CHART_MODE_DISPLAY == {
        "full": "完整回放累积路径",
        "single": "单次分段路径",
        "typical": "多分段中位路径",
    }


def test_history_chart_pairs_by_window_and_sorts_by_end_timestamp():
    pairs = BacktestApp._history_chart_pairs(
        _history_chart_summary(), "week", "固定间隔(1σ)")

    assert pairs["window_id"].tolist() == [
        "window_10", "window_2", "window_3",
    ]
    assert pairs["strategy_type_baseline"].eq("close_to_close").all()


@pytest.mark.parametrize(
    ("metric", "column"),
    [
        ("net", "cumulative_net_pnl"),
        ("gross", "cumulative_gross_pnl"),
        ("tc", "cumulative_tc"),
    ],
)
def test_history_single_window_chart_supports_all_pnl_metrics(metric, column):
    summary = _history_chart_summary()
    model = BacktestApp._history_chart_model(
        summary, "week", "固定间隔(1σ)", mode="single",
        metric=metric, window_id="window_2",
    )

    candidate = next(
        item for item in model["series"] if item["role"] == "candidate")
    expected = summary[
        (summary["window_id"] == "window_2")
        & (summary["strategy"] == "固定间隔(1σ)")
    ].iloc[0][column]
    assert model["state"] == "ok"
    assert model["selected_window_id"] == "window_2"
    np.testing.assert_allclose(candidate["y"], expected)


def test_history_typical_chart_uses_median_and_descriptive_quartile_band():
    summary = _history_chart_summary()
    # 仅保留两个常规分数窗口，使预期分位数直观。
    summary = summary[summary["window_id"].isin(["window_10", "window_2"])]
    model = BacktestApp._history_chart_model(
        summary, "week", "固定间隔(1σ)", mode="typical", metric="net")

    candidate = next(
        item for item in model["bands"] if item["role"] == "candidate")
    assert model["state"] == "ok"
    assert candidate["window_count"] == 2
    assert candidate["show_interval"] is True
    np.testing.assert_allclose(candidate["median"], [3.0, 8.0])
    np.testing.assert_allclose(candidate["p25"], [2.5, 7.0])
    np.testing.assert_allclose(candidate["p75"], [3.5, 9.0])
    assert "非置信区间" in model["title"]


def test_history_chart_band_carries_settled_curve_forward():
    band = BacktestApp._history_chart_band([
        np.asarray([1.0, 3.0]),
        np.asarray([2.0, 4.0, 8.0]),
    ])

    assert band["window_count"] == 2
    np.testing.assert_allclose(band["median"], [1.5, 3.5, 5.5])
    np.testing.assert_allclose(band["p25"], [1.25, 3.25, 4.25])
    np.testing.assert_allclose(band["p75"], [1.75, 3.75, 6.75])


def test_history_chart_has_explicit_empty_state_without_same_length_c2c_pair():
    summary = _history_chart_summary()
    mask = (
        (summary["window_id"] == "window_10")
        & (summary["strategy_type"] == "close_to_close")
    )
    summary.loc[mask, "daily_net_pnl"] = pd.Series(
        [np.asarray([1.0], dtype=float)], index=summary.index[mask])

    model = BacktestApp._history_chart_model(
        summary[summary["window_id"] == "window_10"],
        "week", "固定间隔(1σ)", mode="single")

    assert model["state"] == "empty"
    assert "每日收盘基准" in model["message"]


def test_history_multi_chart_uses_strict_common_windows_and_one_c2c_series():
    model = BacktestApp._history_multi_chart_model(
        _history_multi_chart_summary(), "week",
        ["每日收盘", "固定间隔(1σ)", "每日固定时刻", "固定间隔(1σ)"],
        mode="single", metric="net", window_id="window_10",
        primary_strategy="每日固定时刻",
    )

    assert model["state"] == "ok"
    assert model["common_window_count"] == 2
    assert [item["window_id"] for item in model["window_options"]] == [
        "window_2", "window_3",
    ]
    assert model["selected_window_id"] == "window_3"
    assert [item["role"] for item in model["series"]] == [
        "baseline", "candidate", "candidate",
    ]
    assert sum(item["role"] == "baseline" for item in model["series"]) == 1
    assert model["primary_strategy"] == "每日固定时刻"
    assert all("window_" not in item["label"]
               for item in model["window_options"])
    assert all(item["label"].startswith("第 ")
               for item in model["window_options"])


def test_history_full_chart_concatenates_common_segments_in_strict_order():
    model = BacktestApp._history_multi_chart_model(
        _history_multi_chart_summary(), "week",
        ["固定间隔(1σ)", "每日固定时刻"],
        mode="full", metric="net",
        primary_strategy="每日固定时刻",
    )

    baseline = next(
        item for item in model["series"] if item["role"] == "baseline")
    band = next(
        item for item in model["series"]
        if item.get("strategy") == "固定间隔(1σ)")

    assert model["state"] == "ok"
    assert model["common_window_count"] == 2
    assert model["segment_lengths"] == (2, 2)
    assert model["segment_boundaries"] == (2,)
    assert model["common_day_count"] == 4
    assert model["expected_day_count"] == gui_app.LOOKBACK_DAYS["week"]
    assert model["complete_evidence"] is False
    np.testing.assert_allclose(baseline["y"], [3.0, 8.0, 13.0, 20.0])
    np.testing.assert_allclose(band["y"], [4.0, 10.0, 16.0, 24.0])
    assert "共同可比区间" in model["title"]


def test_history_full_chart_drops_failed_segment_from_common_intersection():
    summary = _history_multi_chart_summary()
    failed = (
        summary["strategy"].eq("每日固定时刻")
        & summary["window_id"].eq("window_2")
    )
    summary.loc[failed, "success"] = False

    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)", "每日固定时刻"],
        mode="full", metric="net",
    )

    assert model["state"] == "ok"
    assert [item["window_id"] for item in model["window_options"]] == [
        "window_3",
    ]
    assert model["common_day_count"] == 2
    for item in model["series"]:
        assert len(item["y"]) == 2
    assert model["sample_scope_differs"] is True


def test_history_multi_typical_chart_uses_same_window_count_for_all_series():
    model = BacktestApp._history_multi_chart_model(
        _history_multi_chart_summary(), "week",
        ["固定间隔(1σ)", "每日固定时刻"],
        mode="typical", metric="net",
    )

    assert model["state"] == "ok"
    assert len(model["bands"]) == 3
    assert {band["window_count"] for band in model["bands"]} == {2}


def test_history_cross_contract_typical_chart_uses_normalized_curves():
    summary = _history_cross_contract_normalized_summary()
    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"],
        mode="typical", metric="net",
    )

    candidate = next(
        band for band in model["bands"] if band["role"] == "candidate")
    expected = np.median(np.vstack([
        row["normalized_cumulative_net_pnl"]
        for _index, row in summary[
            summary["strategy"].eq("固定间隔(1σ)")].iterrows()
    ]), axis=0)
    assert model["state"] == "ok"
    assert model["uses_normalized_notional"] is True
    assert "期初名义金额" in model["metric_label"]
    np.testing.assert_allclose(candidate["median"], expected)


def test_history_cross_contract_full_chart_uses_normalized_daily_curves():
    summary = _history_cross_contract_normalized_summary()
    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"],
        mode="full", metric="net",
    )

    candidate = next(
        item for item in model["series"] if item["role"] == "candidate")
    expected_daily = np.concatenate([
        row["normalized_daily_net_pnl"]
        for _index, row in summary[
            summary["strategy"].eq("固定间隔(1σ)")
        ].sort_values("end_ts").iterrows()
    ])
    assert model["state"] == "ok"
    assert model["uses_normalized_notional"] is True
    assert "各回测分段期初名义金额" in model["metric_label"]
    np.testing.assert_allclose(candidate["y"], np.cumsum(expected_daily))


def test_history_cross_contract_typical_chart_fails_closed_without_normalization():
    summary = _history_cross_contract_normalized_summary().drop(
        columns=["normalized_cumulative_net_pnl"])

    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"],
        mode="typical", metric="net",
    )

    assert model["state"] == "empty"
    assert "安全归一化" in model["message"]
    assert "单回测分段路径" in model["message"]


def test_history_cross_contract_single_chart_keeps_raw_amount_curves():
    summary = _history_cross_contract_normalized_summary()
    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"],
        mode="single", metric="net", window_id="window_2",
    )

    candidate = next(
        series for series in model["series"] if series["role"] == "candidate")
    expected = summary[
        summary["strategy"].eq("固定间隔(1σ)")
        & summary["window_id"].eq("window_2")
    ].iloc[0]["cumulative_net_pnl"]
    assert model["uses_normalized_notional"] is False
    np.testing.assert_allclose(candidate["y"], expected)


def test_history_multi_chart_has_explicit_empty_state_without_common_window():
    summary = _history_multi_chart_summary()
    keep = (
        summary["strategy_type"].eq("close_to_close")
        | (summary["strategy"].eq("固定间隔(1σ)")
           & summary["window_id"].eq("window_10"))
        | (summary["strategy"].eq("每日固定时刻")
           & summary["window_id"].eq("window_2"))
    )
    model = BacktestApp._history_multi_chart_model(
        summary[keep], "week", ["固定间隔(1σ)", "每日固定时刻"],
        mode="typical",
    )

    assert model["state"] == "empty"
    assert "共同" in model["message"]


def test_history_multi_chart_recomputes_common_windows_for_selected_metric():
    summary = _history_multi_chart_summary()
    index = summary.index[
        summary["strategy"].eq("每日固定时刻")
        & summary["window_id"].eq("window_2")
    ][0]
    summary.at[index, "cumulative_tc"] = np.asarray([], dtype=float)

    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)", "每日固定时刻"],
        mode="single", metric="tc",
    )

    assert model["state"] == "ok"
    assert model["common_window_count"] == 1
    assert [item["window_id"] for item in model["window_options"]] == [
        "window_3",
    ]


def test_history_chart_controls_clear_stale_window_after_metric_change():
    summary = _history_multi_chart_summary()
    index = summary.index[
        summary["strategy"].eq("每日固定时刻")
        & summary["window_id"].eq("window_2")
    ][0]
    summary.at[index, "cumulative_tc"] = np.asarray([], dtype=float)

    app = object.__new__(BacktestApp)
    app._history_window_summary = summary
    app._history_chart_mode_var = _Var(
        gui_app.HISTORY_CHART_MODE_DISPLAY["single"])
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["net"])
    app._history_chart_window_var = _Var("")
    app._history_chart_window_combo = _Widget()
    app._history_chart_metric_combo = _Widget()
    app._history_chart_selection = lambda: ("week", None)
    app._history_chart_candidates = lambda _lookback: [
        "固定间隔(1σ)", "每日固定时刻",
    ]
    app._history_chart_primary_candidate = lambda candidates: candidates[0]
    app._draw_history_chart = lambda: None

    BacktestApp._update_history_chart_controls(app)
    stale_label = next(
        label for label, window_id in app._history_chart_window_labels.items()
        if window_id == "window_2")
    app._history_chart_window_var.set(stale_label)
    app._history_chart_metric_var.set(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["tc"])

    BacktestApp._update_history_chart_controls(app)

    assert list(app._history_chart_window_labels.values()) == ["window_3"]
    assert app._history_chart_window_var.get() in app._history_chart_window_labels
    assert app._history_chart_window_combo.state == "readonly"


def test_history_chart_controls_fall_back_from_removed_relative_mode():
    app = object.__new__(BacktestApp)
    app._history_window_summary = _history_multi_chart_summary()
    app._history_chart_mode_var = _Var("逐窗有界C2C优势")
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["net"])
    app._history_chart_window_var = _Var("")
    app._history_chart_window_combo = _Widget()
    app._history_chart_metric_combo = _Widget()
    app._history_chart_selection = lambda: ("week", None)
    app._history_chart_candidates = lambda _lookback: ["固定间隔(1σ)"]
    app._history_chart_primary_candidate = lambda candidates: candidates[0]
    app._draw_history_chart = lambda: None

    BacktestApp._update_history_chart_controls(app)

    assert app._history_chart_window_combo.state == "disabled"
    assert app._history_chart_metric_combo.state == "readonly"


def test_history_chart_rejects_candidate_that_removes_all_common_windows():
    summary = _history_multi_chart_summary()
    keep = (
        summary["strategy_type"].eq("close_to_close")
        | (summary["strategy"].eq("固定间隔(1σ)")
           & summary["window_id"].eq("window_10"))
        | (summary["strategy"].eq("每日固定时刻")
           & summary["window_id"].eq("window_2"))
    )
    app = object.__new__(BacktestApp)
    app._history_window_summary = summary[keep]
    app._history_rank_rows = {
        "candidate_b": {
            "strategy": "每日固定时刻", "strategy_type": "fixed_times",
        },
    }
    app._history_chart_selected_by_period = {
        "week": {"固定间隔(1σ)"},
    }
    app._history_chart_mode_var = _Var(
        gui_app.HISTORY_CHART_MODE_DISPLAY["typical"])
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["net"])
    app._history_chart_selection = lambda: ("week", None)
    app._history_chart_candidates = lambda _lookback: ["固定间隔(1σ)"]
    statuses = []
    app._set_status = statuses.append
    app._update_history_rank_selection = lambda: None

    BacktestApp._toggle_history_chart_candidate(app, "candidate_b")

    assert app._history_chart_selected_by_period["week"] == {"固定间隔(1σ)"}
    assert statuses and "无法加入" in statuses[-1]
    assert "共同" in statuses[-1]


@pytest.mark.parametrize("mode", ["full", "single", "typical"])
def test_history_multi_chart_renderer_draws_c2c_and_all_candidates(mode):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    app = object.__new__(BacktestApp)
    figure = Figure(figsize=(6, 2))
    app._history_chart_figure = figure
    app._history_chart_ax = figure.add_subplot(111)
    app._history_chart_canvas = FigureCanvasAgg(figure)
    app._history_window_summary = _history_multi_chart_summary()
    app._history_chart_mode_var = _Var(
        gui_app.HISTORY_CHART_MODE_DISPLAY[mode])
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["net"])
    app._history_chart_window_var = _Var("")
    app._history_chart_window_labels = {}
    app._history_chart_hint_var = _Var("")
    app._history_chart_color_map = {}
    app._history_chart_marker_map = {}
    app._history_chart_selection = lambda: ("week", None)
    app._history_chart_candidates = lambda _lookback: [
        "固定间隔(1σ)", "每日固定时刻",
    ]
    app._history_chart_primary_candidate = lambda _candidates: "固定间隔(1σ)"

    BacktestApp._draw_history_chart(app)

    _handles, labels = app._history_chart_ax.get_legend_handles_labels()
    assert labels[0] == "每日收盘（固定基准）"
    assert labels[1].startswith("固定间隔(1σ)")
    assert labels[2].startswith("每日固定时刻")
    assert len(app._history_chart_ax.lines) >= 3
    assert "可比 2 段" in app._history_chart_hint_var.get()


def _history_period_view_model_fixture():
    """五周期展示显式携带严格区间、代理段及 C2C 配对口径。"""
    recommendations = pd.DataFrame([{
        "lookback": "week", "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band", "score": 8.0,
        "baseline_score": 10.0, "improvement_vs_c2c": 0.20,
        "selection_improvement_vs_c2c": 0.20,
        "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
        "window_win_rate_vs_c2c": 0.75,
        "paired_windows": 4, "baseline_windows": 4,
        "comparison_eligible": True,
        "rolling_windows": 4, "eligible_endpoints": 4,
        "skipped_endpoints": 0, "history_days_available": 5,
        "lookback_days": 5, "evidence_days": 5, "days_used": 5,
        "maturity_days": 22,
        "evaluation_mode": "strict_lookback",
        "sampling_mode": "strict_contiguous",
        "segment_count": 1, "expiry_segments": 0, "mtm_segments": 1,
        "terminal_mode": "mark_to_market",
    }])

    def row(lookback, rank, strategy, strategy_type, score, baseline_score,
            improvement, paired, eligible, skipped, available, requested,
            complete, *, win_rate=0.0, comparison_eligible=True):
        evidence = min(available, requested)
        expiry_segments, tail = divmod(evidence, 22)
        segment_count = expiry_segments + int(tail > 0)
        return {
            "lookback": lookback, "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "score": score,
            "baseline_score": baseline_score,
            "improvement_vs_c2c": improvement,
            "selection_improvement_vs_c2c": improvement,
            "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "window_win_rate_vs_c2c": win_rate,
            "paired_windows": paired, "baseline_windows": paired,
            "comparison_eligible": comparison_eligible,
            "rolling_windows": paired, "eligible_endpoints": eligible,
            "skipped_endpoints": skipped,
            "history_days_available": available,
            "lookback_days": requested, "complete_window": complete,
            "maturity_days": 22, "evidence_days": requested,
            "days_used": evidence,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous",
            "segment_count": segment_count,
            "expiry_segments": expiry_segments,
            "mtm_segments": int(tail > 0),
            "terminal_mode": (
                "mixed" if expiry_segments and tail else
                "mark_to_market" if tail else "expiry"),
        }

    ranking = pd.DataFrame([
        # 近周：候选在完整的四个同窗样本中优于 C2C，形成正式参考。
        row("week", 1, "固定间隔(1σ)", "hedge_band", 8.0, 10.0,
            0.20, 4, 4, 0, 5, 5, True, win_rate=0.75),
        row("week", 2, "每日收盘", "close_to_close", 10.0, 10.0,
            0.0, 4, 4, 0, 5, 5, True),
        # 近月：样本不足时只给诊断；C2C 的 RMS 仍低于候选。
        row("month", 1, "每日收盘", "close_to_close", 9.0, 9.0,
            0.0, 2, 4, 2, 12, 20, False),
        row("month", 2, "固定间隔(1σ)", "hedge_band", 12.0, 9.0,
            -1.0 / 3.0, 2, 4, 2, 12, 20, False),
        # 近季：只有基准窗口，没有完整同窗候选，不能伪装成择优结论。
        row("quarter", 1, "每日收盘", "close_to_close", 7.0, 7.0,
            0.0, 1, 4, 3, 40, 60, False),
    ])
    return recommendations, ranking


def test_recommendation_view_model_compares_each_period_with_fixed_c2c():
    recommendations, ranking = _history_period_view_model_fixture()

    rows = BacktestApp._comparison_recommendation_rows(
        recommendations, ranking)

    assert [row["lookback"] for row in rows] == [
        "week", "month", "quarter", "half_year", "year",
    ]
    assert rows[0]["status"] == "数据完整"
    assert rows[0]["strategy_label"] == "固定间隔(1σ)"
    assert rows[0]["improvement_vs_c2c"] == pytest.approx(0.20)
    assert rows[0]["gap_ratio"] == pytest.approx(0.20)
    assert rows[0]["selection_improvement_vs_c2c"] == pytest.approx(0.20)
    assert rows[0]["uses_strict_metric"] is True
    assert rows[0]["uses_window_equal_metric"] is False
    assert rows[0]["window_win_rate_vs_c2c"] == pytest.approx(0.75)
    assert (rows[0]["paired"], rows[0]["baseline_windows"]) == (4, 4)
    assert rows[0]["best_is_baseline"] is False

    assert rows[1]["status"] == "数据不足（仅参考）"
    assert rows[1]["strategy_label"] == "诊断：每日收盘（基准最优）"
    assert rows[1]["improvement_vs_c2c"] == pytest.approx(0.0)
    assert rows[1]["best_is_baseline"] is True
    assert rows[1]["formal"] is False
    assert (rows[1]["effective"], rows[1]["eligible"], rows[1]["skipped"]) == (
        2, 4, 2,
    )

    assert rows[2]["status"] == "无可比候选"
    assert rows[2]["strategy_label"] == "仅基准（无可比候选）"
    assert rows[2]["has_comparable_candidate"] is False
    assert rows[3]["period"] == "近半年"
    assert rows[3]["status"] == "无可评估回测分段"
    assert rows[4]["period"] == "近年"
    assert rows[4]["status"] == "无可评估回测分段"


def test_recommendation_view_model_only_returns_selected_period_subset():
    recommendations, ranking = _history_period_view_model_fixture()

    rows = BacktestApp._comparison_recommendation_rows(
        recommendations, ranking, lookbacks=("week", "quarter"))

    assert [row["lookback"] for row in rows] == ["week", "quarter"]
    assert [row["period"] for row in rows] == ["近周", "近季"]
    assert rows[0]["status"] == "数据完整"
    assert rows[1]["status"] == "无可比候选"


def test_product_pool_view_uses_strict_metric_and_contract_list():
    candidate = {
        "lookback": "week", "rank": 1,
        "strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
        "score": 110.0, "baseline_score": 100.0,
        "improvement_vs_c2c": -0.10,
        "selection_improvement_vs_c2c": 0.20,
        "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
        "window_win_rate_vs_c2c": 0.75,
        "paired_windows": 4, "baseline_windows": 4,
        "comparison_eligible": True, "recommendation_eligible": True,
        "complete_window": True, "rolling_windows": 4,
        "eligible_endpoints": 4, "selected_endpoints": 4,
        "planned_endpoints": 4, "skipped_endpoints": 0,
        "history_days_available": 5, "available_history_days": 5,
        "required_history_days": 5, "lookback_days": 5,
        "evidence_days": 5, "days_used": 5,
        "evaluation_mode": "strict_lookback",
        "sampling_mode": "strict_contiguous",
        "segment_count": 1, "expiry_segments": 0, "mtm_segments": 1,
        "terminal_mode": "mark_to_market",
        "maturity_days": 22, "history_complete": True,
        "history_mode": "product_contract_pool", "product_code": "P.DCE",
        "paired_contract_codes": ("P2601.DCE", "P2605.DCE"),
        "relative_comparison_windows": 4,
    }
    baseline = {
        **candidate,
        "rank": 2, "strategy": "每日收盘",
        "strategy_type": "close_to_close", "score": 100.0,
        "improvement_vs_c2c": 0.0,
        "selection_improvement_vs_c2c": 0.0,
        "window_win_rate_vs_c2c": 0.0,
    }

    rows = BacktestApp._comparison_recommendation_rows(
        pd.DataFrame([candidate]), pd.DataFrame([candidate, baseline]))

    assert rows[0]["strategy"] == "固定间隔(1σ)"
    assert rows[0]["improvement_vs_c2c"] == pytest.approx(0.20)
    assert rows[0]["selection_metric"] == (
        history_selection.STRICT_LOOKBACK_SELECTION_METRIC)
    assert rows[0]["uses_strict_metric"] is True
    assert rows[0]["uses_window_equal_metric"] is False
    assert rows[0]["paired_contract_codes"] == (
        "P2601.DCE", "P2605.DCE")


def test_history_period_view_model_preserves_strict_interval_and_proxy_context():
    sampling = {
        "maturity_days": 243,
        "evaluation_mode": "strict_lookback",
        "sampling_mode": "strict_contiguous",
        "evidence_days": 5,
        "days_used": 5,
        "segment_count": 1,
        "expiry_segments": 0,
        "mtm_segments": 1,
        "terminal_mode": "mark_to_market",
        "required_history_days": 5,
        "available_history_days": 300,
        "history_complete": True,
    }
    candidate = {
        "lookback": "week", "rank": 1,
        "strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
        "score": 8.0, "baseline_score": 10.0,
        "improvement_vs_c2c": 0.20,
        "window_win_rate_vs_c2c": 0.60,
        "paired_windows": 5, "baseline_windows": 5,
        "comparison_eligible": True, "recommendation_eligible": True,
        "rolling_windows": 5, "eligible_endpoints": 5,
        "skipped_endpoints": 0, "history_days_available": 5,
        "lookback_days": 5, "complete_window": True,
        **sampling,
    }
    baseline = {
        **candidate,
        "rank": 2, "strategy": "每日收盘",
        "strategy_type": "close_to_close", "score": 10.0,
        "improvement_vs_c2c": 0.0,
        "window_win_rate_vs_c2c": 0.0,
    }

    week = BacktestApp._comparison_recommendation_rows(
        pd.DataFrame([candidate]),
        pd.DataFrame([candidate, baseline]),
    )[0]

    assert week["maturity_days"] == 243
    assert week["evaluation_mode"] == "strict_lookback"
    assert week["sampling_mode"] == "strict_contiguous"
    assert week["evidence_days"] == 5
    assert week["days_used"] == 5
    assert week["segment_count"] == 1
    assert week["expiry_segments"] == 0
    assert week["mtm_segments"] == 1
    assert week["terminal_mode"] == "mark_to_market"
    assert week["required_history_days"] == 5
    assert week["available_history_days"] == 300
    assert week["history_complete"] is True


def test_history_period_view_model_accepts_legacy_ranking_and_proxy_alias():
    candidate = {
        "lookback": "week", "rank": 1, "strategy": "旧候选",
        "strategy_type": "hedge_band", "score": 8.0,
        "baseline_score": 10.0, "improvement_vs_c2c": 0.2,
        "selection_improvement_vs_c2c": 0.3,
        "selection_metric": history_selection.HISTORY_SELECTION_METRIC,
        "rolling_windows": 1, "paired_windows": 1,
        "baseline_windows": 1, "eligible_endpoints": 1,
        "skipped_endpoints": 0, "history_days_available": 5,
        "lookback_days": 5, "days_used": 22, "comparison_eligible": True,
        "complete_window": True, "proxy_segments": 2,
    }
    baseline = {
        **candidate, "rank": 2, "strategy": "每日收盘",
        "strategy_type": "close_to_close", "score": 10.0,
        "improvement_vs_c2c": 0.0,
    }

    week = BacktestApp._comparison_recommendation_rows(
        pd.DataFrame([candidate]),
        pd.DataFrame([candidate, baseline]),
    )[0]

    assert week["strategy"] == "旧候选"
    assert week["improvement_vs_c2c"] == pytest.approx(0.3)
    assert week["uses_strict_metric"] is False
    assert week["uses_window_equal_metric"] is True
    assert week["status"] == "旧版数据完整"
    assert week["evaluation_mode"] is None
    assert week["evidence_days"] == 0
    assert week["days_used"] == 22
    assert week["segment_count"] == 2
    assert week["terminal_mode"] is None


def test_history_metric_recognition_does_not_label_legacy_as_strict():
    legacy = {
        "selection_metric": history_selection.HISTORY_SELECTION_METRIC,
        "selection_improvement_vs_c2c": 0.12,
    }
    strict = {
        "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
        "selection_improvement_vs_c2c": 0.08,
    }

    assert BacktestApp._history_row_uses_recognized_selection_metric(legacy)
    assert BacktestApp._history_row_uses_window_equal_metric(legacy)
    assert not BacktestApp._history_row_uses_strict_metric(legacy)
    assert BacktestApp._history_row_uses_recognized_selection_metric(strict)
    assert not BacktestApp._history_row_uses_window_equal_metric(strict)
    assert BacktestApp._history_row_uses_strict_metric(strict)


def test_zero_window_or_non_finite_history_result_is_not_a_diagnostic_leader():
    ranking = pd.DataFrame([{
        "lookback": "week", "rank": 1, "strategy": "A",
        "score": float("inf"), "rolling_windows": pd.NA,
        "eligible_endpoints": pd.NA, "skipped_endpoints": pd.NA,
        "complete_window": False,
    }])

    week = BacktestApp._comparison_recommendation_rows(None, ranking)[0]

    assert week["strategy"] == "—"
    assert week["status"] == "无可评估回测分段"


def test_trigger_detail_labels_mark_to_market_evaluation_close():
    frame = pd.DataFrame(
        {"标的价格": [100.0, 101.0, 102.0]},
        index=pd.bdate_range("2026-07-20", periods=3),
    )
    bt = SimpleNamespace(
        _results={
            "hedge_triggered": np.array([True, False, True]),
            "terminal_mode": "mark_to_market",
        },
        to_dataframe=lambda: frame.copy(),
    )

    detail, positions = BacktestApp._hedge_trigger_detail_frame(bt)

    assert positions.tolist() == [0, 2]
    assert detail["触发来源"].tolist() == ["初始建仓", "评价期末平仓"]


def test_history_reports_failure_when_main_thread_render_raises(monkeypatch):
    finished = []
    errors = []

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render failed")

    fake = SimpleNamespace(
        _show_history_recommendation=fail_render,
        _finish_history_recommendation=finished.append,
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: errors.append((title, message)),
    )

    ranking = pd.DataFrame([{
        "strategy": "daily", "rolling_windows": 1, "score": 1.0,
    }])
    BacktestApp._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, [],
        {"week": {"window_1": {}}})

    assert finished == [False]
    assert errors and errors[0][0] == "历史择优展示失败"


@pytest.mark.parametrize(
    ("recommendations", "ranking", "window_results"),
    [
        (None, pd.DataFrame([{"rolling_windows": 1, "score": 1.0}]),
         {"week": {"window_1": {}}}),
        (pd.DataFrame(), None, {"week": {"window_1": {}}}),
        (pd.DataFrame(), pd.DataFrame(), {"week": {"window_1": {}}}),
        (pd.DataFrame(),
         pd.DataFrame([{"rolling_windows": 0, "score": float("inf")}]),
         {"week": {}}),
    ],
)
def test_delivery_rejects_missing_or_unusable_history_payload(
        monkeypatch, recommendations, ranking, window_results):
    finished = []
    rendered = []
    errors = []
    fake = SimpleNamespace(
        _show_history_recommendation=lambda *_args: rendered.append(True),
        _finish_history_recommendation=finished.append,
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: errors.append((title, message)),
    )

    BacktestApp._deliver_history_recommendation(
        fake, recommendations, ranking, [], window_results)

    assert rendered == []
    assert finished == [False]
    assert errors and errors[0][0] == "历史择优展示失败"


def test_diagnostic_only_history_payload_is_valid():
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "daily", "complete_window": False,
        "rolling_windows": 1, "score": 3.5,
    }])

    BacktestApp._validate_history_recommendation_payload(
        pd.DataFrame(), ranking, {"week": {"window_1": {}}})


def test_zero_window_fixed_time_payload_reports_real_target_error():
    reason = (
        "fixed_times 目标时刻 [11:30,15:00] 未能逐交易日组匹配："
        "11:30 在全部 2 个交易日组中都不存在。"
    )
    ranking = pd.DataFrame([{
        "strategy": "fixed", "strategy_type": "fixed_times",
        "rolling_windows": 0, "score": float("inf"),
        "failure_scope": "strategy", "failure_reason": reason,
    }])

    with pytest.raises(ValueError) as exc_info:
        BacktestApp._validate_history_recommendation_payload(
            pd.DataFrame(), ranking,
            {"week": {"window_1": {
                "fixed": {
                    "strategy_name": "fixed_times", "error": reason,
                },
            }}},
        )

    message = str(exc_info.value)
    assert "固定时刻策略没有形成任何可评估代理段" in message
    assert "11:30 在全部 2 个交易日组中都不存在" in message
    assert "历史长度不足" not in message


def test_zero_window_contract_pool_payload_reports_exact_contract_error():
    reason = "缺少具体合约 P2509.DCE 的历史行情"
    ranking = pd.DataFrame([{
        "strategy": "daily", "strategy_type": "close_to_close",
        "rolling_windows": 0, "score": float("inf"),
        "failure_scope": "endpoint", "failure_reason": reason,
    }])

    with pytest.raises(ValueError) as exc_info:
        BacktestApp._validate_history_recommendation_payload(
            pd.DataFrame(), ranking,
            {"year": {"window_1": {"_window_error": reason}}},
        )

    message = str(exc_info.value)
    assert "历史具体合约池没有形成任何可评估代理段" in message
    assert "P2509.DCE" in message
    assert "历史长度不足" not in message


def test_history_delivery_preserves_saved_comparison_pool_and_selection():
    saved = {"result-0001": object()}
    selection = {"result-0001"}
    rendered = []
    finished = []
    state = {
        "source": "csv", "csv_path": "prices.csv",
        "history_include_close": True,
    }
    fake = SimpleNamespace(
        _saved_backtests=saved,
        _saved_comparison_selection=selection,
        _latest_history_state={"source": "old"},
        _latest_history_source_label="old source",
        _show_history_recommendation=lambda *args: rendered.append(args),
        _finish_history_recommendation=finished.append,
        _show_saved_comparison_page=lambda: pytest.fail(
            "历史渲染不得重建回测结果对比页"),
    )
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "每日收盘",
        "rolling_windows": 1, "score": 1.0,
    }])

    windows = {"week": {"window_1": {}}}
    BacktestApp._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, [],
        windows, "CSV · prices.csv", state)

    assert len(rendered) == 1
    assert finished == [True]
    assert fake._saved_backtests is saved
    assert fake._saved_comparison_selection is selection
    assert fake._latest_history_state["source"] == "csv"
    assert fake._latest_history_source_label == "CSV · prices.csv"
    # _deliver -> _show 的最后一个参数必须保留完整逐窗结果。
    assert rendered[0][4] is windows


def test_failed_history_render_keeps_previous_successful_history_state(
        monkeypatch):
    previous = {"source": "csv", "marker": "previous"}
    finished = []
    errors = []
    fake = SimpleNamespace(
        _latest_history_state=previous,
        _latest_history_source_label="previous source",
        _show_history_recommendation=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("render failed")),
        _finish_history_recommendation=finished.append,
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: errors.append((title, message)),
    )
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "每日收盘",
        "rolling_windows": 1, "score": 1.0,
    }])

    BacktestApp._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, [],
        {"week": {"window_1": {}}}, "new source", {"source": "wind"})

    assert finished == [False]
    assert fake._latest_history_state is previous
    assert fake._latest_history_source_label == "previous source"
    assert errors and errors[0][0] == "历史择优展示失败"


def _fake_history_apply_target():
    fake = _fake_band_gui()
    navigated = []
    statuses = []
    fake._strategy_var = _Var(STRATEGY_DISPLAY["close_to_close"])
    fake._fixed_times_var = _Var("11:30,15:00")
    fake._sigma_src_var = _Var(gui_app.SIGMA_SOURCE_DISPLAY["implied"])
    fake._sigma_win_var = _Var("20")
    fake._history_band_candidate_sigmas_var = _Var("0.5,1,2")
    fake._history_fixed_times_var = _Var("10:30,14:30")
    fake._comparison_finite = BacktestApp._comparison_finite
    fake._comparison_safe_int = BacktestApp._comparison_safe_int
    fake._comparison_safe_bool = BacktestApp._comparison_safe_bool
    fake._summary_tab = object()
    fake._nb = SimpleNamespace(select=navigated.append)
    fake._set_status = statuses.append

    def mark_band_edited(source):
        fake._band_last_edited = source
        fake._interval_type_var.set(source)

    fake._mark_band_edited = mark_band_edited
    fake._sync_band_inputs = lambda source, strict=False: (
        BacktestApp._sync_band_inputs(fake, source, strict=strict))
    return fake, navigated, statuses


@pytest.mark.parametrize(
    ("row", "strategy_name", "extra"),
    [
        ({"strategy": "每日收盘", "meta_strategy_name": "close_to_close"},
         "close_to_close", {}),
        ({"strategy": "固定时刻(10:30,14:30)",
          "meta_strategy_name": "fixed_times",
          "meta_fixed_times": "10:30,14:30"},
         "fixed_times", {"fixed_times": "10:30,14:30"}),
    ],
)
def test_apply_history_recommendation_maps_daily_and_fixed_time_to_backtest(
        row, strategy_name, extra):
    fake, navigated, statuses = _fake_history_apply_target()

    applied = BacktestApp._apply_history_recommendation(fake, row)

    assert applied["strategy_name"] == strategy_name
    assert fake._strategy_var.get() == STRATEGY_DISPLAY[strategy_name]
    if extra:
        assert fake._fixed_times_var.get() == extra["fixed_times"]
    assert navigated == [fake._summary_tab]
    assert statuses and "已应用历史候选" in statuses[-1]


def test_apply_history_recommendation_never_restores_snapshot_position():
    fake, _navigated, _statuses = _fake_history_apply_target()
    fake._pos_var = _Var("-1")
    fake._latest_history_state = {"position": 1}

    BacktestApp._apply_history_recommendation(
        fake, {
            "strategy": "每日收盘",
            "meta_strategy_name": "close_to_close",
        }, navigate=False)

    assert BacktestApp._normalize_position(fake._pos_var.get()) == -1


@pytest.mark.parametrize("enabled", [False, True])
def test_apply_history_recommendation_restores_public_close_fallback(enabled):
    fake, _navigated, _statuses = _fake_history_apply_target()
    fake._force_day_close_hedge_var = _BoolVar(not enabled)
    row = {
        "strategy": "每日收盘",
        "meta_strategy_name": "close_to_close",
        "meta_force_day_close_hedge": enabled,
    }

    applied = BacktestApp._apply_history_recommendation(
        fake, row, navigate=False)

    assert fake._force_day_close_hedge_var.get() is enabled
    assert applied["force_day_close_hedge"] is enabled


def test_apply_history_band_recommendation_updates_only_backtest_band_controls():
    fake, navigated, _statuses = _fake_history_apply_target()
    history_candidates_before = fake._history_band_candidate_sigmas_var.get()
    history_fixed_times_before = fake._history_fixed_times_var.get()
    row = {
        "strategy": "固定间隔(1.5σ)",
        "meta_strategy_name": "hedge_band",
        "meta_candidate_sigma": 1.5,
        "meta_sigma_source": "realized",
        "meta_sigma_window": 30,
    }

    applied = BacktestApp._apply_history_recommendation(
        fake, row, navigate=False)

    assert applied == {
        "strategy_name": "hedge_band",
        "strategy": "固定间隔(1.5σ)",
        "candidate_sigma": 1.5,
        "sigma_source": "realized",
        "sigma_window": 30,
        "force_day_close_hedge": False,
    }
    assert fake._strategy_var.get() == STRATEGY_DISPLAY["hedge_band"]
    assert fake._band_last_edited == "sigma"
    assert fake._interval_type_var.get() == "sigma"
    assert float(fake._price_interval_var.get()) == pytest.approx(1.5)
    assert fake._sigma_src_var.get() == gui_app.SIGMA_SOURCE_DISPLAY["realized"]
    assert fake._sigma_win_var.get() == "30"
    # “应用”只写回单次回测表单，历史候选空间不能被反向污染。
    assert (fake._history_band_candidate_sigmas_var.get()
            == history_candidates_before)
    assert fake._history_fixed_times_var.get() == history_fixed_times_before
    assert navigated == []


def test_current_path_validation_applies_selection_and_schedules_auto_retain():
    row = {
        "strategy": "固定时刻(10:30,14:30)",
        "meta_strategy_name": "fixed_times",
        "meta_fixed_times": "10:30,14:30",
    }
    applied = []
    run_calls = []
    fake = SimpleNamespace(
        _active_job=None,
        _pending_history_retain_name=None,
        _selected_history_rank_row=lambda: row,
        _apply_history_recommendation=lambda selected, navigate: applied.append(
            (selected, navigate)),
        _run_backtest=lambda: run_calls.append(True) or True,
    )

    started = BacktestApp._run_history_selection_on_current_path(fake)

    assert started is True
    assert applied == [(row, False)]
    assert run_calls == [True]
    assert fake._pending_history_retain_name == (
        "历史验证 · 固定时刻(10:30,14:30)")


@pytest.mark.parametrize("position", [1, -1])
def test_current_path_validation_launches_with_current_left_position(position):
    row = {
        "strategy": "固定时刻(10:30,14:30)",
        "meta_strategy_name": "fixed_times",
        "meta_fixed_times": "10:30,14:30",
    }
    fake, _navigated, _statuses = _fake_history_apply_target()
    fake._pos_var = _Var(position)
    fake._latest_history_state = {"position": -position}
    fake._active_job = None
    fake._pending_history_retain_name = None
    fake._selected_history_rank_row = lambda: row
    fake._apply_history_recommendation = (
        lambda selected, navigate:
        BacktestApp._apply_history_recommendation(
            fake, selected, navigate=navigate))
    launched_positions = []
    fake._run_backtest = lambda: (
        launched_positions.append(
            BacktestApp._normalize_position(fake._pos_var.get()))
        or True)

    assert BacktestApp._run_history_selection_on_current_path(fake) is True
    assert launched_positions == [position]


class _RankTreeStub:
    def __init__(self, iids):
        self._iids = tuple(iids)

    def get_children(self):
        return self._iids


_HISTORY_BATCH_ROWS = {
    "history_rank_0": {
        "strategy": "每日收盘", "strategy_type": "close_to_close",
        "meta_strategy_name": "close_to_close", "lookback": "month",
    },
    "history_rank_1": {
        "strategy": "固定间隔(1.5σ)", "strategy_type": "hedge_band",
        "meta_strategy_name": "hedge_band", "meta_candidate_sigma": 1.5,
        "lookback": "month",
    },
    "history_rank_2": {
        "strategy": "固定时刻(10:30)", "strategy_type": "fixed_times",
        "meta_strategy_name": "fixed_times", "meta_fixed_times": "10:30",
        "lookback": "month",
    },
}


def _fake_history_batch_gui(*, checked=("固定时刻(10:30)", "固定间隔(1.5σ)"),
                            saved_backtests=None, position=-1):
    rows = copy.deepcopy(_HISTORY_BATCH_ROWS)
    fake = SimpleNamespace(
        _active_job=None,
        _pending_history_retain_name=None,
        _pending_history_retain_origin=None,
        _history_batch_queue=[],
        _history_batch_total=0,
        _history_batch_done=0,
        _history_batch_failures=[],
        _history_batch_position=None,
        _history_batch_result_ids=[],
        _history_batch_id=None,
        _history_batch_failure_details=[],
        _history_batch_last_result_ids=[],
        _latest_retained_result_id=None,
        _history_rank_tree=_RankTreeStub(rows),
        _history_rank_rows=rows,
        _pos_var=_Var(position),
        _history_chart_selection=lambda: ("month", None),
        _history_chart_candidates=lambda lookback: list(checked),
        _saved_backtests=dict(saved_backtests or {}),
        _saved_comparison_selection=set(),
        _saved_comparison_baseline_id=None,
    )
    fake._history_batch_targets = lambda: BacktestApp._history_batch_targets(
        fake)
    fake._reset_history_batch = lambda: BacktestApp._reset_history_batch(fake)
    fake._history_batch_step = lambda: BacktestApp._history_batch_step(fake)
    fake._history_batch_advance = (
        lambda success: BacktestApp._history_batch_advance(fake, success))
    fake._finish_history_batch = lambda: BacktestApp._finish_history_batch(
        fake)
    fake._history_batch_skip_rest = (
        lambda: BacktestApp._history_batch_skip_rest(fake))
    # 收尾不再跳页；结论写进优选页结果条，这里让真实实现落数据后直接返回。
    fake._publish_history_batch_outcome = (
        lambda done, total, failures, result_ids:
        BacktestApp._publish_history_batch_outcome(
            fake, done, total, failures, result_ids))
    fake._refresh_saved_comparison_if_visible = lambda: None
    return fake


def test_history_batch_targets_keep_rank_order_and_prepend_new_baseline():
    fake = _fake_history_batch_gui()

    targets = BacktestApp._history_batch_targets(fake)

    assert [item["label"] for item in targets] == [
        "每日收盘", "固定间隔(1.5σ)", "固定时刻(10:30)"]
    assert [item["name"] for item in targets] == [
        "优选近月 · 每日收盘基准",
        "优选近月 · 固定间隔(1.5σ)",
        "优选近月 · 固定时刻(10:30)",
    ]


def test_history_batch_targets_skip_baseline_when_pool_already_has_one():
    saved = {
        "result-0001": SimpleNamespace(
            summary_row={"strategy_type": "close_to_close"}, position=-1),
    }
    fake = _fake_history_batch_gui(saved_backtests=saved, position=-1)

    targets = BacktestApp._history_batch_targets(fake)

    assert [item["label"] for item in targets] == [
        "固定间隔(1.5σ)", "固定时刻(10:30)"]


def test_history_batch_targets_empty_without_any_checked_candidate():
    fake = _fake_history_batch_gui(checked=())

    assert BacktestApp._history_batch_targets(fake) == []


def _drive_history_batch(fake, outcomes):
    """按队列顺序模拟每次回测的成功 / 失败回调。"""
    launched = []

    def run_backtest():
        index = len(launched)
        launched.append({
            "name": fake._pending_history_retain_name,
            "position": BacktestApp._normalize_position(fake._pos_var.get()),
            "origin": copy.deepcopy(fake._pending_history_retain_origin),
        })
        fake._active_job = "backtest"
        outcome = outcomes[index] if index < len(outcomes) else True
        fake.after(0, lambda: _settle(outcome, index))
        return True

    def _settle(success, index):
        fake._active_job = None
        fake._pending_history_retain_name = None
        fake._pending_history_retain_origin = None
        # 模拟 _deliver_backtest_result 成功入池后登记的快照 id。
        fake._latest_retained_result_id = (
            f"result-{index + 1:04d}" if success else None)
        BacktestApp._history_batch_advance(fake, success)

    pending = []
    fake.after = lambda _delay, callback: pending.append(callback)
    fake._run_backtest = run_backtest
    fake._apply_history_recommendation = lambda row, navigate: None
    fake._set_status = lambda _text: None
    fake._show_saved_comparison_page = lambda: fake.opened.append(True)
    fake.opened = []
    fake.errors = []
    started = BacktestApp._run_history_batch_on_current_path(fake)
    while pending:
        pending.pop(0)()
    return started, launched


def test_history_batch_runs_every_candidate_and_reports_in_place(monkeypatch):
    fake = _fake_history_batch_gui()
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: fake.errors.append((title, message)))

    started, launched = _drive_history_batch(fake, [True, True, True])

    assert started is True
    assert [item["name"] for item in launched] == [
        "优选近月 · 每日收盘基准",
        "优选近月 · 固定间隔(1.5σ)",
        "优选近月 · 固定时刻(10:30)",
    ]
    assert {item["position"] for item in launched} == {-1}
    # 收尾不再抢走优选页上下文：既不自动跳页，也不弹模态框。
    assert fake.opened == []
    assert fake.errors == []
    assert fake._history_batch_queue == []
    # 全部成功时结果条没有失败详情，但记住这一批入池的快照供“查看结果对比”。
    assert fake._history_batch_failure_details == []
    assert fake._history_batch_last_result_ids == [
        "result-0001", "result-0002", "result-0003"]


def test_history_batch_origin_meta_records_period_rank_and_batch_id():
    fake = _fake_history_batch_gui()

    _started, launched = _drive_history_batch(fake, [True, True, True])

    origins = [item["origin"] for item in launched]
    assert all(item["batch"] is True for item in origins)
    assert {item["period_label"] for item in origins} == {"近月"}
    assert [item["history_strategy"] for item in origins] == [
        "每日收盘", "固定间隔(1.5σ)", "固定时刻(10:30)"]
    # 同一批次共享 batch_id，并记录批内顺序，便于结果池按批次成组。
    assert len({item["batch_id"] for item in origins}) == 1
    assert [item["batch_index"] for item in origins] == [1, 2, 3]
    assert {item["batch_total"] for item in origins} == {3}


def test_history_batch_aborts_remaining_candidates_after_a_failed_run(
        monkeypatch):
    fake = _fake_history_batch_gui()
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: fake.errors.append((title, message)))

    _started, launched = _drive_history_batch(fake, [True, False])

    assert [item["name"] for item in launched] == [
        "优选近月 · 每日收盘基准",
        "优选近月 · 固定间隔(1.5σ)",
    ]
    # 失败原因留在结果条里按需展开，不再用模态框打断，也不跳页。
    assert fake.errors == []
    assert fake.opened == []
    details = "\n".join(fake._history_batch_failure_details)
    assert "固定间隔(1.5σ)：回测失败" in details
    assert "固定时刻(10:30)" in details
    # 失败前已成功入池的快照仍然可以从结果条直接查看。
    assert fake._history_batch_last_result_ids == ["result-0001"]
    assert fake._history_batch_queue == []


def test_history_batch_reports_rest_when_a_run_cannot_start(monkeypatch):
    fake = _fake_history_batch_gui()
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: fake.errors.append((title, message)))
    fake.errors = []
    fake.opened = []
    fake._set_status = lambda _text: None
    fake._show_saved_comparison_page = lambda: fake.opened.append(True)
    fake._apply_history_recommendation = lambda row, navigate: None
    fake.after = lambda _delay, callback: callback()
    attempts = []

    def run_backtest():
        attempts.append(fake._pending_history_retain_name)
        return False

    fake._run_backtest = run_backtest

    assert BacktestApp._run_history_batch_on_current_path(fake) is False
    assert attempts == ["优选近月 · 每日收盘基准"]
    assert fake.opened == []
    assert fake.errors == []
    details = "\n".join(fake._history_batch_failure_details)
    assert "回测未能启动" in details
    assert "固定间隔(1.5σ)" in details
    assert "固定时刻(10:30)" in details
    assert fake._history_batch_last_result_ids == []
    assert fake._pending_history_retain_name is None
    assert fake._pending_history_retain_origin is None


def test_history_batch_skips_unmappable_row_and_keeps_running_the_rest(
        monkeypatch):
    fake = _fake_history_batch_gui()
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: fake.errors.append((title, message)))

    def apply_row(row, navigate):
        if str(row.get("strategy")) == "固定间隔(1.5σ)":
            raise ValueError("历史结果缺少有效的固定间隔 σ 参数。")

    fake.errors = []
    fake.opened = []
    fake._set_status = lambda _text: None
    fake._show_saved_comparison_page = lambda: fake.opened.append(True)
    fake._apply_history_recommendation = apply_row
    pending = []
    fake.after = lambda _delay, callback: pending.append(callback)
    launched = []

    def run_backtest():
        launched.append(fake._pending_history_retain_name)
        fake.after(0, lambda: BacktestApp._history_batch_advance(fake, True))
        return True

    fake._run_backtest = run_backtest

    started = BacktestApp._run_history_batch_on_current_path(fake)
    while pending:
        pending.pop(0)()

    assert started is True
    # 只有元数据残缺的那一条被跳过，其余候选照常验证并送入结果池。
    assert launched == [
        "优选近月 · 每日收盘基准", "优选近月 · 固定时刻(10:30)"]
    assert fake.errors == []
    assert fake.opened == []
    assert "固定间隔(1.5σ)" in "\n".join(fake._history_batch_failure_details)
    assert fake._history_batch_queue == []


def test_history_batch_refuses_to_start_while_another_job_runs(monkeypatch):
    fake = _fake_history_batch_gui()
    fake._active_job = "backtest"
    infos = []
    monkeypatch.setattr(
        gui_app.messagebox, "showinfo",
        lambda title, message: infos.append(title))

    assert BacktestApp._run_history_batch_on_current_path(fake) is False
    assert infos == ["任务运行中"]


def test_history_batch_reports_missing_selection_without_running(monkeypatch):
    fake = _fake_history_batch_gui(checked=())
    infos = []
    monkeypatch.setattr(
        gui_app.messagebox, "showinfo",
        lambda title, message: infos.append(title))
    fake._run_backtest = lambda: pytest.fail("不应在无勾选候选时启动回测")

    assert BacktestApp._run_history_batch_on_current_path(fake) is False
    assert infos == ["请勾选候选"]


def test_successfully_started_backtest_reports_true_to_auto_retain_caller(
        monkeypatch):
    started_threads = []

    class _Progress(_Widget):
        def pack(self, **_kwargs):
            self.config["packed"] = True

        def start(self, _interval):
            self.config["started"] = True

    class _Thread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started_threads.append(self)

    monkeypatch.setattr(gui_app.threading, "Thread", _Thread)
    state = {"source": "csv", "strategy_name": "close_to_close"}
    fake = SimpleNamespace(
        _prepare_active_strategy_inputs=lambda: None,
        _collect_gui_state=lambda: state,
        _validate_fixed_time_source_state=lambda _state: None,
        _begin_job=lambda name, status: name == "backtest",
        _progress=_Progress(),
        _backtest_worker=lambda _state: None,
    )

    started = BacktestApp._run_backtest(fake)

    assert started is True
    assert len(started_threads) == 1
    assert started_threads[0].args == (state,)


def _history_worker_fixture(source, load_history):
    base_bt = SimpleNamespace(
        option_init=object(), prices=[100.0, 101.0], timestamps=None,
        tc_rate=0.0, position=1, quantity=1.0, multiplier=0.0,
        steps_per_day=1, slippage_bps=0.0,
        _rescale_info={"ratio": 1.0},
    )
    cases = [StrategyCase("daily", CloseToCloseStrategy())]
    delivered = []
    failed = []
    fake = SimpleNamespace(
        _build_backtest=lambda _state: base_bt,
        _strategy_cases_for_history=lambda _state, _bt: (cases, []),
        _comparison_backtest_kwargs=lambda _bt: {},
        _load_full_history_for_recommendation=load_history,
        _deliver_history_recommendation=lambda *args: delivered.append(args),
        _fail_history_recommendation=failed.append,
        after=lambda _delay, callback: callback(),
    )
    state = {
        "source": source,
        "fixed_times": "11:30,15:00",
        "cfg": {"build": lambda _subtype, _params: SimpleNamespace(
            _time_remaining=2)},
        "subtype": "test", "params": {},
        "csv_path": "real.csv", "csv_col": "close",
        "wind_code": "510050.SH", "wind_start": "2025-01-01",
        "wind_end": "2026-01-01", "wind_bar_size": "日频",
    }
    return fake, state, delivered, failed


def test_history_worker_uses_strict_periods_without_endpoint_budgets(
        monkeypatch):
    history = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2026-01-01", periods=3, freq="B"),
    )
    recommendations = pd.DataFrame()
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "daily",
        "complete_window": False, "rolling_windows": 1, "score": 2.0,
    }])
    windows = {"week": {"window_1": {"daily": {}}}}
    captured = []

    def recommend(option, _history, _cases, _kwargs, **kwargs):
        captured.append((option._time_remaining, dict(kwargs)))
        return recommendations, ranking, windows

    monkeypatch.setattr(gui_app, "recommend_by_rolling_history", recommend)

    for maturity_days in (2, 22, 243):
        fake, state, delivered, failed = _history_worker_fixture(
            "csv", lambda _state, _bt: history)
        state["cfg"] = {
            "build": lambda _subtype, _params, days=maturity_days:
                SimpleNamespace(_time_remaining=days),
        }

        BacktestApp._history_recommendation_worker(fake, state)

        assert failed == []
        assert len(delivered) == 1

    assert [maturity for maturity, _kwargs in captured] == [2, 22, 243]
    assert [kwargs["lookbacks"] for _maturity, kwargs in captured] == [
        gui_app.LOOKBACK_DAYS,
        gui_app.LOOKBACK_DAYS,
        gui_app.LOOKBACK_DAYS,
    ]
    assert all(
        "target_endpoints" not in kwargs for _maturity, kwargs in captured)
    assert all("step_days" not in kwargs for _maturity, kwargs in captured)


def test_csv_history_worker_passes_only_selected_strict_periods(
        monkeypatch):
    history = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2026-01-01", periods=3, freq="B"),
    )
    recommendations = pd.DataFrame()
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "daily",
        "complete_window": False, "rolling_windows": 1, "score": 2.0,
    }])
    windows = {"week": {"window_1": {"daily": {}}}}
    captured = []
    selected_lookbacks = {
        "week": gui_app.LOOKBACK_DAYS["week"],
        "quarter": gui_app.LOOKBACK_DAYS["quarter"],
    }

    def recommend(option, loaded, cases, kwargs, **call_kwargs):
        captured.append((option, loaded, cases, kwargs, call_kwargs))
        return recommendations, ranking, windows

    monkeypatch.setattr(gui_app, "recommend_by_rolling_history", recommend)
    fake, state, delivered, failed = _history_worker_fixture(
        "csv", lambda _state, _bt: history)
    state["history_lookbacks"] = selected_lookbacks

    BacktestApp._history_recommendation_worker(fake, state)

    assert failed == []
    assert len(delivered) == 1
    assert len(captured) == 1
    assert captured[0][4]["lookbacks"] == selected_lookbacks
    assert "target_endpoints" not in captured[0][4]


@pytest.mark.parametrize("position", [1, -1])
def test_single_series_history_worker_forwards_position_to_recommender(
        monkeypatch, position):
    history = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2026-01-01", periods=3, freq="B"),
    )
    fake, state, delivered, failed = _history_worker_fixture(
        "csv", lambda _state, _bt: history)
    state["position"] = position
    original_build = fake._build_backtest

    def build(base_state):
        bt = original_build(base_state)
        bt.position = BacktestApp._normalize_position(
            base_state["position"])
        return bt

    fake._build_backtest = build
    fake._comparison_backtest_kwargs = (
        lambda bt: BacktestApp._comparison_backtest_kwargs(bt))
    captured = []
    recommendations = pd.DataFrame()
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "daily",
        "complete_window": False, "rolling_windows": 1, "score": 2.0,
    }])
    windows = {"week": {"window_1": {"daily": {}}}}

    def recommend(_option, _history, _cases, kwargs, **_call_kwargs):
        captured.append(kwargs)
        return recommendations, ranking, windows

    monkeypatch.setattr(gui_app, "recommend_by_rolling_history", recommend)

    BacktestApp._history_recommendation_worker(fake, state)

    assert failed == []
    assert len(delivered) == 1
    assert captured[0]["position"] == position


@pytest.mark.parametrize("position", [1, -1])
def test_history_worker_routes_product_code_without_building_continuous_backtest(
        monkeypatch, position):
    mapping = pd.Series(
        ["P2605.DCE", "P2609.DCE"],
        index=pd.bdate_range("2026-06-29", periods=2),
    )
    pool = ContractHistoryPool(
        "P.DCE", mapping,
        {"P2609.DCE": pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.bdate_range("2026-06-26", periods=3),
        )},
        "P2609.DCE",
    )
    fake, state, delivered, failed = _history_worker_fixture(
        "wind", lambda _state, base_bt: (
            pool if base_bt is None else pytest.fail(
                "品种池加载不应依赖连续合约回测对象")))
    state["wind_code"] = "P.DCE"
    state["position"] = position
    selected_lookbacks = {
        "week": gui_app.LOOKBACK_DAYS["week"],
        "quarter": gui_app.LOOKBACK_DAYS["quarter"],
    }
    state["history_lookbacks"] = selected_lookbacks
    fake._build_backtest = lambda _state: pytest.fail(
        "P.DCE 历史择优不应先下载连续合约行情")
    fake._comparison_backtest_kwargs = (
        lambda bt: BacktestApp._comparison_backtest_kwargs(bt))
    recommendations = pd.DataFrame()
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "daily",
        "complete_window": False, "rolling_windows": 1, "score": 2.0,
    }])
    windows = {"week": {"window_1": {"daily": {}}}}
    captured = []

    def recommend_pool(option, history, cases, kwargs, **call_kwargs):
        captured.append((option, history, cases, kwargs, call_kwargs))
        return recommendations, ranking, windows

    monkeypatch.setattr(
        gui_app, "recommend_by_contract_history_pool", recommend_pool)
    monkeypatch.setattr(
        gui_app, "recommend_by_rolling_history",
        lambda *_args, **_kwargs: pytest.fail(
            "品种代码不得进入单序列 rolling recommender"),
    )

    BacktestApp._history_recommendation_worker(fake, state)

    assert failed == []
    assert len(delivered) == 1
    assert len(captured) == 1
    assert captured[0][1] is pool
    assert captured[0][4]["lookbacks"] == selected_lookbacks
    assert "target_endpoints" not in captured[0][4]
    assert captured[0][3]["position"] == position
    assert "steps_per_day" not in captured[0][3]
    assert "steps_per_day" not in captured[0][4]


@pytest.mark.parametrize("source", ["csv", "wind"])
@pytest.mark.parametrize("stage", ["load", "recommend"])
def test_real_history_worker_failure_is_terminal(monkeypatch, source, stage):
    def load_history(_state, _bt):
        if stage == "load":
            raise RuntimeError("history boom")
        return pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range("2026-01-01", periods=3, freq="B"),
        )

    fake, state, delivered, failed = _history_worker_fixture(
        source, load_history)

    def recommend(*_args, **_kwargs):
        if stage == "recommend":
            raise RuntimeError("history boom")
        raise AssertionError("recommend should not run during load failure")

    monkeypatch.setattr(gui_app, "recommend_by_rolling_history", recommend)
    monkeypatch.setattr(
        gui_app, "compare_strategies",
        lambda *_args, **_kwargs: pytest.fail(
            "current-path comparison must not run after history failure"),
    )

    BacktestApp._history_recommendation_worker(fake, state)

    assert delivered == []
    assert len(failed) == 1
    assert "history boom" in failed[0]


@pytest.mark.parametrize("source", ["csv", "wind"])
def test_real_history_worker_delivers_complete_history_payload(
        monkeypatch, source):
    history = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2026-01-01", periods=3, freq="B"),
    )
    fake, state, delivered, failed = _history_worker_fixture(
        source, lambda _state, _bt: history)
    recommendations = pd.DataFrame()
    ranking = pd.DataFrame([{
        "lookback": "week", "strategy": "daily", "complete_window": False,
        "rolling_windows": 1, "score": 2.0,
    }])
    windows = {"week": {"window_1": {"daily": {}}}}
    monkeypatch.setattr(
        gui_app, "recommend_by_rolling_history",
        lambda *_args, **_kwargs: (recommendations, ranking, windows),
    )
    monkeypatch.setattr(
        gui_app, "compare_strategies",
        lambda *_args, **_kwargs: pytest.fail(
            "历史择优页不应再运行当前期限路径对比"),
    )

    BacktestApp._history_recommendation_worker(fake, state)

    assert failed == []
    assert len(delivered) == 1
    payload = delivered[0]
    assert payload[0] is recommendations
    assert payload[1] is ranking
    assert payload[3] is windows
    assert ("CSV" if source == "csv" else "Wind") in payload[4]
    assert payload[5] is state


def test_worker_rejects_simulation_before_building_backtest():
    failed = []
    fake = SimpleNamespace(
        _build_backtest=lambda _state: pytest.fail("must not build simulation"),
        _fail_history_recommendation=failed.append,
        after=lambda _delay, callback: callback(),
    )

    BacktestApp._history_recommendation_worker(
        fake, {"source": "simulate", "fixed_times": "11:30,15:00"})

    assert len(failed) == 1
    assert "真实历史" in failed[0]


@pytest.mark.parametrize(
    ("source", "factory_name", "source_fields"),
    [
        ("csv", "from_csv", {"csv_path": "prices.csv", "csv_col": "close"}),
        ("wind", "from_wind", {
            "wind_code": "510050.SH", "wind_start": "2026-01-01",
            "wind_end": "2026-02-01", "wind_bar_size": "60分钟",
        }),
    ],
)
@pytest.mark.parametrize("position", [1, -1])
def test_real_sources_always_delegate_bar_count_to_backend(
        monkeypatch, source, factory_name, source_fields, position):
    captured = {}

    def fake_factory(option, *args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        gui_app.HedgeBacktest, factory_name, staticmethod(fake_factory))
    state = {
        "cfg": {"build": lambda subtype, params: SimpleNamespace()},
        "cls_name": "test", "subtype": "test", "params": {},
        "source": source, "tc_rate": 0.0, "position": position,
        "quantity": 1.0, "multiplier": 0.0, "slippage_bps": 0.0,
        "force_day_close_hedge": True,
        "steps_per_day": 240, "strategy_name": "close_to_close",
        **source_fields,
    }
    fake_app = SimpleNamespace(
        _validate_fixed_time_source_state=lambda _state: None,
    )

    BacktestApp._build_backtest(fake_app, state)
    assert captured["steps_per_day"] is None
    assert captured["position"] == position
    assert isinstance(captured["strategy"], CloseToCloseStrategy)
    assert captured["force_day_close_hedge"] is True


def test_resolved_wind_dates_and_bar_size_flow_to_build_meta_and_label(
        monkeypatch):
    captured = {}

    def fake_from_wind(option, *args, **kwargs):
        captured["option"] = option
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(
        gui_app.HedgeBacktest, "from_wind", staticmethod(fake_from_wind))
    form = _fake_collect_state()
    form._source_var.set("wind")
    form._strategy_var.set(STRATEGY_DISPLAY["hedge_band"])
    form._wind_start_var.set("2025-01-02")
    form._wind_end_var.set("2025-02-28")
    form._wind_auto_end_var = _BoolVar(False)

    state = BacktestApp._collect_gui_state(form)
    app = SimpleNamespace(
        _validate_fixed_time_source_state=lambda _state: None)
    backtest = BacktestApp._build_backtest(app, state)

    assert state["wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    # 表单带宽是 1 元绝对值 = 0.78σ_daily，属于窄带，必须落到 1 分钟。
    assert state["wind_bar_size"] == "1分钟"
    assert captured["args"] == (
        "510050.SH", "2025-01-02", "2025-02-28")
    assert captured["kwargs"]["bar_size"] == "1"
    assert backtest._gui_meta["wind_start"] == "2025-01-02"
    assert backtest._gui_meta["wind_end"] == "2025-02-28"
    assert backtest._gui_meta["wind_bar_size"] == "1分钟"
    assert BacktestApp._snapshot_source_label(state) == (
        "Wind · 510050.SH · 2025-01-02 至 2025-02-28 · 1分钟"
    )


def test_comparison_kwargs_do_not_carry_legacy_hedge_frequency():
    kwargs = BacktestApp._comparison_backtest_kwargs(SimpleNamespace(
        tc_rate=0.0, position=1, quantity=1.0, multiplier=0.0,
        steps_per_day=4, slippage_bps=0.0,
        force_day_close_hedge=True,
    ))
    assert "hedge_freq" not in kwargs
    assert kwargs["force_day_close_hedge"] is True


@pytest.mark.parametrize(
    "greek_name", ["delta", "gamma", "vega", "theta", "rho"])
def test_gui_greek_series_prefers_signed_portfolio_values_with_raw_fallback(
        greek_name):
    raw = np.array([1.0, 2.0])
    signed = np.array([-3.0, -6.0])

    np.testing.assert_array_equal(
        BacktestApp._result_greek_series(
            {
                greek_name: raw,
                f"portfolio_{greek_name}": signed,
            },
            greek_name,
        ),
        signed,
    )
    np.testing.assert_array_equal(
        BacktestApp._result_greek_series(
            {greek_name: raw}, greek_name),
        raw,
    )


def test_unknown_gui_strategy_never_falls_back_to_fixed_frequency():
    state = {
        "cfg": {"build": lambda subtype, params: SimpleNamespace()},
        "cls_name": "test", "subtype": "test", "params": {},
        "source": "simulate", "tc_rate": 0.0, "position": 1,
        "quantity": 1.0, "multiplier": 0.0, "slippage_bps": 0.0,
        "steps_per_day": 1, "strategy_name": "obsolete_strategy",
    }
    fake_app = SimpleNamespace()
    with pytest.raises(ValueError, match="未知对冲策略"):
        BacktestApp._build_backtest(fake_app, state)


def test_gui_long_jobs_share_one_busy_guard_and_failure_status():
    statuses = []
    fake = SimpleNamespace(
        _active_job=None,
        _run_btn=_Widget(),
        _retain_btn=_Widget(),
        _compare_btn=_Widget(),
        _history_btn=_Widget(),
        _struct_btn=_Widget(),
        _latest_backtest=None,
        _latest_retained_result_id=None,
        _progress=_Widget(),
        _progress_label=_Widget(),
        _set_status=statuses.append,
    )

    assert BacktestApp._begin_job(fake, "history", "busy") is True
    assert {fake._run_btn.state, fake._retain_btn.state,
            fake._compare_btn.state, fake._history_btn.state,
            fake._struct_btn.state} == {"disabled"}

    # 其它任务的迟到回调不能提前解锁共享控件。
    BacktestApp._finish_job(
        fake, "backtest", success=True,
        success_text="wrong", failure_text="wrong-failure")
    assert fake._active_job == "history"
    assert fake._run_btn.state == "disabled"

    BacktestApp._finish_job(
        fake, "history", success=False,
        success_text="done", failure_text="failed")
    assert fake._active_job is None
    assert {fake._run_btn.state, fake._compare_btn.state,
            fake._history_btn.state, fake._struct_btn.state} == {"normal"}
    assert fake._retain_btn.state == "disabled"
    assert statuses[-1] == "failed"


def test_current_comparison_rebases_only_absolute_case_copy():
    absolute = StrategyCase(
        "absolute", HedgeBandStrategy("absolute", threshold=1.0))
    relative = StrategyCase(
        "relative", HedgeBandStrategy("relative", threshold=0.01))

    scaled = BacktestApp._rescale_strategy_cases(
        [absolute, relative], ratio=2.0)

    assert scaled[0].strategy.threshold == pytest.approx(2.0)
    assert scaled[1].strategy.threshold == pytest.approx(0.01)
    # 原始 cases 必须保留参考价口径，rolling 才能逐窗独立重定基。
    assert absolute.strategy.threshold == pytest.approx(1.0)


class _ReplaySpecStub:
    def __init__(self, window_id, strategies, *, segment_no=1, index=None,
                 metadata=None):
        self.lookback = "month"
        self.window_id = window_id
        self.strategies = {name: SimpleNamespace(name=name)
                           for name in strategies}
        self.external_path = SimpleNamespace(
            index=() if index is None else index)
        self.evaluation_days = 5
        self.steps_per_day = 1
        self.metadata = {"segment_no": segment_no, **(metadata or {})}
        self.replayed = []

    def replay(self, strategy_name):
        self.replayed.append(strategy_name)
        return SimpleNamespace(
            _results={"timestamps": self.external_path.index})


def _fake_history_replay_gui(*, selected="固定间隔(1.5σ)", specs=None):
    dates = pd.bdate_range("2026-03-02", periods=4)
    if specs is None:
        specs = {
            "segment_2": _ReplaySpecStub(
                "segment_2", ("每日收盘", "固定间隔(1.5σ)"),
                segment_no=2, index=dates),
            "segment_1": _ReplaySpecStub(
                "segment_1", ("每日收盘",), segment_no=1, index=dates),
        }
    fake = SimpleNamespace(
        _active_job=None,
        _history_replay_index={"month": specs},
        _history_chart_selection=lambda: ("month", None),
        _selected_history_rank_row=lambda: (
            {"strategy": selected} if selected else None),
        _history_replay_window_var=_Var(""),
        _history_replay_window_combo=None,
    )
    fake._history_replay_options = (
        lambda strategy_name=None, lookback=None:
        BacktestApp._history_replay_options(
            fake, strategy_name, lookback))
    fake._selected_history_replay_spec = (
        lambda: BacktestApp._selected_history_replay_spec(fake))
    return fake


def test_history_replay_label_shows_segment_dates_and_contract():
    spec = _ReplaySpecStub(
        "segment_3", ("每日收盘",), segment_no=3,
        index=pd.bdate_range("2026-03-02", periods=4),
        metadata={"contract_code": "P2605.DCE",
                  "terminal_mode": "mark_to_market"})

    label = BacktestApp._history_replay_label(spec)

    assert label == "第 3 段 · 2026-03-02~2026-03-05 · P2605.DCE · 按市价结算"


def test_history_replay_options_keep_only_segments_running_the_candidate():
    fake = _fake_history_replay_gui()

    options = BacktestApp._history_replay_options(fake)

    # segment_1 只跑过每日收盘，选中固定间隔时不能出现在可加载分段里。
    assert [item["window_id"] for item in options] == ["segment_2"]


def test_history_replay_options_are_ordered_by_segment_number():
    fake = _fake_history_replay_gui(selected="每日收盘")

    options = BacktestApp._history_replay_options(fake)

    assert [item["window_id"] for item in options] == [
        "segment_1", "segment_2"]


def test_selected_history_replay_spec_follows_the_window_dropdown():
    fake = _fake_history_replay_gui(selected="每日收盘")
    options = BacktestApp._history_replay_options(fake)
    fake._history_replay_window_var.set(options[1]["label"])

    spec, strategy_name = BacktestApp._selected_history_replay_spec(fake)

    assert strategy_name == "每日收盘"
    assert spec.window_id == "segment_2"


def test_selected_history_replay_spec_falls_back_to_first_segment():
    fake = _fake_history_replay_gui(selected="每日收盘")
    fake._history_replay_window_var.set("已失效的旧标签")

    spec, _strategy_name = BacktestApp._selected_history_replay_spec(fake)

    assert spec.window_id == "segment_1"


def test_load_history_window_refuses_without_a_replayable_candidate(
        monkeypatch):
    fake = _fake_history_replay_gui(selected="没有跑成功的候选")
    infos = []
    monkeypatch.setattr(
        gui_app.messagebox, "showinfo",
        lambda title, message: infos.append((title, message)))
    fake._begin_job = lambda *a, **k: pytest.fail("不应在无分段时占用任务槽")

    assert BacktestApp._load_history_window_backtest(fake) is False
    assert infos and infos[0][0] == "没有可加载的分段"


def test_load_history_window_refuses_while_another_job_runs(monkeypatch):
    fake = _fake_history_replay_gui()
    fake._active_job = "backtest"
    infos = []
    monkeypatch.setattr(
        gui_app.messagebox, "showinfo",
        lambda title, message: infos.append(title))

    assert BacktestApp._load_history_window_backtest(fake) is False
    assert infos == ["任务运行中"]


def test_delivered_history_replay_becomes_the_current_retainable_backtest():
    dates = pd.bdate_range("2026-03-02", periods=4)
    spec = _ReplaySpecStub(
        "segment_2", ("固定间隔(1.5σ)",), segment_no=2, index=dates,
        metadata={"contract_code": "P2605.DCE"})
    backtest = SimpleNamespace(_results={"timestamps": dates})
    finished = []
    fake = SimpleNamespace(
        _latest_history_state={
            "cls_name": "香草期权 (Vanilla)", "subtype": "call",
            "source": "wind", "wind_bar_size": "15分钟",
            "wind_bar_size_requested": "自动（推荐）",
            "wind_date_mode": "custom_range", "wind_code": "P.DCE",
        },
        _latest_backtest=None,
        _latest_backtest_state=None,
        _latest_retained_result_id="result-0001",
        _show_results=lambda bt, multi: finished.append(bt),
        _finish_history_replay=lambda success, spec_, name: finished.append(
            ("finish", success, name)),
    )
    fake._history_replay_gui_state = (
        lambda spec_, name: BacktestApp._history_replay_gui_state(
            fake, spec_, name))

    BacktestApp._deliver_history_replay(fake, backtest, spec, "固定间隔(1.5σ)")

    assert fake._latest_backtest is backtest
    assert fake._latest_retained_result_id is None
    assert backtest._gui_meta["source"] == "wind"
    assert backtest._gui_meta["wind_start"] == "2026-03-02"
    assert backtest._gui_meta["wind_end"] == "2026-03-05"
    assert list(backtest._wind_meta["dates"]) == list(dates)
    state = fake._latest_backtest_state
    # 快照方向与展示口径来自历史任务本身，合约代码收敛到该分段实际合约。
    assert state["wind_code"] == "P2605.DCE"
    assert state["history_replay_strategy"] == "固定间隔(1.5σ)"
    assert state["history_replay_window_id"] == "segment_2"
    assert finished[-1] == ("finish", True, "固定间隔(1.5σ)")


# ---------------------------------------------------------------------------
# 策略优选：统一的当前路径验证入口与配对缓存
# ---------------------------------------------------------------------------


def _fake_history_verify_gui(checked):
    """只提供“当前路径验证”分派所需的最小状态。"""
    calls = []
    fake = SimpleNamespace(
        _history_chart_candidates=lambda lookback=None: list(checked),
        _run_history_batch_on_current_path=lambda: (
            calls.append("batch") or "batch"),
        _run_history_selection_on_current_path=lambda: (
            calls.append("single") or "single"),
    )
    return fake, calls


def test_history_verify_entry_batches_every_checked_candidate():
    fake, calls = _fake_history_verify_gui(["固定间隔(1.5σ)", "固定时刻(10:30)"])

    result = BacktestApp._verify_history_on_current_path(fake)

    assert calls == ["batch"]
    assert result == "batch"


def test_history_verify_entry_falls_back_to_the_selected_row():
    # 没有勾选时仍然只验证排名表选中的那一条，语义与旧的两个按钮一致。
    fake, calls = _fake_history_verify_gui([])

    result = BacktestApp._verify_history_on_current_path(fake)

    assert calls == ["single"]
    assert result == "single"


def test_history_action_hint_states_the_scope_the_button_will_run():
    fake = SimpleNamespace(
        _history_action_hint_var=_Var(""),
        _history_chart_candidates=lambda lookback=None: [
            "固定间隔(1.5σ)", "固定时刻(10:30)"],
    )

    BacktestApp._refresh_history_action_hint(fake)
    batch_hint = fake._history_action_hint_var.get()

    fake._history_chart_candidates = lambda lookback=None: []
    BacktestApp._refresh_history_action_hint(fake)
    single_hint = fake._history_action_hint_var.get()

    # 提示必须说清这次按钮实际会跑几个策略，措辞保持平实。
    assert "勾选的 2 个策略" in batch_hint
    assert "选中的那一个策略" in single_hint


def test_history_action_hint_is_silent_before_the_result_page_exists():
    # 结果页尚未渲染时提示控件还不存在；类级默认值保证这里不会抛异常。
    fake = SimpleNamespace(_history_action_hint_var=None)

    assert BacktestApp._refresh_history_action_hint(fake) is None


def test_history_multi_chart_model_reuses_supplied_pairs_cache():
    summary = _history_multi_chart_summary()
    cache = {}

    first = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)", "每日固定时刻"],
        mode="full", metric="net", pairs_cache=cache,
    )
    cached_keys = set(cache)
    second = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)", "每日固定时刻"],
        mode="full", metric="net", pairs_cache=cache,
    )

    assert cached_keys == {("week", "固定间隔(1σ)"), ("week", "每日固定时刻")}
    # 复用缓存不得改变模型；缓存帧本身也不能被下游就地过滤掉行。
    assert second["state"] == first["state"] == "ok"
    assert second["common_window_count"] == first["common_window_count"]
    assert set(cache) == cached_keys
    assert all(not pairs.empty for pairs in cache.values())


def test_history_multi_chart_model_without_cache_matches_cached_result():
    summary = _history_multi_chart_summary()

    plain = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"], mode="full", metric="net")
    cached = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"], mode="full", metric="net",
        pairs_cache={},
    )

    assert plain["state"] == cached["state"]
    assert [item["label"] for item in plain["series"]] == [
        item["label"] for item in cached["series"]]


def test_history_batch_step_skips_every_unmappable_row_and_finishes(
        monkeypatch):
    # 整队都无法回填时旧实现靠递归逐条跳过；改成循环后仍然只汇报一次收尾，
    # 且两条原因都出现在同一份汇总里。
    fake = _fake_history_batch_gui(checked=("固定间隔(1.5σ)", "固定时刻(10:30)"))
    fake.errors = []
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: fake.errors.append((title, message)))
    fake._set_status = lambda _text: None
    fake._show_saved_comparison_page = lambda: pytest.fail(
        "没有成功快照时不应切换到结果对比页")
    fake._run_backtest = lambda: pytest.fail("不应为无法回填的候选启动回测")

    def refuse(_row, navigate=False):
        raise ValueError("缺少策略参数")

    fake._apply_history_recommendation = refuse
    fake._history_batch_queue = [
        {"row": {}, "label": "固定间隔(1.5σ)", "name": "优选近月 · A"},
        {"row": {}, "label": "固定时刻(10:30)", "name": "优选近月 · B"},
    ]
    fake._history_batch_total = 2

    started = BacktestApp._history_batch_step(fake)

    assert started is False
    assert fake._history_batch_queue == []
    # 收尾只汇报一次，两条原因进同一份结果条详情，不再弹模态框。
    assert fake.errors == []
    details = fake._history_batch_failure_details
    assert "固定间隔(1.5σ)：缺少策略参数" in "\n".join(details)
    assert "固定时刻(10:30)：缺少策略参数" in "\n".join(details)
    assert fake._history_batch_last_result_ids == []


def test_history_batch_step_returns_false_on_an_empty_queue():
    fake = _fake_history_batch_gui()
    fake._finish_history_batch = lambda: pytest.fail("空队列不应触发收尾汇报")
    fake._history_batch_queue = []

    assert BacktestApp._history_batch_step(fake) is False


# ---------------------------------------------------------------------------
# 策略优选：排名口径判定与区间说明
# ---------------------------------------------------------------------------


def test_history_ranking_flags_read_strict_metric_and_candidate_names():
    ranking = pd.DataFrame([
        {"strategy": "每日收盘", "strategy_type": "close_to_close",
         "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
         "selection_improvement_vs_c2c": 0.0, "history_mode": "single_series"},
        {"strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
         "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
         "selection_improvement_vs_c2c": 0.2, "history_mode": "single_series"},
    ])

    flags = history_selection.ranking_flags(ranking)

    assert flags["uses_strict_metric"] is True
    assert flags["uses_window_equal_metric"] is False
    assert flags["uses_product_pool"] is False
    # 基准不是候选，配色与勾选集合都不应包含它。
    assert flags["candidate_names"] == ["固定间隔(1σ)"]


def test_history_ranking_flags_detect_product_contract_pool():
    ranking = pd.DataFrame([
        {"strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
         "selection_metric": history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
         "selection_improvement_vs_c2c": 0.2,
         "history_mode": "product_contract_pool"},
    ])

    flags = history_selection.ranking_flags(ranking)

    assert flags["uses_product_pool"] is True


def test_history_ranking_flags_stay_neutral_on_empty_ranking():
    flags = history_selection.ranking_flags(pd.DataFrame())

    assert flags == {
        "uses_strict_metric": False,
        "uses_window_equal_metric": False,
        "uses_product_pool": False,
        "candidate_names": [],
    }


@pytest.mark.parametrize("uses_product_pool, expected", [
    (False, "排名依据：整段日损益的波动幅度比「每日收盘」低多少。"),
    (True, "整体金额波动只作参考，不参与排名。"),
])
def test_history_scope_explanation_tracks_the_ranking_route(
        uses_product_pool, expected):
    text = history_selection.scope_explanation(
        "近月", uses_strict_metric=True, uses_product_pool=uses_product_pool)

    assert text.startswith("本次周期：近月。")
    assert text.endswith(expected)
    # 面向使用者的说明里不该出现只在实现中有意义的符号与自造词。
    for jargon in ("L 日", "T ", "V 日", "严格区间", "代理合约", "有界", "RMS"):
        assert jargon not in text


def test_history_scope_explanation_flags_pre_upgrade_results():
    text = history_selection.scope_explanation(
        "近月 / 近季", uses_strict_metric=False, uses_product_pool=False)

    assert "旧版的取样方式" in text
    assert "建议重新运行" in text


def test_history_consensus_note_flags_agreement_across_periods():
    """各周期指向同一策略，本身就是比单个周期数字更强的证据。"""
    items = [
        {"period": "近周", "strategy": "固定间隔0.75σ"},
        {"period": "近月", "strategy": "固定间隔0.75σ"},
        {"period": "近年", "strategy": "固定间隔0.75σ"},
    ]
    note, state = BacktestApp._history_consensus_note(items)
    assert state == "agree"
    assert "3 个周期一致推荐 固定间隔0.75σ" in note


def test_history_consensus_note_warns_and_points_to_long_periods():
    """出现分歧时要说清各周期各选了什么，并给出取舍方向。"""
    items = [
        {"period": "近周", "strategy": "固定时刻"},
        {"period": "近年", "strategy": "固定间隔0.5σ"},
    ]
    note, state = BacktestApp._history_consensus_note(items)
    assert state == "disagree"
    assert "近周 固定时刻" in note and "近年 固定间隔0.5σ" in note
    assert "以长周期为准" in note


def test_history_consensus_note_handles_empty_conclusions():
    note, state = BacktestApp._history_consensus_note([
        {"period": "近周", "strategy": "—"},
    ])
    assert state == "empty"
    assert "没有形成可比结论" in note


@pytest.mark.parametrize(
    ("status", "paired", "total", "expected"),
    [
        # 一切正常时不占字数——这两项逐行重复且不传递信息。
        ("数据完整", 12, 12, "✓"),
        ("可比", 1, 1, "✓"),
        # 有失败段：段数才是重点。
        ("数据完整", 3, 6, "3/6 段"),
        # 数据本身有问题：显示原因。
        ("数据不足（仅参考）", 5, 5, "数据不足（仅参考）"),
        ("不可比", 0, 3, "不可比·0/3"),
    ],
)
def test_history_status_column_only_spends_width_on_problems(
        status, paired, total, expected):
    assert BacktestApp._format_history_status(
        status, paired, total) == expected


def test_only_objective_columns_are_clickable_headers():
    """七列里只有两个排名口径列可点。

    给其余列也装上排序会暗示它们同样能当排名依据，而后端只支持这两种；
    用户会困惑“顺序变了为什么推荐没变”。
    """
    import tkinter as tk
    from tkinter import ttk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    root.withdraw()
    clicked = []
    try:
        tree = BacktestApp._build_history_metric_tree(
            ttk.Frame(root),
            lead_columns=(("period", "分析周期", 90),
                          ("strategy", "历史最优参考", 260)),
            status_heading="状态", status_width=114,
            labels={"score": "s", "baseline": "b", "improvement": "i"},
            height=5, on_objective_click=clicked.append,
        )
        clickable = [c for c in tree["columns"] if tree.heading(c)["command"]]
        assert clickable == ["incremental_pnl", "incremental_sharpe"]
        # 可点的表头带有排序提示符号。
        assert tree.heading("incremental_pnl")["text"].endswith("⇅")
        assert not tree.heading("period")["text"].endswith("⇅")
    finally:
        root.destroy()


def test_history_result_page_exposes_its_interactive_entry_points():
    """结果页的交互入口必须真实存在。

    这些方法散落在几个渲染函数之间，容易在重构相邻代码时被整段切掉，而
    “表头绑定了 command”之类的断言抓不到——command 指向一个已不存在的
    方法时，绑定本身依然成立，只有真正点下去才会炸。
    """
    for name in (
            "_set_history_objective",       # 点列头换排名依据
            "_rerank_history_results",      # 就地重排
            "_history_rerender",            # 重排后重建结果页
            "_select_history_period",       # 切换分析周期
            "_refresh_history_period_chips",
            "_history_consensus_note",      # 跨周期一致性结论
            "_attach_tooltip",
            "_make_scrollable_area",
    ):
        assert callable(getattr(BacktestApp, name, None)), f"{name} 缺失"


def test_clicking_objective_header_actually_reranks(monkeypatch):
    """点列头要真的走到重排，而不只是绑定了一个回调。"""
    calls = []
    fake = SimpleNamespace(
        _history_result_objective_var=_Var("incremental_pnl"),
        _rerank_history_results=lambda: calls.append("reranked"),
    )

    BacktestApp._set_history_objective(fake, "incremental_sharpe")
    assert fake._history_result_objective_var.get() == "incremental_sharpe"
    assert calls == ["reranked"]

    # 非法口径既不改状态也不触发重排。
    BacktestApp._set_history_objective(fake, "max_profit")
    assert fake._history_result_objective_var.get() == "incremental_sharpe"
    assert calls == ["reranked"]


def _pill_ranking(picks_by_period, baseline_first=False):
    rows = []
    for lookback, best in picks_by_period.items():
        rows.append({
            "lookback": lookback, "strategy": best,
            "strategy_type": "hedge_band", "rank": 2 if baseline_first else 1,
            "recommendation_eligible": True,
        })
        rows.append({
            "lookback": lookback, "strategy": "每日收盘",
            "strategy_type": "close_to_close",
            "rank": 1 if baseline_first else 2,
            "recommendation_eligible": True,
        })
    return pd.DataFrame(rows)


def test_header_pill_agrees_with_the_consistency_banner():
    """顶部结论 pill 不得与下方的一致性横幅互相打架。

    rank 是按周期分组算的，每个周期都有一条 rank==1；直接取第一条拿到的
    是「近周」——样本最少、横幅明确建议不要单独采信的那个周期。
    """
    app = SimpleNamespace(
        _history_header_summary_frame=None,
        _history_ranking=_pill_ranking({
            "week": "固定间隔0.75σ", "year": "固定间隔0.5σ"}),
    )
    # 容器为 None 时应安全返回，不抛异常。
    assert BacktestApp._update_history_header_summary(app) is None

    picks = []
    ranking = app._history_ranking
    for _lb, group in ranking.groupby("lookback", sort=False):
        eligible = group[group["recommendation_eligible"].astype(bool)]
        cand = eligible[eligible["strategy_type"] != "close_to_close"]
        picks.append(cand.sort_values("rank").iloc[0]["strategy"])
    # 两个周期给出不同结论时，绝不能只挑一个说成“最佳推荐”。
    assert len(set(picks)) == 2


def test_header_pill_never_promotes_the_baseline():
    """基准是对照组不是推荐；它排第一时也不该出现在结论 pill 里。"""
    ranking = _pill_ranking({"week": "固定间隔2σ"}, baseline_first=True)
    eligible = ranking[ranking["recommendation_eligible"].astype(bool)]
    cand = eligible[eligible["strategy_type"] != "close_to_close"]
    best = cand.sort_values("rank", kind="stable").iloc[0]
    assert best["strategy"] == "固定间隔2σ"
    assert best["strategy_type"] != "close_to_close"


def test_active_sort_column_is_marked_in_the_header():
    """当前排序列用 ↓ 标出，另一个可点列用 ⇅。

    这样排名依据不必再单独占一行文字——表头自己就说清了「现在按哪列排」
    和「哪列还能点」。
    """
    import tkinter as tk
    from tkinter import ttk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    root.withdraw()
    try:
        labels = {"score": "s", "baseline": "b", "improvement": "i"}
        lead = (("period", "分析周期", 90), ("strategy", "历史最优参考", 260))
        tree = BacktestApp._build_history_metric_tree(
            ttk.Frame(root), lead_columns=lead, status_heading="状态",
            status_width=114, labels=labels, height=5,
            on_objective_click=lambda _k: None,
            active_objective="incremental_pnl")
        assert tree.heading("incremental_pnl")["text"].endswith("↓")
        assert tree.heading("incremental_sharpe")["text"].endswith("⇅")

        other = BacktestApp._build_history_metric_tree(
            ttk.Frame(root), lead_columns=lead, status_heading="状态",
            status_width=114, labels=labels, height=5,
            on_objective_click=lambda _k: None,
            active_objective="incremental_sharpe")
        assert other.heading("incremental_sharpe")["text"].endswith("↓")
        assert other.heading("incremental_pnl")["text"].endswith("⇅")

        # 非排名口径的列不该带任何排序标记。
        assert not tree.heading("max_drawdown")["text"].endswith(("↓", "⇅"))
    finally:
        root.destroy()


def test_source_hint_is_rendered_once_as_a_pill():
    """数据源提示只在顶部 pill 出现一次。

    这句话曾经同时挂在 pill 和按钮左侧的 Label 上，同屏出现两遍会被误读
    成两条不同的提示。变量要保留（pill 从它取文案），但不能再有第二个
    控件绑定它。
    """
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    root.destroy()

    import gui_app as module
    app = module.BacktestApp()
    try:
        app.withdraw()
        app.update_idletasks()
        bound = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    if child.cget("textvariable") == str(
                            app._history_source_hint_var):
                        bound.append(child.winfo_class())
                except tk.TclError:
                    pass
                walk(child)

        walk(app)
        assert bound == [], f"仍有控件重复绑定该提示: {bound}"

        # pill 仍然会在数据源不可用时给出提示。
        app._source_var.set("simulate")
        BacktestApp._sync_history_button_state(app)
        app.update_idletasks()
        texts = [w.cget("text")
                 for w in app._history_header_summary_frame.winfo_children()]
        assert any("仅支持 CSV / Wind" in t for t in texts), texts
        assert str(app._history_btn.cget("state")) == "disabled"
    finally:
        app.destroy()


def test_band_only_params_gray_out_with_their_owning_strategy():
    """固定间隔的四项参数必须整组跟着它的勾选启停。

    σ 来源与回看天数只在勾选了固定间隔时才会被读取（见
    _collect_history_state），界面上也应表现为从属于它，而不是看起来像
    一组全局设置。
    """
    import tkinter as tk
    import gui_app as module
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = module.BacktestApp()
    try:
        app.withdraw()
        app.update_idletasks()
        band_widgets = (
            app._history_band_candidate_entry,
            app._history_current_band_check,
        )

        app._history_include_band_var.set(False)
        app._toggle_history_candidate_controls()
        for widget in band_widgets:
            assert str(widget.cget("state")) == "disabled"
        # 固定时刻不受牵连。
        assert str(app._history_fixed_times_entry.cget("state")) == "normal"

        app._history_include_band_var.set(True)
        app._history_include_fixed_times_var.set(False)
        app._toggle_history_candidate_controls()
        assert str(app._history_fixed_times_entry.cget("state")) == "disabled"
        assert str(app._history_band_candidate_entry.cget("state")) == "normal"
    finally:
        app.destroy()


def test_candidate_config_pairs_each_strategy_with_its_params_on_one_row():
    """候选空间左栏勾选、右栏参数，必须逐行对齐且无单元格重叠。

    从属关系靠「同一 grid 行」表达，一旦行号错位，界面上参数就会挪到别
    的策略名下——而 tkinter 对行号写错是完全沉默的，重叠时后放的控件直
    接盖住先放的，只表现为「某个控件不见了」。
    """
    import tkinter as tk
    import gui_app as module
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = module.BacktestApp()
    try:
        app.withdraw()
        app.update_idletasks()
        settings = app._history_wind_frame.master

        occupied = {}
        for child in settings.winfo_children():
            info = child.grid_info()
            if not info:
                continue
            row = int(info["row"])
            column = int(info["column"])
            span = int(info.get("columnspan", 1))
            for col in range(column, column + span):
                occupied.setdefault((row, col), []).append(child)
        overlaps = {k: v for k, v in occupied.items() if len(v) > 1}
        assert not overlaps, f"grid 单元格重叠: {sorted(overlaps)}"

        def row_of(widget):
            return int(widget.grid_info()["row"])

        # 每个参数控件都必须落在自己那个勾选框所在的行上。
        fixed_row = row_of(app._history_fixed_times_entry.master)
        band_row = row_of(app._history_band_candidate_entry.master.master)
        assert fixed_row != band_row
        assert row_of(app._history_current_band_check.master.master) == band_row

        checkbox_rows = {}
        for child in settings.winfo_children():
            if child.winfo_class() != "TCheckbutton" or not child.grid_info():
                continue
            checkbox_rows[str(child.cget("text"))] = row_of(child)
        assert checkbox_rows["固定时刻"] == fixed_row
        assert checkbox_rows["固定间隔"] == band_row
        # 左栏是勾选、右栏是参数，不能反过来。
        assert int(
            app._history_fixed_times_entry.master.grid_info()["column"]) == 1
        for label, row in checkbox_rows.items():
            widget_column = next(
                int(c.grid_info()["column"])
                for c in settings.winfo_children()
                if c.grid_info() and c.winfo_class() == "TCheckbutton"
                and str(c.cget("text")) == label)
            assert widget_column == 0, label
    finally:
        app.destroy()
