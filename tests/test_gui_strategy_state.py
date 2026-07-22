from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gui_app
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
    with pytest.raises(ValueError, match="日内|分钟"):
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
        SimpleNamespace(),
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
        SimpleNamespace(),
        {
            "source": "wind",
            "wind_code": "NO_NIGHT.TEST",
            "wind_bar_size": "15min",
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

    with pytest.raises(ValueError, match="只有每日收盘 C2C 基准") as exc_info:
        BacktestApp._strategy_cases_for_history(
            SimpleNamespace(),
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
        ("fixed_times", "11:30,15:00", "15min"),
        ("fixed_times", "10:10", "5min"),
        ("fixed_times", "10:07", "1min"),
        ("hedge_band", "", "15min"),
    ],
)
def test_wind_auto_bar_size_follows_strategy_observation_needs(
        strategy_name, fixed_times, expected):
    assert BacktestApp._resolve_wind_bar_size(
        WIND_AUTO_BAR_SIZE,
        strategy_name=strategy_name,
        fixed_times=fixed_times,
    ) == expected


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
    ) == "15min"
    # 自动频率在 GUI 主线程解析，只读本地已知 session，不能触发 Wind wss。
    assert calls == [("NO_NIGHT.TEST", False)]


def test_manual_daily_wind_rejects_fixed_time_but_keeps_band_choice():
    with pytest.raises(ValueError, match="固定时刻策略需要分钟行情"):
        BacktestApp._resolve_wind_bar_size(
            "日频", strategy_name="fixed_times",
            fixed_times="11:30,15:00",
        )

    assert BacktestApp._resolve_wind_bar_size(
        "日频", strategy_name="hedge_band") == "日频"


def test_history_wind_auto_range_covers_year_plus_option_maturity():
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
    required = ANNUAL_DAYS + maturity_days
    expected_span = BacktestApp._calendar_span_for_trading_days(required)

    assert resolved["wind_required_trade_days"] == required
    assert resolved["wind_end"] == asof.isoformat()
    assert resolved["wind_start"] == (
        asof - datetime.timedelta(days=expected_span)).isoformat()
    assert resolved["wind_bar_size"] == "15min"
    assert resolved["wind_date_mode"] == "history_auto_year_plus_maturity"
    assert resolved["wind_sigma_warmup_days"] == 0


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
    required = gui_app.LOOKBACK_DAYS["quarter"] + maturity_days

    assert resolved["history_lookbacks"] == selected
    assert resolved["history_max_lookback_days"] == gui_app.LOOKBACK_DAYS[
        "quarter"]
    assert resolved["wind_required_trade_days"] == required
    assert resolved["wind_start"] == (
        asof - datetime.timedelta(
            days=BacktestApp._calendar_span_for_trading_days(required))
    ).isoformat()
    assert resolved["wind_date_mode"] == (
        "history_auto_selected_period_plus_maturity")


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
    required = ANNUAL_DAYS + 22 + 30

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
        fake,
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
        fake,
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
    cases = BacktestApp._band_cases_for_comparison({
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
    cases = BacktestApp._band_cases_for_comparison({
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
    cases = BacktestApp._band_cases_for_comparison({
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
        BacktestApp._band_cases_for_comparison({
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
        _wind_end_var=_Var("2026-01-01"), _wind_bar_size_var=_Var("日频"),
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
    fake._history_sigma_src_var = _Var(
        gui_app.SIGMA_SOURCE_DISPLAY["realized"])
    fake._history_sigma_win_var = _Var("30")
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


def test_history_collects_its_own_candidates_times_and_sigma_configuration():
    fake = _fake_history_collect_state()
    # 候选时刻与 σ 配置独立于普通回测区；收盘兜底是唯一共享例外。
    fake._fixed_times_var.set("09:45")
    fake._sigma_src_var.set(gui_app.SIGMA_SOURCE_DISPLAY["implied"])
    fake._sigma_win_var.set("20")

    state = BacktestApp._collect_history_state(fake)

    assert state["band_candidate_sigmas"] == (0.5, 1.0, 2.0)
    assert state["sigma_source"] == "realized"
    assert state["sigma_window"] == 30
    assert state["fixed_times"] == "10:30,14:30"
    assert state["history_include_close"] is True
    assert state["history_include_fixed_times"] is True
    assert state["history_include_band"] is True
    assert state["history_include_current_band"] is True


def test_history_default_period_controls_freeze_all_lookbacks_and_budgets():
    fake = _fake_history_collect_state()
    fake._history_period_vars = _fake_history_period_vars()
    expected_lookbacks = {
        key: gui_app.LOOKBACK_DAYS[key]
        for key, _label in gui_app.HISTORY_PERIOD_DEFS
    }
    expected_targets = {
        key: gui_app.HISTORY_TARGET_ENDPOINTS[key]
        for key in expected_lookbacks
    }

    state = BacktestApp._collect_history_state(fake)

    assert state["history_lookbacks"] == expected_lookbacks
    assert state["history_target_endpoints"] == expected_targets
    # 任务快照必须是普通字典；启动后再改复选框不能篡改本次实验。
    for variable in fake._history_period_vars.values():
        variable.set(False)
    assert state["history_lookbacks"] == expected_lookbacks
    assert state["history_target_endpoints"] == expected_targets


def test_history_collects_only_checked_periods_in_canonical_order():
    fake = _fake_history_collect_state()
    fake._history_period_vars = _fake_history_period_vars(
        selected=("half_year", "month"))

    state = BacktestApp._collect_history_state(fake)

    assert state["history_lookbacks"] == {
        "month": gui_app.LOOKBACK_DAYS["month"],
        "half_year": gui_app.LOOKBACK_DAYS["half_year"],
    }
    assert state["history_target_endpoints"] == {
        "month": gui_app.HISTORY_TARGET_ENDPOINTS["month"],
        "half_year": gui_app.HISTORY_TARGET_ENDPOINTS["half_year"],
    }


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
    fake._wind_bar_size_var.set(WIND_AUTO_BAR_SIZE)
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
    fake._wind_bar_size_var.set("日频")
    fake._history_wind_asof_var = _Var("2025-06-30")
    fake._history_wind_start_var = _Var("not-used-while-auto")
    fake._history_wind_auto_start_var = _BoolVar(True)
    fake._history_wind_bar_size_var = _Var(WIND_AUTO_BAR_SIZE)

    state = BacktestApp._collect_history_state(fake)
    required = (
        ANNUAL_DAYS + state["params"]["T_days"] + state["sigma_window"])
    expected_start = (
        datetime.date(2025, 6, 30) - datetime.timedelta(
            days=BacktestApp._calendar_span_for_trading_days(required))
    ).isoformat()

    assert state["history_wind_asof"] == "2025-06-30"
    assert state["wind_end"] == "2025-06-30"
    assert state["history_wind_start"] == expected_start
    assert state["wind_start"] == expected_start
    assert state["wind_required_trade_days"] == required
    assert state["wind_sigma_warmup_days"] == state["sigma_window"]
    assert state["history_wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    # 默认固定时刻 + 固定间隔候选冻结为同一实际粒度，保持公平比较。
    assert state["wind_bar_size"] == "15min"
    assert BacktestApp._history_recommendation_source_label(state) == (
        f"Wind · 510050.SH · {expected_start} 至 2025-06-30 · 15min"
    )


def test_history_manual_daily_with_fixed_time_fails_during_state_collection():
    fake = _fake_history_collect_state()
    fake._source_var.set("wind")
    fake._history_wind_asof_var = _Var("2025-06-30")
    fake._history_wind_start_var = _Var("2024-01-02")
    fake._history_wind_auto_start_var = _BoolVar(False)
    fake._history_wind_bar_size_var = _Var("日频")

    with pytest.raises(ValueError, match="固定时刻策略需要分钟行情"):
        BacktestApp._collect_history_state(fake)


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
    assert collected_history["history_target_endpoints"] == {
        key: gui_app.HISTORY_TARGET_ENDPOINTS[key]
        for key, _label in gui_app.HISTORY_PERIOD_DEFS
    }


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
        SimpleNamespace(), state, SimpleNamespace())

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
    fake._history_sigma_src_var = _ForbiddenVar()
    fake._history_sigma_win_var = _ForbiddenVar()

    with pytest.raises(ValueError, match="至少再启用一种候选策略"):
        BacktestApp._collect_history_state(fake)


def test_history_implied_sigma_does_not_read_unused_hv_window():
    class _ForbiddenVar:
        def get(self):
            raise AssertionError("implied σ 不应读取 HV 窗口")

    fake = _fake_history_collect_state()
    fake._history_sigma_src_var.set(
        gui_app.SIGMA_SOURCE_DISPLAY["implied"])
    fake._history_sigma_win_var = _ForbiddenVar()

    state = BacktestApp._collect_history_state(fake)

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
        _history_sigma_src_var=_Var(
            gui_app.SIGMA_SOURCE_DISPLAY["realized"]),
        _history_fixed_times_entry=_Widget(),
        _history_band_candidate_entry=_Widget(),
        _history_current_band_check=_Widget(),
        _history_sigma_src_combo=_Widget(),
        _history_sigma_win_entry=_Widget(),
    )

    BacktestApp._toggle_history_candidate_controls(fake)

    assert fake._history_fixed_times_entry.state == "disabled"
    assert fake._history_band_candidate_entry.state == "normal"
    assert fake._history_current_band_check.state == "normal"
    assert fake._history_sigma_src_combo.state == "readonly"
    assert fake._history_sigma_win_entry.state == "normal"

    fake._history_sigma_src_var.set(
        gui_app.SIGMA_SOURCE_DISPLAY["implied"])
    BacktestApp._toggle_history_candidate_controls(fake)
    assert fake._history_sigma_win_entry.state == "disabled"

    fake._history_include_fixed_times_var.set(True)
    fake._history_include_band_var.set(False)
    BacktestApp._toggle_history_candidate_controls(fake)

    assert fake._history_fixed_times_entry.state == "normal"
    assert fake._history_band_candidate_entry.state == "disabled"
    assert fake._history_current_band_check.state == "disabled"
    assert fake._history_sigma_src_combo.state == "disabled"
    assert fake._history_sigma_win_entry.state == "disabled"


def test_ordinary_backtest_does_not_parse_hidden_history_candidates():
    fake = _fake_collect_state()
    fake._history_band_candidate_sigmas_var = _Var("not-a-number")
    fake._history_fixed_times_var = _Var("not-a-time")
    fake._history_sigma_src_var = _Var("invalid-hidden-value")
    fake._history_sigma_win_var = _Var("also-invalid")

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
    assert errors and errors[0][0] == "历史择优不可用"
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

    assert fake._history_btn.state == "normal"
    assert "真实历史来源已就绪" in fake._history_source_hint_var.get()


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
        "wind_bar_size": "15min",
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
    ):
        summary[normalized] = pd.Series([
            np.asarray(values, dtype=float) / denominator
            for values, denominator in zip(summary[raw], denominators)
        ], dtype=object)
    return summary


def test_history_chart_exposes_only_single_sample_and_multi_sample_modes():
    assert gui_app.HISTORY_CHART_MODE_DISPLAY == {
        "single": "单样本路径",
        "typical": "多样本中位路径",
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
    assert all(item["label"].startswith("样本 ")
               for item in model["window_options"])


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


def test_history_cross_contract_typical_chart_fails_closed_without_normalization():
    summary = _history_cross_contract_normalized_summary().drop(
        columns=["normalized_cumulative_net_pnl"])

    model = BacktestApp._history_multi_chart_model(
        summary, "week", ["固定间隔(1σ)"],
        mode="typical", metric="net",
    )

    assert model["state"] == "empty"
    assert "安全归一化" in model["message"]
    assert "单样本路径" in model["message"]


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


@pytest.mark.parametrize("mode", ["single", "typical"])
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
    assert labels[0] == "每日收盘（C2C基准）"
    assert labels[1].startswith("固定间隔(1σ)")
    assert labels[2].startswith("每日固定时刻")
    assert len(app._history_chart_ax.lines) >= 3
    assert "共同样本 2 个" in app._history_chart_hint_var.get()


def _history_period_view_model_fixture():
    """五周期展示样本显式携带 C2C 基准及配对比较口径。"""
    recommendations = pd.DataFrame([{
        "lookback": "week", "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band", "score": 8.0,
        "baseline_score": 10.0, "improvement_vs_c2c": 0.20,
        "window_win_rate_vs_c2c": 0.75,
        "paired_windows": 4, "baseline_windows": 4,
        "comparison_eligible": True,
        "rolling_windows": 4, "eligible_endpoints": 4,
        "skipped_endpoints": 0, "history_days_available": 5,
        "lookback_days": 5, "maturity_days": 22, "step_days": 5,
    }])

    def row(lookback, rank, strategy, strategy_type, score, baseline_score,
            improvement, paired, eligible, skipped, available, requested,
            complete, *, win_rate=0.0, comparison_eligible=True):
        return {
            "lookback": lookback, "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "score": score,
            "baseline_score": baseline_score,
            "improvement_vs_c2c": improvement,
            "window_win_rate_vs_c2c": win_rate,
            "paired_windows": paired, "baseline_windows": paired,
            "comparison_eligible": comparison_eligible,
            "rolling_windows": paired, "eligible_endpoints": eligible,
            "skipped_endpoints": skipped,
            "history_days_available": available,
            "lookback_days": requested, "complete_window": complete,
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
    assert rows[0]["status"] == "历史样本完整"
    assert rows[0]["strategy_label"] == "固定间隔(1σ)"
    assert rows[0]["improvement_vs_c2c"] == pytest.approx(0.20)
    assert rows[0]["gap_ratio"] == pytest.approx(0.20)
    assert rows[0]["selection_improvement_vs_c2c"] is None
    assert rows[0]["uses_window_equal_metric"] is False
    assert rows[0]["window_win_rate_vs_c2c"] == pytest.approx(0.75)
    assert (rows[0]["paired"], rows[0]["baseline_windows"]) == (4, 4)
    assert rows[0]["best_is_baseline"] is False

    assert rows[1]["status"] == "样本不足，仅诊断"
    assert rows[1]["strategy_label"] == "诊断：每日收盘（基准最优）"
    assert rows[1]["improvement_vs_c2c"] == pytest.approx(0.0)
    assert rows[1]["best_is_baseline"] is True
    assert rows[1]["formal"] is False
    assert (rows[1]["effective"], rows[1]["eligible"], rows[1]["skipped"]) == (
        2, 4, 2,
    )

    assert rows[2]["status"] == "无完整配对候选"
    assert rows[2]["strategy_label"] == "仅基准（无完整配对候选）"
    assert rows[2]["has_comparable_candidate"] is False
    assert rows[3]["period"] == "近半年"
    assert rows[3]["status"] == "无可评估样本"
    assert rows[4]["period"] == "近年"
    assert rows[4]["status"] == "无可评估样本"


def test_recommendation_view_model_only_returns_selected_period_subset():
    recommendations, ranking = _history_period_view_model_fixture()

    rows = BacktestApp._comparison_recommendation_rows(
        recommendations, ranking, lookbacks=("week", "quarter"))

    assert [row["lookback"] for row in rows] == ["week", "quarter"]
    assert [row["period"] for row in rows] == ["近周", "近季"]
    assert rows[0]["status"] == "历史样本完整"
    assert rows[1]["status"] == "无完整配对候选"


def test_product_pool_view_uses_window_equal_metric_and_contract_list():
    candidate = {
        "lookback": "week", "rank": 1,
        "strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
        "score": 110.0, "baseline_score": 100.0,
        "improvement_vs_c2c": -0.10,
        "selection_improvement_vs_c2c": 0.20,
        "selection_metric": "mean_bounded_window_advantage_vs_c2c",
        "window_win_rate_vs_c2c": 0.75,
        "paired_windows": 4, "baseline_windows": 4,
        "comparison_eligible": True, "recommendation_eligible": True,
        "complete_window": True, "rolling_windows": 4,
        "eligible_endpoints": 4, "selected_endpoints": 4,
        "planned_endpoints": 4, "skipped_endpoints": 0,
        "history_days_available": 5, "available_history_days": 27,
        "required_history_days": 27, "lookback_days": 5,
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
        "mean_bounded_window_advantage_vs_c2c")
    assert rows[0]["uses_window_equal_metric"] is True
    assert rows[0]["paired_contract_codes"] == (
        "P2601.DCE", "P2605.DCE")


def test_history_period_view_model_preserves_target_sampling_and_t_context():
    sampling = {
        "maturity_days": 243,
        "sampling_mode": "target_count",
        "step_days": np.nan,
        "target_endpoints": 12,
        "planned_endpoints": 5,
        "selected_endpoints": 5,
        "endpoint_spacing_min": 1,
        "endpoint_spacing_max": 1,
        "window_overlap_min_ratio": 242 / 243,
        "window_overlap_max_ratio": 242 / 243,
        "maturity_exceeds_lookback": True,
        "required_history_days": 248,
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
    assert week["sampling_mode"] == "target_count"
    assert np.isnan(week["step_days"])
    assert week["target_endpoints"] == 12
    assert week["planned_endpoints"] == 5
    assert week["selected_endpoints"] == 5
    assert week["endpoint_spacing_min"] == 1
    assert week["endpoint_spacing_max"] == 1
    assert week["window_overlap_min_ratio"] == pytest.approx(242 / 243)
    assert week["window_overlap_max_ratio"] == pytest.approx(242 / 243)
    assert week["maturity_exceeds_lookback"] is True
    assert week["required_history_days"] == 248
    assert week["available_history_days"] == 300
    assert week["history_complete"] is True


def test_zero_window_or_non_finite_history_result_is_not_a_diagnostic_leader():
    ranking = pd.DataFrame([{
        "lookback": "week", "rank": 1, "strategy": "A",
        "score": float("inf"), "rolling_windows": pd.NA,
        "eligible_endpoints": pd.NA, "skipped_endpoints": pd.NA,
        "complete_window": False,
    }])

    week = BacktestApp._comparison_recommendation_rows(None, ranking)[0]

    assert week["strategy"] == "—"
    assert week["status"] == "无可评估样本"


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
        "fixed_times 目标时刻 [11:30,15:00] 未逐交易日组完整匹配；"
        "第1个交易日组缺失 [11:30]。"
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
    assert "固定时刻策略没有形成任何可评估滚动样本" in message
    assert "缺失 [11:30]" in message
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
    assert "历史具体合约池没有形成任何可评估样本" in message
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
    fake._history_sigma_src_var = _Var(
        gui_app.SIGMA_SOURCE_DISPLAY["implied"])
    fake._history_sigma_win_var = _Var("20")
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
    history_sigma_source_before = fake._history_sigma_src_var.get()
    history_sigma_window_before = fake._history_sigma_win_var.get()
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
    assert fake._history_sigma_src_var.get() == history_sigma_source_before
    assert fake._history_sigma_win_var.get() == history_sigma_window_before
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


def test_history_worker_uses_period_target_budgets_for_every_maturity(
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
    assert [kwargs["target_endpoints"] for _maturity, kwargs in captured] == [
        gui_app.HISTORY_TARGET_ENDPOINTS,
        gui_app.HISTORY_TARGET_ENDPOINTS,
        gui_app.HISTORY_TARGET_ENDPOINTS,
    ]
    assert [kwargs["lookbacks"] for _maturity, kwargs in captured] == [
        gui_app.LOOKBACK_DAYS,
        gui_app.LOOKBACK_DAYS,
        gui_app.LOOKBACK_DAYS,
    ]
    assert all("step_days" not in kwargs for _maturity, kwargs in captured)


def test_csv_history_worker_passes_only_selected_periods_and_budgets(
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
    selected_targets = {
        "week": gui_app.HISTORY_TARGET_ENDPOINTS["week"],
        "quarter": gui_app.HISTORY_TARGET_ENDPOINTS["quarter"],
    }

    def recommend(option, loaded, cases, kwargs, **call_kwargs):
        captured.append((option, loaded, cases, kwargs, call_kwargs))
        return recommendations, ranking, windows

    monkeypatch.setattr(gui_app, "recommend_by_rolling_history", recommend)
    fake, state, delivered, failed = _history_worker_fixture(
        "csv", lambda _state, _bt: history)
    state.update({
        "history_lookbacks": selected_lookbacks,
        "history_target_endpoints": selected_targets,
    })

    BacktestApp._history_recommendation_worker(fake, state)

    assert failed == []
    assert len(delivered) == 1
    assert len(captured) == 1
    assert captured[0][4]["lookbacks"] == selected_lookbacks
    assert captured[0][4]["target_endpoints"] == selected_targets


def test_history_worker_routes_product_code_without_building_continuous_backtest(
        monkeypatch):
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
    selected_lookbacks = {
        "week": gui_app.LOOKBACK_DAYS["week"],
        "quarter": gui_app.LOOKBACK_DAYS["quarter"],
    }
    selected_targets = {
        "week": gui_app.HISTORY_TARGET_ENDPOINTS["week"],
        "quarter": gui_app.HISTORY_TARGET_ENDPOINTS["quarter"],
    }
    state.update({
        "history_lookbacks": selected_lookbacks,
        "history_target_endpoints": selected_targets,
    })
    fake._build_backtest = lambda _state: pytest.fail(
        "P.DCE 历史择优不应先下载连续合约行情")
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
    assert captured[0][4]["target_endpoints"] == selected_targets
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
            "wind_end": "2026-02-01", "wind_bar_size": "60min",
        }),
    ],
)
def test_real_sources_always_delegate_bar_count_to_backend(
        monkeypatch, source, factory_name, source_fields):
    captured = {}

    def fake_factory(option, *args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        gui_app.HedgeBacktest, factory_name, staticmethod(fake_factory))
    state = {
        "cfg": {"build": lambda subtype, params: SimpleNamespace()},
        "cls_name": "test", "subtype": "test", "params": {},
        "source": source, "tc_rate": 0.0, "position": 1,
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
    form._wind_bar_size_var.set(WIND_AUTO_BAR_SIZE)

    state = BacktestApp._collect_gui_state(form)
    app = SimpleNamespace(
        _validate_fixed_time_source_state=lambda _state: None)
    backtest = BacktestApp._build_backtest(app, state)

    assert state["wind_bar_size_requested"] == WIND_AUTO_BAR_SIZE
    assert state["wind_bar_size"] == "15min"
    assert captured["args"] == (
        "510050.SH", "2025-01-02", "2025-02-28")
    assert captured["kwargs"]["bar_size"] == "15"
    assert backtest._gui_meta["wind_start"] == "2025-01-02"
    assert backtest._gui_meta["wind_end"] == "2025-02-28"
    assert backtest._gui_meta["wind_bar_size"] == "15min"
    assert BacktestApp._snapshot_source_label(state) == (
        "Wind · 510050.SH · 2025-01-02 至 2025-02-28 · 15min"
    )


def test_comparison_kwargs_do_not_carry_legacy_hedge_frequency():
    kwargs = BacktestApp._comparison_backtest_kwargs(SimpleNamespace(
        tc_rate=0.0, position=1, quantity=1.0, multiplier=0.0,
        steps_per_day=4, slippage_bps=0.0,
        force_day_close_hedge=True,
    ))
    assert "hedge_freq" not in kwargs
    assert kwargs["force_day_close_hedge"] is True


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
