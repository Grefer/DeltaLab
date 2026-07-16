from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import gui_app
from gui_app import (
    BacktestApp,
    DEFAULT_BAND_CANDIDATE_SIGMAS,
    MAX_BAND_CANDIDATES,
    STRATEGY_DISPLAY,
)
from pricing import (
    CloseToCloseStrategy,
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
            "history_include_band": False,
            # 未勾选项即使值不可用也不得参与或生成跳过说明。
            "fixed_times": "not-a-time",
            "band_candidate_sigmas": (0.5, 1.0),
            "params": {"s0": 100.0, "sigma": 0.2},
        },
        SimpleNamespace(),
    )

    assert [case.name for case in cases] == ["每日收盘"]
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
        _slip_var=_Var("0"),
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


def test_history_collects_its_own_candidates_times_and_sigma_configuration():
    fake = _fake_history_collect_state()
    # 普通回测区故意放入不同值；历史状态必须只采用 `_history_*` 输入。
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


def test_history_does_not_read_disabled_candidate_controls():
    class _ForbiddenVar:
        def get(self):
            raise AssertionError("未勾选的历史候选参数不应被读取")

    fake = _fake_history_collect_state()
    fake._history_include_fixed_times_var.set(False)
    fake._history_include_band_var.set(False)
    # 左侧即便停留在另一个策略且其专属输入无效，历史页的“仅每日收盘”
    # 候选也应按公共环境收集，不读取该隐藏输入。
    fake._strategy_var.set(STRATEGY_DISPLAY["fixed_times"])
    fake._fixed_times_var = _ForbiddenVar()
    fake._history_fixed_times_var = _ForbiddenVar()
    fake._history_band_candidate_sigmas_var = _ForbiddenVar()
    fake._history_include_current_band_var = _ForbiddenVar()
    fake._history_sigma_src_var = _ForbiddenVar()
    fake._history_sigma_win_var = _ForbiddenVar()

    state = BacktestApp._collect_history_state(fake)

    assert state["history_include_close"] is True
    assert state["strategy_name"] == "close_to_close"
    assert state["history_include_fixed_times"] is False
    assert state["history_include_band"] is False
    assert state["fixed_times"] == ""
    assert state["band_candidate_sigmas"] == ()


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


def test_history_requires_at_least_one_candidate_strategy():
    fake = _fake_history_collect_state()
    fake._history_include_close_var.set(False)
    fake._history_include_fixed_times_var.set(False)
    fake._history_include_band_var.set(False)

    with pytest.raises(ValueError, match="至少要选择一种"):
        BacktestApp._collect_history_state(fake)


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
    state = {"source": "csv"}

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


def test_selected_strategy_delta_uses_current_best_as_denominator():
    best = {"score": 8.0, "total_tc": 100.0}
    selected = {"score": 10.0, "total_tc": 80.0}

    score = BacktestApp._comparison_relative_delta(selected, best, "score")
    cost = BacktestApp._comparison_relative_delta(selected, best, "total_tc")

    assert score == {"value": 10.0, "delta": 2.0, "ratio": 0.25}
    assert cost == {"value": 80.0, "delta": -20.0, "ratio": -0.20}


def test_equal_non_best_metric_is_not_labeled_as_baseline():
    best = {"strategy": "A", "score": 8.0}
    selected = {"strategy": "B", "score": 8.0}

    assert BacktestApp._comparison_delta_display(
        selected, best, "score") == "8.00（与最优相同）"


def test_recommendation_view_model_separates_formal_from_diagnostic_leader():
    recommendations = pd.DataFrame([{
        "lookback": "week", "strategy": "A", "score": 8.0,
        "rolling_windows": 4, "eligible_endpoints": 4,
        "skipped_endpoints": 0, "history_days_available": 5,
        "lookback_days": 5, "maturity_days": 22, "step_days": 5,
    }])
    ranking = pd.DataFrame([
        {"lookback": "week", "rank": 1, "strategy": "A", "score": 8.0,
         "rolling_windows": 4, "eligible_endpoints": 4,
         "skipped_endpoints": 0, "history_days_available": 5,
         "lookback_days": 5, "complete_window": True},
        {"lookback": "week", "rank": 2, "strategy": "B", "score": 10.0,
         "rolling_windows": 4, "eligible_endpoints": 4,
         "skipped_endpoints": 0, "history_days_available": 5,
         "lookback_days": 5, "complete_window": True},
        {"lookback": "month", "rank": 1, "strategy": "B", "score": 9.0,
         "rolling_windows": 2, "eligible_endpoints": 4,
         "skipped_endpoints": 2, "history_days_available": 12,
         "lookback_days": 20, "complete_window": False},
        {"lookback": "month", "rank": 2, "strategy": "A", "score": 12.0,
         "rolling_windows": 2, "eligible_endpoints": 4,
         "skipped_endpoints": 2, "history_days_available": 12,
         "lookback_days": 20, "complete_window": False},
    ])

    rows = BacktestApp._comparison_recommendation_rows(
        recommendations, ranking)

    assert [row["lookback"] for row in rows] == [
        "week", "month", "quarter", "year",
    ]
    assert rows[0]["status"] == "正式推荐"
    assert rows[0]["strategy_label"] == "A"
    assert rows[0]["gap_ratio"] == pytest.approx(0.20)
    assert rows[1]["status"] == "样本不足（诊断）"
    assert rows[1]["strategy_label"] == "诊断领先：B"
    assert (rows[1]["effective"], rows[1]["eligible"], rows[1]["skipped"]) == (
        2, 4, 2,
    )
    assert rows[2]["status"] == "无可评估窗口"


def test_zero_window_or_non_finite_history_result_is_not_a_diagnostic_leader():
    ranking = pd.DataFrame([{
        "lookback": "week", "rank": 1, "strategy": "A",
        "score": float("inf"), "rolling_windows": pd.NA,
        "eligible_endpoints": pd.NA, "skipped_endpoints": pd.NA,
        "complete_window": False,
    }])

    week = BacktestApp._comparison_recommendation_rows(None, ranking)[0]

    assert week["strategy"] == "—"
    assert week["status"] == "无可评估窗口"


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

    BacktestApp._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, [],
        {"week": {"window_1": {}}}, "CSV · prices.csv", state)

    assert len(rendered) == 1
    assert finished == [True]
    assert fake._saved_backtests is saved
    assert fake._saved_comparison_selection is selection
    assert fake._latest_history_state["source"] == "csv"
    assert fake._latest_history_source_label == "CSV · prices.csv"


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
        "steps_per_day": 240, "strategy_name": "close_to_close",
        **source_fields,
    }
    fake_app = SimpleNamespace(
        _validate_fixed_time_source_state=lambda _state: None,
    )

    BacktestApp._build_backtest(fake_app, state)
    assert captured["steps_per_day"] is None
    assert isinstance(captured["strategy"], CloseToCloseStrategy)


def test_comparison_kwargs_do_not_carry_legacy_hedge_frequency():
    kwargs = BacktestApp._comparison_backtest_kwargs(SimpleNamespace(
        tc_rate=0.0, position=1, quantity=1.0, multiplier=0.0,
        steps_per_day=4, slippage_bps=0.0,
    ))
    assert "hedge_freq" not in kwargs


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
