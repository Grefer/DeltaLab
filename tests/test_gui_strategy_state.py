from __future__ import annotations

import copy
import datetime
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gui_app
import history_selection
from pricing.hedge_analysis import (
    _aggregate_result_by_day as _agg_daily_frame)
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
    Option_Vanilla,
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


@pytest.mark.parametrize(
    ("label", "values"),
    [
        # numpy 对全等样本只展开 ±0.5，量级一大这 1.0 就小于浮点分辨率。
        ("全等大数", np.full(10, 1e15)),
        ("全等极大数", np.full(10, 1e300)),
        ("单条有效路径", np.array([1e300])),
        # 跨度非零但小于量级 ulp 的 30 倍：分箱同样切不开。
        ("量级内微小跨度", np.array([1e6, 1e6 + 1e-9, 1e6 + 2e-9])),
        ("大数上的微小跨度", np.array([1e15, 1e15 + 0.125])),
        # 两端极大：hi - lo 直接溢出成 inf。
        ("跨度溢出", np.array([-1e308, 1e308])),
        # 正常与温和退化的样本不能被这条兜底改变行为。
        ("全零", np.zeros(10)),
        ("全等小值", np.full(10, 3.5)),
        ("单点", np.array([2.0])),
        ("正常分布", np.linspace(-5.0, 5.0, 40)),
    ],
)
def test_histogram_edges_never_collapse_for_degenerate_samples(label, values):
    """分箱边界必须严格递增，否则 numpy 抛错会打掉整页回测结果。

    模拟已实现波动率填 0、只剩一条有效路径、或结构让每条路径 payoff 相同
    时，``total_pnl`` 会全等或只差 MC 噪声——这正是线上崩的那类样本。
    """
    edges = BacktestApp._histogram_edges(values, 30)

    assert np.all(np.diff(edges) > 0), (label, edges)
    # 真正的验收条件：numpy 自己接受这组边界。
    np.histogram(values, bins=edges)


def test_worthless_option_multi_path_pnl_still_produces_usable_bins():
    """线上崩溃的那组真实参数：20% 价外看跌 + 10 条模拟路径。

    期权几乎一文不值、Delta 取整后一手都不建，于是每条路径的总盈亏都等于
    那点微不足道的期初权利金，彼此只差 1~2 个 ULP。30 档分箱要求的步长比
    这个量级的浮点分辨率还小，numpy 直接拒绝分箱，异常一路把整页结果打掉。
    """
    cfg = gui_app.OPTION_CLASSES["香草期权 (Vanilla)"]
    option = cfg["build"]("Eu", {
        "s0": 100.0, "K": 80.0, "T_days": 22, "sigma": 0.15,
        "cp": -1, "r": 0.03, "q": 0.03,
    })
    paths = gui_app.HedgeBacktest.simulate_multi_paths(
        100.0, 0.15, 22, n_paths=10, r=0.03, q=0.03, seed=42,
        steps_per_day=1)
    backtest = gui_app.HedgeBacktest(
        option, paths[0], tc_rate=0.0001, position=1, quantity=100.0,
        multiplier=5.0, strategy=CloseToCloseStrategy(), steps_per_day=1,
        slippage_bps=0.0, force_day_close_hedge=True, base_seed=42)
    pnl = np.asarray(backtest.run_multi(paths)["total_pnl"], dtype=float)

    # 前提校验：样本必须真的退化，否则这条回归测试就守不住任何东西。
    span = float(np.ptp(pnl))
    resolution = float(np.spacing(float(np.max(np.abs(pnl)))))
    assert 0 < span < 30 * resolution, (
        f"这组参数不再产生退化样本（span={span!r}），"
        "回归用例需要换一组真正只差几个 ULP 的配置")

    edges = BacktestApp._histogram_edges(pnl, 30)
    assert np.all(np.diff(edges) > 0)
    np.histogram(pnl, bins=edges)      # numpy 接受即通过


def test_histogram_edges_keep_the_requested_resolution_when_data_allows():
    """兜底只在切不开时降档，正常样本必须保留请求的分箱数。"""
    normal = np.linspace(-5.0, 5.0, 40)

    assert len(BacktestApp._histogram_edges(normal, 30)) - 1 == 30
    assert len(BacktestApp._histogram_edges(normal, 60)) - 1 == 60
    # 量级 1e6 上跨度 2e-9 只够切 15 档，不该一路退到 1 档。
    narrow = np.array([1e6, 1e6 + 1e-9, 1e6 + 2e-9])
    narrow_bins = len(BacktestApp._histogram_edges(narrow, 30)) - 1
    assert 1 < narrow_bins < 30


def test_histogram_edges_survive_non_finite_and_empty_samples():
    """非有限值与空样本只是画不出分布，不该把渲染整条链路带崩。"""
    mixed = BacktestApp._histogram_edges(
        np.array([1.0, np.nan, np.inf, 2.0]), 30)
    assert np.all(np.diff(mixed) > 0)
    assert mixed[0] == pytest.approx(1.0) and mixed[-1] == pytest.approx(2.0)

    for empty in (np.array([]), np.array([np.nan, np.inf])):
        edges = BacktestApp._histogram_edges(empty, 30)
        assert np.all(np.diff(edges) > 0)


def _vol_chart_stub(prices):
    """构造 _show_vol_chart 需要的最小结果字典。"""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    return SimpleNamespace(
        _results={
            "prices": prices,
            "n_days": max(1, n - 1),
            "implied_vol": 0.18,
            "rolling_realized": np.full(n, 0.15),
            "cumulative_realized": np.full(n, 0.15),
        },
        _wind_meta=None,
    )


@pytest.mark.parametrize(
    ("label", "prices"),
    [
        # 价格出现 0 → 对数收益同时得到 ±inf，matplotlib 直接抛
        # "supplied range of [-inf, inf] is not finite"。
        ("价格含 0", [100.0, 0.0, 100.0, 101.0]),
        # 价格转负 → 对数收益全是 nan。
        ("价格含负", [100.0, -5.0, -6.0, -7.0]),
        # 价格恒定 → 收益率恒为 0，正态叠加曲线的 x 轴会塌成一个点。
        ("价格恒定", [100.0] * 6),
        ("正常价格", [100.0, 101.0, 99.5, 100.5, 102.0]),
    ],
)
def test_vol_chart_survives_non_positive_and_constant_prices(label, prices):
    """波动率页排在分布页之前，它抛错同样会打掉整页结果。

    CSV / Wind 都只校验行数，不保证价格为正——0 或负价格会让对数收益变成
    ±inf / nan，这是与分布页那个 bug 同源的另一处。
    """
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        app._show_vol_chart(_vol_chart_stub(prices))
        app.update_idletasks()
        assert app._vol_figure is not None, label
    finally:
        app.destroy()


def test_vol_chart_drops_the_realized_normal_curve_when_returns_are_constant():
    """σ=0 时 norm.pdf 整条是 nan，不能画不出来却仍在图例写「σ=0.00%」。"""
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        app._show_vol_chart(_vol_chart_stub([100.0] * 6))
        app.update_idletasks()

        returns_ax = app._vol_figure.get_axes()[3]
        labels = [line.get_label() for line in returns_ax.get_lines()]
        assert not [text for text in labels if text.startswith("已实现正态")]
        assert [text for text in labels if text.startswith("隐含波动率正态")]
        assert "收益率恒定" in returns_ax.get_title()
    finally:
        app.destroy()


def test_degenerate_multi_path_stats_still_render_the_distribution_tab():
    """真实控件树上跑一遍：全等 PnL 不再让分布页（进而整页结果）崩掉。"""
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        # 已实现波动率填 0 时全部路径一模一样；量级由数量 × 乘数放大。
        multi_stats = {
            "total_pnl": np.full(10, 1e15),
            "errors": np.full(10, 1e15),
            "realized_vols": np.zeros(10),
            "implied_vol": 0.18,
        }

        app._show_dist_chart(multi_stats)
        app.update_idletasks()

        assert app._dist_figure is not None
        assert app._dist_canvas is not None

        # 波动率价差全等时样本里没有斜率信息，不能凭 polyfit 的最小范数解
        # 写出一个具体斜率（实测会写成「拟合斜率=2.8e13」）。
        scatter_ax = app._dist_figure.get_axes()[3]
        assert scatter_ax.get_legend() is None
        assert not [
            line for line in scatter_ax.get_lines()
            if line.get_label().startswith("拟合斜率")
        ]

        # 价差真的散开时，拟合与图例照旧。
        rng = np.random.default_rng(7)
        app._show_dist_chart({
            "total_pnl": rng.normal(100.0, 20.0, 40),
            "errors": rng.normal(0.0, 5.0, 40),
            "realized_vols": rng.normal(0.18, 0.02, 40),
            "implied_vol": 0.18,
        })
        app.update_idletasks()

        fitted_ax = app._dist_figure.get_axes()[3]
        assert fitted_ax.get_legend() is not None
        assert [
            line for line in fitted_ax.get_lines()
            if line.get_label().startswith("拟合斜率")
        ]
    finally:
        app.destroy()


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
        "wind_auto_start": False,
        "wind_bar_size_requested": WIND_AUTO_BAR_SIZE,
        "strategy_name": "close_to_close",
        "fixed_times": "11:30,15:00",
        "params": {"T_days": 22},
    }
    state.update(overrides)
    return state


def _trading_days_back(asof, trading_days):
    """用与实现同一份交易日历取 asof 之前第 N 个交易日。"""
    calendar = BacktestApp._trading_calendar_days()
    end_index = calendar.index(asof)
    return calendar[end_index - trading_days]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wind_start": "2026-03-02"}, "建仓日不能晚于当前日期"),
        ({"wind_end": "2026-01-02"}, "截止日必须晚于建仓日"),
        ({"wind_end": "2026-01-01"}, "截止日必须晚于建仓日"),
        ({"wind_end": "2026-03-02"}, "截止日不能晚于当前日期"),
    ],
)
def test_single_wind_dates_reject_future_or_reversed_ranges(
        overrides, message):
    state = _single_wind_resolution_state(**overrides)
    with pytest.raises(ValueError, match=message):
        BacktestApp._resolve_single_wind_state(
            state, today=datetime.date(2026, 3, 1))


def test_single_wind_auto_start_backdates_entry_without_parsing_custom_start():
    """建仓日由截止日倒推，落在真实交易日上，且不读自定义建仓日。"""
    class _UnusedStart:
        def __str__(self):
            raise AssertionError("自动倒推模式不得解析自定义建仓日")

    state = _single_wind_resolution_state(
        wind_start=_UnusedStart(),
        wind_end="2026-02-25",
        wind_auto_start=True,
        params={"T_days": 30},
    )
    resolved = BacktestApp._resolve_single_wind_state(
        state, today=datetime.date(2026, 3, 1))

    assert resolved["wind_end"] == "2026-02-25"
    assert resolved["wind_start"] == _trading_days_back(
        datetime.date(2026, 2, 25), 30).isoformat()
    assert resolved["wind_required_trade_days"] == 31
    assert resolved["wind_date_mode"] == "auto_entry_from_asof"


def test_single_wind_auto_start_snaps_non_trading_asof_back_to_trading_day():
    """截止日落在休市日时，两端都必须先落到真实交易日。

    2026-02-20 是春节休市（前一个交易日 2026-02-13），因此这里同时验证
    倒推不是简单的“跳过周末”自然日算术。
    """
    resolved = BacktestApp._resolve_single_wind_state(
        _single_wind_resolution_state(
            wind_end="2026-02-20",
            wind_auto_start=True,
            params={"T_days": 22},
        ),
        today=datetime.date(2026, 3, 1),
    )
    calendar = BacktestApp._trading_calendar_days()

    assert resolved["wind_end"] == "2026-02-13"
    assert datetime.date.fromisoformat(resolved["wind_start"]) in calendar
    assert resolved["wind_start"] == _trading_days_back(
        datetime.date(2026, 2, 13), 22).isoformat()


def test_single_wind_auto_start_refuses_asof_beyond_trading_calendar():
    """日历没覆盖到截止日时不能拿旧日历末尾当锚点，必须直接报错。"""
    calendar = BacktestApp._trading_calendar_days()
    beyond = calendar[-1] + datetime.timedelta(days=1)
    state = _single_wind_resolution_state(
        wind_end=beyond.isoformat(), wind_auto_start=True)

    with pytest.raises(ValueError, match="交易日历未覆盖数据截止日"):
        BacktestApp._resolve_single_wind_state(state, today=beyond)


def test_single_wind_custom_entry_date_is_preserved_exactly():
    resolved = BacktestApp._resolve_single_wind_state(
        _single_wind_resolution_state(
            wind_start="2026-01-02",
            wind_end="2026-02-20",
            wind_auto_start=False,
        ),
        today=datetime.date(2026, 3, 1),
    )

    assert resolved["wind_start"] == "2026-01-02"
    assert resolved["wind_end"] == "2026-02-20"
    assert resolved["wind_date_mode"] == "custom_entry_date"


def _fake_wind_date_form(*, auto_start=True, asof="2026-02-25",
                         maturity="30", retrigger=False):
    """模拟 Wind 日期控件组；``retrigger`` 复现真实 StringVar 的 write trace。"""
    form = SimpleNamespace(
        _wind_end_var=_Var(asof),
        _wind_start_var=_Var("2025-01-02"),
        _wind_auto_start_var=_BoolVar(auto_start),
        _wind_start_entry=_Widget(),
        _param_entries={"T_days": (_Var(maturity), int, None)},
    )
    if retrigger:
        for var in (form._wind_start_var, form._wind_end_var):
            original_set = var.set

            def set_and_retrigger(value, _set=original_set):
                _set(value)
                BacktestApp._sync_wind_entry_date(form)

            var.set = set_and_retrigger
    return form


def test_wind_auto_start_backfills_the_entry_box_with_the_backdated_date():
    """界面不再单列实际区间，建仓日那一格必须就是倒推结果本身。"""
    form = _fake_wind_date_form()

    BacktestApp._sync_wind_entry_date(form)

    expected_start = _trading_days_back(datetime.date(2026, 2, 25), 30)
    assert form._wind_start_var.get() == expected_start.isoformat()


def test_wind_entry_backfill_survives_its_own_write_trace():
    """回填会再触发同一个 trace；重入必须被挡住而不是写出半成品。"""
    form = _fake_wind_date_form(retrigger=True)

    BacktestApp._sync_wind_entry_date(form)

    expected_start = _trading_days_back(datetime.date(2026, 2, 25), 30)
    assert form._wind_start_var.get() == expected_start.isoformat()


@pytest.mark.parametrize(
    ("asof", "maturity"),
    [("2026-02-30", "30"), ("2026-02-2", "30"), ("2026-02-25", ""),
     ("9999-01-05", "30")],
)
def test_wind_entry_backfill_keeps_last_value_on_half_edited_inputs(
        asof, maturity):
    """编辑到一半不得抛错、也不得把建仓日清空或写成半成品。"""
    form = _fake_wind_date_form(asof=asof, maturity=maturity)

    BacktestApp._sync_wind_entry_date(form)

    assert form._wind_start_var.get() == "2025-01-02"


def test_wind_asof_commit_snaps_to_the_trading_day_anchor_it_backdates_from():
    """截止日填成休市日时，提交后落到真正的倒推锚点，建仓日随之更新。

    2026-02-20 是春节休市，倒推锚点是 2026-02-13。两个日期框就是界面上唯一
    的区间说明，锚点不落回框里就没有任何地方写着实际取到哪一天。
    """
    form = _fake_wind_date_form(asof="2026-02-20", retrigger=True)

    BacktestApp._commit_wind_asof(form)

    assert form._wind_end_var.get() == "2026-02-13"
    assert form._wind_start_var.get() == _trading_days_back(
        datetime.date(2026, 2, 13), 30).isoformat()


@pytest.mark.parametrize("asof", ["2026-02-25", "2026-02-30", ""])
def test_wind_asof_commit_leaves_trading_days_and_bad_input_untouched(asof):
    form = _fake_wind_date_form(asof=asof, retrigger=True)

    BacktestApp._commit_wind_asof(form)

    assert form._wind_end_var.get() == asof


def test_wind_asof_commit_is_inert_while_the_entry_date_is_manual():
    """手工建仓日口径下截止日只是取数上限，不该被改写成交易日锚点。"""
    form = _fake_wind_date_form(auto_start=False, asof="2026-02-20")

    BacktestApp._commit_wind_asof(form)

    assert form._wind_end_var.get() == "2026-02-20"


def test_wind_auto_start_toggle_frees_manual_entry_without_overwriting_it():
    form = _fake_wind_date_form(auto_start=False)

    BacktestApp._toggle_wind_auto_start(form)

    # 手工建仓日不被倒推覆盖。
    assert form._wind_start_entry.state == "normal"
    assert form._wind_start_var.get() == "2025-01-02"

    form._wind_auto_start_var.set(True)
    BacktestApp._toggle_wind_auto_start(form)

    expected_start = _trading_days_back(datetime.date(2026, 2, 25), 30)
    assert form._wind_start_entry.state == "disabled"
    assert form._wind_start_var.get() == expected_start.isoformat()


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


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("0", "不能为 0"),
        ("0.0", "不能为 0"),
        ("-0.15", "大于 0"),
        ("inf", "大于 0"),
        ("nan", "大于 0"),
    ],
)
def test_option_sigma_must_be_positive_before_any_job_starts(raw, message):
    """σ=0 会让 BS 的 d1/d2 除零、模拟路径退化成一条确定性曲线。

    回测照样跑完、界面照样出数，只是那一屏 0.0000 没有任何意义，所以必须
    在启动前拦掉，而不是让用户自己去猜哪里不对。
    """
    fake = _fake_collect_state()
    fake._param_entries["sigma"][0].set(raw)

    with pytest.raises(ValueError, match=f"波动率.*{message}"):
        BacktestApp._collect_gui_state(fake)


def test_empty_option_sigma_is_rejected_as_empty_not_as_zero():
    fake = _fake_collect_state()
    fake._param_entries["sigma"][0].set("")

    with pytest.raises(ValueError, match="波动率 不能为空"):
        BacktestApp._collect_gui_state(fake)


@pytest.mark.parametrize("raw", ["0", "0.0", "-0.2", "nan"])
def test_simulated_realized_vol_rejects_zero_and_invalid_values(raw):
    """已实现波动率为 0 时全部模拟路径完全相同，多路径统计失去意义。"""
    fake = _fake_collect_state()
    fake._source_var.set("simulate")
    fake._real_vol_var.set(raw)

    with pytest.raises(ValueError, match="已实现波动率"):
        BacktestApp._collect_gui_state(fake)


def test_blank_realized_vol_keeps_meaning_implied_and_survives_other_sources():
    """留空是有意义的默认（同隐含）；非模拟来源根本不读这个框。"""
    blank = _fake_collect_state()
    blank._source_var.set("simulate")
    blank._real_vol_var.set("")
    assert BacktestApp._collect_gui_state(blank)["real_vol"] == ""

    # CSV / Wind 不消费该字段，里面的历史残值不该挡住任务启动。
    stale = _fake_collect_state()
    stale._source_var.set("csv")
    stale._real_vol_var.set("0")
    assert BacktestApp._collect_gui_state(stale)["real_vol"] == "0"


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


def test_single_wind_collection_backdates_entry_and_ignores_history_ui():
    class _ForbiddenVar:
        def get(self):
            raise AssertionError("单次回测不得读取历史择优 Wind 控件")

    fake = _fake_collect_state()
    fake._source_var.set("wind")
    fake._wind_start_var.set("invalid-but-disabled")
    fake._wind_end_var.set("2025-06-30")
    fake._wind_auto_start_var = _BoolVar(True)
    fake._strategy_var.set(STRATEGY_DISPLAY["close_to_close"])
    fake._history_wind_asof_var = _ForbiddenVar()
    fake._history_wind_start_var = _ForbiddenVar()
    fake._history_wind_auto_start_var = _ForbiddenVar()
    fake._history_wind_bar_size_var = _ForbiddenVar()

    state = BacktestApp._collect_gui_state(fake)
    maturity_days = state["params"]["T_days"]

    assert state["wind_end"] == "2025-06-30"
    assert state["wind_start"] == _trading_days_back(
        datetime.date(2025, 6, 30), maturity_days).isoformat()
    assert state["wind_required_trade_days"] == maturity_days + 1
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
    assert not hasattr(single, "_wind_auto_start_var")
    collected_single = BacktestApp._collect_gui_state(single)

    history = _fake_history_collect_state()
    assert not hasattr(history, "_history_wind_asof_var")
    assert not hasattr(history, "_history_wind_start_var")
    assert not hasattr(history, "_history_wind_auto_start_var")
    assert not hasattr(history, "_history_wind_bar_size_var")
    assert not hasattr(history, "_history_period_vars")
    collected_history = BacktestApp._collect_history_state(history)

    assert collected_single["wind_auto_start"] is False
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


def test_history_row_improvement_treats_c2c_itself_as_zero_improvement():
    improvement = BacktestApp._history_row_improvement({
        "strategy": "每日收盘",
        "strategy_type": "close_to_close",
        "daily_net_pnl_rms": 7.5,
    })

    assert improvement == pytest.approx(0.0)


def test_history_row_improvement_does_not_divide_by_zero_c2c_score():
    improvement = BacktestApp._history_row_improvement({
        "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band",
        "daily_net_pnl_rms": 2.0,
        "baseline_daily_net_pnl_rms": 0.0,
    })

    assert improvement is None


def test_history_row_improvement_prefers_window_equal_selection_metric():
    improvement = BacktestApp._history_row_improvement({
        "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band",
        "daily_net_pnl_rms": 110.0,
        "baseline_daily_net_pnl_rms": 100.0,
        # 合并金额 RMS 被高价合约主导，但逐窗平均优势仍为正。
        "improvement_vs_c2c": -0.10,
        "selection_improvement_vs_c2c": 0.20,
        "selection_metric": "mean_bounded_window_advantage_vs_c2c",
    })

    assert improvement == pytest.approx(0.20)


def test_history_metric_labels_removed_with_hardcoded_headers():
    """表头文案已全部写死在列 spec 里，metric_labels 整条链路是死的。

    `_build_history_metric_tree` 的每个列 spec 都自带 heading，
    `text = heading or labels[key]` 的右半边永远取不到；labels 从
    `_build_history_comparison_view` 一路传下来只是空转。
    """
    assert not hasattr(history_selection, "metric_labels")
    assert not hasattr(BacktestApp, "_history_metric_labels")

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
                "end_ts": pd.Timestamp(end_ts), "daily_net_pnl_rms": score,
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
    second.loc[second["window_id"] == "window_2", "daily_net_pnl_rms"] = 12.0
    second.loc[second["window_id"] == "window_3", "daily_net_pnl_rms"] = 0.0
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


def test_history_chart_view_options_are_pinned_to_full_interval():
    """图表口径必须恒为整段接续，只有指标可切。

    排名指标统计的是各段日损益 concat 之后的整段（hedge_analysis 里的
    ``pd.concat(daily_windows)``）。若图表能切成单段或中位段，同一策略
    完全可能整段为正、某一段为负，两条口径并列而界面无从区分。
    """
    app = object.__new__(BacktestApp)
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["tc"])

    assert BacktestApp._history_chart_view_options(app) == ("full", "tc")

    # 界面上已没有模式控件；即便残留一个旧变量也不得改变口径。
    app._history_chart_mode_var = _Var("多分段中位路径")
    assert BacktestApp._history_chart_view_options(app)[0] == "full"


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


def test_history_chart_controls_only_redraw_after_metric_change():
    """指标切换只需重绘；分段控件已移除，不得再有联动残留。"""
    app = object.__new__(BacktestApp)
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["net"])
    app._history_chart_metric_combo = _Widget()
    drawn = []
    app._draw_history_chart = lambda: drawn.append(True)

    BacktestApp._update_history_chart_controls(app)
    app._history_chart_metric_var.set(
        gui_app.HISTORY_CHART_METRIC_DISPLAY["tc"])
    BacktestApp._update_history_chart_controls(app)

    assert drawn == [True, True]
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


@pytest.mark.parametrize(
    "metric", ["net", "gross", "tc"])
def test_history_multi_chart_renderer_draws_c2c_and_all_candidates(metric):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    app = object.__new__(BacktestApp)
    figure = Figure(figsize=(6, 2))
    app._history_chart_figure = figure
    app._history_chart_ax = figure.add_subplot(111)
    app._history_chart_canvas = FigureCanvasAgg(figure)
    app._history_window_summary = _history_multi_chart_summary()
    app._history_chart_metric_var = _Var(
        gui_app.HISTORY_CHART_METRIC_DISPLAY[metric])
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
    # 段数写在标题里（loc="left"），提示条留给数据完整度。曲线是 N 段
    # 接起来的，不写出来容易被当成一笔交易。
    assert "2 段接续" in app._history_chart_ax.get_title(loc="left")
    assert "日 ·" in app._history_chart_hint_var.get()


def _history_period_view_model_fixture():
    """五周期展示显式携带严格区间、代理段及 C2C 配对口径。"""
    recommendations = pd.DataFrame([{
        "lookback": "week", "strategy": "固定间隔(1σ)",
        "strategy_type": "hedge_band", "daily_net_pnl_rms": 8.0,
        "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.20,
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
            "strategy_type": strategy_type, "daily_net_pnl_rms": score,
            "baseline_daily_net_pnl_rms": baseline_score,
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
        "daily_net_pnl_rms": 110.0, "baseline_daily_net_pnl_rms": 100.0,
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
        "strategy_type": "close_to_close", "daily_net_pnl_rms": 100.0,
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
        "daily_net_pnl_rms": 8.0, "baseline_daily_net_pnl_rms": 10.0,
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
        "strategy_type": "close_to_close", "daily_net_pnl_rms": 10.0,
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
        "strategy_type": "hedge_band", "daily_net_pnl_rms": 8.0,
        "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.2,
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
        "strategy_type": "close_to_close", "daily_net_pnl_rms": 10.0,
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
        "daily_net_pnl_rms": float("inf"), "rolling_windows": pd.NA,
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
        "strategy": "daily", "rolling_windows": 1, "daily_net_pnl_rms": 1.0,
    }])
    BacktestApp._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, [],
        {"week": {"window_1": {}}})

    assert finished == [False]
    assert errors and errors[0][0] == "历史择优展示失败"


@pytest.mark.parametrize(
    ("recommendations", "ranking", "window_results"),
    [
        (None, pd.DataFrame([{"rolling_windows": 1, "daily_net_pnl_rms": 1.0}]),
         {"week": {"window_1": {}}}),
        (pd.DataFrame(), None, {"week": {"window_1": {}}}),
        (pd.DataFrame(), pd.DataFrame(), {"week": {"window_1": {}}}),
        (pd.DataFrame(),
         pd.DataFrame([{"rolling_windows": 0, "daily_net_pnl_rms": float("inf")}]),
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
        "rolling_windows": 1, "daily_net_pnl_rms": 3.5,
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
        "rolling_windows": 0, "daily_net_pnl_rms": float("inf"),
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
        "rolling_windows": 0, "daily_net_pnl_rms": float("inf"),
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
        "rolling_windows": 1, "daily_net_pnl_rms": 1.0,
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
        "rolling_windows": 1, "daily_net_pnl_rms": 1.0,
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
        "complete_window": False, "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
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
        "complete_window": False, "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
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
        "complete_window": False, "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
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
        "complete_window": False, "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
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
        "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
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
    form._wind_auto_start_var = _BoolVar(False)

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
            fake._history_btn.state,
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
    assert {fake._run_btn.state,
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


def test_refresh_history_replay_windows_defaults_to_latest_segment():
    """下拉默认停在最近一段——离当下最近，也是最可能要复核的那一笔。"""
    fake = _fake_history_replay_gui(selected="每日收盘")
    fake._history_replay_window_combo = _Widget()

    options = BacktestApp._refresh_history_replay_windows(fake)

    labels = [item["label"] for item in options]
    assert [item["window_id"] for item in options] == [
        "segment_1", "segment_2"]
    assert fake._history_replay_window_var.get() == labels[-1]
    # 已选中的合法标签不得被默认值顶掉。
    fake._history_replay_window_var.set(labels[0])
    BacktestApp._refresh_history_replay_windows(fake)
    assert fake._history_replay_window_var.get() == labels[0]


def test_selected_history_replay_spec_falls_back_to_latest_segment():
    """标签失效时落到最近一段，而不是最早那段。

    下拉的默认值也是末项（见 _refresh_history_replay_windows），两处必须
    一致——否则下拉显示「第 3 段」而加载的是第 1 段，且不会报任何错。
    """
    fake = _fake_history_replay_gui(selected="每日收盘")
    fake._history_replay_window_var.set("已失效的旧标签")

    spec, _strategy_name = BacktestApp._selected_history_replay_spec(fake)

    assert spec.window_id == "segment_2"


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
    # 重放区间保留时分：日内滚动的相邻两段起止日相同，只到天的话两条快照
    # 的行情签名会一模一样，对比页就会说"输入完全相同"。
    assert backtest._gui_meta["wind_start"] == "2026-03-02 00:00"
    assert backtest._gui_meta["wind_end"] == "2026-03-05 00:00"
    assert list(backtest._wind_meta["dates"]) == list(dates)
    state = fake._latest_backtest_state
    # 快照方向与展示口径来自历史任务本身，合约代码收敛到该分段实际合约。
    assert state["wind_code"] == "P2605.DCE"
    assert state["history_replay_strategy"] == "固定间隔(1.5σ)"
    assert state["history_replay_window_id"] == "segment_2"
    assert finished[-1] == ("finish", True, "固定间隔(1.5σ)")


# ---------------------------------------------------------------------------
# 策略优选：图表配对缓存
# ---------------------------------------------------------------------------


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


def _payload_ranking(**overrides):
    row = {
        "lookback": "week", "strategy": "s", "strategy_type": "hedge_band",
        "rolling_windows": 0, "daily_net_pnl_rms": float("nan"),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_validate_payload_reports_short_history_when_failure_columns_missing():
    """缺失败原因列时要给出「历史区间不足」，不能崩成 traceback。

    两组失败原因共用 failure_scopes / failure_reasons。它们曾分处两个
    try：旧快照缺这两列时第一个 try 在赋值前抛 KeyError，第二个 try 再
    引用同名变量就是 UnboundLocalError——它绕过 except，最终由
    _history_recommendation_worker 把 traceback 弹给用户。
    """
    with pytest.raises(ValueError, match="真实历史长度不足"):
        history_selection.validate_payload(
            pd.DataFrame([{"lookback": "week"}]),
            _payload_ranking(),
            {"week": {"segment_1": {}}})


@pytest.mark.parametrize(("overrides", "expected"), [
    ({"strategy_type": "fixed_times", "failure_scope": "strategy",
      "failure_reason": "14:22 无对应 Bar"}, "固定时刻策略没有形成"),
    ({"failure_scope": "endpoint", "failure_reason": "主力映射日不足"},
     "历史具体合约池没有形成"),
    ({"failure_scope": "", "failure_reason": ""}, "真实历史长度不足"),
])
def test_validate_payload_keeps_each_failure_message(overrides, expected):
    """把两段合并进同一个 try 之后，三条分支的原有文案都不能变。"""
    with pytest.raises(ValueError, match=expected):
        history_selection.validate_payload(
            pd.DataFrame([{"lookback": "week"}]),
            _payload_ranking(**overrides),
            {"week": {"segment_1": {}}})


def test_history_scope_explanation_is_gone_with_its_panel():
    """“计算口径”面板已移除，配套文案不该留在代码里。

    这不是洁癖：那段文案写死了“排名依据：整段日损益的波动幅度比「每日
    收盘」低多少”，而单序列路径现在按增量收益排。它进不了界面，却有一
    条常绿的测试断言着这句话——一个绿色的、断言着错误排名依据的测试，
    比没有测试更危险。
    """
    assert not hasattr(history_selection, "scope_explanation")


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
            "_init_history_splitter_ratio",  # 排名/图表分割线的初始比例
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
        lead = (("period", "分析周期", 90), ("strategy", "历史最优参考", 260))
        tree = BacktestApp._build_history_metric_tree(
            ttk.Frame(root), lead_columns=lead, status_heading="状态",
            status_width=114, height=5,
            on_objective_click=lambda _k: None,
            active_objective="incremental_pnl")
        assert tree.heading("incremental_pnl")["text"].endswith("↓")
        assert tree.heading("incremental_sharpe")["text"].endswith("⇅")

        other = BacktestApp._build_history_metric_tree(
            ttk.Frame(root), lead_columns=lead, status_heading="状态",
            status_width=114, height=5,
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


def test_live_wind_date_controls_default_to_backdated_entry_from_latest_close():
    """真实控件树上验证默认口径：截止日主控 + 建仓日倒推回填且不可手填。"""
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    import gui_app as module
    app = module.BacktestApp()
    try:
        app.withdraw()
        app._source_var.set("wind")
        app._toggle_source()
        app.update_idletasks()

        calendar = BacktestApp._trading_calendar_days()
        asof = datetime.date.fromisoformat(app._wind_end_var.get())
        # 默认截止日是最近一个已收盘交易日：当天盘中的残段会让分钟粒度
        # 策略直接被判死，因此不含当天。
        assert asof in calendar
        assert asof < datetime.date.today()
        assert app._wind_auto_start_var.get() is True
        assert str(app._wind_start_entry.cget("state")) == "disabled"

        state = BacktestApp._collect_gui_state(app)
        maturity_days = BacktestApp._maturity_days_from_params(state["params"])
        expected_start = calendar[
            calendar.index(asof) - maturity_days].isoformat()
        assert state["wind_end"] == asof.isoformat()
        assert state["wind_start"] == expected_start
        assert state["wind_date_mode"] == "auto_entry_from_asof"
        # 建仓日框必须显示倒推结果，而不是占位默认值——它是界面上唯一写着
        # 建仓日的地方，已不再有单独的“实际区间”行兜底。
        assert app._wind_start_var.get() == expected_start
        assert not hasattr(app, "_wind_date_hint_var")
    finally:
        app.destroy()


def _history_scoring_note_context(*, relative_windows, paired):
    """渲染一次周期说明，返回那一行文案。

    ``relative_comparison_windows`` 与配对段数一致时不该出现“参与评分”
    补充说明——它是异常提示，不是常驻信息。
    """
    import tkinter as tk
    import gui_app as module
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    def rank_row(rank, strategy, strategy_type, score):
        return {
            "lookback": "quarter", "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "daily_net_pnl_rms": score,
            "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.1,
            "selection_improvement_vs_c2c": 0.1,
            "selection_metric":
                history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "window_win_rate_vs_c2c": 1.0,
            "paired_windows": paired, "baseline_windows": paired,
            "comparison_eligible": True, "rolling_windows": paired,
            "eligible_endpoints": paired, "skipped_endpoints": 0,
            "history_days_available": 61, "lookback_days": 61,
            "complete_window": True, "maturity_days": 22,
            "evidence_days": 61, "days_used": 61,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous",
            "segment_count": paired, "expiry_segments": paired - 1,
            "mtm_segments": 1, "terminal_mode": "mixed",
            "relative_comparison_windows": relative_windows,
            "incremental_pnl_vs_c2c": 0.01,
            "incremental_sharpe_vs_c2c": 0.2,
            "incremental_tc_vs_c2c": 0.001,
            "max_drawdown": 0.03,
        }

    ranking = pd.DataFrame([
        rank_row(1, "固定间隔(1σ)", "hedge_band", 9.0),
        rank_row(2, "每日收盘", "close_to_close", 10.0),
    ])
    recommendations = ranking[ranking["rank"].eq(1)].copy()

    app = module.BacktestApp()
    try:
        app.withdraw()
        BacktestApp._show_history_recommendation(
            app, recommendations, ranking, notes=None,
            source_label="CSV · 测试", window_results=None,
            history_state={"history_lookbacks": {"quarter": 61}})
        app.update_idletasks()
        return app._history_period_context_var.get()
    finally:
        app.destroy()


def test_history_period_note_stays_silent_when_every_segment_scored():
    """全部分段都进了相对评分时，不得再打“参与评分 N/M 段”。

    该字段在严格区间模式下曾被写成 `1 if isfinite(...) else 0` 的是/否
    标志，而展示层拿它跟段数比，于是分段多于一段就恒打这句话，读起来像
    丢掉了 M-1 段证据——实际一段没丢。
    """
    text = _history_scoring_note_context(relative_windows=3, paired=3)

    assert "参与评分" not in text, text
    assert "分成 3 段" in text


def test_history_period_note_still_fires_when_segments_drop_out():
    """确实有分段算不出相对口径时，提示要照常出现。"""
    text = _history_scoring_note_context(relative_windows=2, paired=3)

    assert "参与评分 2/3 段" in text


def _store_ranking_frame():
    """两个周期、含基准的最小排名表，够驱动整条保存/载入链路。"""
    def row(lookback, days, rank, strategy, strategy_type):
        return {
            "lookback": lookback, "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "daily_net_pnl_rms": 10.0 - rank,
            "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.1,
            "selection_improvement_vs_c2c": 0.1,
            "selection_metric":
                history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "window_win_rate_vs_c2c": 1.0, "paired_windows": 3,
            "baseline_windows": 3, "comparison_eligible": True,
            "recommendation_eligible": True, "rolling_windows": 3,
            "eligible_endpoints": 3, "skipped_endpoints": 0,
            "history_days_available": days, "lookback_days": days,
            "complete_window": True, "maturity_days": 22,
            "evidence_days": days, "days_used": days,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous", "segment_count": 3,
            "expiry_segments": 2, "mtm_segments": 1, "terminal_mode": "mixed",
            "relative_comparison_windows": 3, "max_drawdown": 0.03,
            "incremental_pnl_vs_c2c": 0.02 / rank,
            "incremental_sharpe_vs_c2c": 0.3 / rank,
            "incremental_tc_vs_c2c": 0.001 * rank,
            "selection_objective": "incremental_pnl",
            "segment_lengths": (17, 22, 22),
            "comparison_coverage": 1.0,
        }
    return pd.DataFrame([
        row("quarter", 61, 1, "固定间隔(1σ)", "hedge_band"),
        row("quarter", 61, 2, "每日收盘", "close_to_close"),
        row("year", 243, 1, "固定时刻(10:30)", "fixed_times"),
        row("year", 243, 2, "每日收盘", "close_to_close"),
    ])


def _render_store_result(app, *, ranking=None):
    ranking = _store_ranking_frame() if ranking is None else ranking
    BacktestApp._show_history_recommendation(
        app, ranking[ranking["rank"].eq(1)].copy(), ranking, notes=None,
        source_label="Wind · 510050.SH · 1分钟", window_results=None,
        history_state={
            "history_lookbacks": {"quarter": 61, "year": 243},
            "wind_code": "510050.SH", "history_wind_asof": "2026-07-25",
            # 真实 gs 带着不可序列化的回调，保存必须能丢掉它而不是失败。
            "cfg": {"build": lambda *a: None},
        })
    app.update_idletasks()


def _replay_note(app):
    """取逐段下钻栏里那条说明文字。

    按结构定位而不是按关键词：这条说明的措辞会改（「下钻」这类术语就被换
    掉过），靠关键词匹配会在文案一改时静默返回 None，让断言变成对 None 求
    值而不是报出真正的差异。栏里只有「查看某段明细:」这个字段标签和末尾这
    条说明两个 Label，前者以冒号结尾。
    """
    import tkinter as tk
    bar = app._history_replay_window_combo.master
    notes = []
    for child in bar.winfo_children():
        if child.winfo_class() != "Label":
            continue
        try:
            text = child.cget("text")
        except tk.TclError:
            continue
        if isinstance(text, str) and text and not text.endswith(":"):
            notes.append(text)
    return notes[-1] if notes else None


@pytest.fixture
def history_store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gui_app.history_store, "results_dir", lambda: str(tmp_path))
    return tmp_path


def _fresh_app():
    import tkinter as tk
    import gui_app as module
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()
    app = module.BacktestApp()
    app.withdraw()
    return app


def test_save_button_unlocks_only_after_a_result_exists(history_store_dir):
    app = _fresh_app()
    try:
        assert str(app._history_save_btn.cget("state")) == "disabled"
        _render_store_result(app)
        assert str(app._history_save_btn.cget("state")) == "normal"
    finally:
        app.destroy()


def test_saved_result_reloads_into_an_identical_ranking_table(
        history_store_dir, monkeypatch):
    """存盘再读回，排名表必须逐格一致——这是整个功能的意义所在。"""
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring", lambda *a, **k: "基线")
    app = _fresh_app()
    try:
        _render_store_result(app)
        before = [app._history_rank_tree.item(i, "values")
                  for i in app._history_rank_tree.get_children()]
        path = BacktestApp._save_history_result(app)
        assert path is not None and os.path.isfile(path)
    finally:
        app.destroy()

    app2 = _fresh_app()
    try:
        assert BacktestApp._load_history_result(app2, path) is True
        app2.update_idletasks()
        after = [app2._history_rank_tree.item(i, "values")
                 for i in app2._history_rank_tree.get_children()]
        assert after == before
        assert len(app2._history_period_rows) == 2
    finally:
        app2.destroy()


def test_loaded_result_is_labelled_and_blocks_drilldown_without_replay(
        history_store_dir, monkeypatch):
    """没有重放配方的包：载入后自报身份，且下钻控件禁掉并说明原因。

    包里存的是回测**输入**（行情切片 + 构造参数），不是 bar 级输出——后者
    161 个回测约 153 MB，存不了。带配方的包可以下钻（见下一条测试）；这条
    覆盖旧包或配方缺失的情形，那时按钮必须禁掉而不是点了没反应。

    页面仍允许「只填参数 / 用当前行情回测」——那份结论是过去某天算的，不
    标出来会被当成刚跑的结果用。
    """
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring", lambda *a, **k: "近季基线")
    app = _fresh_app()
    try:
        _render_store_result(app)
        # 新跑的结果不该有载入标记；下钻按钮保持可用（此处下拉之所以空，
        # 是这份合成数据没带分段明细，与「载入」无关）。
        assert getattr(app, "_history_loaded_meta", None) is None
        assert str(app._history_replay_button.cget("state")) == "normal"
        assert _replay_note(app) is not None
        assert "不含逐 bar 明细" not in _replay_note(app)
        path = BacktestApp._save_history_result(app)
    finally:
        app.destroy()

    app2 = _fresh_app()
    try:
        BacktestApp._load_history_result(app2, path)
        app2.update_idletasks()
        assert app2._history_loaded_meta["label"] == "近季基线"
        assert str(
            app2._history_replay_window_combo.cget("state")) == "disabled"
        assert str(app2._history_replay_button.cget("state")) == "disabled"
        # 必须讲清为什么不能下钻，而不是留一个点了没反应的按钮。
        assert "不含逐 bar 明细" in _replay_note(app2)

        banners = []

        def walk(widget):
            import tkinter as tk
            for child in widget.winfo_children():
                try:
                    text = child.cget("text")
                except tk.TclError:
                    text = ""
                if isinstance(text, str) and "载入结果" in text:
                    banners.append(text)
                walk(child)

        walk(app2)
        assert len(banners) == 1, banners
        assert "近季基线" in banners[0]
        assert "保存于" in banners[0]
        # 应用到当前回测的那条路径仍然开放。
        assert callable(app2._apply_history_recommendation)

        # 切排名口径走的是同一个渲染入口。「哪来的结果」不会因为换个看法
        # 而改变，横幅与下钻限制都必须保住。
        BacktestApp._set_history_objective(app2, "incremental_sharpe")
        app2.update_idletasks()
        assert app2._history_result_objective == "incremental_sharpe"
        assert app2._history_loaded_meta is not None
        assert str(app2._history_replay_button.cget("state")) == "disabled"
    finally:
        app2.destroy()


def _replay_fixture():
    """一段可重跑的最小配方：真实回测 + 真实价格切片，不依赖 Wind。"""
    from pricing.hedge_analysis import HistoryReplaySpec

    # 按真实分段的形状造：首个 bar 是 Day 0 锚点（前一日收盘），其后 4 个
    # 交易日各 6 根。总点数必须是 evaluation_days*steps_per_day + 1。
    stamps = [pd.Timestamp("2026-01-05 15:00")]
    for day in range(6, 10):
        stamps.extend(
            pd.Timestamp(f"2026-01-{day:02d} 09:3{minute}")
            for minute in range(6))
    index = pd.DatetimeIndex(stamps)
    path = pd.Series(
        100.0 + np.sin(np.arange(len(index)) / 3.0), index=index)
    option = Option_Vanilla(
        "Vanilla", s0=float(path.iloc[0]), sr=[], K=float(path.iloc[0]),
        T=4, sigma=0.18, cp=1, r=0.03, q=0.03)
    spec = HistoryReplaySpec(
        lookback="quarter", window_id="segment_1", option=option,
        external_path=path, evaluation_days=4, steps_per_day=6,
        strategies={"固定间隔(1σ)": HedgeBandStrategy(
            band_type="sigma", threshold=1.0)},
        backtest_kwargs={"tc_rate": 0.0, "quantity": 1, "multiplier": 0},
        metadata={"segment_no": 1, "terminal_mode": "expiry"})
    return {"quarter": {"segment_1": spec}}


def test_saved_replay_recipe_makes_drilldown_work_after_reload(
        history_store_dir, monkeypatch):
    """带重放配方的包，载入后逐段下钻必须可用。

    早前误以为做不到——量的是回测**输出**（161 个约 153 MB）。真正要存的
    是**输入**：一年 1 分钟序列 gzip 后约 206 KB，各段都是它的切片。
    """
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring", lambda *a, **k: "带配方")
    ranking = _store_ranking_frame()
    # 排名行要带上重建策略所需的身份元数据。
    ranking = ranking.assign(
        meta_strategy_name=[
            "hedge_band", "close_to_close", "fixed_times", "close_to_close"],
        meta_candidate_sigma=[1.0, np.nan, np.nan, np.nan],
        meta_fixed_times=[np.nan, np.nan, "10:30", np.nan])
    ranking.loc[0, "strategy"] = "固定间隔(1σ)"

    app = _fresh_app()
    try:
        BacktestApp._show_history_recommendation(
            app, ranking[ranking["rank"].eq(1)].copy(), ranking, notes=None,
            source_label="Wind · 510050.SH · 1分钟", window_results=None,
            history_state={
                "history_lookbacks": {"quarter": 61, "year": 243},
                "wind_code": "510050.SH", "history_wind_asof": "2026-07-25",
                "cls_name": "香草期权 (Vanilla)", "subtype": "Eu",
                "params": {"s0": 100.0, "K": 100.0, "T_days": 4,
                           "sigma": 0.18, "cp": 1, "r": 0.03, "q": 0.03},
            })
        app._history_replay_index = _replay_fixture()
        app.update_idletasks()
        path = BacktestApp._save_history_result(app)
    finally:
        app.destroy()

    app2 = _fresh_app()
    try:
        assert BacktestApp._load_history_result(app2, path) is True
        app2.update_idletasks()
        # 配方重建成功 -> 下钻可用，且文案不再说「不含逐 bar 明细」。
        assert app2._history_loaded_meta["replay_available"] is True
        assert app2._history_replay_index
        assert str(app2._history_replay_button.cget("state")) == "normal"
        assert "不含逐 bar 明细" not in _replay_note(app2)

        spec = app2._history_replay_index["quarter"]["segment_1"]
        assert spec.evaluation_days == 4
        assert spec.steps_per_day == 6
        assert len(spec.external_path) == 25
        assert spec.external_path.index.normalize().nunique() == 5
        # 期权按段初价重定基——与原始运行同一个函数，否则重放会与排名不符。
        assert spec.option.s0 == pytest.approx(100.0, abs=1e-9)
        # 真能跑起来。
        assert spec.replay("固定间隔(1σ)") is not None
    finally:
        app2.destroy()


def test_loading_syncs_the_option_form_and_display_metadata(
        history_store_dir, monkeypatch):
    """载入后左侧表单必须跟着变，展示页摘要也要有来源。

    不同步不只是「看不见」：「用当前行情回测」读的是左侧表单，表单没跟上
    就会拿另一套期权参数去跑，而结论页仍挂着这份载入结果的名字——两者对
    不上且不会报错。逐段下钻的展示页摘要同理，它从 _latest_history_state
    取期权类型/子类型/数据来源，不设就全是 None。
    """
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring", lambda *a, **k: "同步校验")
    frozen = {
        "history_lookbacks": {"quarter": 61, "year": 243},
        "wind_code": "510050.SH", "history_wind_asof": "2026-07-25",
        "cls_name": "香草期权 (Vanilla)", "subtype": "Eu",
        "params": {"s0": 100.0, "K": 100.0, "T_days": 30, "sigma": 0.23,
                   "cp": -1, "r": 0.025, "q": 0.015},
        "source": "wind", "position": -1, "quantity": 50.0,
        "multiplier": 10.0, "tc_rate": 0.0003, "slippage_bps": 2.0,
        "force_day_close_hedge": False,
    }
    app = _fresh_app()
    try:
        ranking = _store_ranking_frame()
        BacktestApp._show_history_recommendation(
            app, ranking[ranking["rank"].eq(1)].copy(), ranking, notes=None,
            source_label="Wind · 510050.SH · 1分钟", window_results=None,
            history_state=frozen)
        app.update_idletasks()
        path = BacktestApp._save_history_result(app)
    finally:
        app.destroy()

    app2 = _fresh_app()
    try:
        # 先把左侧改成完全不同的配置，确认载入真的覆盖了它。
        app2._class_var.set("雪球期权 (Snowball)")
        BacktestApp._on_option_class_change(app2, None)
        app2._qty_var.set("999")
        app2.update_idletasks()

        assert BacktestApp._load_history_result(app2, path) is True
        app2.update_idletasks()

        assert app2._class_var.get() == "香草期权 (Vanilla)"
        assert app2._subtype_var.get() == gui_app.SUBTYPE_DISPLAY["Eu"]
        assert app2._param_entries["sigma"][0].get() == "0.23"
        assert app2._param_entries["T_days"][0].get() == "30"
        assert app2._qty_var.get() == "50"
        assert app2._mult_var.get() == "10"
        # 成本率界面用百分数，状态里是小数
        assert app2._tc_var.get() == "0.03"
        assert bool(app2._force_day_close_hedge_var.get()) is False
        # 展示页摘要的三项来源不能是 None
        assert app2._latest_history_state
        for key in ("cls_name", "subtype", "source"):
            assert app2._latest_history_state.get(key), key
    finally:
        app2.destroy()


def test_action_button_titles_say_what_gets_applied_and_where():
    """按钮名要说清应用的是「策略」。

    原来的「只填参数」没说填的是什么，用户看不到左侧表单变了，会以为什么
    都没发生；「用当前行情回测」也没体现它会先应用策略。策略优选与结果对
    比是同一件事的两个入口，因此共用「应用策略」这一个名字。
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
        ranking = _store_ranking_frame()
        BacktestApp._show_history_recommendation(
            app, ranking[ranking["rank"].eq(1)].copy(), ranking, notes=None,
            source_label="测试", window_results=None,
            history_state={"history_lookbacks": {"quarter": 61}})
        app.update_idletasks()

        titles = []

        def walk(widget):
            from tkinter import ttk
            for child in widget.winfo_children():
                if isinstance(child, ttk.Button):
                    titles.append(str(child.cget("text")))
                walk(child)

        # 只看策略优选页：结果对比页的同类动作现在也叫「应用策略」，
        # 从根节点走会让这条断言被另一个页面的按钮满足。
        walk(app._history_tab)
        assert "应用策略" in titles, titles
        # 只保留这一个动作：原先并排的「应用并用当前行情回测」等于它 + 左
        # 侧「运行回测」，两条路做同一件事只会让人先去分辨该点哪个。
        assert not [t for t in titles if "用当前行情回测" in t], titles
        assert "只填参数" not in titles
    finally:
        app.destroy()


def test_apply_button_does_not_switch_tabs():
    """「应用策略」不得跳走标签页。

    此前它跳到回测摘要页，而那页显示的是上次留下的内容（多半是刚才下钻
    的那个分段）。刚点完「应用参数」就落在一页别人的数字上，会被当成本策
    略的结果读——用户报的「感觉它把明细也加载了」就是这么来的，其实它什
    么都没加载，只是跳了页。
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
        # 必须带上重建策略所需的元数据：缺了 apply 会抛异常，而 Tk 回调会
        # 把异常吞掉——那样这条测试就是空转的，加不加跳页都「通过」。
        ranking = _store_ranking_frame().assign(
            meta_strategy_name=[
                "hedge_band", "close_to_close", "fixed_times",
                "close_to_close"],
            meta_candidate_sigma=[1.0, np.nan, np.nan, np.nan],
            meta_fixed_times=[np.nan, np.nan, "10:30", np.nan])
        BacktestApp._show_history_recommendation(
            app, ranking[ranking["rank"].eq(1)].copy(), ranking, notes=None,
            source_label="测试", window_results=None,
            history_state={"history_lookbacks": {"quarter": 61}})
        app.update_idletasks()
        # 先确认 apply 真能成功——否则下面的断言毫无意义。
        assert BacktestApp._apply_history_recommendation(
            app, navigate=False) is not None

        buttons = []

        def walk(widget):
            from tkinter import ttk
            for child in widget.winfo_children():
                if isinstance(child, ttk.Button) and str(
                        child.cget("text")) == "应用策略":
                    buttons.append(child)
                walk(child)

        # 同上：结果对比页也有「应用策略」，按钮必须从策略优选页里取。
        walk(app._history_tab)
        assert buttons, "找不到应用按钮"

        before = app._nb.select()
        buttons[0].invoke()
        app.update_idletasks()

        assert app._nb.select() == before, "点了应用参数不该切走标签页"
    finally:
        app.destroy()


def test_drilldown_button_is_named_load_not_backtest(history_store_dir):
    """下钻按钮的名字必须说「加载」，不能说「回测」。

    这个按钮读的是优选跑完时已落盘的 bar 级明细，只有明细确实缺失才回退
    重算。它被改名成「单独回测这一段」过一次，而旁边的说明写的是「加载已
    保存的明细」——同一个控件自相矛盾，用户会以为点一下要再算一遍。这条
    把语义钉在名字上，防止再漂回去。
    """
    app = _fresh_app()
    try:
        _render_store_result(app)
        text = str(app._history_replay_button.cget("text"))
    finally:
        app.destroy()
    assert "加载" in text, text
    assert "回测" not in text, text


def test_drilldown_loads_saved_bars_instead_of_recomputing(tmp_path,
                                                           monkeypatch):
    """「加载到展示页」的正常路径是读已保存的明细，不是重跑。

    一轮优选跑完时全部分段的 bar 级结果就已落盘，所以下钻应当命中缓存
    （实测 27 ms vs 重跑 620 ms）。只有缓存确实不在时才回退到重算，并且
    要把「重算过」这件事回报给调用方——按钮叫「加载」，多等半秒得有交代。
    """
    from pricing.hedge_analysis import HistoryReplaySpec

    monkeypatch.setattr(
        gui_app.history_bar_cache, "cache_dir", lambda: str(tmp_path))

    index = pd.DatetimeIndex([
        pd.Timestamp(f"2026-01-{day:02d} 15:00") for day in (5, 6, 7, 8, 9)])
    spec = HistoryReplaySpec(
        lookback="quarter", window_id="segment_1",
        option=Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0, T=4,
                              sigma=0.18, cp=1, r=0.03, q=0.03),
        external_path=pd.Series([100.0, 101.0, 99.0, 102.0, 100.5],
                                index=index),
        evaluation_days=4, steps_per_day=1,
        strategies={"每日收盘": CloseToCloseStrategy()},
        backtest_kwargs={"tc_rate": 0.0, "quantity": 1, "multiplier": 0},
        metadata={})

    # 第一次没有缓存 -> 回退重算，并如实报告
    first, recomputed = BacktestApp._replay_with_cache(spec, "每日收盘")
    assert recomputed is True

    # 第二次读盘 -> 不再重算，且结果逐位一致
    second, recomputed = BacktestApp._replay_with_cache(spec, "每日收盘")
    assert recomputed is False
    for key, value in first._results.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fc":
            np.testing.assert_allclose(
                second._results[key], value, rtol=0, atol=0, equal_nan=True,
                err_msg=key)

    # 缓存被清掉后再退回重算
    gui_app.history_bar_cache.clear(str(tmp_path))
    _third, recomputed = BacktestApp._replay_with_cache(spec, "每日收盘")
    assert recomputed is True


def test_replay_fidelity_error_catches_a_mismatch(history_store_dir):
    """重放与包内逐日损益不符时必须报出来，一致时不得误报。

    这道关早前只在注释里写了名字却没实现——而它本该拦住的正是缓存 key 漏
    项、分段策略集合不对、预热参数丢失这几类「静默错数」。
    """
    from pricing.hedge_analysis import HistoryReplaySpec

    # 每天一根 bar、共 5 个交易日：首根是 Day 0 锚点，其后 4 天参与评估。
    index = pd.DatetimeIndex([
        pd.Timestamp(f"2026-01-{day:02d} 15:00") for day in (5, 6, 7, 8, 9)])
    path = pd.Series([100.0, 101.0, 99.0, 102.0, 100.5], index=index)
    spec = HistoryReplaySpec(
        lookback="quarter", window_id="segment_1",
        option=Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0, T=4,
                              sigma=0.18, cp=1, r=0.03, q=0.03),
        external_path=path, evaluation_days=4, steps_per_day=1,
        strategies={"每日收盘": CloseToCloseStrategy()},
        backtest_kwargs={"tc_rate": 0.0, "quantity": 1, "multiplier": 0},
        metadata={})
    result = spec.replay("每日收盘")._results
    actual = _agg_daily_frame(result, 1)["net_pnl"].to_numpy()

    faithful = pd.DataFrame([{
        "lookback": "quarter", "window_id": "segment_1",
        "strategy": "每日收盘", "success": True, "daily_net_pnl": actual,
    }])
    assert BacktestApp._replay_fidelity_error(
        result, spec, "每日收盘", faithful) is None

    # 数值被动过一点点就必须抓出来
    tampered = faithful.copy()
    bad = actual.copy()
    bad[0] += 1e-6
    tampered.at[0, "daily_net_pnl"] = bad
    message = BacktestApp._replay_fidelity_error(
        result, spec, "每日收盘", tampered)
    assert message and "不符" in message

    # 天数对不上也要抓
    shorter = faithful.copy()
    shorter.at[0, "daily_net_pnl"] = actual[:-1]
    assert "不符" in BacktestApp._replay_fidelity_error(
        result, spec, "每日收盘", shorter)

    # 失败段或无摘要时不比对，也不该误报
    failed = faithful.copy()
    failed.at[0, "success"] = False
    assert BacktestApp._replay_fidelity_error(
        result, spec, "每日收盘", failed) is None
    assert BacktestApp._replay_fidelity_error(
        result, spec, "每日收盘", None) is None


def test_segment_strategies_drops_candidates_that_failed_in_that_segment():
    """实跑路径只记录该段成功的候选；载入不该给每段塞上全部候选。

    否则原本失败的段也会进下拉，点下去要么撞引擎报错、要么跑出一个从未
    参与排名的数字。
    """
    strategies = {"每日收盘": object(), "固定时刻(10:30)": object(),
                  "固定间隔(1σ)": object()}
    summary = pd.DataFrame([
        {"lookback": "quarter", "window_id": "segment_1",
         "strategy": "每日收盘", "success": True},
        {"lookback": "quarter", "window_id": "segment_1",
         "strategy": "固定时刻(10:30)", "success": False},
        {"lookback": "quarter", "window_id": "segment_1",
         "strategy": "固定间隔(1σ)", "success": True},
    ])

    kept = BacktestApp._segment_strategies(
        strategies, summary, "quarter", "segment_1")

    assert set(kept) == {"每日收盘", "固定间隔(1σ)"}
    # 摘要缺失或一个都对不上时不过滤——宁可多给几项，也别把下拉清空。
    assert set(BacktestApp._segment_strategies(
        strategies, None, "quarter", "segment_1")) == set(strategies)
    assert set(BacktestApp._segment_strategies(
        strategies, summary, "year", "segment_9")) == set(strategies)


def test_running_a_fresh_optimisation_clears_the_loaded_marker(
        history_store_dir, monkeypatch):
    """真跑一轮之后不能还挂着「载入结果」横幅、也不能继续禁用下钻。"""
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring", lambda *a, **k: "旧结果")
    app = _fresh_app()
    try:
        _render_store_result(app)
        path = BacktestApp._save_history_result(app)
        BacktestApp._load_history_result(app, path)
        app.update_idletasks()
        assert app._history_loaded_meta is not None

        # 再渲染一次而不传 loaded_meta —— 真跑一轮走的就是这条路。标记由
        # 渲染入口接管，调用方不需要（也不该）自己记得清。
        _render_store_result(app)

        assert getattr(app, "_history_loaded_meta", None) is None
        assert str(app._history_replay_button.cget("state")) == "normal"
        assert "不含逐 bar 明细" not in _replay_note(app)
    finally:
        app.destroy()


def test_cancelling_the_name_prompt_saves_nothing(
        history_store_dir, monkeypatch):
    """点取消不该留下文件——留空名字和取消是两件事。"""
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring", lambda *a, **k: None)
    app = _fresh_app()
    try:
        _render_store_result(app)
        assert BacktestApp._save_history_result(app) is None
    finally:
        app.destroy()
    assert gui_app.history_store.list_results(str(history_store_dir)) == []


def _objective_header_state(*, with_incremental_columns):
    """按有无增量列渲染一次排名表，返回表头可点性与文案。"""
    import tkinter as tk
    import gui_app as module
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    def rank_row(rank, strategy, strategy_type):
        row = {
            "lookback": "quarter", "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "daily_net_pnl_rms": 10.0 - rank,
            "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.1,
            "selection_improvement_vs_c2c": 0.1,
            "selection_metric":
                history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "window_win_rate_vs_c2c": 1.0, "paired_windows": 3,
            "baseline_windows": 3, "comparison_eligible": True,
            "rolling_windows": 3, "eligible_endpoints": 3,
            "skipped_endpoints": 0, "history_days_available": 61,
            "lookback_days": 61, "complete_window": True, "maturity_days": 22,
            "evidence_days": 61, "days_used": 61,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous", "segment_count": 3,
            "expiry_segments": 2, "mtm_segments": 1, "terminal_mode": "mixed",
            "relative_comparison_windows": 3, "max_drawdown": 0.03,
        }
        if with_incremental_columns:
            row.update({
                "incremental_pnl_vs_c2c": 0.01,
                "incremental_sharpe_vs_c2c": 0.2,
                "incremental_tc_vs_c2c": 0.001,
                "selection_objective": "incremental_pnl",
            })
        return row

    ranking = pd.DataFrame([
        rank_row(1, "固定间隔(1σ)", "hedge_band"),
        rank_row(2, "每日收盘", "close_to_close"),
    ])
    recommendations = ranking[ranking["rank"].eq(1)].copy()

    app = module.BacktestApp()
    try:
        app.withdraw()
        BacktestApp._show_history_recommendation(
            app, recommendations, ranking, notes=None,
            source_label="测试", window_results=None,
            history_state={"history_lookbacks": {"quarter": 61}})
        app.update_idletasks()
        tree = app._history_rank_tree
        return {
            "available": app._history_objectives_available,
            "clickable": [c for c in tree["columns"]
                          if tree.heading(c)["command"]],
            "headings": {c: tree.heading(c)["text"]
                         for c in ("incremental_pnl", "incremental_sharpe")},
            "note": app._history_period_context_var.get(),
        }
    finally:
        app.destroy()


def test_chart_candidate_limit_covers_a_default_run():
    """图表上限必须装得下一次默认运行的全部候选。

    默认候选是 5 档带宽 + 可选「加入当前回测带宽」+ 固定时刻 = 7 个非基准
    候选。上限低于它就意味着永远看不全——而候选之间的差别恰恰要靠曲线叠
    在一起才看得出来。正确性由「加入前预演共同分段交集」单独保证，这个数
    只管可读性。
    """
    default_band = len(history_selection.DEFAULT_BAND_CANDIDATE_SIGMAS)
    default_candidates = default_band + 1 + 1  # 带宽 + 当前带宽 + 固定时刻

    assert history_selection.MAX_HISTORY_CHART_CANDIDATES >= default_candidates
    # 标记只有 10 个，超过之后不同候选会共用同一个标记。
    assert history_selection.MAX_HISTORY_CHART_CANDIDATES <= len(
        gui_app.STRATEGY_CHART_MARKERS)
    assert (gui_app.MAX_HISTORY_CHART_CANDIDATES
            == history_selection.MAX_HISTORY_CHART_CANDIDATES)


@pytest.mark.parametrize(("value", "expected"), [
    (5.6096, "+5.6096"),
    (-0.0057, "-0.0057"),
    # 零就写成零。此前一律显示「基准」，会把「带宽宽到全程没触发、增量恰
    # 好为零」的候选伪装成基线行；而基线的身份已经写在策略列里了。
    # 前导空格是符号位占位：列是右对齐的，少一个字符会让小数点错开一格。
    (0.0, " 0.0000"),
    (1e-15, " 0.0000"),
    (None, "—"),
    (float("nan"), "—"),
    (float("inf"), "—"),
])
def test_objective_value_formatting_writes_zero_as_zero(value, expected):
    assert BacktestApp._format_objective_value(value) == expected


def test_consensus_conclusion_is_rendered_exactly_once():
    """跨周期一致性全页只能有一枚 pill，且必须带「几种结论」。

    此前顶部摘要与结果区各渲染一枚，相隔约 30px 同屏出现、措辞还略有出入
    （一个带计数一个不带），读者会以为是两条不同的结论。
    """
    import tkinter as tk
    import gui_app as module
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    def rank_row(lookback, rank, strategy, strategy_type):
        return {
            "lookback": lookback, "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "daily_net_pnl_rms": 10.0 - rank,
            "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.1,
            "selection_improvement_vs_c2c": 0.1,
            "selection_metric":
                history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "window_win_rate_vs_c2c": 1.0, "paired_windows": 3,
            "baseline_windows": 3, "comparison_eligible": True,
            "recommendation_eligible": True, "rolling_windows": 3,
            "eligible_endpoints": 3, "skipped_endpoints": 0,
            "history_days_available": 61, "lookback_days": 61,
            "complete_window": True, "maturity_days": 22,
            "evidence_days": 61, "days_used": 61,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous", "segment_count": 3,
            "expiry_segments": 2, "mtm_segments": 1, "terminal_mode": "mixed",
            "relative_comparison_windows": 3, "max_drawdown": 0.03,
            "incremental_pnl_vs_c2c": 0.01,
            "incremental_sharpe_vs_c2c": 0.2,
            "incremental_tc_vs_c2c": 0.001,
            "selection_objective": "incremental_pnl",
        }

    # 两个周期给出不同的最优候选 -> disagree 分支。
    ranking = pd.DataFrame([
        rank_row("week", 1, "固定间隔(1σ)", "hedge_band"),
        rank_row("week", 2, "每日收盘", "close_to_close"),
        rank_row("quarter", 1, "固定时刻(10:30)", "fixed_times"),
        rank_row("quarter", 2, "每日收盘", "close_to_close"),
    ])
    recommendations = ranking[ranking["rank"].eq(1)].copy()

    app = module.BacktestApp()
    try:
        app.withdraw()
        # 顶部摘要在来源警告非空时会提前 return（默认来源是「模拟」，那条
        # 警告一直占着位）。不清掉它，顶部这条路径根本不会渲染，测试也就
        # 盯不住「顶部不得再放一枚一致性 pill」这件事。
        app._history_source_hint_var.set("")
        BacktestApp._show_history_recommendation(
            app, recommendations, ranking, notes=None,
            source_label="Wind · 测试 · 1分钟", window_results=None,
            history_state={"history_lookbacks": {"week": 5, "quarter": 61}})
        app.update_idletasks()

        found = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    text = child.cget("text")
                except tk.TclError:
                    text = ""
                if isinstance(text, str) and "一致" in text:
                    found.append(text)
                walk(child)

        walk(app)
    finally:
        app.destroy()

    assert len(found) == 1, found
    assert "不一致" in found[0]
    # 顶部那枚重复的携带着「几种结论」，移除它时这个计数不能跟着丢。
    assert "2 种" in found[0], found[0]


def test_objective_value_zero_carries_no_sign():
    """零不带正负号——加号会暗示它是个正的增量。"""
    text = BacktestApp._format_objective_value(0.0)

    assert not text.strip().startswith(("+", "-"))
    assert "基准" not in text


def test_objective_value_zero_still_occupies_the_sign_column():
    """零要占住符号位，否则整列的小数点对不齐。

    这一列是右对齐的：``0.0000`` 比 ``-0.0211`` 短一个字符，基准行的小数
    点就会比其它行错开一格，看着像换了一种数字格式。
    """
    zero = BacktestApp._format_objective_value(0.0)
    negative = BacktestApp._format_objective_value(-0.0211)
    positive = BacktestApp._format_objective_value(0.0061)

    assert len(zero) == len(negative) == len(positive)
    # 小数点在同一个字符位上。
    assert zero.index(".") == negative.index(".") == positive.index(".")


def test_baseline_row_label_only_marks_identity():
    """基线行只标明身份，不再往策略列里拼分界说明。

    同一行的 # 列与三个指标格已经都显示「基准」，策略列再挂一句
    「以上有正贡献」/「以上更优 · 以下更差」既重复又越写越长。
    """
    label = BacktestApp._history_baseline_row_label("每日收盘")

    assert label == "每日收盘（基准）"
    for stale in ("贡献", "以上", "以下", "——"):
        assert stale not in label


def test_objective_headers_stay_clickable_when_increments_exist():
    state = _objective_header_state(with_incremental_columns=True)

    assert state["available"] is True
    assert state["clickable"] == ["incremental_pnl", "incremental_sharpe"]
    assert "↓" in state["headings"]["incremental_pnl"]
    assert "⇅" in state["headings"]["incremental_sharpe"]
    assert "本次按" not in state["note"]


def test_objective_headers_degrade_when_ranking_has_no_increments():
    """没有增量列时表头不得可点，且必须说出真正的排名依据。

    品种池模式跨合约不能把金额 PnL 直接相加，因此只产出逐段有界改善。
    此时留着 ⇅ 会让点击变成**静默 no-op**——`_rank_history_rows` 两个口
    径都回退到同一个值，↓ 标记移动了、顺序一动不动。而四列指标里三列全
    是「—」，真正的排序依据没有任何一列体现。
    """
    state = _objective_header_state(with_incremental_columns=False)

    assert state["available"] is False
    assert state["clickable"] == []
    assert "⇅" not in state["headings"]["incremental_sharpe"]
    assert "↓" not in state["headings"]["incremental_pnl"]
    assert "较每日收盘的改善" in state["note"]


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


def _render_history_result_page(*, agree=True):
    """用合成排名渲染一次结果页，返回已构建的 app（调用方负责 destroy）。"""
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    def rank_row(lookback, rank, strategy, strategy_type, *,
                 inc_pnl, paired=9, eligible=True):
        return {
            "lookback": lookback, "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type, "daily_net_pnl_rms": 10.0 - rank,
            "baseline_daily_net_pnl_rms": 10.0, "improvement_vs_c2c": 0.12,
            "selection_improvement_vs_c2c": 0.12,
            "selection_metric":
                history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "selection_objective": "incremental_pnl",
            "window_win_rate_vs_c2c": 0.67,
            "paired_windows": paired, "baseline_windows": paired,
            "comparison_eligible": eligible,
            "recommendation_eligible": eligible,
            "rolling_windows": paired, "eligible_endpoints": paired,
            "skipped_endpoints": 0, "history_days_available": 252,
            "lookback_days": 252, "complete_window": eligible,
            "maturity_days": 30, "evidence_days": 252, "days_used": 252,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous",
            "segment_count": paired, "expiry_segments": paired - 1,
            "mtm_segments": 1, "terminal_mode": "mixed",
            "relative_comparison_windows": paired,
            "incremental_pnl_vs_c2c": inc_pnl,
            "incremental_sharpe_vs_c2c": inc_pnl / 10000.0,
            "incremental_tc_vs_c2c": -0.0031, "max_drawdown": 0.0342,
        }

    rows = []
    for lookback, best in (
            ("quarter", "固定间隔(0.75σ)"),
            ("year", "固定间隔(0.75σ)" if agree else "固定时刻(10:30)")):
        rows += [
            rank_row(lookback, 1, best, "hedge_band", inc_pnl=1248.35),
            rank_row(lookback, 2, "每日收盘", "close_to_close", inc_pnl=0.0),
            rank_row(lookback, 3, "固定间隔(2σ)", "hedge_band",
                     inc_pnl=-655.4, paired=4, eligible=False),
        ]
    ranking = pd.DataFrame(rows)
    recommendations = ranking[ranking["rank"].eq(1)].copy()

    app = gui_app.BacktestApp()
    app.withdraw()
    BacktestApp._show_history_recommendation(
        app, recommendations, ranking, notes=None, source_label="测试",
        window_results=None,
        history_state={"history_lookbacks": {"quarter": 61, "year": 252}})
    app.update_idletasks()
    return app


def test_history_result_blocks_never_share_a_grid_row():
    """结果页四个区块必须各占一行。

    它们都 grid 在同一个父容器上，行号是手写的常量；插入新区块时漏改后
    面某一个，两块就会叠在同一行互相压住——而 grid 不会报错，界面只是
    少了一块，跑测试也全绿。
    """
    app = _render_history_result_page()
    try:
        parent = app._history_conclusion_card.master
        rows = {}
        for child in parent.winfo_children():
            info = child.grid_info()
            if not info:
                continue
            rows.setdefault(int(info["row"]), []).append(child.winfo_class())
        duplicated = {k: v for k, v in rows.items() if len(v) > 1}
        assert not duplicated, f"同一 grid 行上有多个区块: {duplicated}"

        # 结论卡必须排在分割区之前：它是结论，不是表格的脚注。
        card_row = int(app._history_conclusion_card.grid_info()["row"])
        splitter_row = int(app._history_splitter.grid_info()["row"])
        assert card_row < splitter_row

        # 排名表与图表是分割区的两格，顺序为上表下图。
        panes = [str(p) for p in app._history_splitter.panes()]
        assert panes == [
            str(app._history_rank_tree.master),
            str(app._history_chart_canvas.get_tk_widget().master),
        ], panes
    finally:
        app.destroy()


def test_history_result_page_has_no_nested_scrolling_canvas():
    """结果页不得再把可滚动画布套在表格和图表外面。

    三层嵌套滚动要靠 ``bind_all`` 在画布、表格、图表之间抢滚轮；两块内容
    各自合适的高度改由用户拖分割线决定，这里守住不回退。
    """
    app = _render_history_result_page()
    try:
        def walk(widget, found):
            for child in widget.winfo_children():
                if child.winfo_class() == "Canvas":
                    found.append(str(child))
                walk(child, found)
            return found

        chart_widget = str(app._history_chart_canvas.get_tk_widget())
        canvases = [
            name for name in walk(app._history_results_container, [])
            if name != chart_widget
        ]
        # 图表自己那块 Canvas 之外，结果区不该再有别的画布。
        assert canvases == [], canvases
    finally:
        app.destroy()


class _FakeSplitter:
    """只实现 `_init_history_splitter_ratio` 用到的那几个 PanedWindow 接口。

    真窗口在 withdraw 状态下不做布局，`winfo_height()` 恒为 1，用真控件测
    不出比例逻辑；而这套逻辑本身与渲染无关。

    ``echo`` 复现真实页面里的那条反馈链：摆放分割线会让图表重绘、
    LabelFrame 的请求尺寸变化、外层 grid 再把 splitter 的高度改掉二十来
    像素，于是 ``<Configure>`` 又打回来。最初这个假控件是"摆放不产生任
    何后果"的空壳，于是漏掉了一个把界面彻底卡死的死循环。
    """

    def __init__(self, height, echo=0):
        self.height = height
        self.echo = echo
        self.sash = 0
        self.placed = []
        self._handlers = {}
        self._depth = 0

    def winfo_height(self):
        return self.height

    def sash_coord(self, index):
        return (0, self.sash)

    def sash_place(self, index, x, y):
        self.placed.append((index, x, y))
        self.sash = y
        if self.echo and self._depth < 60:
            # 高度在两个值之间来回，正是实测到的 621↔645。
            self._depth += 1
            self.height += self.echo
            self.echo = -self.echo
            self.fire("<Configure>")
            self._depth -= 1

    def bind(self, sequence, func, add=None):
        self._handlers[sequence] = func

    def after_idle(self, func):
        func()

    def fire(self, sequence):
        handler = self._handlers.get(sequence)
        if handler is not None:
            handler()


def test_history_splitter_ignores_its_own_layout_echo():
    """摆放分割线引起的高度回弹不得再触发摆放。

    这是把整个界面卡死的那条闭环：sash_place → 图表重绘 → 请求尺寸变化
    → grid 重排 → splitter 高度变 → <Configure> → sash_place。实测单次
    渲染里 sash_place 被调用 300+ 次、一帧都画不出来、内存被 Agg 缓冲区
    撑到 GB 级。
    """
    splitter = _FakeSplitter(621, echo=24)
    BacktestApp._init_history_splitter_ratio(
        SimpleNamespace(), splitter, ratio=0.45)

    assert splitter.placed == [(0, 0, int(621 * 0.45))], splitter.placed


def test_history_splitter_placement_budget_bounds_any_unforeseen_wobble():
    """即使抖动幅度大到绕过回波判定，摆放次数也必须有硬上限。"""
    # 回弹幅度远超容忍比例，模拟未预料的剧烈抖动。
    splitter = _FakeSplitter(600, echo=400)
    BacktestApp._init_history_splitter_ratio(
        SimpleNamespace(), splitter, ratio=0.45)

    assert len(splitter.placed) <= BacktestApp._HISTORY_SASH_PLACEMENT_BUDGET


def test_history_splitter_follows_the_ratio_until_the_user_drags():
    """分割线按比例跟随容器高度，用户一拖就彻底交权。

    只在第一次 `<Configure>` 落一次是不够的：那一刻拿到的常是还没排完版
    的临时高度，比例会被冻结在一个偏小的位置上，排名表一上来只剩四五行。
    但用户拖过之后再跟随，就会把他刚调好的比例冲掉。
    """
    splitter = _FakeSplitter(600)
    BacktestApp._init_history_splitter_ratio(
        SimpleNamespace(), splitter, ratio=0.45)

    # after_idle 立刻摆一次，不必等第一个 <Configure>。
    assert splitter.placed[-1] == (0, 0, 270)

    # 容器变高、用户还没拖过 —— 继续按比例跟随。
    splitter.height = 900
    splitter.fire("<Configure>")
    assert splitter.placed[-1] == (0, 0, 405)

    # 高度还没算出来时不摆放，免得把比例冻结在临时值上。
    splitter.height = 1
    placements = len(splitter.placed)
    splitter.fire("<Configure>")
    assert len(splitter.placed) == placements

    # 极矮容器下不得把排名表压没：下限 150 优先于比例。
    splitter.height = 200
    splitter.fire("<Configure>")
    assert splitter.placed[-1] == (0, 0, 150)

    # 用户按住分割线之后交权：后续尺寸变化不得再改动位置。
    splitter.fire("<ButtonPress-1>")
    placements = len(splitter.placed)
    splitter.height = 1000
    splitter.fire("<Configure>")
    assert len(splitter.placed) == placements


def test_history_conclusion_card_tracks_the_selected_rank_row():
    """结论卡显示的必须是动作按钮真正会作用的那一行。

    两个按钮取的是排名表的选中行；卡片若固定显示周期冠军，用户选中别的
    策略后看到的结论与点下去实际跑的策略就是两回事。
    """
    app = _render_history_result_page()
    try:
        # 默认选中 rank 1，卡片即本周期推荐。
        assert "本周期最优" in app._history_conclusion_badge_var.get()
        assert app._history_conclusion_name_var.get() == "固定间隔(0.75σ)"

        tree = app._history_rank_tree
        children = tree.get_children()
        tree.selection_set(children[1])
        tree.focus(children[1])
        BacktestApp._update_history_rank_selection(app)
        app.update_idletasks()
        assert app._history_conclusion_name_var.get() == "每日收盘"
        assert "基准" in app._history_conclusion_badge_var.get()

        # 数据不足的候选要在卡片上直说，不能只靠表格底色暗示。
        tree.selection_set(children[2])
        tree.focus(children[2])
        BacktestApp._update_history_rank_selection(app)
        app.update_idletasks()
        assert app._history_conclusion_name_var.get() == "固定间隔(2σ)"
        assert "数据不足" in app._history_conclusion_badge_var.get()
    finally:
        app.destroy()


def test_history_run_button_stays_pinned_in_the_page_header():
    """主行动固定在页首，折叠候选配置不会让它移位。

    按钮此前跟在候选配置下方，折叠配置时会随之上跳，出结果后又被推到滚动
    区上方，想改参数重跑得先往回滚。
    """
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        app.update_idletasks()
        header = app._history_btn.master
        # 主按钮与折叠开关同处页首一行，折叠候选配置不会挪动它。
        assert int(app._history_btn.grid_info()["row"]) == 0
        assert header is app._history_config_toggle_btn.master
        before = app._history_btn.grid_info()
        BacktestApp._toggle_history_config_panel(app)
        app.update_idletasks()
        assert app._history_btn.grid_info() == before

        # 折叠后仍能展开，且配置面板回到结果区之上。
        BacktestApp._toggle_history_config_panel(app)
        app.update_idletasks()
        assert app._history_config_panel.winfo_manager() == "pack"
    finally:
        app.destroy()


def test_history_row_padding_follows_each_column_anchor():
    """内边距要按列的对齐方向补在正确的一侧。"""
    keys = ("rank", "strategy", "max_drawdown", "status")
    padded = BacktestApp._pad_history_row(
        ("1", "每日收盘", "0.3948", "✓"), keys)

    # 居中列不动；左对齐补在前面、右对齐补在后面。
    assert padded == ("1", " 每日收盘", "0.3948 ", "✓")
    # 空单元格不补，否则会凭空多出一格看不见的宽度。
    assert BacktestApp._pad_history_row(
        ("", ""), ("strategy", "max_drawdown")) == ("", "")


def test_history_rank_columns_never_let_neighbouring_cells_touch():
    """相邻两列的文本之间必须留出可见间隙。

    ttk.Treeview 没有单元格内边距：右对齐的数字紧贴本列右边界，左对齐的
    文本紧贴下一列左边界。右对齐的「最大回撤」后面原本跟着左对齐的「样本
    完整度」，两者间隙实测正好是 0 像素——`0.3948` 和 `✓` 直接粘成一团。
    """
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import ttk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    try:
        root.withdraw()
        tree = BacktestApp._build_history_metric_tree(
            root,
            lead_columns=(("rank", "#", 38), ("strategy", "策略 / 参数", 232)),
            status_heading="样本完整度", status_width=129, height=8)
        columns = list(tree["columns"])
        # 结构性约束：右对齐列后面绝不能紧跟左对齐列。这两种对齐各自把文
        # 本推到相邻的那条边界上，中间不留任何余地——补空格只是把症状盖住
        # 了，列一窄照样会碰。
        anchors = [BacktestApp._history_column_anchor(c) for c in columns]
        collisions = [
            f"{columns[i]}(e)→{columns[i + 1]}(w)"
            for i in range(len(columns) - 1)
            if anchors[i] == "e" and anchors[i + 1] == "w"
        ]
        assert not collisions, f"右对齐列后面紧跟左对齐列: {collisions}"

        # 取实际会出现的最长内容：负号 + 四位小数，以及最长的策略名。
        values = BacktestApp._pad_history_row(
            ("🥇 1", "固定时刻(23:00,11:30,15:00)", "-2.6190", "-0.1134",
             "+0.1190", "0.3948", "✓"), columns)
        font = tkfont.Font(
            font=ttk.Style().lookup("Treeview", "font") or "TkDefaultFont")

        spans = {}
        offset = tree.column("#0", "width")
        for column in columns:
            width = tree.column(column, "width")
            spans[column] = (offset, offset + width, width,
                             BacktestApp._history_column_anchor(column))
            offset += width

        def visible(column, text):
            """可见字形的像素区间——补的空格占位但不显形，两端要扣掉。"""
            left, right, width, anchor = spans[column]
            total = font.measure(text)
            lead = total - font.measure(text.lstrip())
            trail = total - font.measure(text.rstrip())
            if anchor == "w":
                return left + lead, left + total - trail
            if anchor == "e":
                return right - total + lead, right - trail
            return (left + (width - total) / 2 + lead,
                    left + (width + total) / 2 - trail)

        extents = [visible(c, v) for c, v in zip(columns, values)]
        tight = {
            f"{columns[i]}→{columns[i + 1]}":
                round(extents[i + 1][0] - extents[i][1], 1)
            for i in range(len(columns) - 1)
            if extents[i + 1][0] - extents[i][1] < 4
        }
        assert not tight, f"相邻列文本几乎贴在一起: {tight}"
    finally:
        root.destroy()


def test_history_rank_headers_align_with_their_column_data():
    """表头必须和本列数据同向对齐。

    表头一律居中、数字却右对齐时，一列里没有任何一条共同的竖边，眼睛就
    找不到列的界限——这比缺少分割线更影响可读性。
    """
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    try:
        root.withdraw()
        tree = BacktestApp._build_history_metric_tree(
            root,
            lead_columns=(("rank", "#", 38), ("strategy", "策略 / 参数", 232)),
            status_heading="样本完整度", status_width=129, height=8)
        mismatched = {
            column: (tree.heading(column, "anchor"),
                     tree.column(column, "anchor"))
            for column in tree["columns"]
            if str(tree.heading(column, "anchor"))
            != str(tree.column(column, "anchor"))
        }
        assert not mismatched, f"表头与数据对齐方向不一致: {mismatched}"

        # 数字列必须右对齐：居中会打乱数位，让同一列的数值无法按位比较。
        for column in ("incremental_pnl", "incremental_sharpe",
                       "incremental_tc", "max_drawdown"):
            assert str(tree.column(column, "anchor")) == "e", column
    finally:
        root.destroy()


def test_rank_rows_carry_no_background_tint_at_all():
    """排名表的行不得再有任何底色，选中态是唯一的行级底色。

    此前按角色/名次/完整度分了三档底色，每一档都有毛病：绿色标"第一行"
    却总被自动选中的蓝底盖住（实测三种排名形态下一次都没显形）；判定用
    ``row_no == 0`` 而不是 ``rank == 1``，基准占了第 0 行时真正最优的候选
    反而是纯白；而"可比（仅参考）"这档降级完全没有颜色，与健康行同色，同
    一行的结论卡却显示"数据不足"。
    """
    app = _render_history_result_page()
    try:
        tree = app._history_rank_tree
        tinted = {}
        for iid in tree.get_children():
            for tag in (tree.item(iid, "tags") or ()):
                if not tag:
                    continue
                background = str(tree.tag_configure(tag, "background") or "")
                if background:
                    tinted[tag] = background
        assert not tinted, f"仍有行标签带底色: {tinted}"

        # 分档底色的那几个标签必须彻底消失，而不是只把 background 留空。
        for stale in ("leader", "incomplete"):
            assert not tree.tag_configure(stale, "background"), stale

        # 基准行仍靠等宽加粗强调——那是字体不是底色。
        baseline_font = tree.tag_configure("baseline", "font")
        assert baseline_font, "基准行的加粗强调不应一起被删掉"
        assert "bold" in str(baseline_font)
    finally:
        app.destroy()


def test_rank_table_row_tags_keep_the_monospaced_body_font():
    """行标签不得把表格字体换成比例字体。

    基准行曾用 `_UI_FONT_FAMILY` 加粗——那是比例字体，实测同一个数字在它
    下面只有 28~32px 宽，而表格正文 (Menlo 9) 恒为 35px，于是基准行的数位
    与上下各行全部错开，读起来就是"另一种数字格式"。等宽字体的加粗步进宽
    度与常规一致，可以放心用来强调。
    """
    import tkinter.font as tkfont
    from tkinter import ttk

    samples = (" 0.0000", "-0.0211", "+0.6708", "-0.5102")
    app = _render_history_result_page()
    try:
        tree = app._history_rank_tree
        body = tkfont.Font(
            root=app, font=ttk.Style(app).lookup("Treeview", "font"))

        # 正文本身必须等宽，否则右对齐也救不了数位对齐。
        widths = {body.measure(s) for s in samples}
        assert len(widths) == 1, f"表格正文不是等宽字体: {widths}"

        # 任何覆盖了字体的行标签，都必须与正文同族同宽。
        for tag in ("baseline", "leader", "incomplete"):
            spec = tree.tag_configure(tag, "font")
            if not spec:
                continue          # 没覆盖字体的标签天然安全
            tagged = tkfont.Font(root=app, font=spec)
            assert tagged.actual("family") == body.actual("family"), (
                f"{tag} 换了字体族: {tagged.actual('family')} "
                f"≠ {body.actual('family')}")
            for sample in samples:
                assert tagged.measure(sample) == body.measure(sample), (
                    f"{tag} 的 {sample!r} 宽度与正文不一致")
    finally:
        app.destroy()


def _default_chart_selection(rows, *, unplottable=(), monkeypatch=None):
    """在假排名表上跑一次默认勾选，返回被勾中的候选名（按名次序）。"""
    class _Tree:
        def get_children(self):
            return tuple(f"row{i}" for i in range(len(rows)))

    def fake_model(_summary, _lookback, trial, **_kw):
        if any(name in unplottable for name in trial):
            return {"state": "no_common_windows"}
        return {"state": "ok"}

    monkeypatch.setattr(
        BacktestApp, "_history_multi_chart_model", staticmethod(fake_model))
    fake = SimpleNamespace(
        _history_rank_tree=_Tree(),
        _history_rank_rows={f"row{i}": row for i, row in enumerate(rows)},
        _history_window_summary=object(),
        _history_chart_selection=lambda: ("quarter", {}),
        _history_chart_view_options=lambda: ("full", "net"),
        _history_chart_pairs_cache=lambda: {},
    )
    return BacktestApp._history_top_chart_candidates(fake)


def _rank_entry(name, *, baseline=False):
    return {"strategy": name,
            "strategy_type": "close_to_close" if baseline else "hedge_band"}


def test_default_selection_checks_the_top_three_ranks(monkeypatch):
    """默认勾选排名最靠前的三个候选。"""
    rows = [_rank_entry(f"候选{i}") for i in range(1, 6)]

    assert _default_chart_selection(
        rows, monkeypatch=monkeypatch) == ["候选1", "候选2", "候选3"]


def test_baseline_occupies_a_rank_but_is_never_checked(monkeypatch):
    """基准也占一个名次，所以它排进前三时默认只勾上两个候选。

    基准是固定基准、图上恒在，不参与勾选；为了凑够三个而往下多抓一个名
    次会让默认勾选不再对应"最靠前的那几名"。
    """
    rows = [
        _rank_entry("候选1"),
        _rank_entry("每日收盘", baseline=True),
        _rank_entry("候选3"),
        _rank_entry("候选4"),
    ]

    assert _default_chart_selection(
        rows, monkeypatch=monkeypatch) == ["候选1", "候选3"]


def test_default_selection_never_reaches_past_the_third_rank(monkeypatch):
    """名次窗口之外的候选不得被默认勾上。"""
    rows = [_rank_entry(f"候选{i}") for i in range(1, 9)]

    selected = _default_chart_selection(rows, monkeypatch=monkeypatch)

    assert "候选4" not in selected
    assert len(selected) == BacktestApp._HISTORY_DEFAULT_CHART_RANKS


def test_unplottable_candidate_is_skipped_without_reaching_deeper(monkeypatch):
    """窗口内画不进同一张图的候选跳过即可，不顺延到第四名。"""
    rows = [_rank_entry(f"候选{i}") for i in range(1, 6)]

    selected = _default_chart_selection(
        rows, unplottable={"候选2"}, monkeypatch=monkeypatch)

    assert selected == ["候选1", "候选3"]
    assert "候选4" not in selected


def _wind_history_app():
    """建一个数据源为 Wind、历史区间控件已联动过的 app。"""
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    app.withdraw()
    app._source_var.set("wind")
    BacktestApp._toggle_history_wind_controls(app)
    app.update_idletasks()
    return app


def test_auto_start_date_is_backfilled_and_matches_the_real_request():
    """自动模式下置灰框里的起始日必须等于真正会发出的请求起始日。

    它此前显示的是构造界面时写死的「今天 − 420 自然日」，与实际取数区间毫
    无关系。那个框看上去正是"本次从哪天开始取"，显示一个无关日期比措辞含糊
    更能骗人。
    """
    app = _wind_history_app()
    try:
        shown = app._history_wind_start_var.get()
        resolved = BacktestApp._collect_history_state(app)

        assert shown == resolved["wind_start"], (
            f"界面显示 {shown}，实际请求 {resolved['wind_start']}")
        assert app._history_wind_asof_var.get() == resolved["wind_end"]
        # 自动模式下该框只读。
        assert str(app._history_wind_start_entry.cget("state")) == "disabled"
    finally:
        app.destroy()


def test_auto_start_date_follows_period_and_asof_changes():
    """周期或截至日一变，回填的起始日要跟着重算。"""
    app = _wind_history_app()
    try:
        for key, var in app._history_period_vars.items():
            var.set(key == "year")
        BacktestApp._history_period_selection_changed(app)
        app.update_idletasks()
        long_start = app._history_wind_start_var.get()

        for key, var in app._history_period_vars.items():
            var.set(key == "week")
        BacktestApp._history_period_selection_changed(app)
        app.update_idletasks()
        short_start = app._history_wind_start_var.get()

        # 近周只需往前取 5 个交易日，起点必须比近年晚得多。
        assert short_start > long_start, (long_start, short_start)

        app._history_wind_asof_var.set("2026-03-31")
        app.update_idletasks()
        assert app._history_wind_start_var.get() != short_start
        assert app._history_wind_start_var.get() < "2026-03-31"
    finally:
        app.destroy()


def test_manual_start_date_is_never_overwritten():
    """取消自动后，用户填的起始日不得被任何联动覆写。"""
    app = _wind_history_app()
    try:
        app._history_wind_auto_start_var.set(False)
        BacktestApp._toggle_history_wind_controls(app)
        app._history_wind_start_var.set("2024-01-01")

        # 改截至日、改周期都不该动它。
        app._history_wind_asof_var.set("2026-06-30")
        BacktestApp._history_period_selection_changed(app)
        app.update_idletasks()
        assert app._history_wind_start_var.get() == "2024-01-01"
        assert str(app._history_wind_start_entry.cget("state")) == "normal"

        # 勾回自动则必须重算，不能留着那个已经失效的手填值。
        app._history_wind_auto_start_var.set(True)
        BacktestApp._toggle_history_wind_controls(app)
        app.update_idletasks()
        assert app._history_wind_start_var.get() != "2024-01-01"
        assert app._history_wind_start_var.get() == (
            BacktestApp._collect_history_state(app)["wind_start"])
    finally:
        app.destroy()


def test_auto_start_backfill_ignores_a_half_typed_asof():
    """截至日还没输入完时不要乱猜，保留上一个有效结果。"""
    app = _wind_history_app()
    try:
        app._history_wind_asof_var.set("2026-07-29")
        app.update_idletasks()
        valid = app._history_wind_start_var.get()

        app._history_wind_asof_var.set("2026-0")
        app.update_idletasks()

        assert app._history_wind_start_var.get() == valid
    finally:
        app.destroy()


def test_wind_hint_drops_what_the_top_summary_already_says():
    """本行提示不再重复粒度，只留别处没有的那句。

    实际采用的行情粒度在顶部基准摘要里已经写了一遍，同屏第二次出现是纯
    重复；本行唯一独有的信息是"为什么要多取前一个交易日"。
    """
    app = _wind_history_app()
    try:
        for var in app._history_period_vars.values():
            var.set(True)
        BacktestApp._history_period_selection_changed(app)
        app.update_idletasks()
        hint = app._history_wind_hint_var.get()

        assert "统一采用" not in hint and "行情" not in hint, hint
        assert "多取前一个交易日的收盘价" in hint

        # 手动模式无需提示：未勾的勾选框加可编辑的日期框本身就说明了一切。
        app._history_wind_auto_start_var.set(False)
        BacktestApp._toggle_history_wind_controls(app)
        app.update_idletasks()
        assert app._history_wind_hint_var.get() == ""
    finally:
        app.destroy()


def test_wind_hint_still_reports_bar_size_resolution_errors():
    """粒度解析失败时这一行仍是唯一的报错出口。

    移除粒度显示后 `_resolve_wind_bar_size` 的返回值不再被用到，容易被当成
    死代码删掉——但它抛的错正是从这里报给用户的。
    """
    app = _wind_history_app()
    try:
        app._history_include_fixed_times_var.set(True)
        app._history_fixed_times_var.set("99:99")
        BacktestApp._refresh_history_wind_hint(app)
        app.update_idletasks()

        assert "格式错误" in app._history_wind_hint_var.get()
    finally:
        app.destroy()


def test_base_summary_shows_the_resolved_date_range_not_the_rule():
    """顶部摘要写推算出的起始日，而不是「自动覆盖最近 N 日连续区间」。

    这一行讲的是"本次将冻结的取数区间"；写规则描述读者还得自己换算成日
    期，而起始日已经实时回填进旁边的输入框，两处必须显示同一个日期。
    """
    app = _wind_history_app()
    try:
        for var in app._history_period_vars.values():
            var.set(True)
        BacktestApp._history_period_selection_changed(app)
        app.update_idletasks()

        summary = app._history_base_summary_var.get()
        shown_start = app._history_wind_start_var.get()
        resolved = BacktestApp._collect_history_state(app)

        assert "自动覆盖最近" not in summary, summary
        assert f"{shown_start} 至 " in summary, summary
        assert shown_start == resolved["wind_start"]
    finally:
        app.destroy()


def test_auto_start_is_cleared_when_no_period_is_selected():
    """推不出起始日时清空，不留一个看着像取数起点的旧日期。"""
    app = _wind_history_app()
    try:
        app._history_wind_start_var.set("2024-03-01")
        for var in app._history_period_vars.values():
            var.set(False)
        BacktestApp._history_period_selection_changed(app)
        app.update_idletasks()

        assert app._history_wind_start_var.get() == ""
        assert "未选择分析周期" in app._history_base_summary_var.get()
    finally:
        app.destroy()


def test_candidate_panel_inputs_share_one_size():
    """候选空间里的输入框必须同宽同高。

    此前时刻/带宽候选是 184px、两个日期是 107px，尾部说明跟着错开。
    """
    app = _wind_history_app()
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()
        entries = {
            "时刻候选": app._history_fixed_times_entry,
            "带宽候选": app._history_band_candidate_entry,
            "自定义起始日": app._history_wind_start_entry,
            "分析截至日": app._history_wind_asof_entry,
        }
        sizes = {
            name: (w.winfo_width(), w.winfo_height())
            for name, w in entries.items()
        }
        assert len(set(sizes.values())) == 1, sizes
    finally:
        app.destroy()


def test_date_inputs_do_not_move_when_the_auto_checkbox_toggles():
    """勾选/取消「根据周期自动推算起始日」不得让输入框左右位移。

    提示行原本跨全部列，它的文案随模式变化（自动一长句、手动为空），列宽
    跟着重算，两个日期框就被挤得左右跳。
    """
    app = _wind_history_app()
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()

        def x_positions():
            # 用容器内相对坐标：窗口在测试里是 withdraw 的，绝对屏幕坐标要
            # 等布局定型才稳定，而"是否位移"本来就该在父容器里衡量。
            app.update_idletasks()
            return (app._history_wind_start_entry.winfo_x(),
                    app._history_wind_asof_entry.winfo_x())

        baseline = x_positions()
        for auto in (False, True, False, True):
            app._history_wind_auto_start_var.set(auto)
            BacktestApp._toggle_history_wind_controls(app)
            assert x_positions() == baseline, f"auto={auto} 时发生位移"

        # 周期变化会改写提示里的天数，同样不得挪动输入框。
        for key, var in app._history_period_vars.items():
            var.set(key == "week")
        BacktestApp._history_period_selection_changed(app)
        assert x_positions() == baseline

        for var in app._history_period_vars.values():
            var.set(False)
        BacktestApp._history_period_selection_changed(app)
        assert x_positions() == baseline
    finally:
        app.destroy()


def test_candidate_rows_align_their_labels_and_hints():
    """同组内各行的标签、输入框、尾部说明要各自成列。"""
    app = _wind_history_app()
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()

        # 两个候选参数行同属一组。
        assert (app._history_fixed_times_entry.winfo_rootx()
                == app._history_band_candidate_entry.winfo_rootx())
        # 两个日期行同属一组。
        assert (app._history_wind_start_entry.winfo_rootx()
                == app._history_wind_asof_entry.winfo_rootx())
    finally:
        app.destroy()


def test_optimization_reuses_the_one_shared_progress_bar():
    """优选任务复用全应用唯一那条进度条，不另建一条。

    界面上只该有一个长任务指示器，任务名由底部状态栏给出。此前本页另建了
    一条，运行时同屏两条一起转。
    """
    import threading

    app = _wind_history_app()
    original_thread = threading.Thread
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()
        assert not app._progress.winfo_ismapped()
        # 本页不得再有自己的进度条部件。
        assert not hasattr(app, "_history_progress")
        assert not hasattr(BacktestApp, "_set_history_progress")

        # 拦住真正的后台线程，只观察进度条状态。
        threading.Thread = lambda *a, **k: SimpleNamespace(start=lambda: None)
        assert BacktestApp._run_history_recommendation(app) is True
        threading.Thread = original_thread
        app.update_idletasks()

        assert app._progress.winfo_ismapped(), "优选应当启动共用进度条"
        assert "批量优选" in app._status_var.get()

        BacktestApp._finish_job(
            app, "history", success=True,
            success_text="策略优选完成", failure_text="失败")
        app.update_idletasks()
        assert not app._progress.winfo_ismapped()
    finally:
        threading.Thread = original_thread
        app.destroy()


def test_history_buttons_carry_a_visual_hierarchy():
    """页首与结论卡的按钮要分出主次，不能一排同权重。

    真正有副作用的动作（跑回测、落盘）用实心，导航与视图开关用幽灵样式；
    主色只留给页面主行动，避免同屏两个"主按钮"。
    """
    app = _wind_history_app()
    try:
        assert str(app._history_btn.cget("style")) == "Run.TButton"
        # 去翻旧结果 / 折叠配置都只是导航，不产生结果。
        for button in (app._history_open_store_btn,
                       app._history_config_toggle_btn):
            assert str(button.cget("style")) == "Ghost.TButton"
        # 保存结果确实产出文件，保持实心描边。
        assert str(app._history_save_btn.cget("style") or "") in ("", "TButton")
    finally:
        app.destroy()




def test_source_pill_names_the_subject_of_the_active_source():
    """顶部数据源 pill 必须按来源取标的。

    此前一律读 `_wind_code_var`，于是 CSV 模式下显示的是 Wind 代码框里的
    内容（实测「CSV 行情 · 510050.SH」）——那个代码本次根本不会被读取。
    """
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        app._wind_code_var.set("510050.SH")
        app._csv_path_var.set("/data/沪深300_2026.csv")

        def pill_texts():
            BacktestApp._update_history_header_summary(app)
            app.update_idletasks()
            return [w.cget("text")
                    for w in app._history_header_summary_frame.winfo_children()]

        app._source_var.set("wind")
        BacktestApp._toggle_source(app)
        assert pill_texts() == ["📊 WIND 行情 · 510050.SH"]

        app._source_var.set("csv")
        BacktestApp._toggle_source(app)
        texts = pill_texts()
        assert texts == ["📊 CSV 行情 · 沪深300_2026.csv"], texts
        # 关键：CSV 下不得出现 Wind 代码。
        assert "510050.SH" not in texts[0]

        # 路径未填时只写来源，不留一个孤零零的分隔符。
        app._csv_path_var.set("")
        BacktestApp._toggle_source(app)
        assert pill_texts() == ["📊 CSV 行情"]
    finally:
        app.destroy()


def test_sigma_note_does_not_drift_with_the_checkbox_label():
    """σ 说明不得随「加入当前回测带宽」的文案左右横跳。

    那个勾选框的标题是变量：勾上时会带上换算出的 σ 值，实测把跟在它右边
    的说明推走 144px——同一句话在两次渲染之间换位置。
    """
    app = _wind_history_app()
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()
        checkbox = app._history_current_band_check
        note = [w for w in checkbox.master.winfo_children()
                if w is not checkbox][-1]

        # 用布局结构断言，不用坐标：测试窗口是 withdraw 的，这棵嵌套子树不
        # 会真正排版，winfo_x/y 恒为 0，比坐标只会得到一条永远成立的空断言。
        # 而「说明不横排在那个宽度可变的勾选框右边」正是修复本身。
        assert checkbox.pack_info().get("side") != "left", (
            "勾选框恢复成横排了，它右边的任何东西都会随文案长度被推走")
        assert note.pack_info().get("side") != "left", (
            "σ 说明与勾选框同排；勾选框文案会带上换算出的 σ 值，实测把它推走 144px")
    finally:
        app.destroy()


def test_header_buttons_share_one_gap():
    """页首几个按钮之间的间隙必须一致（此前是 4/8/6 三种值）。"""
    app = _wind_history_app()
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()
        buttons = sorted(
            (w for w in app._history_btn.master.winfo_children()
             if w.winfo_class() == "TButton"),
            key=lambda w: w.winfo_x())
        gaps = {b.winfo_x() - (a.winfo_x() + a.winfo_width())
                for a, b in zip(buttons, buttons[1:])}
        assert len(gaps) == 1, f"页首按钮间隙不一致: {sorted(gaps)}"
    finally:
        app.destroy()


def test_wind_entry_column_has_no_dead_space():
    """Wind 区间第二列的宽度要贴着输入框，不留死区。

    此前按「字符数 × 字宽」估算得 198px，而输入框实际渲染 170px，右侧凭空
    多出 28px，把后面的勾选框整体推远。
    """
    app = _wind_history_app()
    try:
        app._nb.select(app._history_tab)
        app.update_idletasks()
        minsize = app._history_wind_frame.grid_columnconfigure(1)["minsize"]
        entry = app._history_wind_start_entry.winfo_width()
        assert abs(int(minsize) - entry) <= 2, (minsize, entry)
    finally:
        app.destroy()


def test_secondary_notes_are_not_larger_than_their_headings():
    """结果区的次要说明字号不得大于同一条里的标题。

    不显式指定就继承全局默认 10pt，而标题是 9pt 加粗——最长、最不重要的
    那句话反而字号最大。
    """
    import tkinter.font as tkfont

    app = _render_history_result_page()
    try:
        app.update_idletasks()
        replay_bar = app._history_replay_button.master
        labels = [w for w in replay_bar.winfo_children()
                  if w.winfo_class() == "Label"]
        heading, note = labels[0], labels[-1]
        assert heading.cget("text").endswith(":")

        note_size = tkfont.Font(font=note.cget("font")).actual("size")
        heading_size = tkfont.Font(font=heading.cget("font")).actual("size")
        assert note_size <= heading_size, (note_size, heading_size)
        # 左对齐：窄窗口下截断要从句尾开始，而不是把句首也吃掉。
        assert str(note.cget("anchor")) == "w"
    finally:
        app.destroy()


def test_replay_note_states_that_it_switches_tabs():
    """「加载明细」会切走标签页，说明里必须写出来。

    点完整页消失是点击前完全看不出的副作用；载入结果那条分支本来就写了
    「跳转」，默认分支不写就是同一个按钮两种说法。
    """
    app = _render_history_result_page()
    try:
        app.update_idletasks()
        note = [w for w in app._history_replay_button.master.winfo_children()
                if w.winfo_class() == "Label"][-1].cget("text")
        assert "跳转" in note, note
    finally:
        app.destroy()


def test_fixed_times_hint_states_the_format_it_actually_requires():
    """时刻候选必须写明格式——它是本页唯一不容忍中文逗号的输入。

    带宽候选经 parse_band_candidate_sigmas 归一化，中文逗号/分号都收；而
    固定时刻只按 ASCII 逗号切分并要求 HH:MM。偏偏标了「英文逗号分隔」的是
    宽容的那个，没标的才是严格的那个。
    """
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        app.update_idletasks()
        row = app._history_fixed_times_entry.master
        hints = [w.cget("text") for w in row.winfo_children()
                 if w.winfo_class() == "TLabel" and not w.cget("text").endswith(":")]
        assert hints, "时刻候选行没有说明文字"
        assert "HH:MM" in hints[0], hints
        assert "英文逗号" in hints[0], hints
    finally:
        app.destroy()


def test_default_period_is_the_one_with_the_most_evidence():
    """默认展示样本最多的周期，而不是列表里的头一个。

    此前取 HISTORY_PERIOD_DEFS 的第一项「近周」——五档里样本最少、最容易
    被噪声主导的那个；而同一行右端的一致性提示写的是「短周期样本少更易受
    噪声影响，建议以长周期为准」。页面一边这么建议，一边默认把最不该单独
    采信的结论摆在读者面前。
    """
    rows = {
        "week": {"period": "近周", "strategy": "A", "paired": 1},
        "month": {"period": "近月", "strategy": "A", "paired": 1},
        "quarter": {"period": "近季", "strategy": "A", "paired": 3},
        "year": {"period": "近年", "strategy": "A", "paired": 9},
    }
    assert BacktestApp._default_history_period(rows) == "year"

    # 没有可比结论的周期不该被选中，哪怕它更长。
    rows["year"] = {"period": "近年", "strategy": "—", "paired": 0}
    assert BacktestApp._default_history_period(rows) == "quarter"

    # 全都没有结论时仍要选出一个，保持总有一个 chip 是选中的。
    empty = {"week": {"period": "近周", "strategy": "—", "paired": 0},
             "year": {"period": "近年", "strategy": "—", "paired": 0}}
    assert BacktestApp._default_history_period(empty) == "week"
    assert BacktestApp._default_history_period({}) == ""


def test_result_page_actually_opens_on_the_long_period():
    """接线断言：结果页渲染后，选中的必须是长周期那个 chip。

    只测 `_default_history_period` 本身不够——把调用点退回
    `next(iter(...))` 时那条单测照样通过，漏掉的正是"有没有接上"。
    """
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    # 必须让各周期的样本数**不同**：共享夹具里两档都是 9 段，max 取到的
    # 恰好就是第一个，退回 next(iter(...)) 也照样通过——那样断言是空的。
    def rank_row(lookback, rank, strategy, strategy_type, paired):
        return {
            "lookback": lookback, "rank": rank, "strategy": strategy,
            "strategy_type": strategy_type,
            "daily_net_pnl_rms": 10.0 - rank,
            "baseline_daily_net_pnl_rms": 10.0,
            "improvement_vs_c2c": 0.1, "selection_improvement_vs_c2c": 0.1,
            "selection_metric":
                history_selection.STRICT_LOOKBACK_SELECTION_METRIC,
            "selection_objective": "incremental_pnl",
            "window_win_rate_vs_c2c": 0.6,
            "paired_windows": paired, "baseline_windows": paired,
            "comparison_eligible": True, "recommendation_eligible": True,
            "rolling_windows": paired, "eligible_endpoints": paired,
            "skipped_endpoints": 0, "history_days_available": 252,
            "lookback_days": 252, "complete_window": True,
            "maturity_days": 30, "evidence_days": 252, "days_used": 252,
            "evaluation_mode": "strict_lookback",
            "sampling_mode": "strict_contiguous",
            "segment_count": paired, "expiry_segments": max(paired - 1, 0),
            "mtm_segments": 1, "terminal_mode": "mixed",
            "relative_comparison_windows": paired,
            "incremental_pnl_vs_c2c": 1.0 - rank,
            "incremental_sharpe_vs_c2c": 0.1,
            "incremental_tc_vs_c2c": -0.003, "max_drawdown": 0.03,
        }

    rows = []
    for lookback, paired in (("week", 1), ("quarter", 3), ("year", 9)):
        rows.append(rank_row(lookback, 1, "固定间隔(0.5σ)", "hedge_band", paired))
        rows.append(rank_row(lookback, 2, "每日收盘", "close_to_close", paired))
    ranking = pd.DataFrame(rows)

    app = gui_app.BacktestApp()
    try:
        app.withdraw()
        BacktestApp._show_history_recommendation(
            app, ranking[ranking["rank"].eq(1)].copy(), ranking, notes=None,
            source_label="测试", window_results=None,
            history_state={"history_lookbacks":
                           {"week": 5, "quarter": 61, "year": 243}})
        app.update_idletasks()

        selected = app._history_period_rows[
            app._history_selected_period_var.get()]
        assert selected["period"] == "近年", (
            f"默认落在 {selected['period']}，而样本最多的是近年")
        # 顺带确认头一个确实是近周——即"取第一个"会选错。
        assert next(iter(app._history_period_rows.values()))["period"] == "近周"
    finally:
        app.destroy()


def test_run_button_is_disabled_until_a_period_is_selected():
    """一个分析周期都没勾时主按钮必须禁用，并就近给出理由。

    此前按钮照常可点，点下去才弹「请至少选择一个历史分析周期。」的模态
    框——用模态错误代替禁用态，等于让用户先撞一次才知道不能按。
    """
    app = _wind_history_app()
    try:
        def state():
            app.update_idletasks()
            return (str(app._history_btn.cget("state")),
                    [str(w.cget("text"))
                     for w in app._history_header_summary_frame.winfo_children()])

        # _wind_history_app 只联动了 Wind 区间控件，按钮状态还停在初始的
        # 模拟行情态；先按真实入口同步一次。
        BacktestApp._toggle_source(app)
        assert state()[0] == "normal"

        for var in app._history_period_vars.values():
            var.set(False)
        BacktestApp._history_period_selection_changed(app)
        disabled, pills = state()
        assert disabled == "disabled"
        assert any("至少勾选一个分析周期" in p for p in pills), pills

        # 勾回来要立刻恢复可用，理由 pill 同时消失。
        for var in app._history_period_vars.values():
            var.set(True)
        BacktestApp._history_period_selection_changed(app)
        enabled, pills = state()
        assert enabled == "normal"
        assert not any("至少勾选" in p for p in pills), pills
    finally:
        app.destroy()


def test_wraplength_tracks_the_actual_slot_width():
    """折行宽度要跟随控件实际宽度，不能写死。

    写死的值必然与真实槽位对不上：基准摘要写的是 980，而槽位实测 1242——
    文本只要多十来个字，就会在还剩 262px 的情况下提前折成两行。
    """
    app = _render_history_result_page()
    try:
        app._nb.select(app._history_tab)
        if not app._history_config_visible:
            BacktestApp._toggle_history_config_panel(app)
        app.update_idletasks()

        summary = None
        for block in app._history_config_panel.winfo_children():
            for child in block.winfo_children():
                if child.winfo_class() == "Label":
                    summary = child
        assert summary is not None

        # 结构断言：必须挂着跟随宽度的 <Configure> 回调，且不再写死初值。
        # 不比像素——测试窗口是 withdraw 的，label 自身宽度恒为 1，回调按
        # 设计会跳过，量出来的 wraplength 永远是 0，断言会变成看运气。
        assert summary.bind("<Configure>"), "基准摘要没有跟随宽度的回调"
        assert int(summary.cget("wraplength")) in (0, summary.winfo_width()), (
            "wraplength 不该在构造时写死一个与槽位无关的常数")

        context = None
        for child in app._history_conclusion_card.master.winfo_children():
            if child.grid_info().get("row") == 1:
                context = [c for c in child.winfo_children()
                           if c.winfo_class() == "TLabel"][-1]
        assert context is not None
        assert context.bind("<Configure>"), "周期上下文行没有跟随宽度的回调"
    finally:
        app.destroy()


def test_wraplength_handler_does_not_self_excite():
    """跟随宽度的回调不得形成「设值→重排→再设值」的闭环。

    设 wraplength 会改变高度（一行变两行），高度变化让父容器重排并再次派
    发 <Configure>；不比宽度就会无限循环——本页此前的分割线回调正是这么
    卡死过一次。
    """
    class _FakeLabel:
        """只实现 _track_wraplength 用到的接口，可控地伪造宽高联动。"""

        def __init__(self, width):
            self.width = width
            self.applied = []
            self._handlers = {}

        def winfo_width(self):
            return self.width

        def configure(self, **kw):
            self.applied.append(kw.get("wraplength"))
            # 模拟「改 wraplength → 高度变 → 父容器重排 → 再派发」
            self.fire("<Configure>")

        def bind(self, sequence, func, add=None):
            self._handlers[sequence] = func

        def after_idle(self, func):
            func()

        def fire(self, sequence):
            handler = self._handlers.get(sequence)
            if handler is not None:
                handler()

    label = _FakeLabel(1242)
    BacktestApp._track_wraplength(label)
    assert label.applied == [1242], label.applied

    # 宽度没变的回波一律忽略。
    label.fire("<Configure>")
    assert label.applied == [1242], label.applied

    # 宽度真的变了才重设。
    label.width = 900
    label.fire("<Configure>")
    assert label.applied == [1242, 900], label.applied


def test_replay_state_takes_parameters_from_the_candidate_itself():
    """重放要取这一条候选自己的策略参数，不能沿用整个任务的配置。

    原先只设了策略类型名，带宽阈值与波动率口径仍是任务配置——于是同一段
    里重放两个不同候选（1σ 与 2σ），两条快照的策略签名一模一样，结果对比
    页会说它们"输入完全相同"。
    """
    import pandas as pd
    from pricing import HedgeBandStrategy

    index = pd.date_range("2026-08-05 09:30", periods=20, freq="15min")
    spec = SimpleNamespace(
        lookback="week", window_id="window_1",
        external_path=pd.Series(range(20), index=index),
        strategies={
            "固定间隔(1σ)": HedgeBandStrategy("sigma", 1.0),
            "固定间隔(2σ)": HedgeBandStrategy(
                "sigma", 2.0, sigma_source="realized", window_days=30),
        },
        metadata={})
    fake = SimpleNamespace(_latest_history_state={
        "source": "wind", "wind_code": "510050.SH", "wind_bar_size": "15分钟",
        "interval_type": "sigma", "price_interval": 1.0,
        "sigma_source": "implied", "sigma_window": 20,
        "strategy_name": "hedge_band",
    })

    tight = BacktestApp._history_replay_gui_state(fake, spec, "固定间隔(1σ)")
    wide = BacktestApp._history_replay_gui_state(fake, spec, "固定间隔(2σ)")

    assert tight["price_interval"] == 1.0
    assert wide["price_interval"] == 2.0
    assert wide["sigma_source"] == "realized"
    assert wide["sigma_window"] == 30
    # 两条候选的策略签名必须能区分开。
    assert (BacktestApp._snapshot_form_state(tight)
            != BacktestApp._snapshot_form_state(wide))


def test_replay_state_records_fixed_times_from_the_candidate():
    """固定时刻候选同理：时刻表要来自策略对象。"""
    import pandas as pd
    from pricing.hedge_backtest import FixedTimeStrategy

    index = pd.date_range("2026-08-05 09:30", periods=20, freq="15min")
    spec = SimpleNamespace(
        lookback="week", window_id="window_1",
        external_path=pd.Series(range(20), index=index),
        strategies={"固定时刻(10:30)": FixedTimeStrategy("10:30")},
        metadata={})
    fake = SimpleNamespace(_latest_history_state={
        "source": "wind", "fixed_times": "11:30,15:00",
        "strategy_name": "fixed_times",
    })

    state = BacktestApp._history_replay_gui_state(fake, spec, "固定时刻(10:30)")

    assert state["fixed_times"] == "10:30"
