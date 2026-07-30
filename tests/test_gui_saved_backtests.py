from __future__ import annotations

import copy
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gui_app
import history_selection
from gui_app import BacktestApp
from pricing import HedgeBandStrategy


def _result(*, strategy="close_to_close", price_shift=0.0):
    return {
        "net_daily": np.array([-1.0, 2.0, 1.0]),
        "tc_paid": np.array([1.0, 0.2, 0.1]),
        "steps_per_day": 1,
        "hedge_triggered": np.array([True, True, True]),
        "shares": np.array([1.0, 0.5, 0.0]),
        "prices": np.array([100.0, 101.0, 102.0]) + price_shift,
        "strategy_name": strategy,
        "hedging_error": 2.5,
        "trading_day_groups": np.array([0, 1, 2]),
        # 不属于轻量快照，验证保存时不会把完整结果无差别复制进去。
        "gamma": np.array([9.0, 8.0, 7.0]),
    }


def _state(*, strategy="close_to_close", threshold=1.0,
           force_day_close_hedge=False, position=-1):
    return {
        "cls_name": "香草期权 (Vanilla)",
        "subtype": "Eu",
        "params": {"s0": 100.0, "K": 100.0, "sigma": 0.2, "T_days": 2},
        "source": "simulate",
        "seed": "42",
        "strategy_name": strategy,
        "fixed_times": "11:30,15:00",
        "interval_type": "absolute",
        "price_interval": threshold,
        "position": position,
        "quantity": 2.0,
        "multiplier": 100.0,
        "tc_rate": 0.001,
        "slippage_bps": 1.5,
        "force_day_close_hedge": force_day_close_hedge,
    }


def _snapshot(result_id="result-0001", name="结果 A", *, state=None,
              result=None, timestamps=None):
    state = _state() if state is None else state
    result = _result(strategy=state["strategy_name"]) if result is None else result
    bt = SimpleNamespace(
        _results=result,
        prices=result["prices"],
        timestamps=(pd.DatetimeIndex([
            "2026-01-02", "2026-01-05", "2026-01-06",
        ]) if timestamps is None else timestamps),
    )
    return BacktestApp._make_saved_backtest_result(
        bt, state, result_id, name,
        saved_at=datetime.datetime(2026, 7, 15, 12, 0),
    )


def test_saved_backtest_is_compact_deep_copy_with_run_time_parameters():
    state = _state(strategy="hedge_band", threshold=2.0)
    original = _result(strategy="hedge_band")
    snapshot = _snapshot(state=state, result=original)

    original["net_daily"][0] = 999.0
    state["params"]["K"] = 120.0

    assert snapshot.daily_frame["net_pnl"].sum() == pytest.approx(2.0)
    assert snapshot.summary_row["total_net_pnl"] == pytest.approx(2.0)
    assert all(
        not isinstance(value, np.ndarray)
        for value in snapshot.summary_row.values())
    assert snapshot.strategy_label == "固定间隔"
    assert snapshot.position == -1
    assert "绝对 2" in snapshot.parameter_summary
    assert "相对 2.0000%" in snapshot.parameter_summary
    assert snapshot.source_label == "模拟 · seed 42"


def test_saved_snapshot_rejects_result_direction_mismatching_left_state():
    state = _state(position=-1)
    result = _result()
    result["position"] = 1

    with pytest.raises(ValueError, match="头寸方向.*不一致"):
        _snapshot(state=state, result=result)


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(False, "收盘兜底：关闭"), (True, "收盘兜底：开启")],
)
def test_saved_snapshot_parameter_summary_records_close_fallback_rule(
        enabled, expected):
    snapshot = _snapshot(state=_state(
        strategy="hedge_band", threshold=2.0,
        force_day_close_hedge=enabled,
    ), result=_result(strategy="hedge_band"))

    assert expected in snapshot.parameter_summary


def test_saved_wind_snapshot_records_resolved_dates_and_actual_bar_size():
    state = _state(strategy="hedge_band", threshold=1.5)
    state.update({
        "source": "wind",
        "wind_code": "510050.SH",
        "wind_start": "2025-01-02",
        "wind_end": "2025-02-28",
        "wind_bar_size_requested": gui_app.WIND_AUTO_BAR_SIZE,
        "wind_bar_size": "15分钟",
        "wind_date_mode": "custom_range",
    })

    snapshot = _snapshot(
        state=state, result=_result(strategy="hedge_band"))

    assert snapshot.source_label == (
        "Wind · 510050.SH · 2025-01-02 至 2025-02-28 · 15分钟"
    )


def test_saved_payload_uses_cached_results_without_running_backtest(monkeypatch):
    first = _snapshot()
    second_state = _state(strategy="hedge_band", threshold=2.0)
    second = _snapshot(
        "result-0002", "结果 B", state=second_state,
        result=_result(strategy="hedge_band"),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("选择保存结果时不得重新运行回测")

    monkeypatch.setattr(gui_app, "compare_strategies", unexpected)
    monkeypatch.setattr(gui_app.HedgeBacktest, "run", unexpected)
    monkeypatch.setattr(gui_app, "summarize_strategy_result", unexpected)
    monkeypatch.setattr(gui_app, "result_daily_frame", unexpected)

    summary, daily_curves = BacktestApp._saved_comparison_payload([first, second])

    assert set(daily_curves) == {"结果 A", "结果 B"}
    assert summary["rank"].tolist() == [1, 2]
    assert set(summary["strategy"]) == {"结果 A", "结果 B"}
    assert set(summary["actual_trade_count"]) == {3}


def test_saved_payload_keeps_score_ranking_but_marks_close_to_close_baseline():
    baseline = _snapshot("result-0001", "每日收盘 · 基准")
    candidate = _snapshot(
        "result-0002", "固定间隔 · 更优",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )
    baseline.summary_row["score"] = 12.0
    candidate.summary_row["score"] = 8.0

    summary, _curves = BacktestApp._saved_comparison_payload(
        [candidate, baseline])
    headline = BacktestApp._comparison_headline(summary)
    fixed = BacktestApp._comparison_baseline(summary)

    assert headline["best"]["meta_result_id"] == candidate.result_id
    assert summary.iloc[0]["meta_result_id"] == candidate.result_id
    assert fixed["meta_result_id"] == baseline.result_id
    assert fixed["score"] == pytest.approx(12.0)

    baseline.name = "已重命名的每日策略"
    candidate.name = "close-to-close（只是名称）"
    renamed, _curves = BacktestApp._saved_comparison_payload(
        [candidate, baseline])
    assert BacktestApp._comparison_baseline(
        renamed)["meta_result_id"] == baseline.result_id


def test_multiple_close_to_close_results_use_oldest_stable_id_as_baseline():
    first = _snapshot("result-0001", "较晚显示名称")
    second = _snapshot("result-0002", "较早显示名称")
    first.summary_row["score"] = 15.0
    second.summary_row["score"] = 5.0
    second.saved_at = first.saved_at - datetime.timedelta(days=1)

    summary, _curves = BacktestApp._saved_comparison_payload([second, first])
    baseline = BacktestApp._comparison_baseline(summary)
    warnings = BacktestApp._saved_comparison_warnings([second, first])

    assert baseline["meta_result_id"] == "result-0001"
    assert summary.iloc[0]["meta_result_id"] == "result-0002"
    assert any("多条 close-to-close" in warning for warning in warnings)
    assert any("较晚显示名称" in warning for warning in warnings)


def test_missing_close_to_close_keeps_absolute_ranking_without_fake_baseline():
    band = _snapshot(
        "result-0001", "固定间隔",
        state=_state(strategy="hedge_band"),
        result=_result(strategy="hedge_band"),
    )
    fixed = _snapshot(
        "result-0002", "固定时刻",
        state=_state(strategy="fixed_times"),
        result=_result(strategy="fixed_times"),
    )

    summary, _curves = BacktestApp._saved_comparison_payload([band, fixed])

    assert BacktestApp._comparison_headline(summary)["best"] is not None
    assert BacktestApp._comparison_baseline(summary) is None
    assert not summary["meta_is_comparison_baseline"].any()
    assert any(
        "未包含 close-to-close 基准" in warning
        for warning in BacktestApp._saved_comparison_warnings([band, fixed])
    )


def test_path_signature_ignores_timestamps_after_early_termination():
    state = _state()
    result = _result()
    short = SimpleNamespace(
        _results=result,
        prices=result["prices"],
        timestamps=pd.DatetimeIndex([
            "2026-01-02", "2026-01-05", "2026-01-06",
        ]),
    )
    with_unused_tail = SimpleNamespace(
        _results=result,
        prices=result["prices"],
        timestamps=pd.DatetimeIndex([
            "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
        ]),
    )

    assert BacktestApp._backtest_path_key(
        short, state) == BacktestApp._backtest_path_key(with_unused_tail, state)


def test_strategy_parameters_are_comparable_but_context_differences_warn():
    first = _snapshot()
    band = _snapshot(
        "result-0002", "带宽",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )
    assert BacktestApp._saved_comparison_warnings([first, band]) == []

    different_path = copy.deepcopy(band)
    different_path.path_key = ("simulate", 3, "different")
    path_warnings = BacktestApp._saved_comparison_warnings(
        [first, different_path])
    assert any("行情路径" in warning for warning in path_warnings)

    different_contract = copy.deepcopy(band)
    different_contract.contract_key = ("different-contract",)
    contract_warnings = BacktestApp._saved_comparison_warnings(
        [first, different_contract])
    assert any("期权结构参数" in warning for warning in contract_warnings)

    different_economics = copy.deepcopy(band)
    different_economics.economics_key = ("different-economics",)
    economics_warnings = BacktestApp._saved_comparison_warnings(
        [first, different_economics])
    assert any("头寸" in warning for warning in economics_warnings)


def test_rename_does_not_change_tied_result_identity_or_rank():
    first = _snapshot("result-0001", "Z 结果")
    second = _snapshot("result-0002", "A 结果")

    before, _curves = BacktestApp._saved_comparison_payload([first, second])
    before_best = BacktestApp._comparison_headline(before)["best"]
    assert before_best["meta_result_id"] == "result-0001"

    first.name = "ZZZ"
    second.name = "AAA"
    after, _curves = BacktestApp._saved_comparison_payload([first, second])
    after_best = BacktestApp._comparison_headline(after)["best"]

    assert after_best["meta_result_id"] == "result-0001"
    assert after_best["strategy"] == "ZZZ"
    assert after.iloc[0]["meta_result_id"] == "result-0001"


def test_rename_and_delete_use_stable_ids_and_keep_selection_consistent():
    first = _snapshot()
    second = _snapshot("result-0002", "结果 B")
    fake = SimpleNamespace(
        _saved_backtests={first.result_id: first, second.result_id: second},
        _saved_comparison_selection={first.result_id, second.result_id},
        _latest_retained_result_id=first.result_id,
        _latest_backtest=SimpleNamespace(),
        _active_job=None,
        _compare_btn=None,
        _retain_btn=None,
    )

    renamed = BacktestApp._rename_saved_backtest(fake, first.result_id, "新名称")
    assert renamed.result_id == "result-0001"
    assert renamed.name == "新名称"
    assert first.result_id in fake._saved_comparison_selection
    with pytest.raises(ValueError, match="已存在"):
        BacktestApp._rename_saved_backtest(fake, first.result_id, "结果 B")

    deleted = BacktestApp._delete_saved_backtest(fake, first.result_id)
    assert deleted.name == "新名称"
    assert first.result_id not in fake._saved_backtests
    assert first.result_id not in fake._saved_comparison_selection
    assert fake._saved_comparison_baseline_id == second.result_id
    assert second.result_id in fake._saved_comparison_selection
    assert fake._latest_retained_result_id is None


def test_selected_candidates_automatically_keep_fixed_baseline_in_view():
    baseline = _snapshot("result-0001", "每日收盘")
    candidate = _snapshot(
        "result-0002", "固定间隔",
        state=_state(strategy="hedge_band"),
        result=_result(strategy="hedge_band"),
    )
    statuses = []
    fake = SimpleNamespace(
        _saved_backtests={
            baseline.result_id: baseline,
            candidate.result_id: candidate,
        },
        _saved_comparison_selection={candidate.result_id},
        _saved_comparison_baseline_id=None,
        _set_status=statuses.append,
        _refresh_saved_pool_tree=lambda: None,
        _refresh_saved_comparison_view=lambda: None,
    )

    selected = BacktestApp._selected_saved_backtests(fake)
    assert [snapshot.result_id for snapshot in selected] == [
        baseline.result_id, candidate.result_id,
    ]
    assert fake._saved_comparison_selection == {
        baseline.result_id, candidate.result_id,
    }

    BacktestApp._toggle_saved_backtest_selection(fake, baseline.result_id)
    assert fake._saved_comparison_selection == {
        baseline.result_id, candidate.result_id,
    }
    assert statuses and "固定基准" in statuses[-1]


def test_saved_payload_rejects_cross_position_baseline_and_candidate():
    sell_baseline = _snapshot(
        "result-0001", "卖方每日收盘",
        state=_state(position=1),
    )
    buy_candidate = _snapshot(
        "result-0002", "买方固定间隔",
        state=_state(
            strategy="hedge_band", threshold=2.0, position=-1),
        result=_result(strategy="hedge_band"),
    )

    with pytest.raises(ValueError, match="不能混选"):
        BacktestApp._saved_comparison_payload(
            [sell_baseline, buy_candidate],
            baseline_result_id=sell_baseline.result_id,
        )
    assert any(
        "不能混选" in warning
        for warning in BacktestApp._saved_comparison_warnings(
            [sell_baseline, buy_candidate],
            baseline_result_id=sell_baseline.result_id,
        )
    )


def test_saved_selection_switches_position_group_and_uses_matching_baseline():
    sell_baseline = _snapshot(
        "result-0001", "卖方每日收盘",
        state=_state(position=1),
    )
    buy_baseline = _snapshot(
        "result-0002", "买方每日收盘",
        state=_state(position=-1),
    )
    buy_candidate = _snapshot(
        "result-0003", "买方固定间隔",
        state=_state(strategy="hedge_band", position=-1),
        result=_result(strategy="hedge_band"),
    )
    statuses = []
    fake = SimpleNamespace(
        _saved_backtests={
            sell_baseline.result_id: sell_baseline,
            buy_baseline.result_id: buy_baseline,
            buy_candidate.result_id: buy_candidate,
        },
        _saved_comparison_selection={sell_baseline.result_id},
        _saved_comparison_baseline_id=sell_baseline.result_id,
        _set_status=statuses.append,
        _refresh_saved_pool_tree=lambda: None,
        _refresh_saved_comparison_view=lambda: None,
    )

    BacktestApp._toggle_saved_backtest_selection(
        fake, buy_candidate.result_id)
    selected = BacktestApp._selected_saved_backtests(fake)

    assert {
        snapshot.result_id for snapshot in selected
    } == {buy_baseline.result_id, buy_candidate.result_id}
    assert sell_baseline.result_id not in fake._saved_comparison_selection
    assert fake._saved_comparison_baseline_id == buy_baseline.result_id
    assert statuses and "不混合比较" in statuses[-1]


def test_busy_result_pool_actions_do_not_touch_selection_or_open_dialogs(
        monkeypatch):
    first = _snapshot()
    messages = []
    fake = SimpleNamespace(
        _active_job="history",
        _saved_backtests={first.result_id: first},
        _saved_comparison_selection=set(),
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showinfo",
        lambda title, message: messages.append((title, message)),
    )
    monkeypatch.setattr(
        gui_app.simpledialog, "askstring",
        lambda *_args, **_kwargs: pytest.fail("忙碌时不应打开重命名对话框"),
    )
    monkeypatch.setattr(
        gui_app.messagebox, "askyesno",
        lambda *_args, **_kwargs: pytest.fail("忙碌时不应打开删除对话框"),
    )

    BacktestApp._select_all_saved_backtests(fake)
    BacktestApp._clear_saved_backtest_selection(fake)
    BacktestApp._prompt_rename_saved_backtest(fake)
    BacktestApp._prompt_delete_saved_backtest(fake)

    assert fake._saved_comparison_selection == set()
    assert len(fake._saved_backtests) == 1
    assert len(messages) == 4


def test_backtest_delivery_only_marks_rendered_result_as_latest(monkeypatch):
    finished = []
    errors = []
    bt = SimpleNamespace()
    state = {"cfg": object(), "params": {"K": 100.0}}
    fake = SimpleNamespace(
        _show_results=lambda *_args: None,
        _finish_run=finished.append,
        _latest_backtest=None,
        _latest_backtest_state=None,
        _latest_retained_result_id="old",
    )

    BacktestApp._deliver_backtest_result(fake, bt, None, state)

    assert finished == [True]
    assert fake._latest_backtest is bt
    assert fake._latest_backtest_state == {"params": {"K": 100.0}}
    assert fake._latest_retained_result_id is None

    def fail_render(*_args):
        raise RuntimeError("render failed")

    failed = SimpleNamespace(
        _show_results=fail_render,
        _finish_run=finished.append,
        _latest_backtest="previous",
        _latest_backtest_state={"previous": True},
        _latest_retained_result_id="old",
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: errors.append((title, message)),
    )

    BacktestApp._deliver_backtest_result(failed, bt, None, state)

    assert finished[-1] is False
    assert failed._latest_backtest == "previous"
    assert failed._latest_backtest_state == {"previous": True}
    assert failed._latest_retained_result_id == "old"
    assert errors and errors[-1][0] == "回测结果展示失败"


def test_history_validation_delivery_auto_retain_uses_completed_result_only(
        monkeypatch):
    stored_names = []
    finished = []
    bt = SimpleNamespace()
    state = {"source": "csv", "strategy_name": "close_to_close"}

    def unexpected(*_args, **_kwargs):
        raise AssertionError("自动入池不得重新运行回测或历史择优")

    monkeypatch.setattr(gui_app.HedgeBacktest, "run", unexpected)
    monkeypatch.setattr(gui_app, "compare_strategies", unexpected)
    monkeypatch.setattr(gui_app, "recommend_by_rolling_history", unexpected)
    stored_kwargs = []
    fake = SimpleNamespace(
        _pending_history_retain_name="历史验证 · 每日收盘",
        _pending_history_retain_origin={
            "lookback": "month", "period_label": "近月", "batch": False},
        _show_results=lambda *_args: None,
        _finish_run=finished.append,
        _store_current_backtest=lambda name, **kwargs: (
            stored_names.append(name), stored_kwargs.append(kwargs)),
        _latest_backtest=None,
        _latest_backtest_state=None,
        _latest_retained_result_id=None,
        _saved_backtest_sequence=3,
        _saved_backtests={},
    )

    BacktestApp._deliver_backtest_result(fake, bt, None, state)

    assert finished == [True]
    assert fake._latest_backtest is bt
    assert fake._latest_backtest_state == state
    assert fake._pending_history_retain_name is None
    assert fake._pending_history_retain_origin is None
    assert stored_names == ["历史验证 · 每日收盘 #04"]
    # 自动入池的快照必须带上结构化来源，重命名后仍能追溯到优选周期。
    assert stored_kwargs == [{
        "origin": gui_app.SNAPSHOT_ORIGIN_HISTORY_VERIFY,
        "origin_meta": {
            "lookback": "month", "period_label": "近月", "batch": False},
    }]


def test_failed_history_validation_render_does_not_retain(monkeypatch):
    finished = []
    errors = []
    fake = SimpleNamespace(
        _pending_history_retain_name="历史验证 · 每日收盘",
        _pending_history_retain_origin={"period_label": "近月"},
        _show_results=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("render failed")),
        _finish_run=finished.append,
        _store_current_backtest=lambda _name, **_kwargs: pytest.fail(
            "展示失败的结果不得进入对比池"),
        _latest_backtest="previous",
        _latest_backtest_state={"previous": True},
        _latest_retained_result_id="previous-result",
        _saved_backtest_sequence=0,
        _saved_backtests={},
    )
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda title, message: errors.append((title, message)),
    )

    BacktestApp._deliver_backtest_result(
        fake, SimpleNamespace(), None,
        {"source": "csv", "strategy_name": "close_to_close"},
    )

    assert finished == [False]
    assert fake._latest_backtest == "previous"
    assert fake._latest_backtest_state == {"previous": True}
    assert fake._latest_retained_result_id == "previous-result"
    assert fake._pending_history_retain_name is None
    assert fake._pending_history_retain_origin is None
    assert errors and errors[0][0] == "回测结果展示失败"


# ---------------------------------------------------------------------------
#  结果对比 与 策略优选 的衔接：改善方向、跨页配色、结构化来源
# ---------------------------------------------------------------------------


def test_comparison_improvement_is_positive_when_beating_the_baseline():
    """对比页排名列与策略优选页同名指标必须同向：正值表示优于基准。"""
    improvement = BacktestApp._comparison_improvement_vs_baseline(8.0, 10.0)

    assert improvement == pytest.approx(0.2)
    # 比基准差时为负；与基准持平为 0。
    assert BacktestApp._comparison_improvement_vs_baseline(
        12.0, 10.0) == pytest.approx(-0.2)
    assert BacktestApp._comparison_improvement_vs_baseline(
        10.0, 10.0) == pytest.approx(0.0)


@pytest.mark.parametrize("score,baseline", [
    (None, 10.0), (8.0, None), (8.0, 0.0),
    (float("nan"), 10.0), (8.0, float("inf")),
])
def test_comparison_improvement_is_undefined_without_a_usable_baseline(
        score, baseline):
    assert BacktestApp._comparison_improvement_vs_baseline(
        score, baseline) is None


def test_comparison_improvement_matches_history_page_direction():
    """同一对 RMS 在两页得到同符号结论，避免用户反着读。"""
    candidate_rms, baseline_rms = 8.0, 10.0
    comparison = BacktestApp._comparison_improvement_vs_baseline(
        candidate_rms, baseline_rms)
    history = history_selection.row_improvement({
        "score": candidate_rms, "baseline_score": baseline_rms,
        "strategy_type": "hedge_band",
    })

    assert comparison > 0 and history > 0


def test_snapshot_and_history_row_share_one_style_key_per_strategy():
    """同一策略在两页必须取到同一配色键，否则曲线无法跨页对应。"""
    band_state = _state(strategy="hedge_band", threshold=1.0)
    band_state["interval_type"] = "sigma"
    snapshot_key = BacktestApp._snapshot_style_key(band_state)
    history_key = BacktestApp._history_row_style_key({
        "meta_strategy_name": "hedge_band", "meta_candidate_sigma": 1.0,
    })

    assert snapshot_key == history_key == "hedge_band:1σ"

    fixed_state = _state(strategy="fixed_times")
    fixed_state["fixed_times"] = "11:30,15:00"
    assert BacktestApp._snapshot_style_key(fixed_state) == (
        BacktestApp._history_row_style_key({
            "meta_strategy_name": "fixed_times",
            "meta_fixed_times": " 11:30 , 15:00 ",
        }))

    assert BacktestApp._snapshot_style_key(
        _state(strategy="close_to_close")) == (
        BacktestApp._history_row_style_key({
            "meta_strategy_name": "close_to_close"}))


def test_absolute_band_input_shares_color_with_equivalent_sigma_candidate():
    """绝对 / 相对带宽换算到同一 σ 时共享配色，不同带宽仍然分色。"""
    state = _state(strategy="hedge_band", threshold=1.0)
    state["interval_type"] = "sigma"
    sigma_key = BacktestApp._snapshot_style_key(state)

    params = state["params"]
    absolute = HedgeBandStrategy.convert_threshold(
        1.0, "sigma", float(params["s0"]), float(params["sigma"]),
    )["absolute"]
    absolute_state = _state(strategy="hedge_band", threshold=absolute)
    absolute_state["interval_type"] = "absolute"

    assert BacktestApp._snapshot_style_key(absolute_state) == sigma_key

    other = _state(strategy="hedge_band", threshold=2.0)
    other["interval_type"] = "sigma"
    assert BacktestApp._snapshot_style_key(other) != sigma_key


def test_strategy_style_registry_is_stable_and_pins_the_baseline_color():
    """基准配色不随登记顺序漂移，同一键重复取值不变。"""
    first = object.__new__(BacktestApp)
    first._strategy_style_registry = None
    baseline_style = first._strategy_style("close_to_close")
    band_style = first._strategy_style("hedge_band:1σ")

    assert baseline_style != band_style
    assert first._strategy_style("close_to_close") == baseline_style

    # 另一个会话先登记候选，基准仍然拿到同一个颜色。
    second = object.__new__(BacktestApp)
    second._strategy_style_registry = None
    second._strategy_style("hedge_band:1σ")

    assert second._strategy_style("close_to_close") == baseline_style
    assert second._strategy_style("hedge_band:1σ") == band_style


def test_saved_snapshot_defaults_to_manual_origin_with_stable_style_key():
    snapshot = _snapshot(state=_state(strategy="hedge_band", threshold=1.0))

    assert snapshot.origin == gui_app.SNAPSHOT_ORIGIN_MANUAL
    assert snapshot.origin_meta == {}
    assert snapshot.style_key.startswith("hedge_band:")
    assert BacktestApp._saved_snapshot_origin_label(snapshot) == "手工回测"


def test_saved_snapshot_origin_survives_rename_and_names_its_period():
    """来源是结构化字段，改名不会像旧的名称前缀那样丢失。"""
    bt = SimpleNamespace(
        _results=_result(), prices=_result()["prices"],
        timestamps=pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    snapshot = BacktestApp._make_saved_backtest_result(
        bt, _state(), "result-0007", "优选近月 · 每日收盘基准",
        origin=gui_app.SNAPSHOT_ORIGIN_HISTORY_VERIFY,
        origin_meta={
            "period_label": "近月", "lookback": "month",
            "history_rank": 1, "batch": True,
        },
    )

    assert BacktestApp._saved_snapshot_origin_label(snapshot) == "优选验证·近月"

    snapshot.name = "随便改个名字"
    assert snapshot.origin == gui_app.SNAPSHOT_ORIGIN_HISTORY_VERIFY
    assert snapshot.origin_meta["history_rank"] == 1
    assert BacktestApp._saved_snapshot_origin_label(snapshot) == "优选验证·近月"


def test_snapshot_origin_meta_is_decoupled_from_the_caller_dict():
    meta = {"period_label": "近月"}
    bt = SimpleNamespace(
        _results=_result(), prices=_result()["prices"],
        timestamps=pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    snapshot = BacktestApp._make_saved_backtest_result(
        bt, _state(), "result-0008", "结果",
        origin=gui_app.SNAPSHOT_ORIGIN_HISTORY_VERIFY, origin_meta=meta)

    meta["period_label"] = "近年"

    assert snapshot.origin_meta == {"period_label": "近月"}


def test_replay_state_marks_snapshot_as_segment_replay():
    """手工保留重放结果时来源应是分段重放，而不是笼统的手工回测。"""
    state = _state()
    state.update({
        "history_replay_strategy": "固定间隔(1.5σ)",
        "history_replay_lookback": "quarter",
        "history_replay_window_id": "w-3",
    })

    origin, meta = BacktestApp._snapshot_origin_from_state(state)

    assert origin == gui_app.SNAPSHOT_ORIGIN_HISTORY_REPLAY
    assert meta["period_label"] == "近季"
    assert meta["history_strategy"] == "固定间隔(1.5σ)"
    assert meta["window_id"] == "w-3"

    plain_origin, plain_meta = BacktestApp._snapshot_origin_from_state(
        _state())
    assert plain_origin == gui_app.SNAPSHOT_ORIGIN_MANUAL
    assert plain_meta == {}


def test_saved_payload_exposes_style_key_and_origin_for_the_chart():
    """曲线配色必须走稳定的策略身份键，而不是用户可改的结果名。"""
    baseline = _snapshot("result-0001", "每日收盘 · 基准")
    candidate = _snapshot(
        "result-0002", "固定间隔 · 更优",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )

    summary, _curves = BacktestApp._saved_comparison_payload(
        [baseline, candidate])
    keys = dict(zip(summary["strategy"], summary["meta_style_key"]))

    assert keys["每日收盘 · 基准"] == "close_to_close"
    assert keys["固定间隔 · 更优"].startswith("hedge_band:")
    assert set(summary["meta_origin"]) == {gui_app.SNAPSHOT_ORIGIN_MANUAL}
