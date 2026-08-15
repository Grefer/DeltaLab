from __future__ import annotations

import copy
import datetime
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest_pool_store
import gui_app
import history_selection
import history_bar_cache
from deltalab_ui import snapshot_store
from gui_app import BacktestApp
from pricing import HedgeBandStrategy


# 本文件的对比页用例会构造真实 BacktestApp（内部是 tk.Tk 子类），拿不到窗口
# 服务器的环境（无 DISPLAY 的 Linux、macOS 沙箱）会整进程 abort 而不是失败
# 退出。打上 gui 标记，这类环境下可用 `pytest -m "not gui"` 跳过。
pytestmark = pytest.mark.gui


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
              result=None, timestamps=None, origin=None, origin_meta=None):
    state = _state() if state is None else state
    result = _result(strategy=state["strategy_name"]) if result is None else result
    bt = SimpleNamespace(
        _results=result,
        prices=result["prices"],
        timestamps=(pd.DatetimeIndex([
            "2026-01-02", "2026-01-05", "2026-01-06",
        ]) if timestamps is None else timestamps),
    )
    extra = {} if origin is None else {"origin": origin}
    if origin_meta is not None:
        extra["origin_meta"] = origin_meta
    return BacktestApp._make_saved_backtest_result(
        bt, state, result_id, name,
        saved_at=datetime.datetime(2026, 7, 15, 12, 0), **extra,
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

    # 后两条必须打在 snapshot_store 上：_saved_comparison_payload 与它调用的
    # _snapshot_comparison_data 都在 deltalab_ui/snapshot_store.py，那两个名字
    # 从**那个模块**的全局里取。补在 gui_app 上不会报错，只会让这条「不得重新
    # 运行回测」的断言悄悄失效——本用例是反向断言，失效即假绿。
    #
    # HedgeBacktest.run 打的是类对象，改哪个模块引用它都看得见。
    monkeypatch.setattr(gui_app.HedgeBacktest, "run", unexpected)
    monkeypatch.setattr(
        snapshot_store, "summarize_strategy_result", unexpected)
    monkeypatch.setattr(snapshot_store, "result_daily_frame", unexpected)

    summary, daily_curves = BacktestApp._saved_comparison_payload([first, second])

    assert set(daily_curves) == {"结果 A", "结果 B"}
    assert set(summary["strategy"]) == {"结果 A", "结果 B"}
    assert set(summary["actual_trade_count"]) == {3}
    # 本页不设固定基准，也就没有"名次"这一说：# 列是显示顺序的行号。
    assert "rank" not in summary.columns
    assert "meta_is_comparison_baseline" not in summary.columns


def test_saved_payload_keeps_the_order_results_were_saved_in():
    """默认按保存顺序，不按任何指标排。

    按某个指标排会隐含"这一列越高/越低越好"的暗示，而本页不替用户下这个
    判断；保存顺序还能和上方结果池上下对照着看。重命名只改标签，不动顺序。
    """
    low = _snapshot("result-0001", "先存的")
    high = _snapshot(
        "result-0002", "后存的",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )
    low.summary_row["total_net_pnl"] = 2.0
    high.summary_row["total_net_pnl"] = 9.0

    # 收益更高的那条排在后面——因为它后存。
    summary, _curves = BacktestApp._saved_comparison_payload([low, high])
    assert summary["meta_result_id"].tolist() == ["result-0001", "result-0002"]

    # 传入顺序颠倒也不影响：顺序由保存序号决定。
    reversed_input, _curves = BacktestApp._saved_comparison_payload(
        [high, low])
    assert reversed_input["meta_result_id"].tolist() == [
        "result-0001", "result-0002"]

    high.name = "改了个名字"
    renamed, _curves = BacktestApp._saved_comparison_payload([low, high])
    assert renamed["meta_result_id"].tolist() == ["result-0001", "result-0002"]
    assert renamed.iloc[1]["strategy"] == "改了个名字"


def test_close_to_close_is_just_another_result_now():
    """每日收盘不再是基准，池子里没有它也不影响任何展示。"""
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
    warnings = BacktestApp._saved_comparison_warnings([band, fixed])

    assert len(summary) == 2
    assert not any("基准" in warning for warning in warnings)
    assert not any("close-to-close" in warning for warning in warnings)


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


def test_variable_summary_names_the_single_property_that_changed():
    """只有一项不同时直接说出是哪一项，其余一致的也列出来。"""
    close = _snapshot()
    band = _snapshot(
        "result-0002", "带宽",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )

    summary = BacktestApp._comparison_variable_summary([close, band])

    assert summary["state"] == "single"
    # 标题只说属性，长度恒定；具体字段与取值在 fields 里。
    assert summary["headline"] == "本次对比的变量：对冲策略"
    names = [label for label, _values in summary["fields"]]
    assert "策略类型" in names
    assert "行情" in summary["rest"] and "期权" in summary["rest"]


def test_variable_summary_reports_every_property_when_more_than_one_moved():
    """同时变两项就说不清谁造成的差异，要如实说出来。"""
    close = _snapshot()
    other = _snapshot(
        "result-0002", "另一个期权 + 另一个策略",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )
    other.contract_key = (("cls_name", "雪球"), ("subtype", "SN"))

    summary = BacktestApp._comparison_variable_summary([close, other])

    assert summary["state"] == "multiple"
    assert "同时有 2 项不同" in summary["headline"]
    assert "期权" in summary["headline"] and "对冲策略" in summary["headline"]


def test_variable_summary_says_so_when_nothing_differs():
    """输入完全一样时明说，免得用户以为页面没算。"""
    first = _snapshot()
    second = _snapshot("result-0002", "换了个名字")

    summary = BacktestApp._comparison_variable_summary([first, second])

    assert summary["state"] == "identical"
    assert "完全相同" in summary["headline"]


def test_variable_summary_is_silent_for_a_single_result():
    assert BacktestApp._comparison_variable_summary([_snapshot()]) is None


def test_field_level_diff_falls_back_when_the_key_set_itself_changes():
    """换期权大类会整体换掉参数集，说不出"某个参数不同"，只报到属性级。"""
    same_keys = [
        (("cls_name", "香草"), ("K", 100.0)),
        (("cls_name", "香草"), ("K", 110.0)),
    ]
    # 参数标签取自 OPTION_CLASSES 的定义，K 会译成"行权价"。
    assert BacktestApp._differing_field_names(same_keys) == [
        ("行权价", ["100", "110"])]

    different_keys = [
        (("cls_name", "香草"), ("K", 100.0)),
        (("cls_name", "雪球"), ("coupon", 0.08)),
    ]
    assert BacktestApp._differing_field_names(different_keys) == []


def test_warnings_cover_length_and_direction_not_the_path_hash():
    """属性差异交给变量结论；这里只留会误导读数的两件事。

    不报"行情路径不同"：模拟行情下改 σ 就会重新生成价格序列，Wind/CSV 换
    区间也必然换数据——那是上游输入变化的必然结果，单独说一句会和"其余
    一致：行情"直接打架。
    """
    first = _snapshot()
    band = _snapshot(
        "result-0002", "带宽",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"),
    )
    # 期权参数不同必然带来不同的价格序列，但这不该单独报一句。
    different_path = copy.deepcopy(band)
    different_path.path_key = ("simulate", 3, "different")
    assert BacktestApp._saved_comparison_warnings(
        [first, different_path]) == []

    # 长度不同才真的影响读数：曲线按序号对齐，累计类指标跨度也不同。
    shorter = copy.deepcopy(band)
    shorter.summary_row["n_trade_days"] = 10
    warnings = BacktestApp._saved_comparison_warnings([first, shorter])
    assert any("交易日数不同" in warning for warning in warnings)

    sell = _snapshot("result-0003", "卖方", state=_state(position=1))
    mixed = BacktestApp._saved_comparison_warnings([first, sell])
    assert any("同时包含买入与卖出" in warning for warning in mixed)


def test_rename_does_not_change_tied_result_order_or_identity():
    """指标完全相同的两条靠稳定 ID 定序，改名不会让它们互换位置。"""
    first = _snapshot("result-0001", "Z 结果")
    second = _snapshot("result-0002", "A 结果")

    before, _curves = BacktestApp._saved_comparison_payload([first, second])
    assert before["meta_result_id"].tolist() == ["result-0001", "result-0002"]

    first.name = "ZZZ"
    second.name = "AAA"
    after, _curves = BacktestApp._saved_comparison_payload([first, second])

    assert after["meta_result_id"].tolist() == ["result-0001", "result-0002"]
    assert after.iloc[0]["strategy"] == "ZZZ"


def test_rename_and_delete_use_stable_ids_and_keep_selection_consistent():
    first = _snapshot()
    second = _snapshot("result-0002", "结果 B")
    fake = SimpleNamespace(
        _saved_backtests={first.result_id: first, second.result_id: second},
        _saved_comparison_selection={first.result_id, second.result_id},
        _latest_retained_result_id=first.result_id,
        _latest_backtest=SimpleNamespace(),
        _active_job=None,
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
    assert fake._saved_comparison_selection == {second.result_id}
    assert second.result_id in fake._saved_comparison_selection
    assert fake._latest_retained_result_id is None


def _retain_dialog_default(saved_results, gui_state, monkeypatch):
    """走一遍「保留回测结果」对话框，返回它预填的默认名。"""
    prompted = []

    def _capture(_title, _message, *, initialvalue, parent=None):
        prompted.append(initialvalue)
        return None                       # 当作用户按了取消，不真的入池

    monkeypatch.setattr(gui_app.simpledialog, "askstring", _capture)
    fake = SimpleNamespace(
        _latest_backtest=SimpleNamespace(),
        _latest_backtest_state=gui_state,
        _latest_retained_result_id=None,
        _saved_backtests=saved_results,
        # 入池序号远高于同前缀编号：默认名不该受它影响。
        _saved_backtest_sequence=40,
        _strategy_snapshot_labels=BacktestApp._strategy_snapshot_labels,
    )
    BacktestApp._retain_current_backtest(fake)
    assert len(prompted) == 1
    return prompted[0]


def test_default_result_name_continues_the_numbering_of_its_own_prefix(
        monkeypatch):
    """默认名接着同前缀的最大编号往下走，别的策略存了多少条都不算数。"""
    band_state = _state(strategy="hedge_band")
    saved = {
        "result-0001": _snapshot("result-0001", "每日收盘 #05"),
        # 改过名的那条没有编号，不参与计数。
        "result-0002": _snapshot("result-0002", "每日收盘 最终版"),
        "result-0031": _snapshot(
            "result-0031", "固定间隔 · 绝对 1 #12",
            state=band_state, result=_result(strategy="hedge_band")),
    }

    assert _retain_dialog_default(saved, _state(), monkeypatch) == "每日收盘 #06"
    assert _retain_dialog_default(saved, band_state, monkeypatch) == (
        "固定间隔 · 绝对 1 #13")


def test_default_result_name_starts_at_one_for_a_prefix_never_saved(
        monkeypatch):
    saved = {"result-0001": _snapshot("result-0001", "固定间隔 · 绝对 1 #07",
                                      state=_state(strategy="hedge_band"),
                                      result=_result(strategy="hedge_band"))}

    assert _retain_dialog_default(saved, _state(), monkeypatch) == "每日收盘 #01"


def test_default_result_numbering_keeps_going_past_two_digits():
    """补零只是下限：编号过百也要继续往上，不能被格式截断。"""
    saved = {
        "result-0001": SimpleNamespace(name="每日收盘 #99"),
        "result-0002": SimpleNamespace(name="每日收盘 #100"),
    }

    assert BacktestApp._next_default_name_number(saved, "每日收盘") == 101


def test_buy_and_sell_can_now_sit_in_the_same_table():
    """头寸方向是一个可比属性，不再被硬拦——但要说清它是本次的变量。"""
    sell = _snapshot("result-0001", "卖方每日收盘", state=_state(position=1))
    buy = _snapshot("result-0002", "买方每日收盘", state=_state(position=-1))

    summary, _curves = BacktestApp._saved_comparison_payload([sell, buy])
    variable = BacktestApp._comparison_variable_summary([sell, buy])
    warnings = BacktestApp._saved_comparison_warnings([sell, buy])

    assert len(summary) == 2
    assert variable["state"] == "single"
    assert "头寸方向" in variable["headline"]
    # 损益类指标的正负来自方向本身，必须提醒该看哪几列。
    assert any("同时包含买入与卖出" in warning for warning in warnings)
    assert any("总成本" in warning for warning in warnings)


def test_selecting_another_direction_just_adds_it():
    """点另一方向的结果只是把它加进来，不再清空当前勾选。"""
    sell = _snapshot("result-0001", "卖方每日收盘", state=_state(position=1))
    buy = _snapshot("result-0002", "买方固定间隔",
                    state=_state(strategy="hedge_band", position=-1),
                    result=_result(strategy="hedge_band"))
    fake = SimpleNamespace(
        _saved_backtests={
            snapshot.result_id: snapshot for snapshot in (sell, buy)},
        _saved_comparison_selection={sell.result_id},
        _set_status=lambda _text: None,
        _refresh_saved_pool_tree=lambda: None,
        _refresh_saved_comparison_view=lambda: None,
    )

    BacktestApp._toggle_saved_backtest_selection(fake, buy.result_id)

    assert fake._saved_comparison_selection == {
        sell.result_id, buy.result_id}
    assert len(BacktestApp._selected_saved_backtests(fake)) == 2


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
    BacktestApp._clear_saved_backtest_pool(fake)

    assert fake._saved_comparison_selection == set()
    assert len(fake._saved_backtests) == 1
    assert len(messages) == 5


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


# ---------------------------------------------------------------------------
#  结果对比 与 策略优选 的衔接：改善方向、跨页配色、结构化来源
# ---------------------------------------------------------------------------


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


def test_saved_snapshot_origin_survives_rename():
    """来源是结构化字段，改名不会像旧的名称前缀那样丢失。"""
    bt = SimpleNamespace(
        _results=_result(), prices=_result()["prices"],
        timestamps=pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    snapshot = BacktestApp._make_saved_backtest_result(
        bt, _state(), "result-0007", "近月 window_1 · 每日收盘",
        origin=gui_app.SNAPSHOT_ORIGIN_HISTORY_REPLAY,
        origin_meta={
            "period_label": "近月", "lookback": "month",
            "window_id": "window_1",
        },
    )

    assert BacktestApp._saved_snapshot_origin_label(snapshot) == "分段重放"

    snapshot.name = "随便改个名字"
    assert snapshot.origin == gui_app.SNAPSHOT_ORIGIN_HISTORY_REPLAY
    assert snapshot.origin_meta["window_id"] == "window_1"
    assert BacktestApp._saved_snapshot_origin_label(snapshot) == "分段重放"


def test_snapshot_origin_meta_is_decoupled_from_the_caller_dict():
    meta = {"period_label": "近月"}
    bt = SimpleNamespace(
        _results=_result(), prices=_result()["prices"],
        timestamps=pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    snapshot = BacktestApp._make_saved_backtest_result(
        bt, _state(), "result-0008", "结果",
        origin=gui_app.SNAPSHOT_ORIGIN_HISTORY_REPLAY, origin_meta=meta)

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


# ---------------------------------------------------------------------------
#  页面入口与刷新链路：标签直达、滚动、以及刷新路径上不弹模态框
# ---------------------------------------------------------------------------


def _tab_fake(*, snapshots=None, tree=None):
    """伪造刚好够 Notebook 回调分派用的 app。"""
    built = []
    fake = SimpleNamespace(
        _saved_backtests=dict(snapshots or {}),
        _saved_pool_tree=tree,
        _history_tab="history-tab",
        _compare_tab="compare-tab",
        _nb=SimpleNamespace(select=lambda: "compare-tab"),
        _show_saved_comparison_page=(
            lambda **kwargs: built.append(kwargs)),
    )
    return fake, built


def test_compare_tab_builds_the_page_when_entered_from_the_notebook():
    """从标签直达也要看到结果，而不是「先保留结果」占位符。

    页面此前只由左侧按钮构建：按钮上写着 (N) 条，点标签进去却是空占位页。
    """
    snapshot = _snapshot()
    fake, built = _tab_fake(snapshots={snapshot.result_id: snapshot})

    BacktestApp._on_notebook_tab_changed(fake)

    # 补建时不得反过来抢占当前页——本来就已经在这一页上了。
    assert built == [{"navigate": False}]


def test_saved_result_count_is_disclosed_by_the_compare_tab_title():
    """左侧「结果对比」按钮已移除，条数改由标签页标题常驻披露。"""
    titles = []
    fake = SimpleNamespace(
        _saved_backtests={"result-0001": object(), "result-0002": object()},
        _compare_tab="compare-tab",
        _nb=SimpleNamespace(tab=lambda tab, text: titles.append((tab, text))),
    )

    BacktestApp._update_saved_result_count(fake)
    assert titles[-1] == ("compare-tab", " 🆚 结果对比 (2) ")

    # 空池不写 (0)：那时占位符已经在说「先保留结果」。
    fake._saved_backtests = {}
    BacktestApp._update_saved_result_count(fake)
    assert titles[-1] == ("compare-tab", BacktestApp._COMPARE_TAB_TITLE)


def test_saved_result_count_survives_a_destroyed_notebook():
    """计数只是展示，标签页已随窗口销毁时不该反过来打断清理流程。"""
    def _destroyed(_tab, text=None):
        raise gui_app.tk.TclError("标签页已销毁")

    fake = SimpleNamespace(
        _saved_backtests={"result-0001": object()},
        _compare_tab="compare-tab",
        _nb=SimpleNamespace(tab=_destroyed),
    )

    BacktestApp._update_saved_result_count(fake)


def test_existing_comparison_page_is_not_rebuilt_on_every_tab_visit():
    """已构建的页面由结果池自己的刷新链路维护，重进标签不重建。"""
    snapshot = _snapshot()
    fake, built = _tab_fake(
        snapshots={snapshot.result_id: snapshot},
        tree=SimpleNamespace(winfo_exists=lambda: True),
    )

    BacktestApp._on_notebook_tab_changed(fake)

    assert built == []


def test_empty_result_pool_keeps_the_placeholder_instead_of_an_empty_page():
    """池子为空时占位符那句提示正是该说的话，不要用空页面盖掉它。"""
    fake, built = _tab_fake()

    BacktestApp._on_notebook_tab_changed(fake)

    assert built == []


def test_stale_pool_tree_reference_still_builds_the_page():
    """旧渲染销毁过 widget 时按未构建处理，不能因 TclError 中断分派。"""
    snapshot = _snapshot()

    def _destroyed():
        raise gui_app.tk.TclError("widget 已销毁")

    fake, built = _tab_fake(
        snapshots={snapshot.result_id: snapshot},
        tree=SimpleNamespace(winfo_exists=_destroyed),
    )

    BacktestApp._on_notebook_tab_changed(fake)

    assert built == [{"navigate": False}]


def test_history_tab_entry_still_refreshes_its_frozen_context():
    """加了对比页分支后，历史页原有的刷新不能丢，也不得顺带建对比页。

    第一步的 Wind 控件联动走 ``BacktestApp._toggle_history_wind_controls``
    静态调用，不经过这里的替身，因此只观察其余两步。
    """
    calls = []
    fake = SimpleNamespace(
        _history_tab="history-tab",
        _compare_tab="compare-tab",
        _nb=SimpleNamespace(select=lambda: "history-tab"),
        _refresh_history_base_summary=lambda: calls.append("summary"),
        _refresh_history_current_band_label=lambda: calls.append("band"),
        _show_saved_comparison_page=lambda **_kwargs: pytest.fail(
            "历史页不得构建结果对比页"),
    )

    BacktestApp._on_notebook_tab_changed(fake)

    assert calls == ["summary", "band"]


def test_refresh_failure_lands_in_the_empty_state_not_a_modal(monkeypatch):
    """刷新链路每次勾选都会走，出错弹窗会把人挡在页面外。"""
    monkeypatch.setattr(
        gui_app.messagebox, "showerror",
        lambda *_args, **_kwargs: pytest.fail("刷新链路不得弹出模态框"))
    empties = []
    fake = SimpleNamespace(
        _selected_saved_backtests=lambda: (_ for _ in ()).throw(
            ValueError("读取快照失败")),
        _render_saved_comparison_empty=(
            lambda title, detail: empties.append((title, detail))),
    )

    BacktestApp._refresh_saved_comparison_view(fake)

    assert empties == [("无法生成对比", "读取快照失败")]


def _comparison_app(*snapshots):
    """建一个结果池里已有指定快照、但还没打开过对比页的真实 app。

    **界面测试一律用 ``update_idletasks()``，不要用 ``update()``。**

    ``update()`` 在这里会永不返回，而且信号打不断它——测试会挂到超时被杀。
    这不是 DeltaLab 的问题，零应用代码就能复现，四个条件缺一不可：

    1. 进程里先有过一个已 ``destroy()`` 的 ``tk.Tk``；
    2. 当前这个是第二个 ``tk.Tk`` 实例；
    3. 它有个 ``Canvas``，里面用 ``create_window`` 嵌了 widget；
    4. 该 widget 有子控件，且窗口处于 ``withdraw()`` 状态。

    去掉任何一条都正常返回。DeltaLab 命中它是因为左侧参数面板正是
    Canvas + 嵌入 Frame 的滚动结构，而测试里每个 app 都 withdraw，一个
    文件里又不止建一个 app。真实运行只有一个实例、窗口也是显示的，两条
    都不满足，所以产品不受影响。

    代价是虚拟事件（``<<NotebookTabChanged>>`` 之类）在测试里派发不了，
    要验证这类回调就显式调用它，绑定本身由构建界面时的 ``bind`` 负责。
    """
    import tkinter as tk

    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    app.withdraw()
    for snapshot in snapshots:
        app._saved_backtests[snapshot.result_id] = snapshot
        app._saved_comparison_selection.add(snapshot.result_id)
    app.update_idletasks()
    return app


def test_landing_on_the_compare_tab_builds_a_scrollable_ranking_table():
    """在真实控件树上验证：落到对比页就该看到结果池和能滚的排名表。

    这里显式调用标签回调而不是等 Tk 派发 ``<<NotebookTabChanged>>``：虚拟
    事件要 ``update()`` 才会被处理，而这里不能用它（原因见 ``_comparison_app``）。
    事件绑定本身由构建界面时的 ``_nb.bind`` 负责，历史页走的是同一条绑定。
    """
    snapshots = [_snapshot("result-0001", "每日收盘 · 基准")] + [
        _snapshot(
            f"result-{index:04d}", f"固定间隔 {index}",
            state=_state(strategy="hedge_band", threshold=float(index)),
            result=_result(strategy="hedge_band"),
        )
        for index in range(2, 12)
    ]
    app = _comparison_app(*snapshots)
    try:
        assert getattr(app, "_saved_pool_tree", None) is None
        # 左侧不再有对比按钮：标签页是唯一入口，标题带着条数。
        assert not hasattr(app, "_compare_btn")
        BacktestApp._update_saved_result_count(app)
        assert app._nb.tab(app._compare_tab, "text") == (
            f"{BacktestApp._COMPARE_TAB_TITLE.rstrip()} ({len(snapshots)}) ")

        app._nb.select(app._compare_tab)
        BacktestApp._on_notebook_tab_changed(app)
        app.update_idletasks()

        assert len(app._saved_pool_tree.get_children()) == len(snapshots)
        tree = app._comparison_tree
        assert len(tree.get_children()) == len(snapshots)
        assert tree.cget("yscrollcommand"), "排名表没有接上滚动条"
        scrollbars = [
            child for child in tree.master.winfo_children()
            if isinstance(child, gui_app.ttk.Scrollbar)
        ]
        assert len(scrollbars) == 1
    finally:
        app.destroy()


def test_row_actions_live_in_the_right_click_menu():
    """加载明细 / 应用策略 / 重命名只在行右键菜单里，工具栏不放它们。

    六个并排按钮里有四个要先点中一行才有意义，靠一句「点击其它列聚焦后…」
    解释；收进右键菜单后作用对象就是点中的那一行，不用再解释。第一项的
    文案按该行当前是否显示改写，否则菜单永远写着一个方向。

    「删除选中」是例外，它常驻工具栏：删除是这一页唯一一个"想得起来要做、
    却猜不到藏在右键里"的动作，而它的作用对象与右键那项完全相同（行选择）。

    这里把 ``identify_row`` 换掉而不是造真实坐标：``_comparison_app`` 里的
    窗口是 withdraw 的，未映射的 Treeview 拿不到 ``bbox``。
    """
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        # 乙改成"不显示"，两种勾选状态各验一次首项文案。
        app._saved_comparison_selection.discard("result-0002")

        titles = []

        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, gui_app.ttk.Button):
                    titles.append(str(child.cget("text")))
                walk(child)

        walk(app._compare_tab)
        assert titles == [
            "全选", "取消全选", "删除选中", "全部清空", "导出 CSV"], titles

        menu = app._saved_pool_menu
        labels = [
            menu.entrycget(index, "label")
            for index in range(menu.index("end") + 1)
            if menu.type(index) == "command"
        ]
        assert labels == [
            "显示", "参数详情…", "加载明细", "应用策略", "重命名…", "删除"], labels

        tree = app._saved_pool_tree
        opened = []
        menu.tk_popup = lambda x_root, y_root: opened.append((x_root, y_root))
        event = SimpleNamespace(x=10, y=10, x_root=0, y_root=0)

        tree.identify_row = lambda _y: "result-0002"
        BacktestApp._popup_saved_pool_menu(app, event)
        assert tree.focus() == "result-0002", "右键要把该行设成作用对象"
        assert menu.entrycget(0, "label") == "显示"

        tree.identify_row = lambda _y: "result-0001"
        BacktestApp._popup_saved_pool_menu(app, event)
        assert tree.focus() == "result-0001"
        assert menu.entrycget(0, "label") == "取消显示"
        assert len(opened) == 2

        # 「加载明细」按该行有没有重放配方启停：本功能上线前保留的快照只有
        # 汇总层，点了只能弹一句「没有配方」，不如让它点不动。
        detail_index = BacktestApp._POOL_MENU_LOAD_DETAIL_INDEX
        assert menu.entrycget(detail_index, "label") == "加载明细"
        assert str(menu.entrycget(detail_index, "state")) == "normal"

        app._saved_backtests["result-0001"].replay = {}
        BacktestApp._popup_saved_pool_menu(app, event)
        assert str(menu.entrycget(detail_index, "state")) == "disabled"
        # 「参数详情」读的是签名，没有配方也照样打得开——它不能跟着一起灰。
        params_index = detail_index - 1
        assert menu.entrycget(params_index, "label") == "参数详情…"
        assert str(menu.entrycget(params_index, "state")) == "normal"

        # 空白处右键没有作用对象：弹一份点了只会提示"请选择结果"的菜单，
        # 比不弹更费解。
        popped_before = len(opened)
        tree.identify_row = lambda _y: ""
        BacktestApp._popup_saved_pool_menu(app, event)
        assert len(opened) == popped_before
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
#  详情区：快照溯源，以及差值该往哪个方向读
# ---------------------------------------------------------------------------


def _variable_text(snapshots):
    """把说明卡的三层拼回一起，供只关心文案内容的断言使用。"""
    summary = BacktestApp._comparison_variable_summary(snapshots)
    fields = "；".join(
        label if values is None else f"{label}：{' vs '.join(values)}"
        for label, values in summary["fields"])
    return f"{summary['headline']}（{fields}） {summary['rest']}"


def _verified_snapshot(result_id="result-0002", name="1年 window_3 · 固定间隔"):
    """一条由优选页「加载分段到展示页」重放后保留下来的快照。"""
    return _snapshot(
        result_id, name,
        state=_state(strategy="hedge_band", threshold=1.5),
        result=_result(strategy="hedge_band"),
        origin=gui_app.SNAPSHOT_ORIGIN_HISTORY_REPLAY,
        origin_meta={
            "lookback": "year", "period_label": "1年",
            "history_strategy": "固定间隔(1.5σ)", "window_id": "window_3",
        },
    )


def test_replay_snapshot_names_its_segment_and_manual_one_stays_quiet():
    """分段重放要标出是哪一段；手工回测没有可溯的上游，不硬凑一句。"""
    replay_origin, replay_meta = BacktestApp._snapshot_origin_from_state({
        **_state(), "history_replay_strategy": "固定时刻(10:30)",
        "history_replay_lookback": "quarter",
        "history_replay_window_id": "window_3",
    })
    replay = _snapshot(
        "result-0003", "分段结果",
        origin=replay_origin, origin_meta=replay_meta)

    detail = BacktestApp._snapshot_origin_detail(replay)
    assert "分段 window_3" in detail
    assert "候选『固定时刻(10:30)』" in detail
    # 批次编号属于批量验证，重放没有这回事。
    assert "批量验证" not in detail

    assert BacktestApp._snapshot_origin_detail(_snapshot()) == ""


def test_saved_payload_carries_provenance_without_touching_the_metrics():
    """溯源字段随快照进入 payload，但不得改动任何一项指标。"""
    manual = _snapshot("result-0001", "手工结果")
    verified = _verified_snapshot()
    manual.summary_row["total_net_pnl"] = 12.0
    verified.summary_row["total_net_pnl"] = 8.0

    summary, _curves = BacktestApp._saved_comparison_payload(
        [manual, verified])
    rows = {row["meta_result_id"]: row for _i, row in summary.iterrows()}

    assert rows["result-0002"]["total_net_pnl"] == pytest.approx(8.0)
    assert "分段 window_3" in rows["result-0002"]["meta_origin_detail"]
    # 手工快照没有上游可溯，这一格是空串而不是编造一句。
    assert rows["result-0001"]["meta_origin_detail"] == ""
    # 优选当时的改善依赖基准，本页已经不算它了。
    assert "meta_origin_improvement" not in summary.columns


# ---------------------------------------------------------------------------
#  渲染骨架：建一次、之后只换数据
# ---------------------------------------------------------------------------


def test_discarding_the_frame_clears_every_reference_it_owns():
    """丢弃骨架必须把图表、表格、统计卡的引用一起清干净。

    留下任何一个指向已销毁 widget 的引用，下一次刷新就会以为骨架还在，
    然后往一个不存在的表里写数据。
    """
    cleared = []
    fake = SimpleNamespace(
        _comparison_frame=object(),
        _comparison_chart_figure=SimpleNamespace(
            clear=lambda: cleared.append(True)),
        _comparison_chart_ax=object(),
        _comparison_chart_canvas=object(),
        _comparison_tree=object(),
        _comparison_rows={"strategy_0": {}},
        _comparison_stats=object(),
        _comparison_variable_var=object(),
        _comparison_variable_accent=object(),
    )

    BacktestApp._discard_comparison_frame(fake)

    assert cleared == [True]
    assert fake._comparison_frame is None
    assert fake._comparison_chart_figure is None
    assert fake._comparison_chart_ax is None
    assert fake._comparison_chart_canvas is None
    assert fake._comparison_tree is None
    assert fake._comparison_rows == {}
    assert fake._comparison_variable_var is None
    assert fake._comparison_variable_accent is None


def test_selection_callback_stays_quiet_while_the_table_is_refilled():
    """重填期间 Tk 会发选择事件，那时表只有半张，不能拿去画图。"""
    fake = SimpleNamespace(
        _comparison_populating=True,
        _comparison_tree=SimpleNamespace(
            selection=lambda: pytest.fail("重填期间不应读取选中行")),
        _draw_comparison_cumulative_chart=lambda: pytest.fail(
            "重填期间不应重画图表"),
    )

    BacktestApp._update_comparison_selection(fake)


def test_toggling_selection_reuses_the_chart_and_table_widgets():
    """勾选切换只换数据：图表、表格、骨架都必须是同一批 widget。

    每次重建都要新造一个 matplotlib Figure 和整张表，页面还会可见地重排
    一次；批量验证十条候选就是十次这样的重建。
    """
    snapshots = [_snapshot("result-0001", "每日收盘 · 基准")] + [
        _snapshot(
            f"result-{index:04d}", f"固定间隔 {index}",
            state=_state(strategy="hedge_band", threshold=float(index)),
            result=_result(strategy="hedge_band"),
        )
        for index in range(2, 6)
    ]
    app = _comparison_app(*snapshots)
    try:
        app._saved_comparison_selection.intersection_update({"result-0001"})
        app._nb.select(app._compare_tab)
        BacktestApp._on_notebook_tab_changed(app)
        app.update_idletasks()
        figure = app._comparison_chart_figure
        tree = app._comparison_tree
        frame = app._comparison_frame
        assert len(tree.get_children()) == 1

        for index in range(2, 6):
            BacktestApp._toggle_saved_backtest_selection(
                app, f"result-{index:04d}")
            app.update_idletasks()

        assert app._comparison_chart_figure is figure
        assert app._comparison_tree is tree
        assert app._comparison_frame is frame
        assert len(tree.get_children()) == len(snapshots)
        assert len(app._comparison_daily_curves) == len(snapshots)

        # 清空会退回空状态，骨架随之丢弃；再勾选时重新建一套可用的。
        BacktestApp._clear_saved_backtest_selection(app)
        app.update_idletasks()
        assert app._comparison_frame is None

        BacktestApp._toggle_saved_backtest_selection(app, "result-0002")
        app.update_idletasks()
        assert app._comparison_frame is not None
        assert app._comparison_chart_figure is not figure
        # 勾一条就是一条，不再自动把基准拽进来。
        assert len(app._comparison_tree.get_children()) == 1
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
#  口径收敛：数得对、读不改状态、基准不串方向
# ---------------------------------------------------------------------------


def test_reading_the_selection_never_changes_it():
    """取出勾选快照是纯读：它跑在结果池画完之后，改了状态树就对不上。"""
    baseline = _snapshot("result-0001", "每日收盘")
    candidate = _snapshot(
        "result-0002", "固定间隔",
        state=_state(strategy="hedge_band"),
        result=_result(strategy="hedge_band"))
    fake = SimpleNamespace(
        _saved_backtests={
            baseline.result_id: baseline, candidate.result_id: candidate},
        _saved_comparison_selection={candidate.result_id},
        _saved_comparison_baseline_id=None,
    )

    selected = BacktestApp._selected_saved_backtests(fake)

    assert [snapshot.result_id for snapshot in selected] == ["result-0002"]
    assert fake._saved_comparison_selection == {"result-0002"}


# ---------------------------------------------------------------------------
#  出口：导出、回填左侧表单、列头排序
# ---------------------------------------------------------------------------


def test_snapshot_keeps_hedge_inputs_in_their_original_unit():
    """回填要的是能写回表单的结构化输入，展示串回不了表单。"""
    band = _snapshot(
        state=dict(_state(strategy="hedge_band", threshold=1.5),
                   interval_type="sigma"),
        result=_result(strategy="hedge_band"))

    assert band.form_state == {
        "strategy_name": "hedge_band",
        "fixed_times": "11:30,15:00",
        "interval_type": "sigma",
        "price_interval": 1.5,
        "force_day_close_hedge": False,
        # 波动率口径决定带宽怎么换算，实际改变回测结果，必须一起记下来：
        # 不然两条只差这个口径的结果，逐字段比下来会是"完全一致"。
        "sigma_source": "implied",
        "sigma_window": 20,
    }


def _form_fake():
    written = {}
    return written, SimpleNamespace(
        _strategy_var=SimpleNamespace(
            set=lambda v: written.__setitem__("strategy", v)),
        _fixed_times_var=SimpleNamespace(
            set=lambda v: written.__setitem__("fixed_times", v)),
        _band_abs_var=SimpleNamespace(
            set=lambda v: written.__setitem__("absolute", v)),
        _band_rel_var=SimpleNamespace(
            set=lambda v: written.__setitem__("relative", v)),
        _band_sigma_var=SimpleNamespace(
            set=lambda v: written.__setitem__("sigma", v)),
        _sigma_src_var=SimpleNamespace(
            set=lambda v: written.__setitem__("sigma_source", v)),
        _sigma_win_var=SimpleNamespace(
            set=lambda v: written.__setitem__("sigma_window", v)),
        _force_day_close_hedge_var=SimpleNamespace(
            set=lambda v: written.__setitem__("fallback", v)),
        _mark_band_edited=lambda kind: written.__setitem__("edited", kind),
        _sync_band_inputs=lambda kind, strict=False: written.__setitem__(
            "synced", (kind, strict)),
        _toggle_strategy=lambda: written.__setitem__("toggled", True),
    )


def test_backfilling_a_band_snapshot_writes_back_its_own_unit():
    """按保存时那一种单位写回：换算过一手，显示值就和当初跑的差个舍入。"""
    band = _snapshot(
        state=dict(_state(strategy="hedge_band", threshold=1.5),
                   interval_type="sigma", force_day_close_hedge=True),
        result=_result(strategy="hedge_band"))
    written, fake = _form_fake()

    assert BacktestApp._apply_snapshot_strategy_to_form(
        fake, band) == "hedge_band"

    assert written["sigma"] == "1.5"
    assert "absolute" not in written and "relative" not in written
    assert written["edited"] == "sigma"
    assert written["synced"] == ("sigma", True)
    # 波动率口径也要写回，且在带宽同步之前——换算依赖它。
    assert written["sigma_source"] == gui_app.SIGMA_SOURCE_DISPLAY["implied"]
    assert written["sigma_window"] == "20"
    assert written["fallback"] is True
    assert written["toggled"] is True


def test_backfilling_a_fixed_time_snapshot_restores_its_times():
    fixed = _snapshot(
        state=dict(_state(strategy="fixed_times"), fixed_times="10:30,14:00"),
        result=_result(strategy="fixed_times"))
    written, fake = _form_fake()

    assert BacktestApp._apply_snapshot_strategy_to_form(
        fake, fixed) == "fixed_times"

    assert written["fixed_times"] == "10:30,14:00"


def test_backfilling_an_old_snapshot_explains_itself_instead_of_crashing():
    """本功能之前保留的快照没有结构化参数，要说清楚怎么办。"""
    legacy = _snapshot()
    legacy.form_state = {}
    _written, fake = _form_fake()

    with pytest.raises(ValueError, match="重新运行一次回测并保留"):
        BacktestApp._apply_snapshot_strategy_to_form(fake, legacy)


def _sort_summary():
    return pd.DataFrame([
        {"rank": 1, "strategy": "B 候选", "daily_net_pnl_rms": 8.0, "total_tc": 9.0},
        {"rank": 2, "strategy": "A 候选", "daily_net_pnl_rms": 10.0, "total_tc": 2.0},
        {"rank": 3, "strategy": "基准", "daily_net_pnl_rms": 12.0, "total_tc": 5.0},
    ])


def test_sorting_by_a_column_never_renumbers_the_rank():
    """名次是按 RMS 定下的那一份；换个列看只是换查看顺序。

    两者混在一起的话，按成本排完第一行会写着 #1，而它并不是排名第一。
    """
    rows = BacktestApp._comparison_sorted_rows(
        _sort_summary(), "total_tc", False)

    assert [row["strategy"] for row in rows] == ["A 候选", "基准", "B 候选"]
    assert [row["rank"] for row in rows] == [2, 3, 1]


def test_missing_values_sort_last_in_both_directions():
    """缺失值不是最好也不是最差，两个方向都排在末尾。"""
    summary = pd.DataFrame([
        {"rank": 1, "strategy": "有值", "total_tc": 5.0},
        {"rank": 2, "strategy": "缺失", "total_tc": float("nan")},
        {"rank": 3, "strategy": "更小", "total_tc": 1.0},
    ])

    ascending = BacktestApp._comparison_sorted_rows(
        summary, "total_tc", False)
    descending = BacktestApp._comparison_sorted_rows(
        summary, "total_tc", True)

    assert [row["strategy"] for row in ascending] == ["更小", "有值", "缺失"]
    assert [row["strategy"] for row in descending] == ["有值", "更小", "缺失"]


def test_clicking_a_header_twice_flips_direction_and_switching_resets_it():
    repopulated = []
    fake = SimpleNamespace(
        _comparison_summary=_sort_summary(),
        _comparison_daily_curves={},
        _populate_comparison_view=(
            lambda summary, curves: repopulated.append(True)),
    )

    BacktestApp._sort_comparison_by(fake, "total_tc")
    assert fake._comparison_sort_column == "total_tc"
    # 越低越好的列先给升序，第一次点就看到最省成本的那一头。
    assert fake._comparison_sort_descending is False

    BacktestApp._sort_comparison_by(fake, "total_tc")
    assert fake._comparison_sort_descending is True

    # 越高越好的列反过来。
    BacktestApp._sort_comparison_by(fake, "total_net_pnl")
    assert fake._comparison_sort_column == "total_net_pnl"
    assert fake._comparison_sort_descending is True
    assert len(repopulated) == 3


def test_export_carries_raw_numbers_and_provenance():
    """导出给的是原始数值而非表里那串格式化文本，并带上溯源。"""
    manual = _snapshot("result-0001", "手工结果")
    verified = _verified_snapshot()
    manual.summary_row["total_net_pnl"] = 12.0
    verified.summary_row["total_net_pnl"] = 8.0
    summary, curves = BacktestApp._saved_comparison_payload(
        [manual, verified])
    fake = SimpleNamespace(
        _comparison_summary=summary, _comparison_daily_curves=curves)

    ranking, curve_frame = BacktestApp._comparison_export_frames(fake)

    # 默认顺序就是期末净损益从高到低。
    assert list(ranking["结果名称"]) == [
        "手工结果", "1年 window_3 · 固定间隔"]
    assert ranking["期末净损益"].tolist() == [12.0, 8.0]
    assert "分段 window_3" in ranking["来源溯源"].iloc[1]
    # 基准派生的两列已经不复存在。
    assert "较每日收盘改善" not in ranking.columns
    assert "是否固定基准" not in ranking.columns
    assert "日净损益RMS" not in ranking.columns
    # 曲线导成宽表：一列一条结果，行号就是交易日序号。
    assert set(curve_frame.columns) == {"手工结果", "1年 window_3 · 固定间隔"}
    assert curve_frame.index.name == "交易日序号"
    assert curve_frame.index.tolist() == [1, 2]


def test_export_refuses_when_there_is_nothing_selected():
    fake = SimpleNamespace(_comparison_summary=None)

    with pytest.raises(ValueError, match="没有可导出的对比结果"):
        BacktestApp._comparison_export_frames(fake)


def test_the_row_number_column_is_declared_unsortable():
    """`#` 排不动，就不能在表头上装出可排序的样子。

    _comparison_sorted_rows 对 rank 直接原样返回（它是当前显示顺序的行号，
    不是数据字段），而表头此前照样绑了排序命令、打了 ⇅——点下去毫无反应。
    这条断言把"声明"和"实际行为"钉在一起：哪天 rank 真的能排了，就该从
    这个集合里拿掉。
    """
    summary = pd.DataFrame([
        {"strategy": "A", "total_net_pnl": 1.0},
        {"strategy": "B", "total_net_pnl": 3.0},
        {"strategy": "C", "total_net_pnl": 2.0},
    ])

    def names(key):
        return [row["strategy"] for row in
                BacktestApp._comparison_sorted_rows(summary, key, True)]

    assert names("rank") == ["A", "B", "C"]           # 原样返回
    assert names("total_net_pnl") == ["B", "C", "A"]  # 真的会排
    assert "rank" in BacktestApp._COMPARISON_UNSORTABLE_COLUMNS
    assert "total_net_pnl" not in BacktestApp._COMPARISON_UNSORTABLE_COLUMNS


def test_export_survives_results_with_different_trading_day_counts():
    """交易日数不同是本页显式支持的状态，导出不能因此整体失败。

    曾经用 dict-of-arrays 直接建表，pandas 要求各列等长，于是抛
    ``All arrays must be of the same length``；调用方的 except 又把它渲染成
    「没有可导出的结果」，连同已经拼好的排名表一起丢掉——而结果明明就在
    屏幕上，警示语还专门写着"各条的交易日数不同：曲线按序号对齐"。
    换区间做对比正是本页头号用途，所以这条路径必须通。
    """
    fake = SimpleNamespace(
        _comparison_summary=pd.DataFrame([
            {"name": "长的"}, {"name": "短的"}]),
        _comparison_daily_curves={
            "长的": {"cumulative_net_pnl": np.array([1.0, 2.0, 3.0, 4.0])},
            "短的": {"cumulative_net_pnl": np.array([5.0, 6.0])},
        },
    )

    ranking, curves = BacktestApp._comparison_export_frames(fake)

    assert len(ranking) == 2
    assert curves.index.name == "交易日序号"
    # 索引即交易日序号，短的那条补 NaN——与绘图侧 x = arange(1, len+1) 同口径。
    assert curves.index.tolist() == [1, 2, 3, 4]
    assert curves["长的"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert curves["短的"].iloc[:2].tolist() == [5.0, 6.0]
    assert curves["短的"].iloc[2:].isna().all()


def test_strategy_aspect_ignores_fields_the_strategy_never_reads():
    """两条同策略同数字的结果不能因为无关残值被报成「对冲策略不同」。

    _snapshot_form_state 是给回填表单用的，不看 strategy_name 就全量记录
    固定时刻与带宽三项；而 _collect_history_state 每轮优选都会把本轮候选
    配置写回 state，重放快照因此带着与自己无关的残值。两轮优选只有固定时
    刻候选文本不同、都选每日收盘基线加载同一段，逐日损益完全相同，页面却
    报「本次对比的变量：对冲策略」还染成绿色（干净的单变量实验）。
    """
    def snap(**overrides):
        state = {
            "strategy_name": "close_to_close", "fixed_times": "",
            "interval_type": "absolute", "price_interval": 1.0,
            "force_day_close_hedge": True,
            "sigma_source": "implied", "sigma_window": 20,
        }
        state.update(overrides)
        return SimpleNamespace(form_state=state)

    signature = BacktestApp._snapshot_strategy_signature
    # close_to_close 根本不读 fixed_times / 带宽三项
    assert signature(snap(fixed_times="10:00,14:00")) == signature(
        snap(fixed_times="11:00"))
    assert signature(snap(price_interval=1.0)) == signature(
        snap(price_interval=99.0))
    # 但真正生效的输入必须照常算差异
    assert signature(snap()) != signature(snap(force_day_close_hedge=False))
    assert signature(
        snap(strategy_name="fixed_times", fixed_times="10:00")) != signature(
        snap(strategy_name="fixed_times", fixed_times="11:00"))
    assert signature(
        snap(strategy_name="hedge_band", price_interval=1.0)) != signature(
        snap(strategy_name="hedge_band", price_interval=2.0))
    # 带宽策略下的固定时刻残值同样无关
    assert signature(
        snap(strategy_name="hedge_band", fixed_times="X")) == signature(
        snap(strategy_name="hedge_band", fixed_times="Y"))


def test_strategy_aspect_keeps_a_stable_key_set_across_strategies():
    """无关字段填 None 而不是删键——删键会让字段级差异定位整体失效。

    _differing_field_names 在两侧键集合不同时直接放弃字段级定位、只报到
    属性级。若按策略删键，「每日收盘 vs 固定间隔」这种正当的跨策略对比就
    再也说不出「策略类型 / 带宽单位 / 带宽阈值变了」。
    """
    def snap(name, **overrides):
        state = {
            "strategy_name": name, "fixed_times": "10:00",
            "interval_type": "absolute", "price_interval": 1.0,
            "force_day_close_hedge": True,
            "sigma_source": "implied", "sigma_window": 20,
        }
        state.update(overrides)
        return SimpleNamespace(form_state=state)

    keys = [
        {key for key, _value in BacktestApp._snapshot_strategy_signature(
            snap(name))}
        for name in ("close_to_close", "fixed_times", "hedge_band")
    ]
    assert keys[0] == keys[1] == keys[2]
    # 跨策略仍能逐字段点名
    assert BacktestApp._differing_field_names([
        BacktestApp._snapshot_strategy_signature(snap("close_to_close")),
        BacktestApp._snapshot_strategy_signature(snap("hedge_band")),
    ])


def test_sorting_keeps_the_row_the_user_was_looking_at():
    """换个列排序不该把选中行甩掉——行号会变，快照 ID 不会。"""
    snapshots = [
        _snapshot("result-0001", "结果 A"),
        _snapshot("result-0002", "结果 B",
                  state=_state(strategy="hedge_band", threshold=1.5),
                  result=_result(strategy="hedge_band")),
        _snapshot("result-0003", "结果 C",
                  state=_state(strategy="hedge_band", threshold=2.5),
                  result=_result(strategy="hedge_band")),
    ]
    snapshots[0].summary_row.update({"total_net_pnl": 12.0, "total_tc": 5.0})
    snapshots[1].summary_row.update({"total_net_pnl": 8.0, "total_tc": 9.0})
    snapshots[2].summary_row.update({"total_net_pnl": 10.0, "total_tc": 2.0})
    app = _comparison_app(*snapshots)
    try:
        app._nb.select(app._compare_tab)
        BacktestApp._on_notebook_tab_changed(app)
        app.update_idletasks()
        tree = app._comparison_tree
        # 默认按保存顺序。单元格文本带内边距空格，比较前先剥掉。
        assert [
            tree.item(iid, "values")[1].strip()
            for iid in tree.get_children()] == ["结果 A", "结果 B", "结果 C"]

        target = next(
            iid for iid, row in app._comparison_rows.items()
            if row.get("meta_result_id") == "result-0003")
        tree.selection_set(target)
        tree.focus(target)
        BacktestApp._update_comparison_selection(app)
        app.update_idletasks()

        BacktestApp._sort_comparison_by(app, "total_tc")
        app.update_idletasks()

        still = app._comparison_rows[tree.selection()[0]]
        assert still["meta_result_id"] == "result-0003"
        # 成本最低，排到了第一行；# 列跟着显示顺序走。
        assert tree.get_children()[0] == tree.selection()[0]
        assert tree.item(tree.selection()[0], "values")[0].strip() == "1"
    finally:
        app.destroy()


def test_closing_the_window_cancels_its_pending_band_timer():
    """关窗要撤掉带宽换算的防抖定时器。

    Tk 的 destroy 清 widget、也清本实例注册的 Tcl 命令，唯独不动 after
    队列。留下的那条回调指向一个已被删除的命令名，同进程随后再开窗口时
    事件循环会执行到它并报 invalid command name。
    """
    app = _comparison_app()
    pending = app._band_reference_after_id
    assert pending is not None, "带宽输入的防抖定时器此时应当挂着"

    cancelled = []
    original_cancel = app.after_cancel
    app.after_cancel = lambda after_id: (
        cancelled.append(after_id), original_cancel(after_id))[1]
    app.destroy()

    assert cancelled == [pending]


# ---------------------------------------------------------------------------
#  结论卡与表格排版：与策略优选页对齐
# ---------------------------------------------------------------------------


def _extreme_summary():
    high_pnl = _snapshot("result-0001", "高收益")
    low_cost = _snapshot(
        "result-0002", "低成本",
        state=_state(strategy="hedge_band"),
        result=_result(strategy="hedge_band"))
    plain = _snapshot(
        "result-0003", "都不突出",
        state=_state(strategy="fixed_times"),
        result=_result(strategy="fixed_times"))
    high_pnl.summary_row.update(
        {"total_net_pnl": 18.0, "total_tc": 9.0, "max_drawdown": 1.5})
    low_cost.summary_row.update(
        {"total_net_pnl": 15.0, "total_tc": 2.0, "max_drawdown": 4.0})
    plain.summary_row.update(
        {"total_net_pnl": 12.0, "total_tc": 5.0, "max_drawdown": 3.0})
    summary, _curves = BacktestApp._saved_comparison_payload(
        [high_pnl, low_cost, plain])
    return summary


def test_cells_carry_their_own_padding_on_the_anchored_side():
    """Treeview 没有单元格内边距，右对齐的数字会直接顶到列边界上。"""
    padded = BacktestApp._pad_comparison_row(
        (1, "结果名", "18.00", "9.00", "1.50", "1/3", "1,500.00"))

    # 居中列不补，左对齐补在左侧，右对齐补在右侧。
    assert padded[0] == "1"
    assert padded[1] == " 结果名"
    assert padded[2] == "18.00 "
    assert padded[-1] == "1,500.00 "


def test_headers_align_the_same_way_as_their_column_data():
    """表头与数据同向对齐，一列才有共同的竖边可循。"""
    anchors = {
        key: BacktestApp._comparison_column_anchor(key)
        for key, _text, _width in BacktestApp._COMPARISON_RANKING_COLUMNS
    }

    assert anchors["rank"] == "center"
    assert anchors["strategy"] == "w"
    assert all(
        anchors[key] == "e" for key in
        ("total_net_pnl", "total_tc", "max_drawdown", "turnover"))


def test_variable_card_states_the_variable_and_the_caveats():
    """说明卡占的是原先那张明细卡的位置：摆本页唯一的结论性信息。

    明细卡的五个数字指标表里全有、参数与来源结果池表里全有，独有的只剩一
    个交易日数（已并入指标表）——那块版面该留给"这两条到底差在哪"。
    """
    buy = _snapshot("result-0001", "买方", state=_state(position=-1))
    sell = _snapshot("result-0002", "卖方", state=_state(position=1))
    app = _comparison_app()
    try:
        BacktestApp._build_comparison_variable_card(app, app._compare_container)

        BacktestApp._refresh_comparison_variable_card(app, [buy, sell])
        assert "头寸方向" in app._comparison_variable_var.get()
        assert "其余一致" in app._comparison_same_var.get()
        # 买卖同表时提醒该看哪几列。
        assert "总成本" in app._comparison_caveat_var.get()
        assert app._comparison_variable_accent.cget("bg") == (
            gui_app.PALETTE["success"])

        # 只选一条时给引导，不报变量。
        BacktestApp._refresh_comparison_variable_card(app, [buy])
        assert "再勾选一条" in app._comparison_variable_var.get()
        assert app._comparison_caveat_var.get() == ""
        assert app._comparison_variable_accent.cget("bg") == (
            gui_app.PALETTE["border_soft"])
    finally:
        app.destroy()


def test_every_column_stretches_so_one_column_cannot_eat_the_slack():
    """所有列一起 stretch，余量按比例摊掉。

    只让某一列可拉伸时它会独吞全部剩余空间：本页容器约 970px 而列宽总和
    866px，那 100 来个像素此前全灌进结果名列，把它撑到内容宽度的三倍多，
    数字列却全挤在左边。
    """
    app = _comparison_app(_snapshot())
    try:
        app._nb.select(app._compare_tab)
        BacktestApp._on_notebook_tab_changed(app)
        app.update_idletasks()

        for key, _text, width in BacktestApp._COMPARISON_RANKING_COLUMNS:
            column = app._comparison_tree.column(key)
            assert column["stretch"], f"{key} 列没有参与拉伸"
            # 窗口收窄时也不该被压到看不清内容。
            assert column["minwidth"] == max(40, width - 30)

        for key, _text, width, _anchor in BacktestApp._SAVED_POOL_COLUMNS:
            column = app._saved_pool_tree.column(key)
            assert column["stretch"], f"结果池 {key} 列没有参与拉伸"
            assert column["minwidth"] == max(44, width - 30)
    finally:
        app.destroy()


def test_both_tables_align_headers_with_their_data():
    """两张表的表头都与本列数据同向，一列才有共同的竖边可循。"""
    app = _comparison_app(_snapshot())
    try:
        app._nb.select(app._compare_tab)
        BacktestApp._on_notebook_tab_changed(app)
        app.update_idletasks()

        for key, _text, _width in BacktestApp._COMPARISON_RANKING_COLUMNS:
            assert (app._comparison_tree.heading(key)["anchor"]
                    == app._comparison_tree.column(key)["anchor"])
        for key, _text, _width, anchor in BacktestApp._SAVED_POOL_COLUMNS:
            assert app._saved_pool_tree.heading(key)["anchor"] == anchor
            assert app._saved_pool_tree.column(key)["anchor"] == anchor
    finally:
        app.destroy()


def test_pool_columns_widen_for_the_longest_content_they_hold():
    """列宽按池子里最长的那格取，夹在基准宽与上限之间。

    同一列的内容长度差着数倍：「策略参数」在每日收盘时是一句话，在固定间隔
    时要写下绝对 / 相对 / σ 三种口径的等价换算。定死一个宽度只能二选一——
    要么长的那种被截断，要么短的那种空出一大片。
    """
    floor = {
        key: max(44, width - 30)
        for key, _text, width, _anchor in BacktestApp._SAVED_POOL_COLUMNS
    }
    ceiling = BacktestApp._SAVED_POOL_COLUMN_MAX

    app = _comparison_app(_snapshot("result-0001", "甲"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        tree = app._saved_pool_tree
        narrow = tree.column("parameters", "width")
        assert floor["parameters"] <= narrow <= ceiling["parameters"]

        app._saved_backtests["result-0001"].parameter_summary = (
            "绝对 1.5；等价 绝对 1.5 / 相对 1.5000% / 0.7794σ；收盘兜底：开启")
        BacktestApp._refresh_saved_pool_tree(app)
        widened = tree.column("parameters", "width")

        assert widened > narrow, "更长的参数串没有把这一列撑开"
        assert widened <= ceiling["parameters"], "撑开也不能越过上限"
        # 撑开一列不该顺带改动别的列，那正是「一列独吞余量」的老毛病。
        for key in ("name", "origin", "option", "strategy", "source",
                    "saved_at"):
            assert floor[key] <= tree.column(key, "width") <= ceiling[key]

        # 长的那条删掉之后缩回去：列宽跟的是当前这一池，不是历史最大值。
        BacktestApp._delete_saved_backtest(app, "result-0001")
        BacktestApp._refresh_saved_pool_tree(app)
        assert tree.column("parameters", "width") <= narrow
    finally:
        app.destroy()


def test_columns_are_fitted_into_the_container_instead_of_being_cut_off():
    """总宽超出容器时按比例削，而不是让右边的列消失。

    ttk.Treeview 没有横向滚动条：总宽超了它既不压缩也不提示，直接把右边的
    列裁掉，``minwidth`` 只在手动拖列边界时才起作用。此前「保留时间」整列
    看不见就是这么来的。
    """
    fit = BacktestApp._fit_pool_columns_to_width
    widths = {"a": 200, "b": 300, "c": 100}
    floors = {"a": 100, "b": 150, "c": 60}

    # 宽度还没算出来（构建那一刻 winfo_width() 是 1）：原样返回，等 Configure。
    assert fit(widths, floors, -49) == widths
    # 窄到连下限都排不下：再缩也只是把每列都压成看不清，维持原样。
    assert fit(widths, floors, 200) == widths

    tight = fit(widths, floors, 480)
    assert sum(tight.values()) <= 480
    assert all(tight[key] >= floors[key] for key in floors), "削过头了"
    assert all(tight[key] < widths[key] for key in widths), "一列都没削"

    roomy = fit(widths, floors, 900)
    assert sum(roomy.values()) <= 900
    assert all(roomy[key] > widths[key] for key in widths), "余量没分下去"
    # 余量按需求比例分，内容长的列多吃一点，不是平摊。
    assert roomy["b"] - widths["b"] > roomy["c"] - widths["c"]


def test_pool_cells_carry_padding_on_their_anchored_side():
    """结果池的文本列同样补内边距，不贴着列边界。"""
    padded = BacktestApp._pad_tree_cells(
        ("名称", "手工回测", "欧式", "每日收盘", "参数串",
         "模拟 · seed 42", "07-15 12:00:00"),
        [anchor for _k, _t, _w, anchor in BacktestApp._SAVED_POOL_COLUMNS])

    assert padded[0] == " 名称"
    assert padded[1] == "手工回测"
    assert padded[2] == "欧式"
    assert padded[4] == " 参数串"
    assert padded[6] == "07-15 12:00:00"


def test_pool_table_shows_which_option_each_result_tested():
    """结果池表要能直接看出一条测的是哪种期权。

    ``option_label`` 一直存在快照里，却只在保留成功的提示框和导出的
    meta_description 里露过面。此前想在池子里认出一条是香草还是累计，只能
    右键「加载明细」把当前回测顶掉重跑一遍。
    """
    app = _comparison_app(_snapshot())
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        assert app._saved_pool_tree.set("result-0001", "option").strip() == "欧式"
    finally:
        app.destroy()


def test_snapshot_detail_covers_every_input_group_with_shared_wording():
    """参数详情摊出五组输入，措辞与说明卡报差异时用的是同一套中文名。"""
    sections = BacktestApp._snapshot_detail_sections(_snapshot())
    assert [label for label, _rows in sections] == [
        "行情", "期权类型与参数", "头寸方向", "规模与成本", "对冲策略"]

    fields = {
        name: value
        for _label, rows in sections for name, value in rows
    }
    # 期权参数下钻到具体那一项，而不是一句"合约参数"。
    assert fields["期权类型"] == "欧式"
    assert fields["行权价"] == "100"
    assert fields["波动率"] == "0.2"
    # 取值一律回译成人话，与差异串同一个口径。
    assert fields["头寸方向"] == "买入"
    assert fields["策略类型"] == "每日收盘"
    assert fields["行情来源"] == "模拟"
    assert fields["收盘兜底"] == "关闭"


def _market_rows(**state_overrides):
    """按给定行情来源造一条快照，返回详情窗「行情」组的 {字段: 取值}。

    三个来源框在界面上永远同时填着东西，所以基础状态也照实填满：模拟跑出来
    的快照，Wind 代码框里仍留着上一次的代码。
    """
    state = dict(_state())
    state.update({
        "csv_path": "/data/x.csv", "csv_col": "close",
        "wind_code": "510050.SH", "wind_start": "2026-07-16",
        "wind_end": "2026-08-13", "wind_bar_size": gui_app.WIND_AUTO_BAR_SIZE,
    })
    state.update(state_overrides)
    return dict(dict(
        BacktestApp._snapshot_detail_sections(_snapshot(state=state)))["行情"])


def test_market_detail_lists_only_the_source_actually_used():
    """三种来源的字段互斥，只列本次真正用到的那一组。

    market_key 恒定记全部键（键集合恒定是差异比对的要求），没用到的那几项存
    的是**左侧控件当时的值**而不是空值，所以"渲染成 — 就跳过"拦不住：模拟跑
    出来的快照照样会列出 `CSV 列 close`、`标的代码 510050.SH`，而这次回测根
    本没碰过它们。
    """
    simulated = _market_rows(source="simulate")
    assert set(simulated) == {"行情来源", "随机种子"}
    assert simulated["行情来源"] == "模拟"

    csv = _market_rows(source="csv")
    assert set(csv) == {"行情来源", "CSV 文件", "CSV 列"}
    assert "随机种子" not in csv

    wind = _market_rows(source="wind", wind_bar_size="日频",
                        wind_bar_size_requested="日频")
    assert set(wind) == {
        "行情来源", "标的代码", "起始日", "截止日", "bar 粒度"}
    assert "CSV 列" not in wind and "随机种子" not in wind


def test_market_detail_shows_the_bar_size_auto_actually_resolved_to():
    """选了「自动（推荐）」时要显示推出来的实际粒度，而不是那个占位串。

    快照的 wind_bar_size 存的已经是解析后的实际粒度，下拉原值另存在
    bar_size_requested——只显示实际值看不出是自己挑的还是程序推的，只显示
    「自动（推荐）」更糟，那根本不是一个粒度。
    """
    auto = _market_rows(
        source="wind", wind_bar_size="15分钟",
        wind_bar_size_requested=gui_app.WIND_AUTO_BAR_SIZE)
    assert auto["bar 粒度"] == "15分钟（自动推荐）"

    # 手选的粒度不必加注解：它本来就是用户自己填的那个值。
    picked = _market_rows(source="wind", wind_bar_size="日频",
                          wind_bar_size_requested="日频")
    assert picked["bar 粒度"] == "日频"

    # 本字段上线前保留的快照没记原值，照常显示实际粒度即可。
    legacy = _market_rows(source="wind", wind_bar_size="5分钟")
    assert legacy["bar 粒度"] == "5分钟"

    # 实际粒度也没记下来时诚实说明，不把占位串当成粒度摆出来。
    unresolved = _market_rows(
        source="wind", wind_bar_size=gui_app.WIND_AUTO_BAR_SIZE,
        wind_bar_size_requested=gui_app.WIND_AUTO_BAR_SIZE)
    assert unresolved["bar 粒度"] == f"{gui_app.WIND_AUTO_BAR_SIZE} · 未记录实际粒度"


def test_requested_bar_size_stays_out_of_every_signature():
    """下拉原值是纯展示字段，不能进签名。

    进了 market_key 就改变了签名的键集合，新旧快照混选时
    _differing_field_names 两侧对不上、整体退回属性级——差异比对会被一个纯展
    示需求悄悄削弱一档。
    """
    auto = _snapshot(state=dict(
        _state(), source="wind", wind_code="510050.SH",
        wind_start="2026-07-16", wind_end="2026-08-13",
        wind_bar_size="15分钟",
        wind_bar_size_requested=gui_app.WIND_AUTO_BAR_SIZE))
    picked = _snapshot("result-0002", "手选", state=dict(
        _state(), source="wind", wind_code="510050.SH",
        wind_start="2026-07-16", wind_end="2026-08-13",
        wind_bar_size="15分钟", wind_bar_size_requested="15分钟"))

    assert auto.bar_size_requested == gui_app.WIND_AUTO_BAR_SIZE
    assert picked.bar_size_requested == "15分钟"
    # 实际跑的粒度相同 → 行情属性必须判为一致，与"我在下拉里选的是什么"无关。
    assert auto.market_key == picked.market_key
    summary = BacktestApp._comparison_variable_summary([auto, picked])
    assert summary["state"] == "identical", summary["headline"]


_INTRADAY_INDEX = pd.DatetimeIndex(
    [f"2026-01-0{day} {hour}:00"
     for day in (2, 5, 6) for hour in (10, 11, 13, 14)])


def _economics_rows(*, timestamps=None, steps_per_day=1, days=3,
                    **state_overrides):
    """造一条快照并返回详情窗「规模与成本」组的 {字段: 取值}。

    给了 ``timestamps`` 就按 bar 级重造全部数组：``_result()`` 那套是 3 根
    bar，而日内索引有十几根，长度对不上聚合会直接抛错。``trading_day_groups``
    也要一起给——聚合优先按它分组，缺了就退化成按 steps_per_day 切，那正是
    本组测试要区分的两件事。
    """
    state = dict(_state())
    state.update(state_overrides)
    result = _result()
    result["steps_per_day"] = steps_per_day
    if timestamps is not None:
        bars = len(timestamps)
        per_day = bars // days
        result.update({
            "net_daily": np.linspace(-1.0, 1.0, bars),
            "tc_paid": np.full(bars, 0.1),
            "prices": np.linspace(100.0, 102.0, bars),
            "shares": np.linspace(1.0, 0.0, bars),
            "hedge_triggered": np.full(bars, True),
            "trading_day_groups": np.repeat(np.arange(days), per_day),
        })
    snapshot = _snapshot(
        state=state, result=result, timestamps=timestamps)
    return snapshot, dict(dict(
        BacktestApp._snapshot_detail_sections(snapshot))["规模与成本"])


def test_intraday_steps_show_what_the_engine_actually_used():
    """真实行情的「日内采样」要写引擎实际用的 bar 数，不是那个占位 1。

    GUI 对真实行情一律传占位 1（``_gui_steps_per_day``，CSV 分支干脆传
    ``None``），引擎再自己从时间索引推断。照签名显示的话，一条 15 分钟 Wind
    回测会写着「日内采样 1」，而它每天实际有好几根 bar。
    """
    snapshot, rows = _economics_rows(
        timestamps=_INTRADAY_INDEX, steps_per_day=1,
        source="wind", wind_code="510050.SH", wind_start="2026-01-02",
        wind_end="2026-01-06", wind_bar_size="15分钟")

    assert snapshot.intraday_steps == 4
    assert rows["日内采样"] == "4"
    # 签名不动：它记的是传给引擎的输入，改它会让新旧快照混选时假报差异。
    assert dict(snapshot.economics_key)["steps_per_day"] == 1


def test_intraday_steps_keep_the_simulated_value_users_typed():
    """模拟的采样密度是用户直接填的，签名里那个值就是事实。"""
    _snap, rows = _economics_rows(steps_per_day=8, source="simulate")
    assert rows["日内采样"] == "8"


def test_intraday_steps_admit_when_a_legacy_snapshot_never_recorded_them():
    """旧快照只记了占位值，说不出实际粒度就别假装说得出。"""
    snapshot, _rows = _economics_rows(
        timestamps=_INTRADAY_INDEX, steps_per_day=1,
        source="wind", wind_code="510050.SH", wind_start="2026-01-02",
        wind_end="2026-01-06", wind_bar_size="15分钟")
    snapshot.intraday_steps = 0
    rows = dict(dict(
        BacktestApp._snapshot_detail_sections(snapshot))["规模与成本"])
    assert rows["日内采样"] == "未记录（旧快照存的是占位值）"

    # 模拟来源的旧快照不受影响：它那个值本来就是真的。
    simulated, _r = _economics_rows(steps_per_day=8, source="simulate")
    simulated.intraday_steps = 0
    assert dict(dict(BacktestApp._snapshot_detail_sections(
        simulated))["规模与成本"])["日内采样"] == "8"


def test_fixed_time_strategy_rejects_daily_csv_before_loading_prices(tmp_path):
    """日频 CSV 配固定时刻策略要提前拒绝，与 Wind 日频那条对齐。

    此前 `_validate_fixed_time_source_state` 拦了模拟和 Wind 日频，唯独漏了
    CSV —— 日频 CSV 一声不吭就跑完，而每个交易日只有一行，根本没有可供固定
    时刻命中的日内时间戳。
    """
    def write(name, index):
        path = tmp_path / name
        pd.DataFrame(
            {"close": np.linspace(100.0, 102.0, len(index))},
            index=index).to_csv(path)
        return str(path)

    daily = write("daily.csv", pd.DatetimeIndex(
        ["2026-01-02", "2026-01-05", "2026-01-06"]))
    intraday = write("intraday.csv", _INTRADAY_INDEX)

    with pytest.raises(ValueError, match="日频数据"):
        BacktestApp._validate_fixed_time_source_state(
            {"strategy_name": "fixed_times", "source": "csv",
             "csv_path": daily})

    # 分钟级 CSV 照常放行。
    BacktestApp._validate_fixed_time_source_state(
        {"strategy_name": "fixed_times", "source": "csv",
         "csv_path": intraday})

    # 读不出来时放行，把报错留给随后的 from_csv —— 它给的消息更精确。
    BacktestApp._validate_fixed_time_source_state(
        {"strategy_name": "fixed_times", "source": "csv",
         "csv_path": str(tmp_path / "缺失.csv")})

    # 非固定时刻策略一律不受这条约束。
    BacktestApp._validate_fixed_time_source_state(
        {"strategy_name": "close_to_close", "source": "csv",
         "csv_path": daily})


def test_snapshot_detail_drops_placeholders_and_the_data_digest():
    """签名里的占位项与哈希不进详情窗。

    键集合恒定是差异比对的要求：每日收盘的快照带着 ``fixed_times=None``，
    模拟行情的带着 ``csv_path=None``。人看的详情列出来只会让真正有值的那几
    行更难找。行情数据摘要则是一串 sha256——比对时它还能说"确实不是同一段
    数据"，单看一条时什么也没说。
    """
    names = {
        name
        for _label, rows in BacktestApp._snapshot_detail_sections(_snapshot())
        for name, _value in rows
    }
    assert "随机种子" in names, "模拟行情的种子是有值的，不该被当成占位项"
    for absent in ("固定时刻", "带宽阈值", "CSV 文件", "标的代码", "行情数据"):
        assert absent not in names, f"{absent} 没有取值，不该占一行"


def _typed_snapshot(cls_name, subtype, result_id="result-0001", name="结果 A",
                    **param_overrides):
    """按某个期权大类的默认参数造一条快照，用来验证按类取标签。"""
    params = {
        spec[0]: spec[3] for spec in gui_app.OPTION_CLASSES[cls_name]["params"]
    }
    params.update(param_overrides)
    state = dict(_state(), cls_name=cls_name, subtype=subtype, params=params)
    return _snapshot(result_id, name, state=state)


def test_option_parameters_are_read_with_their_own_class_definitions():
    """同一个键在不同大类下是不同的东西，标签与取值都要按本类的定义取。

    此前 ``_option_param_labels`` 全局 setdefault 合并，按 OPTION_CLASSES 的
    定义序永远是香草那一套胜出。于是亚式的观察日数被叫成「杠杆倍数」（累计的
    `N`），雪球的 `cp=-1` 被译成「看跌 (Put)」——而雪球的 `-1` 是「雪球
    (卖看跌)」，译错的不是名字而是取值本身。
    """
    snowball = dict(BacktestApp._snapshot_detail_sections(
        _typed_snapshot("雪球期权 (Snowball)", "Opt_Snowball")))
    fields = dict(snowball["期权类型与参数"])
    assert fields["方向"] == "雪球 (卖看跌)", "雪球的方向被按别的期权回译了"
    assert "最新价 S0" in fields and "入场价 S00" in fields
    assert "剩余期限(交易日)" in fields

    asian = dict(BacktestApp._snapshot_detail_sections(
        _typed_snapshot("亚式期权 (Asian)", "Asian")))
    assert "观察日数" in dict(asian["期权类型与参数"]), "亚式的 N 不是杠杆倍数"


def test_option_parameters_are_listed_in_full_in_their_form_order():
    """每种期权定义了几项就列几项，且按左侧表单的定义顺序排。

    合约参数全是数值型，不存在"取不到值"的情况，所以详情窗略过空值的规则
    不会让它们漏项。顺序按本类定义走——用香草的顺序去排一条雪球，读起来就
    是打乱的。
    """
    for cls_name, subtype in (
            ("雪球期权 (Snowball)", "Opt_Snowball"),
            ("累计期权 (Decumulator)", "Opt_ASGQ_DFF"),
            ("亚式期权 (Asian)", "Asian"),
            ("气囊期权 (Airbag)", "Opt_Airbag"),
            ("香草期权 (Vanilla)", "Eu")):
        sections = dict(BacktestApp._snapshot_detail_sections(
            _typed_snapshot(cls_name, subtype)))
        shown = [name for name, _v in sections["期权类型与参数"]]
        declared = [
            str(spec[1])
            for spec in gui_app.OPTION_CLASSES[cls_name]["params"]]
        # 组里前两行是大类与类型，其后应当是本类参数的原样顺序。
        assert shown[:2] == ["期权大类", "期权类型"], cls_name
        assert shown[2:] == declared, f"{cls_name} 的参数漏项或乱序"


def test_option_parameter_diff_uses_the_shared_class_definitions():
    """差异串也走按类回译，不然亚式的观察日数会被报成杠杆倍数。"""
    summary = BacktestApp._comparison_variable_summary([
        _typed_snapshot("亚式期权 (Asian)", "Asian", "result-0001", "少", N=10),
        _typed_snapshot("亚式期权 (Asian)", "Asian", "result-0002", "多", N=20),
    ])
    assert ("观察日数", ["10", "20"]) in summary["fields"]


def test_snapshot_params_window_opens_without_a_replay_recipe():
    """参数详情读的是签名，没有重放配方也照样打得开。

    这正是它与「加载明细」的分工：那一条要重放才拿得到逐日损益，本功能上线
    前保留的快照没有配方，点了只能弹一句"没有配方"。
    """
    app = _comparison_app(_snapshot())
    try:
        snapshot = app._saved_backtests["result-0001"]
        snapshot.replay = {}
        window = BacktestApp._open_snapshot_params_window(app, snapshot)
        app.update_idletasks()

        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, gui_app.tk.Text):
                    texts.append(child.get("1.0", "end"))
                walk(child)

        walk(window)
        assert len(texts) == 1
        body = texts[0]
        assert "期权类型与参数" in body and "期权类型" in body and "欧式" in body
        assert "行权价" in body and "100" in body
        window.destroy()
    finally:
        app.destroy()


def test_variable_detail_reads_in_the_declared_field_order():
    """字段按定义顺序而非签名字典序输出，"策略类型、带宽阈值"更顺。"""
    close = _snapshot()
    band = _snapshot(
        "result-0002", "带宽",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"))

    text = _variable_text([close, band])

    assert "对冲策略（策略类型：" in text and "；带宽阈值：" in text


def test_single_field_aspect_does_not_repeat_itself_in_brackets():
    """头寸方向这个属性只有一个同名字段，不该写成「头寸方向（头寸方向）」。"""
    buy = _snapshot("result-0001", "买方", state=_state(position=-1))
    sell = _snapshot("result-0002", "卖方", state=_state(position=1))

    text = _variable_text([buy, sell])

    assert "变量：头寸方向" in text
    assert "头寸方向（头寸方向）" not in text


def test_long_field_lists_are_truncated_with_a_count():
    """一个属性里变了很多字段时不要铺开写满一行。"""
    values = [
        (("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1)),
        (("a", 2), ("b", 2), ("c", 2), ("d", 2), ("e", 2)),
    ]
    names = BacktestApp._differing_field_names(values)
    assert names == [(key, ["1", "2"]) for key in "abcde"]

    band = _snapshot(
        "result-0002", "带宽",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"))
    other = _snapshot(
        "result-0003", "另一个",
        state=dict(_state(strategy="fixed_times"),
                   fixed_times="10:30", sigma_source="realized",
                   sigma_window=30, force_day_close_hedge=True),
        result=_result(strategy="fixed_times"))

    summary = BacktestApp._comparison_variable_summary([band, other])
    # 字段多了就纵向排开，不再截断成"共 N 项不同"。
    assert len(summary["fields"]) > 3


def test_option_parameter_diff_names_the_actual_parameter():
    """期权参数是嵌套结构，差异要下钻到具体那一项并带上取值。

    只比顶层键的话，两条只差波动率的结果会被报成"期权（合约参数）"——说了
    等于没说，用户还得自己去翻是哪个参数。
    """
    def with_params(result_id, name, **overrides):
        params = {"s0": 100.0, "K": 100.0, "sigma": 0.18, "T_days": 22}
        params.update(overrides)
        return _snapshot(result_id, name, state=dict(_state(), params=params))

    base = with_params("result-0001", "基准")

    vol = BacktestApp._comparison_variable_summary(
        [base, with_params("result-0002", "低波动", sigma=0.15)])
    assert vol["headline"] == "本次对比的变量：期权类型与参数"
    assert vol["fields"] == [("波动率", ["0.18", "0.15"])]

    strike = BacktestApp._comparison_variable_summary(
        [base, with_params("result-0003", "高行权价", K=105.0)])
    assert ("行权价", ["100", "105"]) in strike["fields"]

    term = BacktestApp._comparison_variable_summary(
        [base, with_params("result-0004", "长期限", T_days=44)])
    assert ["22", "44"] in [values for _label, values in term["fields"]]


def test_diff_values_are_translated_into_chinese():
    """差异里的取值要说人话，不是 close_to_close / -1 这种内部值。"""
    buy = _snapshot("result-0001", "买方", state=_state(position=-1))
    sell = _snapshot("result-0002", "卖方", state=_state(position=1))
    direction = BacktestApp._comparison_variable_summary([buy, sell])
    assert direction["headline"] == "本次对比的变量：头寸方向"
    assert direction["fields"] == [("头寸方向", ["买入", "卖出"])]

    band = _snapshot(
        "result-0003", "带宽",
        state=_state(strategy="hedge_band", threshold=2.0),
        result=_result(strategy="hedge_band"))
    fields = dict(BacktestApp._comparison_variable_summary(
        [buy, band])["fields"])
    assert fields["策略类型"][0] == "每日收盘"
    assert "close_to_close" not in str(fields["策略类型"])


def test_unlabelled_fields_keep_a_stable_order():
    """路径集合是 frozenset，未列入定义表的字段也必须每次同序。"""
    values = [
        (("zeta", 1), ("alpha", 1), ("mid", 1)),
        (("zeta", 2), ("alpha", 2), ("mid", 2)),
    ]
    first = BacktestApp._differing_field_names(values)
    assert first == BacktestApp._differing_field_names(values)
    assert len(first) == 3


def test_replay_segments_are_told_apart_even_within_one_day():
    """同一天里滚动出来的两段，行情签名必须不同。

    重放区间原先只格式化到天，日内滚动的相邻两段起止日一样，两条快照的
    五组签名就全都相同——页面于是说"所选结果的输入完全相同"，而它们的
    指标明明不同。
    """
    def replay_state(window_id, start, end):
        return dict(
            _state(strategy="hedge_band"),
            source="wind", wind_code="510050.SH",
            wind_start=start, wind_end=end, wind_bar_size="15分钟",
            history_replay_strategy="固定间隔(1σ)",
            history_replay_lookback="week",
            history_replay_window_id=window_id,
        )

    morning = _snapshot(
        "result-0001", "上午段",
        state=replay_state("window_1", "2026-08-05 09:30", "2026-08-05 11:15"),
        result=_result(strategy="hedge_band"))
    afternoon = _snapshot(
        "result-0002", "下午段",
        state=replay_state("window_2", "2026-08-05 13:00", "2026-08-05 14:45"),
        result=_result(strategy="hedge_band"))

    assert morning.market_key != afternoon.market_key
    summary = BacktestApp._comparison_variable_summary([morning, afternoon])
    assert summary["state"] == "single"
    assert "行情" in summary["headline"]


def test_identical_data_is_not_reported_as_a_difference():
    """两段的价格序列逐字节相同时不能报差异，哪怕它们编号不同。

    跨周期取到的末段常常是同一段数据（残段固定放在最老一端），拿分段编号
    当区分依据就会把它们报成"行情不同"——而指标一模一样，用户无从理解。
    """
    def replay(result_id, window_id, price_shift=0.0):
        return _snapshot(
            result_id, result_id,
            state=dict(
                _state(strategy="hedge_band"),
                source="wind", wind_code="510050.SH",
                wind_start="2026-08-05 09:30", wind_end="2026-08-05 11:15",
                history_replay_window_id=window_id),
            result=_result(strategy="hedge_band", price_shift=price_shift))

    # 同一份数据、不同分段编号 → 判为相同。
    same = BacktestApp._comparison_variable_summary(
        [replay("result-0001", "segment_3"),
         replay("result-0002", "segment_6")])
    assert same["state"] == "identical"

    # 数据真的不同 → 认出来，但不把哈希摊到界面上。
    differing = BacktestApp._comparison_variable_summary(
        [replay("result-0003", "segment_1"),
         replay("result-0004", "segment_2", price_shift=5.0)])
    assert differing["state"] == "single"
    assert differing["headline"] == "本次对比的变量：行情"
    # 摘要没有可读取值，只说"不同"。
    assert differing["fields"] == [("行情数据", None)]


def test_readable_fields_win_over_the_opaque_digest():
    """可读字段能说清差异时，不再补一句"行情数据不同"——同一件事说两遍。"""
    def wind(result_id, start, shift):
        return _snapshot(
            result_id, result_id,
            state=dict(_state(), source="wind", wind_code="510050.SH",
                       wind_start=start, wind_end="2026-08-12"),
            result=_result(price_shift=shift))

    summary = BacktestApp._comparison_variable_summary(
        [wind("result-0001", "2026-08-05", 0.0),
         wind("result-0002", "2026-08-06", 3.0)])

    fields = dict(summary["fields"])
    assert fields["起始日"] == ["2026-08-05", "2026-08-06"]
    assert "行情数据" not in fields


def test_manual_and_replay_snapshots_stay_comparable():
    """手工快照与分段重放跑的是同一份数据时判为相同，不该凭空差一项。"""
    manual = _snapshot("result-0001", "手工")
    replayed = _snapshot(
        "result-0002", "重放",
        state=dict(_state(), history_replay_window_id="window_3"))

    assert manual.market_key == replayed.market_key
    assert BacktestApp._comparison_variable_summary(
        [manual, replayed])["state"] == "identical"


def test_replay_snapshot_records_the_segment_scaled_option():
    """分段实际用的期权按该段首价缩放过，快照要存缩放后的值。

    不存的话两段的期权签名恒相同——哪怕实际行权价差了 50%，对比页也会说
    "期权：相同"。
    """
    import pandas as pd
    from pricing import HedgeBandStrategy
    from pricing.Option_Vanilla import Option_Vanilla

    def spec_at(first_price):
        index = pd.date_range("2026-08-05 09:30", periods=20, freq="15min")
        option = Option_Vanilla(
            "Vanilla", s0=first_price, sr=[], K=first_price * 0.9, T=22,
            sigma=0.18, cp=1, r=0.03, q=0.03)
        return SimpleNamespace(
            lookback="week", window_id="window_1",
            external_path=pd.Series([first_price] * 20, index=index),
            strategies={"固定间隔(1σ)": HedgeBandStrategy("sigma", 1.0)},
            metadata={}, option=option)

    fake = SimpleNamespace(_latest_history_state=dict(
        _state(strategy="hedge_band"), source="wind",
        params={"s0": 100.0, "K": 90.0, "sigma": 0.18, "T_days": 22}))

    cheap = BacktestApp._history_replay_gui_state(
        fake, spec_at(100.0), "固定间隔(1σ)")
    rich = BacktestApp._history_replay_gui_state(
        fake, spec_at(150.0), "固定间隔(1σ)")

    assert cheap["params"]["K"] == 90.0
    assert rich["params"]["s0"] == 150.0
    assert rich["params"]["K"] == 135.0
    # 不受缩放影响的参数原样保留。
    assert rich["params"]["sigma"] == 0.18
    assert rich["params"]["T_days"] == 22


def test_scaled_params_tolerate_a_missing_option():
    """载入的结果包里可能没有期权对象，此时原样返回不报错。"""
    params = {"s0": 100.0, "K": 90.0}
    assert BacktestApp._replay_scaled_params(params, None) == params
    assert BacktestApp._replay_scaled_params(None, None) == {}


# ---------------------------------------------------------------------------
#  持久化：重开程序还在、序号不撞车、明细能重放
# ---------------------------------------------------------------------------

def _persisted_app():
    """构造真实 app 并返回它；池目录由 conftest 的 autouse 夹具隔离。"""
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()
    app = gui_app.BacktestApp()
    app.withdraw()
    return app


def _run_and_retain(app, name, *, spd="4"):
    """真跑一次模拟回测并保留，返回快照。"""
    app._spd_var.set(spd)
    state = app._collect_gui_state()
    backtest = app._build_backtest(state)
    backtest.run()
    app._latest_backtest = backtest
    app._latest_backtest_state = state
    app._latest_retained_result_id = None
    return BacktestApp._store_current_backtest(app, name), backtest


def test_retained_result_survives_a_restart():
    """重开程序后结果还在，且字段逐个还原。

    NaN 要单独比：``summary_row["position"]`` 缺失时就是 NaN，而 NaN != NaN
    是 Python 语义，不是往返失真。
    """
    app = _persisted_app()
    try:
        snapshot, _bt = _run_and_retain(app, "重启前 #01")
        assert snapshot.store_path and os.path.exists(snapshot.store_path)
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert list(reopened._saved_backtests) == ["result-0001"]
        back = reopened._saved_backtests["result-0001"]
        assert back.name == "重启前 #01"
        assert back.position == snapshot.position
        assert back.origin == snapshot.origin
        assert back.daily_frame.equals(snapshot.daily_frame)
        # 签名 key 必须仍是嵌套 tuple，否则差异卡认不出嵌套。
        for field_name in ("market_key", "contract_key", "economics_key"):
            value = getattr(back, field_name)
            assert isinstance(value, tuple), field_name
            assert value == getattr(snapshot, field_name), field_name
        for key, expected in snapshot.summary_row.items():
            actual = back.summary_row[key]
            if isinstance(expected, float) and math.isnan(expected):
                assert math.isnan(actual), key
            else:
                assert actual == expected, key
    finally:
        reopened.destroy()


def test_saving_after_a_restart_does_not_overwrite_loaded_results():
    """载入旧池后再保存，绝不能顶掉盘里那条。

    序号计数器每次启动归零，而入池是按 key 直接赋值、没有存在性检查——不把
    计数器推到已有最大值之后，第一次「保留」就会静默吃掉 ``result-0001``。
    静默是这里最要命的部分：用户只会发现池子少了一条。
    """
    app = _persisted_app()
    try:
        _run_and_retain(app, "第一条")
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert reopened._saved_backtest_sequence == 1
        _run_and_retain(reopened, "第二条")
        assert sorted(reopened._saved_backtests) == [
            "result-0001", "result-0002"]
        assert reopened._saved_backtests["result-0001"].name == "第一条"
        assert reopened._saved_backtests["result-0002"].name == "第二条"
    finally:
        reopened.destroy()

    again = _persisted_app()
    try:
        assert len(again._saved_backtests) == 2
    finally:
        again.destroy()


def test_comparison_order_follows_sequence_not_result_id():
    """保存顺序由显式序号决定，不再靠 ``result_id`` 字典序。

    字典序在第 10000 条溢出（``result-10000`` 会排到 ``result-9999`` 前），
    而乱序不会有任何断言挂掉，只是图上的曲线顺序变了。
    """
    late = _snapshot("result-0002", "后保存的")
    late.sequence = 10000
    early = _snapshot("result-10000", "先保存的")
    early.sequence = 2
    summary, _curves = BacktestApp._saved_comparison_payload([late, early])
    assert summary["strategy"].tolist() == ["先保存的", "后保存的"]


def test_deleting_a_result_removes_its_file():
    app = _persisted_app()
    try:
        snapshot, _bt = _run_and_retain(app, "待删除")
        path = snapshot.store_path
        assert os.path.exists(path)
        BacktestApp._delete_saved_backtest(app, snapshot.result_id)
        assert not os.path.exists(path)
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert reopened._saved_backtests == {}
    finally:
        reopened.destroy()


def test_clearing_the_pool_takes_every_file_with_it(monkeypatch):
    """「全部清空」= 逐条删除的批量版：确认之后连文件一起没。

    先验一次"点了取消什么都不发生"——这个按钮不可撤销，确认框失灵比它
    根本没做还糟。
    """
    app = _persisted_app()
    try:
        first, _bt = _run_and_retain(app, "待清空 #01")
        second, _bt = _run_and_retain(app, "待清空 #02")
        paths = [first.store_path, second.store_path]
        app._saved_comparison_selection = set(app._saved_backtests)
        # 按钮只存在于对比页上，清空后要刷新的也是它——先把页面建出来，
        # 否则验的是一条真实点击走不到的路径。
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()

        monkeypatch.setattr(
            gui_app.messagebox, "askyesno", lambda *_a, **_kw: False)
        BacktestApp._clear_saved_backtest_pool(app)
        assert len(app._saved_backtests) == 2, "点了取消不该删任何东西"
        assert all(os.path.exists(path) for path in paths)

        asked = []
        monkeypatch.setattr(
            gui_app.messagebox, "askyesno",
            lambda title, message: asked.append((title, message)) or True)
        BacktestApp._clear_saved_backtest_pool(app)

        assert len(asked) == 1 and "2 条" in asked[0][1], asked
        assert app._saved_backtests == {}
        assert app._saved_comparison_selection == set()
        assert not any(os.path.exists(path) for path in paths)
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert reopened._saved_backtests == {}
    finally:
        reopened.destroy()


def test_chart_draws_only_the_top_curves_and_says_where_the_rest_went():
    """图上有条数闸门，指标表和导出不受限。

    691×226 px 的画布、12 色取模，勾满二十条谁也认不出谁；但「全选 →
    点列头排序 → 看第一行」这条路要留着，所以闸门只加在图上。上限与策略优
    选页取齐——两页对"几条还看得清"该给同一个答案。
    """
    limit = gui_app.MAX_COMPARISON_CHART_CURVES
    app = _comparison_app(*[
        _snapshot(f"result-{index:04d}", f"结果 {index}",
                  result=_result(price_shift=index))
        for index in range(1, limit + 4)])
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()

        drawn = [
            name for name, var in app._comparison_curve_vars.items()
            if var.get()]
        assert len(drawn) == limit
        assert len(app._comparison_daily_curves) == limit + 3
        assert len(app._comparison_rows) == limit + 3, "指标表不受闸门影响"
        hint = app._comparison_curve_hint_var.get()
        assert f"{limit} 条" in hint and "其余 3 条" in hint, hint

        # 闸门跟着排序走：换一列排序，图上就换成那一列的前几名。
        # 这一步要能证伪"闸门按插入序截断"，排序结果就必须与插入序不同。
        # 总成本在这批夹具里恒等（tc_paid 固定），键全相等时稳定排序原样
        # 返回输入序，也就是插入序——照它断言，闸门取插入序前八也能通过。
        # 成交额随 price_shift 逐行递增，但升序同样等于插入序，所以点两下
        # 换成降序：此时前八是成交额最大的八条，与插入序前八没有交集。
        BacktestApp._sort_comparison_by(app, "turnover")
        BacktestApp._sort_comparison_by(app, "turnover")
        app.update_idletasks()
        reordered = [
            name for name, var in app._comparison_curve_vars.items()
            if var.get()]
        assert len(reordered) == limit
        ranked = [
            app._comparison_rows[iid]["strategy"]
            for iid in app._comparison_tree.get_children()]
        assert set(reordered) == set(ranked[:limit])
        # 上一条断言只有在两种顺序确实不同时才有牙——夹具一旦退化成"随便
        # 怎么排前八都一样"，它就会变成一条永远通过的空断言。
        assert set(ranked[:limit]) != set(
            list(app._comparison_daily_curves)[:limit]), "夹具没能区分两种顺序"
    finally:
        app.destroy()


def test_chart_curve_hint_stays_quiet_when_nothing_is_held_back():
    """没超过上限就不该出现那句解释——它是为"少画了几条"存在的。"""
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        assert app._comparison_curve_hint_var.get() == ""
        assert all(var.get() for var in app._comparison_curve_vars.values())
    finally:
        app.destroy()


def test_same_strategy_results_stay_telling_apart_on_the_chart():
    """同一策略跑多次，曲线要分得开。

    配色按策略身份取是有意的（同一策略在优选页与本页同色），但换区间、换行
    情、换方向重跑同一策略恰恰是本页头号用途，不加处理时那几条会拿到完全相
    同的颜色和标记。第一条必须严格保持登记表原色，跨页对照靠的就是它。
    """
    app = _comparison_app(
        _snapshot("result-0001", "控制A"),
        _snapshot("result-0002", "控制B"),
        _snapshot("result-0003", "控制C"),
        _snapshot("result-0004", "带宽",
                  state=_state(strategy="hedge_band", threshold=1.0)),
    )
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        colors, dashes = app._comparison_color_map, app._comparison_dash_map

        same = ["控制A", "控制B", "控制C"]
        assert len({dashes[name] for name in same}) == 3, "同策略得靠线型分开"
        # 同策略也要逐条换明度。此前是每四条才升一档，前四条严格同色、区分全
        # 压在线型上——而虚线与点划线在这块画布的细线宽下基本读不出来。
        assert len({colors[name] for name in same}) == 3, (
            "同策略的几条必须颜色也不同，只靠线型分不开")
        assert colors["控制A"] == app._strategy_style("close_to_close")[0], (
            "每组第一条仍要与策略优选页严格同色，跨页对照靠的就是这一点")
        assert colors["带宽"] != colors["控制A"], "不同策略本来就该不同色"
    finally:
        app.destroy()


def _relative_luminance(color):
    channels = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def test_shade_ladder_alternates_light_and_dark_around_the_base_colour():
    """明度档要一深一浅围着原色走，0 档原样返回。

    一味往浅里推的话第三档就接近白色，浅色画布上直接看不见了——此前那版
    正是单向变浅，而且要等线型用完一轮（每四条）才升一档。
    """
    base = "#2563EB"
    assert BacktestApp._shift_hex(base, 0) == base

    lighter = BacktestApp._shift_hex(base, 1)
    darker = BacktestApp._shift_hex(base, 2)
    assert _relative_luminance(lighter) > _relative_luminance(base)
    assert _relative_luminance(darker) < _relative_luminance(base)

    # 档位循环：3 档与 4 种线型互质，组合到第 12 条才重复。
    assert BacktestApp._shift_hex(base, 3) == base
    assert len(gui_app.STRATEGY_CHART_SHADES) == 3
    assert math.gcd(len(gui_app.STRATEGY_CHART_SHADES),
                    len(gui_app.STRATEGY_CHART_DASHES)) == 1

    # 非法输入不能把整张图带崩，原样退回即可。
    assert BacktestApp._shift_hex("tab:blue", 2) == "tab:blue"


def test_shade_ladder_stays_legible_across_every_palette_colour():
    """三档明度在每种基色下都要拉得开，且不会浅到看不见或黑成一团。

    档数少正是为了调得开：摊成八档时最小亮度差只剩 8，那种深浅在
    691×226 px 的画布上根本读不出来。
    """
    for base in gui_app.STRATEGY_CHART_COLORS:
        shades = [
            BacktestApp._shift_hex(base, step)
            for step in range(len(gui_app.STRATEGY_CHART_SHADES))
        ]
        assert len(set(shades)) == len(shades), f"{base} 的档位撞色"
        lums = sorted(_relative_luminance(color) for color in shades)
        gaps = [high - low for low, high in zip(lums, lums[1:])]
        assert min(gaps) >= 25, f"{base} 相邻档太接近：{gaps}"
        # 白底上太浅会消失，压过头则糊成一团黑。
        assert lums[-1] <= 215, f"{base} 最浅档看不见"
        assert lums[0] >= 20, f"{base} 最深档糊成黑"


def test_same_strategy_curves_never_repeat_a_colour_dash_pairing():
    """八条同策略曲线各有独一无二的 (颜色, 线型)，且相邻两条两样都不同。

    区分力来自 3 档明度与 4 种线型的**组合**（互质，到第 12 条才重复），不是
    八种颜色——八档明度实测最小亮度差只剩 8，在 691×226 px 的画布上读不出来。
    此前明度是 ``index // 4``，前四条严格同色、区分全压在线型上，而虚线与点
    划线在细线宽的密集 PnL 曲线里分不出来。
    """
    app = _comparison_app()
    try:
        app._comparison_daily_curves = {f"第{i}条": None for i in range(8)}
        keys = {name: "close_to_close" for name in app._comparison_daily_curves}
        colors, dashes = BacktestApp._comparison_curve_styles(app, keys)

        ordered = list(app._comparison_daily_curves)
        pairs = [(colors[name], str(dashes[name])) for name in ordered]
        assert len(set(pairs)) == 8, "(颜色, 线型) 组合不得重复"
        assert len(set(colors.values())) == len(gui_app.STRATEGY_CHART_SHADES)

        # 相邻两条必定同时换明度与线型——挨着画的那两条最容易看混。
        for left, right in zip(ordered, ordered[1:]):
            assert colors[left] != colors[right], f"{left}/{right} 同色"
            assert dashes[left] != dashes[right], f"{left}/{right} 同线型"
    finally:
        app.destroy()


def test_deleting_a_multi_row_selection_takes_all_their_files(monkeypatch):
    """多选行后删除：确认框点名、条数对得上、被选中的那几个文件都没了。

    作用对象是行选择而不是「显示」勾选集——勾选集跨会话落盘、每保留一条新
    结果自动勾上，还能被「全选」一键赋成全池，拿它当删除范围就等于给
    「全选 → 删除」这两下配了一条清空整池的近路。
    """
    app = _persisted_app()
    try:
        first, _bt = _run_and_retain(app, "批删 #01")
        second, _bt = _run_and_retain(app, "批删 #02")
        third, _bt = _run_and_retain(app, "批删 #03")
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        app._saved_pool_tree.selection_set(
            [first.result_id, third.result_id])

        asked = []
        monkeypatch.setattr(
            gui_app.messagebox, "askyesno",
            lambda title, message: asked.append((title, message)) or True)
        BacktestApp._prompt_delete_saved_backtest(app)

        assert len(asked) == 1, "一次批量删除只该确认一次"
        title, message = asked[0]
        assert title == "删除选中的结果"
        assert "2 条" in message, message
        assert "『批删 #01』" in message and "『批删 #03』" in message
        assert "删除后结果池还剩 1 条" in message
        assert "此操作不可撤销" in message

        assert set(app._saved_backtests) == {second.result_id}
        assert not os.path.exists(first.store_path)
        assert not os.path.exists(third.store_path)
        assert os.path.exists(second.store_path), "没选中的那条不能受牵连"
    finally:
        app.destroy()


def test_deleting_every_row_says_the_pool_will_be_empty(monkeypatch):
    """选满整池时它和「全部清空」是同一件事，确认框必须说出来。"""
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        app._saved_pool_tree.selection_set(list(app._saved_backtests))

        asked = []
        monkeypatch.setattr(
            gui_app.messagebox, "askyesno",
            lambda title, message: asked.append(message) or True)
        BacktestApp._prompt_delete_saved_backtest(app)

        assert "删除后结果池将变空。" in asked[0], asked
        assert app._saved_backtests == {}
    finally:
        app.destroy()


def test_single_row_delete_keeps_its_original_wording(monkeypatch):
    """单条删除的标题与正文一个字都没变——批量只是多了一条分支。"""
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        app._saved_pool_tree.selection_set("result-0001")

        asked = []
        monkeypatch.setattr(
            gui_app.messagebox, "askyesno",
            lambda title, message: asked.append((title, message)) or False)
        BacktestApp._prompt_delete_saved_backtest(app)

        assert asked == [(
            "删除回测结果",
            "删除『甲』？将同时删掉本机上的结果文件，此操作不可撤销。")]
        assert len(app._saved_backtests) == 2, "点了取消不该删任何东西"
    finally:
        app.destroy()


def test_right_click_keeps_a_multi_row_selection_and_counts_it():
    """右键点在选区内保留选区，点在选区外才重置；单条动作多选时置灰。

    反过来做的话选区在菜单弹出前就被打散了，永远右键不出多条。
    """
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"),
        _snapshot("result-0003", "丙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        tree, menu = app._saved_pool_tree, app._saved_pool_menu
        menu.tk_popup = lambda x_root, y_root: None
        event = SimpleNamespace(x=10, y=10, x_root=0, y_root=0)

        tree.selection_set(["result-0001", "result-0002"])
        tree.identify_row = lambda _y: "result-0002"
        BacktestApp._popup_saved_pool_menu(app, event)
        assert set(tree.selection()) == {"result-0001", "result-0002"}
        assert menu.entrycget(
            BacktestApp._POOL_MENU_DELETE_INDEX, "label") == "删除选中的 2 条"
        for index in BacktestApp._POOL_MENU_SINGLE_INDEXES:
            assert str(menu.entrycget(index, "state")) == "disabled", index

        tree.identify_row = lambda _y: "result-0003"
        BacktestApp._popup_saved_pool_menu(app, event)
        assert tree.selection() == ("result-0003",)
        assert menu.entrycget(
            BacktestApp._POOL_MENU_DELETE_INDEX, "label") == "删除"
        assert str(menu.entrycget(0, "state")) == "normal"
    finally:
        app.destroy()


def test_ticking_show_does_not_collapse_the_row_selection():
    """打勾管的是"画不画到下方"，不该把攒好的多选选区塌掉。"""
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"),
        _snapshot("result-0003", "丙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        tree = app._saved_pool_tree
        tree.selection_set(["result-0001", "result-0002"])

        tree.identify_row = lambda _y: "result-0003"
        tree.identify_column = lambda _x: "#0"
        BacktestApp._toggle_saved_backtest_click(
            app, SimpleNamespace(x=10, y=10))

        assert set(tree.selection()) == {"result-0001", "result-0002"}
        assert "result-0003" not in app._saved_comparison_selection
        assert tree.focus() == "result-0003", "焦点该跟着刚点的那一行走"
    finally:
        app.destroy()


def test_focus_lands_next_to_the_deleted_row_not_at_the_end():
    """删掉一条后焦点落到相邻行。

    此前一律跳到表尾最新那条：逐条清理旧结果时，焦点每删一次就从表头蹦到
    表尾，人得重新找一遍自己删到哪儿了。
    """
    app = _comparison_app(*[
        _snapshot(f"result-{index:04d}", f"结果 {index}")
        for index in range(1, 5)])
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        tree = app._saved_pool_tree

        tree.selection_set("result-0002")
        tree.focus("result-0002")
        BacktestApp._delete_saved_backtest(app, "result-0002")
        BacktestApp._refresh_saved_pool_tree(app)
        assert tree.focus() == "result-0003"

        # 删最后一行时没有下一行，落到上一行。
        tree.selection_set("result-0004")
        tree.focus("result-0004")
        BacktestApp._delete_saved_backtest(app, "result-0004")
        BacktestApp._refresh_saved_pool_tree(app)
        assert tree.focus() == "result-0003"
    finally:
        app.destroy()


def test_toolbar_buttons_grey_out_when_they_have_nothing_to_act_on():
    """有没有可操作对象一律用置灰表达，而不是点了再弹一句提示。"""
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        # _comparison_app 默认把每条都勾上，所以「全选」此时无事可做。
        assert "disabled" in app._saved_pool_show_all_btn.state()
        assert "disabled" not in app._saved_pool_hide_all_btn.state()
        assert "disabled" not in app._saved_pool_export_btn.state()

        BacktestApp._clear_saved_backtest_selection(app)
        assert "disabled" not in app._saved_pool_show_all_btn.state()
        assert "disabled" in app._saved_pool_hide_all_btn.state()
        assert "disabled" in app._saved_pool_export_btn.state(), (
            "一条都不显示时导不出东西")

        app._saved_backtests.clear()
        BacktestApp._refresh_saved_pool_tree(app)
        assert "disabled" in app._saved_pool_delete_btn.state()
        assert "disabled" in app._saved_pool_clear_btn.state()
    finally:
        app.destroy()


def test_delete_button_puts_the_count_in_its_own_label():
    """要删几条得在点下去之前就写在按钮上。"""
    app = _comparison_app(*[
        _snapshot(f"result-{index:04d}", f"结果 {index}")
        for index in range(1, 4)])
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        tree = app._saved_pool_tree

        tree.selection_set("result-0001")
        BacktestApp._sync_saved_pool_buttons(app)
        assert app._saved_pool_delete_btn.cget("text") == "删除选中"

        tree.selection_set(["result-0001", "result-0003"])
        BacktestApp._sync_saved_pool_buttons(app)
        assert app._saved_pool_delete_btn.cget("text") == "删除选中 (2)"
    finally:
        app.destroy()


def test_deselecting_every_row_disables_delete_instead_of_using_the_focus(
        monkeypatch):
    """取消全部选中之后不能还删得动——焦点是看不见的。

    ⌘ 点掉最后一个选中行以后，表上一个高亮都没有，焦点却还留在那一行。作用
    对象一旦回退到焦点，「删除选中」就会在屏幕上什么都没选中的情况下照样可
    点，并删掉一条看不出被选中的结果——而这是真删文件、不可撤销。
    """
    app = _comparison_app(
        _snapshot("result-0001", "甲"), _snapshot("result-0002", "乙"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        tree = app._saved_pool_tree
        tree.selection_set("result-0002")
        tree.focus("result-0002")

        tree.selection_toggle(["result-0002"])      # ⌘ 点掉唯一选中的那行
        BacktestApp._sync_saved_pool_buttons(app)

        assert tree.selection() == ()
        assert tree.focus() == "result-0002", "焦点确实还留着，这正是坑的来源"
        assert BacktestApp._picked_saved_backtest_ids(app) == []
        assert "disabled" in app._saved_pool_delete_btn.state()

        monkeypatch.setattr(
            gui_app.messagebox, "askyesno",
            lambda *_args, **_kwargs: pytest.fail("没选中行时不该弹删除确认框"))
        told = []
        monkeypatch.setattr(
            gui_app.messagebox, "showinfo",
            lambda title, _message: told.append(title))
        BacktestApp._prompt_delete_saved_backtest(app)

        assert told == ["请选择结果"]
        assert len(app._saved_backtests) == 2
    finally:
        app.destroy()


def test_clear_button_greys_out_once_the_pool_is_empty():
    """空池点「全部清空」只能得到一句「清空 0 条」，不如让它点不动。"""
    app = _comparison_app(_snapshot("result-0001", "甲"))
    try:
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        assert "disabled" not in app._saved_pool_clear_btn.state()

        app._saved_backtests.clear()
        BacktestApp._refresh_saved_pool_tree(app)
        assert "disabled" in app._saved_pool_clear_btn.state()
    finally:
        app.destroy()


def test_renaming_rewrites_the_same_file():
    """改展示名不换文件名——文件名带序号，是载入顺序的依据。"""
    app = _persisted_app()
    try:
        snapshot, _bt = _run_and_retain(app, "旧名")
        path = snapshot.store_path
        BacktestApp._rename_saved_backtest(app, snapshot.result_id, "新名")
        assert snapshot.store_path == path
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert reopened._saved_backtests["result-0001"].name == "新名"
    finally:
        reopened.destroy()


def test_reloaded_result_replays_the_exact_same_detail():
    """重开后按配方**重算**出来的逐 bar 结果必须与当时完全一致。

    包里存的是**输入**（行情切片 + 冻结参数）而不是 bar 级输出，所以这条是
    整个设计成立的前提：对不上就等于给了一份看起来像、其实不是的明细。

    必须先清掉 bar 缓存再重放。保留结果时明细已顺手落盘，不清的话这条会命中
    缓存、把当初那份数组原样拿回来——断言恒真，而它本该验的重建路径一步都
    没走。缓存不在时的兜底路径才是这里要守的东西。
    """
    app = _persisted_app()
    try:
        _snap, original = _run_and_retain(app, "可重放 #01")
        original_detail, _mask = app._hedge_trigger_detail_frame(original)
    finally:
        app.destroy()

    history_bar_cache.clear()

    reopened = _persisted_app()
    try:
        loaded = reopened._saved_backtests["result-0001"]
        assert loaded.replay, "重放配方没有落盘"
        replayed, recomputed = BacktestApp._replay_saved_snapshot(
            reopened, loaded)
        assert recomputed, "缓存已清空，这里必须走重算而不是读盘"
        assert np.allclose(
            replayed._results["total_pnl"], original._results["total_pnl"])
        assert np.array_equal(
            replayed._results["hedge_triggered"],
            original._results["hedge_triggered"])
        detail, _mask = reopened._hedge_trigger_detail_frame(replayed)
        assert detail.equals(original_detail)
    finally:
        reopened.destroy()


def test_pool_detail_loads_saved_bars_instead_of_recomputing():
    """结果池的「加载明细」正常路径是读盘，不是重跑。

    保留结果的那一刻 bar 级数组还在手里，落盘约 10 ms；之后读回约 3 ms，
    而按配方重跑一次要几百毫秒。与策略优选那条下钻同一套内容寻址缓存。
    """
    app = _persisted_app()
    try:
        _snap, original = _run_and_retain(app, "可重放 #01")
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        loaded = reopened._saved_backtests["result-0001"]
        first, recomputed = BacktestApp._replay_saved_snapshot(
            reopened, loaded)
        assert not recomputed, "保留时已落盘，这里应当读盘而不是重算"
        assert np.array_equal(
            first._results["net_daily"], original._results["net_daily"])

        # 清掉之后必须仍然可用——回退重算并如实回报。
        history_bar_cache.clear()
        again, recomputed_again = BacktestApp._replay_saved_snapshot(
            reopened, loaded)
        assert recomputed_again
        assert np.allclose(
            again._results["net_daily"], original._results["net_daily"])
    finally:
        reopened.destroy()


def test_pool_detail_cache_separates_option_subtypes():
    """不同期权子类型不能共用一条缓存。

    ``optiontype`` 是定价方法的分派键（Option_AB/DE/SNB 走
    ``getattr(self, self.optiontype)()``），它一度被排除在缓存摘要之外，实测
    Option_DE 的 13 个子类型算出同一个 key——互相读到对方的 bar 级数组且不
    报错。这条把它钉住。
    """
    base = {"prices": [100.0, 101.0, 99.5], "index": None,
            "cls_name": "Snowball", "tc_rate": 0.0001}
    keys = {
        sub: history_bar_cache.key_for_recipe({**base, "subtype": sub})
        for sub in ("Opt_Decumulator", "Opt_Decumulator_Back",
                    "Opt_EnDecumulator")
    }
    assert None not in keys.values(), keys
    assert len(set(keys.values())) == len(keys), keys


def test_replay_without_a_recipe_fails_loudly():
    """没有配方就明确报错，不能返回一个空壳假装加载成功。"""
    app = _persisted_app()
    try:
        stale = _snapshot("result-0001", "上线前保留的")
        stale.replay = {}
        with pytest.raises(ValueError, match="没有重放配方"):
            BacktestApp._replay_saved_snapshot(app, stale)
    finally:
        app.destroy()


def test_failed_write_is_reported_instead_of_claiming_success():
    """写盘失败时状态栏不能仍写「已保留」。

    磁盘满、只读、权限不足都会走到这里。内存池仍保留这条结果（它是真跑出来
    的，扔掉更糟），但「已保留」四个字必须诚实——否则用户下次开机才发现，
    那时已经无从追查。
    """
    app = _persisted_app()
    try:
        def _boom(*_args, **_kwargs):
            raise OSError("磁盘只读")

        app._spd_var.set("4")
        state = app._collect_gui_state()
        backtest = app._build_backtest(state)
        backtest.run()
        app._latest_backtest = backtest
        app._latest_backtest_state = state
        app._latest_retained_result_id = None
        original_write = gui_app.backtest_pool_store.write_snapshot
        gui_app.backtest_pool_store.write_snapshot = _boom
        try:
            snapshot = BacktestApp._store_current_backtest(app, "写不进去")
        finally:
            gui_app.backtest_pool_store.write_snapshot = original_write
        assert snapshot.result_id in app._saved_backtests
        assert snapshot.store_path == ""
        assert "未能写入本机" in app._status_var.get()
        assert "磁盘只读" in app._status_var.get()
    finally:
        app.destroy()


def test_load_failure_is_surfaced_in_the_empty_state(isolate_backtest_pool):
    """载入失败要写进空状态区，不能只表现为「池子是空的」。"""
    isolate_backtest_pool.mkdir(parents=True, exist_ok=True)
    (isolate_backtest_pool / "pool-20260813-000000-0001.json.gz").write_bytes(
        b"not gzip at all")
    app = _persisted_app()
    try:
        assert app._saved_backtests == {}
        assert "未能载入" in app._saved_pool_load_error
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        app.update_idletasks()
        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, gui_app.ttk.Label):
                    texts.append(str(child.cget("text")))
                walk(child)

        walk(app._compare_tab)
        assert any("未能载入" in text for text in texts), texts
    finally:
        app.destroy()


def test_restart_restores_which_results_were_displayed():
    """重开后恢复上次显示的那几条，但不无条件全选。

    全选会把二十条曲线一起铺上去；一条不选又让人以为结果没了——所以恢复
    上次的选择，且空状态会写明「已保留的 N 条都在上方结果池里」。
    """
    app = _persisted_app()
    try:
        first, _bt = _run_and_retain(app, "显示的")
        second, _bt2 = _run_and_retain(app, "隐藏的")
        # 走真实路径：勾选只能从已构建的结果池表格上触发。
        BacktestApp._show_saved_comparison_page(app, navigate=False)
        BacktestApp._toggle_saved_backtest_selection(app, second.result_id)
        assert app._saved_comparison_selection == {first.result_id}
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert len(reopened._saved_backtests) == 2
        assert reopened._saved_comparison_selection == {first.result_id}
    finally:
        reopened.destroy()


def test_deleted_results_drop_out_of_the_restored_selection():
    """删掉的结果不能留在显示集合里——重开后会指向一条不存在的快照。"""
    app = _persisted_app()
    try:
        first, _bt = _run_and_retain(app, "留下的")
        second, _bt2 = _run_and_retain(app, "删掉的")
        BacktestApp._delete_saved_backtest(app, second.result_id)
    finally:
        app.destroy()

    reopened = _persisted_app()
    try:
        assert reopened._saved_comparison_selection == {first.result_id}
    finally:
        reopened.destroy()


def test_disk_eviction_is_mirrored_into_the_memory_pool_and_announced():
    """磁盘淘汰必须同步到内存池，并且说一声。

    内存池此前不受 ``MAX_RESULTS`` 约束，只有磁盘受：存到第 21 条时盘上第
    1 条已被删，标签页却仍写着 (21)、对比页也仍列着它——用户重开程序才发现
    少了一条，而那时已无从追查。留在内存里还有第二个后果：重命名它会走
    ``write_snapshot(path=...)`` 把文件原样写回来，目录随即又超出上限。
    """
    evicted = _snapshot("result-0001", "最旧的一条")
    evicted.store_path = "/tmp/pool/pool-20260715-120000-0001.json.gz"
    kept = _snapshot("result-0002", "留下的一条")
    kept.store_path = "/tmp/pool/pool-20260715-120001-0002.json.gz"
    fake = SimpleNamespace(
        _saved_backtests={evicted.result_id: evicted, kept.result_id: kept},
        _saved_comparison_selection={evicted.result_id, kept.result_id},
        _latest_retained_result_id=evicted.result_id,
    )

    note = BacktestApp._forget_evicted_snapshots(fake, [evicted.store_path])

    assert list(fake._saved_backtests) == [kept.result_id]
    assert fake._saved_comparison_selection == {kept.result_id}
    assert fake._latest_retained_result_id is None
    assert "最旧的一条" in note
    assert str(backtest_pool_store.MAX_RESULTS) in note


def test_nothing_evicted_means_no_status_noise():
    """没淘汰任何东西时不能往状态栏挂一句空话。"""
    kept = _snapshot("result-0002", "留下的一条")
    kept.store_path = "/tmp/pool/pool-20260715-120001-0002.json.gz"
    fake = SimpleNamespace(
        _saved_backtests={kept.result_id: kept},
        _saved_comparison_selection={kept.result_id},
        _latest_retained_result_id=None,
    )

    assert BacktestApp._forget_evicted_snapshots(fake, []) == ""
    # 淘汰了盘上一条本来就不在内存池里的（上次会话留下的），也不该报。
    assert BacktestApp._forget_evicted_snapshots(
        fake, ["/tmp/pool/别的.json.gz"]) == ""
    assert list(fake._saved_backtests) == [kept.result_id]
