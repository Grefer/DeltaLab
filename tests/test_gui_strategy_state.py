from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from gui_app import BacktestApp
from pricing import FixedTimeStrategy, HedgeBandStrategy, StrategyCase


class _Var:
    def __init__(self, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


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


def test_wind_intraday_spd_defaults_to_backend_auto_detection():
    state = {"steps_per_day": 4, "steps_per_day_user_override": False}
    assert BacktestApp._resolve_wind_steps_per_day(state, "60") is None


def test_wind_intraday_spd_preserves_explicit_user_override():
    state = {"steps_per_day": 9, "steps_per_day_user_override": True}
    assert BacktestApp._resolve_wind_steps_per_day(state, "60") == 9


def test_wind_daily_spd_is_always_one():
    state = {"steps_per_day": 9, "steps_per_day_user_override": True}
    assert BacktestApp._resolve_wind_steps_per_day(state, None) == 1


def test_gui_long_jobs_share_one_busy_guard_and_failure_status():
    statuses = []
    fake = SimpleNamespace(
        _active_job=None,
        _run_btn=_Widget(),
        _compare_btn=_Widget(),
        _struct_btn=_Widget(),
        _progress=_Widget(),
        _progress_label=_Widget(),
        _set_status=statuses.append,
    )

    assert BacktestApp._begin_job(fake, "comparison", "busy") is True
    assert {fake._run_btn.state, fake._compare_btn.state,
            fake._struct_btn.state} == {"disabled"}

    # 其它任务的迟到回调不能提前解锁共享控件。
    BacktestApp._finish_job(
        fake, "backtest", success=True,
        success_text="wrong", failure_text="wrong-failure")
    assert fake._active_job == "comparison"
    assert fake._run_btn.state == "disabled"

    BacktestApp._finish_job(
        fake, "comparison", success=False,
        success_text="done", failure_text="failed")
    assert fake._active_job is None
    assert {fake._run_btn.state, fake._compare_btn.state,
            fake._struct_btn.state} == {"normal"}
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
