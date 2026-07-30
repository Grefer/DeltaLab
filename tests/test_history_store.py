"""策略优选结果包的存取。

这里守的核心是**往返保真**：一次五周期全选要跑 85 秒，如果存下来读回去
静默丢了曲线或改了缺失值的表示，那这个功能就是在制造错误结论——比没有
功能更糟。
"""
from __future__ import annotations

import datetime
import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

import history_store as store


def _ranking_frame():
    """带上真实排名表里的三类麻烦列：元组、部分缺失的元数据、NaN 指标。"""
    return pd.DataFrame([
        {
            "lookback": "quarter", "rank": 1, "strategy": "固定间隔(1σ)",
            "strategy_type": "hedge_band",
            "segment_lengths": (17, 22, 22),
            "incremental_pnl_vs_c2c": 0.0888888789,
            "incremental_sharpe_vs_c2c": np.nan,
            "meta_fixed_times": np.nan,
            "meta_candidate_sigma": 1.0,
            "complete_window": True,
        },
        {
            "lookback": "quarter", "rank": 2, "strategy": "固定时刻(10:30)",
            "strategy_type": "fixed_times",
            "segment_lengths": (17, 22, 22),
            "incremental_pnl_vs_c2c": -0.004,
            "incremental_sharpe_vs_c2c": 0.25,
            "meta_fixed_times": "10:30",
            "meta_candidate_sigma": np.nan,
            "complete_window": False,
        },
    ])


def _summary_frame():
    return pd.DataFrame([
        {
            "lookback": "quarter", "window_id": "segment_1",
            "strategy": "固定间隔(1σ)", "success": True,
            "start_ts": pd.Timestamp("2026-04-24 09:30:00"),
            "end_ts": pd.Timestamp("2026-05-22 15:00:00"),
            "history_endpoint_date": pd.NaT,
            "daily_net_pnl": np.array([0.1, -0.2, 0.35]),
            "cumulative_net_pnl": np.array([0.1, -0.1, 0.25]),
            "normalization_reason": "",
        },
        {
            "lookback": "quarter", "window_id": "segment_2",
            "strategy": "固定间隔(1σ)", "success": True,
            "start_ts": pd.Timestamp("2026-05-22 09:30:00"),
            "end_ts": pd.Timestamp("2026-06-24 15:00:00"),
            "history_endpoint_date": pd.Timestamp("2026-06-24"),
            "daily_net_pnl": np.array([]),
            "cumulative_net_pnl": np.array([1.5]),
            "normalization_reason": "multiplier 必须为有限正数且不能为 0",
        },
    ])


def _state():
    return {
        "history_lookbacks": {"quarter": 61},
        "wind_code": "510050.SH", "history_wind_asof": "2026-07-25",
        "band_candidate_sigmas": (0.5, 1.0, 2.0),
        "quantity": 100.0, "multiplier": 5.0, "force_day_close_hedge": True,
    }


def _save(tmp_path, **overrides):
    kwargs = dict(
        ranking=_ranking_frame(), window_summary=_summary_frame(),
        history_state=_state(), source_label="Wind · 510050.SH · 1分钟",
        objective="incremental_pnl", elapsed_seconds=84.6,
    )
    kwargs.update(overrides)
    payload = store.build_payload(**kwargs)
    return store.save_result(payload, directory=str(tmp_path))


def test_ranking_round_trips_including_tuples_and_missing_metadata(tmp_path):
    original = _ranking_frame()
    back = store.load_result(_save(tmp_path))["ranking"]

    assert list(back.columns) == list(original.columns)
    assert back.shape == original.shape
    # 元组列必须还是元组：分段长度会参与展示与断言，变成 list 会静默改变
    # 相等性判断。
    assert back["segment_lengths"].tolist() == [(17, 22, 22)] * 2
    assert all(isinstance(v, tuple) for v in back["segment_lengths"])
    # 浮点保留到足够精度，不能被 JSON 截断。
    assert back["incremental_pnl_vs_c2c"].iloc[0] == pytest.approx(
        0.0888888789, rel=1e-12)
    # 缺失统一还原成 np.nan（不是 None）：两者 pd.isna 都为真，但下游对
    # 元数据列做 str(...) 会分别拿到 "nan" 与 "None"。
    for column in ("meta_fixed_times", "incremental_sharpe_vs_c2c"):
        assert back[column].isna().tolist() == (
            original[column].isna().tolist())
    assert back["meta_fixed_times"].iloc[0] is np.nan or pd.isna(
        back["meta_fixed_times"].iloc[0])
    assert back["meta_fixed_times"].iloc[1] == "10:30"
    assert back["complete_window"].tolist() == [True, False]


def test_summary_curves_round_trip_as_numpy_arrays(tmp_path):
    original = _summary_frame()
    back = store.load_result(_save(tmp_path))["window_summary"]

    for column in ("daily_net_pnl", "cumulative_net_pnl"):
        for index in range(len(original)):
            value = back[column].iloc[index]
            # 图表那边按 ndarray 处理；还原成 list 不报错但会走别的分支。
            assert isinstance(value, np.ndarray), (column, index, type(value))
            np.testing.assert_allclose(
                value, original[column].iloc[index], rtol=1e-12)
    # 空曲线也要还原成空数组，不能变成 None。
    assert back["daily_net_pnl"].iloc[1].size == 0
    # 时间列必须还是时间，NaT 也要保住。
    assert isinstance(back["start_ts"].iloc[0], pd.Timestamp)
    assert back["start_ts"].iloc[0] == pd.Timestamp("2026-04-24 09:30:00")
    assert pd.isna(back["history_endpoint_date"].iloc[0])
    assert back["history_endpoint_date"].iloc[1] == pd.Timestamp("2026-06-24")


def test_frozen_state_and_provenance_survive(tmp_path):
    back = store.load_result(_save(tmp_path))

    assert back["history_state"]["wind_code"] == "510050.SH"
    assert back["history_state"]["history_lookbacks"] == {"quarter": 61}
    assert back["history_state"]["force_day_close_hedge"] is True
    assert back["objective"] == "incremental_pnl"
    assert back["elapsed_seconds"] == pytest.approx(84.6)
    assert back["source_label"].startswith("Wind")
    # 没有冻结状态就无法解释也无法复现这份结果，它必须完整落盘。
    assert set(_state()) <= set(back["history_state"])


def test_unserializable_state_entries_are_dropped_not_crashed(tmp_path):
    state = dict(_state())
    state["cfg"] = {"build": lambda *a: None}   # 回调
    state["_internal"] = object()               # 私有/不可序列化
    state["callback"] = print

    path = _save(tmp_path, history_state=state)
    back = store.load_result(path)

    assert "callback" not in back["history_state"]
    assert "_internal" not in back["history_state"]
    assert back["history_state"]["wind_code"] == "510050.SH"


def test_loading_a_foreign_schema_version_is_refused(tmp_path):
    """列名与口径都改过；硬渲染旧包会静默给出错误结论。"""
    path = _save(tmp_path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["schema_version"] = store.SCHEMA_VERSION + 1
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))

    with pytest.raises(store.HistoryResultVersionError, match="口径可能已改变"):
        store.load_result(path)
    # 明确要求时才允许放行，供排障使用。
    assert store.load_result(path, allow_other_version=True) is not None


def test_listing_reports_metadata_without_decoding_curves(tmp_path):
    _save(tmp_path)
    _save(tmp_path, history_state={**_state(), "wind_code": "OI701.CZC"})

    items = store.list_results(str(tmp_path))

    assert len(items) == 2
    assert {item["wind_code"] for item in items} == {"510050.SH", "OI701.CZC"}
    for item in items:
        assert item["compatible"] is True
        assert item["rows"] == 2
        assert item["bytes"] > 0
        assert item["lookbacks"] == ["quarter"]
    # 倒序：最新的在最前面。
    assert items[0]["saved_at"] >= items[1]["saved_at"]


def test_listing_skips_corrupt_packages(tmp_path):
    good = _save(tmp_path)
    broken = os.path.join(str(tmp_path), "坏包.json.gz")
    with open(broken, "wb") as handle:
        handle.write(b"not gzip at all")

    items = store.list_results(str(tmp_path))

    # 坏包不能让整个列表页崩掉，也不能在列表里冒充好包。
    assert [item["path"] for item in items] == [good]


def test_listing_missing_directory_is_empty_not_an_error(tmp_path):
    assert store.list_results(str(tmp_path / "还不存在")) == []


def test_save_is_atomic_and_leaves_no_partial_file(tmp_path):
    path = _save(tmp_path)

    assert os.path.isfile(path)
    leftovers = [
        name for name in os.listdir(str(tmp_path)) if name.endswith(".part")]
    # 85 秒跑出来的结果不能因为写一半崩掉就留下一个在列表里看着正常的坏包。
    assert leftovers == []


def test_rename_changes_the_label_but_keeps_the_file(tmp_path):
    path = _save(tmp_path)

    store.rename_result(path, "近季 · 换月前基准")
    back = store.load_result(path)

    assert back["label"] == "近季 · 换月前基准"
    # 文件名带时间戳且是排序依据，重命名不该动它。
    assert os.path.isfile(path)
    assert store.list_results(str(tmp_path))[0]["label"] == "近季 · 换月前基准"


def test_delete_removes_the_package_and_tolerates_missing(tmp_path):
    path = _save(tmp_path)

    assert store.delete_result(path) is True
    assert store.list_results(str(tmp_path)) == []
    assert store.delete_result(path) is False


def test_results_dir_is_separate_from_the_wind_cache():
    """结果不是缓存：删缓存不该顺手删掉 85 秒跑出来的产物。"""
    from pricing.wind_data import _CACHE_DIR

    results = os.path.abspath(store.results_dir())
    cache = os.path.abspath(_CACHE_DIR)

    assert results != cache
    assert not results.startswith(cache + os.sep)


def test_default_filename_is_descriptive_and_filesystem_safe(tmp_path):
    payload = store.build_payload(
        ranking=_ranking_frame(), window_summary=_summary_frame(),
        history_state={
            "wind_code": "OI701.CZC", "history_wind_asof": "2026-07-27",
            "history_lookbacks": {"quarter": 61, "year": 243},
        },
        source_label="t", objective="incremental_pnl")

    name = store.default_filename(payload)

    assert name.startswith("OI701.CZC_2026-07-27_2周期_")
    assert name.endswith(".json.gz")
    for illegal in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        assert illegal not in name
