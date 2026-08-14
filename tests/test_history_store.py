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


def _multi_period_ranking(winners):
    """按 {周期: 冠军策略} 造一张带基准的排名表。"""
    rows = []
    for lookback, (days, winner) in winners.items():
        rows.append({
            "lookback": lookback, "lookback_days": days, "rank": 1,
            "strategy": winner, "strategy_type": "hedge_band",
            "recommendation_eligible": True,
        })
        rows.append({
            "lookback": lookback, "lookback_days": days, "rank": 2,
            "strategy": "每日收盘", "strategy_type": "close_to_close",
            "recommendation_eligible": True,
        })
    return pd.DataFrame(rows)


def test_listing_reports_the_longest_period_as_evidence_span(tmp_path):
    """「2 个周期」没用：近周+近月和近半年+近年都是 2 个，可信度差着量级。"""
    path = _save(tmp_path, ranking=_multi_period_ranking({
        "quarter": (61, "固定间隔(1σ)"), "half_year": (122, "固定间隔(1σ)"),
    }))

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["evidence_span"] == "近半年"


def test_listing_verdict_names_the_strategy_only_when_periods_agree(tmp_path):
    path = _save(tmp_path, ranking=_multi_period_ranking({
        "quarter": (61, "固定间隔(1σ)"),
        "half_year": (122, "固定间隔(1σ)"),
        "year": (243, "固定间隔(1σ)"),
    }))

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["verdict"] == "固定间隔(1σ)"
    assert item["evidence_span"] == "近年"


def test_listing_verdict_refuses_to_pick_a_winner_when_periods_disagree(
        tmp_path):
    """有分歧就说有分歧——挑一个报出来等于把弱证据伪装成强结论。"""
    path = _save(tmp_path, ranking=_multi_period_ranking({
        "quarter": (61, "固定间隔(1σ)"),
        "half_year": (122, "固定时刻(10:30)"),
        "year": (243, "固定间隔(2σ)"),
    }))

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["verdict"] == "不一致（3 种）"


def test_listing_verdict_ignores_the_baseline_and_ineligible_rows(tmp_path):
    """基准是对照组不是推荐；不合格的周期也不该贡献结论。"""
    ranking = pd.DataFrame([
        # 只有基准能排第一 —— 不构成推荐
        {"lookback": "quarter", "lookback_days": 61, "rank": 1,
         "strategy": "每日收盘", "strategy_type": "close_to_close",
         "recommendation_eligible": True},
        # 数据不合格的候选不参与
        {"lookback": "year", "lookback_days": 243, "rank": 1,
         "strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
         "recommendation_eligible": False},
    ])
    path = _save(tmp_path, ranking=ranking)

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["verdict"] == "无可比结论"


def test_listing_verdict_says_baseline_when_the_baseline_wins(tmp_path):
    """基准排第一时结论就是「每日收盘」，不能报第二名。

    这里曾经无条件跳过 close_to_close，于是历史证据支持"维持每日收盘"的
    周期，列表结论列显示的是**落败**的那个候选——与结果页的「每日收盘
    （基准最优）」正好相反，用户照它决定要不要改用 2σ 带宽就会做反。

    而且这不是偶发：_history_recommendations 以"该周期存在至少一条
    eligible 且非 C2C 的行"为准入闸门，那条行恰恰就是被误报出来的那行，
    所以只要该周期进得了结论，就必然报错方向。
    """
    ranking = pd.DataFrame([
        {"lookback": "quarter", "lookback_days": 61, "rank": 1,
         "strategy": "每日收盘", "strategy_type": "close_to_close",
         "recommendation_eligible": True},
        {"lookback": "quarter", "lookback_days": 61, "rank": 2,
         "strategy": "固定间隔(2σ)", "strategy_type": "hedge_band",
         "recommendation_eligible": True},
    ])

    path = _save(tmp_path, ranking=ranking)
    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["verdict"] == "每日收盘"


def test_listing_verdict_counts_a_baseline_win_as_its_own_conclusion(tmp_path):
    """一个周期选基准、另一个周期选候选，那就是分歧，不能抹平成单一结论。"""
    ranking = pd.DataFrame([
        {"lookback": "month", "lookback_days": 20, "rank": 1,
         "strategy": "每日收盘", "strategy_type": "close_to_close",
         "recommendation_eligible": True},
        {"lookback": "month", "lookback_days": 20, "rank": 2,
         "strategy": "固定间隔(2σ)", "strategy_type": "hedge_band",
         "recommendation_eligible": True},
        {"lookback": "quarter", "lookback_days": 61, "rank": 1,
         "strategy": "固定间隔(2σ)", "strategy_type": "hedge_band",
         "recommendation_eligible": True},
        {"lookback": "quarter", "lookback_days": 61, "rank": 2,
         "strategy": "每日收盘", "strategy_type": "close_to_close",
         "recommendation_eligible": True},
    ])

    path = _save(tmp_path, ranking=ranking)
    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["verdict"] == "不一致（2 种）"


@pytest.mark.parametrize(("params", "expected"), [
    ({"T_days": 22, "sigma": 0.18}, "T22 σ0.18"),
    ({"T_days": 60, "sigma": 0.24}, "T60 σ0.24"),
    # σ 决定带宽绝对宽度，缺了它两份结果会看着一样
    ({"T_days": 22}, "T22"),
    ({"sigma": 0.12}, "σ0.12"),
    ({}, "—"),
])
def test_listing_summarises_the_option_by_maturity_and_vol(
        tmp_path, params, expected):
    path = _save(tmp_path, history_state={**_state(), "params": params})

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["option_summary"] == expected


@pytest.mark.parametrize(("overrides", "expected"), [
    ({"position": -1, "tc_rate": 0.0001}, "Buy 费0.01%"),
    ({"position": 1, "tc_rate": 0.0005}, "Sell 费0.05%"),
    # 收盘保底默认开启，开着时不占版面
    ({"position": -1, "tc_rate": 0.0001, "force_day_close_hedge": True},
     "Buy 费0.01%"),
    # 关掉了就必须看得见：实测它能让固定间隔 2.0σ 从第 1 名掉到第 6 名
    ({"position": -1, "tc_rate": 0.0001, "force_day_close_hedge": False},
     "Buy 费0.01% ⚠无保底"),
])
def test_listing_flags_position_cost_and_a_disabled_close_fallback(
        tmp_path, overrides, expected):
    state = {k: v for k, v in _state().items()
             if k != "force_day_close_hedge"}
    path = _save(tmp_path, history_state={**state, **overrides})

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["position_summary"] == expected


def test_listing_survives_a_package_that_is_gzip_but_not_utf8(tmp_path):
    """gzip 流完好、内容不是合法 utf-8 时抛 UnicodeDecodeError。

    漏捕它会让整个 list_results 崩掉，而不是跳过这一个坏包——列表页直接
    打不开。
    """
    good = _save(tmp_path)
    broken = os.path.join(str(tmp_path), "非utf8.json.gz")
    with gzip.open(broken, "wb") as handle:
        handle.write(b"\xff\xfe\x00\x01 not utf-8")

    items = store.list_results(str(tmp_path))

    assert [item["path"] for item in items] == [good]


def test_listing_sweeps_crash_leftover_part_files(tmp_path):
    """``.part`` 不在 *.json.gz 的 glob 里，没有别的清理点，只会一直占盘。"""
    _save(tmp_path)
    leftover = os.path.join(str(tmp_path), "崩溃残留.json.gz.part")
    with open(leftover, "wb") as handle:
        handle.write(b"half written")

    store.list_results(str(tmp_path))

    assert not os.path.exists(leftover)


def test_tuple_column_normalises_missing_to_nan_like_scalars(tmp_path):
    """元组列的缺失也要还原成 np.nan，与 scalar 分支一致。"""
    frame = pd.DataFrame([
        {"lookback": "quarter", "segment_lengths": (17, 22)},
        {"lookback": "year", "segment_lengths": None},
    ])
    path = _save(tmp_path, ranking=frame)

    back = store.load_result(path)["ranking"]

    assert back["segment_lengths"].iloc[0] == (17, 22)
    assert pd.isna(back["segment_lengths"].iloc[1])
    assert back["segment_lengths"].iloc[1] is not None


def test_replay_payload_carries_warmup_kwargs(tmp_path):
    """预热参数必须入包。

    实跑路径按候选记录它，HistoryReplaySpec.build 会取；缺了它，realized
    σ 的候选会在没有预热种子的情况下跑完，而实跑路径本来会因
    strict_sigma_warmup=True 直接报错。
    """
    from pricing.hedge_analysis import HistoryReplaySpec
    from pricing import CloseToCloseStrategy, Option_Vanilla

    index = pd.DatetimeIndex([
        pd.Timestamp(f"2026-01-{day:02d} 15:00") for day in (5, 6, 7, 8, 9)])
    path = pd.Series([100.0, 101.0, 99.0, 102.0, 100.5], index=index)
    spec = HistoryReplaySpec(
        lookback="quarter", window_id="segment_1",
        option=Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0, T=4,
                              sigma=0.18, cp=1, r=0.03, q=0.03),
        external_path=path, evaluation_days=4, steps_per_day=1,
        strategies={"每日收盘": CloseToCloseStrategy()},
        backtest_kwargs={"tc_rate": 0.0},
        warmup_kwargs={"每日收盘": {"sigma_warmup_log_returns": [0.01, -0.02],
                                   "strict_sigma_warmup": True}},
        metadata={})

    payload = store.build_payload(
        ranking=_ranking_frame(), window_summary=_summary_frame(),
        history_state=_state(), source_label="t",
        objective="incremental_pnl",
        replay_index={"quarter": {"segment_1": spec}})
    stored = store.save_result(payload, directory=str(tmp_path))
    _series_map, specs = store.restore_replay_payload(
        store.load_result(stored))

    warmup = specs[0]["warmup_kwargs"]["每日收盘"]
    assert warmup["strict_sigma_warmup"] is True
    assert warmup["sigma_warmup_log_returns"] == [0.01, -0.02]


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


def test_saving_twice_in_the_same_second_keeps_both(tmp_path):
    """文件名的时间戳只到秒；撞名必须加序号，不能静默覆盖。

    「存完发现名字打错了立刻重存」是很容易发生的操作序列。此前第二次会
    直接盖掉第一次，用户以为存了两份、实际只剩最后一份——而 85 秒跑出来
    的结果被静默丢掉是最不能接受的失败。
    """
    first = _save(tmp_path, label="第一份")
    second = _save(tmp_path, label="第二份")
    third = _save(tmp_path, label="第三份")

    assert len({first, second, third}) == 3
    labels = {item["label"] for item in store.list_results(str(tmp_path))}
    assert labels == {"第一份", "第二份", "第三份"}


def test_saving_beyond_the_limit_evicts_the_oldest(tmp_path):
    """最多保留 MAX_RESULTS 份，超出后淘汰最旧的。"""
    import time

    paths = []
    for index in range(store.MAX_RESULTS + 3):
        paths.append(_save(tmp_path, label=f"第{index}份"))
        time.sleep(0.002)

    items = store.list_results(str(tmp_path))

    assert len(items) == store.MAX_RESULTS
    labels = {item["label"] for item in items}
    # 最早的三份出局，最新的都在。
    assert {"第0份", "第1份", "第2份"}.isdisjoint(labels)
    assert f"第{store.MAX_RESULTS + 2}份" in labels


def test_enforce_limit_returns_what_it_deleted(tmp_path):
    """删除不可逆。调用方要拿这个清单告诉用户——那是别人 85 秒跑出来的。"""
    import time

    for index in range(5):
        _save(tmp_path, label=f"第{index}份")
        time.sleep(0.002)

    evicted = store.enforce_limit(max_results=3, directory=str(tmp_path))

    assert [item["label"] for item in evicted] == ["第1份", "第0份"]
    assert len(store.list_results(str(tmp_path))) == 3
    # 再跑一次已经在限内，不该再删。
    assert store.enforce_limit(max_results=3, directory=str(tmp_path)) == []


def test_save_with_enforce_false_keeps_everything(tmp_path):
    """调用方要自己收集淘汰清单时，写入不能顺手把旧的删了。"""
    for index in range(store.MAX_RESULTS + 2):
        payload = store.build_payload(
            ranking=_ranking_frame(), window_summary=_summary_frame(),
            history_state=_state(), source_label="t",
            objective="incremental_pnl", label=f"第{index}份")
        store.save_result(
            payload, directory=str(tmp_path), enforce=False)

    assert len(store.list_results(str(tmp_path))) == store.MAX_RESULTS + 2


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


_STAMP = datetime.datetime(2026, 7, 31, 14, 28)


@pytest.mark.parametrize(("state", "expected"), [
    ({"wind_code": "510050.SH", "position": 1},
     "510050.SH Sell 07-31 14:28"),
    ({"wind_code": "OI701.CZC", "position": -1},
     "OI701.CZC Buy 07-31 14:28"),
    # CSV 来源用文件名当品种
    ({"csv_path": "/data/沪深300_2026.csv", "position": -1},
     "沪深300_2026 Buy 07-31 14:28"),
    # 空状态也要给出可用的名字，不能抛
    ({}, "Sell 07-31 14:28"),
])
def test_default_label_is_code_side_and_save_time(state, expected):
    """预填名 = 品种 + 买卖方向 + 保存时刻。

    第三段取保存时刻而不是分析截至日：后者同一天连跑两次就撞，而且它已
    经单独成列。不带年份是为了让名称列容得下——列表最多 20 份，同月同日
    同分钟跨年重名不现实。
    """
    assert store.default_label(state, now=_STAMP) == expected


def test_default_label_appends_a_serial_when_the_name_is_taken():
    """同一分钟内连存两次仍会撞，序号是最后一层兜底。"""
    state = {"wind_code": "510050.SH", "position": -1}
    base = "510050.SH Buy 07-31 14:28"

    assert store.default_label(state, existing=[], now=_STAMP) == base
    assert store.default_label(
        state, existing=[base], now=_STAMP) == f"{base} #2"
    assert store.default_label(
        state, existing=[base, f"{base} #2"], now=_STAMP) == f"{base} #3"
    # 别的名字不影响；空白项不算占用。
    assert store.default_label(
        state, existing=["换月前对照", "  ", ""], now=_STAMP) == base


@pytest.mark.parametrize(("subtype", "expected"), [
    ("Eu", "欧式"),
    ("Opt_EnDecumulator_Fix", "固定赔付增强累计"),
    ("Opt_ASGQ_EFF", "熔断每日双固赔累计"),
    ("Opt_Snowball", "雪球"),
    ("EnhanceAsian", "增强亚式"),
])
def test_listing_reports_the_option_subtype_in_chinese(
        tmp_path, monkeypatch, subtype, expected):
    """同一大类下子型行为差别很大——累计期权就有 13 个子型。

    中文名复用 GUI 已有的 SUBTYPE_DISPLAY（由 gui_app 注入），不另造一套
    ——两套映射迟早会各自演化到对不上。
    """
    import gui_app                                    # 触发注入
    monkeypatch.setattr(
        store, "SUBTYPE_DISPLAY", gui_app.SUBTYPE_DISPLAY)
    path = _save(tmp_path, history_state={**_state(), "subtype": subtype})

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["subtype"] == expected


def test_every_registered_subtype_has_a_chinese_name():
    """注册表里每个子型都必须有中文名，否则列表会露出内部标识。"""
    import gui_app

    missing = [
        subtype
        for cfg in gui_app.OPTION_CLASSES.values()
        for subtype in cfg["subtypes"]
        if subtype not in gui_app.SUBTYPE_DISPLAY
    ]

    assert missing == [], missing


@pytest.mark.parametrize(("subtype", "expected"), [
    # 映射缺失时不能崩，也不该显示空白：去掉无信息的 Opt_ 前缀后原样给出
    ("Opt_未知子型", "未知子型"),
    ("SomethingNew", "SomethingNew"),
    ("", "—"),
])
def test_unmapped_subtype_falls_back_to_the_raw_identifier(
        tmp_path, monkeypatch, subtype, expected):
    monkeypatch.setattr(store, "SUBTYPE_DISPLAY", {})
    path = _save(tmp_path, history_state={**_state(), "subtype": subtype})

    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["subtype"] == expected


def test_default_label_gets_the_position_sign_right():
    """1=卖出、-1=买入（见 HedgeBacktest）。弄反会让名字直接误导。

    用 Buy/Sell 而不是 Long/Short：后者在期权语境里指 gamma 敞口，而同一
    买卖方向在不同产品上敞口相反，拿它当买卖方向的标签必然对一半错一半。
    """
    base = {"wind_code": "510050.SH"}

    assert store.default_label({**base, "position": 1}).split()[1] == "Sell"
    assert store.default_label({**base, "position": -1}).split()[1] == "Buy"
    # 非法/缺失时按卖出处理，而不是崩掉。
    assert store.default_label({**base, "position": None}).split()[1] == "Sell"
    assert store.default_label({**base, "position": "x"}).split()[1] == "Sell"


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


def _pool_replay_index(prices_a, prices_b, *, start_b=3):
    """造一个双合约重放索引：两段日期区间重叠，同一天价格不同（换月）。"""
    from pricing.hedge_analysis import HistoryReplaySpec
    from pricing import CloseToCloseStrategy, Option_Vanilla

    def spec_for(window_id, contract, first_day, values):
        index = pd.DatetimeIndex([
            pd.Timestamp(f"2026-06-{first_day + i:02d} 15:00")
            for i in range(len(values))])
        return HistoryReplaySpec(
            lookback="sample", window_id=window_id,
            option=Option_Vanilla("Vanilla", s0=float(values[0]), sr=[],
                                  K=100.0, T=len(values), sigma=0.18, cp=1,
                                  r=0.03, q=0.03),
            external_path=pd.Series(np.asarray(values, dtype=float),
                                    index=index),
            evaluation_days=len(values) - 1, steps_per_day=1,
            strategies={"每日收盘": CloseToCloseStrategy()},
            backtest_kwargs={"tc_rate": 0.0},
            metadata={"history_mode": "product_contract_pool",
                      "contract_code": contract})

    return {"sample": {
        "c_A": spec_for("c_A", "P2601.DCE", 1, prices_a),
        "c_B": spec_for("c_B", "P2605.DCE", start_b, prices_b),
    }}


def test_contract_pool_segments_survive_the_round_trip():
    """不同合约的分段不能被并成一条序列——换月处会串价。

    原本无条件 ``pd.concat`` 后 ``keep="first"`` 去重，前提是「各段都是同
    一条已拉取序列的精确切片」。那只对单标的成立：品种池各段来自不同合
    约，换月处同一天有两个价，后入的段会被前一段顶掉。实测 P2605 段原本
    ``[105,106,107,108]`` 载回来变成 ``[203,204,107,108]``——串掉的第一根
    还是 Day 0 锚点，``_rescale_option_to_real_s0`` 会拿它给整段期权重定
    基，错的远不止那两根 bar。
    """
    prices_a = [201.0, 202.0, 203.0, 204.0]
    prices_b = [105.0, 106.0, 107.0, 108.0]
    index = _pool_replay_index(prices_a, prices_b)

    payload = store.build_replay_payload(index)
    blob = json.loads(json.dumps({"replay": payload}, default=str))
    series_by_key, specs = store.restore_replay_payload(blob)
    assert specs

    # 必须是**按合约**分的桶。逐段拆开也能得出正确数字，但那是兜底路径，
    # 空间翻倍；这里要确认走的是正路。
    assert set(series_by_key) == {"P2601.DCE", "P2605.DCE"}, series_by_key
    assert {spec["series_key"] for spec in specs} == set(series_by_key)

    for spec in specs:
        original = index["sample"][spec["window_id"]].external_path
        series = series_by_key[spec["series_key"]]
        got = series.loc[pd.Timestamp(spec["start_ts"]):
                         pd.Timestamp(spec["end_ts"])]
        assert len(got) == len(original), spec["window_id"]
        assert np.array_equal(got.to_numpy(), original.to_numpy()), (
            spec["window_id"], list(got.to_numpy()),
            list(original.to_numpy()))


def test_single_instrument_still_shares_one_series():
    """单标的仍然只存一条序列——分桶不能把省下来的空间又吐回去。

    各段都是同一条已拉取序列的切片时合并是对的，一年 1 分钟 gzip 后约
    206 KB。按段各存各的要翻好几倍，只在合并确实会串价时才允许退化。
    """
    from pricing.hedge_analysis import HistoryReplaySpec
    from pricing import CloseToCloseStrategy, Option_Vanilla

    stamps = pd.DatetimeIndex([
        pd.Timestamp(f"2026-06-{day:02d} 15:00") for day in range(1, 9)])
    whole = pd.Series(np.linspace(100.0, 107.0, 8), index=stamps)

    def spec_for(window_id, lo, hi):
        piece = whole.iloc[lo:hi]
        return HistoryReplaySpec(
            lookback="sample", window_id=window_id,
            option=Option_Vanilla("Vanilla", s0=float(piece.iloc[0]), sr=[],
                                  K=100.0, T=len(piece), sigma=0.18, cp=1,
                                  r=0.03, q=0.03),
            external_path=piece, evaluation_days=len(piece) - 1,
            steps_per_day=1,
            strategies={"每日收盘": CloseToCloseStrategy()},
            backtest_kwargs={"tc_rate": 0.0}, metadata={})

    # 刻意重叠：近周那几天也在近月里，这正是当初要去重的原因。
    index = {"sample": {"w1": spec_for("w1", 0, 8),
                        "w2": spec_for("w2", 3, 8)}}
    series_by_key, key_by_window = store.replay_series_buckets(index)
    assert list(series_by_key) == [store.DEFAULT_SERIES_KEY], series_by_key
    assert len(series_by_key[store.DEFAULT_SERIES_KEY]) == 8
    assert set(key_by_window.values()) == {store.DEFAULT_SERIES_KEY}


def test_old_flat_contract_pool_package_is_refused():
    """修复前存的品种池包不可救，必须判定不可重放而不是画错价。

    那时多合约已经在保存时就被并成一条、换月处的价被顶掉了，包里存的就是
    错数——重建时无从分辨。拿它画出来的曲线不会报错，还能被保留进结果对
    比页当成真结果用，所以这里宁可让下钻不可用。
    """
    legacy = {"replay": {
        "price_series": {
            "index": [f"2026-06-0{d} 15:00:00" for d in (1, 2, 3, 4)],
            "values": [201.0, 202.0, 203.0, 204.0],
        },
        "specs": [
            {"lookback": "sample", "window_id": "c_A",
             "start_ts": "2026-06-01 15:00:00", "end_ts": "2026-06-02 15:00:00",
             "metadata": {"contract_code": "P2601.DCE"}},
            {"lookback": "sample", "window_id": "c_B",
             "start_ts": "2026-06-03 15:00:00", "end_ts": "2026-06-04 15:00:00",
             "metadata": {"contract_code": "P2605.DCE"}},
        ],
    }}
    assert store.restore_replay_payload(legacy) == ({}, [])

    # 单标的的旧包不受影响，照常可用。
    single = json.loads(json.dumps(legacy))
    for spec in single["replay"]["specs"]:
        spec["metadata"] = {}
    series_by_key, specs = store.restore_replay_payload(single)
    assert list(series_by_key) == [store.DEFAULT_SERIES_KEY]
    assert len(specs) == 2


def test_conflicting_segments_are_split_even_without_contract_codes():
    """认不出合约时，也不能把互相冲突的段并成一条。

    按合约分桶要靠 ``metadata["contract_code"]``，而它不是每条路径都填。
    所以合并前还要逐段验一遍「能不能原样切回来」——验不过就退化成按段各
    存各的。这两层任缺其一，都会在另一层失效时静默串价。

    这条刻意**不填** contract_code，只留价格冲突，专测兜底那层。
    """
    index = _pool_replay_index([201.0, 202.0, 203.0, 204.0],
                               [105.0, 106.0, 107.0, 108.0])
    for spec in index["sample"].values():
        spec.metadata.clear()

    series_by_key, key_by_window = store.replay_series_buckets(index)
    assert len(series_by_key) == 2, series_by_key
    assert store.DEFAULT_SERIES_KEY not in series_by_key

    for (_lookback, window_id), key in key_by_window.items():
        original = index["sample"][window_id].external_path
        got = series_by_key[key].loc[original.index[0]:original.index[-1]]
        assert np.array_equal(got.to_numpy(), original.to_numpy()), (
            window_id, list(got.to_numpy()), list(original.to_numpy()))


def test_delete_result_reports_failure_instead_of_raising(tmp_path,
                                                          monkeypatch):
    """删不掉要返回 False，不能把异常甩给保存流程。

    只接 FileNotFoundError 是不够的：只读文件或被占用会抛 PermissionError，
    它从 enforce_limit 一路冒到保存流程的 except，弹出「保存失败」——而
    save_result 早就把包写进盘了。用户被告知 85 秒跑出来的结果没保存，实际
    它就在目录里。
    """
    path = _save(tmp_path)
    _save(tmp_path)

    def _denied(_target):
        raise PermissionError("只读")

    monkeypatch.setattr(store.os, "remove", _denied)
    assert store.delete_result(path) is False

    # 淘汰阶段同样不能炸：它在 save_result 写盘成功**之后**才跑，异常冒出去
    # 会让用户看到「保存失败」，而包已经在目录里了。
    assert store.enforce_limit(1, directory=str(tmp_path)) == []
    assert len(store.list_results(str(tmp_path))) == 2


def test_listing_verdict_skips_rows_without_a_rank(tmp_path):
    """缺名次的行必须跳过，不能落成 0 去压过真正的第 1 名。

    ``_jsonable`` 把 NaN 写成 null，读回来是 None，``None or 0`` 得到 0——
    而选优取 rank 最小者，于是一行没有名次的记录稳定压过第 1 名，列表的
    「结论」列报出一个落败候选。
    """
    ranking = pd.DataFrame([
        {"lookback": "year", "lookback_days": 243, "rank": 1,
         "strategy": "固定间隔(1σ)", "strategy_type": "hedge_band",
         "recommendation_eligible": True},
        {"lookback": "year", "lookback_days": 243, "rank": np.nan,
         "strategy": "固定时刻(10:30)", "strategy_type": "fixed_times",
         "recommendation_eligible": True},
    ])

    path = _save(tmp_path, ranking=ranking)
    item = next(i for i in store.list_results(str(tmp_path))
                if i["path"] == path)

    assert item["verdict"] == "固定间隔(1σ)"


def test_limit_does_not_peek_any_package_while_under_the_cap(tmp_path,
                                                             monkeypatch):
    """没到上限时一份都不该读。实测 _peek 约 9 ms/份、30 份 272 ms。"""
    _save(tmp_path)
    _save(tmp_path)

    calls = []
    monkeypatch.setattr(store, "list_results",
                        lambda directory=None: calls.append(directory) or [])
    assert store.enforce_limit(10, directory=str(tmp_path)) == []
    assert calls == []
