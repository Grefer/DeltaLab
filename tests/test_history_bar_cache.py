"""分段 bar 级结果的磁盘缓存。

两条底线：命中必须与重跑**逐位一致**（差一点点就等于给用户看假数据），
输入变了必须 miss（宁可重跑 620 ms，也不能读到不匹配的结果）。
"""
from __future__ import annotations

import copy
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import history_bar_cache as cache
from pricing import (CloseToCloseStrategy, FixedTimeStrategy,
                     HedgeBandStrategy, Option_Vanilla)
from pricing.hedge_analysis import HistoryReplaySpec


def _spec(**overrides):
    stamps = [pd.Timestamp("2026-01-05 15:00")]
    for day in range(6, 10):
        stamps.extend(
            pd.Timestamp(f"2026-01-{day:02d} 09:3{minute}")
            for minute in range(6))
    path = pd.Series(
        100.0 + np.sin(np.arange(len(stamps)) / 3.0),
        index=pd.DatetimeIndex(stamps))
    option = Option_Vanilla(
        "Vanilla", s0=float(path.iloc[0]), sr=[], K=float(path.iloc[0]),
        T=4, sigma=0.18, cp=1, r=0.03, q=0.03)
    base = dict(
        lookback="quarter", window_id="segment_1", option=option,
        external_path=path, evaluation_days=4, steps_per_day=6,
        strategies={
            "固定间隔(1σ)": HedgeBandStrategy(band_type="sigma", threshold=1.0),
            "每日收盘": CloseToCloseStrategy(),
        },
        backtest_kwargs={"tc_rate": 0.0, "quantity": 1, "multiplier": 0},
        metadata={},
    )
    base.update(overrides)
    return HistoryReplaySpec(**base)


NAME = "固定间隔(1σ)"


def _with_changed_option_list_attr(spec):
    """只改期权的列表属性（``sr``）——digest 曾经整类丢弃列表。"""
    option = copy.deepcopy(spec.option)
    option.sr = [0.9, 0.8]
    return replace(spec, option=option)


def test_cached_result_is_bit_identical_to_a_fresh_run(tmp_path):
    """命中必须逐位一致。差一点点就是给用户看假数据，比不缓存更糟。"""
    spec = _spec()
    fresh = spec.replay(NAME)._results

    cache.store(spec, NAME, fresh, directory=str(tmp_path))
    back = cache.load(spec, NAME, directory=str(tmp_path))

    assert back is not None
    arrays = [k for k, v in fresh.items() if isinstance(v, np.ndarray)]
    assert len(arrays) > 10, arrays
    for key in arrays:
        original, restored = fresh[key], back[key]
        assert restored.shape == original.shape, key
        if original.dtype.kind in "fc":
            # rtol=atol=0：要求逐位相同，不是「差不多」。
            np.testing.assert_allclose(
                restored, original, rtol=0, atol=0, equal_nan=True,
                err_msg=key)
        else:
            assert np.array_equal(restored, original), key


def test_scalar_fields_survive_alongside_the_arrays(tmp_path):
    spec = _spec()
    fresh = spec.replay(NAME)._results

    cache.store(spec, NAME, fresh, directory=str(tmp_path))
    back = cache.load(spec, NAME, directory=str(tmp_path))

    for key in ("strategy_name", "position", "position_label", "total_tc",
                "n_bars", "steps_per_day"):
        if key in fresh:
            assert back[key] == fresh[key], key


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: replace(s, external_path=s.external_path * 1.0001),
                 id="价格被改过"),
    pytest.param(lambda s: replace(s, evaluation_days=3), id="评估天数"),
    pytest.param(lambda s: replace(s, steps_per_day=7), id="bar 粒度"),
    pytest.param(lambda s: replace(
        s, backtest_kwargs={**dict(s.backtest_kwargs), "tc_rate": 0.002}),
        id="成本率"),
    pytest.param(lambda s: replace(s, strategies={
        **dict(s.strategies),
        "固定间隔(1σ)": HedgeBandStrategy(band_type="sigma", threshold=2.0)}),
        id="带宽倍数"),
    pytest.param(lambda s: replace(s, option=Option_Vanilla(
        "Vanilla", s0=float(s.external_path.iloc[0]), sr=[],
        K=float(s.external_path.iloc[0]), T=4, sigma=0.25, cp=1,
        r=0.03, q=0.03)), id="波动率"),
    # 补的：原先 5 项恰好没碰到 digest 丢列表属性这个洞——期权那项换的是
    # 整个类，掩盖了「只改一个列表字段」的情形。时刻表那个洞由下面的
    # test_changing_only_the_fixed_time_schedule_misses 单独钉（key 是按
    # 候选算的，改 dict 里另一个策略本就不该影响本候选）。
    pytest.param(_with_changed_option_list_attr, id="期权的列表属性"),
])
def test_any_input_change_misses_the_cache(tmp_path, mutate):
    """key 必须覆盖所有影响结果的输入。

    过度敏感只是白跑 620 ms；不够敏感则会把不匹配的结果当成本次结果显示，
    而且不会报错。所以这里宁可多列几项。
    """
    spec = _spec()
    cache.store(spec, NAME, spec.replay(NAME)._results,
                directory=str(tmp_path))

    assert cache.load(spec, NAME, directory=str(tmp_path)) is not None
    assert cache.load(mutate(spec), NAME, directory=str(tmp_path)) is None


def test_changing_only_the_fixed_time_schedule_misses(tmp_path):
    """时刻表整张不进 key 的话，改时刻重跑后下钻会显示上一次的明细。

    实测过：``_digest_strategy`` 曾去找 ``target_times``/``fixed_times``，
    而 FixedTimeStrategy 的属性叫 ``requested_times``/``effective_times``/
    ``times``，两个名字都不存在，时刻表整个不进 key。
    """
    name = "固定时刻"
    base = _spec(strategies={name: FixedTimeStrategy(["09:31"])})
    other = _spec(strategies={name: FixedTimeStrategy(["09:33"])})

    assert cache.key_for(base, name) != cache.key_for(other, name)


def test_unsummarisable_input_disables_the_cache_instead_of_colliding(
        tmp_path):
    """摘要不了的属性必须让整次缓存放弃，而不是被悄悄丢掉。

    悄悄丢掉的后果是两次不同的运行共用一条缓存、下钻显示别次的数据且不报
    错；放弃缓存的代价只是白跑 620 ms。
    """
    spec = _spec()
    option = copy.deepcopy(spec.option)
    option.某个无法摘要的字段 = object()
    blind = replace(spec, option=option)

    assert cache.key_for(blind, NAME) is None
    # 既不写也不读。
    assert cache.store(blind, NAME, spec.replay(NAME)._results,
                       directory=str(tmp_path)) is None
    assert cache.load(blind, NAME, directory=str(tmp_path)) is None


def test_a_different_strategy_in_the_same_segment_misses(tmp_path):
    spec = _spec()
    cache.store(spec, NAME, spec.replay(NAME)._results,
                directory=str(tmp_path))

    assert cache.load(spec, "每日收盘", directory=str(tmp_path)) is None


def test_a_price_series_with_identical_endpoints_but_different_body_misses(
        tmp_path):
    """只比首尾会漏掉「区间相同但价格被改过」——前复权重算正是这种情形。"""
    spec = _spec()
    cache.store(spec, NAME, spec.replay(NAME)._results,
                directory=str(tmp_path))

    tweaked_values = spec.external_path.to_numpy().copy()
    tweaked_values[len(tweaked_values) // 2] += 0.5   # 只改中间一个点
    tweaked = replace(spec, external_path=pd.Series(
        tweaked_values, index=spec.external_path.index))

    assert cache.load(tweaked, NAME, directory=str(tmp_path)) is None


def test_missing_or_corrupt_entries_return_none_instead_of_raising(tmp_path):
    """缓存只省时间；坏了就当没有，绝不能让下钻整个失败。"""
    spec = _spec()

    assert cache.load(spec, NAME, directory=str(tmp_path)) is None

    cache.store(spec, NAME, spec.replay(NAME)._results,
                directory=str(tmp_path))
    npz = [n for n in os.listdir(str(tmp_path)) if n.endswith(".npz")][0]
    with open(os.path.join(str(tmp_path), npz), "wb") as handle:
        handle.write(b"not an npz at all")

    assert cache.load(spec, NAME, directory=str(tmp_path)) is None


def test_store_failure_is_swallowed_and_reported_as_none(tmp_path):
    """写不进去（磁盘满、只读目录）时返回 None，主流程照常。"""
    spec = _spec()
    blocked = tmp_path / "占位"
    blocked.write_text("我是文件不是目录", encoding="utf-8")

    assert cache.store(
        spec, NAME, spec.replay(NAME)._results, directory=str(blocked)) is None


def test_prune_evicts_least_recently_used_until_within_budget(tmp_path):
    """没有上限的缓存本身就是 bug：一轮近年 38 MB，跑二十轮就 760 MB。"""
    import time

    # 必须是真正不同的输入：lookback / window_id 刻意不进 key——同一份
    # 行情与参数出现在两个周期下时本就该共用一条缓存。
    specs = []
    for i in range(4):
        base = _spec()
        specs.append(replace(
            base, window_id=f"segment_{i}",
            external_path=base.external_path + 0.1 * i))
    for index, spec in enumerate(specs):
        cache.store(spec, NAME, spec.replay(NAME)._results,
                    directory=str(tmp_path))
        time.sleep(0.01)      # 拉开 mtime，LRU 顺序才确定
    # 让最早那份变成"最近用过"，它应当被保住。
    assert cache.load(specs[0], NAME, directory=str(tmp_path)) is not None

    total = cache.usage_bytes(str(tmp_path))
    freed = cache.prune(max_bytes=total // 2, directory=str(tmp_path))

    assert freed > 0
    assert cache.usage_bytes(str(tmp_path)) <= total
    assert cache.load(specs[0], NAME, directory=str(tmp_path)) is not None
    # 最久未用的那些先出局。
    survivors = sum(
        cache.load(spec, NAME, directory=str(tmp_path)) is not None
        for spec in specs)
    assert survivors < len(specs)


def test_clear_removes_everything_and_tolerates_a_missing_directory(tmp_path):
    spec = _spec()
    cache.store(spec, NAME, spec.replay(NAME)._results,
                directory=str(tmp_path))

    assert cache.clear(str(tmp_path)) > 0
    assert cache.load(spec, NAME, directory=str(tmp_path)) is None
    assert cache.clear(str(tmp_path / "不存在")) == 0


def test_cache_dir_sits_under_the_disposable_cache_root():
    """它是真缓存：丢了只是下次重跑，所以放缓存目录而不是结果目录。"""
    import history_store

    bars = os.path.abspath(cache.cache_dir())
    results = os.path.abspath(history_store.results_dir())

    assert "cache" in bars
    assert bars != results
    assert not bars.startswith(results + os.sep)
