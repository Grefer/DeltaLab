"""回测结果池落盘层。

守两件事：**往返必须精确**，**坏包不能连累好包**。

往返之所以要逐类型钉死，是因为丢失都不报错、只让结论变差：签名 key 的嵌套
tuple 变成 list 会让「本次对比的变量」哑火；``inf`` 被写成 null 会让指标表把
「无穷大」显示成缺失。这两种都不会抛异常，只会静默给出更差的页面。
"""
from __future__ import annotations

import datetime
import gzip
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import backtest_pool_store as store


def _payload(sequence, *, saved_at=None, body=None):
    return {
        "schema_version": store.POOL_SCHEMA_VERSION,
        "sequence": sequence,
        "result_id": f"result-{sequence:04d}",
        "saved_at": saved_at or f"2026-08-13T10:00:{sequence % 60:02d}",
        "snapshot": body or {},
    }


# ---------------------------------------------------------------------------
#  编解码：逐类型往返
# ---------------------------------------------------------------------------

def test_nested_tuples_survive_the_round_trip():
    """签名 key 是嵌套 tuple，退化成 list 会让差异卡认不出嵌套。

    对比页靠 ``isinstance(item, tuple)`` 展开嵌套再比键集合。JSON 原生没有
    tuple，不显式标记的话往返后「只有波动率不同」会退化成「期权整组不同」，
    而且不抛异常。
    """
    key = (("source", "wind"), ("params", (("sigma", 0.18), ("K", 100.0))))
    back = store.decode(store.encode(key))
    assert back == key
    assert isinstance(back, tuple)
    assert isinstance(back[1][1], tuple)
    assert isinstance(back[1][1][0], tuple)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinities_are_preserved(value):
    """``daily_net_pnl_rms`` 在空交易日就是 inf，它是取值不是缺失。"""
    assert store.decode(store.encode(value)) == value


def test_nan_stays_nan():
    """``position`` 缺失时是 NaN；转成 None 会让下游拿到 "None" 字符串。"""
    assert math.isnan(store.decode(store.encode(float("nan"))))


def test_numpy_scalars_become_plain_python():
    back = store.decode(store.encode(
        {"i": np.int64(3), "f": np.float64(1.5), "b": np.bool_(True)}))
    assert back == {"i": 3, "f": 1.5, "b": True}
    assert isinstance(back["i"], int) and not isinstance(back["i"], np.integer)


def test_dict_keys_starting_with_dollar_are_not_eaten():
    """编码用 ``$`` 做标签，而快照的键来自用户输入（CSV 列名、结果名）。

    dict 编成键值对列表就是为了这个：不能假设业务键不以 ``$`` 开头。
    """
    back = store.decode(store.encode({"$": 1, "$tup": 2, "正常": 3}))
    assert back == {"$": 1, "$tup": 2, "正常": 3}


def test_callables_are_dropped_instead_of_exploding():
    """``gui_state`` 里嵌着 ``cfg["build"]`` 这类回调，不能等到 dumps 才炸。"""
    back = store.decode(store.encode({"keep": 1, "cb": lambda: None}))
    assert back == {"keep": 1}


def test_dataframe_round_trip_keeps_index_and_name():
    """``daily_frame`` 的索引是交易日序号，曲线横轴就是它。"""
    frame = pd.DataFrame(
        {"net_pnl": [1.5, float("inf")], "tc_paid": [0.1, 0.2]},
        index=pd.Index([0, 1], name="trade_day"))
    back = store.decode(store.encode(frame))
    assert back.equals(frame)
    assert back.index.name == "trade_day"
    assert list(back.columns) == ["net_pnl", "tc_paid"]


def test_timestamp_round_trip():
    stamp = pd.Timestamp("2026-08-13 15:00:00")
    assert store.decode(store.encode(stamp)) == stamp
    day = datetime.date(2026, 8, 13)
    assert store.decode(store.encode(day)) == day


# ---------------------------------------------------------------------------
#  文件读写
# ---------------------------------------------------------------------------

def test_filename_carries_no_business_identifiers(tmp_path):
    """文件名不能出现标的代码或结果名。

    结果池是自动保存的，用户不会逐个命名，而 ``wind_code`` 是持仓标的——
    写进文件名等于把它暴露在目录列表里。展示名在包内。
    """
    body = {
        "name": store.encode("我的雪球 510050"),
        "market_key": store.encode((("wind_code", "510050.SH"),)),
    }
    path = store.write_snapshot(
        _payload(7, body=body), directory=str(tmp_path))
    name = os.path.basename(path)
    assert "510050" not in name
    assert "雪球" not in name
    assert name.startswith("pool-") and name.endswith(".json.gz")


def test_read_all_sorts_by_sequence_not_filename(tmp_path):
    """顺序的权威来源是序号，不是文件名或字典序。"""
    for seq in (3, 1, 12, 2):
        store.write_snapshot(_payload(seq), directory=str(tmp_path))
    payloads, skipped = store.read_all(str(tmp_path))
    assert [item["sequence"] for item in payloads] == [1, 2, 3, 12]
    assert skipped == []


def test_limit_evicts_the_oldest(tmp_path):
    for seq in range(1, store.MAX_RESULTS + 4):
        store.write_snapshot(_payload(seq), directory=str(tmp_path))
    payloads, _skipped = store.read_all(str(tmp_path))
    assert len(payloads) == store.MAX_RESULTS
    assert payloads[0]["sequence"] == 4          # 1..3 被淘汰
    assert payloads[-1]["sequence"] == store.MAX_RESULTS + 3


def test_version_mismatch_is_refused_and_reported(tmp_path):
    """版本不符宁可拒绝：口径改过，硬渲染会静默给出错误结论。

    但必须报出来——静默跳过会让用户以为自己丢了结果，而实际是程序不肯读。
    """
    good = store.write_snapshot(_payload(1), directory=str(tmp_path))
    stale = dict(_payload(2))
    stale["schema_version"] = store.POOL_SCHEMA_VERSION + 99
    store.write_snapshot(stale, directory=str(tmp_path))

    with pytest.raises(store.PoolSchemaError):
        store.read_snapshot(
            os.path.join(str(tmp_path), sorted(os.listdir(str(tmp_path)))[1]))

    payloads, skipped = store.read_all(str(tmp_path))
    assert [item["_path"] for item in payloads] == [good]
    assert len(skipped) == 1
    assert "版本" in skipped[0][1]


def test_corrupt_package_does_not_break_the_whole_load(tmp_path):
    """坏包只跳过自己。整次载入崩在一个坏包上等于把好结果一起弄丢。"""
    good = store.write_snapshot(_payload(1), directory=str(tmp_path))
    (tmp_path / "pool-20260813-000000-0002.json.gz").write_bytes(b"not gzip")
    payloads, skipped = store.read_all(str(tmp_path))
    assert [item["_path"] for item in payloads] == [good]
    assert len(skipped) == 1 and "损坏" in skipped[0][1]


def test_write_is_atomic_and_sweeps_stale_partials(tmp_path):
    """``.part`` 不在 glob 里，没有别的清理点，列目录时顺手扫掉。"""
    store.write_snapshot(_payload(1), directory=str(tmp_path))
    leftover = tmp_path / "pool-20260813-000000-0009.json.gz.part"
    leftover.write_text("崩溃残留")
    store.read_all(str(tmp_path))
    assert not leftover.exists()


def test_same_sequence_never_silently_overwrites(tmp_path):
    first = store.write_snapshot(_payload(1), directory=str(tmp_path))
    second = store.write_snapshot(_payload(1), directory=str(tmp_path))
    assert first != second
    assert len(os.listdir(str(tmp_path))) == 2


def test_written_package_is_gzip_json(tmp_path):
    path = store.write_snapshot(_payload(1), directory=str(tmp_path))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema_version"] == store.POOL_SCHEMA_VERSION


def test_view_state_round_trip(tmp_path):
    """「上次显示了哪几条」单独存：它坏了最多回到全部隐藏，不碰快照。"""
    assert store.read_view_state(str(tmp_path)) == set()
    store.write_view_state({"result-0002", "result-0005"}, str(tmp_path))
    assert store.read_view_state(str(tmp_path)) == {
        "result-0002", "result-0005"}


def test_corrupt_view_state_degrades_to_nothing_selected(tmp_path):
    (tmp_path / "view_state.json").write_text("{ 坏文件", encoding="utf-8")
    assert store.read_view_state(str(tmp_path)) == set()
