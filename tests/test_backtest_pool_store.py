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


def test_string_arrays_survive_the_round_trip():
    """非数值数组曾经**写得进读不出**：包在盘上，载入时抛 ValueError。

    ``decode`` 一律按 float 还原，而 ``encode`` 对任意 dtype 都打同一个标
    签。结果是快照写盘成功、下次启动 ``_payload_to_snapshot`` 抛
    ValueError，这一条被记成「字段无法还原」丢掉，随后还会被上限淘汰。
    """
    original = np.array(["段一", "段二"])
    back = store.decode(store.encode(original))
    assert list(back) == ["段一", "段二"]
    # 数值数组仍旧按数值还原，不能因为记了 dtype 就变成 object。
    floats = store.decode(store.encode(np.array([1.5, 2.5])))
    assert floats.dtype.kind == "f"
    assert list(floats) == [1.5, 2.5]


def test_object_arrays_do_not_explode_at_json_dumps():
    """arr 分支曾是唯一不过滤 ``_DROP`` 的容器，哨兵一路带到 json.dumps。

    其余三个容器都过滤了。漏这一个的后果就是 ``_DROP`` 存在的理由本身：
    「只在顶层判 callable 会让它一路走到 json.dumps 才炸」。
    """
    encoded = store.encode({"x": np.array([1.0, len], dtype=object)})
    text = json.dumps(encoded, ensure_ascii=False)
    assert "1.0" in text


def test_series_and_index_survive_the_round_trip():
    """DataFrame 早就逐类型往返了，Series / Index 却被无声丢掉。"""
    series = pd.Series([1.0, 2.0], index=pd.Index([10, 20], name="trade_day"),
                       name="净损益")
    back = store.decode(store.encode(series))
    assert isinstance(back, pd.Series)
    assert list(back) == [1.0, 2.0]
    assert list(back.index) == [10, 20]
    assert back.index.name == "trade_day"
    assert back.name == "净损益"

    stamps = pd.DatetimeIndex(["2026-05-20 09:30", "2026-05-20 09:31"])
    restored = store.decode(store.encode(stamps))
    assert isinstance(restored, pd.DatetimeIndex)
    assert list(restored) == list(stamps)


def test_non_string_dict_keys_survive_the_round_trip():
    """键被 ``str()`` 掉会让重放配方在重开程序前后算出两个缓存 key。

    ``history_bar_cache.key_for_recipe`` 直接摘这份配方，{1: …} 往返成
    {"1": …} 之后摘要不同——同一条快照「刚保留」时命中的缓存，重开程序后
    永远命中不了，加载明细每次都要重跑 620 ms 且无从察觉。
    """
    original = {"warmup": {1: 0.5, 2: 0.25}, "文字键": "照旧"}
    assert store.decode(store.encode(original)) == original


@pytest.mark.parametrize("text", ["[1, 2]", '{"selected": 3}', '"字符串"'])
def test_view_state_that_is_not_an_object_degrades_to_nothing_selected(
        tmp_path, text):
    """合法 JSON 但形状不对时也只能回到「全部隐藏」。

    ``payload.get`` 对数组抛 AttributeError，集合推导对数字抛 TypeError，
    两者都不在原来的 (OSError, ValueError) 里。而调用方那一句在保护
    ``read_all`` 的 try 之外，漏出去会让整池结果都载不进来。
    """
    (tmp_path / "view_state.json").write_text(text, encoding="utf-8")
    assert store.read_view_state(str(tmp_path)) == set()


def test_limit_does_not_read_any_package_while_under_the_cap(
        tmp_path, monkeypatch):
    """没到上限时一个包都不该解压。

    ``write_snapshot`` 每存一条就调一次 enforce_limit，而 read_all 会把每个
    包（各带一整条价格序列）gunzip + json.load 一遍：实测 20 份一年 1 分钟
    序列的包，光这一趟就 0.14 s，每次「保留结果」都白付。
    """
    for seq in range(1, 4):
        store.write_snapshot(_payload(seq), directory=str(tmp_path))

    calls = []
    original = store.read_all

    def _counted(directory=None):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(store, "read_all", _counted)
    assert store.enforce_limit(directory=str(tmp_path)) == []
    assert calls == []


def test_limit_orders_by_filename_and_matches_the_authoritative_order(
        tmp_path, monkeypatch):
    """满池时也不该解压任何包——而结果池稳态就是满的。

    文件名是 ``default_filename`` 拿 payload 的 sequence 与 saved_at 拼出来
    的，顺序由构造保证一致。这里同时钉死「淘汰结果与读包定序完全一致」，
    免得哪天文件名格式改了、顺序悄悄换成字典序（1, 10, 11, 2 …）。
    """
    for seq in (3, 1, 12, 2, 11):
        store.write_snapshot(_payload(seq), directory=str(tmp_path),
                             enforce=False)

    def _forbidden(directory=None):
        raise AssertionError("满池淘汰不该读包")

    monkeypatch.setattr(store, "read_all", _forbidden)
    evicted = store.enforce_limit(3, directory=str(tmp_path))

    assert len(evicted) == 2
    monkeypatch.undo()
    payloads, _skipped = store.read_all(str(tmp_path))
    assert [item["sequence"] for item in payloads] == [3, 11, 12]


def test_limit_falls_back_to_reading_when_a_name_is_unrecognisable(tmp_path):
    """名字认不出来时必须退回读包定序，绝不按名字猜着删。"""
    paths = [store.write_snapshot(_payload(seq), directory=str(tmp_path),
                                  enforce=False)
             for seq in (1, 2, 3)]
    stray = tmp_path / ("我自己改的名字" + store._SUFFIX)
    stray.write_bytes(b"not gzip")

    evicted = store.enforce_limit(2, directory=str(tmp_path))

    # 走的是 read_all 那条路：坏包不在 payloads 里，于是淘汰的是包内序号最
    # 小的那份，而认不出的文件原样留着——没有按文件名猜着删。
    assert evicted == [paths[0]]
    assert stray.exists()
    payloads, _skipped = store.read_all(str(tmp_path))
    assert [item["sequence"] for item in payloads] == [2, 3]
