# _*_ coding: utf-8 _*_
"""交易日历的降级行为（回归）。

这几条守的是同一件事：**日历不够用时不能悄悄返回一段短的**。仓库预置的
``data/tradingday.csv`` 覆盖到 2026 年底，超期之后每一次单次回测都会走到这
里，而免安装版是 ``console=False``、``sys.stderr is None``，``warnings.warn``
在那儿是个静默 no-op——靠告警传递"日历该更新了"等于没有传递。
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pricing import trade_calendar


@pytest.fixture
def offline_calendar(monkeypatch):
    """把两个联网源都掐掉，并保证不碰进程内缓存与真实文件。"""
    monkeypatch.setattr(trade_calendar, "_CACHE", None, raising=False)
    monkeypatch.setattr(
        trade_calendar, "_fetch_akshare",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(
        trade_calendar, "_fetch_wind",
        lambda start, end: (_ for _ in ()).throw(RuntimeError("offline")))
    return trade_calendar


def test_insufficient_coverage_raises_instead_of_truncating(
        offline_calendar, monkeypatch):
    """覆盖不足 + 联网失败 = 报错，不能沿用旧日历返回短区间。"""
    local = np.arange(
        np.datetime64("2026-12-01"), np.datetime64("2027-01-01"),
        dtype="datetime64[D]")
    monkeypatch.setattr(trade_calendar, "_read_local", lambda: local)

    with pytest.raises(RuntimeError) as excinfo:
        trade_calendar.trade_days("2026-12-28", "2027-01-08")

    message = str(excinfo.value)
    # 报错必须说清"覆盖到哪天"和"改哪个文件"，否则用户只知道跑不了。
    assert "2026-12-31" in message
    assert "2027-01-08" in message
    assert trade_calendar._write_path() in message


def test_sufficient_coverage_still_works_offline(
        offline_calendar, monkeypatch):
    """区间落在本地文件之内时不受影响，也不该触发任何联网。"""
    local = np.arange(
        np.datetime64("2026-12-01"), np.datetime64("2027-01-01"),
        dtype="datetime64[D]")
    monkeypatch.setattr(trade_calendar, "_read_local", lambda: local)

    days = trade_calendar.trade_days("2026-12-28", "2026-12-31")
    np.testing.assert_array_equal(
        days, np.array(["2026-12-28", "2026-12-29", "2026-12-30",
                        "2026-12-31"], dtype="datetime64[D]"))


def test_fetch_survives_unwritable_cache(monkeypatch):
    """抓取成功、回写失败时必须继续使用抓取结果。

    ``_write_local`` 的异常此前一路冒到 ``load_calendar``，把"已经拿到正确
    日历"变成一次彻底失败——而只读的 ``.app`` 目录正是最常撞上这条的场景。
    """
    fetched = np.arange(
        np.datetime64("2026-12-01"), np.datetime64("2027-02-01"),
        dtype="datetime64[D]")
    monkeypatch.setattr(trade_calendar, "_CACHE", None, raising=False)
    monkeypatch.setattr(trade_calendar, "_read_local", lambda: None)
    monkeypatch.setattr(trade_calendar, "_fetch_akshare", lambda: fetched)

    def _boom(arr):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(trade_calendar, "_write_local", _boom)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        days = trade_calendar.trade_days("2026-12-28", "2027-01-08")

    assert days[-1] == np.datetime64("2027-01-08")
    assert any("未能写入本地缓存" in str(w.message) for w in caught)


def test_frozen_build_writes_to_user_directory(monkeypatch):
    """冻结态回写到 ``~/.deltalab``，读取时可写副本优先于随包那份。

    ``.app`` 包与安装目录可能只读；结果池、逐 bar 缓存、运行日志三处早就是
    这个约定，交易日历此前是仓库里唯一还写在模块旁边的例外。
    """
    monkeypatch.setattr("sys.frozen", True, raising=False)
    write_path = trade_calendar._write_path()
    assert ".deltalab" in write_path
    assert trade_calendar._cache_paths()[0] == write_path
    # 随包那份仍然读得到，只是排在可写副本之后。
    assert len(trade_calendar._cache_paths()) > 1
