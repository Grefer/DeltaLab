# _*_ coding: utf-8 _*_
"""
交易日历模块（离线优先）

提供 A 股交易日序列，解析顺序：
    1) 本地缓存文件 data/tradingday.csv（或 pricing/tradingday.csv）—— 纯离线，默认路径
    2) akshare（免费公开数据源，非 Wind 接口）—— 文件缺失/不够新时刷新
    3) WindPy —— 最后兜底
成功联网获取后会回写本地文件，之后即可纯离线使用。

仓库已预置 data/tradingday.csv（覆盖至 2026 年底）。到期日超出文件范围且联网
刷新失败时**直接报错**，不再沿用覆盖不足的旧文件——旧行为（对齐原 MATLAB 的
“tradingday 需要更新”）会返回一个被悄悄截断的区间，而它依赖的那句
``warnings.warn`` 在冻结包里是个 no-op（console=False → sys.stderr is None）。

刷新结果的回写目标：开发态是仓库内 data/tradingday.csv，冻结后是
~/.deltalab/data/tradingday.csv（.app 包只读）。读取时可写副本优先于随包那份。
"""

import os
import sys
import threading
import warnings

import numpy as np
import pandas as pd

_CACHE = None  # 进程内缓存：升序 datetime64[D] 数组
# 定价路径上会读它（Option_SNB 解析敲出观察日），而策略优选现在会并行跑
# 多个分段。加载可能落到联网抓取与写本地文件，两个线程同时进来会重复抓、
# 也可能把 data/tradingday.csv 写坏。
_CACHE_LOCK = threading.RLock()


def _norm(x):
    """把 'YYYYMMDD' / 'YYYY-MM-DD' / datetime / datetime64 统一为 datetime64[D]"""
    if isinstance(x, np.datetime64):
        return x.astype('datetime64[D]')
    if isinstance(x, str):
        s = x.replace('-', '').replace('/', '')
        return np.datetime64(f"{s[0:4]}-{s[4:6]}-{s[6:8]}", 'D')
    return np.datetime64(x).astype('datetime64[D]')


def _to_dt64(seq):
    return np.array([_norm(x) for x in seq], dtype='datetime64[D]')


def _write_path():
    """联网刷新后回写到哪。

    开发态写仓库内的 ``data/tradingday.csv``；打包冻结后写
    ``~/.deltalab/data`` —— ``.app`` 包与安装目录可能只读，把刷新结果往里
    写必然失败。仓库里其它三处运行期产物（``history_store.results_dir``、
    ``history_bar_cache.cache_dir``、``deltalab_log.log_dir``）早就是这个
    约定，此前只有交易日历还写在模块旁边，是唯一的例外。
    """
    if getattr(sys, "frozen", False):
        return os.path.join(
            os.path.expanduser("~"), ".deltalab", "data", "tradingday.csv")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "tradingday.csv")


def _cache_paths():
    """读取顺序：刷新过的可写副本优先，再退回随包发布的那一份。"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    paths = [
        _write_path(),
        os.path.join(root, "data", "tradingday.csv"),
        os.path.join(here, "tradingday.csv"),
    ]
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def _read_local():
    for p in _cache_paths():
        if os.path.exists(p):
            ser = pd.read_csv(p).iloc[:, 0].astype(str)
            arr = _to_dt64(ser)
            arr.sort()
            return arr
    return None


def _write_local(arr):
    p = _write_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ser = pd.Series([str(d).replace('-', '') for d in arr], name='trade_date')
    ser.to_csv(p, index=False)


def _fetch_akshare():
    import akshare as ak
    df = ak.tool_trade_date_hist_sina()
    arr = _to_dt64(df['trade_date'])
    arr.sort()
    return arr


def _fetch_wind(start, end):
    try:
        from . import wind_data
    except ImportError:
        import wind_data
    dates = wind_data.get_trade_dates(start, end)
    arr = _to_dt64(dates)
    arr.sort()
    return arr


def _covers(arr, start, end):
    if arr is None or arr.size == 0:
        return False
    if start is not None and arr[0] > _norm(start):
        return False
    if end is not None and arr[-1] < _norm(end):
        return False
    return True


def load_calendar(start=None, end=None):
    """返回完整交易日 datetime64[D] 升序数组（按离线优先顺序解析）。"""
    global _CACHE
    # 快路径无锁：_CACHE 的重绑定是原子的，命中时读到的要么是旧数组要么是
    # 新数组，两者都是完整可用的。只有需要加载时才进锁。
    if _covers(_CACHE, start, end):
        return _CACHE
    with _CACHE_LOCK:
        return _load_calendar_locked(start, end)


def _load_calendar_locked(start, end):
    global _CACHE
    # 进锁后重查：可能已被另一个线程填好了。
    if _covers(_CACHE, start, end):
        return _CACHE

    local = _read_local()
    if _covers(local, start, end):
        _CACHE = local
        return local

    # 本地缺失或范围不足 —— 尝试联网刷新（akshare 优先，Wind 兜底）
    fetched = None
    for fetch in (_fetch_akshare, lambda: _fetch_wind(start, end)):
        try:
            fetched = fetch()
            if _covers(fetched, start, end):
                break
        except Exception:
            fetched = None

    if fetched is not None and fetched.size:
        try:
            _write_local(fetched)
        except OSError as exc:
            # 抓到了就是抓到了。回写只为下次能离线用，写不进去（只读的 .app、
            # 磁盘满、权限不对）没有任何理由把这次成功的刷新一起作废——此前
            # 这句异常会一路冒出 load_calendar，把"已经拿到正确日历"变成一次
            # 彻底失败。
            warnings.warn(
                f"交易日历已刷新但未能写入本地缓存（{exc}），本次仍使用刷新结果；"
                f"下次启动会再抓一遍。")
        _CACHE = fetched
        return fetched

    if local is not None:
        # 覆盖不足就必须失败，不能"告警后沿用旧日历"。
        #
        # 沿用旧的会返回一个**被悄悄截断**的区间：Option_SNB 拿它换算敲出观
        # 察日会撞出 IndexError 或"观察日不在日历中"，界面上是一句对不上号的
        # 报错；而那句 warnings.warn 在冻结包里根本不存在——deltalab.spec 是
        # console=False，冻结后 sys.stderr is None，警告是个静默 no-op。
        # 于是"已经告过警了"只是开发机上的错觉。这里直接抛，把唯一一次说得
        # 清楚的机会用在真正会看到的地方。
        raise RuntimeError(
            f"交易日历只覆盖到 {local[-1]}，不足以支撑请求区间 "
            f"[{_norm(start) if start is not None else local[0]}, "
            f"{_norm(end) if end is not None else local[-1]}]，"
            f"且 akshare / WindPy 联网刷新均失败。"
            f"请联网后重试，或用更新的交易日历替换 {_write_path()}。"
        )

    raise RuntimeError(
        "无法获取交易日历：请放置 data/tradingday.csv，或安装 akshare / WindPy 以联网获取。"
    )


def trade_days(start, end):
    """返回 [start, end] 闭区间内的交易日 datetime64[D] 升序数组。"""
    cal = load_calendar(start, end)
    s, e = _norm(start), _norm(end)
    return cal[(cal >= s) & (cal <= e)]
