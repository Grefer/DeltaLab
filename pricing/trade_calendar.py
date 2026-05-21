# _*_ coding: utf-8 _*_
"""
交易日历模块（离线优先）

提供 A 股交易日序列，解析顺序：
    1) 本地缓存文件 data/tradingday.csv（或 pricing/tradingday.csv）—— 纯离线，默认路径
    2) akshare（免费公开数据源，非 Wind 接口）—— 文件缺失/不够新时刷新
    3) WindPy —— 最后兜底
成功联网获取后会回写本地文件，之后即可纯离线使用。

仓库已预置 data/tradingday.csv（覆盖至 2026 年底）。到期日超出文件范围时，
若联网不可用则沿用旧文件并告警（行为对齐原 MATLAB 的 “tradingday 需要更新”）。
"""

import os
import warnings

import numpy as np
import pandas as pd

_CACHE = None  # 进程内缓存：升序 datetime64[D] 数组


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


def _cache_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return [
        os.path.join(root, "data", "tradingday.csv"),
        os.path.join(here, "tradingday.csv"),
    ]


def _read_local():
    for p in _cache_paths():
        if os.path.exists(p):
            ser = pd.read_csv(p).iloc[:, 0].astype(str)
            arr = _to_dt64(ser)
            arr.sort()
            return arr
    return None


def _write_local(arr):
    p = _cache_paths()[0]
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
        _write_local(fetched)
        _CACHE = fetched
        return fetched

    if local is not None:
        warnings.warn("交易日历不足且联网刷新失败，沿用旧的 data/tradingday.csv，请尽快更新。")
        _CACHE = local
        return local

    raise RuntimeError(
        "无法获取交易日历：请放置 data/tradingday.csv，或安装 akshare / WindPy 以联网获取。"
    )


def trade_days(start, end):
    """返回 [start, end] 闭区间内的交易日 datetime64[D] 升序数组。"""
    cal = load_calendar(start, end)
    s, e = _norm(start), _norm(end)
    return cal[(cal >= s) & (cal <= e)]
