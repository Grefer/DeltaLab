# _*_ coding: utf-8 _*_
"""
Wind 数据接口模块

封装 WindPy API，提供获取历史行情、交易日历等功能，
供对冲回测和期权定价使用。
"""

import functools
import os

import numpy as np
import pandas as pd


# 缓存目录：data/cache
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "cache",
)


def _ensure_wind():
    """启动 Wind 连接，若未安装则抛出提示"""
    try:
        from WindPy import w
    except ImportError:
        raise ImportError(
            "未安装 WindPy，请安装 Wind 金融终端并配置 Python 插件。\n"
            "  pip install WindPy  或在 Wind 终端中设置 Python 接口。"
        )
    if not w.isconnected():
        result = w.start()
        if result.ErrorCode != 0:
            raise ConnectionError(f"Wind 连接失败: {result.Data}")
    return w


def get_close_prices(code, start_date, end_date, adjust="F"):
    """
    获取标的收盘价序列

    Parameters
    ----------
    code : str
        Wind 代码，例如 "000001.SH", "510050.SH", "IF2406.CFE"
    start_date : str
        起始日期，"YYYY-MM-DD"
    end_date : str
        结束日期，"YYYY-MM-DD"
    adjust : str
        复权方式："F"=前复权, "B"=后复权, ""=不复权

    Returns
    -------
    pd.Series
        index=日期, values=收盘价
    """
    w = _ensure_wind()
    price_adj = f"PriceAdj={adjust}" if adjust else ""
    data = w.wsd(code, "close", start_date, end_date, price_adj)

    if data.ErrorCode != 0:
        raise RuntimeError(f"Wind 数据获取失败 [{code}]: ErrorCode={data.ErrorCode}")

    series = pd.Series(data.Data[0], index=pd.to_datetime(data.Times), name=code)
    series = series.dropna()
    return series


def get_trade_dates(start_date, end_date, exchange="SSE"):
    """
    获取交易日序列

    Parameters
    ----------
    exchange : str
        交易所代码: "SSE"=上交所, "SZSE"=深交所, "CFFEX"=中金所

    Returns
    -------
    list of datetime
    """
    w = _ensure_wind()
    data = w.tdays(start_date, end_date, f"TradingCalendar={exchange}")
    if data.ErrorCode != 0:
        raise RuntimeError(f"Wind 交易日历获取失败: ErrorCode={data.ErrorCode}")
    return data.Data[0]


def get_ohlcv(code, start_date, end_date, adjust="F"):
    """
    获取 OHLCV 全量行情

    Returns
    -------
    pd.DataFrame
        columns: ['open', 'high', 'low', 'close', 'volume']
    """
    w = _ensure_wind()
    fields = "open,high,low,close,volume"
    price_adj = f"PriceAdj={adjust}" if adjust else ""
    data = w.wsd(code, fields, start_date, end_date, price_adj)

    if data.ErrorCode != 0:
        raise RuntimeError(f"Wind 数据获取失败 [{code}]: ErrorCode={data.ErrorCode}")

    df = pd.DataFrame(
        np.array(data.Data).T,
        index=pd.to_datetime(data.Times),
        columns=[f.lower() for f in data.Fields],
    )
    return df.dropna()


def get_hist_vol(code, start_date, end_date, window=20, adjust="F"):
    """
    计算历史波动率

    Parameters
    ----------
    window : int
        滚动窗口天数 (交易日)

    Returns
    -------
    pd.Series
        年化历史波动率
    """
    prices = get_close_prices(code, start_date, end_date, adjust)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    vol = log_ret.rolling(window).std() * np.sqrt(243)
    vol.name = f"HV{window}"
    return vol.dropna()


# =============================================================================
# 滚动历史回测：缓存 / 收益 / rebase / 合约规格
# =============================================================================

def _cache_path(code, start, end, asset_type):
    """返回 parquet 缓存文件路径"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # 文件名中不能有冒号等特殊字符
    safe_code = code.replace("/", "_").replace("\\", "_")
    fname = f"{safe_code}_{start}_{end}_{asset_type}.parquet"
    return os.path.join(_CACHE_DIR, fname)


def load_history_cached(code, start, end, asset_type="equity"):
    """
    带 parquet 缓存的历史收盘价读取

    Parameters
    ----------
    code : str
        Wind 代码
    start, end : str
        "YYYY-MM-DD"
    asset_type : str
        "equity" -> 后复权 (PriceAdj=B)
        "future" -> 不传 PriceAdj (期货无分红)

    Returns
    -------
    pd.Series
        index 为 DatetimeIndex, values 为收盘价
    """
    path = _cache_path(code, start, end, asset_type)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        # 只有一列；兼容 index 名称
        ser = df.iloc[:, 0]
        ser.index = pd.to_datetime(ser.index)
        ser.name = code
        return ser

    w = _ensure_wind()
    if asset_type == "equity":
        data = w.wsd(code, "close", start, end, "PriceAdj=B")
    elif asset_type == "future":
        # 期货不传 PriceAdj
        data = w.wsd(code, "close", start, end, "")
    else:
        raise ValueError(f"未知 asset_type: {asset_type}")

    if data.ErrorCode != 0:
        raise RuntimeError(
            f"Wind 数据获取失败 [{code}]: ErrorCode={data.ErrorCode}"
        )

    ser = pd.Series(
        data.Data[0],
        index=pd.to_datetime(data.Times),
        name=code,
    ).dropna()

    # 写入缓存
    ser.to_frame(name="close").to_parquet(path)
    return ser


def get_log_returns(code, start, end, asset_type="equity"):
    """
    读取历史收盘价并返回对数收益序列（从 t=1 开始，已 dropna）

    Returns
    -------
    pd.Series
        log(P[t]/P[t-1])
    """
    prices = load_history_cached(code, start, end, asset_type)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    log_ret.name = f"{code}_logret"
    return log_ret


def rebase_path(log_return_slice, s0):
    """
    把一段 log return 序列 rebase 到指定的 s0

    约定：log_return_slice 的第 0 个元素对应 t=1 的收益，
    因此返回序列长度为 len(log_return_slice)+1：
        S[0] = s0
        S[k] = s0 * exp(sum(log_return_slice[:k]))    for k>=1

    Parameters
    ----------
    log_return_slice : pd.Series
        对数收益切片
    s0 : float
        起点价格

    Returns
    -------
    pd.Series
        rebase 后的价格序列
    """
    if len(log_return_slice) == 0:
        return pd.Series([s0], name="rebased")

    cum = np.cumsum(log_return_slice.values)
    prices = np.concatenate([[s0], s0 * np.exp(cum)])

    # index：在前面补一个 t0（用 log_return 第一天往前推 1 天作为占位，
    # 若没有可用的日期则用 RangeIndex）
    try:
        first_date = log_return_slice.index[0]
        # 前面补一个"虚拟起点"，用 NaT 会影响后续切片；这里直接用第一个日期作为 t0
        # 更稳妥：向前退一个位置，用 BDay
        t0 = first_date - pd.Timedelta(days=1)
        idx = pd.DatetimeIndex([t0]).append(pd.DatetimeIndex(log_return_slice.index))
    except Exception:
        idx = pd.RangeIndex(len(prices))

    return pd.Series(prices, index=idx, name="rebased")


@functools.lru_cache(maxsize=128)
def get_contract_spec(code):
    """
    查询合约规格：期货取 contractmultiplier，股票/ETF 返回 1.0

    判定规则：代码后缀是 .CFE / .SHF / .DCE / .CZC / .INE / .GFE 视为期货。

    Returns
    -------
    dict
        {"multiplier": float, "is_future": bool}
    """
    future_suffixes = (".CFE", ".SHF", ".DCE", ".CZC", ".INE", ".GFE")
    upper = code.upper()
    is_future = any(upper.endswith(s) for s in future_suffixes)

    if not is_future:
        return {"multiplier": 1.0, "is_future": False}

    try:
        w = _ensure_wind()
        data = w.wsd(code, "contractmultiplier")
        if data.ErrorCode != 0:
            raise RuntimeError(
                f"Wind contractmultiplier 查询失败 [{code}]: "
                f"ErrorCode={data.ErrorCode}"
            )
        mult = float(data.Data[0][0])
    except ImportError:
        # WindPy 不可用时返回占位，调用方需自行处理
        mult = 1.0

    return {"multiplier": mult, "is_future": True}
