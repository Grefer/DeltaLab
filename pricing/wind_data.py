# _*_ coding: utf-8 _*_
"""
Wind 数据接口模块

封装 WindPy API，提供获取历史行情、交易日历等功能，
供对冲回测和期权定价使用。
"""

import numpy as np
import pandas as pd


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
