# _*_ coding: utf-8 _*_
"""
Wind 数据接口模块

封装 WindPy API，提供获取历史行情、交易日历等功能，
供对冲回测和期权定价使用。
"""

import functools
import os
import re

import numpy as np
import pandas as pd


# 缓存目录：data/cache
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "cache",
)


# =============================================================================
# 日内可交易分钟数常量表
# =============================================================================
# (exch_eng, sec_type) -> 日内总交易分钟数
# 数据基于常规交易所时段（日盘 + 夜盘，不含集合竞价），用于由 bar_size 推导
# 每日 bar 数（steps_per_day）。sec_type 为 Wind wss 返回的中文字段，实测可能与
# 此表略有差异，如命中失败请按实测值修正。
_TRADING_MINUTES_TABLE: dict[tuple[str, str], int] = {
    # A 股 / ETF（沪深两市 9:30-11:30, 13:00-15:00 共 240 分钟）
    ("SSE", "基金"): 240,
    ("SSE", "股票"): 240,
    ("SZSE", "基金"): 240,
    ("SZSE", "股票"): 240,
    # 金融期货（中金所日盘 9:30-11:30, 13:00-15:00 共 240 分钟，无夜盘）
    ("CFFEX", "指数类"): 240,
    ("CFFEX", "国债类"): 240,
    # 上期所
    ("SHFE", "贵金属"): 570,    # au / ag 夜盘 21:00-02:30，日盘 4h -> 330+240=570
    ("SHFE", "有色"): 480,      # cu/al/zn/pb/ni/sn 夜盘 21:00-01:00 -> 240+240=480
    ("SHFE", "煤焦钢矿"): 360,  # rb/hc/bu 夜盘 21:00-23:00 -> 120+240=360
    # 上海国际能源交易中心
    ("INE", "能源"): 570,       # sc 夜盘 21:00-02:30（lu 例外，见 overrides）
    # 大商所
    ("DCE", "煤焦钢矿"): 360,
    ("DCE", "农产品"): 360,
    ("DCE", "化工"): 360,
    # 郑商所
    ("CZCE", "化工"): 360,
    ("CZCE", "农产品"): 360,
    # 广期所（无夜盘，日盘 4h）
    ("GFEX", "有色"): 240,      # si / lc
}

# 个别品种细分 override（按 code 前缀字母，不区分大小写）
# 用于"同 (exch_eng, sec_type) 但夜盘时长不同"的情况
_SYMBOL_OVERRIDES: dict[str, int] = {
    "LU": 360,  # 低硫燃油，INE 但夜盘 21:00-23:00，共 120+240=360
}


def _extract_symbol_prefix(code: str) -> str:
    """从 Wind 代码中提取字母前缀（大写），如 'au2412.SHF' -> 'AU'。"""
    m = re.match(r"^[A-Za-z]+", str(code))
    return m.group(0).upper() if m else ""


@functools.lru_cache(maxsize=512)
def get_trading_minutes_per_day(code: str) -> int | None:
    """通过 Wind wss 查 exch_eng+sec_type，结合本地常量表返回日内可交易分钟数。

    Wind 不可用 / 字段缺失 / 表里无匹配 -> 返回 None，由调用方决定降级策略。

    Parameters
    ----------
    code : str
        Wind 代码，例如 "510050.SH" / "au2412.SHF"。

    Returns
    -------
    int or None
        日内总交易分钟数；无法判定时返回 None。
    """
    try:
        w = _ensure_wind()
    except Exception:
        return None

    try:
        data = w.wss(code, "exch_eng,sec_type")
    except Exception:
        return None

    if getattr(data, "ErrorCode", -1) != 0:
        return None

    try:
        fields = [f.lower() for f in data.Fields]
        values = [row[0] for row in data.Data]
        rec = dict(zip(fields, values))
        exch = str(rec.get("exch_eng", "")).strip().upper()
        sec_type = str(rec.get("sec_type", "")).strip()
    except Exception:
        return None

    if not exch or not sec_type:
        return None

    minutes = _TRADING_MINUTES_TABLE.get((exch, sec_type))

    # 代码前缀 override（高于 (exch, sec_type)）
    prefix = _extract_symbol_prefix(code)
    if prefix and prefix in _SYMBOL_OVERRIDES:
        minutes = _SYMBOL_OVERRIDES[prefix]

    return minutes


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


def _normalize_intraday_datetime(dt_str, is_start):
    """
    Wind `wsi` 的日期参数格式通常是 "YYYY-MM-DD HH:MM:SS"。
    若用户只传了 "YYYY-MM-DD"，自动补全：
      - is_start=True  -> 09:30:00
      - is_start=False -> 15:00:00
    """
    s = str(dt_str).strip()
    if len(s) <= 10:
        # 仅有日期部分
        s = f"{s} 09:30:00" if is_start else f"{s} 15:00:00"
    return s


def get_intraday_bars(code, start_date, end_date, bar_size="60",
                      fields="close", adjust="F"):
    """
    获取分钟 K 线（intraday bar）

    底层使用 `w.wsi`，支持 1/5/15/30/60 分钟等 BarSize。

    Parameters
    ----------
    code : str
        Wind 代码，如 "510050.SH"
    start_date : str
        起始日期，"YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"。
        仅日期时自动补 "09:30:00"。
    end_date : str
        结束日期，"YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"。
        仅日期时自动补 "15:00:00"。
    bar_size : str
        分钟 bar 大小，"1" / "5" / "15" / "30" / "60" 等（Wind 规范）。
    fields : str
        字段列表（逗号分隔），默认 "close"。
    adjust : str
        复权方式："F"=前复权, "B"=后复权, ""=不复权。
        注意：`wsi` 的复权参数写法与 `wsd` 不完全一致，这里按照
        WindPy 常用写法 `"PriceAdj=B"` 传参；若遇到兼容问题，传 ""
        （不复权）是最保守的选项。

    Returns
    -------
    pd.DataFrame
        index=pd.DatetimeIndex（精确到分钟），columns=fields（小写）
    """
    w = _ensure_wind()
    start_dt = _normalize_intraday_datetime(start_date, is_start=True)
    end_dt = _normalize_intraday_datetime(end_date, is_start=False)

    opts = [f"BarSize={bar_size}"]
    if adjust:
        opts.append(f"PriceAdj={adjust}")
    opt_str = ";".join(opts)

    data = w.wsi(code, fields, start_dt, end_dt, opt_str)
    if data.ErrorCode != 0:
        # 复权键名是 wsi 兼容性的常见踩点：若 PriceAdj 不被识别，先尝试 adjust=""
        hint = ""
        if adjust:
            hint = (
                f"\n  提示：若错误来自复权参数，请尝试 adjust=''（不复权）"
                f"或确认你的 WindPy 版本是否使用 PriceAdj={adjust} 之外的键名。"
            )
        raise RuntimeError(
            f"Wind intraday 数据获取失败 [{code} {bar_size}min]: "
            f"ErrorCode={data.ErrorCode}{hint}"
        )

    df = pd.DataFrame(
        np.array(data.Data).T,
        index=pd.to_datetime(data.Times),
        columns=[f.lower() for f in data.Fields],
    )
    return df.dropna(how="all")


def _intraday_cache_path(code, start, end, bar_size, adjust="F"):
    """intraday 缓存文件路径

    key 必须包含 adjust：F/B/'' 三种复权口径不能混用同一缓存，否则后续
    读回来的序列复权方式与调用方预期不一致。adjust='' 的无复权在文件名
    里固化为 'NA'，避免空串被 OS 文件系统解释出奇怪结果。
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe_code = code.replace("/", "_").replace("\\", "_")
    # 起止日期保留 YYYY-MM-DD 部分，避免 HH:MM 写进文件名
    safe_start = str(start)[:10]
    safe_end = str(end)[:10]
    safe_adj = str(adjust).strip() or "NA"
    fname = (
        f"{safe_code}_{safe_start}_{safe_end}_intraday_"
        f"{bar_size}_{safe_adj}.parquet"
    )
    return os.path.join(_CACHE_DIR, fname)


def get_intraday_close(code, start, end, bar_size="60", adjust="F"):
    """
    获取 intraday 收盘价 `pd.Series`，带 parquet 缓存。

    缓存 key 包含 bar_size，避免不同频率互相污染。

    Parameters
    ----------
    code : str
    start, end : str
        "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"
    bar_size : str
        "1" / "5" / "15" / "30" / "60" 等
    adjust : str
        "F" / "B" / ""

    Returns
    -------
    pd.Series
        index=DatetimeIndex (精确到分钟), values=close
    """
    path = _intraday_cache_path(code, start, end, bar_size, adjust=adjust)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        ser = df["close"] if "close" in df.columns else df.iloc[:, 0]
        ser.index = pd.to_datetime(ser.index)
        ser.name = code
        return ser

    df = get_intraday_bars(code, start, end, bar_size=bar_size,
                           fields="close", adjust=adjust)
    if "close" not in df.columns:
        # 兜底：取第一列作为 close
        ser = df.iloc[:, 0]
    else:
        ser = df["close"]
    ser = ser.dropna()
    ser.name = code

    # 写入缓存
    try:
        ser.to_frame(name="close").to_parquet(path)
    except Exception:
        # 缓存失败不影响主流程
        pass
    return ser


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

def _cache_path(code, start, end, asset_type, adjust=None):
    """返回 parquet 缓存文件路径

    key 在 asset_type 之外再拼入 adjust：当前 load_history_cached 内部对
    equity 写死 PriceAdj=B、future 写死不复权，但把这两个"事实上的复权口径"
    也编码进文件名可以避免以后暴露 adjust 参数后缓存污染。
    TODO: 若后续把 adjust 提成 load_history_cached 的显式入参，调用方必须
    同步把值传到此 key，以免跨口径读到旧缓存。
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # 文件名中不能有冒号等特殊字符
    safe_code = code.replace("/", "_").replace("\\", "_")
    if adjust is None:
        # 按 asset_type 回退到内部默认值，保证 key 唯一
        adjust = "B" if asset_type == "equity" else ""
    safe_adj = str(adjust).strip() or "NA"
    fname = f"{safe_code}_{start}_{end}_{asset_type}_{safe_adj}.parquet"
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
