# _*_ coding: utf-8 _*_
"""
Wind 数据接口模块

封装 WindPy API，提供获取历史行情、交易日历等功能，
供对冲回测和期权定价使用。
"""

import datetime as _datetime
import functools
import os
import re
import sys

import numpy as np
import pandas as pd


# 缓存目录：开发态为 <repo>/data/cache; 打包后(PyInstaller 冻结)切换到
# 用户主目录 ~/.deltalab/cache, 避免写入只读的 .app 包/安装目录.
def _default_cache_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), ".deltalab", "cache")
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "cache",
    )


_CACHE_DIR = _default_cache_dir()


_FUTURES_HISTORY_SUFFIXES = ("CFE", "SHF", "DCE", "CZC", "INE", "GFE")
_FUTURES_PRODUCT_CODE_RE = re.compile(
    rf"^(?P<product>[A-Z]+)\.(?P<exchange>{'|'.join(_FUTURES_HISTORY_SUFFIXES)})$"
)
_FUTURES_CONTRACT_CODE_RE = re.compile(
    rf"^(?P<product>[A-Z]+)(?P<delivery>\d{{3,4}})\."
    rf"(?P<exchange>{'|'.join(_FUTURES_HISTORY_SUFFIXES)})$"
)


def classify_wind_history_code(code):
    """判定历史择优代码是具体标的还是期货品种样本池。

    ``P.DCE`` 这类“字母品种 + 期货交易所”代码返回 ``product_pool``；
    ``P2609.DCE`` 以及股票、ETF 等其它 Wind 代码返回 ``single``。具体
    期货合约必须带 3 或 4 位交割年月数字，避免把 ``P00.DCE`` 等连续
    合约别名误识别为可跨合约汇集的品种入口。
    """
    normalized = str(code or "").strip().upper()
    if not normalized:
        raise ValueError("Wind 代码不能为空")
    product_match = _FUTURES_PRODUCT_CODE_RE.fullmatch(normalized)
    if product_match:
        return {
            "mode": "product_pool",
            "code": normalized,
            "product": product_match.group("product"),
            "exchange": product_match.group("exchange"),
        }
    contract_match = _FUTURES_CONTRACT_CODE_RE.fullmatch(normalized)
    if contract_match:
        return {
            "mode": "single",
            "code": normalized,
            "product": contract_match.group("product"),
            "exchange": contract_match.group("exchange"),
            "delivery": contract_match.group("delivery"),
            "is_futures_contract": True,
        }
    return {
        "mode": "single",
        "code": normalized,
        "is_futures_contract": False,
    }


def get_main_contract_history(product_code, start_date, end_date):
    """返回品种连续代码在各历史交易日实际对应的主力合约。

    Wind 的 ``trade_hiscode`` 字段按查询日期返回当时可见的历史主力代码，
    因而截止日之后才成为主力的合约不会进入结果。返回值索引是交易日期，
    值是标准化后的具体合约 Wind 代码。
    """
    classification = classify_wind_history_code(product_code)
    if classification["mode"] != "product_pool":
        raise ValueError(
            "主力合约历史只能查询品种代码，例如 P.DCE；"
            f"当前为 {classification['code']!r}"
        )
    start_ts, end_ts, _, _ = _validate_intraday_range(start_date, end_date)
    w = _ensure_wind()
    data = w.wsd(
        classification["code"], "trade_hiscode",
        start_date, end_date, "",
    )
    if getattr(data, "ErrorCode", -1) != 0:
        raise RuntimeError(
            f"Wind 主力合约历史获取失败 [{classification['code']}]: "
            f"ErrorCode={getattr(data, 'ErrorCode', None)}"
        )
    try:
        values = data.Data[0]
        index = pd.to_datetime(data.Times)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Wind 主力合约历史返回结构异常 [{classification['code']}]"
        ) from exc

    series = pd.Series(values, index=index, name="main_contract")
    normalized_index = pd.DatetimeIndex(series.index).normalize()
    series = series.loc[
        (normalized_index >= start_ts.normalize())
        & (normalized_index <= end_ts.normalize())
    ]
    series = series.dropna().map(lambda value: str(value).strip().upper())
    series = series[series.ne("")]
    valid = []
    for value in series:
        parsed = classify_wind_history_code(value)
        valid.append(bool(
            parsed.get("is_futures_contract")
            and parsed.get("product") == classification["product"]
            and parsed.get("exchange") == classification["exchange"]
        ))
    series = series.loc[np.asarray(valid, dtype=bool)]
    if series.empty:
        raise ValueError(
            f"{classification['code']} 在 {start_date} 至 {end_date} "
            "没有可用的历史主力合约映射"
        )
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    return series[~series.index.duplicated(keep="last")]


# =============================================================================
# 日内连续交易段常量表
# =============================================================================
# Wind 的 BarSize 会在每个连续 session 结束时保留不足一个 BarSize 的尾
# bar，因此每日 bar 数必须按 sum(ceil(segment / bar_size)) 计算，不能只
# 对总分钟数做一次 ceil。sec_type 是 Wind wss 的实测中文值。
_SECURITY_DAY_SESSIONS = (120, 120)
_COMMODITY_DAY_SESSIONS = (75, 60, 90)
_COMMODITY_NIGHT_120_SESSIONS = (120, *_COMMODITY_DAY_SESSIONS)
_COMMODITY_NIGHT_240_SESSIONS = (240, *_COMMODITY_DAY_SESSIONS)
_COMMODITY_NIGHT_330_SESSIONS = (330, *_COMMODITY_DAY_SESSIONS)
_CFFEX_BOND_SESSIONS = (120, 135)

# 与上面的连续交易段分钟数一一对应的墙钟时段。这里不另建一套
# ``(exchange, sec_type)`` 分类表；公开查询函数先走现有的分钟段分类，
# 再由该映射取得时钟范围，避免 bar 数和“某时刻是否交易”逐渐漂移。
_SECURITY_DAY_CLOCK_RANGES = (
    (_datetime.time(9, 30), _datetime.time(11, 30)),
    (_datetime.time(13, 0), _datetime.time(15, 0)),
)
_COMMODITY_DAY_CLOCK_RANGES = (
    (_datetime.time(9, 0), _datetime.time(10, 15)),
    (_datetime.time(10, 30), _datetime.time(11, 30)),
    (_datetime.time(13, 30), _datetime.time(15, 0)),
)
_CFFEX_BOND_CLOCK_RANGES = (
    (_datetime.time(9, 30), _datetime.time(11, 30)),
    (_datetime.time(13, 0), _datetime.time(15, 15)),
)
_TRADING_SESSION_CLOCK_RANGES_BY_MINUTES = {
    _SECURITY_DAY_SESSIONS: _SECURITY_DAY_CLOCK_RANGES,
    _COMMODITY_DAY_SESSIONS: _COMMODITY_DAY_CLOCK_RANGES,
    _COMMODITY_NIGHT_120_SESSIONS: (
        (_datetime.time(21, 0), _datetime.time(23, 0)),
        *_COMMODITY_DAY_CLOCK_RANGES,
    ),
    _COMMODITY_NIGHT_240_SESSIONS: (
        (_datetime.time(21, 0), _datetime.time(1, 0)),
        *_COMMODITY_DAY_CLOCK_RANGES,
    ),
    _COMMODITY_NIGHT_330_SESSIONS: (
        (_datetime.time(21, 0), _datetime.time(2, 30)),
        *_COMMODITY_DAY_CLOCK_RANGES,
    ),
    _CFFEX_BOND_SESSIONS: _CFFEX_BOND_CLOCK_RANGES,
}

_TRADING_SESSION_MINUTES_TABLE: dict[tuple[str, str], tuple[int, ...]] = {
    # A 股 / ETF：9:30-11:30、13:00-15:00。
    ("SSE", "基金"): _SECURITY_DAY_SESSIONS,
    ("SSE", "股票"): _SECURITY_DAY_SESSIONS,
    ("SZSE", "基金"): _SECURITY_DAY_SESSIONS,
    ("SZSE", "股票"): _SECURITY_DAY_SESSIONS,
    # 中金所：指数类下午到 15:00，国债/利率类到 15:15。
    ("CFFEX", "指数类"): _SECURITY_DAY_SESSIONS,
    ("CFFEX", "国债类"): _CFFEX_BOND_SESSIONS,
    ("CFFEX", "利率类"): _CFFEX_BOND_SESSIONS,
    # 少数 Wind 版本返回交易所简称 CFE。
    ("CFE", "指数类"): _SECURITY_DAY_SESSIONS,
    ("CFE", "国债类"): _CFFEX_BOND_SESSIONS,
    ("CFE", "利率类"): _CFFEX_BOND_SESSIONS,
    # 商品期货日盘：09:00-10:15、10:30-11:30、13:30-15:00。
    ("SHFE", "贵金属"): _COMMODITY_NIGHT_330_SESSIONS,
    ("SHFE", "有色"): _COMMODITY_NIGHT_240_SESSIONS,
    ("SHFE", "煤焦钢矿"): _COMMODITY_NIGHT_120_SESSIONS,
    ("INE", "能源"): _COMMODITY_NIGHT_330_SESSIONS,
    ("DCE", "煤焦钢矿"): _COMMODITY_NIGHT_120_SESSIONS,
    ("DCE", "农产品"): _COMMODITY_NIGHT_120_SESSIONS,
    ("DCE", "油脂油料"): _COMMODITY_NIGHT_120_SESSIONS,
    ("DCE", "化工"): _COMMODITY_NIGHT_120_SESSIONS,
    ("CZCE", "化工"): _COMMODITY_NIGHT_120_SESSIONS,
    ("CZCE", "农产品"): _COMMODITY_NIGHT_120_SESSIONS,
    ("GFEX", "有色"): _COMMODITY_DAY_SESSIONS,
}

# 个别品种细分 override（按 code 前缀字母，不区分大小写）。这些信息本地
# 即可确定，Wind 静态字段不可用时仍能返回可靠 session。
_SYMBOL_SESSION_OVERRIDES: dict[str, tuple[int, ...]] = {
    "LU": _COMMODITY_NIGHT_120_SESSIONS,
    "P": _COMMODITY_NIGHT_120_SESSIONS,
    # 上期能源同一大类下的连续交易长度并不相同；EC 没有夜盘。
    "SC": _COMMODITY_NIGHT_330_SESSIONS,
    "NR": _COMMODITY_NIGHT_120_SESSIONS,
    "BC": _COMMODITY_NIGHT_240_SESSIONS,
    "EC": _COMMODITY_DAY_SESSIONS,
    # 上期所线材没有夜盘；胶版印刷纸为 21:00-23:00。
    "WR": _COMMODITY_DAY_SESSIONS,
    "OP": _COMMODITY_NIGHT_120_SESSIONS,
    # 大商所 / 郑商所中明确仅有日盘的品种。交易所大类 sec_type 较宽，
    # 若不在品种层覆盖，会被同类中存在夜盘的合约误判。
    "JD": _COMMODITY_DAY_SESSIONS,
    "FB": _COMMODITY_DAY_SESSIONS,
    "BB": _COMMODITY_DAY_SESSIONS,
    "LH": _COMMODITY_DAY_SESSIONS,
    "AP": _COMMODITY_DAY_SESSIONS,
    "CJ": _COMMODITY_DAY_SESSIONS,
    "PK": _COMMODITY_DAY_SESSIONS,
    "SF": _COMMODITY_DAY_SESSIONS,
    "SM": _COMMODITY_DAY_SESSIONS,
    "UR": _COMMODITY_DAY_SESSIONS,
    "PM": _COMMODITY_DAY_SESSIONS,
    "WH": _COMMODITY_DAY_SESSIONS,
    "RI": _COMMODITY_DAY_SESSIONS,
    "JR": _COMMODITY_DAY_SESSIONS,
    "LR": _COMMODITY_DAY_SESSIONS,
    "RS": _COMMODITY_DAY_SESSIONS,
    # 广期所当前常用品种均为商品日盘，无固定夜盘。
    "SI": _COMMODITY_DAY_SESSIONS,
    "LC": _COMMODITY_DAY_SESSIONS,
    "PS": _COMMODITY_DAY_SESSIONS,
    "T": _CFFEX_BOND_SESSIONS,
    "TF": _CFFEX_BOND_SESSIONS,
    "TS": _CFFEX_BOND_SESSIONS,
    "TL": _CFFEX_BOND_SESSIONS,
}

# 旧常量仍供诊断工具与兼容调用读取；权威 bar 数使用连续交易段表。
_TRADING_MINUTES_TABLE: dict[tuple[str, str], int] = {
    key: sum(segments)
    for key, segments in _TRADING_SESSION_MINUTES_TABLE.items()
}
_SYMBOL_OVERRIDES: dict[str, int] = {
    prefix: sum(segments)
    for prefix, segments in _SYMBOL_SESSION_OVERRIDES.items()
}


def _extract_symbol_prefix(code: str) -> str:
    """从 Wind 代码中提取字母前缀（大写），如 'au2412.SHF' -> 'AU'。"""
    m = re.match(r"^[A-Za-z]+", str(code))
    return m.group(0).upper() if m else ""


@functools.lru_cache(maxsize=512)
def _get_wind_market_classification(code: str):
    """返回 Wind ``(exch_eng, sec_type)``；无法识别时返回 None。"""
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

    return (exch, sec_type) if exch and sec_type else None


@functools.lru_cache(maxsize=512)
def _get_trading_session_minutes(code: str):
    """按代码返回连续交易段分钟数；未知元数据返回 None。"""
    local = _get_local_trading_session_minutes(code)
    if local is not None:
        return local

    classification = _get_wind_market_classification(str(code))
    if classification is None:
        return None
    return _TRADING_SESSION_MINUTES_TABLE.get(classification)


def _get_local_trading_session_minutes(code: str):
    """仅用代码本身可确定的规则返回交易段，不连接 Wind。"""
    normalized = str(code).strip().upper()
    prefix = _extract_symbol_prefix(code)
    override = _SYMBOL_SESSION_OVERRIDES.get(prefix)
    if override is not None:
        return override

    # 沪深证券和中金所指数代码的交易所后缀足以判定。中金所国债品种
    # T/TF/TS/TL 已在上面的 prefix override 中优先识别。
    if normalized.endswith((".SH", ".SZ", ".CFE")):
        return _SECURITY_DAY_SESSIONS
    return None


@functools.lru_cache(maxsize=1024)
def get_trading_session_clock_ranges(code: str, *, allow_wind: bool = True):
    """返回代码的常规交易墙钟时段；未知元数据返回 ``None``。

    每一项均为 ``(start_time, end_time)``，端点属于交易时段。夜盘结束
    早于开始（例如 ``21:00 -> 01:00``）表示跨午夜。返回值与
    :func:`get_trading_bars_per_day` 复用同一份代码分类 / 品种 override。
    ``allow_wind=False`` 时仅使用代码前缀 / 交易所后缀可确定的本地规则，
    适合 GUI 主线程的即时参数联动，不会启动 Wind 或调用 ``wss``。调用方
    在 ``None`` 时应保持原有严格校验，不能把未知误当成休市。
    """
    if allow_wind:
        sessions = _get_trading_session_minutes(str(code))
    else:
        sessions = _get_local_trading_session_minutes(str(code))
    if sessions is None:
        return None
    return _TRADING_SESSION_CLOCK_RANGES_BY_MINUTES.get(sessions)


def _coerce_session_clock(value) -> _datetime.time:
    """把 ``time`` / datetime / ``HH:MM[:SS]`` 规范为无时区墙钟。"""
    if isinstance(value, _datetime.datetime):
        if value.tzinfo is not None:
            raise ValueError("target_time 不支持时区信息，请传入本地时间")
        return value.time()
    if isinstance(value, _datetime.time):
        if value.tzinfo is not None:
            raise ValueError("target_time 不支持时区信息，请传入本地时间")
        return value
    if isinstance(value, str):
        raw = value.strip()
        for fmt in ("%H:%M", "%H:%M:%S", "%H:%M:%S.%f"):
            try:
                return _datetime.datetime.strptime(raw, fmt).time()
            except ValueError:
                pass
    raise ValueError(
        f"非法 target_time={value!r}，需要 datetime.time 或 HH:MM[:SS]"
    )


def is_time_in_trading_session(
        code: str, target_time, *, allow_wind: bool = True) -> bool | None:
    """判断墙钟时刻是否处于代码的常规交易时段。

    已知交易时段时返回布尔值；Wind 元数据 / 本地分类未知时返回
    ``None``，以便上层继续执行严格的实际 Bar 校验。交易段端点按闭区间
    处理，所以 11:30、15:00 和夜盘收盘均属于有效目标时刻。
    """
    ranges = get_trading_session_clock_ranges(
        str(code), allow_wind=allow_wind
    )
    if ranges is None:
        return None
    clock = _coerce_session_clock(target_time)
    for start, end in ranges:
        if start <= end:
            if start <= clock <= end:
                return True
        elif clock >= start or clock <= end:
            return True
    return False


@functools.lru_cache(maxsize=512)
def get_trading_minutes_per_day(code: str) -> int | None:
    """返回总交易分钟数；未知元数据返回 None。

    该兼容 API 适合展示，不应直接用于推导粗粒度 bar 数；跨休市段的
    BarSize 请调用 :func:`get_trading_bars_per_day`。
    """
    sessions = _get_trading_session_minutes(str(code))
    return None if sessions is None else int(sum(sessions))


@functools.lru_cache(maxsize=2048)
def get_trading_bars_per_day(code: str, bar_size) -> int | None:
    """按连续交易段计算 Wind 每交易日的预期 bar 数。

    每段独立向上取整，以覆盖 10:15、11:30、15:00/15:15 和夜盘收盘
    等不足完整 BarSize 的尾 bar。未知元数据返回 None，调用方应依赖真实
    DatetimeIndex，而不是套用某个市场的硬编码下限。
    """
    try:
        bar_minutes = int(str(bar_size).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"bar_size 必须为正整数分钟数，收到 {bar_size!r}") from exc
    if bar_minutes <= 0:
        raise ValueError(f"bar_size 必须为正整数分钟数，收到 {bar_size!r}")

    sessions = _get_trading_session_minutes(str(code))
    if sessions is None:
        return None
    return int(sum(
        (segment + bar_minutes - 1) // bar_minutes
        for segment in sessions
    ))


def _ensure_wind():
    """启动 Wind 连接，若未安装则抛出提示

    直接 ``import WindPy`` 失败时先做一次**本机发现**再重试: 发布包由没有
    Wind 终端的 CI 构建, 包里不含 WindPy, 但用户机器上装了 Wind 终端就一定
    有一份 —— :mod:`pricing.windpy_locator` 负责把它接进当前进程。

    两次都失败才报错。冻结构建里的失败往往是 DLL 加载 (Windows) 或 dylib
    找不到 (macOS), 而非单纯"模块缺失", 所以把原始异常和扫描过的路径一并
    写进提示, 让用户能直接对症处理。
    """
    frozen = bool(getattr(sys, "frozen", False))
    try:
        from WindPy import w
    except Exception as e:
        from . import windpy_locator

        module = windpy_locator.bootstrap()
        if module is not None and hasattr(module, "w"):
            w = module.w
        else:
            located = windpy_locator.last_error()
            detail = f"{type(e).__name__}: {e}"
            if located is not None:
                detail += f"\n  本机 WindPy 加载失败: {type(located).__name__}: {located}"
            # 限 6 条: 这段文本最终进的是 GUI 弹窗, 后面还要接一整段
            # traceback, 全量候选(二三十条)会把对话框撑爆屏幕.
            hint = (
                "\n  已扫描以下位置但未找到可用的 WindPy:\n"
                f"{windpy_locator.describe_search(limit=6)}"
                "\n  若 Wind 终端装在别处, 可设环境变量 DELTALAB_WIND_DIR "
                "指向含 WindPy.py / WindPy.dll 的目录 (形如 "
                r"C:\Wind\Wind.NET.Client\WindNET\x64) 后重启本程序。"
            )
            if frozen:
                hint += (
                    "\n  [frozen build] 本发布包未内置 WindPy, 运行时依赖本机的"
                    " Wind 金融终端; 请确认终端已安装、已登录, 并在终端里执行过"
                    "「设置 Python 接口 / 修复」。"
                )
            raise ImportError(
                "未安装 WindPy，请安装 Wind 金融终端并配置 Python 插件。\n"
                "  pip install WindPy  或在 Wind 终端中设置 Python 接口。"
                f"{hint}\n  原始错误: {detail}"
            ) from e
    if not w.isconnected():
        result = w.start()
        if getattr(result, "ErrorCode", -1) != 0:
            raise ConnectionError(
                f"Wind 连接失败: ErrorCode={result.ErrorCode}, Data={result.Data}"
                + (
                    "\n  [frozen build] 请确认 Wind 终端已在本机启动并已登录."
                    if frozen else ""
                )
            )
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
    # 与分钟接口保持一致：无效范围在启动 / 连接 Wind 终端前暴露。
    _validate_intraday_range(start_date, end_date)
    w = _ensure_wind()
    price_adj = f"PriceAdj={adjust}" if adjust else ""
    data = w.wsd(code, "close", start_date, end_date, price_adj)

    if data.ErrorCode != 0:
        raise RuntimeError(f"Wind 数据获取失败 [{code}]: ErrorCode={data.ErrorCode}")

    series = pd.Series(data.Data[0], index=pd.to_datetime(data.Times), name=code)
    series = series.dropna()
    return series


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T].+)?$")
_INTRADAY_LOOKBACK_BUFFER_DAYS = 14


def _parse_intraday_boundary(value, *, name):
    """解析 Wind 日内查询边界，并保留“纯日期”这一调用语义。"""
    if value is None:
        raise ValueError(f"{name} 不能为空")

    is_date_only = False
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{name} 不能为空")
        if not _ISO_DATETIME_RE.fullmatch(raw):
            raise ValueError(
                f"非法 {name}={value!r}，需要 YYYY-MM-DD 或精确 datetime"
            )
        is_date_only = bool(_DATE_ONLY_RE.fullmatch(raw))
        value_to_parse = raw
    elif isinstance(value, _datetime.date) and not isinstance(
            value, _datetime.datetime):
        is_date_only = True
        value_to_parse = value
    else:
        if not isinstance(value, (pd.Timestamp, np.datetime64,
                                  _datetime.datetime)):
            raise ValueError(
                f"非法 {name}={value!r}，需要 YYYY-MM-DD 或精确 datetime"
            )
        value_to_parse = value

    try:
        timestamp = pd.Timestamp(value_to_parse)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"非法 {name}={value!r}，需要 YYYY-MM-DD 或精确 datetime"
        ) from exc

    if pd.isna(timestamp):
        raise ValueError(f"非法 {name}={value!r}")
    if timestamp.tzinfo is not None:
        raise ValueError(f"{name} 不支持时区信息，请传入本地时间")
    return timestamp, is_date_only


def _validate_intraday_range(start, end):
    """校验并解析日内查询范围；必须在连接 Wind 前调用。"""
    start_ts, start_is_date = _parse_intraday_boundary(
        start, name="start_date"
    )
    end_ts, end_is_date = _parse_intraday_boundary(end, name="end_date")
    if start_ts > end_ts:
        raise ValueError(
            f"start_date 不能晚于 end_date: {start!r} > {end!r}"
        )
    return start_ts, end_ts, start_is_date, end_is_date


def _format_wind_datetime(timestamp):
    """格式化为 Wind wsi 接受的本地 datetime 字符串。"""
    return pd.Timestamp(timestamp).isoformat(sep=" ")


def _normalize_intraday_datetime(dt_str, is_start):
    """
    Wind `wsi` 的日期参数格式通常是 "YYYY-MM-DD HH:MM:SS"。
    若用户只传了 "YYYY-MM-DD"，自动补全：
      - is_start=True  -> 09:30:00
      - is_start=False -> 15:00:00
    """
    timestamp, is_date_only = _parse_intraday_boundary(
        dt_str, name="start_date" if is_start else "end_date"
    )
    if is_date_only:
        clock = _datetime.time(9, 30) if is_start else _datetime.time(15, 0)
        timestamp = pd.Timestamp.combine(timestamp.date(), clock)
    return _format_wind_datetime(timestamp)


def _intraday_query_plan(start, end):
    """返回 Wind 实际抓取边界及是否需按交易日筛选。"""
    start_ts, end_ts, start_is_date, end_is_date = _validate_intraday_range(
        start, end
    )
    whole_trading_dates = start_is_date and end_is_date
    if whole_trading_dates:
        # 夜盘属于下一个交易日。向前多抓足够的日历日，可把周末/长假前
        # 的夜盘 opener 一并交给连续交易日分组，而不是从 00:00 的残段
        # 开始误切。查询结束覆盖当日全部时间，随后会排除属于下一交易日
        # 的晚间 session。
        query_start = (
            start_ts.normalize()
            - pd.Timedelta(days=_INTRADAY_LOOKBACK_BUFFER_DAYS)
        )
        query_end = end_ts.normalize() + pd.Timedelta(days=1, seconds=-1)
    else:
        # 只要调用者给出了具体时刻，就不扩边界。混合输入仍沿用原本对
        # 纯日期端补 09:30 / 15:00 的兼容行为。
        query_start = (
            pd.Timestamp.combine(start_ts.date(), _datetime.time(9, 30))
            if start_is_date else start_ts
        )
        query_end = (
            pd.Timestamp.combine(end_ts.date(), _datetime.time(15, 0))
            if end_is_date else end_ts
        )
        if query_start > query_end:
            raise ValueError(
                "补全日内时间后 start_date 不能晚于 end_date: "
                f"{_format_wind_datetime(query_start)} > "
                f"{_format_wind_datetime(query_end)}"
            )
    return query_start, query_end, whole_trading_dates, start_ts, end_ts


def _filter_complete_trading_date_groups(frame, start_date, end_date):
    """保留 trading-date 落在闭区间内、且已进入日盘的完整交易日组。"""
    if frame.empty:
        return frame

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.hasnans:
        raise ValueError("Wind intraday 返回了无效时间戳")
    if not index.is_monotonic_increasing:
        frame = frame.sort_index()
        index = pd.DatetimeIndex(frame.index)

    # 局部导入避免 wind_data <-> hedge_backtest 的模块级循环依赖。
    try:
        from .hedge_backtest import (
            _DAY_SESSION_START,
            _EVENING_SESSION_START,
            _trading_day_groups,
        )
    except ImportError:
        from hedge_backtest import (  # type: ignore
            _DAY_SESSION_START,
            _EVENING_SESSION_START,
            _trading_day_groups,
        )

    groups = _trading_day_groups(index)
    keep = np.zeros(len(index), dtype=bool)
    lower = pd.Timestamp(start_date).date()
    upper = pd.Timestamp(end_date).date()

    for group_id in np.unique(groups):
        positions = np.flatnonzero(groups == group_id)
        group_times = index[positions]
        daytime_dates = [
            timestamp.date()
            for timestamp in group_times
            if _DAY_SESSION_START
            <= timestamp.time().replace(tzinfo=None)
            < _EVENING_SESSION_START
        ]
        # 一个纯晚间的尾组尚未走到其 trading-date 日盘，无法证明组已
        # 完整，因此必须排除；这也会自然剔除 end_date 当晚属于下一交易
        # 日的 opener。
        if not daytime_dates:
            continue
        trading_date = max(daytime_dates)
        if lower <= trading_date <= upper:
            keep[positions] = True

    return frame.iloc[keep]


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
        起止均为纯日期时按交易日取完整 session（含属于该交易日的前夜
        夜盘）；带具体时刻时严格保留该边界。
    end_date : str
        结束日期，"YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"。
        起止均为纯日期时按 trading-date 闭区间筛选完整 session。
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
    (
        query_start,
        query_end,
        whole_trading_dates,
        requested_start,
        requested_end,
    ) = _intraday_query_plan(start_date, end_date)
    # 日期/范围错误必须在连接 Wind 前暴露，避免无效输入触发终端启动。
    w = _ensure_wind()
    start_dt = _format_wind_datetime(query_start)
    end_dt = _format_wind_datetime(query_end)

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
    df = df.dropna(how="all")
    if whole_trading_dates:
        df = _filter_complete_trading_date_groups(
            df, requested_start, requested_end
        )
    return df


_INTRADAY_CACHE_WARNED = False


def _warn_intraday_cache_unavailable(exc):
    """日内缓存不可用时提示一次，避免静默退化成每次重拉 Wind。"""
    global _INTRADAY_CACHE_WARNED
    if _INTRADAY_CACHE_WARNED:
        return
    _INTRADAY_CACHE_WARNED = True
    print(
        "[wind_data] 警告：日内行情缓存写入失败，本次运行的每次取数都会重新"
        "请求 Wind。1 分钟粒度下批量回测会慢一个数量级。\n"
        f"  原因: {type(exc).__name__}: {exc}\n"
        "  修复: pip install pyarrow（requirements.txt 已声明该依赖）"
    )


def _intraday_cache_path(code, start, end, bar_size, adjust="F"):
    """intraday 缓存文件路径

    key 必须包含 adjust：F/B/'' 三种复权口径不能混用同一缓存，否则后续
    读回来的序列复权方式与调用方预期不一致。adjust='' 的无复权在文件名
    里固化为 'NA'，避免空串被 OS 文件系统解释出奇怪结果。起止边界会
    保留完整时间精度，并区分纯日期的“完整交易日”语义和精确午夜时刻。
    """
    start_ts, end_ts, start_is_date, end_is_date = _validate_intraday_range(
        start, end
    )
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe_code = code.replace("/", "_").replace("\\", "_")

    def _boundary_token(timestamp, is_date_only):
        if is_date_only:
            return f"D{timestamp:%Y%m%d}"
        # 微秒与剩余纳秒均进入 key，防止不同精确边界共用缓存；T 前缀也
        # 让“整交易日 2024-01-01”区别于“精确到当日 00:00:00”。
        return (
            f"T{timestamp:%Y%m%d_%H%M%S_}{timestamp.microsecond:06d}_"
            f"{timestamp.nanosecond:03d}"
        )

    safe_start = _boundary_token(start_ts, start_is_date)
    safe_end = _boundary_token(end_ts, end_is_date)
    safe_adj = str(adjust).strip() or "NA"
    fname = (
        f"{safe_code}_{safe_start}_{safe_end}_intraday_v2_"
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
        "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"。起止均为纯日期时
        返回 trading-date 闭区间内的完整 session（包括前夜夜盘）。
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
    except Exception as exc:
        # 缓存失败不影响主流程，但必须让调用方知道：没有缓存时每次取数
        # 都会重新请求 Wind，1 分钟粒度的批量回测会慢一个数量级。
        _warn_intraday_cache_unavailable(exc)
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
    except ImportError:
        # WindPy 不可用时返回占位，调用方需自行处理
        mult = 1.0
    else:
        data = w.wsd(code, "contractmultiplier")
        if data.ErrorCode != 0:
            raise RuntimeError(
                f"Wind contractmultiplier 查询失败 [{code}]: "
                f"ErrorCode={data.ErrorCode}"
            )
        try:
            mult = float(data.Data[0][0])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Wind contractmultiplier 返回数据异常 [{code}]: "
                f"{getattr(data, 'Data', None)!r}"
            ) from exc

    return {"multiplier": mult, "is_future": True}
