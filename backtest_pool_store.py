"""回测结果池的落盘存取（纯逻辑，不依赖 tkinter）。

结果池过去只活在会话里，关掉程序就没了。改成默认持久化后，两件事必须先
定死，否则「保存」比「不保存」更糟：

**一、往返必须精确。** 快照里有三样 JSON 原生表达不了的东西，任何一样丢
了都不报错、只是让结论变差：

- 四个签名 key（``market_key`` / ``contract_key`` / ``economics_key`` /
  ``path_key``）是**嵌套 tuple**。对比页靠 ``isinstance(item, tuple)`` 判
  嵌套来算「本次对比的变量差在哪一项」，JSON 往返成 list 后嵌套不再展开，
  差异卡会从「只有波动率不同」退化成说不清。
- ``summary_row`` 里的 ``daily_net_pnl_rms`` / ``avg_daily_tc`` 在空交易日
  时是 ``inf``，``position`` 缺失时是 ``NaN``。指标表对 inf 和缺失的显示与
  排序都不同，不能混成一个。
- ``daily_frame`` 是 DataFrame，索引是交易日序号而不是日期。

所以这里不复用 ``history_store._jsonable``——它把 inf/NaN 一律写成 null，
把 tuple 一律写成 list，那是给「排名表」用的有损口径。本模块用带标签的编
码，逐类型精确还原。

**二、存输入不存输出。** 与 ``history_store`` 同一笔账：一次回测的 bar 级
结果数组平均 971 KB，而重跑它所需的输入——价格序列——一年 1 分钟 gzip 后
约 206 KB。所以「加载明细」存的是行情切片加构造参数，用时重跑（约 1 秒），
而不是把 Greeks 和逐 bar 持仓写进盘里。

**三、文件名不带业务信息。** 与 ``history_store.default_filename`` 相反，
这里只写时间戳和序号。结果池是自动保存的，用户不会逐个命名，而 ``wind_code``
是持仓标的——把它写进文件名等于把标的暴露在文件列表里。展示名在包内。
"""
from __future__ import annotations

import datetime
import glob
import gzip
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

# 包格式版本。快照的字段一直在加（origin、form_state、sequence、replay 都是
# 后补的），旧包用新代码渲染会静默给出错误口径。载入时硬校验，版本不符宁可
# 拒绝——这正是当初不落盘的理由，落盘后它变成前置条件而不是可选项。
POOL_SCHEMA_VERSION = 1

# 最多保留多少条。超出后按序号淘汰最旧的，与策略优选同约定。
MAX_RESULTS = 20

_SUFFIX = ".json.gz"


def pool_dir() -> str:
    """结果池目录。与 ``history_store.results_dir`` 同款约定，但另开一个。

    两者 schema 完全不同、生命周期也不同（这边自动保存、那边显式保存），
    混在一个目录里会让双方的列目录都要先按 schema 过滤一遍。

    开发态放仓库内 ``data/backtest_pool``（记得 .gitignore），打包冻结后放
    ``~/.deltalab/pool``——``.app`` 包与安装目录可能只读。
    """
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), ".deltalab", "pool")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "backtest_pool",
    )


class PoolSchemaError(ValueError):
    """包的 schema 版本与当前代码不符。"""


# ============================================================
#  带标签的编解码：tuple / 非有限浮点 / 时间戳 / DataFrame 精确往返
# ============================================================

# 标签键。dict 一律编码成「键值对列表」而不是原生对象，这样既保序，也不会
# 和标签键撞名——快照里的 key 来自用户输入（CSV 列名、结果名），不能假设
# 它们不以 "$" 开头。
_TAG = "$"
_DROP = object()


def encode(value):
    """把快照里的值编码成可 JSON 序列化的结构。

    不可序列化的对象返回 ``_DROP``，由容器负责丢弃——``gui_state`` 里嵌着
    ``cfg["build"]`` 这类回调，只在顶层判 callable 会让它一路走到
    ``json.dumps`` 才炸。
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if callable(value):
        return _DROP
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # inf / NaN 是真实取值而不是缺失，必须原样带回来。
        if math.isnan(value):
            return {_TAG: "f", "v": "nan"}
        if math.isinf(value):
            return {_TAG: "f", "v": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.ndarray):
        # dtype 必须记下来：解码侧曾经一律按 float 还原，于是字符串 / object
        # 数组**写得进读不出**——包在盘上，载入时 np.asarray(['a'], float) 抛
        # ValueError，整条快照被判「字段无法还原」而丢掉。
        # ``_DROP`` 也要在这里过滤：其余三个容器都过滤了，只有这一条把哨兵
        # 原样塞进列表，一路带到 json.dumps 才炸——那正是 _DROP 要避免的。
        return {
            _TAG: "arr",
            "dtype": value.dtype.str,
            "v": [item for item in (encode(v) for v in value.tolist())
                  if item is not _DROP],
        }
    if isinstance(value, pd.Series):
        # DataFrame 早就逐类型往返了，Series / Index 却没有分支，会一路落到
        # 末尾的 _DROP 被容器无声丢掉——同一批 pandas 容器，两个存储层给了
        # 不同答案（history_bar_cache.store_by_key 是逐类型显式处理的）。
        return {
            _TAG: "ser",
            "index": [encode(item) for item in value.index.tolist()],
            "index_name": (None if value.index.name is None
                           else str(value.index.name)),
            "name": None if value.name is None else str(value.name),
            "v": [encode(item) for item in value.tolist()],
        }
    if isinstance(value, pd.Index):
        # DatetimeIndex 是 Index 的子类，靠 kind 区分而不是靠分支顺序。
        return {
            _TAG: "idx",
            "kind": ("datetimeindex" if isinstance(value, pd.DatetimeIndex)
                     else "index"),
            "name": None if value.name is None else str(value.name),
            "v": [encode(item) for item in value.tolist()],
        }
    if isinstance(value, tuple):
        return {
            _TAG: "tup",
            "v": [item for item in (encode(v) for v in value)
                  if item is not _DROP],
        }
    if isinstance(value, list):
        return [item for item in (encode(v) for v in value)
                if item is not _DROP]
    if isinstance(value, pd.DataFrame):
        return encode_frame(value)
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            converted = encode(item)
            if converted is _DROP:
                continue
            # 键也走 encode：``str(key)`` 会让 {1: …} 往返成 {"1": …}，而
            # history_bar_cache.key_for_recipe 直接摘这份配方——同一条快照在
            # 「刚保留」与「重开程序后」算出两个 key，加载明细每次都得重跑。
            # 字符串键 encode 后仍是它自己，包格式对旧包保持一致。
            encoded_key = encode(key)
            if encoded_key is _DROP:
                encoded_key = str(key)
            pairs.append([encoded_key, converted])
        return {_TAG: "map", "v": pairs}
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        if pd.isna(value):
            return None
        return {_TAG: "ts", "v": pd.Timestamp(value).isoformat()}
    if isinstance(value, datetime.date):
        return {_TAG: "date", "v": value.isoformat()}
    # 到这里是 JSON 不认识的对象。丢掉而不是让整次保存失败：快照的价值在
    # 那几张表和冻结参数上，不在某个附带对象上。
    return _DROP


def decode(value):
    """``encode`` 的逆。"""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TAG)
    if tag is None:
        # 没有标签的 dict 只可能来自手工写的包；按原样还原，别静默吞掉。
        return {k: decode(v) for k, v in value.items()}
    if tag == "f":
        raw = str(value.get("v"))
        return float("nan") if raw == "nan" else float(raw)
    if tag == "tup":
        return tuple(decode(item) for item in value.get("v", ()))
    if tag == "arr":
        items = [decode(item) for item in value.get("v", ())]
        # 包里记的 dtype 优先；旧包没这个字段，按老行为先试 float，装不下
        # 再退到 object——比原来直接抛 ValueError 多救回一批旧包。
        for dtype in ([value.get("dtype")] if value.get("dtype") else []) + [
                float, object]:
            try:
                return np.asarray(items, dtype=dtype)
            except (TypeError, ValueError):
                continue
        return np.asarray(items, dtype=object)
    if tag == "ser":
        series = pd.Series(
            [decode(item) for item in value.get("v", ())],
            name=value.get("name"))
        index = [decode(item) for item in value.get("index", ())]
        if len(index) == len(series):
            series.index = pd.Index(index, name=value.get("index_name"))
        return series
    if tag == "idx":
        items = [decode(item) for item in value.get("v", ())]
        if value.get("kind") == "datetimeindex":
            return pd.DatetimeIndex(items, name=value.get("name"))
        return pd.Index(items, name=value.get("name"))
    if tag == "map":
        return {decode(k): decode(v) for k, v in value.get("v", ())}
    if tag == "ts":
        return pd.Timestamp(value.get("v"))
    if tag == "date":
        return datetime.date.fromisoformat(str(value.get("v")))
    if tag == "df":
        return decode_frame(value)
    return value


def encode_frame(frame):
    """DataFrame 连同索引一起编码。

    索引不能丢：``daily_frame`` 的索引是交易日序号（``index.name`` 为
    ``trade_day``），曲线的横轴就是它。
    """
    if frame is None:
        return None
    return {
        _TAG: "df",
        "columns": [str(name) for name in frame.columns],
        "index": [encode(item) for item in frame.index.tolist()],
        "index_name": (None if frame.index.name is None
                       else str(frame.index.name)),
        "data": {
            str(name): [encode(item) for item in frame[name].tolist()]
            for name in frame.columns
        },
    }


def decode_frame(payload):
    if not payload:
        return None
    columns = [str(name) for name in payload.get("columns", ())]
    data = {
        name: [decode(item) for item in payload.get("data", {}).get(name, ())]
        for name in columns
    }
    index = [decode(item) for item in payload.get("index", ())]
    frame = pd.DataFrame(data, columns=columns or None)
    if len(index) == len(frame):
        frame.index = pd.Index(index, name=payload.get("index_name"))
    return frame


# ============================================================
#  文件读写
# ============================================================

def default_filename(sequence, saved_at=None):
    """``pool-<时间戳>-<序号>.json.gz``。

    刻意不含结果名与标的代码：结果池是自动保存的，文件名会被当成目录列表
    直接看到，而 ``wind_code`` 是持仓标的。
    """
    stamp = pd.Timestamp(saved_at or datetime.datetime.now())
    return (f"pool-{stamp.strftime('%Y%m%d-%H%M%S')}-"
            f"{int(sequence):04d}{_SUFFIX}")


def write_snapshot(payload, *, directory=None, path=None, enforce=True):
    """写入一条快照，返回完整路径。

    ``path`` 显式给出时原地覆盖（重命名走这条），否则按序号+时间戳新建。
    先写 ``.part`` 再 ``os.replace``：写一半崩掉只会留下一个 ``.part``，而
    坏包在列表里看起来和好包一样。
    """
    directory = directory or pool_dir()
    os.makedirs(directory, exist_ok=True)
    if path is None:
        filename = default_filename(
            payload.get("sequence", 0), payload.get("saved_at"))
        path = os.path.join(directory, filename)
        # 同一秒里连存两条（换个参数立刻再跑一次很常见）不能互相覆盖。
        # 序号已经进文件名，撞名说明序号也撞了——那是调用方的 bug，这里
        # 仍然让它落到另一个名字上，不静默吃掉数据。
        if os.path.exists(path):
            stem = os.path.basename(path)[:-len(_SUFFIX)]
            for serial in range(2, 1000):
                candidate = os.path.join(
                    directory, f"{stem}-{serial}{_SUFFIX}")
                if not os.path.exists(candidate):
                    path = candidate
                    break
            else:
                raise OSError(f"同名快照过多，无法保存: {stem}")
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        handle.write(text)
    os.replace(tmp, path)
    if enforce:
        enforce_limit(directory=directory)
    return path


def _sweep_partials(directory):
    """清掉进程崩溃残留的 ``.part``；它们不在 glob 里，没有别的清理点。"""
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if not name.endswith(".part"):
            continue
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            continue


def read_snapshot(path, *, allow_other_version=False):
    """读回一条快照。版本不符默认拒绝。"""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    version = payload.get("schema_version")
    if version != POOL_SCHEMA_VERSION and not allow_other_version:
        raise PoolSchemaError(
            f"结果包版本 {version} 与当前程序（{POOL_SCHEMA_VERSION}）不一致")
    payload["_path"] = path
    return payload


def read_all(directory=None):
    """按序号升序读回全部快照。

    返回 ``(payloads, skipped)``。``skipped`` 是 ``(文件名, 原因)`` 列表：
    坏包和版本不符的包都不能静默跳过——用户会以为自己丢了结果，而实际是
    程序不肯读。调用方负责把它报到界面上。
    """
    directory = directory or pool_dir()
    if not os.path.isdir(directory):
        return [], []
    _sweep_partials(directory)
    payloads, skipped = [], []
    for path in sorted(glob.glob(os.path.join(directory, "*" + _SUFFIX))):
        try:
            payloads.append(read_snapshot(path))
        except PoolSchemaError as exc:
            skipped.append((os.path.basename(path), str(exc)))
        # ValueError 覆盖 JSONDecodeError 与 UnicodeDecodeError；
        # BadGzipFile 是 OSError 子类。漏掉会让整次载入崩在一个坏包上。
        except (OSError, EOFError, ValueError) as exc:
            skipped.append((os.path.basename(path), f"文件损坏：{exc}"))
    payloads.sort(key=lambda item: (
        int(item.get("sequence", 0) or 0), str(item.get("saved_at", ""))))
    return payloads, skipped


_VIEW_STATE_NAME = "view_state.json"


def read_view_state(directory=None):
    """读回「上次显示了哪几条」。读不到就返回空集合。

    这是视图状态不是数据，所以单独一个小文件、坏了也只是回到「全部隐藏」，
    不影响快照本身。
    """
    directory = directory or pool_dir()
    path = os.path.join(directory, _VIEW_STATE_NAME)
    # 兜底范围要覆盖到「合法 JSON 但不是预期形状」：文件内容是数组时
    # ``payload.get`` 抛 AttributeError，``selected`` 是个数字时集合推导抛
    # TypeError，两者都不在 (OSError, ValueError) 里。而调用方
    # （gui_app._load_saved_pool）这一句在保护 read_all 的 try 之外，漏出去
    # 会让整池结果都载不进来——本函数承诺的是「最多回到全部隐藏」。
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return set()
        return {str(item) for item in payload.get("selected", ())}
    except (OSError, ValueError, TypeError, AttributeError):
        return set()


def write_view_state(selected, directory=None):
    """记下当前显示了哪几条；失败静默——它只是个便利，不值得打断操作。"""
    directory = directory or pool_dir()
    try:
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, _VIEW_STATE_NAME + ".part")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"selected": sorted(str(item) for item in selected)},
                      handle, ensure_ascii=False)
        os.replace(tmp, os.path.join(directory, _VIEW_STATE_NAME))
    except OSError:
        pass


def delete_snapshot(path):
    """删除一条快照；不存在时静默返回 False。"""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# ``default_filename`` 的形状，外加 ``write_snapshot`` 撞名时补的那个序号。
_NAME_PATTERN = re.compile(
    r"^pool-(\d{8}-\d{6})-(\d{4})(?:-(\d+))?" + re.escape(_SUFFIX) + r"$")


def _order_by_filename(paths):
    """按文件名推出与 ``read_all`` 相同的先后顺序；认不出就返回 None。

    文件名是 ``default_filename`` 拿 payload 里的 ``sequence`` 和 ``saved_at``
    拼出来的，所以对本程序写下的包，它与包内顺序**由构造保证**一致，排序键
    也照 read_all 取（先序号后时刻）。手工改过名的文件匹配不上，那时退回去
    逐个读包——删错一份是不可逆的，宁可慢。
    """
    ordered = []
    for path in paths:
        matched = _NAME_PATTERN.match(os.path.basename(path))
        if matched is None:
            return None
        ordered.append((
            int(matched.group(2)),          # sequence
            matched.group(1),               # 保存时刻
            int(matched.group(3) or 0),     # 撞名补的序号
            path,
        ))
    ordered.sort()
    return [item[-1] for item in ordered]


def enforce_limit(max_results=MAX_RESULTS, *, directory=None):
    """只保留最近 ``max_results`` 条，返回被淘汰的路径列表。

    删除不可逆，调用方应当把返回值报给用户——按规则删的也要说一声。
    """
    directory = directory or pool_dir()
    if max_results is None or max_results <= 0:
        return []
    # 一个包都不读就能定序。``write_snapshot`` 每存一条就调一次这里，而
    # read_all 会把每个包（各带一整条价格序列）gunzip + json.load 一遍：实测
    # 20 份一年 1 分钟序列的包，光这一趟就 0.14 s，每次「保留结果」都白付，
    # 而结果池稳态就是满的，这笔钱躲不掉。文件数先判要不要动手，真要淘汰时
    # 再从文件名取顺序。
    try:
        names = glob.glob(os.path.join(directory, "*" + _SUFFIX))
    except OSError:
        return []
    if len(names) <= max_results:
        return []
    evicted = []
    ordered = _order_by_filename(names)
    if ordered is not None:
        # 这条路上坏包也算数、也会被淘汰——上限本来就是给磁盘占用设的，而
        # 用户看到的那句话是「最多 20 条」。
        for path in ordered[:len(ordered) - max_results]:    # 升序，旧的在前
            if delete_snapshot(path):
                evicted.append(path)
        return evicted
    # 有名字认不出来，退回权威顺序。注意这条路上坏包与版本不符的包不进
    # payloads，因此不会被淘汰——目录可能长期超出上限，代价是慢，不是删错。
    payloads, _skipped = read_all(directory)
    if len(payloads) <= max_results:
        return []
    for payload in payloads[:len(payloads) - max_results]:   # 升序，旧的在前
        path = payload.get("_path")
        if path and delete_snapshot(path):
            evicted.append(path)
    return evicted
