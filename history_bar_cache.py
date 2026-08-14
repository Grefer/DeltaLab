"""策略优选分段的 bar 级结果磁盘缓存（纯逻辑，不依赖 tkinter）。

**为什么要有它。** 一轮近年全选是 161 次回测，bar 级数组合计约 153 MB，
全留在内存代价太大，所以历史择优跑完就释放、只保留「重放配方」，想看某段
明细时现跑一次（实测 620 ms）。把这些结果压缩落盘后，同一段再看只要 3 ms
——快 200 倍，而整轮压缩后约 38 MB（数组重复度高，1052 KB 压到 239 KB）。

**放在缓存目录，与结果包分开。** 这里的东西丢了能重跑，是真正的缓存；结果
包不可重建（Wind 区间随分析截至日移动），所以在别处。

**按内容哈希做 key。** 不用「第几次运行」这类外部标识：同一份输入无论来自
刚跑完的结果还是载入的结果包，都命中同一条缓存；输入变了则必然 miss，不会
读到过期数据。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# 磁盘占用不在这里设上限——按产品决定，改由「最多保留 20 份优选记录」
# 间接约束（见 history_store.MAX_RESULTS）。``prune`` 与 ``clear`` 仍然
# 可用，只是不再自动调用；需要时一行就能接回去。
DEFAULT_MAX_BYTES = 2 * 1024 ** 3
_SUFFIX = ".npz"
_META_SUFFIX = ".json"


def cache_dir() -> str:
    """与 Wind intraday 缓存同一个父目录，但另开一层。"""
    if getattr(sys, "frozen", False):
        base = os.path.join(os.path.expanduser("~"), ".deltalab", "cache")
    else:
        base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "cache")
    return os.path.join(base, "history_bars")


# 摘要不了的值用它标记。**默认方向必须是「放弃缓存」而不是「悄悄丢
# 掉」**：漏掉一个决定行为的属性，就会让两次不同的运行共用一条缓存，下钻
# 页显示的是别次的逐 bar 明细，而且不报错。代价只是偶尔多跑 620 ms。
_GIVE_UP = object()

# 只有确实不影响回测结果的属性才允许跳过。名单要短，且每一项都要能说出
# 为什么无关。
_IGNORED_ATTRS = frozenset({
    "name",            # 策略类型名，已由类名覆盖
})
# ``optiontype`` 曾经在这张名单里，理由写的是「期权的展示标签」——那只对
# Option_Vanilla 成立，它是唯一从不读 self.optiontype 的类。实际上它是定价
# 方法的**分派键**：Option_AB/DE/SNB 走 ``getattr(self, self.optiontype)()``，
# Option_AS 走 ``match self.optiontype``。排除它的后果实测是 Option_DE 的
# 13 个子类型算出同一个 key，互相读到对方的 bar 级数组且不报错。


def _all_numbers(seq):
    """序列是否全是数字（bool 不算——它是 int 的子类但语义不同）。"""
    if not len(seq):
        return False
    return all(
        isinstance(item, (int, float, np.integer, np.floating))
        and not isinstance(item, (bool, np.bool_))
        for item in seq)


def _digest_numbers(seq):
    """把数字序列按 float64 摘要；装不进 float64 就放弃缓存。"""
    try:
        values = np.asarray(seq, dtype=np.float64)
    except (TypeError, ValueError):
        return _GIVE_UP
    return f"n{values.shape}{hashlib.sha1(values.tobytes()).hexdigest()}"


def _digest_value(value, depth=0):
    """把任意值压成可比较的字符串；压不了就返回 ``_GIVE_UP``。"""
    if depth > 6:
        return _GIVE_UP
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return f"b{bool(value)}"
    if isinstance(value, (int, np.integer)):
        return f"i{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "fnan" if not np.isfinite(number) else f"f{number:.17g}"
    if isinstance(value, str):
        return f"s{value}"
    if isinstance(value, (datetime.time, datetime.date, datetime.datetime)):
        return f"t{value.isoformat()}"
    if isinstance(value, pd.Timestamp):
        return "T" + ("NaT" if pd.isna(value) else value.isoformat())
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return _digest_value(value.tolist(), depth + 1)
        if value.dtype.kind in "fiu":
            return _digest_numbers(value)
        return f"a{value.shape}{hashlib.sha1(value.tobytes()).hexdigest()}"
    if isinstance(value, (list, tuple)) and _all_numbers(value):
        # 数字序列一律按 float64 数组摘要，容器类型不参与。包里存的预热对数
        # 收益是 list，实跑时是 ndarray，同一串数走两条分支就会算出不同的
        # key——载入结果包后 realized σ 候选因此**永远**命中不了缓存，每次
        # 下钻都在重算。装在 list 还是 ndarray 里不影响回测结果，不该换 key。
        return _digest_numbers(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(
            value, (set, frozenset)) else value
        parts = [_digest_value(item, depth + 1) for item in items]
        if any(part is _GIVE_UP for part in parts):
            return _GIVE_UP
        return "[" + ",".join(parts) + "]"
    if isinstance(value, dict):
        parts = []
        for key in sorted(value, key=repr):
            sub = _digest_value(value[key], depth + 1)
            if sub is _GIVE_UP:
                return _GIVE_UP
            parts.append(f"{key!r}:{sub}")
        return "{" + ",".join(parts) + "}"
    return _GIVE_UP


def _digest_object(obj):
    """枚举对象的全部公开非可调用属性；有一项摘要不了就整体放弃。

    此前这里按类型白名单挑（只收 int/float/str，还显式排除 bool），认不出
    的静默丢弃。实测漏掉的是 ``sr``（逐期敲出线）、``ko_observ``（观察
    日）、``margin_call``，以及固定时刻策略的整张时刻表——两个完全不同的
    配置会算出同一个 key。现在改成全枚举 + 遇到不认识的就放弃缓存。
    """
    parts = [type(obj).__name__]
    for name in sorted(dir(obj)):
        if name.startswith("_") or name in _IGNORED_ATTRS:
            continue
        try:
            value = getattr(obj, name)
        except Exception:                              # noqa: BLE001
            return None
        if callable(value):
            continue
        digested = _digest_value(value)
        if digested is _GIVE_UP:
            return None
        parts.append(f"{name}={digested}")
    return "|".join(parts)


def _digest_path(path):
    """行情切片身份：长度、首尾时间戳，加数值本身的哈希。

    只取首尾会漏掉「区间相同但价格被改过」的情况——那正是复权口径变化会
    造成的，而它必须导致 miss。
    """
    values = np.asarray(path, dtype=float)
    index = getattr(path, "index", None)
    head = str(index[0]) if index is not None and len(index) else ""
    tail = str(index[-1]) if index is not None and len(index) else ""
    body = hashlib.sha1(values.tobytes()).hexdigest()
    return f"{len(values)}|{head}|{tail}|{body}"


def key_for(spec, strategy_name):
    """把一次重放的全部输入压成一个 key；无法完整刻画时返回 ``None``。

    返回 None 表示这次不缓存（既不读也不写）。覆盖：期权、行情切片、评估
    天数、bar 粒度、策略身份、回测参数、预热参数。任何一项变化都会换
    key，任何一项摘要不了就整体放弃——宁可白跑 620 ms，也不能读到不匹配
    的结果。
    """
    strategy = spec.strategies.get(str(strategy_name))
    if strategy is None:
        return None
    option_digest = _digest_object(spec.option)
    strategy_digest = _digest_object(strategy)
    if option_digest is None or strategy_digest is None:
        return None
    kwargs_digest = _digest_value(dict(spec.backtest_kwargs))
    warmup_digest = _digest_value(
        dict(spec.warmup_kwargs.get(strategy_name, {}) or {}))
    if kwargs_digest is _GIVE_UP or warmup_digest is _GIVE_UP:
        return None
    material = "\n".join([
        option_digest,
        _digest_path(spec.external_path),
        f"eval={int(spec.evaluation_days)}",
        f"spd={int(spec.steps_per_day)}",
        strategy_digest,
        kwargs_digest,
        warmup_digest,
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


# 快照配方里唯一不影响回测数字的键。它是展示元数据——重放时由配方原样塞
# 回 ``bt._gui_meta``，不参与任何计算。算进 key 只会凭空制造 miss。名单要
# 短，且每一项都要能说出为什么无关（与 ``_IGNORED_ATTRS`` 同一条纪律）。
_RECIPE_IGNORED_KEYS = frozenset({"gui_meta"})


def key_for_recipe(recipe):
    """把结果池快照的重放配方压成 key；刻画不全时返回 ``None``。

    与 ``key_for`` 同样保守失败：宁可白跑一次，也不能读到不匹配的结果。
    配方本身已经是可序列化的纯数据（它要进快照包），所以直接摘它，不必像
    ``HistoryReplaySpec`` 那样枚举对象属性。

    价格与时间戳单独哈希：一条序列几千个点，走 ``_digest_value`` 会先拼出
    一个几百 KB 的中间字符串，白费内存。
    """
    if not recipe:
        return None
    try:
        body = dict(recipe)
    except (TypeError, ValueError):
        return None
    prices = body.pop("prices", None)
    index = body.pop("index", None)
    for name in _RECIPE_IGNORED_KEYS:
        body.pop(name, None)
    try:
        values = np.asarray(list(prices or ()), dtype=float)
    except (TypeError, ValueError):
        return None
    if values.size < 2:
        return None
    if index is None:
        stamps = "noindex"
    else:
        try:
            joined = "\u0000".join(str(item) for item in index)
        except TypeError:
            return None
        stamps = (f"{len(index)}|"
                  f"{hashlib.sha1(joined.encode('utf-8')).hexdigest()}")
    rest = _digest_value(body)
    if rest is _GIVE_UP:
        return None
    material = "\n".join([
        "recipe",
        f"{values.size}|{hashlib.sha1(values.tobytes()).hexdigest()}",
        stamps,
        rest,
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def store_recipe(recipe, result, *, directory=None):
    """按快照配方落盘一次回测的 bar 级结果。"""
    return store_by_key(key_for_recipe(recipe), result, directory=directory)


def load_recipe(recipe, *, directory=None):
    """按快照配方读回 bar 级结果；未命中返回 None（调用方重跑即可）。"""
    return load_by_key(key_for_recipe(recipe), directory=directory)


def _paths(key, directory=None):
    directory = directory or cache_dir()
    stem = os.path.join(directory, key)
    return stem + _SUFFIX, stem + _META_SUFFIX


def store(spec, strategy_name, result, *, directory=None):
    """把一段回测结果写进缓存，返回路径；写失败时返回 None。

    缓存写失败绝不能让主流程出错——它只是省时间的东西，没有它一切照常。
    """
    return store_by_key(
        key_for(spec, strategy_name), result, directory=directory)


def store_by_key(key, result, *, directory=None):
    """按已算好的 key 落盘。``key`` 为 None 表示这次不缓存，直接返回 None。"""
    directory = directory or cache_dir()
    if key is None:
        return None
    npz_path, meta_path = _paths(key, directory)
    # npz 只认 ndarray，pandas 的容器类型进去就化成裸数组了。把原始容器类
    # 型单独记下来，load 才能还原成**与真跑逐位且同型**的结果——本模块开
    # 头就承诺"命中与重跑一致"，而 `timestamps` 真跑是 DatetimeIndex、命中
    # 回来却是 ndarray，下游那些 isinstance(..., pd.DatetimeIndex) 的分支
    # 会因此走向另一条路（例如 _history_trading_day_groups 的日内分组）。
    arrays, scalars, containers = {}, {}, {}
    for name, value in dict(result).items():
        if isinstance(value, np.ndarray):
            arrays[str(name)] = value
        elif isinstance(value, pd.Series):
            arrays[str(name)] = value.to_numpy()
        # DatetimeIndex 是 Index 的子类，必须先判它。
        elif isinstance(value, pd.DatetimeIndex):
            arrays[str(name)] = np.asarray(value)
            containers[str(name)] = "datetimeindex"
        elif isinstance(value, pd.Index):
            arrays[str(name)] = np.asarray(value)
            containers[str(name)] = "index"
        else:
            scalars[str(name)] = value
    try:
        os.makedirs(directory, exist_ok=True)
        tmp = npz_path + ".part"
        np.savez_compressed(tmp, **arrays)
        # np.savez 会补 .npz 后缀，这里统一回目标名再原子替换。
        produced = tmp if os.path.exists(tmp) else tmp + ".npz"
        os.replace(produced, npz_path)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"scalars": _jsonable_scalars(scalars),
                 "containers": containers,
                 "written_at": time.time()},
                handle, ensure_ascii=False)
    except Exception:                                  # noqa: BLE001
        return None
    return npz_path


def _jsonable_scalars(scalars):
    out = {}
    for key, value in scalars.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            number = float(value)
            out[key] = None if not np.isfinite(number) else number
        elif isinstance(value, (np.bool_,)):
            out[key] = bool(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [
                v for v in (
                    _jsonable_scalars({"x": item}).get("x") for item in value)
            ]
        elif isinstance(value, dict):
            out[key] = _jsonable_scalars(value)
        # 其它类型（策略对象、时间戳序列…）不进 meta：它们要么在 arrays
        # 里，要么本来就能从配方重建。
    return out


def load(spec, strategy_name, *, directory=None):
    """读回一段结果；未命中或文件损坏时返回 None（调用方重跑即可）。"""
    return load_by_key(key_for(spec, strategy_name), directory=directory)


def load_by_key(key, *, directory=None):
    """按已算好的 key 读回；未命中、损坏或 key 为 None 时返回 None。"""
    if key is None:
        return None
    npz_path, meta_path = _paths(key, directory)
    if not os.path.isfile(npz_path):
        return None
    try:
        result = {}
        with np.load(npz_path, allow_pickle=False) as bundle:
            for name in bundle.files:
                result[name] = bundle[name]
        containers = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            result.update(dict(meta.get("scalars", {})))
            containers = dict(meta.get("containers", {}))
        # 还原 pandas 容器类型。缺 containers 的是本字段之前写下的老条目，
        # 保持 ndarray 与旧行为一致（摘要改过之后它们本就不该再被命中）。
        for name, kind in containers.items():
            if name not in result:
                continue
            if kind == "datetimeindex":
                result[name] = pd.DatetimeIndex(result[name])
            elif kind == "index":
                result[name] = pd.Index(result[name])
        # 命中即刷新 mtime，LRU 才有意义。
        os.utime(npz_path, None)
    except Exception:                                  # noqa: BLE001
        return None
    return result


def usage_bytes(directory=None):
    directory = directory or cache_dir()
    if not os.path.isdir(directory):
        return 0
    total = 0
    for name in os.listdir(directory):
        try:
            total += os.path.getsize(os.path.join(directory, name))
        except OSError:
            continue
    return total


def prune(max_bytes=DEFAULT_MAX_BYTES, *, directory=None):
    """按最近使用时间裁到容量上限内，返回删掉的字节数。"""
    directory = directory or cache_dir()
    if not os.path.isdir(directory):
        return 0
    entries = []
    for name in os.listdir(directory):
        if not name.endswith(_SUFFIX):
            continue
        path = os.path.join(directory, name)
        try:
            entries.append((os.path.getmtime(path), path,
                            os.path.getsize(path)))
        except OSError:
            continue
    total = usage_bytes(directory)
    if total <= max_bytes:
        return 0
    freed = 0
    for _mtime, path, size in sorted(entries):
        if total - freed <= max_bytes:
            break
        meta = path[:-len(_SUFFIX)] + _META_SUFFIX
        for target in (path, meta):
            try:
                freed += os.path.getsize(target)
                os.remove(target)
            except OSError:
                continue
    return freed


def clear(directory=None):
    """清空缓存；返回删掉的文件数。丢了只是下次要重跑，不丢数据。"""
    directory = directory or cache_dir()
    if not os.path.isdir(directory):
        return 0
    removed = 0
    for name in os.listdir(directory):
        if name.endswith((_SUFFIX, _META_SUFFIX, ".part")):
            try:
                os.remove(os.path.join(directory, name))
                removed += 1
            except OSError:
                continue
    return removed
