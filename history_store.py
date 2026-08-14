"""策略优选结果的落盘存取（纯逻辑，不依赖 tkinter）。

一次五周期全选要跑 23 个分段 × 7 个候选 = 161 次回测，实测 85 秒，而结果
包 gzip 后只有约 137 KB。所以值得存：137 KB 换 85 秒。

**存输入，不存输出。** ``window_results`` 里每个回测的完整数组（Greeks、逐
bar 价格、持仓）平均 971 KB，161 个约 153 MB——那个存不了。但逐段下钻要的
是**输入**：行情切片加构造参数。底层 1 分钟序列一年 gzip 后约 206 KB，各段
都是它的切片，存一份即可。这份输入是**兜底**：正常下钻读的是 history_bar_cache
里那份可丢弃的 bar 级缓存（3 ms），缓存不在时才用这里的配方重算选中那一段（约
620 ms），而不是 161 段。两层缺一不可——缓存能重建所以不进包，包不可重建（Wind
前复权会让重取拿到另一串数）所以必须自带输入。

（早前把这两者搞混，据此误判成「载入后无法下钻」，这里记一笔免得再犯。）

**不放在缓存目录。** intraday 缓存可丢弃，删了从 Wind 重拉即可；优选结果
不可重建——Wind 区间是从「分析截至日」往回数的，下个月用同样参数跑出来的
不是同一段数据。放在名字叫 cache 的目录里迟早被人清掉。
"""
from __future__ import annotations

import datetime
import glob
import gzip
import io
import json
import os
import sys

import numpy as np
import pandas as pd

# 包格式版本。列名与口径这些东西是会变的（增量性价比→增量信噪比、分段
# 方向从正推改成倒推），旧包用新代码渲染会静默显示错误口径——那样这个功
# 能就是在制造错误结论。载入时必须校验。
SCHEMA_VERSION = 2

_MANIFEST_NAME = "manifest.json"
_SUFFIX = ".json.gz"

# 最多保留多少份优选记录。超出后按保存时间淘汰最旧的——磁盘占用由此间接
# 收敛（结果包本身几十到几百 KB，真正的大头是它带动的 bar 级缓存）。
MAX_RESULTS = 20


def results_dir() -> str:
    """结果目录。与 pricing.wind_data 的缓存目录同款约定，但另开一个。

    开发态放仓库内 ``data/history_results``，打包冻结后放
    ``~/.deltalab/results``——``.app`` 包与安装目录可能只读。
    """
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), ".deltalab", "results")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "history_results",
    )


def _column_kind(series):
    """取一列的还原类型。

    按列名硬编码「哪些是曲线列」很脆——以后新增一列曲线就会被静默漏掉，
    读回来是 list 而不是 ndarray，图表那边不报错但会走出别的分支。这里改
    成保存时探测、写进包里，还原完全由数据驱动。
    """
    for value in series:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, np.ndarray):
            return "array"
        if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
            return "timestamp"
        if isinstance(value, tuple):
            return "tuple"
        if isinstance(value, (list, dict)):
            return "json"
        return "scalar"
    return "scalar"


# 无法序列化的值用它标记，由容器负责丢掉。真实的 history_state 里带着
# ``cfg["build"]`` 这类回调，而它嵌在字典里——只在顶层判断 callable 会让
# 它一路走到 json.dumps 才炸，那时 85 秒的结果已经跑完了。
_DROP = object()


def _jsonable(value):
    if value is None:
        return None
    if callable(value):
        return _DROP
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [
            item for item in (_jsonable(v) for v in value)
            if item is not _DROP
        ]
    if isinstance(value, dict):
        return {
            str(k): converted
            for k, converted in (
                (k, _jsonable(v)) for k, v in value.items())
            if converted is not _DROP
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    # 到这里说明是个 json 不认识的对象。宁可丢掉也不要让整次保存失败——
    # 结果包的价值在那两张表和冻结参数上，不在某个附带对象上。
    return _DROP


def _frame_to_payload(frame):
    """把 DataFrame 连同每列的还原类型一起打包。"""
    if frame is None:
        return None
    kinds = {str(name): _column_kind(frame[name]) for name in frame.columns}
    records = []
    for _index, row in frame.iterrows():
        record = {}
        for key, value in row.items():
            converted = _jsonable(value)
            # 表格的列结构必须完整：某个格子不可序列化时写 null，而不是
            # 把这一列从该行里抽掉——那会让还原出来的 DataFrame 缺列。
            record[str(key)] = None if converted is _DROP else converted
        records.append(record)
    return {
        "columns": [str(name) for name in frame.columns],
        "kinds": kinds,
        "records": records,
    }


def _payload_to_frame(payload):
    """按包里记录的类型还原 DataFrame。"""
    if not payload:
        return None
    columns = [str(name) for name in payload.get("columns", [])]
    kinds = dict(payload.get("kinds", {}))
    records = list(payload.get("records", []))
    frame = pd.DataFrame(records, columns=columns or None)
    if frame.empty:
        return frame
    for name in frame.columns:
        kind = kinds.get(str(name), "scalar")
        if kind == "array":
            frame[name] = [
                np.asarray(v, dtype=float) if v is not None
                else np.asarray([], dtype=float)
                for v in frame[name]
            ]
        elif kind == "timestamp":
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
        elif kind == "tuple":
            # 缺失同样归一成 np.nan，与下面 scalar 分支保持一致——那里的
            # 注释解释了为什么必须统一。
            frame[name] = [
                tuple(v) if isinstance(v, list)
                else (np.nan if v is None else v)
                for v in frame[name]
            ]
        else:
            # JSON 的 null 读回来是 None，而 pandas 原生的缺失是 np.nan。
            # 两者 pd.isna 都为真，但 repr 不同——下游若对元数据列做
            # str(...) 会分别拿到 "None" 和 "nan" 两种垃圾字符串。统一成
            # np.nan，往返才是精确的，不必逐个审计消费方。
            frame[name] = [
                np.nan if v is None else v for v in frame[name]
            ]
    return frame


def _safe_token(text, fallback="未命名"):
    token = "".join(
        ch for ch in str(text or "") if ch.isalnum() or ch in "._-"
    ).strip("._-")
    return token or fallback


def build_payload(*, ranking, window_summary, history_state, source_label,
                  objective, notes=None, label=None, elapsed_seconds=None,
                  replay_index=None):
    """组装结果包。``history_state`` 是本次运行冻结的全部输入。"""
    # 回调、控件、期权对象这类不可序列化的东西不进包；它们也不是结果的一
    # 部分。真实状态里的 cfg["build"] 是嵌在字典里的回调，所以过滤必须靠
    # _jsonable 递归下去，只在顶层判断 callable 会漏。
    state = {}
    for key, value in dict(history_state or {}).items():
        if str(key).startswith("_"):
            continue
        converted = _jsonable(value)
        if converted is not _DROP:
            state[str(key)] = converted
    return {
        "schema_version": SCHEMA_VERSION,
        # 精度到微秒：秒级精度下同一秒内连存几份会完全同时刻，「最旧的
        # 是哪份」就没了定义，淘汰会变成随机删。
        "saved_at": datetime.datetime.now().isoformat(timespec="microseconds"),
        "label": str(label or "").strip(),
        "source_label": str(source_label or ""),
        "objective": str(objective or ""),
        "elapsed_seconds": (
            float(elapsed_seconds) if elapsed_seconds is not None else None),
        "notes": [str(n) for n in (notes or [])],
        "history_state": state,
        "ranking": _frame_to_payload(ranking),
        "window_summary": _frame_to_payload(window_summary),
        # 逐段下钻要的是回测**输入**（行情切片 + 构造参数），不是输出。
        "replay": build_replay_payload(replay_index),
    }


DEFAULT_SERIES_KEY = "__all__"


def _series_bucket_key(spec):
    """这一段属于哪条底层序列。

    单标的时各段都是同一条已拉取序列的精确切片（实测 15/15 完全一致），并
    成一条存最省。品种池不成立：各段来自**不同合约**，换月处同一天有两个
    价，因此按合约分桶。
    """
    meta = dict(getattr(spec, "metadata", None) or {})
    code = str(meta.get("contract_code") or "").strip()
    return code or DEFAULT_SERIES_KEY


def _slice_round_trips(merged, piece):
    """``piece`` 能否按首尾时间戳从 ``merged`` 原样切回来。"""
    got = merged.loc[piece.index[0]:piece.index[-1]]
    return (len(got) == len(piece)
            and np.array_equal(got.to_numpy(), piece.to_numpy()))


def replay_series_buckets(replay_index):
    """把逐分段行情切片按底层序列分桶合并。

    返回 ``(series_by_key, key_by_window)``，后者是 ``{(lookback, window_id):
    series_key}``。跨 lookback 会重叠（近周那 5 天也在近年里），所以桶内必
    须按索引去重，否则同一天会被存好几遍。

    **合并只有在每段都能原样切回来时才成立。** 早前无条件并成一条并
    ``keep="first"``，前提写的是「各段都是同一条序列的切片」——那只对单标
    的成立。品种池实测：P2605 段原本 ``[105,106,107,108]``，载回来变成
    ``[203,204,107,108]``，头两根是 P2601 的价；串掉的第一根还是 Day 0 锚
    点，``_rescale_option_to_real_s0`` 会拿它给整段期权重定基，错的不止那两
    根 bar。所以这里逐段校验，一旦切不回来就退化成按段各存各的：多占些空
    间，但绝不能让某段读到别段的价格。
    """
    buckets = {}
    for windows in (replay_index or {}).values():
        for spec in (windows or {}).values():
            path = getattr(spec, "external_path", None)
            if path is None or not len(path):
                continue
            buckets.setdefault(_series_bucket_key(spec), []).append(
                (spec, pd.Series(path)))
    series_by_key, key_by_window = {}, {}
    for key, entries in buckets.items():
        merged = pd.concat([piece for _spec, piece in entries])
        merged = merged[~merged.index.duplicated(keep="first")].sort_index()
        if all(_slice_round_trips(merged, piece) for _spec, piece in entries):
            series_by_key[key] = merged
            for spec, _piece in entries:
                key_by_window[(str(spec.lookback), str(spec.window_id))] = key
            continue
        for spec, piece in entries:
            window = (str(spec.lookback), str(spec.window_id))
            per_key = f"{spec.lookback}|{spec.window_id}"
            series_by_key[per_key] = piece
            key_by_window[window] = per_key
    return series_by_key, key_by_window


def series_from_replay_index(replay_index):
    """把逐分段行情切片并回一条底层序列；分桶不止一条时返回 None。

    只保留给「确实只有一条底层序列」的调用方（单标的）。多合约请改用
    ``replay_series_buckets``——并成一条会串价。
    """
    series_by_key, _keys = replay_series_buckets(replay_index)
    if len(series_by_key) != 1:
        return None
    return next(iter(series_by_key.values()))


def build_replay_payload(replay_index):
    """把重放配方序列化。

    存的是**输入**不是输出：底层价格序列一份，加上每段的起止时间戳与
    ``evaluation_days`` / ``steps_per_day`` / 回测参数。实测一年 1 分钟序
    列 gzip 后约 206 KB，而 bar 级**输出**是 153 MB——早前把这两者搞混，
    才误判成「载入后无法下钻」。

    分段的起止用时间戳而不是位置：位置依赖序列本身，时间戳则可以直接
    ``series.loc[start:end]`` 切出来，与序列如何拼接无关。
    """
    series_by_key, key_by_window = replay_series_buckets(replay_index)
    if not series_by_key:
        return None
    specs = []
    for lookback, windows in (replay_index or {}).items():
        for window_id, spec in (windows or {}).items():
            path = spec.external_path
            specs.append({
                "lookback": str(lookback),
                "window_id": str(window_id),
                "start_ts": pd.Timestamp(path.index[0]).isoformat(),
                "end_ts": pd.Timestamp(path.index[-1]).isoformat(),
                "evaluation_days": int(spec.evaluation_days),
                "steps_per_day": int(spec.steps_per_day),
                "option_s0": float(getattr(spec.option, "s0", float("nan"))),
                "strategies": [str(name) for name in spec.strategies],
                "backtest_kwargs": _jsonable(dict(spec.backtest_kwargs)),
                # 预热参数必须入包：实跑路径按候选记录它，重放时
                # HistoryReplaySpec.build 会取。缺了它，realized σ 的候选
                # 会在没有预热种子的情况下跑完，而实跑路径本来会因
                # strict_sigma_warmup=True 直接报错。
                "warmup_kwargs": _jsonable(
                    {str(k): v for k, v in dict(spec.warmup_kwargs).items()}),
                "metadata": _jsonable(dict(spec.metadata)),
                # 这一段该从哪条序列上切。单标的只有一条（DEFAULT_SERIES_KEY），
                # 品种池按合约一条——不记这个字段就没法知道该切哪条。
                "series_key": key_by_window.get(
                    (str(lookback), str(window_id)), DEFAULT_SERIES_KEY),
            })
    return {
        "price_series_by_key": {
            str(key): {
                "index": [pd.Timestamp(ts).isoformat() for ts in series.index],
                "values": [float(v) for v in series.to_numpy()],
            }
            for key, series in series_by_key.items()
        },
        "specs": specs,
    }


def _series_from_block(raw):
    index = pd.to_datetime(list((raw or {}).get("index", ())))
    values = np.asarray(list((raw or {}).get("values", ())), dtype=float)
    if not len(index) or len(index) != len(values):
        return None
    return pd.Series(values, index=index)


def restore_replay_payload(payload):
    """还原成 ``({序列键: 价格序列}, 分段配方列表)``；没有重放数据时 ``({}, [])``。

    兼容旧包：本修复之前写的包只有一条扁平的 ``price_series``，那时的合并
    方式对**单标的**是对的，直接挂到 ``DEFAULT_SERIES_KEY`` 上照常用。

    但旧的**品种池**包不可救：多合约当初就已经被并成一条、换月处的价被顶
    掉了，包里存的就是错数。这种情况直接判定不可重放，而不是拿错价格画一
    条看起来正常的曲线——后者不会报错，还能被保留进结果对比。
    """
    block = (payload or {}).get("replay")
    if not block:
        return {}, []
    specs = list(block.get("specs", ()))
    keyed = block.get("price_series_by_key")
    if keyed:
        out = {}
        for key, raw in dict(keyed).items():
            series = _series_from_block(raw)
            if series is not None:
                out[str(key)] = series
        return (out, specs) if out else ({}, [])

    series = _series_from_block(block.get("price_series"))
    if series is None:
        return {}, []
    contracts = {
        str((dict(spec.get("metadata") or {})).get("contract_code") or "")
        for spec in specs
    }
    contracts.discard("")
    if len(contracts) > 1:
        return {}, []
    return {DEFAULT_SERIES_KEY: series}, specs


def default_label(state, existing=None, now=None):
    """保存对话框的预填名：``品种 Buy/Sell 保存时刻``。

    第三段取保存时刻（``MM-DD HH:MM``）。它天生唯一，把「防重名」这件事
    一次解决；分析截至日试过，但同一天连跑两次就撞，而且它已经单独成列。
    不带年份是有意的：列表最多留 20 份，同月同日同分钟跨年重名不现实，
    而少四个字符能让名称列容得下。

    ``existing`` 传入已有的名字，完全同名时追加 ``#2``/``#3``——同一分钟
    内连存两次仍会撞，这层兜底保留。

    ``position`` 是**买卖方向**：1=卖出、-1=买入（见 HedgeBacktest）。这里
    用 Buy/Sell 而不是 Long/Short——后者在期权语境里通常指 gamma 敞口，而
    同一个买卖方向在不同产品上敞口相反（实测卖出时香草是 short gamma、雪
    球是 long gamma），拿它当买卖方向的标签必然对一半错一半。
    """
    state = dict(state or {})
    code = str(state.get("wind_code") or "").strip()
    if not code:
        csv_path = str(state.get("csv_path") or "").strip()
        code = os.path.splitext(os.path.basename(csv_path))[0] if csv_path else ""
    try:
        position = int(state.get("position", 1))
    except (TypeError, ValueError):
        position = 1
    side = "Sell" if position == 1 else "Buy"
    stamp = (now or datetime.datetime.now()).strftime("%m-%d %H:%M")
    base = " ".join(piece for piece in (code, side, stamp) if piece)
    if not base:
        base = side
    taken = {str(name).strip() for name in (existing or ()) if str(name).strip()}
    if base not in taken:
        return base
    for serial in range(2, 1000):
        candidate = f"{base} #{serial}"
        if candidate not in taken:
            return candidate
    return base


def default_filename(payload):
    """``<标的>_<截至日>_<周期数>_<时间戳>.json.gz``。"""
    state = dict(payload.get("history_state") or {})
    code = _safe_token(state.get("wind_code") or state.get("csv_path") or "结果")
    asof = _safe_token(
        state.get("history_wind_asof") or state.get("wind_end") or "", "无区间")
    lookbacks = state.get("history_lookbacks") or {}
    periods = f"{len(lookbacks)}周期" if lookbacks else "无周期"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{code}_{asof}_{periods}_{stamp}{_SUFFIX}"


def save_result(payload, *, directory=None, filename=None, enforce=True):
    """写入结果包，返回完整路径。

    ``enforce=False`` 时不在这里淘汰旧记录——调用方需要拿到被淘汰的清单
    报给用户时，自己调 ``enforce_limit``。
    """
    directory = directory or results_dir()
    os.makedirs(directory, exist_ok=True)
    filename = filename or default_filename(payload)
    if not filename.endswith(_SUFFIX):
        filename += _SUFFIX
    path = os.path.join(directory, filename)
    # 文件名的时间戳只到秒。同一秒里连存两份（很容易——存完发现名字打错了
    # 立刻重存）会静默覆盖，用户以为存了两份、实际只剩最后一份。撞名就往
    # 后加序号，绝不覆盖已有结果。
    if os.path.exists(path):
        stem = filename[:-len(_SUFFIX)]
        for serial in range(2, 1000):
            candidate = os.path.join(directory, f"{stem}-{serial}{_SUFFIX}")
            if not os.path.exists(candidate):
                path = candidate
                break
        else:
            raise OSError(f"同名结果包过多，无法保存: {filename}")
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 先写临时文件再原子替换：85 秒跑出来的结果不能因为写一半崩掉而只留下
    # 一个坏包，而坏包在列表里看起来和好包一样。
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        handle.write(text)
    os.replace(tmp, path)
    if enforce:
        enforce_limit(directory=directory)
    return path


def enforce_limit(max_results=MAX_RESULTS, *, directory=None):
    """只保留最近 ``max_results`` 份记录，返回被淘汰的路径列表。

    删除是不可逆的，所以调用方应当把返回值报给用户——静默删掉别人 85 秒
    跑出来的东西，即使是按规则删的，也必须说一声。
    """
    items = list_results(directory)
    if max_results is None or max_results <= 0 or len(items) <= max_results:
        return []
    evicted = []
    for item in items[max_results:]:          # list_results 已按时间倒序
        if delete_result(item["path"]):
            evicted.append(item)
    return evicted


class HistoryResultVersionError(ValueError):
    """包的 schema 版本与当前代码不符。"""


def load_result(path, *, allow_other_version=False):
    """读回结果包并还原两张表。

    版本不符时默认拒绝：列名与口径都变过，硬渲染会静默给出错误结论。
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION and not allow_other_version:
        raise HistoryResultVersionError(
            f"结果包版本 {version} 与当前程序（{SCHEMA_VERSION}）不一致，"
            "口径可能已改变；请重新运行策略优选。")
    payload["ranking"] = _payload_to_frame(payload.get("ranking"))
    payload["window_summary"] = _payload_to_frame(
        payload.get("window_summary"))
    return payload


_PERIOD_LABELS = dict(
    (("week", "近周"), ("month", "近月"), ("quarter", "近季"),
     ("half_year", "近半年"), ("year", "近年")))


def _evidence_span(records, lookbacks):
    """证据跨度 = 最长的那个周期。

    列表里写「2 个周期」没用：近周+近月与近半年+近年都是「2 个」，可信度
    却差着量级。决定这份结果值不值得信的是最长那一段。
    """
    longest, best_days = "", -1
    for record in records:
        try:
            days = int(record.get("lookback_days") or 0)
        except (TypeError, ValueError):
            continue
        if days > best_days:
            best_days, longest = days, str(record.get("lookback") or "")
    if not longest:
        keys = list(lookbacks or ())
        longest = keys[-1] if keys else ""
    return _PERIOD_LABELS.get(longest, longest or "—")


def _verdict(records):
    """保存时那个排名口径下的跨周期结论，与结果页那枚一致性 pill 同口径。

    各周期都指向同一策略才写策略名；有分歧就写「不一致（N 种）」而不是
    挑一个报出来——把弱证据伪装成强结论，正是这个页面一直在避免的事。
    """
    best_by_period = {}
    for record in records:
        if not record.get("recommendation_eligible"):
            continue
        if str(record.get("strategy_type") or "") == "close_to_close":
            continue
        name = str(record.get("strategy") or "").strip()
        if not name or name == "—":
            continue
        period = str(record.get("lookback") or "")
        try:
            rank = int(record.get("rank") or 0)
        except (TypeError, ValueError):
            continue
        current = best_by_period.get(period)
        if current is None or rank < current[0]:
            best_by_period[period] = (rank, name)
    picks = [name for _rank, name in best_by_period.values()]
    if not picks:
        return "无可比结论"
    unique = set(picks)
    if len(unique) == 1:
        return picks[0]
    return f"不一致（{len(unique)} 种）"


def _option_summary(state):
    """``T22 σ0.18``。

    两者都是决定性的：T 决定 gamma 剖面；σ 直接决定带宽的绝对宽度
    （``k·S·σ/√243``）——同一个「0.75σ」在 σ=0.12 与 σ=0.24 下是两条完全
    不同的带，不写出来两份结果会看着一样。
    """
    params = dict(state.get("params") or {})
    pieces = []
    days = params.get("T_days", params.get("T"))
    try:
        if days is not None:
            pieces.append(f"T{int(days)}")
    except (TypeError, ValueError):
        pass
    try:
        sigma = float(params.get("sigma"))
        if np.isfinite(sigma):
            pieces.append(f"σ{sigma:g}")
    except (TypeError, ValueError):
        pass
    return " ".join(pieces) or "—"


def _subtype_label(state, display_map=None):
    """期权子类型的中文名。

    ``display_map`` 由调用方注入（GUI 侧的 ``SUBTYPE_DISPLAY``），这里不
    直接 import gui_app——本模块要保持纯逻辑、可独立测试。映射里没有的原
    样返回，只去掉注册表统一的 ``Opt_`` 前缀（那四个字符不携带信息）。
    """
    subtype = str(state.get("subtype") or "").strip()
    if not subtype:
        return "—"
    if display_map and subtype in display_map:
        return str(display_map[subtype])
    return subtype[4:] if subtype.startswith("Opt_") else subtype


def _position_summary(state):
    """``Buy 费0.01%``，收盘保底被关掉时追加警示。

    买卖方向决定对冲逻辑的方向；成本率决定「多调仓划不划算」这个核心权衡，
    0.01% 与 0.05% 会选出不同的最优带宽。数量与乘数不入列——实测它们只
    线性缩放金额，不改变名次。

    收盘保底默认开启且很少改，所以只在**关闭**时标记：正常情况下不占版
    面，而一旦不同就必须看得见——实测关掉它能让固定间隔 2.0σ 从第 1 名
    掉到第 6 名。
    """
    try:
        position = int(state.get("position", 1))
    except (TypeError, ValueError):
        position = 1
    pieces = ["Sell" if position == 1 else "Buy"]
    try:
        rate = float(state.get("tc_rate"))
        if np.isfinite(rate):
            pieces.append(f"费{rate * 100:g}%")
    except (TypeError, ValueError):
        pass
    if "force_day_close_hedge" in state and not state.get(
            "force_day_close_hedge"):
        pieces.append("⚠无保底")
    return " ".join(pieces)


# 子类型中文名的来源。GUI 在 import 时注入自己的 SUBTYPE_DISPLAY——本模
# 块不反向依赖 gui_app，未注入时退回原始标识，列表仍可用。
SUBTYPE_DISPLAY = {}


def _peek(path):
    """读元数据供列表页使用；不还原成 DataFrame。

    注意它仍会 ``json.load`` 整个包（曲线也会解成 list），只是不做类型还
    原。实测约 9 ms/份、30 份 272 ms——量级可以接受，所以没有为列表另建
    索引文件或 sidecar；真到几百份再说。
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    # ValueError 一并覆盖 JSONDecodeError 与 UnicodeDecodeError——gzip 流
    # 完好但内容不是合法 utf-8 时抛后者，漏掉会让整个 list_results 崩掉，
    # 而不是跳过这一个坏包。BadGzipFile 是 OSError 子类，无需单列。
    except (OSError, EOFError, ValueError):
        return None
    state = dict(payload.get("history_state") or {})
    ranking = payload.get("ranking") or {}
    records = list(ranking.get("records", ()))
    lookbacks = state.get("history_lookbacks") or {}
    return {
        "path": path,
        "filename": os.path.basename(path),
        "schema_version": payload.get("schema_version"),
        "compatible": payload.get("schema_version") == SCHEMA_VERSION,
        "saved_at": payload.get("saved_at") or "",
        "label": payload.get("label") or "",
        "source_label": payload.get("source_label") or "",
        "objective": payload.get("objective") or "",
        "wind_code": state.get("wind_code") or "",
        "asof": (state.get("history_wind_asof")
                 or state.get("wind_end") or ""),
        "lookbacks": list(lookbacks.keys()),
        "option_summary": _option_summary(state),
        "subtype": _subtype_label(state, SUBTYPE_DISPLAY),
        "position_summary": _position_summary(state),
        "evidence_span": _evidence_span(records, lookbacks),
        "verdict": _verdict(records),
        "rows": len(records),
        "bytes": os.path.getsize(path),
        "_mtime": os.path.getmtime(path),
    }


def _sweep_partials(directory):
    """清掉进程崩溃残留的 ``.part``。

    它们不会出现在 ``*.json.gz`` 的 glob 里，也就没有别的清理点，只会一直
    占着盘。列目录时顺手扫一次即可。
    """
    for name in os.listdir(directory):
        if not name.endswith(".part"):
            continue
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            continue


def list_results(directory=None):
    """按保存时间倒序列出结果包；坏包与非法文件跳过。"""
    directory = directory or results_dir()
    if not os.path.isdir(directory):
        return []
    _sweep_partials(directory)
    items = []
    for path in glob.glob(os.path.join(directory, "*" + _SUFFIX)):
        meta = _peek(path)
        if meta is not None:
            items.append(meta)
    # mtime 作次级键：旧包的 saved_at 只精确到秒，同秒的几份仍需确定顺序。
    items.sort(
        key=lambda item: (item["saved_at"], item.get("_mtime", 0.0)),
        reverse=True)
    return items


def delete_result(path):
    """删除一个结果包；不存在时静默返回 False。"""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def rename_result(path, label):
    """只改包内的展示名，不动文件名——文件名带时间戳，是排序依据。"""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["label"] = str(label or "").strip()
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        handle.write(text)
    os.replace(tmp, path)
    return path
