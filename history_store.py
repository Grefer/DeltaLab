"""策略优选结果的落盘存取（纯逻辑，不依赖 tkinter）。

一次五周期全选要跑 23 个分段 × 7 个候选 = 161 次回测，实测 85 秒，而结果
包 gzip 后只有约 137 KB。所以值得存：137 KB 换 85 秒。

**不存 bar 级结果。** ``window_results`` 里每个回测的完整数组（Greeks、逐
bar 价格、持仓）平均 971 KB，161 个约 153 MB。因此载入后「加载某段明细」
不可用——调用方必须在界面上讲明这一点，而不是让按钮点了没反应。

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
SCHEMA_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_SUFFIX = ".json.gz"


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
            frame[name] = [
                tuple(v) if isinstance(v, list) else v for v in frame[name]
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
                  objective, notes=None, label=None, elapsed_seconds=None):
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
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": str(label or "").strip(),
        "source_label": str(source_label or ""),
        "objective": str(objective or ""),
        "elapsed_seconds": (
            float(elapsed_seconds) if elapsed_seconds is not None else None),
        "notes": [str(n) for n in (notes or [])],
        "history_state": state,
        "ranking": _frame_to_payload(ranking),
        "window_summary": _frame_to_payload(window_summary),
    }


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


def save_result(payload, *, directory=None, filename=None):
    """写入结果包，返回完整路径。"""
    directory = directory or results_dir()
    os.makedirs(directory, exist_ok=True)
    filename = filename or default_filename(payload)
    if not filename.endswith(_SUFFIX):
        filename += _SUFFIX
    path = os.path.join(directory, filename)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 先写临时文件再原子替换：85 秒跑出来的结果不能因为写一半崩掉而只留下
    # 一个坏包，而坏包在列表里看起来和好包一样。
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        handle.write(text)
    os.replace(tmp, path)
    return path


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


def _peek(path):
    """只读元数据，不还原表格——列表页不需要把曲线全解出来。"""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, EOFError, json.JSONDecodeError, gzip.BadGzipFile):
        return None
    state = dict(payload.get("history_state") or {})
    ranking = payload.get("ranking") or {}
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
        "lookbacks": list((state.get("history_lookbacks") or {}).keys()),
        "rows": len(ranking.get("records", ())),
        "bytes": os.path.getsize(path),
    }


def list_results(directory=None):
    """按保存时间倒序列出结果包；坏包与非法文件跳过。"""
    directory = directory or results_dir()
    if not os.path.isdir(directory):
        return []
    items = []
    for path in glob.glob(os.path.join(directory, "*" + _SUFFIX)):
        meta = _peek(path)
        if meta is not None:
            items.append(meta)
    items.sort(key=lambda item: item["saved_at"], reverse=True)
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
