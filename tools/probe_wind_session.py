# _*_ coding: utf-8 _*_
"""
Wind 字段两阶段探测脚本 —— 交易所 / 品种类型 / 夜盘 / 日内交易时段
=================================================================

用途
----
本脚本用于在装有 Wind 金融终端的本地环境，绕开 `w.wss` 的
**all-or-nothing** 行为（整组里只要有 1 个字段 Wind 不认识，
就会返回 `ErrorCode=-40522006 invalid indicators` 整批拒绝），
通过两阶段探测把候选字段清单压缩到"真正被 Wind 接受"的子集。

两阶段流程
----------
* **阶段 1：单代码 × 单字段**
  只用 `au2412.SHF`（黄金，最有代表性的夜盘合约），
  对每一个候选字段**单独**调用 `w.wss(code, field)`，
  根据 ErrorCode 与 Data 把字段分类为 OK / INVALID / NODATA / ERROR，
  末尾打印一份"有效字段清单"。

* **阶段 2：多代码 × 批量字段**
  用阶段 1 筛出的有效字段，对剩余 8 个代码一次性
  `w.wss(code, valid_fields)`，按 code 分段打印实际值。
  因为字段已经过验证，这里可以安全 all-in-one。

运行方式
--------
在 Wind 终端已启动的机器上：

    python tools/probe_wind_session.py

预期产物
--------
标准输出中第一段是阶段 1 的单字段扫描结果（约 30 行 + 分类汇总），
第二段是阶段 2 的跨合约字段值对比表。把整段输出复制给开发者
（Claude/quant-expert）即可对齐字段清单并修改
pricing/wind_data.py 的夜盘识别逻辑。

注意
----
* 只调用 `w.wss`（静态字段），Wind 配额开销极小。
* 每次 wss 都被 try/except 包住，单个字段失败不会中断后续探测。
* 输出会比上一版长（约 200~300 行），但这是诊断所必需的。
"""

from __future__ import annotations

import math
import sys
import traceback


# -----------------------------------------------------------------------------
# 阶段 1 所用代码：选最有代表性的夜盘合约
# -----------------------------------------------------------------------------
PROBE_CODE = "au2412.SHF"  # 黄金，长夜盘 21:00-02:30

# -----------------------------------------------------------------------------
# 阶段 2 所用代码：剩余 8 个，覆盖多交易所 / 多夜盘场景
# -----------------------------------------------------------------------------
BATCH_CODES = [
    "510050.SH",   # A 股 ETF，无夜盘
    "IF2412.CFE",  # 沪深 300 股指期货，无夜盘
    "cu2412.SHF",  # 沪铜，21:00-01:00
    "rb2412.SHF",  # 螺纹钢，21:00-23:00
    "i2501.DCE",   # 铁矿石，21:00-23:00
    "MA501.CZC",   # 甲醇，21:00-23:00
    "sc2412.INE",  # 原油，21:00-02:30
    "si2501.GFE",  # 工业硅，无夜盘
]

# -----------------------------------------------------------------------------
# 候选字段：阶段 1 逐个单独试
# 原 19 个 + 额外建议（含元信息、交易时段别名、夜盘别名）
# -----------------------------------------------------------------------------
CANDIDATE_FIELDS = [
    # 基础元信息
    "exch_eng", "exch_cn", "sec_type", "sectype", "sec_name",
    "windcode", "exchange", "exch",
    # 交易时段
    "tradetime", "trade_time", "trading_session",
    "trade_session", "trade_hours",
    "tradinghours", "tradinghour", "session", "marketsession",
    # 夜盘标记
    "nightTrading", "night_trading",
    "ifnight_trade", "ifnighttrade", "nighttrade",
    "night", "isnight", "hasnighttrade",
    # 合约规格 / 元信息
    "contractmultiplier", "punit", "ftmargins", "lasttradingdate",
    "mfprice", "changelt", "prevclose",
]


# -----------------------------------------------------------------------------
# 输出辅助
# -----------------------------------------------------------------------------
def _fmt_value(v):
    """把 Wind 返回的单值格式化成一行可读字符串。"""
    if v is None:
        return "<None>"
    try:
        s = str(v)
    except Exception:
        s = repr(v)
    if len(s) > 200:
        s = s[:200] + "...<truncated>"
    return s


def _print_section(title: str):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def _print_subsection(title: str):
    print(f"\n--- {title} ---")


def _is_empty_value(v) -> bool:
    """判断 Wind 返回的单值是否等同于"无数据"。"""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return True
    return False


def _is_invalid_indicator(data) -> bool:
    """识别 ErrorCode=-40522006 的 invalid indicators 场景。"""
    if getattr(data, "ErrorCode", 0) == -40522006:
        return True
    # 部分 WindPy 版本把错误信息放在 Data 或 OUTMESSAGE 里
    try:
        msg_parts = []
        for attr in ("Data", "Codes", "Fields"):
            v = getattr(data, attr, None)
            if v is not None:
                msg_parts.append(str(v))
        blob = " ".join(msg_parts).lower()
        if "invalid indicators" in blob:
            return True
    except Exception:
        pass
    return False


# -----------------------------------------------------------------------------
# 阶段 1：单代码 × 单字段扫描
# -----------------------------------------------------------------------------
def probe_fields_one_by_one(w, code: str) -> list[str]:
    """对 code 逐字段调用 wss，返回有效字段清单。"""
    _print_section(f"阶段 1：单代码逐字段探测   CODE = {code}")
    print(f"候选字段共 {len(CANDIDATE_FIELDS)} 个，逐个调用 w.wss(code, field) ...\n")

    ok_fields: list[str] = []
    invalid_fields: list[str] = []
    nodata_fields: list[str] = []
    error_fields: list[tuple[str, int]] = []

    name_width = max(len(f) for f in CANDIDATE_FIELDS) + 2

    for field in CANDIDATE_FIELDS:
        try:
            data = w.wss(code, field)
        except Exception as exc:
            print(f"[EXCEPTION] {field:<{name_width}} w.wss 抛错: {exc!r}")
            error_fields.append((field, -1))
            continue

        ec = getattr(data, "ErrorCode", None)

        if _is_invalid_indicator(data):
            print(f"[INVALID]   {field:<{name_width}} [字段名 Wind 不认识]")
            invalid_fields.append(field)
            continue

        if ec != 0:
            print(f"[ERROR]     {field:<{name_width}} [ErrorCode={ec}]")
            error_fields.append((field, ec))
            continue

        # ErrorCode == 0：检查 Data 有无内容
        val = None
        try:
            if data.Data and data.Data[0]:
                val = data.Data[0][0]
        except Exception:
            val = "<parse error>"

        if _is_empty_value(val):
            print(f"[NODATA]    {field:<{name_width}} [字段存在但本合约无数据]")
            nodata_fields.append(field)
        else:
            print(f"[OK]        {field:<{name_width}} -> {_fmt_value(val)}")
            ok_fields.append(field)

    # 汇总
    _print_subsection("阶段 1 汇总")
    print(f"  OK      ({len(ok_fields):>2}): {ok_fields}")
    print(f"  NODATA  ({len(nodata_fields):>2}): {nodata_fields}")
    print(f"  INVALID ({len(invalid_fields):>2}): {invalid_fields}")
    print(f"  ERROR   ({len(error_fields):>2}): {error_fields}")
    print(f"\n有效字段清单 = {ok_fields + nodata_fields}")
    print("  （OK + NODATA 均视为 Wind 认识的字段，进入阶段 2 批量验证）")

    # 阶段 2 用 OK + NODATA：字段合法，只是 au 上可能没值，其他合约或许有值
    return ok_fields + nodata_fields


# -----------------------------------------------------------------------------
# 阶段 2：多代码 × 批量字段
# -----------------------------------------------------------------------------
def probe_codes_with_valid_fields(w, codes: list[str], valid_fields: list[str]):
    """用 valid_fields 对一批 codes 逐个批量 wss 并打印。"""
    _print_section(
        f"阶段 2：批量复用有效字段   共 {len(codes)} 个代码 × "
        f"{len(valid_fields)} 个字段"
    )

    if not valid_fields:
        print("[WARN] 阶段 1 没有筛出任何有效字段，跳过阶段 2。")
        return

    name_width = max(len(f) for f in valid_fields) + 2
    fields_str = ",".join(valid_fields)

    for code in codes:
        _print_subsection(f"CODE = {code}")
        try:
            data = w.wss(code, fields_str)
        except Exception as exc:
            print(f"[EXCEPTION] w.wss({code}) 抛错: {exc!r}")
            traceback.print_exc()
            continue

        ec = getattr(data, "ErrorCode", None)
        print(f"  ErrorCode = {ec}")
        if ec != 0:
            print(f"  [WARN] 批量调用失败，Data = {getattr(data, 'Data', None)}")
            continue

        returned_fields = [str(f).lower() for f in (data.Fields or [])]
        value_map = {}
        for i, fld in enumerate(returned_fields):
            try:
                value_map[fld] = (
                    data.Data[i][0] if data.Data and data.Data[i] else None
                )
            except Exception:
                value_map[fld] = "<parse error>"

        for f in valid_fields:
            key = f.lower()
            if key in value_map:
                print(f"  {f:<{name_width}} -> {_fmt_value(value_map[key])}")
            else:
                print(f"  {f:<{name_width}} -> <field not returned>")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    try:
        from WindPy import w
    except ImportError:
        print("[FATAL] 未安装 WindPy，请在 Wind 金融终端环境运行本脚本。")
        sys.exit(1)

    print("正在启动 Wind 连接 ...")
    try:
        start_ret = w.start()
        print(f"w.start() -> ErrorCode={getattr(start_ret, 'ErrorCode', '?')} "
              f"Data={getattr(start_ret, 'Data', '?')}")
    except Exception as exc:
        print(f"[FATAL] w.start() 抛错: {exc!r}")
        traceback.print_exc()
        sys.exit(1)

    if not w.isconnected():
        print("[FATAL] Wind 未连接，请先登录 Wind 终端后重试。")
        sys.exit(1)

    print(f"Wind 已连接。\n"
          f"  阶段 1 探测代码：{PROBE_CODE}\n"
          f"  阶段 1 候选字段数：{len(CANDIDATE_FIELDS)}\n"
          f"  阶段 2 复用代码数：{len(BATCH_CODES)}")

    # 阶段 1
    try:
        valid_fields = probe_fields_one_by_one(w, PROBE_CODE)
    except Exception as exc:
        print(f"[EXCEPTION] 阶段 1 失败: {exc!r}")
        traceback.print_exc()
        valid_fields = []

    # 阶段 2
    try:
        probe_codes_with_valid_fields(w, BATCH_CODES, valid_fields)
    except Exception as exc:
        print(f"[EXCEPTION] 阶段 2 失败: {exc!r}")
        traceback.print_exc()

    try:
        w.stop()
        print("\nWind 连接已关闭。")
    except Exception as exc:
        print(f"[WARN] w.stop() 抛错: {exc!r}")


if __name__ == "__main__":
    main()
