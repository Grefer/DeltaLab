# _*_ coding: utf-8 _*_
"""
Smoke 测试：验证 get_trading_minutes_per_day 在 Wind 终端下返回值是否符合预期。

用法（需在已登录 Wind 金融终端的环境执行）：
    python tools/smoke_trading_minutes.py

预期输出（每行两列：code -> minutes）：
    510050.SH  -> 240
    IF2412.CFE -> 240
    au2412.SHF -> 570
    cu2412.SHF -> 480
    rb2412.SHF -> 360
    i2501.DCE  -> 360
    MA501.CZC  -> 360
    sc2412.INE -> 570
    si2501.GFE -> 240

如果某行返回 None 或与预期不符，很可能是 Wind wss 返回的 sec_type 中文字符串
与常量表里预设的差了一点（例如"指数"vs"指数类"、"ETF"vs"基金"等），请按实测
值调整 pricing/wind_data.py 中的 _TRADING_MINUTES_TABLE。
"""

import os
import sys

# 允许直接 `python tools/smoke_trading_minutes.py` 运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pricing.wind_data import get_trading_minutes_per_day


TEST_CASES = [
    ("510050.SH", 240),
    ("IF2412.CFE", 240),
    ("au2412.SHF", 570),
    ("cu2412.SHF", 480),
    ("rb2412.SHF", 360),
    ("i2501.DCE", 360),
    ("MA501.CZC", 360),
    ("sc2412.INE", 570),
    ("si2501.GFE", 240),
]


def main():
    print(f"{'code':<14} {'expect':>6}  {'actual':>6}  status")
    print("-" * 40)
    mismatches = []
    for code, expect in TEST_CASES:
        try:
            actual = get_trading_minutes_per_day(code)
        except Exception as e:
            actual = f"ERR: {e!r}"
        status = "OK" if actual == expect else "MISMATCH"
        if status != "OK":
            mismatches.append((code, expect, actual))
        print(f"{code:<14} {expect:>6}  {str(actual):>6}  {status}")

    print("-" * 40)
    if mismatches:
        print(f"共 {len(mismatches)} 个不匹配，请检查 Wind wss 返回的 "
              f"exch_eng/sec_type 是否与 _TRADING_MINUTES_TABLE 一致。")
    else:
        print("全部匹配。")


if __name__ == "__main__":
    main()
