# _*_ coding: utf-8 _*_
"""
DeltaLab - 期权对冲回测 GUI 应用

基于 tkinter 构建，支持选择不同期权类型、回测方式（模拟/历史数据），
并以图表和表格形式展示回测结果。
"""

import sys
import os
import bisect
import copy
import hashlib
import platform
import threading
import datetime
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
import numpy as np
import matplotlib
# 拿不到窗口服务器时（CI 的 headless Linux、无 DISPLAY 的批处理调用），
# TkAgg 会抛 ImportError："Cannot load backend 'TkAgg' ... as 'headless' is
# currently running"。回退到 Agg 只为让 **import 本身**成功——真要画图的那批
# 测试仍在 xvfb 下跑，那时 DISPLAY 有效，选中的照样是 TkAgg。
#
# 这里不回退的话，`pytest -m "not gui"` 也救不了：marker 要把模块 import 完
# 才读得到，而炸点恰恰就在 import gui_app 这一步，于是连一条纯逻辑测试都收集
# 不起来（实测 CI 上 4 个模块全部 collection error）。
try:
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _resource_path(*parts: str) -> str:
    # PyInstaller 解压到 sys._MEIPASS; 开发态以源文件所在目录为根.
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)

# ---- 跨平台中文字体设置 ----
_SYSTEM = platform.system()
if _SYSTEM == "Darwin":
    _CJK_CANDIDATES = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
                       "Hiragino Sans GB", "Songti SC"]
elif _SYSTEM == "Windows":
    _CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun"]
else:
    _CJK_CANDIDATES = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                       "WenQuanYi Micro Hei", "Source Han Sans SC"]

_AVAILABLE_FONTS = {f.name for f in font_manager.fontManager.ttflist}
_CJK_FALLBACK = [f for f in _CJK_CANDIDATES if f in _AVAILABLE_FONTS] + ["DejaVu Sans"]
plt.rcParams['font.sans-serif'] = _CJK_FALLBACK
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 显式绑定一个 CJK 字体文件, 供 matplotlib text() 等接口通过 fontproperties 强制使用,
# 避免某些调用路径回退到不含 CJK 字形的默认字体导致乱码.
_MPL_CJK_FP = None
for _f in font_manager.fontManager.ttflist:
    if _f.name in _CJK_CANDIDATES:
        _MPL_CJK_FP = font_manager.FontProperties(fname=_f.fname)
        break

# Tk/ttk 使用的中文 UI 字体族（取第一个可用的 CJK 字体）
_UI_FONT_FAMILY = _CJK_FALLBACK[0] if _CJK_FALLBACK[0] != "DejaVu Sans" else "TkDefaultFont"

# 等宽字体 (用于摘要/结构文本)
if _SYSTEM == "Windows":
    _MONO_CANDIDATES = ["Cascadia Mono", "Consolas", "Courier New"]
elif _SYSTEM == "Darwin":
    _MONO_CANDIDATES = ["Menlo", "Monaco", "Courier New"]
else:
    _MONO_CANDIDATES = ["DejaVu Sans Mono", "Liberation Mono", "Courier New"]
_MONO_FONT_FAMILY = next((f for f in _MONO_CANDIDATES if f in _AVAILABLE_FONTS), "Courier")

# ---- 统一视觉调色板 (现代扁平风格, 蓝灰系) ----
PALETTE = {
    "bg":           "#F3F5F9",   # 窗口底色
    "surface":      "#FFFFFF",   # 卡片/面板
    "surface_alt":  "#F8FAFC",   # 次级表面 (Text 背景, 斑马行)
    "border":       "#D8DEE8",   # 边框
    "border_soft":  "#E5E9F0",   # 轻分割线
    "text":         "#1F2937",   # 主文字
    "text_muted":   "#6B7280",   # 次要文字
    "text_light":   "#9CA3AF",   # 更浅文字 (占位符)
    "primary":      "#2563EB",   # 主色 (运行按钮)
    "primary_hov":  "#1D4ED8",
    "primary_act":  "#1E40AF",
    "primary_light":"#EFF6FF",   # 主色浅底
    "accent":       "#0EA5E9",   # 次级按钮
    "accent_hov":   "#0284C7",
    "success":      "#16A34A",
    "success_light":"#F0FDF4",   # 成功浅底
    "warning":      "#D97706",
    "warning_light":"#FFFBEB",   # 警告浅底
    "danger":       "#DC2626",
    "danger_light": "#FEF2F2",   # 危险浅底
    "selected":     "#DBEAFE",   # 选中高亮
    "gold":         "#B8860B",   # 金色 (装饰线)
    "tab_inactive": "#E2E8F0",   # 未选中 tab 底色
}

# ---- 左侧参数面板的统一度量 (像素) ----
# 三个分组 (期权类型 / 期权参数 / 回测设置) 以及它们内部的子面板共用同一套
# 列宽, 输入框的左右边缘才能跨分组对齐; 列宽由 grid 的 minsize 强制, 各控件
# 自己的 width= 只是字符数下限, 因此统一改这里就能整体调节表单尺寸。
FORM_LABEL_W     = 144   # 标签列宽 (含标签右侧留白; 容得下最长的雪球保证金标签)
FORM_INPUT_W     = 168   # 输入列宽: 所有 Entry / Combobox 等宽
FORM_LABEL_GAP   = 10    # 标签与输入框之间的留白
FORM_HINT_GAP    = 8     # 输入框与右侧说明文字之间的留白
FORM_ROW_PADY    = 4     # 每行上下留白 => 相邻控件间隔恒为 8
FORM_SECTION_PAD = 12    # 分组内边距
FORM_SECTION_GAP = 10    # 分组之间的间距
FORM_ENTRY_CHARS = 8     # 字符宽度只作下限, 实际宽度取 FORM_INPUT_W


def _form_grid(container):
    """把容器配成统一的三列表单: 标签 | 等宽输入 | 说明文字。

    只有第三列可伸缩, 面板拉宽时输入框不会跟着变形, 跨分组始终等宽对齐。
    """
    container.columnconfigure(0, minsize=FORM_LABEL_W, weight=0)
    container.columnconfigure(1, minsize=FORM_INPUT_W, weight=0)
    container.columnconfigure(2, weight=1)


def _form_label(parent, text, row, column=0, columnspan=1,
                style="Surface.TLabel"):
    """表单标签: 统一右侧留白与行距。"""
    widget = ttk.Label(parent, text=text, style=style)
    widget.grid(row=row, column=column, columnspan=columnspan, sticky="w",
                padx=(0, FORM_LABEL_GAP), pady=FORM_ROW_PADY)
    return widget


def _form_input(widget, row, column=1, columnspan=1, sticky="ew"):
    """表单输入: 贴住输入列左边缘, 宽度由列宽统一。

    多控件组合 (单选组、带勾选框的行) 传 columnspan=2 + sticky="w",
    让它们向右溢出到说明列, 而不是把输入列撑宽。
    """
    widget.grid(row=row, column=column, columnspan=columnspan,
                sticky=sticky, pady=FORM_ROW_PADY)
    return widget


def _form_hint(parent, row, text, column=2, columnspan=1):
    """输入框右侧的浅色说明文字。"""
    widget = ttk.Label(parent, text=text, style="SurfaceMuted.TLabel")
    widget.grid(row=row, column=column, columnspan=columnspan, sticky="w",
                padx=(FORM_HINT_GAP, 0), pady=FORM_ROW_PADY)
    return widget


def _form_separator(parent, row, columnspan=3):
    """分组内的轻分割线: 上下留白一致。"""
    widget = ttk.Separator(parent, orient="horizontal")
    widget.grid(row=row, column=0, columnspan=columnspan, sticky="ew",
                pady=(FORM_ROW_PADY + 4, FORM_ROW_PADY + 2))
    return widget


# matplotlib 整体风格配置 (与 Tk 主题协调)
plt.rcParams['axes.facecolor']   = PALETTE["surface"]
plt.rcParams['figure.facecolor'] = PALETTE["surface"]
plt.rcParams['axes.edgecolor']   = PALETTE["border"]
plt.rcParams['axes.labelcolor']  = PALETTE["text"]
plt.rcParams['xtick.color']      = PALETTE["text_muted"]
plt.rcParams['ytick.color']      = PALETTE["text_muted"]
plt.rcParams['axes.titlecolor']  = PALETTE["text"]
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['grid.color']       = PALETTE["border_soft"]
plt.rcParams['grid.linestyle']   = '--'
plt.rcParams['grid.linewidth']   = 0.6
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# 确保 pricing 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing import (
    Option_AB, Option_AS, Option_DE, Option_SNB, Option_Vanilla, HedgeBacktest,
    CloseToCloseStrategy, FixedTimeStrategy,
    HedgeBandStrategy, StrategyCase, ContractHistoryPool,
    compare_strategies, result_daily_frame,
    summarize_strategy_result, history_window_summary,
    history_replay_index,
)
from pricing.constants import ANNUAL_DAYS
from pricing.hedge_analysis import (
    LOOKBACK_DAYS,
    _aggregate_result_by_day,
    DEFAULT_SELECTION_OBJECTIVE,
    SELECTION_OBJECTIVES,
    HistoryReplaySpec,
    rerank_history,
    recommend_by_contract_history_pool,
    recommend_by_rolling_history,
)
# 载入结果包时按冻结参数重建每段的期权：原始运行也是按段初价重定基的，
# 用同一个函数才能保证重放与排名同源。
from pricing.hedge_backtest import (
    _PRICE_FIELDS_BY_CLS,
    _rescale_option_to_real_s0,
)
from pricing.hedge_backtest import (
    _infer_intraday_steps,
    _rescale_strategy_to_real_s0,
    _validate_fixed_time_data,
    histogram_bin_edges,
    padded_histogram_range,
)
import history_selection
from history_selection import (
    DEFAULT_BAND_CANDIDATE_SIGMAS,
    DEFAULT_FIXED_TIMES,
    HISTORY_PERIOD_DEFS,
    MAX_BAND_CANDIDATES,
    MAX_HISTORY_CHART_CANDIDATES,
)
import history_store
import history_bar_cache
import backtest_pool_store

# ============================================================
#  期权类型注册表
# ============================================================

def _snowball_ko_observ(T, first_obs, period):
    """按"锁定期 + 固定间隔 + 末次=到期"生成敲出观察交易日序号（1-based）。

    全用交易日：首个观察在第 first_obs 日，其后每 period 个交易日一次，并
    强制最后一次落在到期日 T（与到期不齐时末段为短桩）。返回升序去重列表。
    """
    T = int(T)
    first_obs = max(1, int(first_obs))
    period = max(1, int(period))
    days = list(range(first_obs, T + 1, period))
    if not days or days[-1] != T:
        days.append(T)                         # 末次观察 = 到期
    return sorted({d for d in days if 1 <= d <= T})


def _build_snowball(st, p):
    """构造雪球（act=1 交易日计息）。观察日按锁定期+固定间隔生成（按完整交易日计算）；
    ko_step>0 时为降敲，KO 自期初值每观察日递减 ko_step 个点，得逐观察日 KO 向量。"""
    T = int(p["T"])
    ko_observ = _snowball_ko_observ(T, p["first_obs_d"], p["obs_period_d"])
    step = float(p.get("ko_step", 0.0) or 0.0)
    KO = [p["KO"] - step * i for i in range(len(ko_observ))] if step else p["KO"]
    return Option_SNB(
        st, p["s00"], p["s0"], p["K"], p["KI"], KO, T,
        p["sigma"], p["coupon"], p["coupon_ko"], p["margin"], 1, p["cp"],
        r=p["r"], q=p["q"], sr=[], ko_observ=ko_observ, nPath=p["nPath"],
        margin_call=bool(p["margin_call"]),
    )


OPTION_CLASSES = {
    "香草期权 (Vanilla)": {
        "class": Option_Vanilla,
        "subtypes": ["Eu"],
        "params": [
            ("s0",     "初始价格 S0",    float, 100.0),
            ("K",      "行权价",        float, 100.0),
            ("T_days", "期限(交易日)",   int,   22),
            ("sigma",  "波动率",        float, 0.18),
            ("cp",     "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
        ],
        "build": lambda st, p: Option_Vanilla(
            st, p["s0"], [], p["K"], p["T_days"],
            p["sigma"], p["cp"],
            r=p["r"], q=p["q"], exe_mode=st,
        ),
    },
    "累计期权 (Decumulator)": {
        "class": Option_DE,
        "subtypes": [
            "Opt_Decumulator", "Opt_Decumulator_Back",
            "Opt_Decumulator_Fix", "Opt_Decumulator_Fix_E",
            "Opt_EnDecumulator", "Opt_EnDecumulator_Fix",
            "Opt_ASGQ_call_put", "Opt_ASGQ_EP", "Opt_ASGQ_EF", "Opt_ASGQ_EFF",
            "Opt_ASGQ_DP", "Opt_ASGQ_DF", "Opt_ASGQ_DFF",
        ],
        "params": [
            ("s0",     "初始价格 S0",    float, 100.0),
            ("K",      "行权价",        float, 90.0),
            ("T_days", "剩余期限(交易日)", int, 20),
            ("T_over", "已过天数",       int,   0),
            ("sigma",  "波动率",        float, 0.18),
            ("H",      "障碍价格",       float, 110.0),
            ("N",      "杠杆倍数",       int,   2),
            ("cp",     "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("fix",    "固定赔付(可选)",  float, 0.0),
            ("P",      "保障价格(可选)",  float, 0.0),
            ("amount", "固定金额(可选)",  float, 0.0),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
            ("nPath",  "定价路径数 (MC)", int,   100000),
        ],
        "build": lambda st, p: Option_DE(
            st, p["s0"], [], p["K"], p["T_over"], p["T_days"],
            list(range(1, p["T_days"] + p["T_over"] + 1)),
            p["sigma"], p["H"], p["N"], p["cp"],
            r=p["r"], q=p["q"], nPath=p["nPath"],
            fix=p["fix"] if p["fix"] else None,
            P=p["P"] if p["P"] else None,
            amount=p["amount"] if p["amount"] else None,
        ),
    },
    "亚式期权 (Asian)": {
        "class": Option_AS,
        "subtypes": ["Asian", "EnhanceAsian"],
        "params": [
            ("s0",     "初始价格 S0",    float, 100.0),
            ("K",      "行权价",        float, 100.0),
            ("E",      "增强价(Enhanced)", float, 100.0),
            ("T",      "期限(交易日)",   int,   22),
            ("N",      "观察日数",       int,   22),
            ("sigma",  "波动率",        float, 0.15),
            ("cp",     "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("minPay", "最低赔付",       float, 0.0),
            ("maxPay", "最高赔付",       float, 999999.0),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
            ("nPath",  "定价路径数 (MC)", int,   100000),
        ],
        "build": lambda st, p: Option_AS(
            st, p["s0"], [], p["K"], p["E"], p["T"], p["N"],
            p["sigma"], p["cp"], p["minPay"], p["maxPay"],
            r=p["r"], q=p["q"], nPath=p["nPath"]
        ),
    },
    "气囊期权 (Airbag)": {
        "class": Option_AB,
        "subtypes": ["Opt_Airbag"],
        "params": [
            ("s0",    "初始价格 S0",    float, 100.0),
            ("K",     "行权价",        float, 100.0),
            ("KI",    "敲入价",        float, 90.0),
            ("T_days","期限(交易日)",   int,   20),
            ("sigma", "波动率",        float, 0.18),
            ("pr",    "参与率",        float, 0.8),
            ("pr_ki", "敲入参与率",     float, 1.0),
            ("cp",    "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("r",     "无风险利率",     float, 0.03),
            ("q",     "分红率",        float, 0.03),
            ("nPath", "定价路径数 (MC)", int,   100000),
        ],
        "build": lambda st, p: Option_AB(
            st, p["s0"], [], p["K"], p["KI"], p["T_days"],
            list(range(1, p["T_days"] + 1)),
            p["sigma"], p["pr"], p["pr_ki"], p["cp"],
            r=p["r"], q=p["q"], nPath=p["nPath"]
        ),
    },
    "雪球期权 (Snowball)": {
        "class": Option_SNB,
        "subtypes": ["Opt_Snowball"],
        "params": [
            ("s00",        "入场价 S00",        float, 100.0),
            ("s0",         "最新价 S0",         float, 100.0),
            ("K",          "行权价",            float, 100.0),
            ("KI",         "敲入价",            float, 80.0),
            ("KO",         "期初敲出价",         float, 103.0),
            ("T",          "剩余期限(交易日)",    int,   243),
            # 锁定期/观察间隔：值用交易日(与引擎一致)，下拉给月度预设辅助输入，
            # 可编辑——既能选预设也能手填自定义交易日数（21 交易日 ≈ 1 个月）。
            ("first_obs_d","首次敲出观察",        int,   63,
             {"锁1月 (21)": 21, "锁2月 (42)": 42, "锁3月 (63)": 63, "锁6月 (126)": 126},
             {"editable": True}),
            ("obs_period_d","观察间隔",          int,   21,
             {"月度 (21)": 21, "双月 (42)": 42, "季度 (63)": 63, "半年 (126)": 126},
             {"editable": True}),
            ("ko_step",    "每期降敲(点,0=平敲)", float, 0.0),
            ("sigma",      "波动率",            float, 0.15),
            ("coupon",     "未敲出票息率(年化)",  float, 0.15),
            ("coupon_ko",  "敲出票息率(年化)",    float, 0.15),
            ("margin_call", "保证金模式",          int,   1, {"追保(亏损不封顶)": 1, "不追保(有限亏损)": 0}),
            ("margin",     "保证金比例(不追保封顶)", float, 0.2),
            ("cp",         "方向",             int,   -1, {"雪球 (卖看跌)": -1, "反雪球 (卖看涨)": 1}),
            ("r",          "无风险利率",        float, 0.03),
            ("q",          "分红率",            float, 0.03),
            ("nPath",      "定价路径数 (MC)",    int,   20000),
        ],
        # act 固定为 1（交易日计息，无需交易日历）；观察日=锁定期+固定间隔+末次到期，
        # ko_step>0 时为降敲（逐观察日 KO 递减），见 _build_snowball。
        "build": _build_snowball,
    },
}


# ============================================================
#  GUI 显示名 ↔ 后端内部键 映射
#  说明：后端 (hedge_backtest / Option_* 类) 使用英文/方法名做字符串匹配，
#  这里仅影响界面显示；读取 Combobox 值后需通过 *_FROM_DISPLAY 反向映射
#  还原为内部键再传给后端。
# ============================================================

SUBTYPE_DISPLAY = {
    "Eu":                    "欧式",
    "Opt_Decumulator":       "普通累计",
    "Opt_Decumulator_Back":  "回归累计",
    "Opt_Decumulator_Fix":   "固定赔付回归累计",
    "Opt_Decumulator_Fix_E": "固赔到期结算累计",
    "Opt_EnDecumulator":     "增强回归累计",
    "Opt_EnDecumulator_Fix": "固定赔付增强累计",
    "Opt_ASGQ_call_put":     "到期熔断保障累计",
    "Opt_ASGQ_EP":           "熔断每日保障累计",
    "Opt_ASGQ_EF":           "熔断每日固赔累计",
    "Opt_ASGQ_EFF":          "熔断每日双固赔累计",
    "Opt_ASGQ_DP":           "每日熔断保障累计",
    "Opt_ASGQ_DF":           "每日熔断固赔累计",
    "Opt_ASGQ_DFF":          "每日熔断双固赔累计",
    "Asian":                 "标准亚式",
    "EnhanceAsian":          "增强亚式",
    "Opt_Airbag":            "气囊",
    "Opt_Snowball":          "雪球",
}
SUBTYPE_FROM_DISPLAY = {v: k for k, v in SUBTYPE_DISPLAY.items()}

# 已保存结果列表要显示子类型的中文名。history_store 保持纯逻辑、不反
# 向依赖 GUI，所以由这里把映射注入进去。
history_store.SUBTYPE_DISPLAY = SUBTYPE_DISPLAY

STRATEGY_DISPLAY = {
    "close_to_close": "每日收盘",
    "fixed_times": "每日固定时刻",
    "hedge_band": "价格波动触发调仓",
}
STRATEGY_FROM_DISPLAY = {v: k for k, v in STRATEGY_DISPLAY.items()}


# Wind 仍需要明确的起止边界和 BarSize；GUI 用“自动（推荐）”把这组底层
# 参数按单次回测 / 历史择优语义解析后，再传给既有后端 API。
WIND_AUTO_BAR_SIZE = "自动（推荐）"
WIND_BAR_SIZE_OPTIONS = (
    WIND_AUTO_BAR_SIZE, "日频", "60分钟", "30分钟", "15分钟", "5分钟", "1分钟",
)
_WIND_BAR_MINUTES = {
    "60分钟": 60, "30分钟": 30, "15分钟": 15, "5分钟": 5, "1分钟": 1,
}
# 采样粒度只有 WIND_BAR_SIZE_OPTIONS 一套标签；自动推荐与 Wind 请求参数
# 都从 _WIND_BAR_MINUTES 派生，避免下拉改名后留下对不上的字面量。
_WIND_FIXED_TIME_BAR_LABELS = ("15分钟", "5分钟", "1分钟")
# 价格波动触发只比较 bar 收盘价（HedgeBandStrategy.should_hedge），bar 内
# 穿带后回落的行情完全不可见，且漏掉的方向是单边的：少调仓 -> 低估交易
# 成本 -> 高估策略表现。因此自动粒度一律取最细的 1 分钟，不按带宽分档：
# 分档会让同一个候选的评分依赖“本批次里最窄的候选是谁”，而宽带下省下
# 的数据量也换不到可测量的精度。带宽与粒度的关系只用于量化手动选粗时
# 的代价，见 _WIND_BAND_MISS_RATES。
_WIND_BAND_BAR_LABEL = "1分钟"
_WIND_DATE_BUFFER_DAYS = 21

# 进程内交易日历缓存：("ok", [date...]) | ("error", 原因)。
_LOCAL_TRADING_CALENDAR = None


def _local_trading_calendar():
    """返回本地交易日历（升序 ``datetime.date`` 列表）与解析状态。

    单次回测的建仓日必须落在真实交易日上，``_calendar_span_for_trading_days``
    那套“自然日 + 节假日缓冲”只能用来估计取数区间，不能用来定位某一天。

    解析结果（含失败原因）在进程内只算一次：日期提示挂在 StringVar trace 上，
    每次击键都会调用，不能每次都去读文件，更不能反复触发联网刷新。
    """
    global _LOCAL_TRADING_CALENDAR
    if _LOCAL_TRADING_CALENDAR is None:
        try:
            from pricing.trade_calendar import load_calendar
            days = load_calendar().astype("datetime64[D]").astype(object)
            _LOCAL_TRADING_CALENDAR = ("ok", list(days))
        except Exception as exc:      # 缺日历文件且联网刷新也失败
            _LOCAL_TRADING_CALENDAR = (
                "error", str(exc) or exc.__class__.__name__)
    return _LOCAL_TRADING_CALENDAR

# 排名依据的中英映射。两个口径都相对“日内不动”的基线取增量，区别只在于
# 看绝对多赚了多少，还是看这份增量的信噪比（每单位波动换来多少增量）。
HISTORY_OBJECTIVE_DISPLAY = {
    "incremental_pnl": "增量收益（赚更多）",
    "incremental_sharpe": "增量信噪比（更稳）",
}
HISTORY_OBJECTIVE_FROM_DISPLAY = {
    value: key for key, value in HISTORY_OBJECTIVE_DISPLAY.items()
}
# 表格里对应两个排名口径的列，其表头可点击切换排名依据。
_OBJECTIVE_COLUMN_KEYS = {
    "incremental_pnl": "incremental_pnl",
    "incremental_sharpe": "incremental_sharpe",
}
# ranking 里承载这两个口径的列名。品种池模式不产出它们（跨合约不能直接
# 把金额 PnL 相加），展示层据此降级——见 _build_history_metric_tree。
_OBJECTIVE_RANKING_COLUMNS = (
    "incremental_pnl_vs_c2c",
    "incremental_sharpe_vs_c2c",
)

# 图表口径固定为整段接续（模式 "full"），不再有模式下拉，因此这里只留
# 指标的显示名。history_selection 的模型层仍实现着 single / typical，供
# 直接调用，但界面不再提供入口——它们与排名口径不一致。
HISTORY_CHART_METRIC_DISPLAY = {
    "net": "净损益",
    "gross": "成本前损益",
    "tc": "交易成本",
}
HISTORY_CHART_METRIC_FROM_DISPLAY = {
    value: key for key, value in HISTORY_CHART_METRIC_DISPLAY.items()
}

# 结果对比页与策略优选页此前各有一套曲线配色，同一个策略在两页会拿到不同
# 颜色，用户无法把两页的曲线对应起来。两页现在共用这一份色表和标记表，由
# BacktestApp._strategy_style 按会话内首次出现顺序登记。
STRATEGY_CHART_COLORS = (
    "#2563EB", "#D97706", "#7C3AED", "#0F766E",
    "#DB2777", "#DC2626", "#0891B2", "#65A30D",
    "#9333EA", "#C2410C", "#4F46E5", "#047857",
)
STRATEGY_CHART_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
# 同一策略的多条结果共用**色相**（跨页同色是有意的），组内靠明度加线型分开。
# 换区间、换行情、换方向重跑同一个策略恰恰是结果对比页的头号用途，不分开的话
# 一勾多条就是一团完全同色的线。
STRATEGY_CHART_DASHES = (
    "solid", (0, (5, 2)), (0, (1, 1.4)), (0, (7, 2, 1.5, 2)))
# 组内明度档：原色 / 提亮 / 压暗。正数往白里混，负数往黑里压。
#
# **只要三档**是因为线型有四种，3 与 4 互质，(明度, 线型) 的组合到第 12 条才
# 重复，而图上最多同时画 8 条（MAX_COMPARISON_CHART_CURVES）——够用了。档数少
# 才调得开：实测六种基色下三档的最小亮度差是 34，摊成八档只剩 8，那种深浅在
# 691×226 px 的画布上根本读不出来。相邻两条因此明度与线型必定同时不同。
#
# 第 0 档恒为原色：跨页对照靠同一策略的第一条与策略优选页严格同色。
STRATEGY_CHART_SHADES = (0.0, 0.34, -0.40)
# 对比图同时画的曲线上限。指标表不受限——20 行照排照导，「全选 → 点列头
# 排序 → 看第一行」这条路要留着；闸门只加在图上：691×226 px 的画布、12 色取
# 模，20 条叠上去谁也认不出谁。数字与策略优选页的
# MAX_HISTORY_CHART_CANDIDATES 取齐，两页对"几条还看得清"该给同一个答案。
MAX_COMPARISON_CHART_CURVES = 8
# 每日收盘在两页都是固定基准，必须始终占用同一个颜色位，不能因为登记顺序
# 不同而换色。
BASELINE_STRATEGY_STYLE_KEY = "close_to_close"

# 快照来源：只有 origin 是结构化字段，可供结果池分组与回跳使用；此前来源
# 只体现在用户可改的结果名前缀里，改名即丢失。
SNAPSHOT_ORIGIN_MANUAL = "manual"
SNAPSHOT_ORIGIN_HISTORY_REPLAY = "history_replay"
SNAPSHOT_ORIGIN_DISPLAY = {
    SNAPSHOT_ORIGIN_MANUAL: "手工回测",
    SNAPSHOT_ORIGIN_HISTORY_REPLAY: "分段重放",
}


@dataclass
class SavedBacktestResult:
    """回测结果池里的轻量快照：只保存对比所需字段，外加一份重放配方。

    展示层（``summary_row`` / ``daily_frame`` / 四个签名 key / ``form_state``）
    恒可用，重开程序后画曲线、算差异、应用策略都只读它。``replay`` 是可选
    的第二层，存的是**输入**（行情切片 + 冻结的运行参数），供「加载明细」
    重跑出 bar 级结果——bar 级**输出**平均 971 KB，那个不存。
    """

    result_id: str
    name: str
    saved_at: datetime.datetime
    summary_row: dict
    daily_frame: object
    strategy_label: str
    parameter_summary: str
    source_label: str
    option_label: str
    path_key: tuple
    # 四组可比属性的签名，都保留键名，因此能逐字段 diff 出"差在哪一项"。
    # path_key 是价格数组的哈希，只能判"数据变没变"，定位不了原因。
    market_key: tuple
    contract_key: tuple
    economics_key: tuple
    position: int
    # 跨页配色键：曲线名是用户可改的结果名，不能用来对齐策略优选页的颜色。
    style_key: str = BASELINE_STRATEGY_STYLE_KEY
    # 结构化来源，不随重命名丢失；origin_meta 记录优选周期、批次与当时排名。
    origin: str = SNAPSHOT_ORIGIN_MANUAL
    origin_meta: dict = field(default_factory=dict)
    # 本次运行的对冲策略输入，按原始单位保存，供回填左侧表单继续调参。
    # parameter_summary 是给人看的一句话，回不了表单。
    form_state: dict = field(default_factory=dict)
    # 保存顺序的权威来源。对比表原先按 result_id 字典序排，等于把「顺序」
    # 焊死在 ID 格式上：改 ID 方案会静默乱序，第 10000 条也会排到 9999 前。
    sequence: int = 0
    # 重放配方；缺失（旧包、或行情拿不到）时「加载明细」置灰而不是假装成功。
    replay: dict = field(default_factory=dict)
    # 下拉里选的那个 bar 粒度原值（可能是「自动（推荐）」）。纯展示字段，
    # **不进任何签名**：market_key 里的 wind_bar_size 已经是解析后的实际粒度，
    # 签名要的是"实际跑的是什么"。把它加进 market_key 会改变签名的键集合，
    # 于是新旧快照混选时 _differing_field_names 两侧键集合不同、整体退回属性
    # 级——差异比对会被这个纯展示需求悄悄削弱一档。
    bar_size_requested: str = ""
    # 引擎实际用的每交易日 bar 数。同样是纯展示字段、同样不进签名：
    # economics_key 里的 steps_per_day 记的是 GUI **传进去**的值，而真实行情
    # 下那是占位 1（见 _gui_steps_per_day），引擎自己再从时间索引推断。0 表示
    # 本字段上线前保留的快照没有记。
    intraday_steps: int = 0
    # 该快照在磁盘上的位置，载入时回填；重命名原地覆盖、删除据此删文件。
    store_path: str = ""

SIGMA_SOURCE_DISPLAY = {
    "implied":  "隐含波动率",
    "realized": "已实现波动率",
}
SIGMA_SOURCE_FROM_DISPLAY = {v: k for k, v in SIGMA_SOURCE_DISPLAY.items()}


# ============================================================
#  期权结构说明文档
# ============================================================

STRUCTURE_DOCS = {
    ("香草期权 (Vanilla)", "Eu"): (
        "【欧式香草期权】\n"
        "• Payoff: Call max(S_T−K,0) / Put max(K−S_T,0)\n"
        "• 定价: Black-Scholes 封闭解\n\n"
        "风险特征:\n"
        "  Delta 单调 0→1 (call) 或 −1→0 (put)\n"
        "  Gamma 集中于 ATM (S≈K), 随 T 缩小放大\n"
        "  Vega 对 ATM 最敏感, 随 √T 增长\n"
        "  Theta 为买方持续付出的时间价值"
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator"): (
        "【普通累计 Opt_Decumulator】\n"
        "每日观察 + 每日结算, 敲出即终止存续:\n"
        "  • 首次 S ≥ H (敲出): 当日及之后均停止累计\n"
        "  • K < S < H       :  1 倍 (S − K) 结算\n"
        "  • S ≤ K           :  N 倍杠杆 (S − K) 结算\n\n"
        "与 Back 的差异: Back 敲出仅当日计 0、后续仍继续观察;\n"
        "本结构敲出即彻底了结, 路径依赖更强."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Back"): (
        "【回归累计 Opt_Decumulator_Back】\n"
        "每日观察 + 每日结算, 三段式 cashflow:\n"
        "  • S ≥ H  (敲出障碍):  当日 0 赔付\n"
        "  • K < S < H       :  1 倍 (S − K) 结算\n"
        "  • S ≤ K           :  N 倍杠杆 (S − K) 结算\n\n"
        "总损益 = 所有观察日折现加总.\n"
        "卖方希望标的震荡于 (K, H) 区间, 触 K 承 N 倍下行."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Fix"): (
        "【固定赔付回归累计 Opt_Decumulator_Fix】\n"
        "结构同 Back, 差异:\n"
        "  • K < S < H 区间按固定金额 `fix` 结算, 而非 (S−K)\n"
        "  • 敲出段/杠杆段逻辑不变\n\n"
        "锁定中间段现金流, 便于账务管理."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Fix_E"): (
        "【固赔到期结算累计 Opt_Decumulator_Fix_E】\n"
        "结构同 Fix, 差异在杠杆段结算方式:\n"
        "  • K ≤ S < H 区间: 每日固定金额 `fix`\n"
        "  • 杠杆段 (S ≤ K): 不每日结算, 仅按到期日收盘价\n"
        "    一次性结算 (S_T − K) × 累计天数 × N\n\n"
        "适合到期一次性交割杠杆腿的固赔回归累计."
    ),
    ("累计期权 (Decumulator)", "Opt_EnDecumulator"): (
        "【增强回归累计 Opt_EnDecumulator】\n"
        "三段式每日结算:\n"
        "  • S ≥ H :  (S − H) 1 倍  (敲出后仍给买方正向收益)\n"
        "  • K < S < H: (S − K) 1 倍\n"
        "  • S ≤ K :  (S − K) N 倍\n\n"
        "相比 Back, 保留敲出后上行收益, 故称'增强'."
    ),
    ("累计期权 (Decumulator)", "Opt_EnDecumulator_Fix"): (
        "【固定赔付增强累计 Opt_EnDecumulator_Fix】\n"
        "  • S ≥ H :  (S − H) 1 倍\n"
        "  • K < S < H: 固定金额 `fix`\n"
        "  • S ≤ K :  (S − K) N 倍"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_call_put"): (
        "【到期观察熔断保障累计 ASGQ_call_put】\n"
        "路径依赖 + 到期一次性结算:\n"
        "  • 若路径曾 S ≥ H (熔断): 熔断日后统一按 (S_T − P)\n"
        "  • 从未熔断: 按 (S_T − K), 若 S_T ≤ K 额外 N 倍\n\n"
        "保障价 P 提供下行软保护, ASGQ = 熔断保障累计."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EP"): (
        "【熔断保障累计(每日结算) ASGQ_EP】\n"
        "  • 未熔断部分: 按日 (S − K) 累加\n"
        "  • 熔断日起  : 每日 (S − P) 结算"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EF"): (
        "【熔断固定赔付累计 ASGQ_EF】\n"
        "  • 未熔断部分: 按日 (S − K) 累加\n"
        "  • 熔断日起  : 每日固定金额 `amount`"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EFF"): (
        "【熔断每日双固赔累计 ASGQ_EFF】\n"
        "到期观察 + 每日结算, 双固定赔付:\n"
        "  • K < S < H (区间): 每日固定金额 `fix`\n"
        "  • S ≤ K           : 每日 (S − K) 1 倍\n"
        "  • 熔断日起        : 每日固定金额 `amount`\n"
        "  • 到期 S_T ≤ K 且未熔断: 额外结算\n"
        "    (S_T − K) × 累计天数 × (N − 1) 杠杆腿"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DP"): (
        "【每日观察熔断保障累计 ASGQ_DP】\n"
        "每日观察 + 每日结算:\n"
        "  • 未熔断: (S − K), S ≤ K 时乘 N 倍\n"
        "  • 熔断日起: 每日 (S − P)\n\n"
        "比到期版对路径更敏感, Delta/Gamma 跳跃更剧烈."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DF"): (
        "【每日观察熔断固定赔付累计 ASGQ_DF】\n"
        "  • 未熔断: (S − K), S ≤ K 时 N 倍\n"
        "  • 熔断日起: 每日固定金额 `amount`"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DFF"): (
        "【每日熔断双固赔累计 ASGQ_DFF】\n"
        "每日观察 + 每日结算, 双固定赔付:\n"
        "  • K < S < H (区间): 每日固定金额 `fix`\n"
        "  • S ≤ K           : (S − K) N 倍\n"
        "  • 熔断日起        : 每日固定金额 `amount`"
    ),
    ("亚式期权 (Asian)", "Asian"): (
        "【亚式期权 Asian】\n"
        "Payoff = clip( mean(S[-N:]) − K,  minPay,  maxPay ) × cp\n\n"
        "  • 取最后 N 个交易日均价与 K 的差额\n"
        "  • minPay / maxPay 限定赔付区间\n"
        "  • 平均化显著降低末日价格风险\n"
        "  • Gamma / Vega 远低于同期限 Vanilla"
    ),
    ("亚式期权 (Asian)", "EnhanceAsian"): (
        "【增强亚式 EnhanceAsian】\n"
        "每日先做价格增强:\n"
        "  • Call: 观察价 = max(S, E)\n"
        "  • Put : 观察价 = min(S, E)\n"
        "再求均值与 K 比较, 并 clip 到 [minPay, maxPay].\n\n"
        "E 提供'每日保底'效果, 提升买方期望."
    ),
    ("气囊期权 (Airbag)", "Opt_Airbag"): (
        "【气囊期权 Opt_Airbag】\n"
        "到期结算, 路径判断是否敲入 KI:\n"
        "  • 未敲入 (Call: min(S) > KI):  pr × max(S_T − K, 0)\n"
        "  • 已敲入                    : pr_ki × (S_T − K)\n\n"
        "小幅下行时买方有软垫保护 (payoff=0 而非负);\n"
        "一旦跌破 KI, 转为线性承担下行, 即'气囊爆掉'."
    ),
    ("雪球期权 (Snowball)", "Opt_Snowball"): (
        "【雪球期权 Opt_Snowball】 (cp=-1 雪球 / cp=1 反雪球)\n"
        "MC 定价, 路径依赖, 四种到期情形:\n"
        "  • 未敲入未敲出: 全期票息 s00 × coupon × 期限\n"
        "  • 敲出 (观察日触 KO): 敲出票息 × 持有期, 提前了结\n"
        "  • 敲入且敲出: 同敲出 (敲出优先)\n"
        "  • 敲入未敲出: 追保=承担完整亏损; 不追保=亏损按 margin × s00 封顶\n\n"
        "敲入逐日监测; 敲出按固定间隔观察 (首次=锁定期后, 末次=到期日).\n"
        "每期降敲(ko_step>0)时 KO 自期初值逐期递减, 越往后越易敲出.\n"
        "卖方 (持有者) 短 vega/gamma、正 theta; 现价 ↗ 近 KO 时\n"
        "Delta 与 Gamma 易出现剧烈跳变 (敲出悬崖)."
    ),
}


# ============================================================
#  主窗口
# ============================================================

class BacktestApp(tk.Tk):

    # `tkinter.Misc.__getattr__` 会把未定义属性转发给 self.tk，对尚未渲染
    # 结果页的实例用 getattr(..., None) 兜底会递归。策略优选结果页按需创建
    # 的状态统一在类级给出 None 默认值。
    _history_pairs_cache = None

    def __init__(self):
        super().__init__()
        self.title("DeltaLab - 期权对冲回测系统")
        self.geometry("1600x1000")
        # 左侧面板已启用垂直滚动, 这里可以给一个更宽容的最小尺寸,
        # 即便在高 DPI / 小分辨率屏幕下也不会裁掉底部按钮.
        self.minsize(1200, 720)
        self.configure(bg=PALETTE["bg"])
        self._active_job = None
        self._saved_backtests = {}
        self._saved_comparison_selection = set()
        self._saved_backtest_sequence = 0
        self._latest_backtest = None
        self._latest_backtest_state = None
        self._latest_retained_result_id = None
        self._latest_history_state = None
        self._latest_history_source_label = None
        # 策略配色登记表：结果对比与策略优选共用，使同一策略跨页同色。
        # 仍是会话级——跨会话的配色稳定性从没承诺过，结果池持久化后同一批
        # 结果重开可能换色，这一点写在 GUI_USAGE 的 6.7 里。
        self._strategy_style_registry = None
        # 载入本机已保存的结果池。放在 _build_ui 之前：计数行与占位符在构建
        # 时就要拿到真实条数，否则要么显示 0、要么得再刷一次。
        self._saved_pool_load_error = ""
        self._load_saved_pool()
        self._apply_window_icon()
        self._setup_styles()
        self._build_ui()
        self._param_entries = {}
        self._on_option_class_change(None)
        self._refresh_history_base_summary()
        self._refresh_history_current_band_label()

    # ---- 窗口图标 ----
    def _apply_window_icon(self):
        ico_path = _resource_path("assets", "deltalab.ico")
        # 带透明边距的图标 (符合 macOS 网格), 直接运行脚本时 Dock 图标不会过大
        padded_path = _resource_path("assets", "deltalab_padded.png")
        png_path = padded_path if os.path.exists(padded_path) else _resource_path("assets", "deltalab.png")

        # Windows: .ico 在任务栏/标题栏表现最佳
        if _SYSTEM == "Windows" and os.path.exists(ico_path):
            try:
                self.iconbitmap(default=ico_path)
                return
            except tk.TclError:
                pass

        # 其它平台 (macOS / Linux) 或 Windows 回退: iconphoto + PNG
        if os.path.exists(png_path):
            try:
                self._icon_photo = tk.PhotoImage(file=png_path)
                self.iconphoto(True, self._icon_photo)
            except tk.TclError:
                pass

    # ---- 样式 ----
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        base_font    = (_UI_FONT_FAMILY, 10)
        small_font   = (_UI_FONT_FAMILY, 9)
        title_font   = (_UI_FONT_FAMILY, 18, "bold")
        subtitle_font = (_UI_FONT_FAMILY, 10)
        header_font  = (_UI_FONT_FAMILY, 10, "bold")
        group_font   = (_UI_FONT_FAMILY, 10, "bold")
        tab_font     = (_UI_FONT_FAMILY, 12, "bold")
        btn_font     = (_UI_FONT_FAMILY, 10)
        run_font     = (_UI_FONT_FAMILY, 11, "bold")

        # 默认选项 (供 tk.* 原生控件继承)
        self.option_add("*Font", base_font)
        self.option_add("*TCombobox*Listbox*Font", base_font)

        # ---- 通用 Frame / Label ----
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("Surface.TFrame", background=PALETTE["surface"])
        style.configure("Card.TFrame",
                        background=PALETTE["surface"],
                        relief="flat", borderwidth=1)

        style.configure("TLabel",
                        background=PALETTE["bg"],
                        foreground=PALETTE["text"],
                        font=base_font)
        style.configure("Surface.TLabel",
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"])
        style.configure("Muted.TLabel",
                        background=PALETTE["bg"],
                        foreground=PALETTE["text_muted"],
                        font=small_font)
        style.configure("SurfaceMuted.TLabel",
                        background=PALETTE["surface"],
                        foreground=PALETTE["text_muted"],
                        font=small_font)
        style.configure("Title.TLabel",
                        background=PALETTE["bg"],
                        foreground=PALETTE["text"],
                        font=title_font)
        style.configure("Subtitle.TLabel",
                        background=PALETTE["bg"],
                        foreground=PALETTE["text_muted"],
                        font=subtitle_font)
        style.configure("Header.TLabel",
                        background=PALETTE["bg"],
                        foreground=PALETTE["text"],
                        font=header_font)
        style.configure("Status.TLabel",
                        background=PALETTE["surface"],
                        foreground=PALETTE["text_muted"],
                        font=small_font,
                        padding=(8, 4))

        # ---- LabelFrame (分组容器) ----
        style.configure("TLabelframe",
                        background=PALETTE["surface"],
                        bordercolor=PALETTE["border"],
                        relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label",
                        background=PALETTE["surface"],
                        foreground=PALETTE["primary"],
                        font=group_font,
                        padding=(4, 0))

        # ---- 输入控件 ----
        # Entry 与 Combobox 的纵向内边距取同一个值, 两者才等高;
        # 横向留 6 让文字不贴边 (Combobox 右侧还要放下箭头, 少 2)。
        style.configure("TEntry",
                        fieldbackground=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["border"],
                        darkcolor=PALETTE["border"],
                        padding=(6, 4))
        style.map("TEntry",
                  bordercolor=[("focus", PALETTE["primary"])],
                  lightcolor=[("focus", PALETTE["primary"])])

        style.configure("TCombobox",
                        fieldbackground=PALETTE["surface"],
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        arrowcolor=PALETTE["text_muted"],
                        padding=(6, 4))
        style.map("TCombobox",
                  fieldbackground=[("readonly", PALETTE["surface"])],
                  bordercolor=[("focus", PALETTE["primary"])],
                  arrowcolor=[("active", PALETTE["primary"])])

        # ---- Radio/Check (背景分两套: Frame 背景 vs Surface 背景) ----
        style.configure("TRadiobutton",
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        font=base_font,
                        focuscolor=PALETTE["surface"])
        style.map("TRadiobutton",
                  background=[("active", PALETTE["surface"]),
                              ("disabled", PALETTE["surface"])],
                  foreground=[("disabled", PALETTE["text_light"]),
                              ("active", PALETTE["primary"]),
                              ("selected", PALETTE["text"])])

        # 专门定制 clam 下的原生 Checkbutton 样式（采用无锯齿 Pixel Art 彻底消除 macOS Retina 拉伸模糊）
        try:
            bg_checked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAZElEQVR4nGP8/PXnfwYKABMlmhkYGBhYYAyjvE8kaTw3iY86LqC/AbfmiJBvALpmrAZgU4QsrpbyhrAL0A3BpRmrATBFME34NON0AbohuDTjNABZEz7NeA0gRjNBA4gBjAOeGwEXWyXhhSE6UgAAAABJRU5ErkJggg=="
            bg_unchecked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAASElEQVR4nGP8/PXnfwYKABMlmhkYGBhYYIynL9+TpFFaXBDVAAYGBgZ1RXGiNN+8/xLOptgLowaMGsDAgJYSkVMYsYBxwHMjAKdYEIYsHHJgAAAAAElFTkSuQmCC"
            self._cb_bg_checked = tk.PhotoImage(data=bg_checked_crisp)
            self._cb_bg_unchecked = tk.PhotoImage(data=bg_unchecked_crisp)
            style.element_create("Bg.Indicator", "image", self._cb_bg_unchecked,
                ("selected", self._cb_bg_checked), border=0, sticky="w")
            
            sf_checked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAY0lEQVR4nGP8////fwYKABMlmhkYGBhYYAy1lDckabw1R4Q6LqC/ATCnk2UAumasBmBThCyOHthYXYBuCC7NWA2AKYJpwqcZpwvQDcGXRnAGIkwToQSGNxaISZ0UJyTGAc+NAJGvJhltVByMAAAAAElFTkSuQmCC"
            sf_unchecked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAQklEQVR4nGP8////fwYKABMlmhkYGBhYYIyb91+SpFFdURzVAGRBQgDZMoq9MGrAqAEMDGgpkdTkzMDAwMA44LkRACG7EL4AADlaAAAAAElFTkSuQmCC"
            self._cb_sf_checked = tk.PhotoImage(data=sf_checked_crisp)
            self._cb_sf_unchecked = tk.PhotoImage(data=sf_unchecked_crisp)
            style.element_create("Surface.Indicator", "image", self._cb_sf_unchecked,
                ("selected", self._cb_sf_checked), border=0, sticky="w")
            
            style.layout("TCheckbutton", [
                ('Checkbutton.padding', {'sticky': 'nswe', 'children': [
                    ('Bg.Indicator', {'side': 'left', 'sticky': ''}),
                    ('Checkbutton.focus', {'side': 'left', 'sticky': '', 'children': [
                        ('Checkbutton.label', {'sticky': 'nswe'})
                    ]})
                ]})
            ])
            style.layout("Surface.TCheckbutton", [
                ('Checkbutton.padding', {'sticky': 'nswe', 'children': [
                    ('Surface.Indicator', {'side': 'left', 'sticky': ''}),
                    ('Checkbutton.focus', {'side': 'left', 'sticky': '', 'children': [
                        ('Checkbutton.label', {'sticky': 'nswe'})
                    ]})
                ]})
            ])
        except Exception:
            pass

        def _config_checkbutton(style_name, bg_color):
            style.configure(style_name,
                            background=bg_color,
                            foreground=PALETTE["text"],
                            font=base_font,
                            focuscolor=bg_color)
            style.map(style_name,
                      background=[("active", bg_color),
                                  ("disabled", bg_color)],
                      foreground=[("disabled", PALETTE["text_light"]),
                                  ("active", PALETTE["primary"]),
                                  ("selected", PALETTE["text"])])

        _config_checkbutton("TCheckbutton", PALETTE["bg"])
        _config_checkbutton("Surface.TCheckbutton", PALETTE["surface"])

        # ---- 按钮 ----
        style.configure("TButton",
                        font=btn_font,
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        focusthickness=0,
                        padding=(10, 6),
                        relief="flat")
        style.map("TButton",
                  background=[("active", PALETTE["border_soft"]),
                              ("pressed", PALETTE["border"])],
                  bordercolor=[("active", PALETTE["primary"])])

        # 行内小按钮 (如 CSV 的「浏览…」): 纵向内边距与输入框对齐, 同高
        style.configure("Field.TButton", padding=(10, 4))

        # 主色按钮 (运行)
        style.configure("Run.TButton",
                        font=run_font,
                        foreground="white",
                        background=PALETTE["primary"],
                        bordercolor=PALETTE["primary"],
                        padding=(14, 8),
                        relief="flat")
        style.map("Run.TButton",
                  background=[("active", PALETTE["primary_hov"]),
                              ("pressed", PALETTE["primary_act"]),
                              ("disabled", "#9CA3AF")],
                  foreground=[("disabled", "#E5E7EB")])

        # 次级按钮 (结构图)
        style.configure("Accent.TButton",
                        font=btn_font,
                        foreground="white",
                        background=PALETTE["accent"],
                        bordercolor=PALETTE["accent"],
                        padding=(10, 6),
                        relief="flat")
        style.map("Accent.TButton",
                  background=[("active", PALETTE["accent_hov"]),
                              ("pressed", PALETTE["accent_hov"]),
                              ("disabled", "#9CA3AF")],
                  foreground=[("disabled", "#E5E7EB")])

        # 幽灵按钮：导航与视图开关这类"不产生结果"的操作。它们和真正的动作
        # 用同一种带边框实心按钮时，一排四个同权重，主行动就被稀释了。去掉
        # 边框、文字转为次要色，悬停才浮出主色底。
        style.configure("Ghost.TButton",
                        font=btn_font,
                        background=PALETTE["surface"],
                        foreground=PALETTE["text_muted"],
                        bordercolor=PALETTE["surface"],
                        lightcolor=PALETTE["surface"],
                        darkcolor=PALETTE["surface"],
                        focusthickness=0,
                        padding=(10, 6),
                        relief="flat")
        style.map("Ghost.TButton",
                  background=[("active", PALETTE["primary_light"]),
                              ("pressed", PALETTE["selected"]),
                              ("disabled", PALETTE["surface"])],
                  foreground=[("active", PALETTE["primary"]),
                              ("disabled", PALETTE["text_light"])],
                  bordercolor=[("active", PALETTE["primary_light"]),
                               ("pressed", PALETTE["selected"])],
                  lightcolor=[("active", PALETTE["primary_light"]),
                              ("pressed", PALETTE["selected"])],
                  darkcolor=[("active", PALETTE["primary_light"]),
                             ("pressed", PALETTE["selected"])])

        # 危险按钮：删数据且不可撤销的那一类。做成红字描边而不是实心红——它
        # 不是主行动，实心红摆在一排常规按钮里会比「运行回测」还抢眼；红字
        # 加悬停浅红底，足够在点下去之前把性质说清楚。
        style.configure("Danger.TButton",
                        font=btn_font,
                        background=PALETTE["surface"],
                        foreground=PALETTE["danger"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["surface"],
                        darkcolor=PALETTE["surface"],
                        focusthickness=0,
                        padding=(10, 6),
                        relief="flat")
        style.map("Danger.TButton",
                  background=[("active", PALETTE["danger_light"]),
                              ("pressed", PALETTE["danger_light"]),
                              ("disabled", PALETTE["surface"])],
                  foreground=[("disabled", PALETTE["text_light"])],
                  bordercolor=[("active", PALETTE["danger"]),
                               ("disabled", PALETTE["border_soft"])],
                  lightcolor=[("active", PALETTE["danger_light"]),
                              ("pressed", PALETTE["danger_light"])],
                  darkcolor=[("active", PALETTE["danger_light"]),
                             ("pressed", PALETTE["danger_light"])])

        # ---- Notebook (现代卡片式 Tab) ----
        style.configure("TNotebook",
                        background=PALETTE["bg"],
                        bordercolor=PALETTE["border_soft"],
                        tabmargins=(4, 8, 4, 0))
        style.configure("TNotebook.Tab",
                        font=tab_font,
                        background=PALETTE["tab_inactive"],
                        foreground=PALETTE["text_muted"],
                        bordercolor=PALETTE["border_soft"],
                        padding=(20, 10, 20, 10),
                        focuscolor=PALETTE["bg"])
        style.map("TNotebook.Tab",
                  background=[("selected", PALETTE["surface"]),
                              ("active", PALETTE["border_soft"])],
                  foreground=[("selected", PALETTE["primary"]),
                              ("active", PALETTE["text"])],
                  bordercolor=[("selected", PALETTE["primary"])],
                  expand=[("selected", (0, 2, 0, 0))])

        # ---- Treeview (更宽松的行距, 更美观的表头) ----
        style.configure("Treeview",
                        background=PALETTE["surface"],
                        fieldbackground=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border_soft"],
                        rowheight=28,
                        font=(_MONO_FONT_FAMILY, 9))
        style.configure("Treeview.Heading",
                        background=PALETTE["primary_light"],
                        foreground=PALETTE["primary"],
                        font=header_font,
                        relief="flat",
                        padding=(8, 8))
        style.map("Treeview.Heading",
                  background=[("active", PALETTE["selected"])])
        style.map("Treeview",
                  background=[("selected", PALETTE["selected"])],
                  foreground=[("selected", PALETTE["primary"])])

        # ---- Scrollbar ----
        style.configure("Vertical.TScrollbar",
                        background=PALETTE["bg"],
                        troughcolor=PALETTE["bg"],
                        bordercolor=PALETTE["bg"],
                        arrowcolor=PALETTE["text_muted"],
                        gripcount=0)
        style.map("Vertical.TScrollbar",
                  background=[("active", PALETTE["border"])])
        style.configure("Horizontal.TScrollbar",
                        background=PALETTE["bg"],
                        troughcolor=PALETTE["bg"],
                        bordercolor=PALETTE["bg"],
                        arrowcolor=PALETTE["text_muted"],
                        gripcount=0)
        style.map("Horizontal.TScrollbar",
                  background=[("active", PALETTE["border"])])

        # ---- Progressbar ----
        style.configure("TProgressbar",
                        background=PALETTE["primary"],
                        troughcolor=PALETTE["border_soft"],
                        bordercolor=PALETTE["border_soft"],
                        lightcolor=PALETTE["primary"],
                        darkcolor=PALETTE["primary"])

        # ---- PanedWindow ----
        style.configure("TPanedwindow", background=PALETTE["bg"])
        style.configure("TPanedwindow.Sash",
                        background=PALETTE["border"],
                        sashthickness=6)

        # ---- Separator ----
        style.configure("TSeparator", background=PALETTE["border"])

    # ---- 界面构建 ----
    def _build_ui(self):
        # 顶部标题条 (带底部分隔线)
        header = ttk.Frame(self)
        header.pack(fill="x", padx=0, pady=0)

        title_row = ttk.Frame(header)
        title_row.pack(fill="x", padx=18, pady=(14, 4))
        ttk.Label(title_row, text="DeltaLab",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(title_row,
                  text="期权动态对冲回测系统",
                  style="Subtitle.TLabel").pack(side="left", padx=(12, 0), pady=(8, 0))

        ttk.Separator(header, orient="horizontal").pack(fill="x", padx=0, pady=(6, 0))

        # 底部状态栏
        # 必须排在 body 之前 pack：pack 按调用顺序分配空间，body 带 expand=True
        # 会先吃满自己的请求高度；一旦右侧图表 + 表格的请求高度超过窗口可用高度
        # （最大化到较矮的屏幕、或策略优选页有图有表时），后 pack 的状态栏就会被
        # 挤出窗口底部只露半行。先占位则状态栏恒为 24px，压缩的是可伸缩的 body。
        status_bar = ttk.Frame(self, style="Surface.TFrame")
        status_bar.pack(fill="x", side="bottom")
        ttk.Separator(status_bar, orient="horizontal").pack(fill="x")
        self._status_var = tk.StringVar(
            value="就绪  |  选择期权类型、设置参数后点击『运行回测』")
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Status.TLabel", anchor="w").pack(fill="x", padx=10)

        # 主体：左侧参数 + 右侧结果
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        # ─── 左侧面板 (整体包一层 Canvas + Scrollbar, 解决低分辨率/高 DPI 下底部按钮被裁) ───
        left_outer = ttk.Frame(body)
        body.add(left_outer, weight=1)

        # 外层 Canvas 横向充满, Scrollbar 靠右
        # width 仅作为 PanedWindow 初始 sash 位置的参考, 不阻止后续缩放;
        # 取表单三列 (标签 + 输入 + 说明) 的自然宽度, 初次打开时右侧说明
        # 文字不会被 sash 截断
        self._left_canvas = tk.Canvas(
            left_outer, highlightthickness=0, bd=0,
            bg=PALETTE["surface"], width=480,
        )
        self._left_scrollbar = ttk.Scrollbar(
            left_outer, orient="vertical", command=self._left_canvas.yview
        )
        self._left_canvas.configure(yscrollcommand=self._left_scrollbar.set)
        self._left_scrollbar.pack(side="right", fill="y")
        self._left_canvas.pack(side="left", fill="both", expand=True)

        # inner frame: 真正承载左侧所有 LabelFrame/按钮; 父控件必须是 Canvas 本身.
        self._left_inner = ttk.Frame(self._left_canvas, style="Surface.TFrame")
        self._left_inner_id = self._left_canvas.create_window(
            (0, 0), window=self._left_inner, anchor="nw"
        )

        # inner 尺寸变化 → 更新 scrollregion
        def _on_inner_configure(event):
            self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))
        self._left_inner.bind("<Configure>", _on_inner_configure)

        # canvas 宽度变化 → inner 跟随横向铺满 (否则 inner 默认是内容宽度, 右侧会有空白)
        def _on_canvas_configure(event):
            self._left_canvas.itemconfigure(self._left_inner_id, width=event.width)
        self._left_canvas.bind("<Configure>", _on_canvas_configure)

        # ─── 鼠标滚轮 ───
        # 旧实现是 <Enter> 时 bind_all、<Leave> 时 unbind_all. 问题在于指针只要
        # 从 Canvas 挪到面板内任意子控件 (LabelFrame / Label / Entry / Combobox…),
        # Canvas 就会收到 <Leave> 把滚轮解绑 —— 而参数区表面几乎全被子控件盖住,
        # 于是表现为「鼠标停在参数上滚轮没反应, 只有压着右侧滚动条才滚得动」
        # (滚动条能滚是 Tk 自带的 Scrollbar 类绑定, 与这里无关).
        #
        # 改成常驻一条全局绑定, 每次事件按**指针位置**判断是否落在左侧面板内:
        # 落在外面就原样放行, 不会劫持右侧图表/表格的滚轮.
        # 用指针位置而不是 event.widget, 是因为 Windows 上 Tk 把 <MouseWheel>
        # 投递给**焦点控件**而非指针下的控件, 只看 event.widget 会判错.
        _self_scrolling = {"Text", "Listbox", "Treeview"}

        def _wheel_over_left_panel(event):
            try:
                widget = self.winfo_containing(event.x_root, event.y_root)
                while widget is not None:
                    if widget is self._left_canvas:
                        return True
                    # 指针下方是自带滚动的控件, 让它自己处理, 不抢事件
                    if widget.winfo_class() in _self_scrolling:
                        return False
                    widget = getattr(widget, "master", None)
            except (KeyError, tk.TclError):
                # 指针不在本程序窗口内 / 控件刚被销毁
                return False
            return False

        def _on_left_mousewheel(event):
            if not _wheel_over_left_panel(event):
                return None
            # Windows: delta 为 ±120 的倍数; macOS: 每格 ±1 (触控板还更小).
            step = -event.delta if _SYSTEM == "Darwin" else -event.delta / 120
            units = int(step)
            if units == 0 and step:
                units = 1 if step > 0 else -1
            self._left_canvas.yview_scroll(units, "units")
            return "break"

        def _on_left_mousewheel_linux(event):
            if not _wheel_over_left_panel(event):
                return None
            self._left_canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
            return "break"

        # 供测试/其它代码直接调用的入口 (指向本平台实际生效的那个处理器)
        self._left_wheel_handler = (
            _on_left_mousewheel_linux if _SYSTEM == "Linux" else _on_left_mousewheel)

        if _SYSTEM == "Linux":
            self.bind_all("<Button-4>", _on_left_mousewheel_linux, add="+")
            self.bind_all("<Button-5>", _on_left_mousewheel_linux, add="+")
        else:
            self.bind_all("<MouseWheel>", _on_left_mousewheel, add="+")

        # ttk 的 Combobox/Spinbox 自带滚轮类绑定, 悬停其上滚动会直接改选中值 ——
        # 参数区里全是这种控件, 既会误改「大类」「子类型」等参数, 又会让人以为
        # 面板滚不动. 覆盖成空处理: 不返回 "break", 事件继续走到上面的全局绑定,
        # 于是滚轮在整个界面里统一是「滚面板」而不是「改值」.
        for _wheel_cls in ("TCombobox", "TSpinbox"):
            self.bind_class(_wheel_cls, "<MouseWheel>", lambda _event: None)
            if _SYSTEM == "Linux":
                self.bind_class(_wheel_cls, "<Button-4>", lambda _event: None)
                self.bind_class(_wheel_cls, "<Button-5>", lambda _event: None)

        # 之后所有原本放在 left 里的控件, 父容器改为 left_inner
        left = self._left_inner

        # 1) 期权大类
        sec1 = ttk.LabelFrame(left, text=" 期权类型 ", padding=FORM_SECTION_PAD)
        sec1.pack(fill="x", pady=(0, FORM_SECTION_GAP))
        _form_grid(sec1)

        _form_label(sec1, "大类:", 0)
        self._class_var = tk.StringVar()
        class_cb = ttk.Combobox(sec1, textvariable=self._class_var,
                                width=FORM_ENTRY_CHARS,
                                values=list(OPTION_CLASSES.keys()), state="readonly")
        _form_input(class_cb, 0)
        class_cb.current(0)
        class_cb.bind("<<ComboboxSelected>>", self._on_option_class_change)

        _form_label(sec1, "子类型:", 1)
        self._subtype_var = tk.StringVar()
        self._subtype_cb = ttk.Combobox(sec1, textvariable=self._subtype_var,
                                        width=FORM_ENTRY_CHARS, state="readonly")
        _form_input(self._subtype_cb, 1)
        self._subtype_cb.bind(
            "<<ComboboxSelected>>", lambda _event: self._schedule_band_reference_sync())

        # 2) 期权参数
        # 说明: 外层左侧已经有整体 Canvas+Scrollbar, 这里不再嵌套独立滚动容器,
        # 让参数区按内容自然撑开高度, 整体滚动由外层统一处理.
        sec2 = ttk.LabelFrame(left, text=" 期权参数 ", padding=FORM_SECTION_PAD)
        sec2.pack(fill="x", pady=(0, FORM_SECTION_GAP))

        self._param_frame = ttk.Frame(sec2, style="Surface.TFrame")
        self._param_frame.pack(fill="x", expand=True)
        _form_grid(self._param_frame)

        # 3) 回测设置
        sec3 = ttk.LabelFrame(left, text=" 回测设置 ", padding=FORM_SECTION_PAD)
        sec3.pack(fill="x", pady=(0, FORM_SECTION_GAP))
        _form_grid(sec3)

        _form_label(sec3, "数据来源:", 0)
        self._source_var = tk.StringVar(value="simulate")
        src_frame = ttk.Frame(sec3, style="Surface.TFrame")
        # 单选组比输入列宽, 让它跨到说明列, 免得把输入列撑变形
        _form_input(src_frame, 0, columnspan=2, sticky="w")
        ttk.Radiobutton(src_frame, text="模拟", variable=self._source_var,
                        value="simulate", command=self._toggle_source).pack(side="left")
        ttk.Radiobutton(src_frame, text="CSV", variable=self._source_var,
                        value="csv", command=self._toggle_source).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(src_frame, text="Wind", variable=self._source_var,
                        value="wind", command=self._toggle_source).pack(side="left", padx=(12, 0))

        # 模拟 / CSV / Wind 三个子面板共用同一套列宽, 切换数据来源时
        # 输入框不会左右跳动。
        # 模拟参数
        self._sim_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sim_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        _form_grid(self._sim_frame)
        _form_label(self._sim_frame, "种子:", 0)
        self._seed_var = tk.StringVar(value="42")
        _form_input(ttk.Entry(self._sim_frame, textvariable=self._seed_var,
                              width=FORM_ENTRY_CHARS), 0)
        _form_label(self._sim_frame, "已实现波动率:", 1)
        self._real_vol_var = tk.StringVar(value="")
        _form_input(ttk.Entry(self._sim_frame, textvariable=self._real_vol_var,
                              width=FORM_ENTRY_CHARS), 1)
        _form_hint(self._sim_frame, 1, "空=同隐含")
        _form_label(self._sim_frame, "回测路径数 (MC):", 2)
        self._npaths_var = tk.StringVar(value="10")
        _form_input(ttk.Entry(self._sim_frame, textvariable=self._npaths_var,
                              width=FORM_ENTRY_CHARS), 2)
        _form_label(self._sim_frame, "每日模拟采样点数:", 3)
        self._spd_var = tk.StringVar(value="1")
        self._spd_combo = ttk.Combobox(
            self._sim_frame, textvariable=self._spd_var, width=FORM_ENTRY_CHARS,
            values=("1", "4", "48", "240"), state="readonly")
        _form_input(self._spd_combo, 3)
        _form_hint(self._sim_frame, 3, "每个交易日等分的采样点数")

        # CSV 参数
        self._csv_frame = ttk.Frame(sec3, style="Surface.TFrame")
        _form_grid(self._csv_frame)
        _form_label(self._csv_frame, "文件:", 0)
        self._csv_path_var = tk.StringVar()
        _form_input(ttk.Entry(self._csv_frame, textvariable=self._csv_path_var,
                              width=FORM_ENTRY_CHARS), 0)
        ttk.Button(self._csv_frame, text="浏览…", width=6,
                   style="Field.TButton", command=self._browse_csv).grid(
            row=0, column=2, sticky="w",
            padx=(FORM_HINT_GAP, 0), pady=FORM_ROW_PADY)
        _form_label(self._csv_frame, "价格列:", 1)
        self._csv_col_var = tk.StringVar(value="close")
        _form_input(ttk.Entry(self._csv_frame, textvariable=self._csv_col_var,
                              width=FORM_ENTRY_CHARS), 1)

        # Wind 参数
        self._wind_frame = ttk.Frame(sec3, style="Surface.TFrame")
        _form_grid(self._wind_frame)
        _form_label(self._wind_frame, "代码:", 0)
        self._wind_code_var = tk.StringVar(value="510050.SH")
        _form_input(ttk.Entry(self._wind_frame, textvariable=self._wind_code_var,
                              width=FORM_ENTRY_CHARS), 0)
        # 建仓日排在最上面：它才是读结果时关心的那一天。勾选倒推时这一格
        # 由「数据截止日」按期权期限沿交易日历往前算出来并回填（置灰只读），
        # 开关紧跟其后说明这格是谁算的；下一行才是作为主控的截止日。
        _form_label(self._wind_frame, "建仓日:", 1)
        _today = datetime.date.today()
        # 截止日默认落在最近一个「已收盘」交易日：当天盘中拉行情，最后一个
        # 交易日组只有半天 bar，分钟粒度的策略会被判成盘中残段而直接拒绝。
        # 这也与历史择优默认截至日（不含当天）保持同一口径。
        _wind_asof_default = BacktestApp._latest_trading_day(
            _today - datetime.timedelta(days=1)).isoformat()
        # 只是取消勾选前的占位：勾选倒推时首次刷新就会用倒推结果覆盖它。
        _wind_start_default = (_today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        self._wind_start_var = tk.StringVar(value=_wind_start_default)
        self._wind_start_entry = ttk.Entry(
            self._wind_frame, textvariable=self._wind_start_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._wind_start_entry, 1)
        self._wind_auto_start_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._wind_frame, text="按数据截止日倒推",
            variable=self._wind_auto_start_var,
            command=self._toggle_wind_auto_start,
            style="Surface.TCheckbutton",
        ).grid(row=1, column=2, sticky="w",
               padx=(FORM_HINT_GAP, 0), pady=FORM_ROW_PADY)

        _form_label(self._wind_frame, "数据截止日:", 2)
        self._wind_end_var = tk.StringVar(value=_wind_asof_default)
        self._wind_end_entry = ttk.Entry(
            self._wind_frame, textvariable=self._wind_end_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._wind_end_entry, 2)
        # 界面上不再单列"实际区间"，两个日期框就是本次真实取数区间，因此
        # 截止日填成休市日时也要落到倒推真正用的那个锚点（不晚于它的最近
        # 交易日）。逐次击键就落值会打断输入，只在提交时对齐。
        self._wind_end_entry.bind(
            "<FocusOut>", lambda _event: self._commit_wind_asof())
        self._wind_end_entry.bind(
            "<Return>", lambda _event: self._commit_wind_asof())

        # 行情采样粒度不再由用户选择：每种策略的正确粒度是唯一确定的
        # （收盘=日频、固定时刻=能覆盖目标时刻的最粗、价格波动=1 分钟），
        # 手动调粗只会静默让结论偏乐观。这里只展示本次将要使用的粒度。
        _form_label(self._wind_frame, "行情采样粒度:", 3)
        self._wind_frequency_hint_var = tk.StringVar()
        # 这一格是只读结论, 不是输入框, 因此贴着输入列起点直接铺开
        ttk.Label(
            self._wind_frame, textvariable=self._wind_frequency_hint_var,
            style="SurfaceMuted.TLabel",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=FORM_ROW_PADY)
        self._toggle_wind_auto_start()
        self._refresh_wind_frequency_hint()
        self._wind_code_var.trace_add(
            "write", lambda *_args: self._refresh_wind_frequency_hint())
        self._wind_end_var.trace_add(
            "write", lambda *_args: self._sync_wind_entry_date())

        # 轻分割线
        _form_separator(sec3, 2)

        # 对冲参数
        # 顺序：成本（成本率 + 滑点）→ 仓位（方向 / 数量 / 乘数）→ 策略。
        # 滑点与成本率是同一类按成交量计的执行成本，排在一起；也因此不能留在
        # 策略块之后——那里的专属参数随策略显隐，会带着滑点上下跳一百多像素。
        row_h = 3
        _form_label(sec3, "交易成本率(%):", row_h)
        self._tc_var = tk.StringVar(value="0.01")
        _form_input(ttk.Entry(sec3, textvariable=self._tc_var,
                              width=FORM_ENTRY_CHARS), row_h)

        row_h += 1
        _form_label(sec3, "滑点 (基点):", row_h)
        self._slip_var = tk.StringVar(value="0")
        _form_input(ttk.Entry(sec3, textvariable=self._slip_var,
                              width=FORM_ENTRY_CHARS), row_h)

        row_h += 1
        _form_label(sec3, "头寸方向:", row_h)
        self._pos_var = tk.StringVar(value="1")
        pos_frame = ttk.Frame(sec3, style="Surface.TFrame")
        _form_input(pos_frame, row_h, columnspan=2, sticky="w")
        ttk.Radiobutton(pos_frame, text="卖出 (Sell)", variable=self._pos_var,
                        value="1").pack(side="left")
        ttk.Radiobutton(pos_frame, text="买入 (Buy)", variable=self._pos_var,
                        value="-1").pack(side="left", padx=(12, 0))

        row_h += 1
        _form_label(sec3, "交易数量:", row_h)
        self._qty_var = tk.StringVar(value="100")
        _form_input(ttk.Entry(sec3, textvariable=self._qty_var,
                              width=FORM_ENTRY_CHARS), row_h)

        row_h += 1
        _form_label(sec3, "合约乘数:", row_h)
        self._mult_var = tk.StringVar(value="5")
        _form_input(ttk.Entry(sec3, textvariable=self._mult_var,
                              width=FORM_ENTRY_CHARS), row_h)
        _form_hint(sec3, row_h, "0=不取整")

        # 轻分割线：高级对冲参数（策略 / intraday / 滑点）
        row_h += 1
        _form_separator(sec3, row_h)

        row_h += 1
        _form_label(sec3, "对冲策略:", row_h)
        self._strategy_var = tk.StringVar(
            value=STRATEGY_DISPLAY["close_to_close"])
        self._strategy_combo = ttk.Combobox(
            sec3, textvariable=self._strategy_var, width=FORM_ENTRY_CHARS,
            values=tuple(STRATEGY_DISPLAY.values()), state="readonly",
        )
        _form_input(self._strategy_combo, row_h)
        self._strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_strategy())

        # 紧跟策略下拉：它只修饰所选策略的触发时点，不是第四种策略。原先排在
        # 固定时刻 / 带宽参数之后，会随策略切换在面板里上下跳。
        # 默认开启：日内触发策略若不在收盘补一次，隔夜会裸露一整晚的
        # Δ 敞口，而回测把这段风险记为零成本，结果偏乐观。
        row_h += 1
        self._force_day_close_hedge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sec3, text="每日收盘保底对冲（重复则跳过）",
            variable=self._force_day_close_hedge_var,
            command=self._refresh_history_base_summary,
            style="Surface.TCheckbutton",
        ).grid(row=row_h, column=1, columnspan=2, sticky="w", pady=FORM_ROW_PADY)

        # sigma 单位下的波动率来源参数；带宽数值统一使用下方输入框。
        row_h += 1
        self._sigma_band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sigma_band_frame.grid(row=row_h, column=0, columnspan=3, sticky="ew")
        _form_grid(self._sigma_band_frame)
        self._k_var = tk.StringVar(value="1")  # 旧状态字段兼容，不再单独展示
        _form_label(self._sigma_band_frame, "波动率来源:", 0)
        self._sigma_src_var = tk.StringVar(value=SIGMA_SOURCE_DISPLAY["implied"])
        _form_input(ttk.Combobox(
            self._sigma_band_frame, textvariable=self._sigma_src_var,
            width=FORM_ENTRY_CHARS,
            values=tuple(SIGMA_SOURCE_DISPLAY.values()), state="readonly"), 0)
        _form_label(self._sigma_band_frame, "历史波动率回看天数:", 1)
        self._sigma_win_var = tk.StringVar(value="20")
        _form_input(ttk.Entry(self._sigma_band_frame,
                              textvariable=self._sigma_win_var,
                              width=FORM_ENTRY_CHARS), 1)

        row_h += 1
        self._fixed_time_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._fixed_time_frame.grid(row=row_h, column=0, columnspan=3, sticky="ew")
        _form_grid(self._fixed_time_frame)
        _form_label(self._fixed_time_frame, "固定时刻:", 0)
        self._fixed_times_var = tk.StringVar(value=DEFAULT_FIXED_TIMES)
        self._fixed_times_entry = ttk.Entry(
            self._fixed_time_frame, textvariable=self._fixed_times_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._fixed_times_entry, 0)
        _form_hint(self._fixed_time_frame, 0, "HH:MM，逗号分隔")

        # 三种带宽单位互为换算，逐行排布才能和上面的参数共用同一条对齐线。
        self._band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._band_frame.grid(row=row_h, column=0, columnspan=3, sticky="ew")
        _form_grid(self._band_frame)
        _form_label(self._band_frame, "绝对间隔:", 0)
        self._band_abs_var = tk.StringVar(value="1")
        self._band_abs_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_abs_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._band_abs_entry, 0)
        _form_label(self._band_frame, "相对间隔:", 1)
        self._band_rel_var = tk.StringVar(value="0.01")
        self._band_rel_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_rel_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._band_rel_entry, 1)
        _form_hint(self._band_frame, 1, "0.01 = 1%")
        _form_label(self._band_frame, "日波动率倍数:", 2)
        self._band_sigma_var = tk.StringVar(value="0.779423")
        self._band_sigma_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_sigma_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._band_sigma_entry, 2)
        _form_hint(self._band_frame, 2, "编辑任一项后自动换算")
        # 策略专属参数是本组最后一行：它随策略显隐，后面不能再排别的控件。

        # 保留原状态字段供收集/兼容；其值由最后编辑的联动输入决定。
        self._price_interval_var = tk.StringVar(value="1")
        self._interval_type_var = tk.StringVar(value="absolute")
        self._band_synced = False
        self._band_syncing = False
        self._band_last_edited = "absolute"
        self._band_reference_after_id = None
        for entry, var, kind in (
                (self._band_abs_entry, self._band_abs_var, "absolute"),
                (self._band_rel_entry, self._band_rel_var, "relative"),
                (self._band_sigma_entry, self._band_sigma_var, "sigma")):
            # StringVar trace 同时覆盖键入、粘贴与辅助输入法；
            # 自动换算期间由 _band_syncing 阻断回入。
            var.trace_add(
                "write", lambda *_args, k=kind: self._mark_band_edited(k))
            entry.bind("<FocusOut>", lambda e, k=kind: self._commit_band_input(k))
            entry.bind("<Return>", lambda e, k=kind: self._commit_band_input(k, force=True))

        # 依据默认策略（close_to_close）初始化专用参数可见性
        self._toggle_strategy()

        # 运行按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(2, 6))
        self._run_btn = ttk.Button(btn_frame, text="▶  运行回测", style="Run.TButton",
                                   command=self._run_backtest)
        self._run_btn.pack(fill="x", ipady=4)

        self._retain_btn = ttk.Button(
            btn_frame, text="＋  保留当前结果到对比", style="Accent.TButton",
            command=self._retain_current_backtest, state="disabled",
        )
        self._retain_btn.pack(fill="x", ipady=2, pady=(FORM_SECTION_GAP - 2, 0))

        # 左侧不再有「结果对比」按钮：同名标签页是等效入口，切过去时
        # _on_notebook_tab_changed 会补建页面。已保留条数改挂到标签页标题上，
        # 从任何页面都看得见。
        self._struct_btn = ttk.Button(btn_frame, text="📊  绘制结构图",
                                      style="Accent.TButton",
                                      command=self._plot_structure)
        self._struct_btn.pack(fill="x", ipady=2, pady=(FORM_SECTION_GAP - 2, 0))

        struct_ctrl = ttk.Frame(btn_frame)
        struct_ctrl.pack(fill="x", pady=(FORM_SECTION_GAP - 2, 0))
        ttk.Label(struct_ctrl, text="扫描 ±%:", style="Muted.TLabel").pack(side="left")
        self._struct_range_var = tk.StringVar(value="30")
        ttk.Entry(struct_ctrl, textvariable=self._struct_range_var, width=6).pack(
            side="left", padx=(6, 16))
        ttk.Label(struct_ctrl, text="点数:", style="Muted.TLabel").pack(side="left")
        self._struct_npts_var = tk.StringVar(value="31")
        ttk.Entry(struct_ctrl, textvariable=self._struct_npts_var, width=6).pack(
            side="left", padx=(6, 0))

        self._progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self._progress_label = ttk.Label(btn_frame, text="", anchor="center",
                                          style="Muted.TLabel")

        # ─── 右侧面板 ───
        right = ttk.Frame(body)
        body.add(right, weight=2)

        # Notebook for results
        self._nb = ttk.Notebook(right)
        self._nb.pack(fill="both", expand=True)

        # Tab 定义表: (属性名后缀, tab 标题, 占位提示标题, 占位提示副标题)
        _tab_defs = [
            ("summary", " 📋 回测摘要 ",
             "回测摘要", "运行回测后此处将展示详细的盈亏、Greeks (希腊字母) 和波动率统计"),
            ("chart",   " 📈 对冲图表 ",
             "对冲图表", "运行回测后此处将展示标的价格、Delta、Gamma、累计盈亏等图表"),
            ("table",   " 📃 每日明细 ",
             "每日明细",
             "运行回测后此处将只展示触发对冲的建仓、策略、收盘保底与终止记录"),
            ("struct",  " 🔬 结构分析 ",
             "结构分析", "点击左侧『绘制结构图』按钮以生成期权结构的 Greeks (希腊字母) 曲线"),
            ("vol",     " 📊 波动率分析 ",
             "波动率分析", "运行回测后此处将展示隐含波动率与已实现波动率的对比分析"),
            ("dist",    " 🎲 盈亏分布 ",
             "盈亏分布", "在模拟模式下运行多路径回测后，此处将展示蒙特卡洛盈亏分布"),
            ("compare", BacktestApp._COMPARE_TAB_TITLE,
             "已保存回测结果对比",
             "回测完成后点击『保留当前结果到对比』；换策略或参数继续回测，再到此页勾选结果"),
            ("history", " 🎯 策略优选 ",
             "策略优选",
             "使用 CSV / Wind 真实历史行情搜索已勾选周期的领先对冲方案"),
        ]

        for suffix, title, ph_title, ph_desc in _tab_defs:
            tab = ttk.Frame(self._nb, style="Surface.TFrame")
            self._nb.add(tab, text=title)
            setattr(self, f"_{suffix}_tab", tab)

            # 为需要 container 的 tab 创建内容容器
            if suffix not in ("summary", "table"):
                container = ttk.Frame(tab, style="Surface.TFrame")
                container.pack(fill="both", expand=True, padx=4, pady=4)
                setattr(self, f"_{suffix}_container", container)

            # 历史择优拥有固定的配置区和结果区，不使用通用占位页。
            if suffix == "history":
                continue

            # ==== 占位 / 欢迎界面 ====
            placeholder = ttk.Frame(tab, style="Surface.TFrame")
            placeholder.place(relx=0.5, rely=0.45, anchor="center")
            setattr(self, f"_{suffix}_placeholder", placeholder)

            # 大图标
            icon_map = {
                "summary": "📋", "compare": "🆚", "history": "🎯",
                "chart": "📈", "vol": "📊",
                "dist": "🎲", "struct": "🔬", "table": "📃",
            }
            icon_lbl = tk.Label(placeholder, text=icon_map.get(suffix, "📄"),
                                font=(_UI_FONT_FAMILY, 42),
                                bg=PALETTE["surface"], fg=PALETTE["text_muted"])
            icon_lbl.pack(pady=(0, 10))

            title_lbl = tk.Label(placeholder, text=ph_title,
                                 font=(_UI_FONT_FAMILY, 16, "bold"),
                                 bg=PALETTE["surface"], fg=PALETTE["text"])
            title_lbl.pack(pady=(0, 6))

            desc_lbl = tk.Label(placeholder, text=ph_desc,
                                font=(_UI_FONT_FAMILY, 10),
                                bg=PALETTE["surface"], fg=PALETTE["text_muted"],
                                wraplength=360, justify="center")
            desc_lbl.pack(pady=(0, 0))

        # 摘要 Tab 特有: 预创建 Text 控件 (初始隐藏, 占位符可见)
        self._summary_text = tk.Text(
            self._summary_tab, wrap="word",
            font=(_MONO_FONT_FAMILY, 10),
            state="disabled",
            bg=PALETTE["surface_alt"],
            fg=PALETTE["text"],
            relief="flat", borderwidth=0,
            padx=16, pady=14,
            insertbackground=PALETTE["primary"],
            selectbackground=PALETTE["selected"],
            selectforeground=PALETTE["text"],
            spacing1=2, spacing3=2,
        )
        # 定义 Text 样式标签 (用于 _show_summary 中分段上色)
        self._summary_text.tag_configure(
            "header", foreground=PALETTE["primary"],
            font=(_MONO_FONT_FAMILY, 11, "bold"), spacing1=6, spacing3=4)
        self._summary_text.tag_configure(
            "separator", foreground=PALETTE["gold"],
            font=(_MONO_FONT_FAMILY, 10))
        self._summary_text.tag_configure(
            "section", foreground=PALETTE["accent"],
            font=(_MONO_FONT_FAMILY, 10, "bold"), spacing1=4)
        self._summary_text.tag_configure(
            "value_pos", foreground=PALETTE["success"])
        self._summary_text.tag_configure(
            "value_neg", foreground=PALETTE["danger"])
        self._summary_text.tag_configure(
            "label", foreground=PALETTE["text_muted"])
        self._summary_text.tag_configure(
            "monte_header", foreground="#7C3AED",
            font=(_MONO_FONT_FAMILY, 11, "bold"), spacing1=8, spacing3=4)

        self._build_history_workspace()
        self._nb.bind(
            "<<NotebookTabChanged>>", self._on_notebook_tab_changed, add="+")

        self._toggle_source()

    def _build_history_workspace(self):
        """构建独立历史择优工作区，避免占用回测结果对比容器。"""
        container = self._history_container
        for widget in container.winfo_children():
            widget.destroy()

        header = ttk.Frame(container, style="Surface.TFrame")
        header.pack(fill="x", padx=8, pady=(8, 6))
        header.columnconfigure(0, weight=1)
        for _column in range(1, 5):
            header.columnconfigure(_column, weight=0)

        self._history_header_summary_frame = ttk.Frame(
            header, style="Surface.TFrame")
        self._history_header_summary_frame.grid(
            row=0, column=0, sticky="w")

        # 一次五周期全选要跑八十多秒，跑完只能截图——所以给它一个落盘的去
        # 处。「已保存结果」任何时候都可用（要去翻旧结果）；「保存结果」只
        # 在页面上确实有结果时可用。
        # 「已保存结果」是去翻旧结果的入口，「收起/展开候选配置」只是视图
        # 开关——都用幽灵样式，把实心按钮的分量留给「保存结果」和主行动。
        self._history_open_store_btn = ttk.Button(
            header, text="已保存结果", width=11, style="Ghost.TButton",
            command=self._open_history_result_store,
        )
        self._history_open_store_btn.grid(
            row=0, column=1, sticky="e", padx=(12, 0))
        self._history_save_btn = ttk.Button(
            header, text="保存结果", width=10, state="disabled",
            command=self._save_history_result,
        )
        self._history_save_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))

        self._history_config_visible = True
        self._history_config_toggle_btn = ttk.Button(
            header, text="收起候选配置", width=13, style="Ghost.TButton",
            command=self._toggle_history_config_panel,
        )
        self._history_config_toggle_btn.grid(
            row=0, column=3, sticky="e", padx=(6, 0))

        # 变量保留：顶部摘要里的警示 pill 从它取文案。这里不再重复渲染
        # 同一句话——同屏出现两遍只会让人以为是两条不同的提示。
        self._history_source_hint_var = tk.StringVar()
        # 主行动固定在页首：它此前跟在候选配置下方，折叠配置时会随之上跳，
        # 出结果后又被推到滚动区上方，想改参数重跑得先往回滚。
        self._history_btn = ttk.Button(
            header, text="开始策略优选", style="Run.TButton",
            command=self._run_history_recommendation,
        )
        self._history_btn.grid(
            row=0, column=4, sticky="e", padx=(6, 0), ipadx=12, ipady=2)

        self._history_config_panel = ttk.Frame(
            container, style="Surface.TFrame")
        self._history_config_panel.pack(fill="x", padx=8, pady=(0, 6))

        self._history_base_summary_var = tk.StringVar(value="正在读取当前基准配置…")
        base_box = tk.Frame(
            self._history_config_panel,
            bg=PALETTE["primary_light"],
            highlightbackground=PALETTE["border_soft"], highlightthickness=1,
            padx=14, pady=8)
        base_box.pack(fill="x", pady=(0, 8))
        base_summary = tk.Label(
            base_box, textvariable=self._history_base_summary_var,
            bg=PALETTE["primary_light"], fg=PALETTE["primary"],
            font=(_UI_FONT_FAMILY, 9), anchor="w", justify="left")
        base_summary.pack(fill="x")
        BacktestApp._track_wraplength(base_summary)

        settings = ttk.LabelFrame(
            self._history_config_panel,
            text=" 候选空间（仅属于策略优选） ", padding=14)
        settings.pack(fill="x")
        # 面板统一度量。四个输入框此前宽度不一（时刻/带宽 184px，两个日期
        # 107px），左边缘也各走各的，尾部说明跟着错开。这里用一套常量约束：
        # 同宽输入框 + 同宽标签列 => 每一行的输入框和尾部说明各自成列。
        # 标签列宽按字体实测最长标签算，不写死像素，换字体不会错位。
        row_pady = 4
        entry_width = 22
        label_font = tkfont.nametofont("TkDefaultFont")
        label_gap, hint_gap = 8, 12
        param_label_w = max(
            label_font.measure(text) for text in ("时刻候选:", "带宽候选:")
        ) + label_gap
        date_label_w = max(
            label_font.measure(text) for text in ("自定义起始日:", "分析截至日:")
        ) + label_gap
        # 左栏放勾选、右栏放该策略自己的参数：从属关系由「同一行」表达，
        # 比缩进符号更硬——置灰整行时右侧一起变灰，看得出是同一组。此前
        # σ 来源/回看天数单独占一行、看着像全局设置，而它们只在勾选了固定
        # 间隔时才会被读取。
        settings.columnconfigure(1, weight=1)

        self._history_include_close_var = tk.BooleanVar(value=True)
        self._history_include_fixed_times_var = tk.BooleanVar(value=True)
        self._history_include_band_var = tk.BooleanVar(value=True)

        ttk.Label(
            settings, text="参与策略", style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 9, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            settings, text="相关参数", style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 9, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=(18, 0), pady=(0, 4))
        ttk.Separator(settings, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew",
            pady=(0, row_pady + 2))

        self._history_close_baseline_check = ttk.Checkbutton(
            settings, text="每日收盘",
            variable=self._history_include_close_var,
            state="disabled", style="Surface.TCheckbutton",
        )
        self._history_close_baseline_check.grid(
            row=2, column=0, sticky="w", pady=row_pady)
        ttk.Label(
            settings, text="固定基准，其余策略的增量指标都相对它计算，不可取消",
            style="SurfaceMuted.TLabel",
        ).grid(row=2, column=1, sticky="w", padx=(18, 0), pady=row_pady)

        ttk.Checkbutton(
            settings, text="固定时刻",
            variable=self._history_include_fixed_times_var,
            command=self._toggle_history_candidate_controls,
            style="Surface.TCheckbutton",
        ).grid(row=3, column=0, sticky="w", pady=row_pady)
        # 参数行改用 grid：pack 下每行各自紧排，输入框和尾部说明的左边缘由
        # 前面标签的实际文字宽度决定，行与行对不齐。固定列宽后各成一列。
        fixed_row = ttk.Frame(settings, style="Surface.TFrame")
        fixed_row.grid(row=3, column=1, sticky="w", padx=(18, 0), pady=row_pady)
        fixed_row.columnconfigure(0, minsize=param_label_w)
        ttk.Label(
            fixed_row, text="时刻候选:", style="Surface.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self._history_fixed_times_var = tk.StringVar(value=DEFAULT_FIXED_TIMES)
        self._history_fixed_times_entry = ttk.Entry(
            fixed_row, textvariable=self._history_fixed_times_var,
            width=entry_width)
        self._history_fixed_times_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(
            fixed_row, text="每日在这些时刻各调仓一次，HH:MM 英文逗号分隔",
            style="SurfaceMuted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(hint_gap, 0))

        ttk.Checkbutton(
            settings, text="固定间隔",
            variable=self._history_include_band_var,
            command=self._toggle_history_candidate_controls,
            style="Surface.TCheckbutton",
        ).grid(row=4, column=0, sticky="nw", pady=row_pady)
        band_box = ttk.Frame(settings, style="Surface.TFrame")
        band_box.grid(row=4, column=1, sticky="w", padx=(18, 0), pady=row_pady)
        band_row = ttk.Frame(band_box, style="Surface.TFrame")
        band_row.pack(anchor="w", fill="x")
        band_row.columnconfigure(0, minsize=param_label_w)
        ttk.Label(
            band_row, text="带宽候选:", style="Surface.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self._history_band_candidate_sigmas_var = tk.StringVar(
            value=",".join(
                f"{value:g}" for value in DEFAULT_BAND_CANDIDATE_SIGMAS))
        self._history_band_candidate_entry = ttk.Entry(
            band_row, textvariable=self._history_band_candidate_sigmas_var,
            width=entry_width,
        )
        self._history_band_candidate_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(
            band_row, text="日波动 σ 的倍数，最多 10 档，英文逗号分隔",
            style="SurfaceMuted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(hint_gap, 0))

        # 附加勾选与上面的输入框左对齐，不再顶到标签列。
        band_extra = ttk.Frame(band_box, style="Surface.TFrame")
        band_extra.pack(anchor="w", pady=(row_pady + 2, 0))
        self._history_include_current_band_var = tk.BooleanVar(value=True)
        self._history_current_band_label_var = tk.StringVar(
            value="加入当前回测带宽")
        self._history_current_band_check = ttk.Checkbutton(
            band_extra, textvariable=self._history_current_band_label_var,
            variable=self._history_include_current_band_var,
        )
        self._history_current_band_check.pack(anchor="w")
        # σ 恒取左侧输入的波动率，界面上不再给选项：选「历史波动率」会让
        # 带宽随行情伸缩，那是另一条策略维度，而这里的排名只比带宽倍数，
        # 全部候选共用同一个 σ 才可比。引擎侧仍支持 realized。
        #
        # 这句话另起一行而不是跟在勾选框右边：勾选框的文案是变量，勾上时会
        # 带上换算出的 σ 值（「加入当前回测带宽: 0.866025σ（绝对输入换算）」），
        # 实测把右边这句推走 144px——同一句说明在两次渲染间左右横跳。
        ttk.Label(
            band_extra, text="σ 统一取左侧输入的波动率",
            style="SurfaceMuted.TLabel",
        ).pack(anchor="w")

        ttk.Separator(settings, orient="horizontal").grid(
            row=5, column=0, columnspan=2, sticky="ew",
            pady=(row_pady + 6, row_pady + 2))

        # 分析周期不属于任何一个策略，放在分隔线下面单独一行，避免被误读
        # 成上面某个策略的参数。
        period_bar = ttk.Frame(settings, style="Surface.TFrame")
        period_bar.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=row_pady)
        ttk.Label(
            period_bar, text="分析周期:", style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 9, "bold"),
        ).pack(side="left", padx=(0, 10))
        self._history_period_vars = {}
        for period_key, period_label in HISTORY_PERIOD_DEFS:
            variable = tk.BooleanVar(value=True)
            self._history_period_vars[period_key] = variable
            ttk.Checkbutton(
                period_bar, text=period_label, variable=variable,
                command=self._history_period_selection_changed,
                style="Surface.TCheckbutton",
            ).pack(side="left", padx=(0, 12))
        ttk.Label(
            period_bar, text="（可多选）",
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(2, 0))

        self._history_wind_frame = ttk.LabelFrame(
            settings, text=" Wind 严格历史区间（独立于单次回测） ", padding=(12, 8))
        self._history_wind_frame.grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(row_pady + 4, 2))
        # 钉死前两列的宽度，把所有富余给最后一列。否则跨列的提示行一换文案
        # （自动模式一长句、手动模式为空）就会重算列宽，两个日期输入框跟着
        # 左右跳——勾一下「自动推算」就位移，正是这个原因。
        self._history_wind_frame.columnconfigure(0, minsize=date_label_w)
        # 列宽取输入框的实际请求宽度，而不是「字符数 × 字宽」估算——后者
        # 实测比真实渲染宽 28px（198 vs 170），右侧凭空多出一块死区，把后面
        # 的勾选框整体推远。量完即销毁，不给框里留下一个孤儿控件。
        _probe = ttk.Entry(self._history_wind_frame, width=entry_width)
        entry_px = _probe.winfo_reqwidth()
        _probe.destroy()
        self._history_wind_frame.columnconfigure(1, minsize=entry_px)
        self._history_wind_frame.columnconfigure(2, weight=1)
        # 按区间的自然读法排：先起始日、后截止日。自动推算开关紧跟在它所
        # 控制的那个输入框后面，而不是另起一列。
        ttk.Label(
            self._history_wind_frame, text="自定义起始日:",
            style="Surface.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=row_pady)
        history_start_default = (
            datetime.date.today() - datetime.timedelta(days=420)
        ).isoformat()
        self._history_wind_start_var = tk.StringVar(
            value=history_start_default)
        self._history_wind_start_entry = ttk.Entry(
            self._history_wind_frame,
            textvariable=self._history_wind_start_var, width=entry_width,
        )
        self._history_wind_start_entry.grid(
            row=0, column=1, sticky="w", pady=row_pady)

        self._history_wind_auto_start_var = tk.BooleanVar(value=True)
        # 静态文案：往前取多少天、以及为什么多取一天，都在下方提示行里按当
        # 前勾选实时给出，勾选框自己不必再重复一遍口径。
        self._history_wind_auto_start_check = ttk.Checkbutton(
            self._history_wind_frame,
            text="根据周期自动推算起始日",
            variable=self._history_wind_auto_start_var,
            command=self._toggle_history_wind_controls,
        )
        self._history_wind_auto_start_check.grid(
            row=0, column=2, sticky="w", padx=(hint_gap, 0), pady=row_pady)

        ttk.Label(
            self._history_wind_frame, text="分析截至日:",
            style="Surface.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=row_pady)
        history_asof_default = (datetime.date.today() - datetime.timedelta(
            days=1)).isoformat()
        self._history_wind_asof_var = tk.StringVar(
            value=history_asof_default)
        self._history_wind_asof_entry = ttk.Entry(
            self._history_wind_frame,
            textvariable=self._history_wind_asof_var, width=entry_width,
        )
        self._history_wind_asof_entry.grid(
            row=1, column=1, sticky="w", pady=row_pady)
        self._history_wind_hint_var = tk.StringVar()
        # 提示行只占最后一列：跨列会把它的文字宽度算进前两列的需求，文案一
        # 变就挤动上面的输入框。
        ttk.Label(
            self._history_wind_frame,
            textvariable=self._history_wind_hint_var,
            style="SurfaceMuted.TLabel", wraplength=760, justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w",
               pady=(row_pady, 0))
        # 截至日是自动起始日的推算基准，改一个字就要重算回填。
        self._history_wind_asof_var.trace_add(
            "write",
            lambda *_args: (
                BacktestApp._refresh_history_auto_wind_start(self),
                BacktestApp._refresh_history_base_summary(self),
            ),
        )
        self._wind_code_var.trace_add(
            "write",
            lambda *_args: (
                BacktestApp._refresh_history_wind_hint(self),
                BacktestApp._refresh_history_base_summary(self),
            ),
        )

        self._history_results_container = ttk.Frame(
            container, style="Surface.TFrame")
        self._history_results_container.pack(
            fill="both", expand=True, padx=6, pady=(0, 6))
        self._show_empty_history_results()
        self._history_period_selection_changed()
        self._toggle_history_candidate_controls()
        self._sync_history_button_state()
        BacktestApp._update_history_header_summary(self)

    def _update_history_header_summary(self):
        """更新顶部 Header 区域的 Executive Pills 摘要，消除顶部空白。"""
        container = getattr(self, "_history_header_summary_frame", None)
        if container is None:
            return
        try:
            for w in list(container.winfo_children()):
                w.destroy()
        except (tk.TclError, AttributeError):
            return

        # 载入的历史结果必须一眼看得出来：页面允许「只填参数 / 用当前行情
        # 回测」，而这份结论是过去某天算的、行情早已往前走了。不标出来就会
        # 被当成刚跑的结果用。
        loaded = getattr(self, "_history_loaded_meta", None)
        if loaded:
            stamp = str(loaded.get("saved_at", ""))[:16].replace("T", " ")
            name = str(loaded.get("label") or "").strip()
            asof = str(loaded.get("asof") or "").strip()
            text = "📂 载入结果"
            if name:
                text += f"「{name}」"
            if stamp:
                text += f" · 保存于 {stamp}"
            if asof:
                text += f" · 分析截至 {asof}"
            loaded_pill = tk.Label(
                container, text=text,
                bg=PALETTE["primary_light"], fg=PALETTE["primary"],
                font=(_UI_FONT_FAMILY, 9, "bold"), padx=10, pady=3,
            )
            loaded_pill.pack(side="left", padx=(0, 6))

        hint = getattr(self, "_history_source_hint_var", None)
        hint_text = hint.get().strip() if hint is not None else ""
        if hint_text and not loaded:
            pill = tk.Label(
                container, text=f"⚠ {hint_text}",
                bg=PALETTE["warning_light"], fg=PALETTE["warning"],
                font=(_UI_FONT_FAMILY, 9, "bold"), padx=10, pady=3,
            )
            pill.pack(side="left", padx=(0, 6))
            return

        source_label = getattr(self, "_history_last_source_label", None)
        if not source_label and hasattr(self, "_source_var"):
            source = self._source_var.get()
            # 标的必须按来源取：此前一律读 _wind_code_var，于是 CSV 模式下
            # 这枚 pill 会显示 Wind 代码框里的内容（实测「CSV 行情 ·
            # 510050.SH」），而本次根本不会去碰那个代码。
            if source == "wind":
                code_var = getattr(self, "_wind_code_var", None)
                subject = code_var.get().strip() if code_var is not None else ""
                source_label = "WIND 行情" + (f" · {subject}" if subject else "")
            elif source == "csv":
                path_var = getattr(self, "_csv_path_var", None)
                path = path_var.get().strip() if path_var is not None else ""
                subject = os.path.basename(path) if path else ""
                source_label = "CSV 行情" + (f" · {subject}" if subject else "")

        if source_label:
            src_pill = tk.Label(
                container, text=f"📊 {source_label}",
                bg=PALETTE["primary_light"], fg=PALETTE["primary"],
                font=(_UI_FONT_FAMILY, 9), padx=10, pady=3,
            )
            src_pill.pack(side="left", padx=(0, 6))

        # 跨周期一致性只在结果区渲染一枚 pill（紧挨周期切换条，见
        # _build_history_period_box）。这里曾经再放一枚说同一件事的：两枚
        # 相隔约 30px 同屏出现，措辞还略有出入，读者会以为是两条不同的
        # 结论。顶部这条摘要留给「本次跑的是什么」——数据源与排名口径。
        objective_var = getattr(self, "_history_result_objective_var", None)
        if objective_var is not None:
            obj_val = objective_var.get().strip()
            obj_label = HISTORY_OBJECTIVE_DISPLAY.get(obj_val, obj_val)
            obj_pill = tk.Label(
                container, text=f"🆚 {obj_label}",
                bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
                font=(_UI_FONT_FAMILY, 9), padx=10, pady=3,
            )
            obj_pill.pack(side="left", padx=(0, 6))

    def _toggle_history_config_panel(self):
        panel = getattr(self, "_history_config_panel", None)
        button = getattr(self, "_history_config_toggle_btn", None)
        if panel is None or button is None:
            return
        if getattr(self, "_history_config_visible", True):
            panel.pack_forget()
            self._history_config_visible = False
            button.configure(text="展开候选配置")
        else:
            before = getattr(self, "_history_results_container", None)
            pack_options = {
                "fill": "x", "padx": 8, "pady": (0, 5),
            }
            if before is not None:
                pack_options["before"] = before
            panel.pack(**pack_options)
            self._history_config_visible = True
            button.configure(text="收起候选配置")

    def _sync_history_save_button(self):
        """页面上确实有可保存的结果时才放开「保存结果」。"""
        button = getattr(self, "_history_save_btn", None)
        if button is None:
            return
        ranking = getattr(self, "_history_ranking", None)
        ready = ranking is not None and not getattr(ranking, "empty", True)
        try:
            button.configure(state="normal" if ready else "disabled")
        except tk.TclError:
            pass

    def _save_history_result(self):
        """把当前优选结果落盘。

        只存两张表与冻结参数（gzip 后约 137 KB）。**不存 bar 级结果**：
        ``window_results`` 每个回测的完整数组平均 971 KB，一次五周期全选
        161 个约 153 MB。代价是载入后逐段下钻不可用，载入时会明确说明。
        """
        ranking = getattr(self, "_history_ranking", None)
        if ranking is None or getattr(ranking, "empty", True):
            messagebox.showinfo("没有可保存的结果", "请先运行一次策略优选。")
            return None
        label = simpledialog.askstring(
            "保存优选结果",
            "给这份结果起个名字（可留空，仅用于列表里辨认）:",
            initialvalue=history_store.default_label(
                getattr(self, "_history_last_state", None),
                existing=[item["label"]
                          for item in history_store.list_results()]),
            parent=self)
        if label is None:      # 用户取消，与留空不同
            return None
        try:
            payload = history_store.build_payload(
                ranking=ranking,
                window_summary=getattr(self, "_history_window_summary", None),
                history_state=getattr(self, "_history_last_state", None),
                source_label=getattr(self, "_history_last_source_label", None),
                objective=getattr(self, "_history_result_objective", None),
                notes=getattr(self, "_history_last_notes", None),
                label=label,
                replay_index=getattr(self, "_history_replay_index", None),
            )
            # 先写入再淘汰，顺序不能反：反过来会在写失败时白删一份。
            path = history_store.save_result(payload, enforce=False)
            evicted = history_store.enforce_limit()
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("保存失败", str(exc))
            return None
        size_kb = os.path.getsize(path) / 1024
        message = f"已保存优选结果（{size_kb:.0f} KB）: {os.path.basename(path)}"
        # 超出上限时最旧的记录会被删掉。删除不可逆，即使是按规则删的也必
        # 须说一声——那是别人 85 秒跑出来的东西。
        if evicted:
            names = "、".join(
                item["label"] or item["filename"] for item in evicted[:3])
            more = f" 等 {len(evicted)} 份" if len(evicted) > 3 else ""
            message += (
                f"；已达上限 {history_store.MAX_RESULTS} 份，"
                f"淘汰最旧的「{names}」{more}")
        self._set_status(message)
        return path

    def _load_history_result(self, path):
        """载入结果包并渲染。渲染失败不改动当前页面。"""
        try:
            payload = history_store.load_result(path)
        except history_store.HistoryResultVersionError as exc:
            messagebox.showerror("版本不兼容", str(exc))
            return False
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("载入失败", str(exc))
            return False
        ranking = payload.get("ranking")
        if ranking is None or getattr(ranking, "empty", True):
            messagebox.showerror("载入失败", "结果包里没有排名数据。")
            return False
        state = dict(payload.get("history_state") or {})
        # 用包里存的排名口径重新排一次，顺带拿到每周期首选。走公开的
        # rerank_history 而不是自己重算，保证载入结果与新跑结果同一条路径。
        objective = str(payload.get("objective") or "").strip()
        if objective not in SELECTION_OBJECTIVES:
            objective = DEFAULT_SELECTION_OBJECTIVE
        recommendations, ranking = rerank_history(ranking, objective)
        if ranking is None or getattr(ranking, "empty", True):
            messagebox.showerror("载入失败", "结果包里的排名无法重建。")
            return False
        # 载入的结果没有 window_results，因此 details 直接给出已还原的分段
        # 表与一个空的重放索引：逐段下钻会因此不可用，_build_history_replay_bar
        # 会据此说明原因，而不是留一个点了没反应的按钮。
        meta = {
            "path": path,
            "saved_at": str(payload.get("saved_at") or ""),
            "label": str(payload.get("label") or ""),
            "asof": str(state.get("history_wind_asof")
                        or state.get("wind_end") or ""),
        }
        replay_index = self._replay_index_from_payload(payload, ranking)
        meta["replay_available"] = bool(replay_index)
        try:
            BacktestApp._show_history_recommendation(
                self, recommendations, ranking,
                notes=payload.get("notes") or None,
                source_label=payload.get("source_label"),
                window_results=None, history_state=state,
                details=(payload.get("window_summary"), replay_index),
                loaded_meta=meta,
            )
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("载入失败", str(exc))
            return False
        # 载入的这份结果就是当前上下文：把它的冻结状态装回去，逐段下钻的
        # 展示页摘要（期权类型/子类型/数据来源）才有来源，「用当前行情回
        # 测」也才不会拿另一套参数去跑。
        self._latest_history_state = BacktestApp._copy_snapshot_gui_state(
            state)
        self._latest_history_source_label = payload.get("source_label")
        applied = self._apply_state_to_option_form(state)
        stamp = meta["saved_at"][:16].replace("T", " ")
        message = f"已载入优选结果（保存于 {stamp}）"
        if applied:
            message += "；已同步左侧 " + "、".join(applied)
        self._set_status(message)
        return True

    @staticmethod
    def _strategy_from_ranking_row(row, *, wind_code=None):
        """按排名行的元数据重建策略对象；无法识别时返回 None。

        载入的结果包不带策略对象（它不可序列化），但排名行里存着重建它所
        需的全部身份信息——这正是 _apply_history_recommendation 写回左侧表
        单时用的同一批字段。

        ``wind_code`` 用来还原固定时刻策略的交易时段。**这一步不能省**：
        原始运行会用时段过滤掉标的没有的时刻（510050 没有夜盘，23:00 会被
        剔除），不设时段的话重建出来的策略会因为「23:00 在全部交易日组中
        都不存在」直接构造失败。
        """
        row = dict(row or {})
        name = str(row.get("meta_strategy_name")
                   or row.get("strategy_type") or "").strip()
        if name == "close_to_close":
            return CloseToCloseStrategy()
        if name == "fixed_times":
            times = str(row.get("meta_fixed_times") or "").strip()
            if not times:
                return None
            fixed = FixedTimeStrategy(times)
            code = str(wind_code or "").strip()
            if code:
                from pricing.wind_data import (
                    get_trading_session_clock_ranges,
                )
                try:
                    fixed.set_trading_sessions(
                        get_trading_session_clock_ranges(code))
                except Exception:                      # noqa: BLE001
                    # 取不到时段时退回不过滤：仍能重放存在的时刻，
                    # 缺失时刻会由回测自身的校验明确报错。
                    pass
            return fixed
        if name == "hedge_band":
            sigma = BacktestApp._comparison_finite(
                row.get("meta_candidate_sigma"))
            if sigma is None or sigma <= 0:
                return None
            return HedgeBandStrategy(
                band_type="sigma", threshold=float(sigma),
                sigma_source=str(row.get("meta_sigma_source") or "implied"),
                window_days=int(row.get("meta_sigma_window") or 20))
        return None

    @staticmethod
    def _segment_strategies(strategies, summary, lookback, window_id):
        """只保留在该分段**确实成功**的候选。

        实跑路径的 replay_strategies 是在候选于该段成功之后才写入的，所以
        失败的段本来就不会出现在下拉里。载入时若给每段都塞上全部候选，原
        本失败的段也会进下拉，点下去要么撞引擎报错、要么跑出一个从未参与
        排名的数字。包里的 window_summary 带 success 列，据此过滤。
        """
        if summary is None or getattr(summary, "empty", True):
            return dict(strategies)
        try:
            rows = summary[
                (summary["lookback"].astype(str) == str(lookback))
                & (summary["window_id"].astype(str) == str(window_id))
                & summary["success"].astype(bool)
            ]
            ok = {str(name) for name in rows["strategy"]}
        except (KeyError, TypeError):
            return dict(strategies)
        kept = {name: obj for name, obj in strategies.items() if name in ok}
        # 摘要里一个都对不上时不做过滤：宁可多给几项，也别把下拉清空。
        return kept or dict(strategies)

    def _replay_index_from_payload(self, payload, ranking):
        """用包里存的行情切片与构造参数重建逐段重放索引。

        存的是回测**输入**，所以这里不是「读回结果」而是「重建可重跑的配
        方」——缓存不在时据此重算选中那一段。重建路径与原始运行是两套代码，任何
        一处对不上都会让下钻显示的数字与排名不符且**不会报错**，因此每次
        重放都由 ``_replay_fidelity_error`` 拿包里存的逐日损益核对，不符
        时弹窗说明。
        """
        import pandas as pd

        series_by_key, specs = history_store.restore_replay_payload(payload)
        if not series_by_key or not specs:
            return {}
        summary = payload.get("window_summary")
        state = dict(payload.get("history_state") or {})
        try:
            base_option = state["cfg"]["build"](
                state.get("subtype"), state.get("params"))
        except Exception:                              # noqa: BLE001
            # cfg 带回调，不进包；改由注册表按冻结的类型与参数重建。
            base_option = BacktestApp._option_from_state(state)
        if base_option is None:
            return {}
        strategies = {}
        for _index, row in ranking.iterrows():
            item = row.to_dict()
            built = BacktestApp._strategy_from_ranking_row(
                item, wind_code=state.get("wind_code"))
            if built is not None:
                strategies[str(item.get("strategy", "")).strip()] = built
        if not strategies:
            return {}
        index = {}
        failures = []
        for spec in specs:
            # 这里刻意不吞异常：整段 try/except 一裹，重建失败就退化成「下
            # 钻按钮点了没反应」，和功能没做一样却更难查。失败原因收集起来
            # 报给用户。
            try:
                # 按段各切各的序列。单标的只有一条，品种池按合约分桶——拿
                # 错序列就会切到别的合约的价，而且不报错。
                key = str(spec.get("series_key")
                          or history_store.DEFAULT_SERIES_KEY)
                series = series_by_key.get(key)
                if series is None:
                    failures.append(
                        f"{spec['lookback']}/{spec['window_id']}: "
                        f"包内缺少序列 {key}")
                    continue
                path = series.loc[
                    pd.Timestamp(spec["start_ts"]):pd.Timestamp(spec["end_ts"])]
                if not len(path):
                    failures.append(
                        f"{spec['lookback']}/{spec['window_id']}: 行情切片为空")
                    continue
                option, _info = _rescale_option_to_real_s0(
                    base_option, float(path.iloc[0]))
                index.setdefault(str(spec["lookback"]), {})[
                    str(spec["window_id"])] = HistoryReplaySpec(
                        lookback=str(spec["lookback"]),
                        window_id=str(spec["window_id"]),
                        option=option,
                        external_path=path,
                        evaluation_days=int(spec["evaluation_days"]),
                        steps_per_day=int(spec["steps_per_day"]),
                        strategies=BacktestApp._segment_strategies(
                            strategies, summary,
                            spec["lookback"], spec["window_id"]),
                        backtest_kwargs=dict(spec.get("backtest_kwargs") or {}),
                        warmup_kwargs={
                            str(k): dict(v or {})
                            for k, v in dict(
                                spec.get("warmup_kwargs") or {}).items()},
                        metadata=dict(spec.get("metadata") or {}),
                    )
            except Exception as exc:                   # noqa: BLE001
                failures.append(
                    f"{spec.get('lookback')}/{spec.get('window_id')}: "
                    f"{type(exc).__name__} {exc}")
        if failures and not index:
            self._set_status(f"重放配方重建失败：{failures[0]}")
        return index

    @staticmethod
    def _option_from_state(state):
        """按冻结的期权类型与参数重建期权对象。"""
        cls_name = str(state.get("cls_name") or "").strip()
        cfg = OPTION_CLASSES.get(cls_name)
        if cfg is None:
            return None
        try:
            return cfg["build"](state.get("subtype"), dict(
                state.get("params") or {}))
        except Exception:                              # noqa: BLE001
            return None

    def _apply_state_to_option_form(self, state):
        """把冻结状态里的期权与回测设置写回左侧表单。

        载入结果后不同步是有实际危害的，不只是看不见：「用当前行情回测」
        读的是左侧表单，表单没跟上就会拿另一套期权参数去回测，而结论页仍
        挂着这份载入结果的名字——两者对不上且不会报错。

        只写确实存在于状态里的字段；写不进去的静默跳过（旧包可能缺字段，
        缺一项不该让整次载入失败）。
        """
        state = dict(state or {})
        applied = []
        cls_name = str(state.get("cls_name") or "").strip()
        if cls_name in OPTION_CLASSES and hasattr(self, "_class_var"):
            if self._class_var.get() != cls_name:
                self._class_var.set(cls_name)
                # 换大类要重建子类型下拉与参数控件，走界面原有的回调。
                BacktestApp._on_option_class_change(self, None)
            applied.append("期权大类")
        subtype = str(state.get("subtype") or "").strip()
        if subtype and hasattr(self, "_subtype_var"):
            self._subtype_var.set(SUBTYPE_DISPLAY.get(subtype, subtype))
            applied.append("子类型")
        params = dict(state.get("params") or {})
        entries = getattr(self, "_param_entries", {}) or {}
        for key, value in params.items():
            record = entries.get(key)
            if not record:
                continue
            var, _dtype, choices = record
            try:
                if choices:
                    # 下拉项存的是显示名，反查一次。
                    label = next(
                        (text for text, raw in choices.items() if raw == value),
                        None)
                    var.set(label if label is not None else str(value))
                else:
                    var.set(f"{value:g}" if isinstance(value, float)
                            else str(value))
            except (tk.TclError, TypeError, ValueError):
                continue
        if params:
            applied.append(f"{len(params)} 个期权参数")
        for key, attr, fmt in (
                ("position", "_pos_var", str),
                ("quantity", "_qty_var", lambda v: f"{float(v):g}"),
                ("multiplier", "_mult_var", lambda v: f"{float(v):g}"),
                ("tc_rate", "_tc_var", lambda v: f"{float(v) * 100:g}"),
                ("slippage_bps", "_slip_var", lambda v: f"{float(v):g}"),
                ("force_day_close_hedge", "_force_day_close_hedge_var", bool),
        ):
            if key not in state:
                continue
            var = getattr(self, attr, None)
            if var is None:
                continue
            try:
                var.set(fmt(state[key]))
            except (tk.TclError, TypeError, ValueError):
                continue
        if applied:
            self._refresh_history_base_summary()
        return applied

    def _open_history_result_store(self):
        """已保存结果的列表窗口：载入 / 重命名 / 删除。"""
        try:
            items = history_store.list_results()
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("无法读取已保存结果", str(exc))
            return
        window = tk.Toplevel(self)
        window.title("已保存的优选结果")
        window.geometry("1180x420")
        window.transient(self)
        body = ttk.Frame(window, style="Surface.TFrame", padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text=f"目录: {history_store.results_dir()}",
            style="SurfaceMuted.TLabel",
        ).pack(fill="x", pady=(0, 6))

        # 标的与大小不单独占列：标的已在默认名里（改名后可从悬停找回），
        # 大小是诊断信息、没人靠字节数挑结果。腾出的位置给「证据跨度」和
        # 「结论」——前者决定这份结果值不值得信，后者把文件列表变成研究
        # 日志。「周期 N 个」被跨度取代：近周+近月与近半年+近年都是「2
        # 个」，可信度却差着量级。
        # 一列值不值得占位，看它改不改变你点哪一份。留下的都是「改了会让
        # 结论不可互换」的：多空（买/卖 gamma 是相反逻辑）、成本率（多调
        # 仓划不划算的核心权衡）、T 与 σ（σ 直接决定带宽绝对宽度，同一个
        # 0.75σ 在 σ=0.12 与 0.24 下是两条不同的带）。数量与乘数不入列
        # ——实测只线性缩放金额、不改名次；行权价每段按段初价重定基；粒度
        # 现在是自动推导的因变量。这些都在选中后的细节行里。
        # 保存时刻已经写进名称，不再单独占列；腾出的位置给期权子类型
        # ——同一大类下 Decumulator 有 13 个子型，行为差别很大。
        columns = ("label", "code", "option", "subtype", "position", "asof",
                   "span", "verdict")
        headings = {
            "label": "名称", "code": "标的", "option": "期权",
            "subtype": "子类型", "position": "头寸", "asof": "分析截至日",
            "span": "证据跨度", "verdict": "结论",
        }
        # 子类型用中文名，最长「熔断每日双固赔累计」9 个汉字；列宽按它留。
        widths = {"label": 170, "code": 92, "option": 92, "subtype": 150,
                  "position": 138, "asof": 98, "span": 74, "verdict": 155}
        # 动作行先占位（见下方 actions 的 pack）：树带 expand=True 会吃掉
        # 剩余空腔，排在它后面的按钮行会被压成 0 高度直接消失，而 tkinter
        # 对此完全沉默。
        tree_frame = ttk.Frame(body, style="Surface.TFrame")
        tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=12,
            selectmode="browse")
        for key in columns:
            tree.heading(key, text=headings[key])
            tree.column(
                key, width=widths[key], minwidth=60,
                anchor="w" if key in ("label", "code") else "center",
                stretch=key == "label")
        scrollbar = ttk.Scrollbar(
            tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        tree.tag_configure("incompatible", background=PALETTE["warning_light"])

        by_iid = {}

        def refresh(selected_path=None):
            for child in tree.get_children():
                tree.delete(child)
            by_iid.clear()
            for index, item in enumerate(history_store.list_results()):
                iid = f"result_{index}"
                by_iid[iid] = item
                verdict = item["verdict"]
                if not item["compatible"]:
                    verdict = f"⚠ 版本 {item['schema_version']}，无法载入"
                tree.insert(
                    "", "end", iid=iid,
                    values=(
                        item["label"] or item["filename"],
                        item["wind_code"] or "—",
                        item["option_summary"],
                        item["subtype"],
                        item["position_summary"],
                        item["asof"] or "—",
                        item["evidence_span"],
                        verdict,
                    ),
                    tags=() if item["compatible"] else ("incompatible",))
                if selected_path and item["path"] == selected_path:
                    tree.selection_set(iid)
            BacktestApp._sync_history_store_buttons(
                tree, by_iid, load_btn, rename_btn, delete_btn)

        def selected():
            picked = tree.selection()
            return by_iid.get(picked[0]) if picked else None

        def do_load():
            item = selected()
            if item is None:
                return
            if self._load_history_result(item["path"]):
                window.destroy()

        def do_rename():
            item = selected()
            if item is None:
                return
            label = simpledialog.askstring(
                "重命名", "新名称:", initialvalue=item["label"], parent=window)
            if label is None:
                return
            try:
                history_store.rename_result(item["path"], label)
            except Exception as exc:                  # noqa: BLE001
                messagebox.showerror("重命名失败", str(exc), parent=window)
                return
            refresh(item["path"])

        def do_delete():
            item = selected()
            if item is None:
                return
            name = item["label"] or item["filename"]
            if not messagebox.askyesno(
                    "删除结果", f"删除「{name}」？此操作不可撤销。",
                    parent=window):
                return
            history_store.delete_result(item["path"])
            refresh()

        actions = ttk.Frame(body, style="Surface.TFrame")
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        tree_frame.pack(fill="both", expand=True)
        self._history_store_hint_var = tk.StringVar(value="")
        ttk.Label(
            actions, textvariable=self._history_store_hint_var,
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        delete_btn = ttk.Button(actions, text="删除", width=8,
                                command=do_delete, state="disabled")
        delete_btn.pack(side="right", padx=(6, 0))
        rename_btn = ttk.Button(actions, text="重命名", width=9,
                                command=do_rename, state="disabled")
        rename_btn.pack(side="right", padx=(6, 0))
        load_btn = ttk.Button(actions, text="载入", width=8,
                              command=do_load, state="disabled")
        load_btn.pack(side="right")
        def on_select(_event=None):
            BacktestApp._sync_history_store_buttons(
                tree, by_iid, load_btn, rename_btn, delete_btn)
            item = selected()
            # 标的、周期构成、大小、来源都不单独占列——它们是「确认这是不
            # 是我要的那份」时才看一眼的细节，放在选中后的这一行足够。
            self._history_store_hint_var.set(
                BacktestApp._history_store_detail_line(item) if item else "")

        tree.bind("<<TreeviewSelect>>", on_select)
        tree.bind("<Double-1>", lambda _event: do_load())
        refresh()
        if not by_iid:
            self._history_store_hint_var.set(
                "还没有保存过结果。跑完一次优选后点「保存结果」。")

    @staticmethod
    def _history_store_detail_line(item):
        """选中某份结果后显示的细节：列表里没占列的那些。"""
        if not item:
            return ""
        periods = "、".join(
            history_store._PERIOD_LABELS.get(key, key)
            for key in item.get("lookbacks", ())) or "—"
        parts = [
            f"保存于 {str(item.get('saved_at', ''))[:19].replace('T', ' ')}",
            f"周期 {periods}",
            f"{item.get('rows', 0)} 行 · {item.get('bytes', 0) / 1024:.0f} KB",
        ]
        source = str(item.get("source_label") or "").strip()
        if source:
            parts.append(source)
        if not item.get("compatible", True):
            parts.append(
                f"⚠ 包版本 {item.get('schema_version')}，与当前程序"
                f"（{history_store.SCHEMA_VERSION}）不一致，口径可能已改变")
        return " · ".join(parts)

    @staticmethod
    def _sync_history_store_buttons(tree, by_iid, load_btn, rename_btn,
                                    delete_btn):
        """按选中项启停按钮；版本不兼容的包不给「载入」。"""
        picked = tree.selection()
        item = by_iid.get(picked[0]) if picked else None
        has = item is not None
        for button in (rename_btn, delete_btn):
            try:
                button.configure(state="normal" if has else "disabled")
            except tk.TclError:
                pass
        try:
            load_btn.configure(
                state="normal" if has and item["compatible"] else "disabled")
        except tk.TclError:
            pass

    @staticmethod
    def _track_wraplength(label, slack=0):
        """让 wraplength 跟随控件实际宽度，而不是写死一个数。

        写死的值必然与真实槽位对不上：基准摘要写的是 980，而它的槽位实测
        1242——文本只要多十来个字就在还剩 262px 的情况下提前折成两行。

        防自激：只有宽度真的变了才重设。设 wraplength 会改变**高度**（一行
        变两行），高度变化会让父容器重排并再次派发 <Configure>；不比宽度就
        会形成「设值→重排→再设值」的闭环。
        """
        last = {"width": None}

        def _apply(event=None):
            width = label.winfo_width()
            if width <= 1 or width == last["width"]:
                return
            last["width"] = width
            try:
                label.configure(wraplength=max(80, width - slack))
            except tk.TclError:
                pass

        label.bind("<Configure>", _apply, add="+")
        label.after_idle(_apply)

    @staticmethod
    def _attach_tooltip(widget, text):
        """给控件挂一个轻量悬浮提示。

        用于把过长的辅助信息（例如各周期的最优策略全名）从按钮标题里移
        出去：标题只留周期名，细节悬停可见，按钮宽度就不再被策略名撑爆。
        """
        if not text:
            return
        state = {"tip": None}

        def _show(_event=None):
            if state["tip"] is not None:
                return
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.attributes("-topmost", True)
            tk.Label(
                tip, text=text, bg=PALETTE["text"], fg="#FFFFFF",
                font=(_UI_FONT_FAMILY, 9), padx=8, pady=4, justify="left",
            ).pack()
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            tip.wm_geometry(f"+{x}+{y}")
            state["tip"] = tip

        def _hide(_event=None):
            tip = state.pop("tip", None)
            state["tip"] = None
            if tip is not None:
                tip.destroy()

        widget.bind("<Enter>", _show, add="+")
        widget.bind("<Leave>", _hide, add="+")
        widget.bind("<Destroy>", _hide, add="+")

    def _show_empty_history_results(self):
        container = getattr(self, "_history_results_container", None)
        if container is None:
            return
        for widget in container.winfo_children():
            widget.destroy()
        placeholder = ttk.Frame(container, style="Surface.TFrame")
        placeholder.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(
            placeholder, text="尚未运行策略优选", style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 14, "bold"),
        ).pack(pady=(0, 5))
        ttk.Label(
            placeholder,
            text=("选择 CSV 或 Wind 真实历史行情，设置候选空间后开始。\n"
                  "结果仅显示本次勾选周期的严格连续排名与区间诊断。"),
            style="SurfaceMuted.TLabel", justify="center",
        ).pack()

    # ---- 策略优选纯逻辑的 GUI 侧入口 ----
    # 候选空间构造、任务入参校验、排名展示模型与图表模型都不依赖 tkinter，
    # 实现集中在 history_selection.py。这里按同名 staticmethod 暴露，使
    # 界面回调、后台 worker 与既有调用点仍以统一的 BacktestApp 名字访问。
    _normalize_history_lookbacks = staticmethod(
        history_selection.normalize_lookbacks)
    _validate_history_recommendation_source = staticmethod(
        history_selection.validate_source)
    _validate_fixed_time_history_granularity = staticmethod(
        history_selection.validate_fixed_time_granularity)
    _validate_history_recommendation_payload = staticmethod(
        history_selection.validate_payload)
    _parse_band_candidate_sigmas = staticmethod(
        history_selection.parse_band_candidate_sigmas)
    _band_cases_for_history = staticmethod(history_selection.band_cases)
    _strategy_cases_for_history = staticmethod(
        history_selection.strategy_cases)
    _comparison_finite = staticmethod(history_selection.finite_value)
    _comparison_safe_int = staticmethod(history_selection.safe_int)
    _comparison_safe_bool = staticmethod(history_selection.safe_bool)
    _comparison_baseline = staticmethod(history_selection.ranking_baseline)
    _history_row_improvement = staticmethod(history_selection.row_improvement)
    _history_row_uses_recognized_selection_metric = staticmethod(
        history_selection.row_uses_recognized_metric)
    _history_row_uses_strict_metric = staticmethod(
        history_selection.row_uses_strict_metric)
    _history_row_uses_window_equal_metric = staticmethod(
        history_selection.row_uses_window_equal_metric)
    _history_contract_codes_text = staticmethod(
        history_selection.contract_codes_text)
    _comparison_recommendation_rows = staticmethod(
        history_selection.recommendation_rows)
    _history_chart_pairs = staticmethod(history_selection.chart_pairs)
    _history_chart_array = staticmethod(history_selection.chart_array)
    _history_chart_band = staticmethod(history_selection.chart_band)
    _history_chart_model = staticmethod(history_selection.chart_model)
    _history_multi_chart_model = staticmethod(
        history_selection.multi_chart_model)

    def _history_objective_from_controls(self):
        """本次运行使用的排名依据。

        配置区不再提供选择：首次一律用默认口径，跑完后在结果页点带 ⇅ 的
        列头即可换个看法——两个口径的指标在回测时就一并算好，换口径只是
        重排，没必要在开跑之前先做这个选择。
        """
        variable = getattr(self, "_history_objective_var", None)
        if variable is None:
            return DEFAULT_SELECTION_OBJECTIVE
        objective = HISTORY_OBJECTIVE_FROM_DISPLAY.get(
            variable.get().strip())
        if objective not in SELECTION_OBJECTIVES:
            raise ValueError(f"未知排名依据: {variable.get()!r}")
        return objective

    def _history_lookbacks_from_controls(self, *, require_one=True):
        """读取历史页周期复选框；旧测试替身缺少控件时默认全选。"""
        variables = getattr(self, "_history_period_vars", None)
        if variables is None:
            return BacktestApp._normalize_history_lookbacks()
        flags = {
            key: bool(variable.get())
            for key, variable in variables.items()
        }
        if require_one:
            return BacktestApp._normalize_history_lookbacks(flags)
        try:
            return BacktestApp._normalize_history_lookbacks(flags)
        except ValueError as exc:
            if "至少选择一个" not in str(exc):
                raise
            return {}

    def _refresh_history_auto_wind_start(self):
        """自动模式下把推算出的起始日回填进那个置灰的输入框。

        此前它在自动模式下显示的是构造界面时写死的「今天 − 420 自然日」，
        与真正会用的起始日毫无关系（实测截至日 2026-07-29、近年 243 日时，
        真实起点比它晚了两个多月）。用户看到一个具体日期，很自然会以为就是
        从那天开始取数——比含糊的措辞更能骗人。

        与单次回测的「按数据截止日倒推」同一个模式：置灰只读，推算结果实时
        回填到这一格；取消勾选后这个值就成为手工编辑的起点，不再另存一份用
        户旧输入。手动模式下绝不覆写。
        """
        start_var = getattr(self, "_history_wind_start_var", None)
        auto_var = getattr(self, "_history_wind_auto_start_var", None)
        if start_var is None or auto_var is None:
            return
        try:
            if not bool(auto_var.get()):
                return
        except tk.TclError:
            return
        lookbacks = BacktestApp._history_lookbacks_from_controls(
            self, require_one=False)
        if not lookbacks:
            # 一个周期都没勾时推不出起始日。留着上一次的值会变成又一个"看着
            # 像本次取数起点、实际无关"的假日期——清空，同屏两处提示已经在
            # 说"请先勾选周期"了。
            try:
                start_var.set("")
            except tk.TclError:
                pass
            return
        asof_var = getattr(self, "_history_wind_asof_var", None)
        if asof_var is None:
            return
        try:
            asof = BacktestApp._parse_wind_date(
                asof_var.get(), "历史分析截至日")
        except (ValueError, tk.TclError):
            # 截至日正在输入、还不是合法日期时不猜；用户填完自然会再触发。
            return
        include_band_var = getattr(self, "_history_include_band_var", None)
        warmup_days = BacktestApp._history_realized_warmup_days({
            "history_include_band": (
                bool(include_band_var.get())
                if include_band_var is not None else False),
            # 本页恒用输入波动率，与 _collect_history_state 写入的一致。
            "sigma_source": "implied",
        })
        required = max(lookbacks.values()) + warmup_days + 1
        start = BacktestApp._history_auto_wind_start(
            asof, required_trade_days=required)
        try:
            start_var.set(start.isoformat())
        except tk.TclError:
            pass

    def _history_period_selection_changed(self):
        """同步周期勾选对 Wind 自动范围与基准摘要的影响。

        勾选框文案已固定为「根据周期自动推算起始日」，不再随勾选变化——具体
        往前取多少个交易日、以及为什么要多取一天，都由下方提示行按当前勾选
        实时给出，写在勾选框上只是同屏重复一遍口径。
        """
        BacktestApp._refresh_history_auto_wind_start(self)
        if getattr(self, "_history_wind_hint_var", None) is not None:
            BacktestApp._refresh_history_wind_hint(self)
        if getattr(self, "_history_base_summary_var", None) is not None:
            BacktestApp._refresh_history_base_summary(self)
        # 勾选数为 0 时主按钮要禁用，所以每次改动都得重算按钮状态。
        BacktestApp._sync_history_button_state(self)

    def _toggle_history_candidate_controls(self):
        """只联动历史页自己的候选输入，不影响单次回测控件。"""
        fixed_enabled = bool(self._history_include_fixed_times_var.get())
        band_enabled = bool(self._history_include_band_var.get())
        self._history_fixed_times_entry.configure(
            state="normal" if fixed_enabled else "disabled")
        self._history_band_candidate_entry.configure(
            state="normal" if band_enabled else "disabled")
        self._history_current_band_check.configure(
            state="normal" if band_enabled else "disabled")
        BacktestApp._toggle_history_wind_controls(self)

    def _history_summary_bar_label(self):
        """择优摘要展示的实际粒度；参数尚未就绪时返回占位符。

        粒度不再是可选项，摘要必须显示解析结果而不是“自动”字面量——
        1 分钟与日频差 240 倍数据量，用户在启动前有权知道是哪一档。
        """
        include_fixed_var = getattr(
            self, "_history_include_fixed_times_var", None)
        include_band_var = getattr(self, "_history_include_band_var", None)
        fixed_var = getattr(self, "_history_fixed_times_var", None)
        wind_code_var = getattr(self, "_wind_code_var", None)
        try:
            return BacktestApp._resolve_wind_bar_size(
                WIND_AUTO_BAR_SIZE,
                fixed_times=(
                    fixed_var.get().strip() if fixed_var is not None else ""),
                include_fixed_times=(
                    bool(include_fixed_var.get())
                    if include_fixed_var is not None else False),
                include_band=(
                    bool(include_band_var.get())
                    if include_band_var is not None else False),
                wind_code=(
                    wind_code_var.get().strip()
                    if wind_code_var is not None else ""),
            )
        except ValueError:
            # 固定时刻还没填合法值时不阻塞摘要；启动任务时才是权威校验点。
            return "—"

    def _refresh_history_base_summary(self):
        """刷新历史实验将冻结的公共回测环境摘要。"""
        variable = getattr(self, "_history_base_summary_var", None)
        source_var = getattr(self, "_source_var", None)
        if variable is None or source_var is None:
            return
        source = source_var.get()
        if source == "csv":
            path_var = getattr(self, "_csv_path_var", None)
            path = path_var.get().strip() if path_var is not None else ""
            column_var = getattr(self, "_csv_col_var", None)
            column = column_var.get().strip() if column_var is not None else "close"
            source_text = (
                f"CSV · {os.path.basename(path) or '尚未选择文件'}"
                f" · {column or 'close'}")
        elif source == "wind":
            code_var = getattr(self, "_wind_code_var", None)
            start_var = getattr(self, "_history_wind_start_var", None)
            end_var = getattr(self, "_history_wind_asof_var", None)
            auto_start_var = getattr(
                self, "_history_wind_auto_start_var", None)
            code = code_var.get().strip() if code_var is not None else "—"
            bar = BacktestApp._history_summary_bar_label(self)
            # 自动模式也直接写推算出的起始日，而不是"自动覆盖最近 243 日连续
            # 区间"这种规则描述。这一行讲的是"本次将冻结的取数区间"，规则描
            # 述放在这里读者还得自己换算成日期；而起始日已经实时回填进旁边那
            # 个输入框，两处显示同一个日期才对得上。周期没勾时才退回提示语。
            start = start_var.get().strip() if start_var is not None else "—"
            if auto_start_var is not None and auto_start_var.get():
                lookbacks = BacktestApp._history_lookbacks_from_controls(
                    self, require_one=False)
                if not lookbacks:
                    start = "未选择分析周期"
            end = end_var.get().strip() if end_var is not None else "—"
            source_text = (
                f"Wind · {code or '—'} · {start or '—'} 至 {end or '—'}"
                f" · {bar or '—'}")
        else:
            source_text = "模拟行情（策略优选不可用）"
        subtype_var = getattr(self, "_subtype_var", None)
        subtype = subtype_var.get() if subtype_var is not None else "—"
        position_var = getattr(self, "_pos_var", None)
        position = position_var.get() if position_var is not None else "—"
        tc_var = getattr(self, "_tc_var", None)
        tc = tc_var.get() if tc_var is not None else "—"
        quantity_var = getattr(self, "_qty_var", None)
        quantity = quantity_var.get() if quantity_var is not None else "—"
        multiplier_var = getattr(self, "_mult_var", None)
        multiplier = multiplier_var.get() if multiplier_var is not None else "—"
        slippage_var = getattr(self, "_slip_var", None)
        slippage = slippage_var.get() if slippage_var is not None else "—"
        close_fallback_var = getattr(
            self, "_force_day_close_hedge_var", None)
        close_fallback = (
            "开启" if close_fallback_var is not None
            and bool(close_fallback_var.get()) else "关闭")
        param_entries = getattr(self, "_param_entries", {})
        maturity_entry = (
            param_entries.get("T_days") or param_entries.get("T"))
        maturity = (
            maturity_entry[0].get().strip()
            if maturity_entry is not None else "—")
        selected_lookbacks = BacktestApp._history_lookbacks_from_controls(
            self, require_one=False)
        period_labels = {
            key: label for key, label in HISTORY_PERIOD_DEFS}
        selected_period_text = "、".join(
            period_labels[key] for key in selected_lookbacks)
        if not selected_period_text:
            selected_period_text = "未选择"
        variable.set(
            f"本次运行将冻结：{source_text}  ·  {subtype or '—'}  ·  "
            f"代理期限 T={maturity or '—'} 日  ·  "
            f"严格连续周期 {selected_period_text}  ·  "
            f"收盘保底 {close_fallback}  ·  "
            f"头寸方向 {position}  ·  数量 {quantity}  ·  乘数 {multiplier}  ·  "
            f"成本率 {tc}%  ·  滑点 {slippage} bps  ·  左侧期权参数")

    def _refresh_history_current_band_label(self):
        variable = getattr(self, "_history_current_band_label_var", None)
        if variable is None:
            return
        try:
            sigma_multiple = float(self._band_sigma_var.get().strip())
            if not np.isfinite(sigma_multiple) or sigma_multiple <= 0:
                raise ValueError
            kind_labels = {
                "absolute": "绝对", "relative": "相对", "sigma": "σ",
            }
            source_label = kind_labels.get(self._band_last_edited, "当前")
            variable.set(
                f"加入当前回测带宽：{sigma_multiple:.6g}σ（{source_label}输入换算）")
        except (AttributeError, TypeError, ValueError):
            variable.set("加入当前回测带宽（编辑完成后自动换算）")

    # ---- 事件回调 ----
    def _on_notebook_tab_changed(self, _event=None):
        """按落地页刷新它依赖的会话状态。

        历史页要刷新将被冻结的公共环境与当前带宽摘要；结果对比页此前只由
        左侧按钮构建，从标签直接进来的用户会看到「先保留结果」占位符——哪
        怕按钮上已经写着结果数。
        """
        try:
            current = self._nb.select()
        except (AttributeError, tk.TclError):
            return
        if current == str(self._history_tab):
            BacktestApp._toggle_history_wind_controls(self)
            self._refresh_history_base_summary()
            self._refresh_history_current_band_label()
            return
        if current == str(getattr(self, "_compare_tab", None)):
            BacktestApp._ensure_saved_comparison_page(self)

    def _ensure_saved_comparison_page(self):
        """结果池非空却还没构建过页面时补建一次，不改变任何勾选状态。

        只在页面缺失时动手：已经构建过的页面由结果池自己的刷新链路维护，
        重建会丢掉用户当前的聚焦行。池子为空时保持占位符——那句提示正是
        此时该说的话。
        """
        tree = getattr(self, "_saved_pool_tree", None)
        try:
            if tree is not None and tree.winfo_exists():
                return
        except tk.TclError:
            # widget 已随旧渲染销毁，按未构建处理。
            pass
        if not getattr(self, "_saved_backtests", None):
            return
        self._show_saved_comparison_page(navigate=False)

    def _on_option_class_change(self, event):
        cls_name = self._class_var.get()
        cfg = OPTION_CLASSES[cls_name]
        display_values = [SUBTYPE_DISPLAY[s] if s in SUBTYPE_DISPLAY else str(s)
                          for s in cfg["subtypes"]]
        self._subtype_cb.configure(values=display_values)
        self._subtype_cb.current(0)
        self._rebuild_params(cfg["params"])
        self._refresh_history_base_summary()

    def _rebuild_params(self, params):
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_entries = {}
        self._param_widgets = {}
        for i, spec in enumerate(params):
            key, label, dtype, default = spec[:4]
            choices = spec[4] if len(spec) > 4 else None
            meta = spec[5] if len(spec) > 5 else None
            editable = bool(meta and meta.get("editable"))  # 可编辑下拉=预设+手填
            label_widget = _form_label(self._param_frame, f"{label}:", i)
            if choices:
                val_to_display = {v: k for k, v in choices.items()}
                default_display = val_to_display.get(
                    default, str(default) if editable else next(iter(choices)))
                var = tk.StringVar(value=default_display)
                cb = ttk.Combobox(self._param_frame, textvariable=var,
                                  values=list(choices.keys()),
                                  state="normal" if editable else "readonly",
                                  width=FORM_ENTRY_CHARS)
                _form_input(cb, i)
                input_widget = cb
                if key == "margin_call":
                    cb.bind("<<ComboboxSelected>>",
                            lambda _event: self._sync_snowball_margin_controls())
            else:
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(self._param_frame, textvariable=var,
                                  width=FORM_ENTRY_CHARS)
                _form_input(entry, i)
                input_widget = entry
            self._param_entries[key] = (var, dtype, choices)
            self._param_widgets[key] = {
                "label": label_widget,
                "input": input_widget,
                "base_label": label,
                "choices": choices,
            }
        _form_grid(self._param_frame)
        self._sync_snowball_margin_controls()
        self._bind_band_reference_inputs()
        self._bind_wind_maturity_input()

    def _bind_wind_maturity_input(self):
        """重建期权参数后重新监听剩余期限：它决定倒推出来的建仓日。"""
        if getattr(self, "_wind_start_var", None) is None:
            return
        entry = (self._param_entries.get("T_days")
                 or self._param_entries.get("T"))
        if entry is not None:
            entry[0].trace_add(
                "write", lambda *_args: self._sync_wind_entry_date())
        self._sync_wind_entry_date()

    def _bind_band_reference_inputs(self):
        """重建期权参数后，重新监听带宽换算依赖的 S0 与年化 sigma。"""
        if not hasattr(self, "_band_last_edited"):
            return
        for key in ("s0", "sigma"):
            entry = self._param_entries.get(key)
            if entry is not None:
                entry[0].trace_add(
                    "write", lambda *_args: self._schedule_band_reference_sync())
        self._schedule_band_reference_sync()

    def _schedule_band_reference_sync(self):
        """合并连续键入事件；输入完整后按最后编辑单位刷新另外两项。"""
        if not hasattr(self, "_band_last_edited") or self._band_syncing:
            return
        pending = getattr(self, "_band_reference_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except (tk.TclError, ValueError):
                pass
        self._band_reference_after_id = self.after(120, self._refresh_band_reference)

    def _refresh_band_reference(self):
        self._band_reference_after_id = None
        if self._band_syncing:
            return
        # 参数编辑到一半时不弹窗；运行前会执行 strict 校验并阻止错误输入。
        self._sync_band_inputs(self._band_last_edited, quiet=True)

    def destroy(self):
        """先撤掉挂起的带宽换算定时器，再销毁窗口。

        Tk 的 destroy 清 widget、也清本实例注册的 Tcl 命令，唯独不动 after
        队列。带宽输入的防抖定时器几乎总是挂着一个，窗口关掉后它仍排在事件
        队列里，指向一个已经被删掉的命令名。同一个进程里随后再开一个窗口
        时，那次事件循环就会执行到这条失效的 after 脚本 —— 报
        ``invalid command name``，并且不再返回。
        """
        pending = getattr(self, "_band_reference_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except (tk.TclError, ValueError):
                pass
            self._band_reference_after_id = None
        super().destroy()

    def _sync_snowball_margin_controls(self):
        """雪球保证金模式联动：追保时保证金比例不参与封顶，置灰避免误读。"""
        if not hasattr(self, "_param_entries"):
            return
        if "margin_call" not in self._param_entries or "margin" not in self._param_entries:
            return

        mode_var, _, mode_choices = self._param_entries["margin_call"]
        selected = mode_var.get().strip()
        margin_call = bool(mode_choices.get(selected, 1)) if mode_choices else True

        margin_widgets = getattr(self, "_param_widgets", {}).get("margin", {})
        label_widget = margin_widgets.get("label")
        input_widget = margin_widgets.get("input")
        if input_widget is None:
            return

        if margin_call:
            input_widget.configure(state="disabled")
            if label_widget is not None:
                label_widget.configure(text="保证金比例(追保下不封顶):")
        else:
            input_widget.configure(state="normal")
            if label_widget is not None:
                label_widget.configure(text="保证金比例(最大亏损):")

    def _toggle_wind_auto_start(self):
        entry = getattr(self, "_wind_start_entry", None)
        variable = getattr(self, "_wind_auto_start_var", None)
        if entry is not None and variable is not None:
            entry.configure(state="disabled" if variable.get() else "normal")
        BacktestApp._sync_wind_entry_date(self)

    def _maturity_days_from_controls(self):
        """读取期权参数区的剩余期限；输入未完成时返回 None（只用于联动）。"""
        entries = getattr(self, "_param_entries", {}) or {}
        entry = entries.get("T_days") or entries.get("T")
        if entry is None:
            return None
        try:
            return BacktestApp._maturity_days_from_params(
                {"T_days": entry[0].get().strip()})
        except (AttributeError, TypeError, ValueError):
            return None

    def _sync_wind_entry_date(self):
        """勾选倒推时把倒推结果回填到置灰的建仓日框。

        建仓日就显示在它自己那一格里，界面不再单列"实际区间"——正因如此这
        一格必须是倒推结果本身，留着上次手填的日期就成了假的建仓日。回填会
        再次触发同一个 trace，用 ``_wind_date_syncing`` 挡住重入。

        截止日或期限编辑到一半时保持上一个有效值：这里只负责显示，启动任务
        时的 ``_resolve_single_wind_state`` 才是权威校验点。
        """
        start_var = getattr(self, "_wind_start_var", None)
        if start_var is None or getattr(self, "_wind_date_syncing", False):
            return
        auto_var = getattr(self, "_wind_auto_start_var", None)
        if auto_var is not None and not auto_var.get():
            return
        maturity_days = BacktestApp._maturity_days_from_controls(self)
        if maturity_days is None:
            return
        asof_var = getattr(self, "_wind_end_var", None)
        try:
            asof = BacktestApp._parse_wind_date(
                asof_var.get().strip() if asof_var is not None else "",
                "Wind 数据截止日")
            start, _asof_anchor = BacktestApp._entry_date_from_asof(
                asof, maturity_days)
        except ValueError:
            return
        if start_var.get().strip() != start.isoformat():
            self._wind_date_syncing = True
            try:
                start_var.set(start.isoformat())
            finally:
                self._wind_date_syncing = False

    def _commit_wind_asof(self):
        """编辑完成后把截止日落到倒推真正使用的交易日锚点。"""
        auto_var = getattr(self, "_wind_auto_start_var", None)
        if auto_var is not None and not auto_var.get():
            return
        asof_var = getattr(self, "_wind_end_var", None)
        if asof_var is None:
            return
        try:
            asof = BacktestApp._parse_wind_date(
                asof_var.get().strip(), "Wind 数据截止日")
        except ValueError:
            return
        anchor = BacktestApp._latest_trading_day(asof)
        if anchor != asof:
            # 触发 trace，建仓日随新锚点一起更新。
            asof_var.set(anchor.isoformat())

    def _refresh_wind_frequency_hint(self):
        hint = getattr(self, "_wind_frequency_hint_var", None)
        strategy_var = getattr(self, "_strategy_var", None)
        if hint is None or strategy_var is None:
            return
        strategy_name = STRATEGY_FROM_DISPLAY.get(
            strategy_var.get(), strategy_var.get())
        fixed_var = getattr(self, "_fixed_times_var", None)
        fixed_times = fixed_var.get().strip() if fixed_var is not None else ""
        wind_code_var = getattr(self, "_wind_code_var", None)
        wind_code = (
            wind_code_var.get().strip() if wind_code_var is not None else "")
        try:
            actual = BacktestApp._resolve_wind_bar_size(
                WIND_AUTO_BAR_SIZE, strategy_name=strategy_name,
                fixed_times=fixed_times, wind_code=wind_code)
        except ValueError as exc:
            hint.set(str(exc))
            return
        reason = {
            "close_to_close": "收盘策略",
            "fixed_times": "覆盖目标时刻",
            "hedge_band": "不漏 bar 内穿带",
        }.get(strategy_name, "策略需要")
        hint.set(f"{actual}（{reason}）")

    def _toggle_history_wind_controls(self):
        source_var = getattr(self, "_source_var", None)
        is_wind = source_var is not None and source_var.get() == "wind"
        auto_var = getattr(self, "_history_wind_auto_start_var", None)
        auto_start = bool(auto_var.get()) if auto_var is not None else True

        asof_entry = getattr(self, "_history_wind_asof_entry", None)
        if asof_entry is not None:
            asof_entry.configure(state="normal" if is_wind else "disabled")
        check = getattr(self, "_history_wind_auto_start_check", None)
        if check is not None:
            check.configure(state="normal" if is_wind else "disabled")
        start_entry = getattr(self, "_history_wind_start_entry", None)
        if start_entry is not None:
            start_entry.configure(
                state="normal" if is_wind and not auto_start else "disabled")
        # 勾回自动时必须立刻重算：否则框里留着用户手动模式下填的旧日期，而
        # 那已经不是本次真正会用的起始日了。
        BacktestApp._refresh_history_auto_wind_start(self)
        BacktestApp._refresh_history_wind_hint(self)
        if getattr(self, "_history_base_summary_var", None) is not None:
            BacktestApp._refresh_history_base_summary(self)

    def _refresh_history_wind_hint(self):
        hint = getattr(self, "_history_wind_hint_var", None)
        source_var = getattr(self, "_source_var", None)
        if hint is None or source_var is None:
            return
        if source_var.get() != "wind":
            hint.set("CSV 使用文件内完整日期与时间索引；本组 Wind 参数不参与。")
            return

        include_fixed_var = getattr(
            self, "_history_include_fixed_times_var", None)
        include_band_var = getattr(self, "_history_include_band_var", None)
        include_fixed = bool(
            include_fixed_var.get()) if include_fixed_var is not None else False
        include_band = bool(
            include_band_var.get()) if include_band_var is not None else False
        fixed_var = getattr(self, "_history_fixed_times_var", None)
        fixed_times = fixed_var.get().strip() if fixed_var is not None else ""
        wind_code_var = getattr(self, "_wind_code_var", None)
        wind_code = (
            wind_code_var.get().strip() if wind_code_var is not None else "")
        # 这里仍然要解析一次粒度，但**不是为了显示**：解析不出可用粒度时
        # （例如固定时刻落在该代码的休市区间、任何一档都覆盖不了）会抛错，
        # 而这条错误正是本行唯一的报告出口。别因为返回值没被用掉就删掉它。
        # 实际采用的粒度本身不在这里说——顶部基准摘要已经写了一遍。
        try:
            BacktestApp._resolve_wind_bar_size(
                WIND_AUTO_BAR_SIZE, fixed_times=fixed_times,
                include_fixed_times=include_fixed,
                include_band=include_band, wind_code=wind_code)
        except ValueError as exc:
            hint.set(str(exc))
            return
        auto_start_var = getattr(
            self, "_history_wind_auto_start_var", None)
        if auto_start_var is None or auto_start_var.get():
            lookbacks = BacktestApp._history_lookbacks_from_controls(
                self, require_one=False)
            if not lookbacks:
                hint.set("请至少勾选一个历史分析周期")
                return
            # 「Day 0 锚点」在这里展开成它到底是什么、为什么要多取：首日的
            # 损益要和前一个交易日的收盘比，所以区间前面必须多一天收盘价。
            # 这是本行唯一别处没有的信息。
            hint.set(
                f"起始日往前取满 {max(lookbacks.values())} 个交易日，"
                "并多取前一个交易日的收盘价作为首日损益的比较起点")
        else:
            # 手动模式无需提示：勾选框未勾、输入框可编辑且已填着日期，本身
            # 就说明了一切，再写一句"使用你填写的起始日"是同屏第三遍。
            hint.set("")

    def _toggle_source(self):
        src = self._source_var.get()
        self._sim_frame.grid_remove()
        self._csv_frame.grid_remove()
        self._wind_frame.grid_remove()
        if src == "simulate":
            self._sim_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        elif src == "csv":
            self._csv_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        elif src == "wind":
            self._wind_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        BacktestApp._toggle_strategy(self)
        BacktestApp._toggle_wind_auto_start(self)
        BacktestApp._toggle_history_wind_controls(self)
        self._refresh_history_base_summary()
        BacktestApp._sync_history_button_state(self)

    def _toggle_strategy(self):
        """按策略显示其专用参数。"""
        strategy_key = STRATEGY_FROM_DISPLAY.get(
            self._strategy_var.get(), self._strategy_var.get())
        band_mode = self._interval_type_var.get()
        if strategy_key == "hedge_band" and band_mode == "sigma":
            self._sigma_band_frame.grid()
        else:
            self._sigma_band_frame.grid_remove()
        if strategy_key == "fixed_times":
            self._fixed_time_frame.grid()
            self._band_frame.grid_remove()
        elif strategy_key == "hedge_band":
            self._fixed_time_frame.grid_remove()
            self._band_frame.grid()
        else:
            self._fixed_time_frame.grid_remove()
            self._band_frame.grid_remove()
        band_entries = (self._band_abs_entry, self._band_rel_entry, self._band_sigma_entry)
        if strategy_key == "fixed_times":
            self._fixed_times_entry.configure(state="normal")
            for entry in band_entries:
                entry.configure(state="disabled")
        elif strategy_key == "hedge_band":
            self._fixed_times_entry.configure(state="disabled")
            for entry in band_entries:
                entry.configure(state="normal")
            if not self._band_synced:
                self._band_synced = True
                self._sync_band_inputs(self._band_last_edited)
        BacktestApp._refresh_wind_frequency_hint(self)

    def _mark_band_edited(self, source_type):
        """记住真正的用户输入源，避免回测读到隐藏的旧状态。"""
        if self._band_syncing:
            return
        self._band_last_edited = source_type
        self._interval_type_var.set(source_type)
        variable = getattr(self, "_history_current_band_label_var", None)
        if variable is not None:
            variable.set("加入当前回测带宽（编辑完成后自动换算）")
        self._toggle_strategy()

    def _commit_band_input(self, source_type, force=False):
        if force:
            self._mark_band_edited(source_type)
        if self._band_last_edited == source_type:
            return self._sync_band_inputs(source_type)
        return None

    def _sync_band_inputs(self, source_type=None, *, strict=False, quiet=False):
        """用当前 S0 / sigma 将指定带宽反推为另外两种单位。

        ``strict=True`` 用于运行前的最终校验，失败时直接抛错；
        日常联动在用户输入尚未完整时保持静默。
        """
        source_type = source_type or self._band_last_edited
        if source_type not in ("absolute", "relative", "sigma"):
            exc = ValueError(f"未知带宽单位: {source_type}")
            if strict:
                raise exc
            return None
        vars_by_type = {
            "absolute": self._band_abs_var,
            "relative": self._band_rel_var,
            "sigma": self._band_sigma_var,
        }
        if self._band_syncing:
            return None
        try:
            value = float(vars_by_type[source_type].get().strip())
            if not np.isfinite(value):
                raise ValueError("带宽必须是有限数值")
            if value <= 0:
                raise ValueError("带宽必须大于 0")
            s0_var = self._param_entries["s0"][0]
            sigma_var = self._param_entries["sigma"][0]
            s0 = float(s0_var.get().strip())
            sigma = float(sigma_var.get().strip())
            if not np.isfinite(s0) or s0 <= 0:
                raise ValueError("初始价格 S0 必须大于 0")
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("年化波动率 sigma 必须大于 0")
            converted = HedgeBandStrategy.convert_threshold(value, source_type, s0, sigma)
            self._band_syncing = True
            try:
                self._band_abs_var.set(f"{converted['absolute']:.10g}")
                self._band_rel_var.set(f"{converted['relative']:.10g}")
                self._band_sigma_var.set(f"{converted['sigma']:.10g}")
                self._band_last_edited = source_type
                self._interval_type_var.set(source_type)
                # 后端阈值始终使用最后编辑项的原始单位与数值。
                self._price_interval_var.set(f"{value:.10g}")
            finally:
                self._band_syncing = False
            self._toggle_strategy()
            self._refresh_history_current_band_label()
            if not quiet:
                self._set_status(
                    f"带宽已换算（参考价 {s0:g}，年化 σ {sigma:g}，"
                    f"{ANNUAL_DAYS} 个交易日计算）")
            return converted
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"固定间隔参数无效（{source_type}）：{exc}") from exc
            if not quiet:
                self._set_status(f"带宽换算暂未完成：{exc}")
            return None

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._csv_path_var.set(path)

    # ---- 回测核心 ----
    def _prepare_active_strategy_inputs(self, *, include_band=False):
        """在主线程对当前活动策略做最终同步与校验。"""
        strategy_name = STRATEGY_FROM_DISPLAY.get(
            self._strategy_var.get(), self._strategy_var.get())
        if strategy_name == "hedge_band" or include_band:
            self._sync_band_inputs(self._band_last_edited, strict=True)
        if strategy_name == "fixed_times":
            # 构造策略对 HH:MM 格式和空列表做权威校验。
            FixedTimeStrategy(self._fixed_times_var.get().strip())

    @staticmethod
    def _validate_fixed_time_source_state(gs):
        """在拉取行情前快速拒绝明确不可用的数据源。"""
        if gs.get("strategy_name") != "fixed_times":
            return
        src = gs.get("source")
        if src == "simulate":
            raise ValueError(
                "每日固定时刻策略需要真实日内时间戳；模拟路径不可用。")
        if src == "wind" and gs.get("wind_bar_size", "日频") == "日频":
            raise ValueError(
                "每日固定时刻策略不支持 Wind 日频行情。粒度现在按策略自动"
                "推导，出现该错误说明用的是旧快照里保存的手动日频配置。")
        if src == "csv" and BacktestApp._csv_is_daily(gs.get("csv_path")):
            raise ValueError(
                "每日固定时刻策略需要日内时间戳，所选 CSV 是日频数据"
                "（每个交易日只有一行）。请换用分钟级 CSV，或改用其它策略。")

    # 判定日频只需要看到"连着好些行都是不同的日子"，读前 200 行足矣：日频
    # 的话这就是 200 个交易日，分钟级的话还没走完第一天。整份读进来只为判一
    # 个粒度，大文件上是白等——而这里的职责恰恰是"在拉取行情前**快速**拒绝"。
    _CSV_GRANULARITY_PROBE_ROWS = 200

    @staticmethod
    def _csv_is_daily(path):
        """CSV 是不是日频（每个交易日一行）。读不出来时一律返回 False。

        读不出就放行，把报错留给随后的 ``from_csv``：文件不存在、首列不是时
        间、列名拼错，它给的消息都比"这个文件是不是日频"精确得多。本函数只
        负责拦住"文件没问题、但粒度配不上固定时刻策略"这一种。
        """
        path = str(path or "").strip()
        if not path:
            return False
        try:
            import pandas as pd
            frame = pd.read_csv(
                path, parse_dates=[0], index_col=0,
                nrows=BacktestApp._CSV_GRANULARITY_PROBE_ROWS)
            index = frame.index
            if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
                return False
            return int(_infer_intraday_steps(index)) <= 1
        except Exception:                                  # noqa: BLE001
            return False

    def _sync_history_button_state(self):
        """使策略优选入口始终与真实数据来源及后台任务状态一致。"""
        source_var = getattr(self, "_source_var", None)
        if source_var is None:
            return
        # 一个周期都没勾时也要禁用：此前按钮照常可点，点下去才弹
        # 「请至少选择一个历史分析周期。」的模态框——用模态错误代替禁用态，
        # 等于让用户先撞一次才知道不能按。
        try:
            has_period = bool(BacktestApp._history_lookbacks_from_controls(
                self, require_one=False))
        except (AttributeError, tk.TclError):
            has_period = True      # 控件还没建好时不误禁
        enabled = (
            getattr(self, "_active_job", None) is None
            and source_var.get() in ("csv", "wind")
            and has_period
        )
        state = "normal" if enabled else "disabled"
        for attr in ("_history_btn",):
            button = getattr(self, attr, None)
            if button is None:
                continue
            try:
                exists = getattr(button, "winfo_exists", None)
                if exists is not None and not exists():
                    setattr(self, attr, None)
                    continue
                button.configure(state=state)
            except tk.TclError:
                # 容忍窗口销毁期间的延迟来源联动。
                setattr(self, attr, None)
        hint = getattr(self, "_history_source_hint_var", None)
        if hint is not None:
            if source_var.get() not in ("csv", "wind"):
                hint.set("策略优选仅支持 CSV / Wind，模拟行情下不可运行")
            elif not has_period:
                # 禁用必须有就近可循的理由：这条 pill 紧挨着主按钮。
                hint.set("请至少勾选一个分析周期")
            else:
                # 一切正常时不占版面；能不能跑由按钮自身的可用状态表达。
                hint.set("")
        BacktestApp._update_history_header_summary(self)

    def _begin_job(self, job_name, status_text):
        """原子地进入 GUI 后台任务状态，并锁住所有长任务入口。"""
        active = getattr(self, "_active_job", None)
        if active is not None:
            messagebox.showinfo("任务运行中", "已有任务正在运行，请等待其完成。")
            return False
        self._active_job = job_name
        for button in BacktestApp._job_guarded_buttons(self):
            button.configure(state="disabled")
        BacktestApp._sync_history_button_state(self)
        self._set_status(status_text)
        return True

    def _job_guarded_buttons(self):
        """返回会重建结果视图或改变快照状态的入口按钮。"""
        buttons = []
        for attr in (
                "_run_btn", "_retain_btn", "_history_btn", "_struct_btn"):
            button = getattr(self, attr, None)
            if button is not None:
                buttons.append(button)
        return buttons

    def _sync_retain_button_state(self):
        button = getattr(self, "_retain_btn", None)
        if button is None:
            return
        can_retain = (
            getattr(self, "_active_job", None) is None
            and getattr(self, "_latest_backtest", None) is not None
            and getattr(self, "_latest_retained_result_id", None) is None
        )
        button.configure(state="normal" if can_retain else "disabled")

    def _finish_job(self, job_name, *, success, success_text, failure_text):
        """仅由当前任务恢复共享进度条和入口按钮。"""
        if getattr(self, "_active_job", None) != job_name:
            return
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.pack_forget()
        self._progress_label.pack_forget()
        self._progress_label.configure(text="")
        self._active_job = None
        for button in BacktestApp._job_guarded_buttons(self):
            button.configure(state="normal")
        BacktestApp._sync_retain_button_state(self)
        BacktestApp._sync_history_button_state(self)
        self._set_status(success_text if success else failure_text)

    def _run_backtest(self):
        # 在主线程中收集所有 GUI 参数（tkinter 非线程安全）
        try:
            self._prepare_active_strategy_inputs()
            gui_state = self._collect_gui_state()
            self._validate_fixed_time_source_state(gui_state)
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return False

        if not self._begin_job("backtest", "正在运行回测…"):
            return False
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.start(15)
        threading.Thread(target=self._backtest_worker, args=(gui_state,),
                         daemon=True).start()
        return True

    def _run_history_recommendation(self):
        """用 CSV/Wind 真实历史执行独立的候选策略优选。"""
        if getattr(self, "_active_job", None) is not None:
            messagebox.showinfo("任务运行中", "已有任务正在运行，请等待其完成。")
            return False
        try:
            # 在同步带宽和读取其它表单前先拦截模拟来源；禁用按钮之外仍保留
            # 这道入口校验，防止快捷键、旧页面控件或直接调用绕过。
            source_var = getattr(self, "_source_var", None)
            if source_var is not None:
                BacktestApp._validate_history_recommendation_source(
                    {"source": source_var.get()})
            history_state = self._collect_history_state()
            selected_lookbacks = BacktestApp._normalize_history_lookbacks(
                history_state.get("history_lookbacks"))
            history_state["history_lookbacks"] = selected_lookbacks
            self._refresh_history_base_summary()
            BacktestApp._validate_history_recommendation_source(history_state)
        except Exception as exc:
            messagebox.showerror("策略优选不可用", str(exc))
            return False

        period_labels = {
            key: label for key, label in HISTORY_PERIOD_DEFS}
        selected_text = "、".join(
            period_labels[key] for key in selected_lookbacks)
        if not self._begin_job(
                "history",
                f"正在使用真实历史行情执行批量优选（{selected_text}）…"):
            return False
        # 与单次回测、结构扫描、分段重放共用同一条进度条：全应用只有一个
        # 长任务指示器，任务名由底部状态栏给出。
        self._progress.configure(mode="indeterminate")
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.start(15)
        threading.Thread(
            target=self._history_recommendation_worker,
            args=(history_state,), daemon=True,
        ).start()
        return True

    @staticmethod
    def _rescale_strategy_cases(cases, ratio):
        """为已重定基的当前路径复制策略，保留原 cases 供历史代理段使用。"""
        return [
            StrategyCase(
                case.name,
                _rescale_strategy_to_real_s0(case.strategy, ratio),
                copy.deepcopy(case.metadata),
            )
            for case in cases
        ]

    @staticmethod
    def _comparison_backtest_kwargs(bt):
        return {
            "tc_rate": bt.tc_rate,
            "position": BacktestApp._normalize_position(bt.position),
            "quantity": bt.quantity,
            "multiplier": bt.multiplier,
            "steps_per_day": bt.steps_per_day,
            "slippage_bps": bt.slippage_bps,
            "force_day_close_hedge": bool(
                getattr(bt, "force_day_close_hedge", False)),
        }

    @staticmethod
    def _series_with_backtest_timestamps(bt):
        if getattr(bt, "timestamps", None) is None:
            return np.asarray(bt.prices, dtype=float)
        import pandas as pd
        return pd.Series(
            np.asarray(bt.prices, dtype=float),
            index=pd.DatetimeIndex(bt.timestamps), name="close",
        )

    @staticmethod
    def _load_wind_contract_history_pool(gs):
        """按分析截止日的历史主力映射加载同品种具体合约行情。"""
        import pandas as pd
        from pricing.wind_data import (
            get_close_prices,
            get_intraday_close,
            get_main_contract_history,
        )

        product_code = str(gs["wind_code"]).strip().upper()
        start_date = BacktestApp._parse_wind_date(
            gs["wind_start"], "历史行情起始日")
        end_date = BacktestApp._parse_wind_date(
            gs["wind_end"], "历史分析截至日")
        mapping = get_main_contract_history(
            product_code, start_date.isoformat(), end_date.isoformat())
        # 严格区间只加载本次所选最长证据区间内真实出现的主力合约。
        # 区间之前只为 Day 0 锚点留出必要历史，不能再因代理期限 T 把行情扩
        # 到证据区间之外。下面仍加上 warmup_days 是留给引擎的 realized 通
        # 路——本页 σ 恒取输入值，它恒为 0。
        history_lookbacks = BacktestApp._normalize_history_lookbacks(
            gs.get("history_lookbacks"))
        evidence_days = max(history_lookbacks.values())
        evidence_mapping = mapping.iloc[-min(evidence_days, len(mapping)):]
        contract_codes = list(dict.fromkeys(evidence_mapping.astype(str)))
        warmup_days = BacktestApp._history_realized_warmup_days(gs)
        prehistory_span = BacktestApp._calendar_span_for_trading_days(
            warmup_days + 1)
        bar_label = gs.get("wind_bar_size", "日频")
        contract_prices = {}
        contract_load_errors = {}
        for contract_code in contract_codes:
            main_dates = evidence_mapping.index[
                evidence_mapping.astype(str).eq(contract_code)]
            if not len(main_dates):
                continue
            contract_start = max(
                start_date,
                pd.Timestamp(main_dates[0]).date()
                - datetime.timedelta(days=prehistory_span),
            )
            contract_end = min(
                end_date, pd.Timestamp(main_dates[-1]).date())
            if contract_start >= contract_end:
                contract_load_errors[contract_code] = (
                    f"可用请求区间不足：{contract_start} 至 {contract_end}")
                continue
            try:
                if bar_label == "日频":
                    prices = get_close_prices(
                        contract_code,
                        contract_start.isoformat(), contract_end.isoformat(),
                        "",
                    )
                else:
                    prices = get_intraday_close(
                        contract_code,
                        contract_start.isoformat(), contract_end.isoformat(),
                        bar_size=bar_label.removesuffix("min"), adjust="",
                    )
            except Exception as exc:
                contract_load_errors[contract_code] = str(exc)
                continue
            if len(prices) < 2:
                contract_load_errors[contract_code] = (
                    f"仅返回 {len(prices)} 个价格点")
                continue
            contract_prices[contract_code] = prices

        if not contract_prices:
            detail = next(iter(contract_load_errors.values()), "未返回行情")
            raise ValueError(
                f"{product_code} 在分析截止日前没有可用的具体主力合约行情："
                f"{detail}")
        return ContractHistoryPool(
            product_code=product_code,
            main_contract_by_date=mapping,
            contract_prices=contract_prices,
            main_contract_asof=str(mapping.iloc[-1]),
            contract_load_errors=contract_load_errors,
        )

    @staticmethod
    def _contract_pool_backtest_context(gs, pool):
        """用最近可用具体合约构造候选配置上下文，不请求连续合约行情。"""
        selected = None
        for contract_code in reversed(
                pool.main_contract_by_date.astype(str).tolist()):
            selected = pool.contract_prices.get(contract_code.upper())
            if selected is not None:
                break
        if selected is None:
            selected = next(iter(pool.contract_prices.values()))
        timestamps = getattr(selected, "index", None)
        spd = (
            _infer_intraday_steps(timestamps)
            if timestamps is not None else 1)
        return SimpleNamespace(
            timestamps=timestamps,
            steps_per_day=max(1, int(spd)),
            tc_rate=float(gs.get("tc_rate", 0.0)),
            position=BacktestApp._normalize_position(
                gs.get("position", 1)),
            quantity=float(gs.get("quantity", 1.0)),
            multiplier=float(gs.get("multiplier", 0.0)),
            slippage_bps=float(gs.get("slippage_bps", 0.0)),
            force_day_close_hedge=bool(
                gs.get("force_day_close_hedge", False)),
        )

    @staticmethod
    def _load_full_history_for_recommendation(gs, base_bt):
        """返回未被单个期权期限裁剪的完整历史价格。"""
        BacktestApp._validate_history_recommendation_source(gs)
        if gs.get("source") == "wind":
            from pricing.wind_data import classify_wind_history_code
            classification = classify_wind_history_code(gs.get("wind_code"))
            if classification["mode"] == "product_pool":
                return BacktestApp._load_wind_contract_history_pool(gs)
        retained_meta = getattr(base_bt, "_gui_meta", {}) or {}
        retained_source = retained_meta.get("source")
        if retained_source is not None and retained_source != gs.get("source"):
            raise ValueError(
                "历史行情缓存来源与本次任务来源不一致，已拒绝继续择优。")
        retained = getattr(base_bt, "_full_price_history", None)
        if retained is not None:
            return retained.copy()

        src = gs.get("source")
        if src == "csv":
            import pandas as pd
            path = gs.get("csv_path")
            frame = pd.read_csv(path, parse_dates=[0], index_col=0)
            price_col = gs.get("csv_col", "close")
            if price_col not in frame.columns:
                raise ValueError(
                    f"列 {price_col!r} 不在 CSV 中，可用列: {list(frame.columns)}")
            return frame[price_col].dropna().astype(float)
        if src == "wind":
            from pricing.wind_data import classify_wind_history_code
            classification = classify_wind_history_code(gs["wind_code"])
            adjust = "" if classification.get("is_futures_contract") else "F"
            bar_label = gs.get("wind_bar_size", "日频")
            if bar_label == "日频":
                from pricing.wind_data import get_close_prices
                return get_close_prices(
                    gs["wind_code"], gs["wind_start"], gs["wind_end"], adjust)
            from pricing.wind_data import get_intraday_close
            bar_size = bar_label.removesuffix("min")
            return get_intraday_close(
                gs["wind_code"], gs["wind_start"], gs["wind_end"],
                bar_size=bar_size, adjust=adjust,
            )
        # 来源白名单已在函数入口验证；保留防御性分支以免未来新增来源时
        # 未同步实现完整历史读取。
        raise ValueError(f"尚未实现数据来源 {src!r} 的完整历史读取。")

    @staticmethod
    def _history_recommendation_source_label(gs, history=None):
        """生成随任务快照传递的真实行情来源标签，避免渲染时读取 GUI。"""
        BacktestApp._validate_history_recommendation_source(gs)
        if gs["source"] == "csv":
            filename = os.path.basename(gs.get("csv_path", "")) or "未命名文件"
            return f"CSV · {filename} · {gs.get('csv_col', 'close')}"
        if isinstance(history, ContractHistoryPool):
            failed_count = len(history.contract_load_errors)
            failed_text = f"（另 {failed_count} 个加载失败）" if failed_count else ""
            mapping_asof = str(history.main_contract_by_date.index[-1])[:10]
            return (
                f"Wind 品种样本池 · {history.product_code} · 请求截至 "
                f"{gs.get('wind_end', '—')} · 主力映射截至 {mapping_asof} = "
                f"{history.main_contract_asof} · "
                f"{len(history.contract_prices)} 个历史合约{failed_text} · "
                f"{gs.get('wind_bar_size', '日频')}")
        if gs["source"] == "wind":
            from pricing.wind_data import classify_wind_history_code
            classification = classify_wind_history_code(gs.get("wind_code"))
            if classification.get("is_futures_contract"):
                return (
                    f"Wind 单合约（不自动汇集） · {classification['code']} · "
                    f"{gs.get('wind_start', '—')} 至 {gs.get('wind_end', '—')} · "
                    f"{gs.get('wind_bar_size', '日频')}")
        return (
            f"Wind · {gs.get('wind_code', '—')} · "
            f"{gs.get('wind_start', '—')} 至 {gs.get('wind_end', '—')} · "
            f"{gs.get('wind_bar_size', '日频')}"
        )

    def _history_recommendation_worker(self, gs):
        try:
            BacktestApp._validate_history_recommendation_source(gs)
            history_lookbacks = BacktestApp._normalize_history_lookbacks(
                gs.get("history_lookbacks"))
            # 排名依据在任务启动时随其它参数一起冻结，结果页据此展示。
            objective = gs.get(
                "history_objective", DEFAULT_SELECTION_OBJECTIVE)
            if objective not in SELECTION_OBJECTIVES:
                raise ValueError(f"未知排名依据: {objective!r}")
            original_option = gs["cfg"]["build"](gs["subtype"], gs["params"])
            is_product_pool = False
            if gs.get("source") == "wind":
                from pricing.wind_data import classify_wind_history_code
                is_product_pool = (
                    classify_wind_history_code(gs.get("wind_code"))["mode"]
                    == "product_pool")

            if is_product_pool:
                # 品种代码直接读取逐日主力映射和具体合约，不把连续合约行情
                # 作为隐式前置依赖，也不重复下载整段连续分钟数据。
                history = self._load_full_history_for_recommendation(gs, None)
                if not isinstance(history, ContractHistoryPool):
                    raise TypeError("期货品种代码未返回历史具体合约样本池")
                base_bt = BacktestApp._contract_pool_backtest_context(gs, history)
                cases, notes = self._strategy_cases_for_history(gs, base_bt)
                kwargs = self._comparison_backtest_kwargs(base_bt)
                # 各历史合约可能有不同的完整日 Bar 数；由样本池逐合约推导。
                kwargs.pop("steps_per_day", None)
                recommendations, ranking, window_results = (
                    recommend_by_contract_history_pool(
                        original_option, history, cases, kwargs,
                        lookbacks=history_lookbacks,
                        objective=objective,
                    )
                )
            else:
                # CSV、股票/ETF 与明确具体合约仍是单一标的历史；建立一条与
                # 当前策略无关的基准行情/期权对象以复用其真实采样信息。
                base_state = copy.deepcopy(gs)
                base_state["strategy_name"] = "close_to_close"
                base_bt = self._build_backtest(base_state)
                cases, notes = self._strategy_cases_for_history(gs, base_bt)
                kwargs = self._comparison_backtest_kwargs(base_bt)
                history = self._load_full_history_for_recommendation(gs, base_bt)
                recommendations, ranking, window_results = (
                    recommend_by_rolling_history(
                        original_option, history, cases, kwargs,
                        lookbacks=history_lookbacks,
                        steps_per_day=base_bt.steps_per_day,
                        objective=objective,
                    )
                )
            BacktestApp._validate_history_recommendation_payload(
                recommendations, ranking, window_results)
            # 趁 bar 级结果还在手里落盘：渲染之后它们就被释放了，之后想看
            # 某段明细只能重跑（620 ms/段）。现在写一次约 10 ms/段，读回
            # 只要 3 ms。仍在 worker 线程里，不挡界面。
            BacktestApp._cache_history_bars(window_results)
            source_label = BacktestApp._history_recommendation_source_label(
                gs, history)

            self.after(
                0,
                lambda: self._deliver_history_recommendation(
                    recommendations, ranking, notes, window_results,
                    source_label, gs),
            )
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(0, lambda message=err_msg: self._fail_history_recommendation(
                message))

    def _deliver_history_recommendation(
            self, recommendations, ranking, notes=None, window_results=None,
            source_label=None, history_state=None):
        """在主线程完成渲染；只有视图真正构建成功后才报告完成。"""
        success = False
        try:
            BacktestApp._validate_history_recommendation_payload(
                recommendations, ranking, window_results)
            self._show_history_recommendation(
                recommendations, ranking, notes, source_label,
                window_results, history_state)
            self._latest_history_state = (
                BacktestApp._copy_snapshot_gui_state(history_state)
                if history_state is not None else None)
            self._latest_history_source_label = source_label
            success = True
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            messagebox.showerror("历史择优展示失败", err_msg)
        finally:
            self._finish_history_recommendation(success)

    def _fail_history_recommendation(self, message):
        messagebox.showerror("历史择优失败", message)
        self._finish_history_recommendation(False)

    def _finish_history_recommendation(self, success=True):
        self._finish_job(
            "history", success=success,
            success_text="策略优选完成  |  可在排名表选中策略后应用到左侧参数",
            failure_text="历史择优失败  |  请查看错误信息",
        )

    @staticmethod
    def _parse_wind_date(value, label):
        """严格解析 GUI 的 Wind 日期字段，避免错误延迟到联网请求后。"""
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{label}不能为空。")
        try:
            parsed = datetime.date.fromisoformat(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}格式应为 YYYY-MM-DD，当前为 {text!r}。") from exc
        if parsed.isoformat() != text:
            raise ValueError(f"{label}格式应为 YYYY-MM-DD，当前为 {text!r}。")
        return parsed

    @staticmethod
    def _maturity_days_from_params(params):
        """读取各期权结构统一意义下的剩余交易日数。"""
        value = params.get("T_days")
        if value is None:
            value = params.get("T")
        try:
            days = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("无法从期权参数解析剩余期限。") from exc
        if days <= 0:
            raise ValueError("期权剩余期限必须大于 0 个交易日。")
        return days

    @staticmethod
    def _history_realized_warmup_days(state):
        """返回历史候选在 Day 0 前必须具备的 realized-HV 预热日数。

        界面已不再提供 σ 来源选项，``_collect_history_state`` 恒写入
        ``implied``，因此当前返回值恒为 0。保留这条判断是因为回测引擎仍
        支持 realized，而分析层用的是 ``strict_sigma_warmup=True``——一旦
        有 realized 状态从别处（旧配置、后端直调）流进来却没预热区间，会
        直接抛错而不是降级，那时静默返回 0 比现在更难查。
        """
        if not bool(state.get("history_include_band", False)):
            return 0
        if str(state.get("sigma_source", "implied")) != "realized":
            return 0
        try:
            days = int(state.get("sigma_window", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("历史候选 HV 回看期必须是正整数。") from exc
        if days < 2:
            raise ValueError("历史候选 HV 回看期必须至少为 2 日。")
        return days

    @staticmethod
    def _history_auto_wind_start(asof, *, required_trade_days):
        """自动模式下的历史行情起始日。

        界面回填与真正发出的 Wind 请求共用这一份推算。分成两处各算一套的
        话，置灰框里显示的日期和实际取数区间会各走各的，而那个框看上去正
        是"本次从哪天开始取"。
        """
        span = BacktestApp._calendar_span_for_trading_days(required_trade_days)
        return asof - datetime.timedelta(days=span)

    @staticmethod
    def _validate_sigma_input(value, label):
        """波动率必须是大于 0 的有限数值。

        σ=0 不是"没有波动"这么无害：Black-Scholes 的 d1/d2 要除以
        ``σ√T``，σ=0 时直接除零，Greeks 全部退化成 NaN 或 0；模拟路径也塌
        成一条确定性曲线，多路径统计的每条路径完全相同。回测照样跑完、界面
        照样出数，只是那些数字没有任何意义——所以必须在启动前就拦掉，而不是
        让用户对着一屏 0.0000 猜哪里不对。
        """
        if value is None:
            return
        try:
            sigma = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}必须是大于 0 的数值。") from exc
        if sigma == 0:
            raise ValueError(f"{label}不能为 0，必须是大于 0 的数值。")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"{label}必须是大于 0 的有限数值。")

    @staticmethod
    def _calendar_span_for_trading_days(trading_days):
        """把交易日需求换成带节假日缓冲的自然日拉取跨度。"""
        try:
            days = int(trading_days)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Wind 交易日需求必须是正整数。") from exc
        if days <= 0:
            raise ValueError("Wind 交易日需求必须是正整数。")
        return int(np.ceil(days * 365.0 / ANNUAL_DAYS)) + _WIND_DATE_BUFFER_DAYS

    _CALENDAR_FIX_HINT = (
        "请更新 data/tradingday.csv，或取消勾选「按数据截止日倒推」"
        "后手工指定建仓日。")

    @staticmethod
    def _trading_calendar_days(*, required=True):
        """返回升序交易日列表；``required=False`` 时日历不可用只返回 None。"""
        status, payload = _local_trading_calendar()
        if status == "ok":
            return payload
        if required:
            raise ValueError(
                f"无法读取交易日历（{payload}），不能把数据截止日倒推成"
                f"建仓日。{BacktestApp._CALENDAR_FIX_HINT}")
        return None

    @staticmethod
    def _latest_trading_day(reference):
        """返回不晚于 ``reference`` 的最近交易日。

        只用于预填与提示：日历缺失或尚未覆盖到 ``reference`` 时原样返回，
        不假装知道更近的交易日，也不为一个默认值去联网。
        """
        days = BacktestApp._trading_calendar_days(required=False)
        if not days or reference < days[0] or reference > days[-1]:
            return reference
        return days[bisect.bisect_right(days, reference) - 1]

    @staticmethod
    def _entry_date_from_asof(asof, maturity_days):
        """把数据截止日按交易日历倒推 ``maturity_days`` 个交易日。

        单次回测把行情首日当作建仓日、之后严格 T 个交易日到期（见
        ``HedgeBacktest`` 的评估窗口裁剪），因此这里不能沿用历史页那套
        “自然日跨度 + 节假日缓冲”的取数区间估计：多取的缓冲会整体前移
        建仓日，让期权在截止日之前就到期，界面上却仍写着截止日。

        返回 ``(建仓日, 落到交易日的截止日)``：截止日本身可能是周末或节假日，
        必须先落到不晚于它的最近交易日，倒推出来的两端才都是真实交易日。
        """
        days = BacktestApp._trading_calendar_days()
        if asof < days[0] or asof > days[-1]:
            raise ValueError(
                f"交易日历未覆盖数据截止日 {asof.isoformat()}（可用区间 "
                f"{days[0].isoformat()} ~ {days[-1].isoformat()}）。"
                f"{BacktestApp._CALENDAR_FIX_HINT}")
        end_index = bisect.bisect_right(days, asof) - 1
        if end_index < maturity_days:
            raise ValueError(
                f"交易日历在 {days[end_index].isoformat()} 之前不足 "
                f"{maturity_days} 个交易日，无法倒推建仓日。"
                f"{BacktestApp._CALENDAR_FIX_HINT}")
        return days[end_index - maturity_days], days[end_index]

    @staticmethod
    def _local_trading_sessions(wind_code):
        """只读本地可确定规则的交易时段；无代码或元数据未知时返回 None。

        GUI 主线程的参数联动会频繁调用，因此固定 ``allow_wind=False``，
        绝不为了解析一个下拉值去启动 Wind 终端或发 ``wss``。
        """
        if not str(wind_code or "").strip():
            return None
        from pricing.wind_data import get_trading_session_clock_ranges
        return get_trading_session_clock_ranges(
            str(wind_code).strip(), allow_wind=False)

    @staticmethod
    def _recommended_fixed_time_bar_size(fixed_times, *, sessions=None):
        """返回能稳健覆盖目标 HH:MM 的推荐分钟粒度。

        Wind 的 bar 标签约定随粒度而变（实测 510050.SH）：1 分钟用左标签
        （09:30~14:59），5/15/60 分钟用右标签（09:35/09:45/10:30~15:00）。
        因此 session 开盘那一刻只有 1 分钟粒度取得到——整除判据只看墙钟，
        会把 09:30 误判成 15 分钟可覆盖。
        """
        strategy = FixedTimeStrategy(fixed_times)
        if sessions is not None:
            strategy.set_trading_sessions(sessions)
        parsed = strategy.effective_times
        opens = {start for start, _end in (strategy.trading_sessions or ())}
        if any(value in opens for value in parsed):
            return _WIND_FIXED_TIME_BAR_LABELS[-1]
        minute_marks = tuple(t.hour * 60 + t.minute for t in parsed)
        for label in _WIND_FIXED_TIME_BAR_LABELS:
            bar_minutes = _WIND_BAR_MINUTES[label]
            if all(mark % bar_minutes == 0 for mark in minute_marks):
                return label
        return _WIND_FIXED_TIME_BAR_LABELS[-1]

    @staticmethod
    def _recommended_band_bar_size():
        """价格波动触发一律用最细粒度观察穿带。

        不按带宽分档：带宽宽窄不改变“bar 内穿带后回落看不见”这一机制，
        只改变漏采的绝对数量，因此没有哪一档粗粒度是无条件安全的；分档
        还会让同一个候选的评分依赖“本批次里最窄的候选是谁”。
        """
        return _WIND_BAND_BAR_LABEL

    @staticmethod
    def _resolve_wind_bar_size(
            requested, *, strategy_name=None, fixed_times="",
            include_fixed_times=False, include_band=False, wind_code=None):
        """把“自动（推荐）”解析成实际 Wind 日频或分钟 BarSize。"""
        requested = str(requested or WIND_AUTO_BAR_SIZE).strip()
        if requested not in WIND_BAR_SIZE_OPTIONS:
            raise ValueError(f"未知 Wind 行情采样粒度: {requested!r}")

        needs_fixed = bool(
            include_fixed_times or strategy_name == "fixed_times")
        needs_band = bool(include_band or strategy_name == "hedge_band")
        # 交易时段只解析一次，供开盘瞬间拦截和固定时刻推荐共用。
        sessions = (
            BacktestApp._local_trading_sessions(wind_code)
            if needs_fixed else None)
        if requested != WIND_AUTO_BAR_SIZE:
            if requested == "日频" and needs_fixed:
                raise ValueError(
                    "固定时刻策略需要分钟行情，收到日频。粒度已改为按策略"
                    "自动推导，显式传入日频的调用方需要改传分钟粒度或"
                    f"{WIND_AUTO_BAR_SIZE!r}。")
            return requested

        recommendations = []
        if needs_band:
            recommendations.append(
                BacktestApp._recommended_band_bar_size())
        if needs_fixed:
            recommendations.append(
                BacktestApp._recommended_fixed_time_bar_size(
                    fixed_times, sessions=sessions))
        if not recommendations:
            return "日频"
        return min(
            recommendations,
            key=lambda label: _WIND_BAR_MINUTES[label],
        )

    @staticmethod
    def _resolve_single_wind_state(state, *, today=None):
        """解析单次回测的数据截止日、倒推建仓日和实际行情粒度。

        日期主控是截止日（默认最新已收盘交易日），建仓日按期权期限往前倒推——与
        策略优选一致：那边也是从分析截至日向前回放严格区间。往后推的旧口径
        会把建仓日钉在过去某天、再拿期限去凑结束日，用户想问的“现在这套
        参数在最近一个完整期限上表现如何”反而要自己反算日期。
        """
        resolved = dict(state)
        if resolved.get("source") != "wind":
            return resolved
        if not str(resolved.get("wind_code", "")).strip():
            raise ValueError("Wind 代码不能为空。")

        today = today or datetime.date.today()
        if not isinstance(today, datetime.date):
            today = BacktestApp._parse_wind_date(today, "当前日期")
        asof = BacktestApp._parse_wind_date(
            resolved.get("wind_end"), "Wind 数据截止日")
        if asof > today:
            raise ValueError("Wind 数据截止日不能晚于当前日期。")

        maturity_days = BacktestApp._maturity_days_from_params(
            resolved.get("params", {}))
        # 缺少开关的调用方（旧 API / 测试替身）自带完整两端日期，保持显式区间。
        auto_start = bool(resolved.get("wind_auto_start", False))
        if auto_start:
            start, asof = BacktestApp._entry_date_from_asof(
                asof, maturity_days)
        else:
            start = BacktestApp._parse_wind_date(
                resolved.get("wind_start"), "Wind 建仓日")
            if start > today:
                raise ValueError("Wind 建仓日不能晚于当前日期。")
        if asof <= start:
            raise ValueError("Wind 数据截止日必须晚于建仓日。")

        requested = resolved.get(
            "wind_bar_size_requested",
            resolved.get("wind_bar_size", WIND_AUTO_BAR_SIZE),
        )
        actual_bar_size = BacktestApp._resolve_wind_bar_size(
            requested,
            strategy_name=resolved.get("strategy_name"),
            fixed_times=resolved.get("fixed_times", ""),
            wind_code=resolved.get("wind_code"),
        )
        resolved.update({
            "wind_start": start.isoformat(),
            "wind_end": asof.isoformat(),
            "wind_auto_start": auto_start,
            "wind_bar_size_requested": str(requested),
            "wind_bar_size": actual_bar_size,
            "wind_date_mode": (
                "auto_entry_from_asof" if auto_start else "custom_entry_date"),
            "wind_required_trade_days": maturity_days + 1,
        })
        return resolved

    @staticmethod
    def _resolve_history_wind_state(state, *, today=None):
        """解析历史择优独立的截至日、所选周期范围和统一行情粒度。"""
        resolved = dict(state)
        if resolved.get("source") != "wind":
            return resolved
        if not str(resolved.get("wind_code", "")).strip():
            raise ValueError("Wind 代码不能为空。")

        today = today or datetime.date.today()
        if not isinstance(today, datetime.date):
            today = BacktestApp._parse_wind_date(today, "当前日期")
        asof = BacktestApp._parse_wind_date(
            resolved.get("history_wind_asof", resolved.get("wind_end")),
            "历史分析截至日",
        )
        if asof > today:
            raise ValueError("历史分析截至日不能晚于当前日期。")

        warmup_days = BacktestApp._history_realized_warmup_days(resolved)
        history_lookbacks = BacktestApp._normalize_history_lookbacks(
            resolved.get("history_lookbacks"))
        max_lookback_days = max(history_lookbacks.values())
        # L 与 V 是评分/估计所需交易日；额外 1 日只作为首日建仓的
        # Day 0 收盘锚点，不计入历史评分区间。
        required_trade_days = max_lookback_days + warmup_days + 1
        auto_start = bool(resolved.get("history_wind_auto_start", True))
        if auto_start:
            start = BacktestApp._history_auto_wind_start(
                asof, required_trade_days=required_trade_days)
        else:
            start = BacktestApp._parse_wind_date(
                resolved.get("history_wind_start", resolved.get("wind_start")),
                "历史行情起始日",
            )
        if start >= asof:
            raise ValueError("历史行情起始日必须早于分析截至日。")

        requested = resolved.get(
            "history_wind_bar_size_requested",
            resolved.get("wind_bar_size_requested",
                         resolved.get("wind_bar_size", WIND_AUTO_BAR_SIZE)),
        )
        actual_bar_size = BacktestApp._resolve_wind_bar_size(
            requested,
            fixed_times=resolved.get("fixed_times", ""),
            include_fixed_times=resolved.get(
                "history_include_fixed_times", False),
            include_band=resolved.get("history_include_band", False),
            wind_code=resolved.get("wind_code"),
        )
        if not auto_start:
            date_mode = "history_custom_range"
        elif max_lookback_days == int(ANNUAL_DAYS):
            date_mode = "history_auto_year_strict_interval"
        else:
            date_mode = "history_auto_selected_strict_interval"
        resolved.update({
            "wind_start": start.isoformat(),
            "wind_end": asof.isoformat(),
            "wind_bar_size": actual_bar_size,
            "history_wind_asof": asof.isoformat(),
            "history_wind_start": start.isoformat(),
            "history_wind_auto_start": auto_start,
            "history_wind_bar_size_requested": str(requested),
            "wind_bar_size_requested": str(requested),
            "wind_date_mode": date_mode,
            "wind_required_trade_days": required_trade_days,
            "wind_sigma_warmup_days": warmup_days,
            "history_lookbacks": history_lookbacks,
            "history_max_lookback_days": max_lookback_days,
        })
        return resolved

    @staticmethod
    def _gui_steps_per_day(source, simulate_value):
        """GUI 仅允许模拟路径选择采样密度；真实行情返回自动占位值 1。"""
        if source != "simulate":
            return 1
        value = int(simulate_value or 1)
        if value <= 0:
            raise ValueError("模拟采样 bar/日必须大于 0。")
        return value

    @staticmethod
    def _normalize_position(value):
        """把所有 GUI / worker 方向入口收敛为唯一的 ``±1`` 表示。"""
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("头寸方向只允许 1（卖出）或 -1（买入）。")
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "头寸方向只允许 1（卖出）或 -1（买入）。") from exc
        if not np.isfinite(normalized) or normalized not in (-1.0, 1.0):
            raise ValueError("头寸方向只允许 1（卖出）或 -1（买入）。")
        return int(normalized)

    def _collect_gui_state(self):
        """收集单次回测状态；不读取或校验任何历史择优控件。"""
        state = BacktestApp._collect_gui_state_for_strategy(self)
        return BacktestApp._resolve_single_wind_state(state)

    def _collect_gui_state_for_strategy(self, strategy_name=None):
        """收集公共回测环境，并只读取指定策略的专属参数。"""
        cls_name = self._class_var.get()
        cfg = OPTION_CLASSES[cls_name]
        subtype_display = self._subtype_var.get()
        subtype = SUBTYPE_FROM_DISPLAY.get(subtype_display, subtype_display)

        params = {}
        param_labels = {spec[0]: spec[1] for spec in cfg["params"]}
        for key, (var, dtype, choices) in self._param_entries.items():
            val_str = var.get().strip()
            if not val_str:
                raise ValueError(f"{param_labels.get(key, key)} 不能为空。")
            if choices and val_str in choices:
                params[key] = choices[val_str]          # 选了预设项
            elif choices:
                # 可编辑下拉里手填的自定义值：按 dtype 解析
                params[key] = float(val_str) if dtype == float else int(val_str)
            elif dtype == float:
                params[key] = float(val_str)
            elif dtype == int:
                params[key] = int(val_str)
            else:
                params[key] = val_str

        BacktestApp._validate_sigma_input(
            params.get("sigma"), param_labels.get("sigma", "波动率"))

        if cls_name == "雪球期权 (Snowball)":
            margin = float(params.get("margin", 0.0))
            if margin < 0:
                raise ValueError("保证金比例必须 >= 0。")
            if not bool(params.get("margin_call", 1)) and margin <= 0:
                raise ValueError("不追保模式必须设置正的保证金比例，用于定义最大亏损封顶。")

        if strategy_name is None:
            strategy_name = STRATEGY_FROM_DISPLAY.get(
                self._strategy_var.get(), self._strategy_var.get())
        if strategy_name not in STRATEGY_DISPLAY:
            raise ValueError(f"未知对冲策略: {strategy_name}")

        interval_type = self._band_last_edited
        needs_band = strategy_name == "hedge_band"
        price_interval = (
            float(self._price_interval_var.get()) if needs_band else 1.0)
        sigma_source = "implied"
        sigma_window = 20
        if needs_band and interval_type == "sigma":
            sigma_source = SIGMA_SOURCE_FROM_DISPLAY.get(
                self._sigma_src_var.get(), self._sigma_src_var.get())
            sigma_window = int(self._sigma_win_var.get())
            if sigma_window < 2:
                raise ValueError("历史波动率回看期必须至少为 2 日。")

        fixed_times = (self._fixed_times_var.get().strip()
                       if strategy_name == "fixed_times" else "")
        source = self._source_var.get()
        real_vol = self._real_vol_var.get().strip()
        # 留空是有意义的默认（同隐含），0 或负数不是：那会让所有模拟路径
        # 退化成同一条确定性曲线。只在模拟来源校验——CSV / Wind 根本不读
        # 这个框，不该被里面的历史残值挡住。
        if source == "simulate" and real_vol:
            BacktestApp._validate_sigma_input(real_vol, "已实现波动率")
        # 真实 CSV/Wind 的 bar 数由后端按时间索引和交易时段自动推导；
        # 只有模拟路径需要用户选择离散粒度。
        steps_per_day = self._gui_steps_per_day(source, self._spd_var.get())
        force_close_var = getattr(
            self, "_force_day_close_hedge_var", None)
        force_day_close_hedge = bool(
            force_close_var.get()) if force_close_var is not None else False
        wind_auto_start_var = getattr(self, "_wind_auto_start_var", None)
        wind_auto_start = bool(
            wind_auto_start_var.get()
        ) if wind_auto_start_var is not None else False
        # 粒度不再可选；解析阶段按当前策略推导实际值。
        wind_bar_size_requested = WIND_AUTO_BAR_SIZE

        return {
            "cls_name": cls_name,
            "cfg": cfg,
            "subtype": subtype,
            "params": params,
            "source": source,
            "tc_rate": float(self._tc_var.get()) / 100.0,
            # 左侧方向控件是唯一来源；任务启动时冻结并规范成严格 ±1。
            "position": BacktestApp._normalize_position(
                self._pos_var.get()),
            "quantity": float(self._qty_var.get()),
            "multiplier": float(self._mult_var.get()),
            "s0": str(params.get("s0", "")),
            "seed": self._seed_var.get().strip(),
            "real_vol": real_vol,
            "n_paths": self._npaths_var.get().strip(),
            "csv_path": self._csv_path_var.get().strip(),
            "csv_col": self._csv_col_var.get().strip() or "close",
            "wind_code": self._wind_code_var.get().strip(),
            "wind_start": self._wind_start_var.get().strip(),
            "wind_end": self._wind_end_var.get().strip(),
            "wind_auto_start": wind_auto_start,
            "wind_bar_size_requested": wind_bar_size_requested,
            # 未经过上下文 resolver 时暂存用户选择；单次/历史入口会在启动
            # 任务前把“自动（推荐）”替换为实际粒度。
            "wind_bar_size": wind_bar_size_requested,
            # --- 新增：对冲策略与 intraday / 滑点 ---
            "strategy_name": strategy_name,
            "sigma_source": sigma_source,
            "sigma_window": sigma_window,
            "steps_per_day": steps_per_day,
            "slippage_bps": float(self._slip_var.get() or 0.0),
            "fixed_times": fixed_times,
            "price_interval": price_interval,
            "interval_type": interval_type,
            "force_day_close_hedge": force_day_close_hedge,
        }

    def _collect_history_state(self):
        """收集历史择优的独立候选配置，并冻结当前公共回测环境。"""
        history_lookbacks = BacktestApp._history_lookbacks_from_controls(self)
        # 每日收盘是历史图表的固定 每日收盘 基准，UI 中只读且状态永远为真。
        include_close = True
        include_fixed_times = bool(
            self._history_include_fixed_times_var.get())
        include_band = bool(self._history_include_band_var.get())
        if not (include_fixed_times or include_band):
            raise ValueError(
                "历史择优会固定运行每日收盘基准；"
                "请至少再启用一种候选策略（固定时刻或固定间隔）。")
        include_current_band = (
            include_band
            and bool(self._history_include_current_band_var.get()))
        # 当前带宽是显式可选候选；只在确实加入时强制同步。
        if include_band and include_current_band:
            self._sync_band_inputs(self._band_last_edited, strict=True)

        # 历史基准与左侧当前单策略无关；用每日收盘只收集公共环境，
        # 避免未参与候选的单次策略隐藏输入阻断历史任务。
        state = BacktestApp._collect_gui_state_for_strategy(
            self, "close_to_close")
        BacktestApp._validate_history_recommendation_source(state)
        # 历史择优与单次回测共用左侧唯一的收盘兜底开关；任务启动时
        # _collect_gui_state_for_strategy 已把其当前值冻结进本次状态。
        force_day_close_hedge = bool(
            state.get("force_day_close_hedge", False))

        fixed_times = ""
        if include_fixed_times:
            fixed_times = self._history_fixed_times_var.get().strip()
            FixedTimeStrategy(fixed_times)

        band_candidates = ()
        # 候选带宽的 σ 恒为左侧输入的波动率：全部候选共用同一个 σ，排名
        # 才是在比带宽倍数本身。引擎仍支持 realized，只是不再从这里进入。
        sigma_source = "implied"
        sigma_window = 20
        if include_band:
            band_candidates = BacktestApp._parse_band_candidate_sigmas(
                self._history_band_candidate_sigmas_var.get())
            if not band_candidates and not include_current_band:
                raise ValueError(
                    "固定间隔已勾选，请至少输入一个 σ 候选，"
                    "或勾选『加入当前回测带宽』。")

        state.update({
            "history_lookbacks": history_lookbacks,
            "history_include_close": include_close,
            "history_include_fixed_times": include_fixed_times,
            "history_include_band": include_band,
            "history_include_current_band": include_current_band,
            "fixed_times": fixed_times,
            "band_candidate_sigmas": band_candidates,
            "sigma_source": sigma_source,
            "sigma_window": sigma_window,
            # 无论当前单次回测选中什么策略，历史页都显式读取
            # 最后编辑的带宽原始单位与数值，仅用于“加入当前带宽”。
            "interval_type": (
                self._band_last_edited if include_current_band else "sigma"),
            "price_interval": (
                float(self._price_interval_var.get())
                if include_current_band else 1.0),
            # 左侧公共开关统一作用于本次历史实验的全部候选。
            "force_day_close_hedge": force_day_close_hedge,
            "history_objective": BacktestApp._history_objective_from_controls(
                self),
        })
        history_asof_var = getattr(self, "_history_wind_asof_var", None)
        history_start_var = getattr(self, "_history_wind_start_var", None)
        history_auto_start_var = getattr(
            self, "_history_wind_auto_start_var", None)
        state.update({
            "history_wind_asof": (
                history_asof_var.get().strip()
                if history_asof_var is not None else state.get("wind_end", "")),
            "history_wind_start": (
                history_start_var.get().strip()
                if history_start_var is not None else state.get("wind_start", "")),
            # 旧测试替身 / API 调用没有历史控件时保留显式起始日；真实 GUI
            # 默认自动覆盖最长已选严格区间与 Day 0 锚点。不含 HV 预热：本页
            # 的 σ 恒取左侧输入值，预热日数恒为 0（见
            # _history_realized_warmup_days）。
            "history_wind_auto_start": (
                bool(history_auto_start_var.get())
                if history_auto_start_var is not None else False),
            # 粒度按本次勾选的候选集合自动取最细的一档。
            "history_wind_bar_size_requested": WIND_AUTO_BAR_SIZE,
        })
        state = BacktestApp._resolve_history_wind_state(state)
        if include_band:
            # 在启动线程前完成当前带宽换算、去重和 10 档上限校验。
            BacktestApp._band_cases_for_history(state)
        return state

    def _backtest_worker(self, gui_state):
        try:
            bt = self._build_backtest(gui_state)
            bt.run()

            # 模拟模式下进行蒙特卡洛多路径分析
            multi_stats = None
            src = gui_state["source"]
            if src == "simulate":
                n_paths = int(gui_state.get("n_paths") or 500)
                if n_paths > 1:
                    params = gui_state["params"]
                    s0 = float(gui_state["s0"])
                    seed = int(gui_state["seed"])
                    T_days = params.get("T_days") or params.get("T") or 20
                    sigma_impl = params.get("sigma", 0.18)
                    r = params.get("r", 0.03)
                    q = params.get("q", 0.03)
                    real_vol_str = gui_state["real_vol"]
                    sigma_real = float(real_vol_str) if real_vol_str else sigma_impl
                    spd_mc = int(gui_state.get("steps_per_day", 1))
                    paths = HedgeBacktest.simulate_multi_paths(
                        s0, sigma_real, T_days, n_paths=n_paths,
                        r=r, q=q, seed=seed, steps_per_day=spd_mc)

                    # 切换为确定进度条
                    def _switch_to_determinate():
                        self._progress.stop()
                        self._progress.configure(mode="determinate", maximum=n_paths, value=0)
                        self._progress_label.configure(text=f"蒙特卡洛模拟: 0/{n_paths}")
                        self._progress_label.pack(fill="x")
                    self.after(0, _switch_to_determinate)

                    def _on_progress(done, total):
                        self.after(0, lambda d=done, t=total: (
                            self._progress.configure(value=d),
                            self._progress_label.configure(text=f"蒙特卡洛模拟: {d}/{t}"),
                        ))

                    multi_stats = bt.run_multi(paths, progress_callback=_on_progress)

            self.after(
                0,
                lambda: self._deliver_backtest_result(
                    bt, multi_stats, gui_state),
            )
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(
                0, lambda message=err_msg: self._fail_backtest(message))

    @staticmethod
    def _copy_snapshot_gui_state(gui_state):
        """去掉不可序列化的构造器，仅保留本次运行的参数快照。"""
        return copy.deepcopy({
            key: value for key, value in gui_state.items() if key != "cfg"
        })

    def _deliver_backtest_result(self, bt, multi_stats, gui_state):
        """在主线程原子地渲染并登记最新结果，避免假成功状态。"""
        success = False
        try:
            state_copy = BacktestApp._copy_snapshot_gui_state(gui_state)
            self._show_results(bt, multi_stats)
            self._latest_backtest = bt
            self._latest_backtest_state = state_copy
            self._latest_retained_result_id = None
            success = True
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            messagebox.showerror("回测结果展示失败", err_msg)
        finally:
            self._finish_run(success)

    def _fail_backtest(self, message):
        messagebox.showerror("回测失败", message)
        self._finish_run(False)

    def _set_status(self, text):
        """更新底部状态栏文字 (线程安全: 调用方需保证在主线程或通过 after)。"""
        try:
            self._status_var.set(text)
        except Exception:
            pass

    def _finish_run(self, success=True):
        self._finish_job(
            "backtest", success=success,
            success_text="回测完成  |  可保留当前结果，随后修改策略或参数继续回测",
            failure_text="回测失败  |  请查看错误信息",
        )

    @staticmethod
    def _freeze_snapshot_value(value):
        """把配置规范化为可稳定比较的不可变值。"""
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            return tuple(sorted(
                (str(key), BacktestApp._freeze_snapshot_value(item))
                for key, item in value.items()
            ))
        if isinstance(value, (list, tuple)):
            return tuple(BacktestApp._freeze_snapshot_value(item) for item in value)
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            return ("ndarray", str(array.dtype), tuple(array.shape), digest)
        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            return value.isoformat()
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @staticmethod
    def _backtest_path_key(bt, gui_state):
        """对实际价格、时间索引和交易日分组生成会话内可比签名。"""
        result = getattr(bt, "_results", {}) or {}
        prices = np.ascontiguousarray(
            np.asarray(result.get("prices", getattr(bt, "prices", [])),
                       dtype="<f8")
        )
        digest = hashlib.sha256()
        source = str(gui_state.get("source", "unknown"))
        digest.update(source.encode("utf-8"))
        digest.update(prices.tobytes())

        groups = result.get("trading_day_groups")
        if groups is not None:
            group_array = np.ascontiguousarray(np.asarray(groups, dtype="<i8"))
            digest.update(b"groups")
            digest.update(group_array.tobytes())

        timestamps = getattr(bt, "timestamps", None)
        if timestamps is not None:
            try:
                import pandas as pd
                # run() 可能因敲出提前终止；签名必须和结果价格使用同一前缀，
                # 不能把未参与本次结果的尾部时间戳混入比较上下文。
                timestamp_index = pd.DatetimeIndex(timestamps)[:len(prices)]
                timestamp_values = np.ascontiguousarray(
                    np.asarray(timestamp_index.asi8, dtype="<i8"))
                digest.update(b"timestamps")
                digest.update(timestamp_values.tobytes())
            except Exception:
                digest.update(
                    repr(tuple(timestamps)[:len(prices)]).encode("utf-8"))
        return source, int(len(prices)), digest.hexdigest()

    @staticmethod
    def _strategy_style_key(strategy_name, *, fixed_times=None, sigma=None):
        """把策略身份压成跨页稳定的配色键。

        结果对比页的曲线名是用户可改的结果名，策略优选页的是候选名，两者都
        不能直接当配色键。这里只取策略类型及其决定性参数，使同一策略在两页
        取到同一颜色和标记。
        """
        name = str(strategy_name or "").strip()
        if name == "close_to_close":
            return BASELINE_STRATEGY_STYLE_KEY
        if name == "fixed_times":
            times = ",".join(
                part.strip()
                for part in str(fixed_times or "").split(",")
                if part.strip()
            )
            return f"fixed_times:{times or '—'}"
        if name == "hedge_band":
            value = BacktestApp._comparison_finite(sigma)
            # σ 是三种带宽输入的共同换算口径；用它当键，绝对/相对输入换算到
            # 同一带宽时也能共享颜色。
            return (
                f"hedge_band:{value:.6g}σ" if value is not None
                else "hedge_band:—")
        return f"other:{name or '—'}"

    @staticmethod
    def _snapshot_origin_from_state(gui_state):
        """从单次回测状态判断快照来源。

        历史分段重放会在状态里留下 history_replay_* 字段；手工点『保留当前
        结果』时据此把它标成分段重放，而不是笼统的手工回测。
        """
        gui_state = dict(gui_state or {})
        strategy = str(
            gui_state.get("history_replay_strategy", "") or "").strip()
        if not strategy:
            return SNAPSHOT_ORIGIN_MANUAL, {}
        lookback = str(gui_state.get("history_replay_lookback", "") or "")
        return SNAPSHOT_ORIGIN_HISTORY_REPLAY, {
            "lookback": lookback,
            "period_label": dict(HISTORY_PERIOD_DEFS).get(lookback, lookback),
            "history_strategy": strategy,
            "window_id": str(
                gui_state.get("history_replay_window_id", "") or ""),
        }

    @staticmethod
    def _snapshot_style_key(gui_state):
        """从单次回测状态取配色键；σ 一律换算成日波动倍数。"""
        strategy = str(gui_state.get("strategy_name", "")).strip()
        if strategy != "hedge_band":
            return BacktestApp._strategy_style_key(
                strategy, fixed_times=gui_state.get("fixed_times"))
        band_type = gui_state.get("interval_type", "absolute")
        threshold = BacktestApp._comparison_finite(
            gui_state.get("price_interval"))
        sigma = None
        if threshold is not None:
            try:
                params = gui_state.get("params", {})
                sigma = HedgeBandStrategy.convert_threshold(
                    threshold, band_type, float(params["s0"]),
                    float(params["sigma"]),
                )["sigma"]
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                # 换算不出 σ 时退回按输入口径区分，绝不让不同带宽共享颜色。
                return BacktestApp._strategy_style_key(
                    f"hedge_band_{band_type}_{threshold:.6g}")
        return BacktestApp._strategy_style_key("hedge_band", sigma=sigma)

    @staticmethod
    def _history_baseline_row_label(strategy_label):
        """基线行在策略列里的显示名。

        基线在三种排序口径下取值都恒为 0——增量收益、增量信噪比、以及品种
        池降级时用的逐段改善率——所以它排在哪一位本身就是分界线。这里只标
        明身份，不再拼接分界说明：那句话越写越长，而同一行的 # 列与三个指
        标格已经都显示「基准」了。
        """
        return f"{strategy_label}（基准）"

    @staticmethod
    def _history_row_style_key(row):
        """从策略优选排名行取配色键，与快照侧使用同一套命名。"""
        row = dict(row or {})
        strategy_name = str(
            row.get("meta_strategy_name")
            or row.get("strategy_type")
            or "").strip()
        return BacktestApp._strategy_style_key(
            strategy_name,
            fixed_times=row.get("meta_fixed_times"),
            sigma=row.get("meta_candidate_sigma"),
        )

    def _strategy_style(self, style_key):
        """按会话内首次出现顺序固定策略配色；两页共用同一登记表。

        登记表随会话保留，因此先在策略优选页看到的候选，之后作为快照进入
        结果对比页时颜色不变。
        """
        registry = getattr(self, "_strategy_style_registry", None)
        if registry is None:
            # 基准先占位，保证 每日收盘 在两页永远是同一个颜色。
            registry = {
                BASELINE_STRATEGY_STYLE_KEY: (
                    STRATEGY_CHART_COLORS[0], STRATEGY_CHART_MARKERS[0]),
            }
            self._strategy_style_registry = registry
        key = str(style_key or BASELINE_STRATEGY_STYLE_KEY)
        if key not in registry:
            index = len(registry)
            registry[key] = (
                STRATEGY_CHART_COLORS[index % len(STRATEGY_CHART_COLORS)],
                STRATEGY_CHART_MARKERS[index % len(STRATEGY_CHART_MARKERS)],
            )
        return registry[key]

    @staticmethod
    def _strategy_snapshot_labels(gui_state):
        strategy = gui_state.get("strategy_name", "unknown")
        fallback_label = (
            "收盘兜底：开启"
            if bool(gui_state.get("force_day_close_hedge", False))
            else "收盘兜底：关闭"
        )

        def _with_fallback(parameters):
            return f"{parameters}；{fallback_label}"

        if strategy == "close_to_close":
            return "每日收盘", _with_fallback("每交易日最后一个采样点")
        if strategy == "fixed_times":
            times = str(gui_state.get("fixed_times", "")).strip() or "—"
            return "固定时刻", _with_fallback(times)
        if strategy == "hedge_band":
            band_type = gui_state.get("interval_type", "absolute")
            threshold = float(gui_state.get("price_interval", 0.0))
            if band_type == "relative":
                primary = f"相对 {threshold:.4%}"
            elif band_type == "sigma":
                primary = f"{threshold:g} 倍波动率（日波动）"
            else:
                primary = f"绝对 {threshold:g}"
            try:
                params = gui_state.get("params", {})
                converted = HedgeBandStrategy.convert_threshold(
                    threshold, band_type, float(params["s0"]),
                    float(params["sigma"]),
                )
                equivalents = (
                    f"绝对 {converted['absolute']:.6g} / "
                    f"相对 {converted['relative']:.4%} / "
                    f"{converted['sigma']:.4g}σ"
                )
                return "固定间隔", _with_fallback(
                    f"{primary}；等价 {equivalents}")
            except (KeyError, TypeError, ValueError):
                return "固定间隔", _with_fallback(primary)
        return (
            STRATEGY_DISPLAY.get(strategy, str(strategy)),
            _with_fallback("—"),
        )

    @staticmethod
    def _snapshot_source_label(gui_state):
        source = gui_state.get("source")
        if source == "simulate":
            return f"模拟 · seed {gui_state.get('seed', '—')}"
        if source == "csv":
            path = str(gui_state.get("csv_path", "")).strip()
            return f"CSV · {os.path.basename(path) or '—'}"
        if source == "wind":
            return (
                f"Wind · {gui_state.get('wind_code', '—')} · "
                f"{gui_state.get('wind_start', '—')} 至 "
                f"{gui_state.get('wind_end', '—')} · "
                f"{gui_state.get('wind_bar_size', '日频')}"
            )
        return str(source or "未知来源")

    @staticmethod
    def _snapshot_comparison_data(result):
        """在入池时缓存指标与日级曲线，避免保留和重复聚合 bar 级数组。"""
        summary_row = summarize_strategy_result(result, "snapshot")
        daily_frame = result_daily_frame(result).copy(deep=True)
        return copy.deepcopy(summary_row), daily_frame

    @staticmethod
    def _snapshot_replay_recipe(bt, gui_state):
        """把重跑这次回测所需的**输入**摘成配方。

        存价格序列本身，而不是「来源标识 + 重新取数」：
        - 模拟的路径由 ``real_vol`` 参与生成，而 ``real_vol`` 从来没进过快照；
        - CSV 文件可能被移动或删除；
        - Wind 非期货走前复权，复权因子随时间变，同一区间换个日期取回来的
          **不是同一串数**。
        存下来就都不依赖了，离线也能精确重现。

        固定时刻存的是 ``effective_times``（已按交易时段剔掉休市目标）而不是
        用户填的原值：重放时拿不到品种 session 元数据，用原值会触发逐日严格
        校验而失败——原始运行明明是成功的。
        """
        import pandas as pd
        results = getattr(bt, "_results", None) or {}
        prices = np.asarray(results.get("prices", ()), dtype=float)
        if prices.size < 2:
            return {}
        timestamps = results.get("timestamps")
        index = None
        if timestamps is not None:
            try:
                index = [pd.Timestamp(ts).isoformat()
                         for ts in pd.DatetimeIndex(timestamps)]
            except (TypeError, ValueError):
                index = None
        if index is not None and len(index) != prices.size:
            index = None
        recipe = {
            "prices": prices.tolist(),
            "index": index,
            "steps_per_day": int(results.get("steps_per_day", 1) or 1),
            "source": gui_state.get("source"),
            "cls_name": gui_state.get("cls_name"),
            "subtype": gui_state.get("subtype"),
            "params": copy.deepcopy(dict(gui_state.get("params", {}))),
            "position": BacktestApp._normalize_position(
                gui_state.get("position")),
            "quantity": gui_state.get("quantity"),
            "multiplier": gui_state.get("multiplier"),
            "tc_rate": gui_state.get("tc_rate"),
            "slippage_bps": gui_state.get("slippage_bps", 0.0),
            "force_day_close_hedge": bool(
                gui_state.get("force_day_close_hedge", False)),
            "evaluation_days": results.get("evaluation_days"),
            "form_state": BacktestApp._snapshot_form_state(gui_state),
            "fixed_times_effective": list(
                results.get("fixed_time_effective_times", ()) or ()),
            # 摘要页直接读 bt._gui_meta，重放出来的对象也得有一份。
            "gui_meta": copy.deepcopy(dict(getattr(bt, "_gui_meta", {}) or {})),
        }
        return recipe

    @staticmethod
    def _strategy_from_form_state(form_state, *, fixed_times_effective=None):
        """按冻结的表单输入重建策略对象。

        固定时刻优先用 ``effective_times``：重放时没有品种交易时段元数据，
        用用户填的原值会让休市目标重新参与逐日严格校验而失败——而原始运行
        是成功的，那一步已经把它们剔掉了。
        """
        name = str(form_state.get("strategy_name", "close_to_close"))
        if name == "close_to_close":
            return CloseToCloseStrategy()
        if name == "fixed_times":
            times = fixed_times_effective or form_state.get(
                "fixed_times", DEFAULT_FIXED_TIMES)
            return FixedTimeStrategy(times)
        if name == "hedge_band":
            return HedgeBandStrategy(
                band_type=form_state.get("interval_type", "absolute"),
                threshold=form_state.get("price_interval", 1.0),
                sigma_source=form_state.get("sigma_source", "implied"),
                window_days=form_state.get("sigma_window", 20),
            )
        raise ValueError(f"未知对冲策略: {name}")

    def _replay_saved_snapshot(self, snapshot):
        """取这条快照的逐 bar 明细，返回 ``(回测对象, 是否重算过)``。

        **正常路径是读已保存的数据，不是重跑。** 保留结果的那一刻 bar 级
        结果就已落盘（见 ``_make_saved_backtest_result``），这里按同一份配
        方算出的 key 读回来，实测 3 ms；真跑一次要几百毫秒。

        只有缓存确实不在时才回退重算——缓存被清过、换了机器、或本功能上线
        前保留的旧快照。回退与「保留结果」时那次运行同源：价格序列是存下来
        的原始切片，期权与策略按同一个 ``_rescale_option_to_real_s0`` 重定
        基，因此逐 bar 明细与当时一致，不依赖 CSV 文件是否还在、也不依赖
        Wind 能否联网。调用方要把「重算过」说出来：用户按的是「加载」。
        """
        import pandas as pd
        recipe = dict(getattr(snapshot, "replay", None) or {})
        if not recipe:
            raise ValueError(
                "这条结果没有重放配方（可能保存于本功能上线之前），"
                "无法加载明细；重新跑一次并保留即可。")
        prices = np.asarray(recipe.get("prices", ()), dtype=float)
        if prices.size < 2:
            raise ValueError("重放配方里的行情序列不完整。")
        index = recipe.get("index")
        series = (pd.Series(prices, index=pd.to_datetime(index))
                  if index else pd.Series(prices))

        cls_name = recipe.get("cls_name")
        cfg = OPTION_CLASSES.get(cls_name)
        if cfg is None:
            raise ValueError(f"未知期权大类：{cls_name}")
        option = cfg["build"](recipe.get("subtype"), dict(recipe.get("params", {})))
        strategy = BacktestApp._strategy_from_form_state(
            dict(recipe.get("form_state", {})),
            fixed_times_effective=recipe.get("fixed_times_effective") or None)

        # 真实行情当时按首价整体缩放过期权与绝对带宽，重放必须走同一步，
        # 否则行权价与带宽会停在参考价口径上，明细和当时对不上。
        if str(recipe.get("source")) in ("csv", "wind"):
            option, info = _rescale_option_to_real_s0(
                option, float(series.iloc[0]))
            strategy = _rescale_strategy_to_real_s0(strategy, info["ratio"])

        bt = HedgeBacktest(
            option, series,
            tc_rate=recipe.get("tc_rate", 0.0),
            position=recipe.get("position", 1),
            quantity=recipe.get("quantity", 1.0),
            multiplier=recipe.get("multiplier", 5),
            strategy=strategy,
            steps_per_day=int(recipe.get("steps_per_day", 1) or 1),
            slippage_bps=recipe.get("slippage_bps", 0.0),
            force_day_close_hedge=bool(
                recipe.get("force_day_close_hedge", False)),
            evaluation_days=recipe.get("evaluation_days"),
        )
        bt._gui_meta = dict(recipe.get("gui_meta", {})) or {
            "cls_name": cls_name,
            "subtype": recipe.get("subtype"),
            "source": recipe.get("source"),
        }
        cached = history_bar_cache.load_recipe(recipe)
        if cached is not None:
            # run() 只设置 _results（hedge_backtest 里唯一一处赋值），所以
            # 把结果塞进未运行的对象与真跑出来的等价。
            bt._results = cached
            return bt, False
        bt.run()
        history_bar_cache.store_recipe(recipe, dict(bt._results or {}))
        return bt, True

    def _load_saved_snapshot_detail(self):
        """右键菜单「加载明细」：重跑选中快照并渲染到结果页。"""
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshot = self._saved_backtests[result_id]
        try:
            bt, recomputed = self._replay_saved_snapshot(snapshot)
        except Exception as exc:                              # noqa: BLE001
            messagebox.showerror("加载明细失败", str(exc))
            return
        self._latest_backtest = bt
        self._show_results(bt)
        if recomputed:
            # 菜单项叫「加载明细」，正常就该是秒开的读盘。走到重算说明保存
            # 的明细不在了，得说一声，否则用户只感觉「怎么卡了一下」。
            self._set_status(
                f"已保存的明细不存在，已按配方重新计算并补存：{snapshot.name}")
        else:
            self._set_status(
                f"已加载『{snapshot.name}』的明细  |  "
                "回测摘要 / 对冲图表 / 每日明细 / 波动率分析均为该次结果")

    @staticmethod
    def _saved_snapshot_strategy_type(snapshot):
        """读取不可被用户重命名影响的快照策略类型。"""
        summary_row = getattr(snapshot, "summary_row", {}) or {}
        return str(summary_row.get("strategy_type", "")).strip()

    @staticmethod
    def _saved_snapshot_position(snapshot):
        """读取快照的经济方向；显示名称和其它参数均不能覆盖它。"""
        return BacktestApp._normalize_position(snapshot.position)

    @staticmethod
    def _saved_snapshot_origin_label(snapshot):
        """产生方式列文案。

        来源读结构化的 origin 字段，用户重命名结果不会让它丢失。
        """
        origin = str(getattr(snapshot, "origin", "") or
                     SNAPSHOT_ORIGIN_MANUAL)
        return SNAPSHOT_ORIGIN_DISPLAY.get(origin, origin)

    @staticmethod
    def _snapshot_origin_detail(snapshot):
        """把快照的结构化来源摊成一句可读的溯源说明。

        分段重放会记下它回放的是哪个周期、哪个候选、哪一段。溯源不参与任何
        排名，它只回答"这条结果是怎么来的"。
        """
        meta = getattr(snapshot, "origin_meta", None) or {}
        parts = []
        period = str(meta.get("period_label", "") or "").strip()
        if period:
            parts.append(f"周期 {period}")
        strategy = str(meta.get("history_strategy", "") or "").strip()
        if strategy:
            parts.append(f"候选『{strategy}』")
        window_id = str(meta.get("window_id", "") or "").strip()
        if window_id:
            parts.append(f"分段 {window_id}")
        return "  ·  ".join(parts)

    @staticmethod
    def _signature_keys_from_state(gui_state, *, data_digest=None,
                                   steps_per_day=None):
        """从一份 gui_state 造出行情 / 期权 / 规模与成本三组签名。

        单独抽出来是因为**策略优选的参数详情也要用它**：那边只有一份运行时
        冻结的 state、没有 bt，但要摊出的字段与结果池详情必须逐项对齐。留在
        ``_make_saved_backtest_result`` 里的话两处各写一份字典，加一个字段只
        同步一边，同一次回测在两个窗口就会显示出不一样的输入。
        """
        return (
            BacktestApp._freeze_snapshot_value({
                "source": gui_state.get("source"),
                # 价格序列本身的摘要。可读字段（标的、区间、粒度）说不清的
                # 差异由它兜底：日内滚动的两段、同一区间取到了不同数据，都能
                # 认出来。反过来，跨周期取到同一段末尾时数据逐字节相同，它也
                # 不会造出假差异——用分段编号就会，那三段编号不同而数据一模
                # 一样。优选那边跑的是多段，没有单一摘要，传 None。
                "data_digest": data_digest,
                "seed": gui_state.get("seed"),
                "csv_path": gui_state.get("csv_path"),
                "csv_col": gui_state.get("csv_col"),
                "wind_code": gui_state.get("wind_code"),
                "wind_start": gui_state.get("wind_start"),
                "wind_end": gui_state.get("wind_end"),
                "wind_bar_size": gui_state.get("wind_bar_size"),
            }),
            BacktestApp._freeze_snapshot_value({
                "cls_name": gui_state.get("cls_name"),
                "subtype": gui_state.get("subtype"),
                "params": gui_state.get("params", {}),
            }),
            BacktestApp._freeze_snapshot_value({
                "quantity": gui_state.get("quantity"),
                "multiplier": gui_state.get("multiplier"),
                "tc_rate": gui_state.get("tc_rate"),
                "slippage_bps": gui_state.get("slippage_bps", 0.0),
                "steps_per_day": steps_per_day,
            }),
        )

    @staticmethod
    def _make_saved_backtest_result(
            bt, gui_state, result_id, name, saved_at=None, *,
            origin=SNAPSHOT_ORIGIN_MANUAL, origin_meta=None, sequence=0):
        snapshot_position = BacktestApp._normalize_position(
            gui_state.get("position"))
        result_position = (getattr(bt, "_results", {}) or {}).get("position")
        if result_position is not None:
            result_position = BacktestApp._normalize_position(result_position)
            if result_position != snapshot_position:
                raise ValueError(
                    "回测结果的头寸方向与本次左侧参数不一致，已拒绝保存。")
        summary_row, daily_frame = BacktestApp._snapshot_comparison_data(
            bt._results)
        strategy_label, parameter_summary = (
            BacktestApp._strategy_snapshot_labels(gui_state))
        path_key = BacktestApp._backtest_path_key(bt, gui_state)
        market_key, contract_key, economics_key = (
            BacktestApp._signature_keys_from_state(
                gui_state,
                data_digest=path_key[2],
                steps_per_day=bt._results.get("steps_per_day"),
            ))
        subtype = gui_state.get("subtype", "—")
        replay_recipe = BacktestApp._snapshot_replay_recipe(bt, gui_state)
        # 保留的这一刻 bar 级结果还在手里，顺手落盘：之后点「加载明细」就
        # 不必按配方重跑（实测 3 ms vs 620 ms）。与策略优选共用同一套内容
        # 寻址缓存，写失败只是白跑一次，不影响保留本身。
        history_bar_cache.store_recipe(
            replay_recipe, dict(getattr(bt, "_results", None) or {}))
        return SavedBacktestResult(
            result_id=result_id,
            name=name,
            saved_at=saved_at or datetime.datetime.now(),
            summary_row=summary_row,
            daily_frame=daily_frame,
            strategy_label=strategy_label,
            parameter_summary=parameter_summary,
            source_label=BacktestApp._snapshot_source_label(gui_state),
            option_label=SUBTYPE_DISPLAY.get(subtype, str(subtype)),
            path_key=path_key,
            market_key=market_key,
            contract_key=contract_key,
            economics_key=economics_key,
            position=snapshot_position,
            style_key=BacktestApp._snapshot_style_key(gui_state),
            origin=str(origin or SNAPSHOT_ORIGIN_MANUAL),
            origin_meta=copy.deepcopy(dict(origin_meta or {})),
            form_state=BacktestApp._snapshot_form_state(gui_state),
            sequence=int(sequence or 0),
            replay=replay_recipe,
            bar_size_requested=str(
                gui_state.get("wind_bar_size_requested", "") or ""),
            intraday_steps=BacktestApp._effective_intraday_steps(bt),
        )

    @staticmethod
    def _effective_intraday_steps(bt):
        """本次回测**实际**的每交易日 bar 数。

        不能只读 ``_results["steps_per_day"]``：那是 GUI 传给引擎的值，而
        ``_gui_steps_per_day`` 对真实行情一律返回占位 1（CSV 分支干脆传
        ``None``），引擎再自己从时间索引推断并取两者较大值。照那个值展示，
        一条 15 分钟 Wind 回测会写着「日内采样 1」，而它每天实际十几根 bar。

        推不出来时退回传入值——模拟路径的索引常常不是 DatetimeIndex，而它那
        个值本来就是用户直接填的真实采样密度。
        """
        results = getattr(bt, "_results", None) or {}
        declared = max(1, int(results.get("steps_per_day", 1) or 1))
        timestamps = getattr(bt, "timestamps", None)
        if timestamps is None or len(timestamps) < 2:
            return declared
        try:
            inferred = int(_infer_intraday_steps(timestamps))
        except (AttributeError, TypeError, ValueError):
            return declared
        return max(declared, inferred)

    @staticmethod
    def _snapshot_source(snapshot):
        """快照的行情来源；取不到时 ``None``。"""
        flat = BacktestApp._flatten_signature(
            tuple(getattr(snapshot, "market_key", ()) or ()))
        source = flat.get(("source",))
        return source if isinstance(source, str) else None

    @staticmethod
    def _snapshot_form_state(gui_state):
        """摘出可回填左侧表单的对冲策略输入，保持运行时的原始单位。

        只取对冲策略这一组：期权结构、方向、成本和行情来源改动面太大，回填
        它们等于把整张表单换掉，而用户点这个按钮想做的是接着这条结果继续调
        对冲参数。
        """
        gui_state = dict(gui_state or {})
        return {
            "strategy_name": str(gui_state.get("strategy_name", "") or ""),
            "fixed_times": str(gui_state.get("fixed_times", "") or ""),
            "interval_type": str(
                gui_state.get("interval_type", "") or "absolute"),
            "price_interval": BacktestApp._comparison_finite(
                gui_state.get("price_interval")),
            "force_day_close_hedge": bool(
                gui_state.get("force_day_close_hedge", False)),
            # 波动率口径决定带宽怎么换算，实际改变回测结果（它直接传给
            # HedgeBandStrategy）。此前它不在快照的任何一处：两条只差这个
            # 口径的结果，逐字段比下来会是"完全一致"，而数字明明不同。
            "sigma_source": str(
                gui_state.get("sigma_source", "") or "implied"),
            "sigma_window": BacktestApp._comparison_safe_int(
                gui_state.get("sigma_window"), 20),
        }

    # 每种对冲策略实际读取的 form_state 字段。用于对比签名——回填不需要它，
    # 回填要的是"把表单恢复原样"，全量记录对回填是正确的。
    _STRATEGY_RELEVANT_FORM_FIELDS = {
        "close_to_close": (),
        "fixed_times": ("fixed_times",),
        "hedge_band": (
            "interval_type", "price_interval", "sigma_source", "sigma_window"),
    }
    # 签名里恒定出现的策略参数字段（无关时填 None，保持键集合稳定）。
    _STRATEGY_SIGNATURE_FIELDS = tuple(dict.fromkeys(
        field
        for fields in _STRATEGY_RELEVANT_FORM_FIELDS.values()
        for field in fields
    ))

    @staticmethod
    def _snapshot_strategy_signature(snapshot):
        """对比用的对冲策略签名：只含**这个策略真正读取**的字段。

        不能拿整份 ``form_state`` 当签名。``_snapshot_form_state`` 是为回填
        表单服务的，它不看 ``strategy_name`` 就全量记录固定时刻与带宽三项；
        而 ``_collect_history_state`` 每轮优选都会把本轮候选配置写回 state，
        重放快照因此带着与自己无关的残值。结果是两条**跑的是同一策略、逐日
        损益完全相同**的结果被报成「本次对比的变量：对冲策略」，还被染成绿色
        （干净的单变量实验）——例如两轮优选只有固定时刻候选文本不同，都选
        每日收盘基线、加载同一段，close_to_close 根本不读 fixed_times。

        未知策略名保守起见回退到全量比较：宁可多报一次差异，也不要把真的
        不同说成相同。

        **键集合必须恒定**，无关字段填 ``None`` 而不是删掉。``
        _differing_field_names`` 在两侧键集合不同时会整体放弃字段级定位、
        只报到属性级；若这里按策略删键，「每日收盘 vs 固定间隔」这种正当的
        跨策略对比就再也说不出「策略类型 / 带宽单位 / 带宽阈值变了」，退化
        成一句光秃秃的「对冲策略」。填 ``None`` 则两边仍可逐字段比：同策略
        时无关项两边都是 ``None`` 不产生假差异，跨策略时 ``None`` 与实际值
        不同，照常点名。
        """
        state = dict(getattr(snapshot, "form_state", None) or {})
        name = str(state.get("strategy_name", "") or "")
        relevant = BacktestApp._STRATEGY_RELEVANT_FORM_FIELDS.get(name)
        if relevant is None:
            return BacktestApp._freeze_snapshot_value(state)
        # 策略名与收盘保底对所有策略都生效，恒参与比较。
        signature = {
            "strategy_name": name,
            "force_day_close_hedge": bool(
                state.get("force_day_close_hedge", False)),
        }
        for key in BacktestApp._STRATEGY_SIGNATURE_FIELDS:
            signature[key] = state.get(key) if key in relevant else None
        return BacktestApp._freeze_snapshot_value(signature)

    @staticmethod
    def _validate_saved_result_name(saved_results, name, exclude_id=None):
        cleaned = str(name or "").strip()
        if not cleaned:
            raise ValueError("结果名称不能为空")
        if any(
                snapshot.result_id != exclude_id and snapshot.name == cleaned
                for snapshot in saved_results.values()):
            raise ValueError(f"结果名称已存在：{cleaned}")
        return cleaned

    _COMPARE_TAB_TITLE = " 🆚 结果对比 "

    def _update_saved_result_count(self):
        """把结果池条数写进标签页标题。

        空池不写 `(0)`：那时标签页里的占位符已经在说"先保留结果"，标题再挂
        一个 0 只是噪声。非空时数字要一直可见——它是"有没有东西可对比"的
        唯一常驻提示，此前挂在左侧按钮上。
        """
        notebook = getattr(self, "_nb", None)
        tab = getattr(self, "_compare_tab", None)
        if notebook is None or tab is None:
            return
        count = len(getattr(self, "_saved_backtests", None) or {})
        title = BacktestApp._COMPARE_TAB_TITLE
        if count:
            title = f"{title.rstrip()} ({count}) "
        try:
            notebook.tab(tab, text=title)
        except tk.TclError:
            # 标签页已随窗口销毁：计数只是展示，不该反过来打断清理流程。
            pass

    # ---- 结果池落盘 ----
    _POOL_FIELDS = (
        "result_id", "name", "saved_at", "summary_row", "daily_frame",
        "strategy_label", "parameter_summary", "source_label", "option_label",
        "path_key", "market_key", "contract_key", "economics_key", "position",
        "style_key", "origin", "origin_meta", "form_state", "sequence",
        "replay", "bar_size_requested", "intraday_steps",
    )

    @staticmethod
    def _snapshot_to_payload(snapshot):
        """快照 → 可写盘的 payload。``store_path`` 不入包（它就是包的位置）。"""
        body = {
            key: backtest_pool_store.encode(getattr(snapshot, key))
            for key in BacktestApp._POOL_FIELDS
        }
        return {
            "schema_version": backtest_pool_store.POOL_SCHEMA_VERSION,
            "sequence": int(getattr(snapshot, "sequence", 0) or 0),
            "result_id": str(snapshot.result_id),
            "saved_at": snapshot.saved_at.isoformat(),
            "snapshot": body,
        }

    @staticmethod
    def _payload_to_snapshot(payload):
        """payload → 快照。缺字段用 dataclass 默认值补，不让旧包整条报废。"""
        body = dict(payload.get("snapshot") or {})
        values = {
            key: backtest_pool_store.decode(body[key])
            for key in BacktestApp._POOL_FIELDS if key in body
        }
        saved_at = values.get("saved_at")
        if not isinstance(saved_at, datetime.datetime):
            values["saved_at"] = datetime.datetime.fromisoformat(
                str(payload.get("saved_at")))
        else:
            values["saved_at"] = saved_at.to_pydatetime() if hasattr(
                saved_at, "to_pydatetime") else saved_at
        snapshot = SavedBacktestResult(**values)
        snapshot.store_path = str(payload.get("_path", "") or "")
        return snapshot

    def _persist_saved_snapshot(self, snapshot):
        """写盘并返回一句状态栏后缀。

        失败不回滚内存池——这一条结果是真跑出来的，扔掉它比留着更糟。但
        「已保留」四个字必须诚实：写不进去就当场说清楚，否则用户下次开机
        才发现，而那时已经无从追查。
        """
        try:
            path = backtest_pool_store.write_snapshot(
                BacktestApp._snapshot_to_payload(snapshot), enforce=False)
        except Exception as exc:                              # noqa: BLE001
            snapshot.store_path = ""
            return f"（未能写入本机：{exc}）"
        snapshot.store_path = path
        # 先写入再淘汰，顺序不能反（与 _save_history_result 同一条约定）：反
        # 过来会在写失败时白删一份。自己调 enforce_limit 而不是让
        # write_snapshot 代劳，是因为要拿到被淘汰的清单——store 那边的约定
        # 就是「调用方应当把返回值报给用户」。
        try:
            evicted = backtest_pool_store.enforce_limit()
        except Exception:                                     # noqa: BLE001
            evicted = []
        return "并已保存到本机" + BacktestApp._forget_evicted_snapshots(
            self, evicted)

    def _forget_evicted_snapshots(self, evicted_paths):
        """把已淘汰出磁盘的快照同步移出内存池，返回一句状态栏后缀。

        内存池此前不受 ``MAX_RESULTS`` 约束，只有磁盘受。存到第 21 条时盘上
        第 1 条已经被删，标签页却仍写着 (21)、对比页也仍列着它——用户重开程
        序才发现少了一条，而那时已无从追查。留在内存里还有第二个后果：重命
        名它会走 ``write_snapshot(path=...)`` 把文件原样写回来，目录随即又超
        出上限。

        删除不可逆，即使是按规则删的也必须说一声。
        """
        paths = {str(path) for path in (evicted_paths or ()) if path}
        if not paths:
            return ""
        dropped = [
            snapshot for snapshot in self._saved_backtests.values()
            if str(getattr(snapshot, "store_path", "") or "") in paths
        ]
        for snapshot in dropped:
            self._saved_backtests.pop(snapshot.result_id, None)
            self._saved_comparison_selection.discard(snapshot.result_id)
            if getattr(self, "_latest_retained_result_id", None) == (
                    snapshot.result_id):
                self._latest_retained_result_id = None
        if not dropped:
            return ""
        names = "、".join(str(snapshot.name) for snapshot in dropped[:3])
        more = f" 等 {len(dropped)} 条" if len(dropped) > 3 else ""
        return (f"；已达上限 {backtest_pool_store.MAX_RESULTS} 条，"
                f"淘汰最旧的「{names}」{more}")

    def _load_saved_pool(self):
        """启动时载入结果池，并把序号推到已有最大值之后。

        序号不推的话，载入 20 条旧结果后第一次「保留」仍会生成
        ``result-0001``，而入池是按 key 直接赋值——盘里那条会被静默顶掉。
        """
        try:
            payloads, skipped = backtest_pool_store.read_all()
        except Exception as exc:                              # noqa: BLE001
            self._saved_pool_load_error = f"载入已保存结果失败：{exc}"
            return
        self._saved_pool_load_error = ""
        max_sequence = 0
        for payload in payloads:
            try:
                snapshot = BacktestApp._payload_to_snapshot(payload)
            except Exception as exc:                          # noqa: BLE001
                skipped.append(
                    (os.path.basename(str(payload.get("_path", ""))),
                     f"字段无法还原：{exc}"))
                continue
            self._saved_backtests[snapshot.result_id] = snapshot
            max_sequence = max(max_sequence, int(snapshot.sequence or 0))
        self._saved_backtest_sequence = max_sequence
        # 恢复上次显示的那几条。与结果本身分开存：它坏了最多回到「全部隐
        # 藏」，而空状态会写明 N 条都还在池里。默认不全选——重开就把二十条
        # 曲线一起铺上去，比让用户点一下更糟。
        restored = backtest_pool_store.read_view_state() & set(
            self._saved_backtests)
        self._saved_comparison_selection = restored
        if skipped:
            self._saved_pool_load_error = (
                f"有 {len(skipped)} 条已保存结果未能载入："
                + "；".join(f"{name}（{why}）" for name, why in skipped[:3])
                + ("…" if len(skipped) > 3 else ""))

    def _store_current_backtest(self, name, *,
                                origin=SNAPSHOT_ORIGIN_MANUAL,
                                origin_meta=None):
        """用已完成的普通回测结果创建快照，绝不重跑回测。"""
        bt = getattr(self, "_latest_backtest", None)
        gui_state = getattr(self, "_latest_backtest_state", None)
        if bt is None or gui_state is None:
            raise ValueError("没有可保留的已完成回测结果。")
        if getattr(self, "_latest_retained_result_id", None) is not None:
            raise ValueError("当前回测结果已经保留。")

        cleaned_name = self._validate_saved_result_name(
            self._saved_backtests, name)
        next_sequence = self._saved_backtest_sequence + 1
        result_id = f"result-{next_sequence:04d}"
        # 序号在载入结果池时已经推到已有最大值之后（见 _load_saved_pool），
        # 所以这里不会撞上盘里那些旧 ID。
        snapshot = self._make_saved_backtest_result(
            bt, gui_state, result_id, cleaned_name,
            origin=origin, origin_meta=origin_meta, sequence=next_sequence)
        self._saved_backtest_sequence = next_sequence
        self._saved_backtests[result_id] = snapshot
        self._saved_comparison_selection.add(result_id)
        self._latest_retained_result_id = result_id
        write_note = self._persist_saved_snapshot(snapshot)
        BacktestApp._persist_pool_view(self)
        self._update_saved_result_count()
        self._sync_retain_button_state()
        self._refresh_saved_comparison_if_visible()
        self._set_status(
            f"已保留『{cleaned_name}』{write_note}  |  可修改策略或参数继续回测"
            f"（共 {len(self._saved_backtests)} 条）")
        return snapshot

    @staticmethod
    def _next_default_name_number(saved_results, base):
        """同前缀名字接着往下编号：池里已有『X #05』就返回 6。

        默认编号不能用入池序号 ``_saved_backtest_sequence``：那是整个结果池
        的保存次序，中间存过别的策略就会跳号——存到「每日收盘 #05」后再存
        十条固定间隔，下一条每日收盘会默认成 #16。用户读这个数字时问的是
        「这是我第几次存这个策略」，所以只数同前缀的那几条。

        只认前缀完全一致、后缀是纯 ASCII 数字的名字：用户改过的名字（如
        「每日收盘 最终版」）不参与编号，也就不会把编号顶到莫名其妙的位置。
        """
        prefix = f"{base} #"
        highest = 0
        for snapshot in saved_results.values():
            name = str(getattr(snapshot, "name", "") or "")
            if not name.startswith(prefix):
                continue
            tail = name[len(prefix):]
            # 光 isdigit() 会放过上标之类的字符，而 int() 认不出它们。
            if tail.isascii() and tail.isdigit():
                highest = max(highest, int(tail))
        return highest + 1

    def _retain_current_backtest(self):
        bt = getattr(self, "_latest_backtest", None)
        gui_state = getattr(self, "_latest_backtest_state", None)
        if bt is None or gui_state is None:
            messagebox.showinfo("没有可保留结果", "请先成功运行一次回测。")
            return
        if getattr(self, "_latest_retained_result_id", None) is not None:
            messagebox.showinfo("结果已保留", "当前回测结果已经在对比结果池中。")
            return

        strategy_label, parameters = self._strategy_snapshot_labels(gui_state)
        short_parameters = parameters.split("；", 1)[0]
        if gui_state.get("strategy_name") == "close_to_close":
            base_name = strategy_label
        else:
            base_name = f"{strategy_label} · {short_parameters}"
        next_number = BacktestApp._next_default_name_number(
            self._saved_backtests, base_name)
        default_name = f"{base_name} #{next_number:02d}"
        while True:
            name = simpledialog.askstring(
                "保留回测结果",
                "为本次结果命名（可稍后在对比页重命名）：",
                initialvalue=default_name, parent=self,
            )
            if name is None:
                return
            try:
                name = self._validate_saved_result_name(
                    self._saved_backtests, name)
                break
            except ValueError as exc:
                messagebox.showerror("名称无效", str(exc))

        origin, origin_meta = BacktestApp._snapshot_origin_from_state(
            gui_state)
        try:
            self._store_current_backtest(
                name, origin=origin, origin_meta=origin_meta)
        except Exception as exc:
            messagebox.showerror("保留结果失败", str(exc))
            return

    @staticmethod
    def _saved_comparison_payload(snapshots):
        """只读取已完成快照生成对比数据，绝不重新运行回测。

        本页展示的是每条结果自己的绝对指标，不设固定基准、不算相对改善：
        谁跟谁比、比哪一项，交给排序和并列的曲线去回答。
        """
        import pandas as pd
        snapshots = list(snapshots)
        rows = []
        daily_curves = {}
        for snapshot in snapshots:
            if snapshot.name in daily_curves:
                raise ValueError(f"保存结果名称重复：{snapshot.name}")
            description = (
                f"{snapshot.strategy_label} · {snapshot.parameter_summary} · "
                f"{snapshot.option_label} · {snapshot.source_label}"
            )
            row = copy.deepcopy(snapshot.summary_row)
            row.update({
                "strategy": snapshot.name,
                "meta_description": description,
                "meta_result_id": snapshot.result_id,
                # 保存顺序的权威来源。别退回按 result_id 字典序排：那把顺序
                # 焊死在 ID 格式上，第 10000 条会排到 9999 前面，跨会话载入
                # 后若换了 ID 方案更会静默乱序——而乱序不会有任何断言挂掉。
                "meta_sequence": int(getattr(snapshot, "sequence", 0) or 0),
                # 曲线名可被用户改写，配色必须走稳定的策略身份键。
                "meta_style_key": getattr(
                    snapshot, "style_key", BASELINE_STRATEGY_STYLE_KEY),
                "meta_origin": getattr(
                    snapshot, "origin", SNAPSHOT_ORIGIN_MANUAL),
                "meta_origin_detail": (
                    BacktestApp._snapshot_origin_detail(snapshot)),
            })
            rows.append(row)
            daily_curves[snapshot.name] = snapshot.daily_frame
        summary = pd.DataFrame(rows)
        if not summary.empty:
            # 默认按保存顺序，与结果池上下对照着看最省事；想看谁高谁低，
            # 点一下列头即可。按某个指标排序会隐含"这一列越高/越低越好"的
            # 暗示，而本页不替用户下这个判断。
            summary = summary.sort_values(
                ["meta_sequence", "meta_result_id"],
                kind="stable").reset_index(drop=True)
        return summary, daily_curves

    # 一次对比里可以变化的五组属性。控制变量的道理：只有一组不同，差异才
    # 归得到那一组头上；同时变两组以上，看到的差就说不清是谁造成的。
    # (属性键, 中文名, 取值函数)
    _COMPARISON_ASPECTS = (
        ("market", "行情", lambda s: getattr(s, "market_key", ())),
        # 这一组统辖的是「是哪一种期权」外加它的全部合约参数，组名必须把两
        # 件事都说到：只写「期权类型」的话，改了执行价、类型没动的那次对比
        # 会被报成「本次对比的变量：期权类型（执行价：100 vs 105）」——句子
        # 本身就在说类型变了。
        ("contract", "期权类型与参数",
         lambda s: getattr(s, "contract_key", ())),
        ("position", "头寸方向", lambda s: (
            ("position", BacktestApp._saved_snapshot_position(s)),)),
        ("economics", "规模与成本", lambda s: getattr(s, "economics_key", ())),
        ("strategy", "对冲策略",
         lambda s: BacktestApp._snapshot_strategy_signature(s)),
    )

    # 字段级差异的中文名。签名保留了键名，所以能说到具体是哪一项不同，
    # 而不只是"期权不同"。期权合约参数的标签直接取自 OPTION_CLASSES 的
    # 定义，新增期权类型时不必在这里补一遍。
    _COMPARISON_FIELD_LABELS = {
        "source": "行情来源", "seed": "随机种子", "csv_path": "CSV 文件",
        "csv_col": "CSV 列", "wind_code": "标的代码",
        "wind_start": "起始日", "wind_end": "截止日",
        "wind_bar_size": "bar 粒度", "data_digest": "行情数据",
        "cls_name": "期权大类", "subtype": "期权类型",
        "quantity": "交易数量", "multiplier": "合约乘数",
        "tc_rate": "成本率", "slippage_bps": "滑点", "steps_per_day": "日内采样",
        "position": "头寸方向",
        "strategy_name": "策略类型", "fixed_times": "固定时刻",
        "interval_type": "带宽单位", "price_interval": "带宽阈值",
        "force_day_close_hedge": "收盘兜底",
        "sigma_source": "波动率口径", "sigma_window": "波动率窗口",
    }

    @staticmethod
    def _option_param_labels(cls_name=None):
        """期权合约参数的键 → 中文名，以及取值的中文回译。

        标签就写在 OPTION_CLASSES 的参数定义里（第二项），带 choices 的参数
        （如方向 1/-1）还能把数值译回"看涨/看跌"。

        **给了大类就以那个大类自己的定义为准。** 同一个键在不同大类下是不同
        的东西，全局合并会张冠李戴：

        - ``N`` 在累计里是杠杆倍数，在亚式里是观察日数；
        - ``T_days`` 在香草里是期限，在累计里是剩余期限；
        - ``s0`` 在雪球里是最新价而非初始价格；
        - ``cp`` 最狠——雪球的 ``-1`` 是「雪球 (卖看跌)」、``+1`` 是「反雪球
          (卖看涨)」，其余各类是「看跌 / 看涨」。译错的不是名字而是取值本身。

        此前一律 ``setdefault`` 且按 OPTION_CLASSES 的定义序合并，等于永远
        用香草那一套去解释所有期权。

        实现是把目标大类排到最前面再合并：它的定义先落进 setdefault 因而胜
        出，其余大类留作回退，未在本类定义的键仍能拿到一个名字而不是裸键名。
        拿不到大类时（跨大类比对）行为与从前一致——那时两侧键集合本来就对不
        上，差异只报到属性级，标签取谁都不影响结论。
        """
        configs = [OPTION_CLASSES[cls_name]] if cls_name in OPTION_CLASSES else []
        configs.extend(
            config for name, config in OPTION_CLASSES.items()
            if name != cls_name)
        labels, choices = {}, {}
        for config in configs:
            for spec in config.get("params", ()):
                key = str(spec[0])
                labels.setdefault(key, str(spec[1]))
                if len(spec) > 4 and isinstance(spec[4], dict):
                    choices.setdefault(key, {
                        value: text for text, value in spec[4].items()})
        return labels, choices

    @staticmethod
    def _signature_cls_name(flats):
        """从若干摊平签名里取共同的期权大类；不一致或没有则 ``None``。

        标签与取值回译要按大类取（见 ``_option_param_labels``），而
        ``_differing_field_names`` 是五组属性共用的——不含 ``cls_name`` 的那
        几组自然拿到 ``None``，正好退回全局合并。
        """
        names = {flat.get(("cls_name",)) for flat in flats}
        if len(names) != 1:
            return None
        name = names.pop()
        return name if isinstance(name, str) else None

    @staticmethod
    def _flatten_signature(value, prefix=()):
        """把嵌套签名摊平成 {键路径: 取值}。

        ``contract_key`` 里的 params 本身又是一组键值对，只比顶层就只能说
        到"合约参数不同"——而真正要说的是"波动率 0.18 vs 0.15"。
        """
        flat = {}
        for key, item in value:
            path = prefix + (str(key),)
            nested = (
                isinstance(item, tuple) and item
                and all(isinstance(entry, tuple) and len(entry) == 2
                        for entry in item))
            if nested:
                flat.update(BacktestApp._flatten_signature(item, path))
            else:
                flat[path] = item
        return flat

    # 取值的中文回译：差异串里写 "close_to_close vs hedge_band" 没人看得舒服。
    _COMPARISON_VALUE_LABELS = {
        "position": {-1: "买入", 1: "卖出"},
        "strategy_name": STRATEGY_DISPLAY,
        "sigma_source": SIGMA_SOURCE_DISPLAY,
        "subtype": SUBTYPE_DISPLAY,
        "interval_type": {
            "absolute": "绝对", "relative": "相对", "sigma": "σ 倍数"},
        "force_day_close_hedge": {True: "开启", False: "关闭"},
        "source": {"simulate": "模拟", "csv": "CSV", "wind": "Wind"},
    }

    @staticmethod
    def _format_signature_value(key, value, cls_name=None):
        """把签名里的取值渲染成人看的字符串。

        ``cls_name`` 决定用哪一套合约参数回译：不给的话 ``cp=-1`` 在雪球上会
        被译成「看跌 (Put)」，而雪球的 ``-1`` 是「雪球 (卖看跌)」。
        """
        mapping = BacktestApp._COMPARISON_VALUE_LABELS.get(key)
        if mapping and value in mapping:
            return str(mapping[value])
        _labels, choices = BacktestApp._option_param_labels(cls_name)
        mapping = choices.get(key)
        if mapping and value in mapping:
            return str(mapping[value])
        if value is None or value == "":
            return "—"
        if isinstance(value, bool):
            return "开启" if value else "关闭"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    @staticmethod
    def _differing_field_names(values):
        """逐字段比出差异，返回 [(中文名, [各条的取值]), ...]。

        取值按传入顺序原样返回，不拼成 "A vs B" ——三条以上时那种串既看不
        出谁是谁，又会把标题撑到七八十字。拼不拼、怎么排版交给展示层。
        不可读的字段（价格序列摘要）取值给 None，表示"只说不同、无值可看"。

        键集合本身不同时（换期权大类会整体换掉参数集）说不出具体字段，返回
        空表示"整组不同"，由调用方按属性级报告。
        """
        flats = [BacktestApp._flatten_signature(value) for value in values]
        paths = {frozenset(flat) for flat in flats}
        if len(paths) != 1:
            return []
        cls_name = BacktestApp._signature_cls_name(flats)
        param_labels, _choices = BacktestApp._option_param_labels(cls_name)
        order = list(BacktestApp._COMPARISON_FIELD_LABELS) + list(param_labels)
        changed = []
        opaque = {"data_digest"}
        for path in next(iter(paths)):
            taken = [flat[path] for flat in flats]
            if len({BacktestApp._freeze_snapshot_value(v) for v in taken}) <= 1:
                continue
            key = path[-1]
            label = BacktestApp._COMPARISON_FIELD_LABELS.get(
                key, param_labels.get(key, key))
            rank = order.index(key) if key in order else len(order)
            if key in opaque:
                # 摘要是一串哈希，摊出来没人看得懂；它只负责"确实不一样"。
                changed.append((rank, label, None))
                continue
            changed.append((rank, label, [
                BacktestApp._format_signature_value(key, value, cls_name)
                for value in taken
            ]))
        # 可读字段已经说清了差异时，就不必再补一句"行情数据不同"——那是
        # 同一件事的两种说法。只有摘要单独不同才值得说：输入看着一样，跑出
        # 来的价格序列却不是同一条。
        if any(shown is not None for _rank, _label, shown in changed):
            changed = [item for item in changed if item[2] is not None]
        # 按定义顺序输出，而不是签名的字典序——"策略类型、带宽阈值"读着顺。
        # 次级键是标签本身：路径集合是 frozenset，迭代序不确定，只按 rank
        # 排的话未列入定义表的字段每次出来的顺序都可能不同。
        changed.sort(key=lambda item: (item[0], item[1]))
        return [(label, shown) for _rank, label, shown in changed]

    # 详情窗里不列的字段。行情数据摘要是一串 sha256：在差异串里它至少还回答
    # 了"这两条确实不是同一段数据"，单看一条时一个哈希什么也没说。
    _SNAPSHOT_DETAIL_HIDDEN_FIELDS = frozenset({"data_digest"})

    # 行情组按来源只列本次真正用到的那几项。三种来源的字段互斥，但
    # market_key 恒定记全部键（键集合恒定是差异比对的要求），未用到的那几项
    # 存的是**左侧控件当时的值**——不是空值，所以"渲染成 — 就跳过"拦不住：
    # 模拟跑出来的快照照样会列出「CSV 列 close」「标的代码 510050.SH」，那是
    # Wind 代码框里恰好还留着的内容，这次回测根本没碰过它。
    _MARKET_DETAIL_FIELDS = {
        "simulate": ("source", "seed"),
        "csv": ("source", "csv_path", "csv_col"),
        "wind": ("source", "wind_code", "wind_start", "wind_end",
                 "wind_bar_size"),
    }

    @staticmethod
    def _bar_size_detail_text(actual, requested):
        """bar 粒度的显示串：把「实际用了什么」和「是不是自动推的」都写上。

        快照的 ``wind_bar_size`` 存的已经是 ``_resolve_wind_bar_size`` 解析后
        的实际粒度（见 ``_resolve_wind_backtest_state``），下拉原值另存在
        ``bar_size_requested``。只显示实际值看不出这个「15分钟」是自己挑的还
        是程序按策略推出来的；只显示「自动（推荐）」更糟——那根本不是一个粒度，
        是个待解析的占位。
        """
        actual = str(actual or "").strip()
        requested = str(requested or "").strip()
        if requested != WIND_AUTO_BAR_SIZE:
            return actual or "—"
        if not actual or actual == WIND_AUTO_BAR_SIZE:
            # 本字段上线前保留的快照没记原值，也可能记的就是未解析的占位：
            # 诚实说没记，不要把「自动（推荐）」摆出来充当一个粒度。
            return f"{WIND_AUTO_BAR_SIZE} · 未记录实际粒度"
        return f"{actual}（自动推荐）"

    @staticmethod
    def _intraday_steps_detail_text(snapshot, declared):
        """日内采样的显示串：写引擎**实际**用的那个数。

        见 ``_effective_intraday_steps``——签名里的值对真实行情是占位 1。
        """
        effective = int(getattr(snapshot, "intraday_steps", 0) or 0)
        if effective > 0:
            return str(effective)
        if declared is None:
            # 优选的一行覆盖多段，没有单一采样密度可言：交回 "—" 让渲染层
            # 整条略过，而不是含糊地说一句"未记录"。
            return "—"
        if BacktestApp._snapshot_source(snapshot) == "simulate":
            # 模拟的采样密度是用户直接填进去的，签名里那个值就是事实。
            return str(max(1, int(declared or 1)))
        # 真实行情的旧快照只记了占位值，说不出实际粒度就别假装说得出。
        return "未记录（旧快照存的是占位值）"

    # 展示层要覆盖签名取值的那几项：签名记的是「当时传给引擎的输入」，而这
    # 两项的输入是占位或待解析的，人要看的是「实际跑的是什么」。收成一张表
    # 而不是在渲染循环里堆 if——第三项迟早会来。
    _DETAIL_VALUE_OVERRIDES = {
        ("market", "wind_bar_size"): (
            lambda snapshot, value: BacktestApp._bar_size_detail_text(
                value, getattr(snapshot, "bar_size_requested", ""))),
        ("economics", "steps_per_day"): (
            lambda snapshot, value: BacktestApp._intraday_steps_detail_text(
                snapshot, value)),
    }

    @staticmethod
    def _format_detail_value(key, value, cls_name=None):
        """详情窗的取值渲染：在差异串那套回译之上补一条多值参数的排版。

        签名里的数组被冻成 ``("ndarray", dtype, shape, 摘要)`` 四元组，普通
        列表冻成元组；两者直接 ``str()`` 出来都是 Python 字面量。
        """
        if isinstance(value, tuple):
            if not value:
                return "—"
            if value[0] == "ndarray" and len(value) == 4:
                count = int(np.prod(value[2])) if value[2] else 1
                return f"数组（{count} 项）"
            return "、".join(
                BacktestApp._format_signature_value(key, item, cls_name)
                for item in value)
        return BacktestApp._format_signature_value(key, value, cls_name)

    @staticmethod
    def _snapshot_detail_sections(snapshot):
        """把一条快照的全部输入摊成 [(组名, [(字段名, 取值), ...]), ...]。

        分组、字段中文名和取值回译全部走对比页那一套（``_COMPARISON_ASPECTS``
        / ``_COMPARISON_FIELD_LABELS`` / ``_option_param_labels`` /
        ``_format_signature_value``），于是详情窗里读到的措辞与说明卡报差异时
        用的一定是同一个词。各写一套中文名的话，改了期权参数标签只同步一边，
        用户就会在两处看到同一项的两个名字。

        渲染成 ``—`` 的字段整条略过：签名为了让键集合恒定（见
        ``_snapshot_strategy_signature``），会给与本条无关的项填 ``None``——
        每日收盘的快照带着 ``fixed_times=None``，Wind 行情的快照带着
        ``csv_path=None``。差异比对需要这些占位，人看的详情不需要，列出来只
        会让真正有值的那几行更难找。合约参数不会因此漏项：它们全是数值型，
        取不到值的情况不存在，所以每种期权定义了几项就列几项（香草 7、累计
        14、亚式 12、气囊 11、雪球 18）。

        标签、取值回译和**行序**都按这条快照自己的期权大类取：``param_labels``
        把该大类的参数排在最前，于是详情里的参数顺序就是左侧表单里的定义顺序。
        """
        cls_name = BacktestApp._signature_cls_name(
            [BacktestApp._flatten_signature(
                tuple(getattr(snapshot, "contract_key", ()) or ()))])
        param_labels, _choices = BacktestApp._option_param_labels(cls_name)
        order = list(BacktestApp._COMPARISON_FIELD_LABELS) + list(param_labels)
        sections = []
        for aspect, label, getter in BacktestApp._COMPARISON_ASPECTS:
            rows = []
            flat = BacktestApp._flatten_signature(tuple(getter(snapshot)))
            # 行情组只列本次来源用得上的那几项；来源不认识时全列，宁可多几行
            # 也不要因为一个没见过的来源名把整组抹空。
            allowed = (
                BacktestApp._MARKET_DETAIL_FIELDS.get(flat.get(("source",)))
                if aspect == "market" else None)
            for path, value in flat.items():
                field = path[-1]
                if field in BacktestApp._SNAPSHOT_DETAIL_HIDDEN_FIELDS:
                    continue
                if allowed is not None and field not in allowed:
                    continue
                override = BacktestApp._DETAIL_VALUE_OVERRIDES.get(
                    (aspect, field))
                shown = (
                    override(snapshot, value) if override
                    else BacktestApp._format_detail_value(
                        field, value, cls_name))
                if shown == "—":
                    continue
                rows.append((
                    order.index(field) if field in order else len(order),
                    BacktestApp._COMPARISON_FIELD_LABELS.get(
                        field, param_labels.get(field, field)),
                    shown,
                ))
            # 与差异串同一个排序依据：先按定义顺序，同 rank 再按文本。
            # _flatten_signature 的键来自 frozenset 化的路径集合，不兜底按文
            # 本排的话，同一条快照两次打开可能排出不同的顺序。
            sections.append(
                (label, [(name, text) for _rank, name, text in sorted(rows)]))
        return sections

    @staticmethod
    def _history_row_form_state(row):
        """把优选排名行的 ``meta_*`` 还原成一份对冲策略 form_state。

        这几个字段正是「应用策略到左侧参数」读的那些，够拼出完整的策略签名。
        候选带宽一律以 σ 表达：整批候选共用左侧输入的波动率，排名比的就是 σ
        倍数本身（见 ``_collect_history_state``）。
        """
        fallback = row.get("meta_force_day_close_hedge")
        return {
            "strategy_name": str(
                row.get("meta_strategy_name")
                or row.get("strategy_type") or "").strip(),
            "fixed_times": str(row.get("meta_fixed_times", "") or ""),
            "interval_type": "sigma",
            "price_interval": BacktestApp._comparison_finite(
                row.get("meta_candidate_sigma")),
            "sigma_source": str(
                row.get("meta_sigma_source", "") or "implied"),
            "sigma_window": BacktestApp._comparison_safe_int(
                row.get("meta_sigma_window"), 20),
            "force_day_close_hedge": (
                bool(fallback)
                if isinstance(fallback, (bool, np.bool_)) else False),
        }

    # 期权参数那一组在优选页必须带的限定。每段实际用的期权是按该段首价整体
    # 缩放过的（见 _history_replay_gui_state / _replay_scaled_params），一行覆
    # 盖十几段就有十几组实际行权价——照单看会拿去对账，对不上。
    #
    # 出口写「加载明细」而不是「回测这一段」：分段的 bar 级明细在优选跑完时
    # 就已落盘，那个按钮是读回来（约 27 ms）不是重算，_build_history_replay_bar
    # 那条注释专门交代过不能叫「回测」。
    _HISTORY_CONTRACT_NOTE = (
        "⚠ 以下是左侧填入的参考值。每段实际用的期权按该段首价整体缩放，"
        "逐段不同。要看某一段实际用的参数，在图表下方「查看某段明细」选中该段"
        "并点「加载明细」，回测摘要页会列出按该段首价伸缩后的期权要素。")

    @staticmethod
    def _history_row_detail_sections(row, history_state):
        """把优选排名的一行摊成可展示的分组。

        与结果池最大的不同是**参数天然分两层**：优选的全部候选共用同一套期权、
        行情与规模成本（那正是控制变量的做法），只有对冲策略逐条不同。所以这
        里拿运行时冻结的 state 造出前四组，再把这一行自己的 ``meta_*`` 拼成
        第五组的 form_state——五组的字段、中文名和取值回译于是与结果池详情完全
        同构，两个窗口不会对同一项给出两种说法。

        末尾另加一组「本行的取样口径」：它回答"这一行的数字是在多大样本上算
        出来的"，是优选特有的、结果池没有的一层。
        """
        history_state = dict(history_state or {})
        row = dict(row or {})
        market_key, contract_key, economics_key = (
            BacktestApp._signature_keys_from_state(history_state))
        try:
            position = BacktestApp._normalize_position(
                history_state.get("position"))
        except ValueError:
            position = 1
        probe = SimpleNamespace(
            market_key=market_key,
            contract_key=contract_key,
            economics_key=economics_key,
            position=position,
            form_state=BacktestApp._history_row_form_state(row),
            bar_size_requested=str(
                history_state.get("wind_bar_size_requested", "") or ""),
            intraday_steps=0,
        )
        sections = BacktestApp._snapshot_detail_sections(probe)

        scope = []
        for key, label in (
                ("period", "周期"),
                ("strategy", "候选"),
        ):
            text = str(row.get(key, "") or "").strip()
            if text:
                scope.append((label, text))
        paired = BacktestApp._comparison_safe_int(
            row.get("paired_windows", row.get("rolling_windows")), 0)
        if paired:
            scope.append(("参与评分段数", str(paired)))
        baseline = BacktestApp._comparison_safe_int(
            row.get("baseline_windows"), 0)
        if baseline:
            scope.append(("基准段数", str(baseline)))
        sections.append(("本行的取样口径", scope))
        return sections

    @staticmethod
    def _comparison_aspect_diff(snapshots):
        """逐属性比对，返回 [(中文名, 是否相同, 差异字段列表), ...]。

        少于两条时没有"异同"可言，返回空表。
        """
        snapshots = list(snapshots)
        if len(snapshots) < 2:
            return []
        report = []
        for _key, label, getter in BacktestApp._COMPARISON_ASPECTS:
            values = [tuple(getter(snapshot)) for snapshot in snapshots]
            same = len(set(values)) == 1
            fields = [] if same else BacktestApp._differing_field_names(values)
            report.append((label, same, fields))
        return report

    @staticmethod
    def _comparison_variable_summary(snapshots):
        """把属性异同拆成三层：标题、逐字段取值、其余一致。

        标题只说"变量是哪个属性"，长度恒定；具体取值交给 ``fields``，由展示
        层按序号排成网格。此前三层挤在一行粗体里，三条结果差两项就是七八十
        个字，还得靠 "A vs B vs C" 硬串——那串既看不出谁是谁，也没法截断。
        """
        report = BacktestApp._comparison_aspect_diff(snapshots)
        if not report:
            return None
        changed = [(label, fields) for label, same, fields in report if not same]
        same_labels = [label for label, same, _f in report if same]

        def _rows(entries):
            """把各属性的字段摊平；字段名与属性名重合时不重复。"""
            rows = []
            for label, fields in entries:
                if not fields:
                    rows.append((label, None))
                    continue
                for name, values in fields:
                    rows.append((name, values))
            return rows

        if not changed:
            return {
                "state": "identical",
                "headline": "所选结果的输入完全相同",
                "fields": [],
                "rest": ("行情、期权类型与参数、头寸方向、规模与成本、"
                         "对冲策略逐项一致；"
                         "若数字仍有差异，只可能来自未记录的输入。"),
            }
        if len(changed) == 1:
            return {
                "state": "single",
                "headline": f"本次对比的变量：{changed[0][0]}",
                "fields": _rows(changed),
                "rest": (f"其余一致：{'、'.join(same_labels)}"
                         if same_labels else ""),
            }
        return {
            "state": "multiple",
            "headline": (
                f"同时有 {len(changed)} 项不同："
                + "、".join(label for label, _fields in changed)),
            "fields": _rows(changed),
            "rest": ("差异无法归因到其中任何一项；"
                     + (f"一致的是：{'、'.join(same_labels)}"
                        if same_labels else "没有任何一项是一致的")),
        }

    @staticmethod
    def _saved_comparison_warnings(snapshots):
        """属性异同之外，还需要单独说明的看数限制。

        不报"行情路径不同"：模拟行情下价格序列由期权参数派生（改 σ 就会重
        新生成路径），Wind/CSV 下换区间也必然换数据——路径差异是上游输入
        变化的必然结果，单独说一句"行情数据本身不同"会和"其余一致：行情"
        直接打架。真正需要提醒的是**长度不同**：那时曲线按序号对齐才有误
        导性，累计类指标也不在同一个跨度上。
        """
        snapshots = list(snapshots)
        if len(snapshots) < 2:
            return []
        warnings = []
        lengths = {
            BacktestApp._comparison_safe_int(
                (snapshot.summary_row or {}).get("n_trade_days"), 0)
            for snapshot in snapshots
        }
        if len(lengths) > 1:
            warnings.append(
                "各条的交易日数不同：曲线按序号对齐，"
                "期末净损益与总成本不在同一个跨度上。")
        positions = {
            BacktestApp._saved_snapshot_position(snapshot)
            for snapshot in snapshots
        }
        if len(positions) > 1:
            warnings.append(
                "同时包含买入与卖出：期末净损益与最大回撤的正负来自方向本身，"
                "横向看应比总成本、再触发与换手额。")
        return warnings

    def _rename_saved_backtest(self, result_id, new_name):
        snapshot = self._saved_backtests[result_id]
        snapshot.name = BacktestApp._validate_saved_result_name(
            self._saved_backtests, new_name, exclude_id=result_id)
        # 原地覆盖同一个文件：文件名带序号，是载入顺序的依据，不能因为改
        # 展示名就换名字。
        if snapshot.store_path:
            try:
                backtest_pool_store.write_snapshot(
                    BacktestApp._snapshot_to_payload(snapshot),
                    path=snapshot.store_path, enforce=False)
            except Exception:                                 # noqa: BLE001
                # 改名失败只影响下次启动看到的名字，不该打断本次会话。
                pass
        return snapshot

    def _delete_saved_backtest(self, result_id):
        snapshot = self._saved_backtests.pop(result_id)
        self._saved_comparison_selection.discard(result_id)
        if getattr(self, "_latest_retained_result_id", None) == result_id:
            self._latest_retained_result_id = None
        if snapshot.store_path:
            backtest_pool_store.delete_snapshot(snapshot.store_path)
        BacktestApp._persist_pool_view(self)
        BacktestApp._update_saved_result_count(self)
        BacktestApp._sync_retain_button_state(self)
        return snapshot

    def _build_backtest(self, gs):
        """根据已收集的 GUI 状态构建 HedgeBacktest 实例（可在任意线程调用）"""
        if (gs.get("source") == "wind"
                and gs.get("wind_bar_size") == WIND_AUTO_BAR_SIZE):
            # 兼容直接调用 _build_backtest 的 API / 测试；正常 GUI 路径在
            # 主线程收集状态时已经完成解析。
            gs = BacktestApp._resolve_single_wind_state(gs)
        cfg = gs["cfg"]
        subtype = gs["subtype"]
        params = gs["params"]
        src = gs["source"]
        tc_rate = gs["tc_rate"]
        position = BacktestApp._normalize_position(gs["position"])
        quantity = gs["quantity"]
        multiplier = gs["multiplier"]
        slippage_bps = gs.get("slippage_bps", 0.0)
        steps_per_day = int(gs.get("steps_per_day", 1))
        force_day_close_hedge = bool(
            gs.get("force_day_close_hedge", False))

        # 组装对冲策略对象
        strat_name = gs.get("strategy_name", "close_to_close")
        if strat_name == "hedge_band":
            strategy = HedgeBandStrategy(
                band_type=gs.get("interval_type", "absolute"),
                threshold=gs.get("price_interval", 1.0),
                sigma_source=gs.get("sigma_source", "implied"),
                window_days=gs.get("sigma_window", 20),
            )
        elif strat_name == "close_to_close":
            strategy = CloseToCloseStrategy()
        elif strat_name == "fixed_times":
            strategy = FixedTimeStrategy(
                gs.get("fixed_times", DEFAULT_FIXED_TIMES))
        else:
            raise ValueError(f"未知对冲策略: {strat_name}")

        if isinstance(strategy, FixedTimeStrategy) and src == "wind":
            from pricing.wind_data import get_trading_session_clock_ranges
            strategy.set_trading_sessions(
                get_trading_session_clock_ranges(gs.get("wind_code", "")))

        # 判断 Wind intraday 频率（None/日频 走日频分支，其它走 wsi）
        wind_bar_label = gs.get("wind_bar_size", "日频") or "日频"
        _bar_label_to_size = {
            "日频": None,
            **{label: str(minutes)
               for label, minutes in _WIND_BAR_MINUTES.items()},
        }
        # 只有 Wind 分支真正消费该粒度。CSV / 模拟来源不经过 Wind resolver，
        # 其 wind_bar_size 仍是下拉原值“自动（推荐）”，不能因此拒绝回测。
        if src == "wind" and wind_bar_label not in _bar_label_to_size:
            raise ValueError(
                f"Wind 行情采样粒度尚未解析: {wind_bar_label!r}")
        wind_bar_size = _bar_label_to_size.get(wind_bar_label)

        self._validate_fixed_time_source_state(gs)

        if src == "simulate":
            s0 = float(params["s0"])
            seed = int(gs["seed"])

            option = cfg["build"](subtype, params)

            # 获取期限天数
            T_days = params.get("T_days") or params.get("T") or 20
            sigma_impl = params.get("sigma", 0.18)
            r = params.get("r", 0.03)
            q = params.get("q", 0.03)
            # 已实现波动率：空则同隐含波动率
            real_vol_str = gs["real_vol"]
            sigma_real = float(real_vol_str) if real_vol_str else sigma_impl
            prices = HedgeBacktest.simulate_prices(
                s0, sigma_real, T_days, r=r, q=q, seed=seed,
                steps_per_day=steps_per_day,
            )
            bt = HedgeBacktest(option, prices, tc_rate=tc_rate, position=position,
                               quantity=quantity, multiplier=multiplier,
                               strategy=strategy, steps_per_day=steps_per_day,
                               slippage_bps=slippage_bps,
                               force_day_close_hedge=force_day_close_hedge)

        elif src == "csv":
            filepath = gs["csv_path"]
            if not filepath:
                raise ValueError("请选择 CSV 文件")
            price_col = gs["csv_col"]
            # 期权参数中的 s0 作为参考价 S_ref，
            # from_csv 会按 ratio = 真实起始价 / S_ref 自动缩放价格量纲要素。
            option = cfg["build"](subtype, params)
            bt = HedgeBacktest.from_csv(option, filepath, price_col=price_col,
                                        tc_rate=tc_rate,
                                        position=position,
                                        quantity=quantity, multiplier=multiplier,
                                        strategy=strategy,
                                        steps_per_day=None,
                                        slippage_bps=slippage_bps,
                                        force_day_close_hedge=(
                                            force_day_close_hedge))

        elif src == "wind":
            code = gs["wind_code"]
            start = gs["wind_start"]
            end = gs["wind_end"]
            # 期权参数中的 s0 作为参考价 S_ref，
            # from_wind 会按 ratio = 真实起始价 / S_ref 自动缩放价格量纲要素。
            option = cfg["build"](subtype, params)
            from pricing.wind_data import classify_wind_history_code
            classification = classify_wind_history_code(code)
            wind_adjust = (
                "" if classification.get("is_futures_contract") else "F")
            # wind_bar_size=None -> 日频；其它值触发 intraday。GUI 不再
            # 提供全局 spd 覆盖，统一交给真实索引/交易时段自动推导。
            bt = HedgeBacktest.from_wind(option, code, start, end,
                                         tc_rate=tc_rate,
                                         position=position,
                                         quantity=quantity, multiplier=multiplier,
                                         strategy=strategy,
                                         steps_per_day=None,
                                         slippage_bps=slippage_bps,
                                         force_day_close_hedge=(
                                             force_day_close_hedge),
                                         adjust=wind_adjust,
                                         bar_size=wind_bar_size)
        else:
            raise ValueError(f"未知数据来源: {src}")

        if strat_name == "fixed_times":
            self._validate_fixed_time_backtest(bt, strategy)

        # 保存元信息用于展示（不再依赖 tkinter 变量）
        bt._gui_meta = {
            "cls_name": gs["cls_name"],
            "subtype": subtype,
            "source": src,
            "wind_start": gs.get("wind_start"),
            "wind_end": gs.get("wind_end"),
            "wind_bar_size": gs.get("wind_bar_size"),
            "wind_bar_size_requested": gs.get(
                "wind_bar_size_requested"),
            "wind_date_mode": gs.get("wind_date_mode"),
        }
        return bt

    @staticmethod
    def _validate_fixed_time_backtest(bt, strategy):
        """用后端同一套交易日组规则验证固定时刻行情。"""
        configured_strategy = getattr(bt, "strategy", None)
        if isinstance(configured_strategy, FixedTimeStrategy):
            strategy = configured_strategy
        timestamps = getattr(bt, "timestamps", None)
        try:
            import pandas as pd
            index = pd.DatetimeIndex(timestamps)
        except Exception as exc:
            raise ValueError("行情时间戳无法解析为 DatetimeIndex。") from exc
        _validate_fixed_time_data(strategy, index)

    # ---- 隐藏占位符工具方法 ----
    def _hide_placeholder(self, suffix):
        """幂等隐藏占位符；已被旧渲染路径销毁时清除失效引用。"""
        attr = f"_{suffix}_placeholder"
        ph = getattr(self, attr, None)
        if ph is not None:
            try:
                ph.place_forget()
            except tk.TclError:
                # Tk widget 对象仍在 Python 中，但底层 window path 已失效。
                setattr(self, attr, None)

    def _clear_tab_content_preserving_placeholder(self, suffix, tab):
        """清理直接挂在 tab 下的动态控件，但保留可复用的占位符。"""
        placeholder = getattr(self, f"_{suffix}_placeholder", None)
        for widget in tab.winfo_children():
            if widget is not placeholder:
                widget.destroy()

    # ---- 动态 Figure 尺寸计算 ----
    _CHART_DPI = 96

    def _container_figsize(self, container, fallback=(10, 7), min_w=6, min_h=4):
        """根据容器的实际像素尺寸返回 (width_inch, height_inch).

        若容器尚未布局 (winfo_width()==1), 则用 Notebook 的已知尺寸估算.
        """
        self.update_idletasks()
        pw = container.winfo_width()
        ph = container.winfo_height()
        # 容器未布局时回退到 Notebook 尺寸 (减去 tab 栏和 padding 的大致高度)
        if pw <= 1 or ph <= 1:
            nb = self._nb
            pw = nb.winfo_width() - 16   # padding 左右各 ~8
            ph = nb.winfo_height() - 52  # tab 栏 + padding 约 52px
            if pw <= 1 or ph <= 1:
                return fallback
        w_inch = max(pw / self._CHART_DPI, min_w)
        h_inch = max(ph / self._CHART_DPI, min_h)
        return (w_inch, h_inch)

    def _reset_figure_container(self, key, container):
        """销毁旧 Tk canvas，并立即释放其 Figure 中的大数组引用。"""
        for widget in container.winfo_children():
            widget.destroy()
        figure_attr = f"_{key}_figure"
        old_figure = getattr(self, figure_attr, None)
        if old_figure is not None:
            old_figure.clear()
        setattr(self, figure_attr, None)
        setattr(self, f"_{key}_canvas", None)

    # ---- 结果展示 ----
    @staticmethod
    def _format_comparison_value(value, digits=2, *, signed=False,
                                 percent=False):
        """稳健格式化对比指标；缺失值统一显示为破折号。"""
        number = BacktestApp._comparison_finite(value)
        if number is None:
            return "—"
        if percent:
            number *= 100.0
        if abs(number) < 0.5 * 10 ** (-digits):
            number = 0.0
        sign = "+" if signed and number > 0 else ""
        suffix = "%" if percent else ""
        return f"{sign}{number:,.{digits}f}{suffix}"

    # 导出用的列与表头。排名表给的是格式化过的展示串，这里导原始数值，
    # 落到表格软件里才能继续算。
    _COMPARISON_EXPORT_COLUMNS = (
        ("strategy", "结果名称"),
        ("meta_description", "策略 / 参数 / 期权 / 行情来源"),
        ("total_net_pnl", "期末净损益"),
        ("total_tc", "总成本"),
        ("max_drawdown", "最大回撤"),
        ("rehedge_count", "再触发次数"),
        ("actual_trade_count", "实际成交次数"),
        ("turnover", "换手额"),
        ("n_trade_days", "交易日数"),
        ("meta_origin", "产生方式"),
        ("meta_origin_detail", "来源溯源"),
    )

    def _comparison_export_frames(self):
        """把当前对比整理成两张可导出的表：指标与曲线。

        导的是原始数值而不是表里那串格式化文本，落到表格软件里才能继续算。
        """
        import pandas as pd
        summary = getattr(self, "_comparison_summary", None)
        if summary is None or getattr(summary, "empty", True):
            raise ValueError("当前没有可导出的对比结果。")
        ranking = pd.DataFrame([
            {
                heading: row.to_dict().get(key)
                for key, heading in BacktestApp._COMPARISON_EXPORT_COLUMNS
            }
            for _index, row in summary.iterrows()
        ])
        # 逐条建 Series 再按索引横向拼，不能用 dict-of-arrays 直接建表：
        # 后者要求各列等长，而"各条交易日数不同"恰恰是本页显式支持的状态
        # （_saved_comparison_warnings 有专门提示、绘图侧对每条各自算
        # x = arange(1, len+1)），换区间做对比更是本页头号用途。等长要求会
        # 让整个导出抛 ValueError，被调用方转成"没有可导出的结果"，连同已
        # 经拼好的排名表一起丢掉——结果明明就在屏幕上。
        # 索引即交易日序号，短的那几条自动补 NaN，与图表的对齐方式一致。
        columns = [
            pd.Series(
                np.asarray(
                    daily.get("cumulative_net_pnl", []), dtype=float),
                name=name,
            ).rename_axis(None)
            for name, daily in getattr(
                self, "_comparison_daily_curves", {}).items()
        ]
        for column in columns:
            column.index = np.arange(1, len(column) + 1)
        curves = pd.concat(columns, axis=1) if columns else pd.DataFrame()
        curves.index.name = "交易日序号"
        return ranking, curves

    def _export_saved_comparison(self):
        """导出当前对比：排名一张表，累计净损益曲线另一张。"""
        try:
            ranking, curves = BacktestApp._comparison_export_frames(self)
        except Exception as exc:
            messagebox.showinfo("没有可导出的结果", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="导出结果对比", defaultextension=".csv",
            initialfile="结果对比.csv",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        base, extension = os.path.splitext(path)
        curves_path = f"{base}_曲线{extension or '.csv'}"
        try:
            ranking.to_csv(path, index=False, encoding="utf-8-sig")
            curves.to_csv(curves_path, encoding="utf-8-sig")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self._set_status(
            f"已导出 {len(ranking)} 条排名与曲线  |  "
            f"{os.path.basename(path)} / {os.path.basename(curves_path)}")
        messagebox.showinfo(
            "导出成功",
            f"排名已保存至:\n{path}\n\n累计净损益曲线已保存至:\n{curves_path}")

    def _apply_snapshot_strategy_to_form(self, snapshot):
        """把快照的对冲策略输入写回左侧表单，不动其它任何一组参数。"""
        form_state = dict(getattr(snapshot, "form_state", None) or {})
        strategy_name = str(form_state.get("strategy_name", "") or "").strip()
        if strategy_name not in STRATEGY_DISPLAY:
            raise ValueError(
                "这条快照保存于本功能之前，没有可回填的策略参数；"
                "重新运行一次回测并保留即可。")
        if strategy_name == "fixed_times":
            fixed_times = str(form_state.get("fixed_times", "") or "").strip()
            if not fixed_times:
                raise ValueError("快照缺少固定时刻参数。")
            FixedTimeStrategy(fixed_times)
            self._fixed_times_var.set(fixed_times)
        elif strategy_name == "hedge_band":
            band_type = str(
                form_state.get("interval_type", "") or "absolute").strip()
            threshold = BacktestApp._comparison_finite(
                form_state.get("price_interval"))
            if band_type not in ("absolute", "relative", "sigma"):
                raise ValueError(f"快照的带宽单位无法识别：{band_type}")
            if threshold is None or threshold <= 0:
                raise ValueError("快照缺少有效的固定间隔带宽。")
            # 波动率口径要先写回：带宽的三口径换算依赖它，顺序反了会按旧
            # 口径换算出另外两种表示。
            sigma_source = str(
                form_state.get("sigma_source", "") or "implied")
            if sigma_source in SIGMA_SOURCE_DISPLAY:
                self._sigma_src_var.set(SIGMA_SOURCE_DISPLAY[sigma_source])
            sigma_window = BacktestApp._comparison_safe_int(
                form_state.get("sigma_window"), 20)
            self._sigma_win_var.set(str(max(2, sigma_window)))
            # 按保存时的那一种输入单位写回，另外两种由既有换算链路补齐；
            # 换算过一手再写回去，显示值会和当初跑的那次差一个舍入。
            {
                "absolute": self._band_abs_var,
                "relative": self._band_rel_var,
                "sigma": self._band_sigma_var,
            }[band_type].set(f"{threshold:.10g}")
            self._mark_band_edited(band_type)
            self._sync_band_inputs(band_type, strict=True)
        self._strategy_var.set(STRATEGY_DISPLAY[strategy_name])
        fallback_var = getattr(self, "_force_day_close_hedge_var", None)
        if fallback_var is not None:
            fallback_var.set(
                bool(form_state.get("force_day_close_hedge", False)))
        self._toggle_strategy()
        return strategy_name

    def _apply_selected_snapshot_to_form(self):
        """把结果池里聚焦那条快照的对冲参数送回左侧继续调。

        与同排的重命名、删除同一个作用对象：结果池里点中的那一行。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshot = self._saved_backtests[result_id]
        try:
            self._apply_snapshot_strategy_to_form(snapshot)
        except Exception as exc:
            messagebox.showerror("无法应用策略", str(exc))
            return
        self._set_status(
            f"已把『{snapshot.name}』的对冲策略应用到左侧参数  |  "
            "期权结构、方向与行情来源保持不变")

    def _show_saved_snapshot_params(self):
        """右键菜单「参数详情」：只读列出聚焦那条快照的全部原始输入。

        与「加载明细」的分工：那一条要的是**结果**（逐日损益、对冲图表），
        因此必须重放，代价是顶掉当前回测并切页，旧快照没有配方还点不动；
        这一条要的是**输入**，四个签名里全存着，读一读就有——不重放、不动
        当前回测、不依赖 ``replay``，上线前保留的快照照样打得开。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        BacktestApp._open_snapshot_params_window(
            self, self._saved_backtests[result_id])

    def _open_snapshot_params_window(self, snapshot):
        """结果池那条快照的参数详情窗。"""
        meta = [
            BacktestApp._saved_snapshot_origin_label(snapshot),
            snapshot.saved_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        detail = BacktestApp._snapshot_origin_detail(snapshot)
        if detail:
            meta.append(detail)
        return BacktestApp._open_params_window(
            self,
            heading=snapshot.name,
            subtitle="  ·  ".join(meta),
            sections=BacktestApp._snapshot_detail_sections(snapshot),
        )

    def _open_params_window(self, *, heading, subtitle, sections, notes=None):
        """摆出参数详情窗。结果池与策略优选共用这一套渲染。

        正文用只读 ``Text`` 而不是一排 ``Label``：内容长度随期权大类差着数
        倍（香草 7 个合约参数，雪球 18 个），Text 自带滚动，而且文本可以选中
        复制——把一条结果的参数贴给别人是这个窗口的第二个用途。

        ``notes`` 是 ``{组名: 说明}``，用于给整组加一句限定。策略优选那边靠它
        说明「期权参数是参考值，每段按段初价缩放」——那句话属于整组，挂在某
        一行上都不对。
        """
        notes = dict(notes or {})
        window = tk.Toplevel(self)
        window.title(f"参数详情 — {heading}")
        window.geometry("560x620")
        window.transient(self)
        body = ttk.Frame(window, style="Surface.TFrame", padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body, text=heading, style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 13, "bold"), anchor="w",
        ).pack(fill="x")
        ttk.Label(
            body, text=subtitle, style="SurfaceMuted.TLabel",
            anchor="w", wraplength=520, justify="left",
        ).pack(fill="x", pady=(2, 8))

        text_frame = ttk.Frame(body, style="Surface.TFrame")
        text_frame.pack(fill="both", expand=True)
        text = tk.Text(
            text_frame, wrap="word", relief="flat", padx=10, pady=8,
            bg=PALETTE["surface_alt"], fg=PALETTE["text"],
            font=(_MONO_FONT_FAMILY, 10), highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text.tag_configure(
            "group", foreground=PALETTE["primary"],
            font=(_UI_FONT_FAMILY, 11, "bold"), spacing1=10, spacing3=4)
        text.tag_configure("label", foreground=PALETTE["text_muted"])
        text.tag_configure("value", foreground=PALETTE["text"])
        text.tag_configure("empty", foreground=PALETTE["text_light"])
        text.tag_configure(
            "note", foreground=PALETTE["warning"], lmargin1=16, lmargin2=16,
            spacing3=2)

        # 取值全部起在同一竖列上。用制表位按**像素**对齐，不能按字符数补空格：
        # 等宽字体里一个汉字占两格而 len() 只算一个，`初始价格 S0`（7 字、11
        # 格）和 `期限(交易日)`（8 字、12 格）补完反而错开。
        # 制表位按全窗最长的字段名算，不是逐组算——逐组算的话五组的取值会各起
        # 各的位置。
        name_font = tkfont.Font(font=(_MONO_FONT_FAMILY, 10))
        stop = 14 + max(
            (name_font.measure(name)
             for _label, rows in sections for name, _v in rows),
            default=0) + 18
        text.configure(tabs=(stop,))
        for label, rows in sections:
            text.insert("end", f"{label}\n", "group")
            note = str(notes.get(label, "") or "").strip()
            if note:
                text.insert("end", f"{note}\n", "note")
            if not rows:
                text.insert("end", "  （本组没有已记录的输入）\n", "empty")
                continue
            for name, shown in rows:
                text.insert("end", f"  {name}\t", "label")
                text.insert("end", f"{shown}\n", "value")
        # 插完再锁：disabled 状态下 insert 是静默无效的。
        text.configure(state="disabled")

        ttk.Button(
            body, text="关闭", width=10, command=window.destroy,
        ).pack(anchor="e", pady=(10, 0))
        return window

    # 结果池表的列：(列键, 表头, 初始宽度, 对齐)。初始宽度按表头与内容实际
    # 需要分配，总和接近容器宽度，配合全列 stretch 使窗口缩放不改变列比例。
    # 这一列宽度同时是自动伸缩的下限，见 _SAVED_POOL_COLUMN_MAX。
    # 「期权类型」排在「策略」前面：一条结果先是"测的哪一种期权"，才轮到
    # "用什么对冲它"。此前这一列完全不在表上——``option_label`` 存了却只在
    # 保留成功的提示框和导出的 meta_description 里露过面，想在池子里认出一条
    # 是香草还是累计，只能右键「加载明细」把当前回测顶掉重跑一遍。
    _SAVED_POOL_COLUMNS = (
        ("name", "结果名称", 175, "w"),
        ("origin", "产生方式", 96, "center"),
        ("option", "期权类型", 110, "center"),
        ("strategy", "策略", 104, "center"),
        ("parameters", "策略参数", 240, "w"),
        ("source", "行情来源", 170, "w"),
        ("saved_at", "保留时间", 104, "center"),
    )
    # 自动伸缩的上限。同一列的内容长度差着数倍：「策略参数」在每日收盘时是
    # 一句话，在固定间隔时要写下绝对 / 相对 / σ 三种口径的等价换算；「行情
    # 来源」在模拟时只有一个 seed，在 Wind 时是代码加起止日加频率。定死一个
    # 宽度只能二选一——要么长的那种被截断，要么短的那种空出一大片。
    # 「期权类型」的上限按最长的子类中文名留：「熔断每日双固赔累计」9 个
    # 汉字，与历史结果窗口那张表给子类型的 150px 是同一个依据。
    # 加进第七列后上限之和（含 #0 列）约 1400px，超出常见窗口的容器宽度——
    # 但七列同时顶到各自上限要求每一格都是最长内容，实际到不了；真到了由
    # _fit_pool_columns_to_width 按比例削，不会像 minwidth 那样把右边的列裁掉。
    _SAVED_POOL_COLUMN_MAX = {
        "name": 230, "origin": 110, "option": 150, "strategy": 130,
        "parameters": 380, "source": 250, "saved_at": 116,
    }
    # 容器放不下、要按比例削的时候，这一列尽量保住的宽度。「策略参数」是这
    # 张表里唯一需要逐字读的列——固定间隔在这里写下绝对 / 相对 / σ 三种口径
    # 的等价换算，被削到基准宽以下就只剩一个「绝对 1.5；等价…」的开头，这一
    # 列等于白占。其余列要么是短标签（产生方式、策略、保留时间），要么截了
    # 还认得出（结果名称、行情来源的 Wind 串），所以只有这一列有这条保护。
    #
    # 300 → 250 是给第七列腾的，而且必须腾。这个数字直接进
    # _fit_pool_columns_to_width 的下限和：下限和一旦超过容器可用宽，那个函数
    # 就放弃缩放原样返回，于是总宽超出、右边的列被 Treeview 直接裁掉（没有横向
    # 滚动条，也不会有任何提示）。加一列就把下限和抬高约 48px，等于「窗口最窄
    # 能缩到多少」跟着退 48px——新增一列不该以此为代价。250 让七列的下限和落在
    # 829px、临界容器宽 879px，与改动前的六列（831 / 881px）持平。
    # 代价只在挤的时候出现：970px 容器下这一列从 305px 变成 294px。宽敞时不受
    # 影响，它照样能长到 _SAVED_POOL_COLUMN_MAX 给的 380。
    _SAVED_POOL_COLUMN_KEEP = {"parameters": 250}

    def _show_saved_comparison_page(self, *, navigate=True):
        """构建并渲染结果对比页；``navigate`` 为假时只构建不抢占当前页。"""
        self._hide_placeholder("compare")
        BacktestApp._discard_comparison_frame(self)
        container = self._compare_container
        for widget in container.winfo_children():
            widget.destroy()

        header = ttk.Frame(container, style="Surface.TFrame")
        header.pack(fill="x", padx=8, pady=(8, 3))
        header.columnconfigure(0, weight=1)
        self._saved_pool_count_var = tk.StringVar()
        ttk.Label(
            header, textvariable=self._saved_pool_count_var,
            style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=(f"自动保存在本机，重开程序后仍在（最多 "
                  f"{backtest_pool_store.MAX_RESULTS} 条，超出淘汰最旧的）；"
                  "显示 / 隐藏只换视图，不会重新回测"),
            style="SurfaceMuted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        pool = ttk.LabelFrame(container, text=" 已保留结果 ", padding=12)
        pool.pack(fill="x", padx=8, pady=(4, 8))
        toolbar = ttk.Frame(pool, style="Surface.TFrame")
        toolbar.pack(fill="x", pady=(0, 4))
        # 这一排分三组，从左到右：两个只换视图的（幽灵样式）、两个删数据的
        # （危险样式，与前面隔开一段单独成组）、一个产出文件的收在最右。
        #
        # 「全选 / 取消全选」管的是「显示」列那一列勾选框，「全部清空」倒的
        # 是整个结果池——两组名字都带各自的作用对象，区分靠的是样式（幽灵 vs
        # 危险红）、间距分组和确认框，不再靠名字本身绕开。
        #
        # 「删除选中」作用于**行选择**而不是「显示」勾选集，这条不能松：勾选
        # 集的语义只有"画不画到下方"，它跨会话落盘、每保留一条新结果自动勾
        # 上，而「全选」一键就能把它赋成全池——让它兼任删除范围，「全选 →
        # 删除选中」两击就清空了整个结果池，还绕过写明条数的确认框。改名之
        # 后这条更要盯紧：「全选」听上去比原先的「全部显示」更像是在给删除选料。
        ttk.Label(
            toolbar,
            text=("右键结果行可显示、看参数详情、加载明细、应用策略、"
                  "重命名或删除；Shift / ⌘ 可多选行"),
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        self._saved_pool_show_all_btn = ttk.Button(
            toolbar, text="全选", width=9, style="Ghost.TButton",
            command=self._select_all_saved_backtests,
        )
        self._saved_pool_show_all_btn.pack(side="left", padx=(4, 0))
        self._saved_pool_hide_all_btn = ttk.Button(
            toolbar, text="取消全选", width=9, style="Ghost.TButton",
            command=self._clear_saved_backtest_selection,
        )
        self._saved_pool_hide_all_btn.pack(side="left", padx=(4, 0))
        # 条数写进按钮文案，这样"要删几条"在点下去之前就已经在屏幕上；宽度
        # 按带条数时的最长文案定死，免得每次改选都抖一下版面。
        self._saved_pool_delete_btn = ttk.Button(
            toolbar, text="删除选中", width=13, style="Danger.TButton",
            command=self._prompt_delete_saved_backtest,
        )
        self._saved_pool_delete_btn.pack(side="left", padx=(20, 0))
        self._saved_pool_clear_btn = ttk.Button(
            toolbar, text="全部清空", width=9, style="Danger.TButton",
            command=self._clear_saved_backtest_pool,
        )
        # 两个红按钮之间也要留出手指宽度：它们的后果差着一整个结果池，而
        # 「删除选中」是这一排最常点的那个。
        self._saved_pool_clear_btn.pack(side="left", padx=(14, 0))
        # 导出收在最右，与两个红按钮隔开：它是这一排唯一不改变任何状态的
        # 动作，挨着删除放会让"点错一格"的代价过大。
        self._saved_pool_export_btn = ttk.Button(
            toolbar, text="导出 CSV", width=9,
            command=self._export_saved_comparison,
        )
        self._saved_pool_export_btn.pack(side="left", padx=(20, 0))

        pool_columns = BacktestApp._SAVED_POOL_COLUMNS
        tree_frame = ttk.Frame(pool, style="Surface.TFrame")
        tree_frame.pack(fill="x")
        tree = ttk.Treeview(
            tree_frame,
            columns=[key for key, _text, _width, _anchor in pool_columns],
            # extended：Shift / ⌘ 多选行，供「删除选中」与右键批量删除使用。
            # 于是本表两套"选中"彻底分家——打勾只管画不画到下方，选行才是
            # 动作的作用对象。
            show="tree headings", height=5, selectmode="extended",
        )
        tree.heading("#0", text="显示")
        tree.column("#0", width=46, minwidth=44, stretch=False, anchor="center")
        for key, text, width, anchor in pool_columns:
            # 与指标表同一套规则：表头跟本列数据同向对齐，且所有列一起
            # stretch——只让其中一两列可拉伸时，余量会全灌进那几列。
            tree.heading(key, text=text, anchor=anchor)
            tree.column(
                key, width=width, minwidth=max(44, width - 30),
                anchor=anchor, stretch=True,
            )
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")
        tree.bind("<Button-1>", self._toggle_saved_backtest_click)
        tree.bind("<space>", self._toggle_focused_saved_backtest)
        # 「删除选中」的文案与启停跟着行选择走，而行选择的变化不经过结果池
        # 刷新链路（用户拖一下选区、按住 ⌘ 点一行都不会重建表）。
        tree.bind("<<TreeviewSelect>>", self._sync_saved_pool_buttons)
        # 列宽要跟着容器走：构建的那一刻还没布局，宽度是 1，真正的宽度是在
        # 第一次 <Configure> 才知道的；之后窗口每次缩放也要重排。
        tree.bind("<Configure>", self._on_saved_pool_resize)
        # 单条快照的动作全部收进右键菜单。三个序列都绑：macOS 的 Tk 在不同
        # 小版本里把右键映射成 Button-2 或 Button-3，Control+左键又是系统级
        # 的右键等价操作；Windows / Linux 只认 Button-3。
        for sequence in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
            tree.bind(sequence, self._popup_saved_pool_menu)
        self._saved_pool_tree = tree
        self._saved_pool_menu = BacktestApp._build_saved_pool_menu(self)

        self._saved_comparison_content = ttk.Frame(
            container, style="Surface.TFrame")
        self._saved_comparison_content.pack(
            fill="both", expand=True, padx=2, pady=(0, 6))
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()
        if navigate:
            self._nb.select(self._compare_tab)

    def _refresh_saved_comparison_if_visible(self):
        tree = getattr(self, "_saved_pool_tree", None)
        try:
            visible = tree is not None and bool(tree.winfo_exists())
        except tk.TclError:
            visible = False
        if visible:
            self._refresh_saved_pool_tree()
            self._refresh_saved_comparison_view()

    @staticmethod
    def _neighbour_pool_row(previous, removed, children):
        """选中的行整批消失后，焦点该落到哪一行。

        取被删那几行里最靠后的位置，先往下找第一条还在的，没有再往上找。
        此前一律落到 ``children[-1]``，也就是表尾最新那条——逐条清理旧结果
        时焦点每删一次就从表头蹦到表尾，人得重新找一遍自己删到哪儿了。
        """
        alive = set(children)
        positions = [
            index for index, item in enumerate(previous) if item in removed]
        anchor = positions[-1] if positions else len(previous)
        for item in previous[anchor + 1:]:
            if item in alive:
                return item
        for item in reversed(previous[:anchor]):
            if item in alive:
                return item
        return children[-1] if children else None

    def _refresh_saved_pool_tree(self):
        tree = getattr(self, "_saved_pool_tree", None)
        if tree is None:
            return
        # 整个选区都要记下来，不能只记焦点：多选是删除类动作的作用对象，而
        # 这条刷新每勾选一次就会跑一遍，只恢复一行等于每打一个勾就把用户的
        # 多选悄悄塌成单选。
        previous = tuple(tree.get_children())
        picked = tuple(tree.selection())
        focus = tree.focus()
        for item in previous:
            tree.delete(item)
        for snapshot in self._saved_backtests.values():
            is_selected = snapshot.result_id in self._saved_comparison_selection
            tree.insert(
                "", "end", iid=snapshot.result_id,
                image=self._cb_sf_checked if is_selected else self._cb_sf_unchecked,
                values=BacktestApp._pad_tree_cells(
                    (
                        snapshot.name,
                        BacktestApp._saved_snapshot_origin_label(snapshot),
                        snapshot.option_label,
                        snapshot.strategy_label,
                        snapshot.parameter_summary, snapshot.source_label,
                        snapshot.saved_at.strftime("%m-%d %H:%M:%S"),
                    ),
                    [anchor for _key, _text, _width, anchor
                     in BacktestApp._SAVED_POOL_COLUMNS],
                ),
            )
        children = tree.get_children()
        survivors = [item for item in picked if item in children]
        if not survivors and focus in children:
            survivors = [focus]
        if not survivors and children:
            neighbour = BacktestApp._neighbour_pool_row(
                previous, set(picked) or {focus}, children)
            survivors = [neighbour] if neighbour else []
        if survivors:
            tree.selection_set(survivors)
            # 焦点优先留在原处：打勾不该把焦点拽到选区的第一行去。
            tree.focus(focus if focus in children else survivors[0])
        count_var = getattr(self, "_saved_pool_count_var", None)
        if count_var is not None:
            count_var.set(
                f"回测结果池 · 已保存 {len(self._saved_backtests)} 条 · "
                f"当前显示 {len(self._saved_comparison_selection)} 条")
        BacktestApp._autosize_saved_pool_columns(self)
        BacktestApp._sync_saved_pool_buttons(self)

    def _saved_pool_fonts(self):
        """结果池表的（单元格字体, 表头字体），供测量列宽用。

        与 style 里给 Treeview / Treeview.Heading 配的那两个保持一致；
        测量对象和渲染对象不是同一个字体的话，量出来的宽度没有意义。
        """
        fonts = getattr(self, "_saved_pool_fonts_cache", None)
        if fonts is None:
            fonts = (
                tkfont.Font(font=(_MONO_FONT_FAMILY, 9)),
                tkfont.Font(font=(_UI_FONT_FAMILY, 10, "bold")),
            )
            self._saved_pool_fonts_cache = fonts
        return fonts

    # 「显示」复选框列的固定宽度，以及表格边框吃掉的那几像素。算可用宽度时
    # 要先扣掉它们，否则每次都会多分出去五十来个像素。
    _SAVED_POOL_CHECK_WIDTH = 46
    _SAVED_POOL_TABLE_CHROME = 4

    def _autosize_saved_pool_columns(self):
        """按内容与当前容器宽度重新分配七列的宽度。

        两件事一起做，缺一件都不成立：

        1. 每列先按当前这一池的最长内容量一个需求宽，夹在基准宽与上限之间。
           同一列的内容长度差着数倍（每日收盘的参数是一句话，固定间隔要写下
           三种口径的等价换算），定死一个宽度只能二选一。
        2. 再把这些需求宽整体缩放到**容器放得下**。ttk.Treeview 没有横向滚
           动条，总宽超出容器时它既不压缩也不提示，直接把右边的列裁掉——此前
           「保留时间」整列看不见就是这么来的，minwidth 只在手动拖列边界时
           才起作用，拦不住这个。富余时按需求比例把余量分下去，长内容的列多
           吃一点；不够时按"高出下限的那部分"等比削减，谁也不会被压到看不清。

        容器宽度要等布局完成才有意义，构建的那一刻拿到的是 1，所以这里在拿
        不到有效宽度时只做第 1 步，第 2 步由 ``<Configure>`` 那次回调补上。
        """
        tree = getattr(self, "_saved_pool_tree", None)
        if tree is None:
            return
        try:
            cell_font, head_font = BacktestApp._saved_pool_fonts(self)
            rows = tree.get_children()
            widths, floors = {}, {}
            for key, text, base, _anchor in BacktestApp._SAVED_POOL_COLUMNS:
                # 表头两侧的留白比单元格宽：排序标记之外还有 Treeview 自己的
                # 表头内边距。
                need = head_font.measure(text) + 26
                for iid in rows:
                    need = max(need, cell_font.measure(tree.set(iid, key)) + 16)
                ceiling = BacktestApp._SAVED_POOL_COLUMN_MAX.get(key, base)
                floor = max(44, base - 30)
                # 下限用 minwidth 而不是基准宽：内容短的列（产生方式、策略、
                # 保留时间）占着基准宽不放，省下的空间本该归内容长的那几列。
                widths[key] = max(floor, min(need, ceiling))
                keep = BacktestApp._SAVED_POOL_COLUMN_KEEP.get(key)
                floors[key] = (
                    max(floor, min(widths[key], keep)) if keep else floor)
            widths = BacktestApp._fit_pool_columns_to_width(
                widths, floors,
                tree.winfo_width() - BacktestApp._SAVED_POOL_CHECK_WIDTH
                - BacktestApp._SAVED_POOL_TABLE_CHROME)
            for key, width in widths.items():
                tree.column(key, width=width)
        except tk.TclError:
            # 表已随窗口销毁：列宽只是展示，不该反过来打断刷新链路。
            pass

    @staticmethod
    def _fit_pool_columns_to_width(widths, floors, available):
        """把各列的需求宽缩放到 ``available``；放不下下限时原样返回。

        纯函数，不碰控件——这样"够宽 / 不够宽 / 宽度还不知道"三种分支可以直
        接测，不必真去摆一个窗口出来。
        """
        total = sum(widths.values())
        if not total or available <= sum(floors.values()):
            # 宽度还没算出来（构建那一刻是 1），或者窄到连下限都排不下：
            # 后者再缩也只是把每列都压成看不清，不如维持原样让人去拉窗口。
            return dict(widths)
        if total > available:
            slack = total - sum(floors.values())
            cut = total - available
            return {
                key: int(width - (width - floors[key]) * cut / slack)
                for key, width in widths.items()
            }
        extra = available - total
        return {
            key: width + int(extra * width / total)
            for key, width in widths.items()
        }

    def _on_saved_pool_resize(self, event):
        """容器宽度变了才重排列宽——``<Configure>`` 高度变化也会发。"""
        if event.width == getattr(self, "_saved_pool_last_width", None):
            return
        self._saved_pool_last_width = event.width
        BacktestApp._autosize_saved_pool_columns(self)

    def _sync_saved_pool_buttons(self, _event=None):
        """把工具栏五个按钮的启停与文案对齐当前状态。

        「有没有可操作对象」在这个项目里一律用置灰表达，而不是点了再弹一句
        提示（历史结果窗口的三个按钮、右键「加载明细」、「＋ 保留当前结果」
        都是这么做的）。此前这一排只有「全部清空」兑现了这条规则，清空之
        后页面上会留下三个照样可点、点了什么也不发生的按钮。

        按钮随对比页重建，旧引用可能已经被销毁；拿不到就当这一排还没建出
        来，静默跳过——状态同步不该反过来打断刷新链路。
        """
        pool = getattr(self, "_saved_backtests", None) or {}
        shown = getattr(self, "_saved_comparison_selection", None) or set()
        picked = len(BacktestApp._picked_saved_backtest_ids(self))
        for attr, enabled in (
            # 全都勾上时「全选」已经没有可勾的了，同「清空 0 条」一样
            # 是个点了不会有任何变化的按钮。
            ("_saved_pool_show_all_btn",
             bool(pool) and any(rid not in shown for rid in pool)),
            ("_saved_pool_hide_all_btn", bool(shown)),
            # 导出的作用域是「当前显示的那几条」，一条不显示时它导不出东西；
            # 原先是点了弹一句「没有可导出的结果」，那句兜底仍然留着防御。
            ("_saved_pool_export_btn", bool(shown)),
            ("_saved_pool_delete_btn", picked > 0),
            ("_saved_pool_clear_btn", bool(pool)),
        ):
            button = getattr(self, attr, None)
            try:
                if button is None or not button.winfo_exists():
                    continue
                button.state(["!disabled"] if enabled else ["disabled"])
            except tk.TclError:
                continue
        delete_btn = getattr(self, "_saved_pool_delete_btn", None)
        try:
            if delete_btn is not None and delete_btn.winfo_exists():
                delete_btn.configure(
                    text=f"删除选中 ({picked})" if picked > 1 else "删除选中")
        except tk.TclError:
            pass

    def _build_saved_pool_menu(self):
        """结果池的行右键菜单：只放作用于「点中那一条」的动作。

        第一项的文案在弹出时按当前勾选状态改写，所以它固定占索引 0。
        """
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="显示",
                         command=self._toggle_focused_saved_backtest)
        menu.add_separator()
        # 「参数详情」排在「加载明细」前面：前者读的是输入、恒可用、不改变
        # 任何状态，后者要重放并顶掉当前回测。先给代价小的那个。
        menu.add_command(label="参数详情…",
                         command=self._show_saved_snapshot_params)
        menu.add_command(label="加载明细",
                         command=self._load_saved_snapshot_detail)
        menu.add_command(label="应用策略",
                         command=self._apply_selected_snapshot_to_form)
        menu.add_command(label="重命名…",
                         command=self._prompt_rename_saved_backtest)
        menu.add_separator()
        menu.add_command(label="删除",
                         command=self._prompt_delete_saved_backtest)
        return menu

    # 只有「加载明细」按有没有重放配方启停，名字要写全：同排的「参数详情」
    # 读的是签名，任何快照都打得开，两者不能共用一个 DETAIL 索引。
    _POOL_MENU_LOAD_DETAIL_INDEX = 3

    # 除末项「删除」外，其余五项都只对一条快照说得通，多选时一并置灰。
    _POOL_MENU_SINGLE_INDEXES = (0, 2, 3, 4, 5)
    _POOL_MENU_DELETE_INDEX = 7

    def _popup_saved_pool_menu(self, event):
        """右键先确定作用对象，再弹菜单。

        空白处右键没有作用对象，直接不弹——弹一份点了只会跳「请选择结果」
        的菜单，比不弹更费解。

        点在**已经选中的行**上时保留整个选区，点在选区外才重置成这一行：
        反过来就永远右键不出多条，攒好的选区在菜单弹出前就被打散了。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return "break"
        tree = getattr(self, "_saved_pool_tree", None)
        menu = getattr(self, "_saved_pool_menu", None)
        if tree is None or menu is None:
            return None
        result_id = tree.identify_row(event.y)
        if not result_id:
            return None
        if result_id not in tree.selection():
            tree.selection_set(result_id)
        tree.focus(result_id)
        picked = BacktestApp._picked_saved_backtest_ids(self)
        multiple = len(picked) > 1
        menu.entryconfigure(
            0,
            label=("取消显示" if result_id in self._saved_comparison_selection
                   else "显示"))
        for index in BacktestApp._POOL_MENU_SINGLE_INDEXES:
            menu.entryconfigure(
                index, state=("disabled" if multiple else "normal"))
        # 没有重放配方就置灰：本功能上线前保留的快照只有汇总层，点了只能
        # 弹一句"没有配方"，不如直接让它点不动。
        snapshot = self._saved_backtests.get(result_id)
        if not multiple:
            menu.entryconfigure(
                BacktestApp._POOL_MENU_LOAD_DETAIL_INDEX,
                state=("normal" if getattr(snapshot, "replay", None)
                       else "disabled"))
        # 末项文案跟着选中条数走，这样"这一下要删几条"在菜单里就已经写着。
        menu.entryconfigure(
            BacktestApp._POOL_MENU_DELETE_INDEX,
            label=(f"删除选中的 {len(picked)} 条" if multiple else "删除"))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _saved_pool_actions_allowed(self):
        if getattr(self, "_active_job", None) is None:
            return True
        messagebox.showinfo(
            "任务运行中", "请等待当前任务完成后再修改或选择保存结果。")
        return False

    def _toggle_saved_backtest_click(self, event):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return "break"
        tree = self._saved_pool_tree
        result_id = tree.identify_row(event.y)
        if not result_id or tree.identify_column(event.x) != "#0":
            return None
        # 只挪焦点、不碰选区：打勾管的是"画不画到下方"，动一下它不该把用户
        # 攒起来的多选选区塌掉。
        tree.focus(result_id)
        self._toggle_saved_backtest_selection(result_id)
        return "break"

    def _toggle_focused_saved_backtest(self, _event=None):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return "break"
        result_id = self._focused_saved_backtest_id()
        if result_id:
            self._toggle_saved_backtest_selection(result_id)
        return "break"

    def _persist_pool_view(self):
        """记下当前显示了哪几条，供下次启动恢复。

        显式调用而不是塞进刷新链路：刷新每次勾选都会走好几遍，写盘该只发生
        在「选择真的变了」的那几个点上。
        """
        backtest_pool_store.write_view_state(self._saved_comparison_selection)

    def _toggle_saved_backtest_selection(self, result_id):
        if result_id in self._saved_comparison_selection:
            self._saved_comparison_selection.remove(result_id)
        elif result_id in self._saved_backtests:
            self._saved_comparison_selection.add(result_id)
        BacktestApp._persist_pool_view(self)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _select_all_saved_backtests(self):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        # 原地改而不是重新绑定属性：「取消全选」走的是 .clear()，两个入口对
        # 同一个集合一个换对象一个改内容，拿着旧引用的地方就会看到两种结果。
        self._saved_comparison_selection.clear()
        self._saved_comparison_selection.update(self._saved_backtests)
        BacktestApp._persist_pool_view(self)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _clear_saved_backtest_selection(self):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        self._saved_comparison_selection.clear()
        BacktestApp._persist_pool_view(self)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _focused_saved_backtest_id(self):
        tree = getattr(self, "_saved_pool_tree", None)
        if tree is None:
            return None
        selection = tree.selection()
        return selection[0] if selection else (tree.focus() or None)

    def _picked_saved_backtest_ids(self):
        """当前行选择里仍然存在的那几条，按结果池顺序返回。

        行选择是删除类动作的唯一作用对象。选区里可能留着已经被删掉的行
        （确认框跑的是嵌套事件循环，开着的时候池子仍可能变），表也可能已经
        销毁，所以这里统一过滤一次再交出去。

        **只认选区，不回退到焦点行。** 焦点是不可见的：⌘ 点掉最后一个选中
        行之后表上一个高亮都没有，焦点却还留在那一行；回退到它，「删除选中」
        就会在屏幕上什么都没选中的情况下照样可点，并删掉一条看不出被选中的
        结果——而删除是真删文件、不可撤销。键盘那条路径（空格键切换显示）走
        的是 ``_focused_saved_backtest_id``，不受这里影响。
        """
        tree = getattr(self, "_saved_pool_tree", None)
        try:
            selection = tuple(tree.selection()) if tree is not None else ()
        except tk.TclError:
            selection = ()
        picked = {str(item) for item in selection}
        pool = getattr(self, "_saved_backtests", None) or {}
        return [result_id for result_id in pool if result_id in picked]

    def _prompt_rename_saved_backtest(self):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshot = self._saved_backtests[result_id]
        while True:
            new_name = simpledialog.askstring(
                "重命名回测结果", "新的结果名称：",
                initialvalue=snapshot.name, parent=self,
            )
            if new_name is None:
                return
            try:
                self._rename_saved_backtest(result_id, new_name)
                break
            except ValueError as exc:
                messagebox.showerror("名称无效", str(exc))
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()
        self._set_status(f"已将回测结果重命名为『{snapshot.name}』")

    @staticmethod
    def _saved_pool_name_digest(snapshots):
        """把一批快照压成一句可读的名字串：前三条点名，其余记数。

        确认框与状态栏共用。必须截断：结果名是用户自由输入且没有长度上限
        （``_validate_saved_result_name`` 只校验非空与唯一），二十个名字会
        把不能滚动的 Tk 消息框撑出屏幕。写法与结果包淘汰那处一致。
        """
        snapshots = list(snapshots)
        names = "、".join(f"『{snapshot.name}』" for snapshot in snapshots[:3])
        more = f" 等 {len(snapshots)} 条" if len(snapshots) > 3 else ""
        return f"{names}{more}"

    def _prompt_delete_saved_backtest(self):
        """删除当前选中的结果：一条和多条走同一条路。

        作用对象是**行选择**，不是「显示」勾选集——理由见工具栏那段注释。
        单条时的标题与正文与此前逐字一致；多条才换成点名前三条的版本。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        picked = BacktestApp._picked_saved_backtest_ids(self)
        if not picked:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshots = [self._saved_backtests[result_id] for result_id in picked]
        # 结果池已落盘，这是真删文件。措辞必须比「从当前会话删除」重——那
        # 句的轻描淡写建立在「反正关程序也会没」之上。
        tail = "将同时删掉本机上的结果文件，此操作不可撤销。"
        if len(picked) == 1:
            title = "删除回测结果"
            message = f"删除『{snapshots[0].name}』？{tail}"
        else:
            rest = len(self._saved_backtests) - len(picked)
            # 选满整池时它和「全部清空」是同一件事，不点名的话用户不会意
            # 识到自己正在清空。
            title = "删除选中的结果"
            message = (
                f"删除选中的 {len(picked)} 条？\n\n"
                f"{BacktestApp._saved_pool_name_digest(snapshots)}\n\n"
                f"{tail}\n"
                + (f"删除后结果池还剩 {rest} 条。" if rest
                   else "删除后结果池将变空。"))
        if not messagebox.askyesno(title, message):
            return
        # 确认框跑的是嵌套事件循环，开着的这段时间 after 回调照常执行，池子
        # 可能已经变了；_delete_saved_backtest 用的是无默认值的 pop，撞上已
        # 经不在的 id 会 KeyError，把批量删除停在半路且不报任何状态。
        deleted = [
            self._delete_saved_backtest(result_id)
            for result_id in picked if result_id in self._saved_backtests
        ]
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()
        if not deleted:
            self._set_status("选中的结果已不在结果池里，未删除任何东西")
            return
        digest = BacktestApp._saved_pool_name_digest(deleted)
        # 删除不可逆，必须说出删了哪些而不只是几条。
        head = (f"已删除{digest}" if len(deleted) == 1
                else f"已删除 {len(deleted)} 条结果：{digest}")
        self._set_status(f"{head}  |  剩余 {len(self._saved_backtests)} 条")

    def _clear_saved_backtest_pool(self):
        """清空整池：逐条删除，同时删掉本机上的结果文件。

        和右键菜单的「删除」是同一件事，只是作用域从一条变成全部——攒到
        二十条时逐条删要点二十次、确认二十次。确认框的标题写全「清空结果
        池」而不是按钮上那个「全部清空」：按钮靠位置和红色说明自己清的是
        什么，弹出来的框没有那两样，只能靠字面。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        total = len(self._saved_backtests)
        if not total:
            return
        if not messagebox.askyesno(
                "清空结果池",
                f"删除全部 {total} 条已保留结果？将同时删掉本机上的结果文件，"
                "此操作不可撤销。"):
            return
        # 先固化 id 列表：_delete_saved_backtest 会就地改这个字典。
        for result_id in list(self._saved_backtests):
            self._delete_saved_backtest(result_id)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()
        self._set_status(f"已清空结果池  |  删除 {total} 条已保留结果")

    def _selected_saved_backtests(self):
        """纯查询：按结果池顺序返回已勾选的快照。"""
        return [
            snapshot for result_id, snapshot in self._saved_backtests.items()
            if result_id in self._saved_comparison_selection
        ]

    def _discard_comparison_frame(self):
        """丢掉结果区骨架与图表引用，让下一次刷新重新建。

        这里不调用 ``plt.close``：图表是直接构造的 ``Figure``，从未登记进
        pyplot 的 figure manager，对它调用 close 是空操作。真正生效的是销毁
        canvas widget 之后由 GC 回收，clear 只是先松开轴上引用的数据。
        """
        figure = getattr(self, "_comparison_chart_figure", None)
        if figure is not None:
            figure.clear()
        self._comparison_frame = None
        self._comparison_chart_figure = None
        self._comparison_chart_ax = None
        self._comparison_chart_canvas = None
        self._comparison_tree = None
        self._comparison_rows = {}
        self._comparison_summary = None
        self._comparison_variable_var = None
        self._comparison_variable_accent = None

    def _render_saved_comparison_empty(self, title, detail):
        """把内容区换成居中的空状态说明，并释放上一张图表。"""
        content = self._saved_comparison_content
        BacktestApp._discard_comparison_frame(self)
        for widget in content.winfo_children():
            widget.destroy()
        empty = ttk.Frame(content, style="Surface.TFrame")
        empty.place(relx=0.5, rely=0.42, anchor="center")
        ttk.Label(
            empty, text=title, style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 14, "bold"),
        ).pack(pady=(0, 5))
        ttk.Label(
            empty, text=detail, style="SurfaceMuted.TLabel", justify="center",
            # 这里也会显示异常原文，长度不可控，必须折行。
            wraplength=420,
        ).pack()

    def _refresh_saved_comparison_view(self):
        # 出错不弹模态框：这条链路每次勾选都会走，弹窗会把人挡在页面外。
        # 直接把原因写进空状态区，它本来就是说明"为什么现在没东西看"的地方。
        try:
            snapshots = self._selected_saved_backtests()
            if not snapshots:
                # 「空」现在有三种含义，必须分清：本机从没存过 / 存过但载入
                # 失败 / 存着但全被隐藏了。第三种最容易被误读成数据没了——
                # 用户刚点完「取消全选」，页面回一句「尚未选择」就像清空了。
                pooled = len(self._saved_backtests)
                load_error = getattr(self, "_saved_pool_load_error", "")
                if pooled:
                    self._render_saved_comparison_empty(
                        "当前没有显示任何结果",
                        f"已保留的 {pooled} 条都在上方结果池里，"
                        "点「显示」列或按「全选」即可放上来。")
                elif load_error:
                    self._render_saved_comparison_empty(
                        "没有可显示的结果", load_error)
                else:
                    self._render_saved_comparison_empty(
                        "尚未保留任何结果",
                        "先运行回测，再点『＋ 保留当前结果到对比』。"
                        "保留的结果会自动存在本机，重开程序后仍在。")
                return
            content = self._saved_comparison_content
            summary, daily_curves = self._saved_comparison_payload(snapshots)
        except Exception as exc:
            self._render_saved_comparison_empty("无法生成对比", str(exc))
            return
        BacktestApp._ensure_comparison_frame(self, content)
        self._populate_comparison_view(summary, daily_curves)

    def _ensure_comparison_frame(self, content):
        """结果区骨架只建一次；已经建好就直接复用。"""
        existing = getattr(self, "_comparison_frame", None)
        try:
            if existing is not None and existing.winfo_exists():
                return
        except tk.TclError:
            pass
        for widget in content.winfo_children():
            widget.destroy()
        self._comparison_results = {}
        self._comparison_ranking = None
        self._build_comparison_variable_card(content)
        frame = ttk.Frame(content, style="Surface.TFrame")
        frame.pack(fill="both", expand=True, padx=8, pady=(1, 0))
        self._build_current_comparison_view(
            frame, show_curve_controls=False,
            ranking_title="已选结果指标（点列头换排序）")
        self._comparison_frame = frame

    def _build_comparison_variable_card(self, parent):
        """对比说明卡：这次比的变量是哪一项，以及看数时要留意什么。

        这个位置原先是一张"选中结果明细卡"，摆的是名称、五个指标和参数串
        ——而那些数字指标表里全有，参数与来源结果池表里全有，独有的只剩一
        个交易日数（已并入指标表）。真正该占这块版面的是本页唯一的结论性
        信息：这两条结果之间到底差在哪。
        """
        card = tk.Frame(
            parent, bg=PALETTE["surface_alt"],
            highlightbackground=PALETTE["border_soft"], highlightthickness=1)
        card.pack(fill="x", padx=8, pady=(8, 4))

        self._comparison_variable_accent = tk.Frame(
            card, bg=PALETTE["border_soft"], width=4)
        self._comparison_variable_accent.pack(side="left", fill="y")

        body = tk.Frame(card, bg=PALETTE["surface_alt"], padx=14, pady=10)
        body.pack(side="left", fill="both", expand=True)
        body.columnconfigure(0, weight=1)

        self._comparison_variable_var = tk.StringVar(value="")
        self._comparison_variable_label = tk.Label(
            body, textvariable=self._comparison_variable_var,
            bg=PALETTE["surface_alt"], fg=PALETTE["text"],
            font=(_UI_FONT_FAMILY, 14, "bold"), anchor="w", justify="left",
            wraplength=980,
        )
        self._comparison_variable_label.grid(row=0, column=0, sticky="w")

        # 逐字段的取值网格：一行一个字段，取值按 #序号 排开，序号与指标表
        # 首列一致。此前这些值拼成 "A vs B vs C" 塞进标题，三条以上就分不
        # 清谁是谁了。
        self._comparison_field_grid = tk.Frame(
            body, bg=PALETTE["surface_alt"])
        self._comparison_field_grid.grid(row=1, column=0, sticky="w",
                                         pady=(6, 0))

        self._comparison_same_var = tk.StringVar(value="")
        tk.Label(
            body, textvariable=self._comparison_same_var,
            bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9), anchor="w", justify="left",
            wraplength=980,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        self._comparison_caveat_var = tk.StringVar(value="")
        self._comparison_caveat_label = tk.Label(
            body, textvariable=self._comparison_caveat_var,
            bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9), anchor="w", justify="left",
            wraplength=980,
        )
        self._comparison_caveat_label.grid(row=3, column=0, sticky="w")

    def _fill_comparison_field_grid(self, fields):
        """一行一个差异字段：字段名 + 各结果的取值，取值按 #序号 排开。

        序号与指标表首列一致，所以三条以上也认得出谁是谁——此前这些值拼成
        "A vs B vs C" 塞在标题里，既分不清对应关系，又能把标题撑到七八十字。
        """
        grid = getattr(self, "_comparison_field_grid", None)
        if grid is None:
            return
        try:
            for widget in list(grid.winfo_children()):
                widget.destroy()
        except tk.TclError:
            return
        for row, (label, values) in enumerate(fields):
            tk.Label(
                grid, text=label, bg=PALETTE["surface_alt"],
                fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 9),
                anchor="w", width=12,
            ).grid(row=row, column=0, sticky="w", pady=(0, 2))
            if values is None:
                # 价格序列摘要之类：只说不同，没有可读的值。
                tk.Label(
                    grid, text="（不同）", bg=PALETTE["surface_alt"],
                    fg=PALETTE["text"], font=(_UI_FONT_FAMILY, 10),
                    anchor="w",
                ).grid(row=row, column=1, sticky="w", pady=(0, 2))
                continue
            for column, value in enumerate(values):
                cell = tk.Frame(grid, bg=PALETTE["surface_alt"])
                cell.grid(row=row, column=column + 1, sticky="w",
                          padx=(0, 18), pady=(0, 2))
                tk.Label(
                    cell, text=f"#{column + 1}", bg=PALETTE["surface_alt"],
                    fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 8),
                ).pack(side="left", padx=(0, 4))
                tk.Label(
                    cell, text=str(value), bg=PALETTE["surface_alt"],
                    fg=PALETTE["text"],
                    font=(_UI_FONT_FAMILY, 10, "bold"),
                ).pack(side="left")

    def _refresh_comparison_variable_card(self, snapshots):
        """把变量结论与看数提醒写进说明卡。"""
        variable_var = getattr(self, "_comparison_variable_var", None)
        accent = getattr(self, "_comparison_variable_accent", None)
        if variable_var is None or accent is None:
            return
        same_var = self._comparison_same_var
        caveat_var = self._comparison_caveat_var
        try:
            if not accent.winfo_exists():
                return
        except tk.TclError:
            return

        if len(snapshots) < 2:
            variable_var.set(
                "再勾选一条即可比对" if snapshots else "尚未选择对比结果")
            same_var.set(
                "页面会说明所选结果之间差在哪一项，数值判断留给你。")
            caveat_var.set("")
            accent.configure(bg=PALETTE["border_soft"])
            BacktestApp._fill_comparison_field_grid(self, [])
            return

        summary = BacktestApp._comparison_variable_summary(snapshots)
        state = summary["state"]
        variable_var.set(summary["headline"])
        same_var.set(summary["rest"])
        BacktestApp._fill_comparison_field_grid(self, summary["fields"])
        accent.configure(bg={
            "single": PALETTE["success"],
            "identical": PALETTE["primary"],
            "multiple": PALETTE["warning"],
        }[state])
        caveats = self._saved_comparison_warnings(snapshots)
        caveat_var.set(
            "\n".join(f"· {caveat}" for caveat in caveats) if caveats else "")

    def _show_history_recommendation(
            self, recommendations, ranking, notes=None, source_label=None,
            window_results=None, history_state=None, details=None,
            loaded_meta=None):
        """只在历史择优页渲染严格区间结论与排名。

        ``loaded_meta`` 非空表示这是从结果包载入的，页面要标明来源并禁掉
        逐段下钻。它由渲染入口统一接管而不是让各调用方自己维护标记——真跑
        一轮却忘了清，新结果就会继续挂着「载入结果」横幅、下钻也被误禁，
        而这类遗漏不会报错。
        """
        self._history_loaded_meta = loaded_meta or None
        container = self._history_results_container
        view_attrs = (
            "_history_ranking", "_history_period_rows",
            "_history_selected_period_var", "_history_detail_var",
            "_history_rank_tree", "_history_rank_rows",
            "_history_window_summary", "_history_replay_index",
            "_history_replay_window_var", "_history_replay_window_combo",
            "_history_chart_figure", "_history_chart_ax",
            "_history_chart_canvas", "_history_chart_metric_var",
            "_history_chart_metric_combo", "_history_chart_hint_var",
            "_history_chart_selected_by_period", "_history_pairs_cache",
            "_history_chart_color_map", "_history_chart_marker_map",
            "_history_conclusion_card", "_history_conclusion_accent",
            "_history_conclusion_badge_var", "_history_conclusion_name_var",
            "_history_conclusion_stats", "_history_splitter",
            "_history_lookbacks", "_history_uses_strict_metric",
            "_history_uses_window_equal_metric",
            # 排名口径也属于视图状态：不在这张清单里就会跨次渲染残留，
            # 渲染失败时也回滚不掉。
            "_history_result_objective", "_history_result_objective_var",
            "_history_objectives_available",
        )
        missing = object()
        old_view_state = {
            name: getattr(self, name, missing) for name in view_attrs
        }
        staging = ttk.Frame(container, style="Surface.TFrame")
        self._history_ranking = ranking
        self._history_last_notes = notes
        self._history_last_source_label = source_label
        self._history_last_state = history_state
        BacktestApp._update_history_header_summary(self)
        BacktestApp._sync_history_save_button(self)
        if notes:
            first_note = str(notes[0])
            short_note = first_note if len(first_note) <= 92 else first_note[:89] + "…"
            note_bar = tk.Frame(staging, bg=PALETTE["warning_light"], bd=0)
            note_bar.pack(fill="x", padx=8, pady=(0, 4))
            tk.Frame(note_bar, bg=PALETTE["warning"], width=3).pack(side="left", fill="y")
            inner = tk.Frame(note_bar, bg=PALETTE["warning_light"], padx=12, pady=6)
            inner.pack(side="left", fill="x", expand=True)
            tk.Label(
                inner, text=f"⚠ {len(notes)} 条说明：{short_note}",
                bg=PALETTE["warning_light"], fg=PALETTE["warning"],
                font=(_UI_FONT_FAMILY, 9), anchor="w",
            ).pack(side="left", fill="x", expand=True)
            ttk.Button(
                inner, text="查看全部", width=8,
                command=lambda: messagebox.showinfo(
                    "历史择优说明", "\n\n".join(str(note) for note in notes)),
            ).pack(side="right", padx=(6, 0))

        # 结果页不再整体滚动：顶部结论固定，排名表与图表由一条可拖分割线
        # 分配剩余高度，各自内部该滚的照旧滚。
        history_body = ttk.Frame(staging, style="Surface.TFrame")
        history_body.pack(fill="both", expand=True, padx=4, pady=(2, 2))
        history_lookbacks = (
            history_state.get("history_lookbacks")
            if isinstance(history_state, Mapping) else None)
        try:
            self._build_history_comparison_view(
                history_body, recommendations, ranking, window_results,
                history_lookbacks=history_lookbacks, details=details)
        except Exception:
            new_figure = getattr(self, "_history_chart_figure", None)
            old_figure = old_view_state.get(
                "_history_chart_figure", missing)
            if (new_figure is not None and new_figure is not old_figure):
                plt.close(new_figure)
            staging.destroy()
            for name, value in old_view_state.items():
                if value is missing:
                    try:
                        delattr(self, name)
                    except AttributeError:
                        pass
                else:
                    setattr(self, name, value)
            raise

        # 顶部摘要必须在视图构建**之后**再刷一次。上面那次调用发生在
        # _build_history_comparison_view 之前，而排名口径要到它里面的
        # _build_history_scope_note 才写进 _history_result_objective_var——
        # 早刷的那次读到的是上一次结果的口径。于是换过一次口径再跑第二次
        # 优选时，表头标的是「增量收益 ↓」，顶部 pill 却还写着「增量信噪
        # 比」，而这两个口径给出的第一名可能正好相反。
        BacktestApp._update_history_header_summary(self)

        # 新视图完整构建后再一次性替换，渲染失败不破坏上次成功页面。
        old_figure = old_view_state.get("_history_chart_figure", missing)
        if (old_figure is not missing and old_figure is not None
                and old_figure is not self._history_chart_figure):
            plt.close(old_figure)
        for widget in list(container.winfo_children()):
            if widget is not staging:
                widget.destroy()
        staging.pack(fill="both", expand=True)
        # 结果就绪后收起候选配置：配置已经冻结进本次实验，继续占据上半屏
        # 只会把排名和图表挤出视口。用户仍可用同一个按钮展开改参数。
        if getattr(self, "_history_config_visible", True):
            BacktestApp._toggle_history_config_panel(self)
        self._nb.select(self._history_tab)

    # 排名表结构与数据无关，骨架建一次就够。展示的全是每条结果自己的绝对
    # 指标；谁跟谁比、比哪一项，交给排序和并列的曲线回答。
    # 各列的初始宽度按表头与内容实际需要分配，总和接近容器宽度——配合下面
    # 的全列 stretch，窗口缩放时平均摊到每列的增量很小，比例不会走样。
    _COMPARISON_RANKING_COLUMNS = (
        ("rank", "#", 46),
        ("strategy", "结果名称", 260),
        ("total_net_pnl", "期末净损益", 110),
        ("total_tc", "总成本", 100),
        ("max_drawdown", "最大回撤", 110),
        ("rehedge_count", "再触发/成交", 115),
        ("turnover", "换手额", 125),
        ("n_trade_days", "交易日数", 90),
    )

    # 各列的对齐方向。未列出的都是数字列，一律右对齐——与优选页同一套规则。
    _COMPARISON_COLUMN_ANCHORS = {"rank": "center", "strategy": "w"}

    # 不可排序的列：`#` 就是当前显示顺序的行号，本身不是数据字段
    # （见 _comparison_sorted_rows）。这些列既不绑排序命令也不打排序标记。
    _COMPARISON_UNSORTABLE_COLUMNS = frozenset({"rank"})

    @staticmethod
    def _comparison_column_anchor(key):
        return BacktestApp._COMPARISON_COLUMN_ANCHORS.get(key, "e")

    @staticmethod
    def _pad_tree_cells(values, anchors):
        """按对齐方向给单元格补一个空格，等价于给每格加内边距。

        ttk.Treeview 没有单元格内边距这回事：文本严格贴着 anchor 那一侧的
        列边界。列宽又随窗口伸缩，靠调宽某一列治不了根。表格值不参与任何
        解析（行数据另存），补空格不影响逻辑。
        """
        padded = []
        for value, anchor in zip(values, anchors):
            text = str(value)
            if not text:
                padded.append(text)
            elif anchor == "e":
                padded.append(f"{text} ")
            elif anchor == "w":
                padded.append(f" {text}")
            else:
                padded.append(text)
        return tuple(padded)

    @staticmethod
    def _pad_comparison_row(values):
        """指标表的单元格内边距。"""
        return BacktestApp._pad_tree_cells(values, [
            BacktestApp._comparison_column_anchor(key)
            for key, _text, _width in
            BacktestApp._COMPARISON_RANKING_COLUMNS
        ])

    # 点列头换排序时，各列第一次点下去该往哪边排。收益是越高越好，先给降序；
    # 成本、回撤、换手、触发次数先给升序——第一次点就看到自己想找的那一头。
    _COMPARISON_SORT_DESCENDING_FIRST = frozenset({"total_net_pnl"})
    # 排序取的是原始数值，不是表里那串格式化文本。
    _COMPARISON_SORT_KEYS = {
        "strategy": "strategy", "total_net_pnl": "total_net_pnl",
        "total_tc": "total_tc", "max_drawdown": "max_drawdown",
        "rehedge_count": "rehedge_count", "turnover": "turnover",
        "n_trade_days": "n_trade_days",
    }

    @staticmethod
    def _comparison_sorted_rows(summary, sort_key, descending):
        """按指定列排出显示顺序；未指定时保持保存顺序。

        ``rank`` 不是数据里的字段，它就是当前显示顺序的行号——本页没有唯一
        的权威排序，按哪一列看是用户自己选的。
        """
        rows = []
        if summary is not None and not getattr(summary, "empty", True):
            rows = [row.to_dict() for _index, row in summary.iterrows()]
        if not sort_key or sort_key == "rank":
            return rows

        def key_of(item):
            if sort_key == "strategy":
                # 名称按字典序；缺失当空串，不与数值列共用缺失规则。
                return (0, str(item.get("strategy", "")))
            value = BacktestApp._comparison_finite(
                item.get(BacktestApp._COMPARISON_SORT_KEYS.get(
                    sort_key, sort_key)))
            # 缺失值恒排在末尾，无论升降序——它们不是"最好"也不是"最差"。
            if value is None:
                return (1, 0.0)
            return (0, -value if descending else value)

        rows.sort(key=key_of)
        return rows

    def _sort_comparison_by(self, column):
        """点列头换排序：同一列再点一次反向，换列则用该列的默认方向。"""
        if getattr(self, "_comparison_sort_column", None) == column:
            self._comparison_sort_descending = not getattr(
                self, "_comparison_sort_descending", False)
        else:
            self._comparison_sort_column = column
            self._comparison_sort_descending = (
                column in BacktestApp._COMPARISON_SORT_DESCENDING_FIRST)
        summary = getattr(self, "_comparison_summary", None)
        if summary is None:
            return
        self._populate_comparison_view(
            summary, getattr(self, "_comparison_daily_curves", {}))

    def _refresh_comparison_sort_markers(self):
        """当前排序列打实心箭头，其余列用 ⇅ 表示可点。"""
        tree = getattr(self, "_comparison_tree", None)
        if tree is None:
            return
        active = getattr(self, "_comparison_sort_column", None)
        descending = getattr(self, "_comparison_sort_descending", False)
        for key, text, _width in BacktestApp._COMPARISON_RANKING_COLUMNS:
            if key in BacktestApp._COMPARISON_UNSORTABLE_COLUMNS:
                marker = ""
            elif key == active:
                marker = " ▼" if descending else " ▲"
            else:
                marker = " ⇅"
            try:
                tree.heading(
                    key, text=f"{text}{marker}",
                    anchor=BacktestApp._comparison_column_anchor(key))
            except tk.TclError:
                return

    def _build_current_comparison_view(
            self, parent, *, show_curve_controls=True,
            ranking_title="当前路径排名（金额与持仓口径一致）"):
        """建结果区骨架：图表画布、排名表、详情格。

        这里一个数据字段都不读。勾选一次就把整块 destroy 重建，代价是每次
        新造一个 matplotlib Figure 和整张表——批量验证十条候选，就是主线程
        上十次这样的重建。骨架与数据分开之后，刷新只剩下改值。
        """
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)

        chart_box = ttk.LabelFrame(
            parent, text=" 累计净损益（按真实交易日汇总，已扣成本） ", padding=12)
        chart_box.grid(row=0, column=0, sticky="nsew", pady=(4, 3))
        controls = ttk.Frame(chart_box, style="Surface.TFrame")
        controls.pack(fill="x", pady=(0, 2))
        self._comparison_curve_hint_var = tk.StringVar(
            value="显示曲线:" if show_curve_controls else "")
        # 不显示曲线勾选框时这一行没有内容可放；条数顶部已经写了。
        ttk.Label(
            controls, textvariable=self._comparison_curve_hint_var,
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(2, 5))
        self._comparison_show_curve_controls = bool(show_curve_controls)
        self._comparison_curve_choices = None
        if show_curve_controls:
            curve_choices = ttk.Frame(chart_box, style="Surface.TFrame")
            curve_choices.pack(fill="x", padx=2, pady=(0, 2))
            for column in range(2):
                curve_choices.columnconfigure(column, weight=1, uniform="curve_choices")
            self._comparison_curve_choices = curve_choices

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._comparison_chart_figure = Figure(
            figsize=(7.2, 2.35), dpi=self._CHART_DPI,
            facecolor=PALETTE["surface"], constrained_layout=True,
        )
        self._comparison_chart_ax = self._comparison_chart_figure.add_subplot(111)
        self._comparison_chart_canvas = FigureCanvasTkAgg(
            self._comparison_chart_figure, master=chart_box)
        self._comparison_chart_canvas.get_tk_widget().pack(
            fill="both", expand=True)

        lower = ttk.Frame(parent, style="Surface.TFrame")
        lower.grid(row=1, column=0, sticky="nsew", pady=(3, 0))
        lower.columnconfigure(0, weight=1)
        lower.rowconfigure(0, weight=1)

        result_box = ttk.LabelFrame(
            lower, text=f" {ranking_title} ", padding=12)
        result_box.grid(row=0, column=0, sticky="nsew")
        # 批量验证一次能送十条进来，而窗口高度未必够。给排名表配滚动条，
        # 高度只作初始请求值——没有它的时候，超出可见区的行既看不到也滚不动。
        tree_frame = ttk.Frame(result_box, style="Surface.TFrame")
        tree_frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(
            tree_frame,
            columns=[key for key, _text, _width in
                     BacktestApp._COMPARISON_RANKING_COLUMNS],
            show="headings", height=3, selectmode="browse",
        )
        for key, text, width in BacktestApp._COMPARISON_RANKING_COLUMNS:
            # 表头与本列数据同向对齐。表头一律居中而数字右对齐时，一列里没
            # 有任何一条共同的竖边，眼睛就找不到列的界限。
            anchor = BacktestApp._comparison_column_anchor(key)
            if key in BacktestApp._COMPARISON_UNSORTABLE_COLUMNS:
                # `#` 是当前显示顺序的行号、不是数据字段，_comparison_sorted_rows
                # 对它直接原样返回。给它绑 command 再打上 ⇅，等于告诉用户
                # "这列可以排"，点下去却毫无反应。
                tree.heading(key, text=text, anchor=anchor)
            else:
                tree.heading(
                    key, text=text, anchor=anchor,
                    command=lambda k=key: self._sort_comparison_by(k))
            # 所有列一起 stretch：只让某一列可拉伸时它会独吞全部剩余空间
            # ——本页的容器约 986px 而列宽总和 642px，那 329px 此前全灌给
            # 了结果名列，把它撑到内容宽度的三倍多，数字列却全挤在左边。
            tree.column(
                key, width=width, minwidth=max(40, width - 30),
                anchor=anchor, stretch=True,
            )
        ranking_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ranking_scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        ranking_scroll.pack(side="right", fill="y")
        # 行底色只由奇偶决定，与优选页同一条规则。极值不进表格：一行可能
        # 在多列最优、多行也可能各占一项，而 Treeview 的 tag 是行级的，
        # 底色说不出"哪一列最优"，只会把整张表染花。这件事交给结论卡——
        # 徽章列出是哪几项，对应的数字瓦片染绿。
        tree.tag_configure("even", background=PALETTE["surface_alt"])
        self._comparison_tree = tree
        self._comparison_rows = {}
        tree.bind("<<TreeviewSelect>>", self._update_comparison_selection)

    def _populate_comparison_view(self, summary, daily_curves):
        """把本次数据灌进已建好的骨架：曲线、配色、排名行、详情。"""
        self._comparison_summary = summary
        self._comparison_daily_curves = dict(daily_curves or {})
        # 曲线配色走与策略优选页共用的登记表，使同一策略跨页同色；summary 未
        # 携带策略身份键时退回按曲线名登记，只保证本页内部不撞色。
        style_keys = {}
        if summary is not None and not getattr(summary, "empty", True):
            for _index, row in summary.iterrows():
                item = row.to_dict()
                name = str(item.get("strategy", ""))
                if name and name not in style_keys:
                    style_keys[name] = str(
                        item.get("meta_style_key") or f"result_name:{name}")
        self._comparison_color_map, self._comparison_dash_map = (
            BacktestApp._comparison_curve_styles(self, style_keys))

        tree = self._comparison_tree
        rows = {}
        # 重填后把用户原来盯着的那一行选回来。行号会随排序和勾选变化，只有
        # 快照 ID 是稳的；不记住它的话，换个列排序选中行就跳到第一行去了。
        previous_id = ""
        if tree.selection():
            previous_id = str(getattr(self, "_comparison_rows", {}).get(
                tree.selection()[0], {}).get("meta_result_id", "") or "")
        ordered = BacktestApp._comparison_sorted_rows(
            summary, getattr(self, "_comparison_sort_column", None),
            getattr(self, "_comparison_sort_descending", False))
        # 曲线闸门要在排序之后定：图上画的是"按当前排序靠前的那几条"。
        self._rebuild_comparison_curve_toggles(
            BacktestApp._comparison_chartable_names(ordered))
        # 清表会让选中行消失并发出 TreeviewSelect；重填期间抑制回调，填完再
        # 主动刷新一次，避免拿着半张表去画图。
        self._comparison_populating = True
        try:
            tree.delete(*tree.get_children())
            for row_no, item in enumerate(ordered):
                iid = f"strategy_{row_no}"
                values = BacktestApp._pad_comparison_row((
                    row_no + 1,
                    item.get("strategy", ""),
                    self._format_comparison_value(item.get("total_net_pnl"), 2),
                    self._format_comparison_value(item.get("total_tc"), 2),
                    self._format_comparison_value(item.get("max_drawdown"), 2),
                    (f"{self._comparison_safe_int(item.get('rehedge_count'), 0)}/"
                     f"{self._comparison_safe_int(item.get('actual_trade_count'), 0)}"),
                    self._format_comparison_value(item.get("turnover"), 2),
                    self._comparison_safe_int(item.get("n_trade_days"), 0),
                ))
                tree.insert(
                    "", "end", iid=iid, values=values,
                    tags=("even",) if row_no % 2 == 0 else ())
                rows[iid] = item
            self._comparison_rows = rows
            tree.configure(height=max(3, min(8, len(rows))))
            BacktestApp._refresh_comparison_sort_markers(self)
            children = tree.get_children()
            if children:
                kept_iid = next((
                    iid for iid, item in rows.items()
                    if previous_id
                    and str(item.get("meta_result_id", "")) == previous_id
                ), None)
                initial_iid = kept_iid or children[0]
                tree.selection_set(initial_iid)
                tree.focus(initial_iid)
        finally:
            self._comparison_populating = False
        # 说明卡的 #序号必须与指标表首列对得上，所以按**显示顺序**取快照：
        # 用户点列头换排序后，两边的编号要一起变。
        BacktestApp._refresh_comparison_variable_card(self, [
            self._saved_backtests[result_id]
            for result_id in (
                str(item.get("meta_result_id", "") or "") for item in ordered)
            if result_id in self._saved_backtests
        ])
        self._update_comparison_selection()

    @staticmethod
    def _shift_hex(color, step):
        """把颜色挪到 ``STRATEGY_CHART_SHADES`` 的第 ``step`` 档明度。

        0 档原样返回：同一策略的第一条曲线要与策略优选页严格同色，跨页对照
        靠的就是这一点。

        既提亮也压暗，不是一味往浅里推：单向变浅到第三档就接近白色，在浅色
        画布上直接看不见了；一深一浅在同一色相里既拉得开距离，又不会跑出
        "这是蓝色系那一组"的识别。
        """
        ratio = STRATEGY_CHART_SHADES[step % len(STRATEGY_CHART_SHADES)]
        if not ratio:
            return color
        try:
            channels = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
        except (TypeError, ValueError, IndexError):
            # 不是 #RRGGBB（例如 matplotlib 的 "tab:blue"）：挪不动就原样退回，
            # 曲线宁可同色也不能画不出来。
            return color
        if ratio > 0:
            return "#" + "".join(
                f"{int(round(value + (255 - value) * ratio)):02X}"
                for value in channels)
        return "#" + "".join(
            f"{int(round(value * (1 + ratio))):02X}" for value in channels)

    def _comparison_curve_styles(self, style_keys):
        """定出每条曲线的颜色与线型，返回两张映射表。

        色相按策略身份取——同一策略在优选页与本页同色是有意设计。但"同一策略
        换区间、换行情、换方向各跑一次"正是本页的头号用途，同组因此常有好几
        条，光靠色相是分不开的。

        所以组内**明度与线型逐条同时变**。此前明度是每四条才升一档
        （``index // 4``），于是前四条严格同色、区分全压在线型上——而虚线与点
        划线在 691×226 px 的画布上、细线宽的密集 PnL 曲线里基本读不出来，实测
        八条同策略只得到两种颜色。改成逐条变之后是八种。

        3 档明度与 4 种线型互质，组合到第 12 条才重复，图上最多 8 条，够用；
        相邻两条则必定明度与线型同时不同。
        """
        colors, dashes = {}, {}
        seen = {}
        for name in self._comparison_daily_curves:
            key = style_keys.get(name, f"result_name:{name}")
            base, _marker = self._strategy_style(key)
            index = seen.get(key, 0)
            seen[key] = index + 1
            dashes[name] = STRATEGY_CHART_DASHES[
                index % len(STRATEGY_CHART_DASHES)]
            colors[name] = BacktestApp._shift_hex(base, index)
        return colors, dashes

    @staticmethod
    def _comparison_chartable_names(ordered):
        """图上该画哪几条：按指标表当前排序取前 N 条的结果名。

        跟着排序走而不是按保留顺序截断，是为了让「全选 → 点期末净损益
        列头 → 看图」这条路成立：换一列排序，图上就换成那一列的前几名。
        """
        names = []
        for item in ordered:
            name = str(item.get("strategy", "") or "")
            if name and name not in names:
                names.append(name)
            if len(names) >= MAX_COMPARISON_CHART_CURVES:
                break
        return names

    def _rebuild_comparison_curve_toggles(self, visible=None):
        """重建曲线勾选框；本页不显示它们时只登记开关状态。

        ``visible`` 是闸门放行的那几条，其余曲线登记为关——指标表与导出照
        样覆盖全部勾选结果，被挡住的只是图上那几根线。
        """
        allowed = None if visible is None else set(visible)
        self._comparison_curve_vars = {
            name: tk.BooleanVar(value=allowed is None or name in allowed)
            for name in self._comparison_daily_curves
        }
        container = getattr(self, "_comparison_curve_choices", None)
        # 挡下了几条一定要说出来——不说的话，"我勾了 11 条、图上只有 8 条"
        # 看起来就是数据丢了。有勾选框时接在「显示曲线:」后面，没有时那行标
        # 签本来就是空的，整句写进去。
        hidden = len(self._comparison_daily_curves) - len(
            allowed if allowed is not None else self._comparison_daily_curves)
        hint_var = getattr(self, "_comparison_curve_hint_var", None)
        if hint_var is not None:
            note = (
                f"图上只画按当前排序靠前的 {MAX_COMPARISON_CHART_CURVES} 条；"
                f"其余 {hidden} 条仍在下方指标表与导出里" if hidden > 0 else "")
            head = "显示曲线:" if container is not None else ""
            hint_var.set("  ".join(part for part in (head, note) if part))
        if container is None:
            return
        for widget in container.winfo_children():
            widget.destroy()
        for index, name in enumerate(self._comparison_daily_curves):
            color = self._comparison_color_map[name]
            tk.Checkbutton(
                container, text=name,
                variable=self._comparison_curve_vars[name],
                command=self._draw_comparison_cumulative_chart,
                bg=PALETTE["surface"], fg=color,
                activebackground=PALETTE["surface"], activeforeground=color,
                selectcolor=PALETTE["surface"],
                font=(_UI_FONT_FAMILY, 8), padx=2, anchor="w",
            ).grid(
                row=index // 2, column=index % 2,
                sticky="w", padx=(0, 8), pady=(0, 1),
            )

    def _draw_comparison_cumulative_chart(self):
        ax = getattr(self, "_comparison_chart_ax", None)
        canvas = getattr(self, "_comparison_chart_canvas", None)
        if ax is None or canvas is None:
            return
        ax.clear()
        tree = getattr(self, "_comparison_tree", None)
        focused_name = None
        if tree is not None and tree.selection():
            row = self._comparison_rows.get(tree.selection()[0], {})
            focused_name = row.get("strategy")
        plotted = 0
        for name, daily in self._comparison_daily_curves.items():
            variable = self._comparison_curve_vars.get(name)
            if variable is not None and not variable.get():
                continue
            y = np.asarray(daily.get("cumulative_net_pnl", []), dtype=float)
            if not len(y):
                continue
            x = np.arange(1, len(y) + 1)
            focused = name == focused_name
            ax.plot(
                x, y, label=name, color=self._comparison_color_map[name],
                linestyle=getattr(
                    self, "_comparison_dash_map", {}).get(name, "solid"),
                linewidth=2.8 if focused else 1.7,
                alpha=1.0 if focused or focused_name is None else 0.58,
                marker="o", markersize=3.5 if focused else 2.5,
            )
            plotted += 1

        ax.axhline(0.0, color=PALETTE["text_muted"], linewidth=0.8, alpha=0.7)
        ax.set_xlabel("交易日", fontsize=8)
        ax.set_ylabel("累计净损益", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.65)
        if plotted:
            ax.legend(loc="best", frameon=False, fontsize=8, ncol=min(3, plotted))
            ax.margins(x=0.02, y=0.14)
        else:
            ax.text(
                0.5, 0.5, "请至少勾选一个策略",
                ha="center", va="center", transform=ax.transAxes,
                color=PALETTE["text_muted"], fontsize=10,
            )
        canvas.draw_idle()

    def _update_comparison_selection(self, _event=None):
        """选中一行只做一件事：在图上把它的曲线突出。

        它的身份与数字，指标表和结果池表里都有，不必再另起一处复述。
        """
        # 重填排名表期间 Tk 会发选择事件，此时表只有半张，画出来的图是过渡
        # 态；填完由 _populate_comparison_view 主动刷新一次。
        if getattr(self, "_comparison_populating", False):
            return
        self._draw_comparison_cumulative_chart()

    # 排名表的指标列。表头文案写死在这里，改口径时只有这一处要动。
    # 两个排名口径并排显示：一列驱动本次排序，另一列供对照——两者给出
    # 相反顺序的地方，正是需要人来判断的地方。RMS 得分与胜率是旧口径的
    # 诊断值，已移入选中行的详情串，不再占表宽。
    # 一套完整的决策依据：多赚多少、信噪比如何、为此多花多少成本、
    # 最坏一段亏多少。只给前两项时右侧会空出大片版面，而后两项恰恰是
    # 判断“这份增量划不划算”缺不了的另一半。
    _HISTORY_METRIC_COLUMNS = (
        ("incremental_pnl", "增量收益", 130),
        ("incremental_sharpe", "增量信噪比", 140),
        ("incremental_tc", "增量成本", 130),
        ("max_drawdown", "最大回撤", 130),
    )

    # 各列的对齐方向。未列出的都是数字列，一律右对齐。
    # 状态列曾经是左对齐，紧跟在右对齐的最大回撤后面——右对齐文本贴着本列
    # 右边界、左对齐文本贴着下一列左边界，中间一个像素都不剩，于是
    # 「0.3948」和「✓」直接粘成一团。改成居中后两侧都留出空隙。
    _HISTORY_COLUMN_ANCHORS = {
        "check": "center", "rank": "center", "period": "center",
        "strategy": "w", "status": "center",
    }
    _HISTORY_NUMERIC_ANCHOR = "e"

    @staticmethod
    def _history_column_anchor(key):
        return BacktestApp._HISTORY_COLUMN_ANCHORS.get(
            key, BacktestApp._HISTORY_NUMERIC_ANCHOR)

    @staticmethod
    def _pad_history_row(values, keys):
        """按对齐方向给单元格补一个空格的内边距，返回可直接写入的值。

        ttk.Treeview 没有单元格内边距这回事：文本严格贴着 anchor 那一侧的
        列边界。列宽又随窗口伸缩，靠把某一列调宽治不了根——窗口一窄，右对
        齐的数字仍然会顶到边界上，和邻列内容挤在一起。这里在**显示文本**
        两端补空格，等价于给每格加内边距；表格值不参与任何解析（行数据另
        存在 ``_history_rank_rows``），补空格不影响逻辑。
        """
        padded = []
        for key, value in zip(keys, values):
            text = str(value)
            anchor = BacktestApp._history_column_anchor(key)
            if not text:
                padded.append(text)
            elif anchor == "e":
                padded.append(f"{text} ")
            elif anchor == "w":
                padded.append(f" {text}")
            else:
                padded.append(text)
        return tuple(padded)

    @staticmethod
    def _format_history_status(status, paired, total):
        """把“可比段数 + 数据完整度”压成一列（表头「样本完整度」）。

        正常情况下这两项分别是 ``n/n`` 和“数据完整”，逐行重复且不传递
        任何信息；只有出现失败段或数据不足时才值得占版面。
        """
        text = str(status or "").strip()
        healthy = text in ("数据完整", "可比", "旧版数据完整")
        try:
            paired_int, total_int = int(paired), int(total)
        except (TypeError, ValueError):
            paired_int = total_int = 0
        complete = total_int > 0 and paired_int == total_int
        if healthy and complete:
            return "✓"
        if healthy and not complete:
            return f"{paired_int}/{total_int} 段"
        if complete:
            return text or "—"
        return f"{text}·{paired_int}/{total_int}" if text else f"{paired_int}/{total_int}"

    @staticmethod
    def _format_drawdown_value(value, digits=4):
        """最大回撤是绝对量且越小越好，不带正负号。"""
        number = BacktestApp._comparison_finite(value)
        if number is None:
            return "—"
        return f"{abs(number):.{digits}f}"

    @staticmethod
    def _format_objective_value(value, digits=4):
        """格式化增量指标；缺失时显示占位符。

        零就写成零。此前零一律显示为「基准」，有两个问题：基线的身份已经
        写在策略列的「（基准）」里，三个指标格再各写一遍是同一行内的重复；
        更要紧的是候选也可能拿到零增量（带宽宽到全程没触发，与基线完全一
        致），那时「基准」会把它伪装成基线行。零不带正负号——加号会暗示它
        是个正的增量。

        但零仍要占住符号位：这一列是右对齐的，``0.0000`` 比 ``-0.0211``
        短一个字符，小数点就会错开一格，基准行看着像换了一种格式。补一个
        空格既不显形，又让整列的小数点对齐。
        """
        number = BacktestApp._comparison_finite(value)
        if number is None:
            return "—"
        if np.isclose(number, 0.0, atol=1e-12):
            return f" {0.0:.{digits}f}"
        return f"{number:+.{digits}f}"

    @staticmethod
    def _build_history_metric_tree(parent, *, lead_columns, status_heading,
                                   status_width, height,
                                   on_objective_click=None,
                                   active_objective=None,
                                   objectives_available=True):
        """构建周期排名表格。

        ``on_objective_click`` 提供时，两个排名口径列的表头可点击切换排名
        依据；其余列的表头保持不可点，避免暗示它们也能当排名依据。

        ``objectives_available=False`` 表示本次结果里根本没有增量指标
        （品种池模式跨合约无法直接相加金额，只产出逐段有界改善）。此时两
        列表头必须一并去掉 ⇅ 和点击回调：留着的话点下去是**静默 no-op**
        ——``_rank_history_rows`` 两个口径都回退到同一个 RMS 改善，标记移
        动了、顺序一动不动。
        """
        if on_objective_click is None or not objectives_available:
            def on_objective_click(_key):
                return None
        specs = (
            tuple(lead_columns)
            + BacktestApp._HISTORY_METRIC_COLUMNS
            + (("status", status_heading, status_width),)
        )
        tree = ttk.Treeview(
            parent, columns=tuple(key for key, _text, _width in specs),
            show="tree headings", height=height, selectmode="browse",
        )
        tree.heading("#0", text="勾选")
        tree.column("#0", width=46, stretch=False, anchor="center")
        for key, heading, width in specs:
            text = heading
            # 表头与本列数据同向对齐。此前表头一律居中而数字右对齐，一列
            # 里没有任何一条共同的竖边，眼睛就找不到列的界限——这比缺少分
            # 割线更要命。同向之后每列的右边界（或左边界）自己连成一条线。
            anchor = BacktestApp._history_column_anchor(key)
            if key in _OBJECTIVE_COLUMN_KEYS and objectives_available:
                # 只有这两列是合法的排名依据，点表头即按它重排；其余列不
                # 可点——七列里给五列装上排序会让人以为推荐也跟着变，而
                # 后端并没有对应的排名口径。
                # 当前排序列用实心 ↓，另一列用 ⇅ 表示「可点、但不是现在
                # 的排序依据」，省掉上方那句重复的文字说明。
                marker = " ↓" if key == active_objective else " ⇅"
                tree.heading(
                    key, text=f"{text}{marker}", anchor=anchor,
                    command=lambda k=key: on_objective_click(k))
            else:
                tree.heading(key, text=text, anchor=anchor)
            # 所有列一起 stretch：只让某一列可拉伸时它会独吞全部剩余空间
            # （实测策略名列被撑到 656px，内容只占 120px）。各列的初始宽度
            # 已按表头与内容实际需要分配，总和接近容器宽度，因此窗口缩放
            # 时平均摊到每列的增量很小，比例不会走样。
            tree.column(
                key, width=width, minwidth=max(40, width - 30),
                anchor=BacktestApp._history_column_anchor(key),
                stretch=True,
            )
        return tree

    def _build_history_comparison_view(
            self, parent, recommendations, ranking, window_results=None,
            history_lookbacks=None, details=None):
        """按“区间说明 / 周期结论 / 周期排名与动作 / 图表”四段装配结果页。"""
        self._history_ranking = ranking
        self._history_lookbacks = BacktestApp._normalize_history_lookbacks(
            history_lookbacks)
        # 原始逐回测分段结果含完整 bar 级价格、Greeks 与 PnL 数组；历史分钟数据下
        # 深拷贝并长期保留会占用大量内存。图表只保存压缩后的独立日级摘要，
        # worker 投递的原始结果在本次渲染结束后即可释放。
        if details is not None:
            # 重排场景：分段明细与排名无关，直接沿用已压缩的结果，避免
            # 长期持有 bar 级原始数组（一年 1 分钟下有数百 MB）。
            self._history_window_summary, self._history_replay_index = details
        else:
            self._history_window_summary = history_window_summary(
                window_results or {})
        # 重放配方只保存构造回测所需的行情切片、期权与策略对象，不驻留每个
        # 候选每段的完整结果数组。逐段明细正常从 history_bar_cache 读（3 ms），
        # 配方是缓存不在时的兜底，据此重算任一分段（620 ms）。
            self._history_replay_index = history_replay_index(
                window_results or {})
        self._history_chart_selected_by_period = {}
        # 配对缓存只对本次 summary 有效，必须随结果页一起重建。
        self._history_pairs_cache = {}

        objectives = ranking.get("selection_objective")
        self._history_result_objective = (
            str(objectives.dropna().iloc[0])
            if objectives is not None and not objectives.dropna().empty
            else None)
        flags = history_selection.ranking_flags(ranking)
        self._history_uses_strict_metric = flags["uses_strict_metric"]
        self._history_uses_window_equal_metric = flags[
            "uses_window_equal_metric"]
        # 按列是否真的存在判断，而不是按模式判断：旧快照同样可能没有增量
        # 列，它和品种池结果在展示上应该走同一条降级路径。
        self._history_objectives_available = any(
            column in ranking.columns
            for column in _OBJECTIVE_RANKING_COLUMNS)
        self._assign_history_chart_styles(flags["candidate_names"], ranking)
        selected_periods = [
            (key, label) for key, label in HISTORY_PERIOD_DEFS
            if key in self._history_lookbacks
        ]

        parent.columnconfigure(0, weight=1)
        # 顶部三块（口径提示 / 周期条 / 结论卡）按自然高度固定，剩余高度全
        # 部交给可拖分割区。
        parent.rowconfigure(3, weight=1)
        # 口径说明是一次性知识，长期占据结果区顶部只会挤压表格与图表。
        # 常驻一行主指标定义，完整口径折进按钮后面按需查看。
        BacktestApp._build_history_scope_note(
            self, parent, uses_strict_metric=flags["uses_strict_metric"])

        self._build_history_period_box(
            parent, recommendations, ranking)

        # 排名表与图表改由一条可拖的分割线分配高度，取代此前“外层滚动画布
        # 里再套一张自带滚动条的表格和一块图表”的三层嵌套：那样滚轮要靠
        # bind_all 在三者之间抢来抢去，而两块内容各自的合适高度只有用户知
        # 道。opaqueresize=False 让拖动时只画一条线，松手才重排——matplotlib
        # 逐帧重绘跟不上拖动。
        splitter = tk.PanedWindow(
            parent, orient="vertical", bg=PALETTE["border_soft"],
            sashwidth=6, sashrelief="flat", borderwidth=0,
            opaqueresize=False)
        splitter.grid(row=3, column=0, sticky="nsew", padx=2, pady=(4, 2))
        self._history_splitter = splitter

        detail_box = self._build_history_detail_box(splitter)
        splitter.add(detail_box, minsize=150, stretch="always")
        # 结论卡占 row=2（表格之上），但必须在排名表之后构建：它内嵌的动作
        # 条要读排名表的勾选状态来决定文案。grid 的位置与构建顺序无关。
        self._build_history_conclusion_card(parent)
        chart_box = self._build_history_chart_box(splitter)
        # 图表那一格的下限要扣掉指标条、下钻条与边框（合计约 120px）才是真
        # 正的绘图高度：给 280 才剩得下约 160px，坐标轴与图例不至于连成一
        # 团。把表格拖到最高时图表仍可读，正是此前给整页套滚动条要解决的
        # 那个问题。
        splitter.add(chart_box, minsize=280, stretch="always")
        self._init_history_splitter_ratio(splitter)

        if getattr(self, "_history_period_rows", None):
            self._update_history_selection()

    def _assign_history_chart_styles(self, candidate_names, ranking=None):
        """为本次候选固定颜色与标记，使排名表与图表配色始终一致。

        配色来自 _strategy_style 这一份会话级登记表，因此同一策略在结果对比
        页拿到的颜色与这里相同。ranking 提供策略身份元数据；缺失时退回按候选
        名登记，只保证本页内部一致。
        """
        identities = {}
        if ranking is not None and not getattr(ranking, "empty", True):
            for _index, row in ranking.iterrows():
                item = row.to_dict()
                name = str(item.get("strategy", "")).strip()
                if name and name not in identities:
                    identities[name] = BacktestApp._history_row_style_key(item)
        color_map = {}
        marker_map = {}
        for strategy in candidate_names:
            style_key = identities.get(
                strategy, f"history_name:{strategy}")
            color, marker = self._strategy_style(style_key)
            color_map[strategy] = color
            marker_map[strategy] = marker
        self._history_chart_color_map = color_map
        self._history_chart_marker_map = marker_map

    def _build_history_scope_note(self, parent, *, uses_strict_metric):
        """初始化当前排名口径。

        排名依据不再单独占一行文字：顶部摘要标签里已有一枚 pill，表格里
        当前排序的那一列也会用 ↓ 标出来，再写一句说明就是第三次重复。
        """
        objective = getattr(self, "_history_result_objective", None)
        self._history_result_objective_var = tk.StringVar(
            value=objective or DEFAULT_SELECTION_OBJECTIVE)
        if not uses_strict_metric:
            # 旧口径结果仍需明确提示，否则会被当成新版排名解读。
            bar = ttk.Frame(parent, style="Surface.TFrame")
            bar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
            note_card = tk.Frame(bar, bg=PALETTE["warning_light"], bd=0)
            note_card.pack(fill="x")
            tk.Label(
                note_card, text="⚠ 旧版取样方式的结果，建议重新运行",
                bg=PALETTE["warning_light"], fg=PALETTE["warning"],
                font=(_UI_FONT_FAMILY, 9), anchor="w", padx=12, pady=6,
            ).pack(side="left", fill="x", expand=True)

    def _set_history_objective(self, objective):
        """点带 ⇅ 的列头切换排名依据。"""
        if objective not in SELECTION_OBJECTIVES:
            return
        variable = getattr(self, "_history_result_objective_var", None)
        if variable is None:
            return
        variable.set(objective)
        self._rerank_history_results()

    def _rerank_history_results(self):
        """切换排名依据时就地重排，不重跑回测。"""
        ranking = getattr(self, "_history_ranking", None)
        if ranking is None or getattr(ranking, "empty", True):
            return
        objective = self._history_result_objective_var.get().strip()
        if objective not in SELECTION_OBJECTIVES:
            return
        if objective == getattr(self, "_history_result_objective", None):
            return
        details = (
            getattr(self, "_history_window_summary", None),
            getattr(self, "_history_replay_index", None),
        )
        period_var = getattr(self, "_history_selected_period_var", None)
        kept_period = period_var.get() if period_var is not None else None
        kept_checked = dict(
            getattr(self, "_history_chart_selected_by_period", {}) or {})
        try:
            recommendations, reranked = rerank_history(ranking, objective)
        except (ValueError, KeyError) as exc:
            messagebox.showerror("重排失败", str(exc))
            return
        self._history_rerender(recommendations, reranked, details)
        if kept_checked:
            self._history_chart_selected_by_period = kept_checked
        period_var = getattr(self, "_history_selected_period_var", None)
        if (period_var is not None and kept_period
                and kept_period in getattr(self, "_history_period_rows", {})):
            period_var.set(kept_period)
            BacktestApp._refresh_history_period_chips(self)
        self._update_history_selection()

    def _history_rerender(self, recommendations, ranking, details):
        """用新的排名结果重建结果页，沿用本次已有的分段明细。

        切排名口径走的也是这条路。载入标记必须原样带过去——它描述的是「这
        份结果哪来的」，换个看法不会把载入的结果变成刚跑的。
        """
        if getattr(self, "_history_results_container", None) is None:
            return
        BacktestApp._show_history_recommendation(
            self, recommendations, ranking,
            notes=getattr(self, "_history_last_notes", None),
            source_label=getattr(self, "_history_last_source_label", None),
            history_state=getattr(self, "_history_last_state", None),
            details=details,
            loaded_meta=getattr(self, "_history_loaded_meta", None),
        )

    @staticmethod
    def _history_consensus_note(items):
        """跨周期一致性提示，返回 ``(文案, 状态)``。"""
        picks = [
            (str(item.get("period", "—")), str(item.get("strategy", "")))
            for item in items
            if item.get("strategy") and str(item.get("strategy")) != "—"
        ]
        if not picks:
            return "各周期都没有形成可比结论", "empty"
        names = {name for _period, name in picks}
        if len(names) == 1:
            return f"{len(picks)} 个周期一致推荐 {picks[0][1]}", "agree"
        detail = "、".join(f"{period} {name}" for period, name in picks)
        return (f"各周期结论不一致（{detail}）；"
                "短周期样本少更易受噪声影响，建议以长周期为准"), "disagree"

    @staticmethod
    def _default_history_period(period_rows):
        """默认展示哪个周期：有可比结论的周期里样本最多的那个。

        此前取 ``next(iter(...))``，也就是 ``HISTORY_PERIOD_DEFS`` 的头一个
        「近周」——五档里样本最少、最容易被噪声主导的那个。而同一行右端的
        一致性提示写的是「短周期样本少更易受噪声影响，建议以长周期为准」：
        页面一边这么建议，一边默认把最不该单独采信的结论摆在读者面前。

        按可比样本段数排而不是按周期长短：某个长周期可能因历史不足而没有
        可比结论，那时它反而不该被选中。全都没有结论就退回第一个，保持
        「总有一个 chip 是选中的」。
        """
        rows = dict(period_rows or {})
        if not rows:
            return ""

        def sample_size(item):
            item = item or {}
            if not str(item.get("strategy", "") or "").strip("—").strip():
                return -1          # 没有可比结论的周期排到最后
            return BacktestApp._comparison_safe_int(item.get("paired", 0), 0)

        best = max(rows, key=lambda key: sample_size(rows[key]))
        return best if sample_size(rows[best]) >= 0 else next(iter(rows))

    def _build_history_period_box(
            self, parent, recommendations, ranking):
        """渲染周期切换条与一枚跨周期一致性 pill。

        一致性此前是一条通栏横幅，和顶部摘要里的 pill 说的是同一件事，两
        者上下相邻地重复占了两行。这里压成周期条右端的一枚 pill，完整措辞
        （含“短周期样本少”的提醒）移进悬浮提示。
        """
        box = ttk.Frame(parent, style="Surface.TFrame")
        box.grid(row=1, column=0, sticky="ew", padx=2, pady=(3, 0))

        items = list(self._comparison_recommendation_rows(
            recommendations, ranking, self._history_lookbacks))
        note, state = BacktestApp._history_consensus_note(items)

        period_bar = ttk.Frame(box, style="Surface.TFrame")
        period_bar.pack(fill="x", pady=(2, 4))
        # 配置区那个「分析周期」是要跑哪些，这里是在看哪一个；同名不同义，
        # 结果区改用「查看周期」。
        ttk.Label(
            period_bar, text="查看周期:", style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 9, "bold"),
        ).pack(side="left", padx=(0, 10))
        self._history_selected_period_var = tk.StringVar()
        self._history_period_rows = {}
        self._history_period_chips = {}
        for item in items:
            key = str(item.get("lookback", ""))
            if not key:
                continue
            self._history_period_rows[key] = item
            chip = tk.Label(
                period_bar, text=str(item.get("period", key)),
                font=(_UI_FONT_FAMILY, 10), padx=16, pady=5,
                cursor="hand2", bd=0,
            )
            chip.pack(side="left", padx=(0, 6))
            chip.bind(
                "<Button-1>",
                lambda _event, k=key: BacktestApp._select_history_period(
                    self, k))
            best = str(item.get("strategy", "") or "").strip()
            BacktestApp._attach_tooltip(
                chip,
                f"{item.get('period', key)} 最优：{best}" if best and best != "—"
                else f"{item.get('period', key)}：无可比结论")
            self._history_period_chips[key] = chip
        if self._history_period_rows:
            self._history_selected_period_var.set(
                BacktestApp._default_history_period(self._history_period_rows))
            BacktestApp._refresh_history_period_chips(self)

        pill_bg = {
            "agree": PALETTE["success_light"],
            "disagree": PALETTE["warning_light"],
        }.get(state, PALETTE["surface_alt"])
        pill_fg = {
            "agree": PALETTE["success"],
            "disagree": PALETTE["warning"],
        }.get(state, PALETTE["text_muted"])
        # 这是全页唯一一枚一致性 pill，因此把「几种结论」也写进来——顶部
        # 那枚重复的已经移除，它携带的计数不能跟着丢。
        #
        # 计数只能数**有结论的**周期，不能用 len(items)：items 恒等于已选周
        # 期数（recommendation_rows 对无 leader 的周期照样 append 一行
        # strategy="—"），而一致性判定只统计有结论的那些。五选全勾、只有近月
        # 与近季形成结论时，pill 会写「5 个周期结论一致」，而同一份数据算出
        # 的悬停提示写的是「2 个周期一致推荐 …」。pill 常驻可见、提示要悬停，
        # 用户会把 2 个周期的证据当成 5 个周期的交叉验证——正是本页在
        # disagree 文案里特意强调要避免的那种误读。
        concluded = [
            str(item.get("strategy", "")) for item in items
            if item.get("strategy") and str(item.get("strategy")) != "—"
        ]
        distinct = len(set(concluded))
        pill_text = {
            "agree": f"✓ {len(concluded)} 个周期结论一致",
            "disagree": f"⚠ 各周期结论不一致（{distinct} 种）",
        }.get(state, "各周期均无可比结论")
        consensus_pill = tk.Label(
            period_bar, text=pill_text, bg=pill_bg, fg=pill_fg,
            font=(_UI_FONT_FAMILY, 9, "bold"), padx=10, pady=4,
        )
        consensus_pill.pack(side="right")
        BacktestApp._attach_tooltip(consensus_pill, note)

        # 周期本身的取样口径（合约期限、回放天数、分段构成…）另起一行：
        # 它常长到七八项，挤在周期按钮同一行会把 pill 顶出可视区。
        self._history_period_context_var = tk.StringVar(value="")
        period_context = ttk.Label(
            box, textvariable=self._history_period_context_var,
            style="SurfaceMuted.TLabel", justify="left")
        period_context.pack(fill="x", pady=(0, 2))
        BacktestApp._track_wraplength(period_context)

    def _build_history_detail_box(self, parent):
        """渲染选中周期的候选排名与图表勾选栏，返回待挂载的区块。

        动作按钮与结论文字已上移到结论卡：它们描述的是“选中的那条策略”，
        放在八行表格下方时，人要先滚过表格才看得到自己刚选出来的结论。

        自己不做布局：它是分割区的一格，由调用方 ``add`` 进去。
        """
        detail_box = ttk.LabelFrame(
            parent, text=" 该周期内各策略排名 ", padding=(14, 8))
        detail_box.columnconfigure(0, weight=1)
        detail_box.rowconfigure(2, weight=1)
        self._history_detail_var = tk.StringVar(value="请选择周期")

        chart_selection_bar = ttk.Frame(detail_box, style="Surface.TFrame")
        chart_selection_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(
            chart_selection_bar,
            text=f"勾选要对比的策略（最多 {MAX_HISTORY_CHART_CANDIDATES} 个）",
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            chart_selection_bar, text="清空勾选", width=8,
            command=self._clear_history_chart_candidates,
        ).pack(side="right", padx=(8, 0))

        rank_tree = BacktestApp._build_history_metric_tree(
            detail_box,
            lead_columns=(
                ("rank", "#", 38),
                ("strategy", "策略 / 参数", 232),
            ),
            status_heading="样本完整度", status_width=129, height=8,
            on_objective_click=self._set_history_objective,
            active_objective=getattr(self, "_history_result_objective", None),
            objectives_available=getattr(
                self, "_history_objectives_available", True),
        )
        rank_tree.grid(row=2, column=0, sticky="nsew")
        rank_scrollbar = ttk.Scrollbar(
            detail_box, orient="vertical", command=rank_tree.yview)
        rank_scrollbar.grid(row=2, column=1, sticky="ns")
        rank_tree.configure(yscrollcommand=rank_scrollbar.set)
        # 行底色只保留选中态（由 ttk 的 Treeview 样式提供），不再按角色/名
        # 次/数据完整度给行上色。此前那三档底色各有毛病：
        #   · 绿色标"第一行"——而第一行渲染后就被自动选中，选中蓝把它整个
        #     盖住，实测三种排名形态下绿色一次都没显形；
        #   · 判定用 row_no==0 而不是 rank==1，基准占了第 0 行时 elif 再也
        #     不命中，真正最优的候选反而是纯白；
        #   · "可比（仅参考）"这档降级完全没有颜色，与健康行同色，而结论卡
        #     对同一行会显示"数据不足"——表格与卡片自相矛盾。
        # 这些信息本来就各有归属：名次看 # 列的奖牌与排序，最优看结论卡，
        # 数据完整度看「样本完整度」列。底色再说一遍只会互相打架。
        # 基准行只保留等宽加粗：它是身份强调而非底色，且与正文同族同宽，
        # 不会破坏数位对齐（换成比例字体会，见 _pad_history_row 一节）。
        rank_tree.tag_configure(
            "baseline", font=(_MONO_FONT_FAMILY, 9, "bold"))
        rank_tree.bind("<Button-1>", self._toggle_history_chart_click)
        rank_tree.bind("<space>", self._toggle_focused_history_chart_candidate)
        rank_tree.bind(
            "<<TreeviewSelect>>", self._update_history_rank_selection)
        # 右键菜单。三个序列都绑，与结果池同理：macOS 的 Tk 在不同小版本里把
        # 右键映射成 Button-2 或 Button-3，Control+左键又是系统级的右键等价
        # 操作；Windows / Linux 只认 Button-3。
        for sequence in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
            rank_tree.bind(sequence, self._popup_history_rank_menu)
        self._history_rank_menu = tk.Menu(self, tearoff=0)
        self._history_rank_menu.add_command(
            label="参数详情…", command=self._show_history_row_params)
        self._history_rank_tree = rank_tree

        # 选中策略的原始损益波动值：表格七列之外的补充，仍留在表格脚下，
        # 但不再套一层灰卡片——同屏的灰底条已经太多。
        ttk.Label(
            detail_box, textvariable=self._history_detail_var,
            style="SurfaceMuted.TLabel", justify="left", wraplength=1100,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        return detail_box

    # 切到某个周期时默认勾选的排名行数。数的是名次，不是候选个数——基准
    # 也占一个名次却不可勾选，所以它排进前三名时默认只勾上两个候选。
    _HISTORY_DEFAULT_CHART_RANKS = 3

    # 自动摆放分割线的次数上限，兜底任何未预料的抖动。正常一次就位。
    _HISTORY_SASH_PLACEMENT_BUDGET = 6
    # 小于这个相对幅度的高度变化视为自己引起的回波，不是窗口真的变了。
    _HISTORY_SASH_ECHO_TOLERANCE = 0.15

    def _init_history_splitter_ratio(self, splitter, ratio=0.45):
        """按比例摆放排名表与图表的初始分割线，直到用户第一次拖它为止。

        不管的话，初始分配走两格各自的请求高度，而图表请求得更高，排名表
        一上来只剩四五行可见。

        **难点全在"不自激"。** ``sash_place`` 改变窗格尺寸 → 图表那一格里
        的 matplotlib 跟着 resize → LabelFrame 的请求尺寸变化 → 外层 grid
        重排 → splitter 自己的高度被改掉二十来像素 → ``<Configure>`` 又落
        回这里。无条件重摆就是一个闭环：实测单次渲染中 ``sash_place`` 被
        调用 300+ 次，高度在 621↔645 之间反复横跳，界面永远画不完一帧，
        内存被 Agg 缓冲区反复分配撑到 GB 级。

        所以只响应**真实**的尺寸变化：相对幅度小于
        ``_HISTORY_SASH_ECHO_TOLERANCE`` 的高度变化是自己引起的回波，直接
        忽略；用户拉窗口那种大幅变化才重新按比例摆。另有两道兜底——已经在
        目标位就不动（minsize 钳住目标时不会反复试），以及一个硬性次数预
        算。平时也不需要持续跟随：两格都是 ``stretch="always"``，窗口缩放
        时 tk 自己会按比例分配。
        """
        state = {"owned": False, "height": None,
                 "left": BacktestApp._HISTORY_SASH_PLACEMENT_BUDGET}

        def _place(_event=None):
            if state["owned"] or state["left"] <= 0:
                return
            height = splitter.winfo_height()
            if height <= 1:
                return
            previous = state["height"]
            if previous is not None and abs(height - previous) <= (
                    previous * BacktestApp._HISTORY_SASH_ECHO_TOLERANCE):
                return
            state["height"] = height
            target = max(150, int(height * ratio))
            try:
                if abs(splitter.sash_coord(0)[1] - target) <= 2:
                    return
                state["left"] -= 1
                splitter.sash_place(0, 0, target)
            except tk.TclError:
                pass

        def _hand_over(_event=None):
            # 分割线属于 PanedWindow 自身；点在表格或图表里是子控件的事件，
            # 不会落到这里。所以这一下就是“用户抓住了分割线”。
            state["owned"] = True

        splitter.bind("<Configure>", _place, add="+")
        splitter.bind("<ButtonPress-1>", _hand_over, add="+")
        splitter.after_idle(_place)

    def _build_history_conclusion_card(self, parent):
        """结论卡：选中策略是什么、凭什么、以及拿它做什么，集中在一处。

        结论此前散在四个地方——顶部 pill、一致性横幅、排名表首行、表格下
        方的详情条，而真正要用它的两个按钮又在表格另一侧。这里把“选中的
        策略 + 它的关键数字 + 两个动作”并成一张卡放在表格上方；表格默认
        选中 rank 1，所以刚出结果时这张卡显示的就是本周期推荐。
        """
        card = tk.Frame(
            parent, bg=PALETTE["surface_alt"],
            highlightbackground=PALETTE["border_soft"], highlightthickness=1)
        card.grid(row=2, column=0, sticky="ew", padx=2, pady=(6, 2))
        self._history_conclusion_card = card

        # 左侧色条随结论性质变色：领先绿、基准蓝、数据不全黄。
        self._history_conclusion_accent = tk.Frame(
            card, bg=PALETTE["border_soft"], width=4)
        self._history_conclusion_accent.pack(side="left", fill="y")

        body = tk.Frame(card, bg=PALETTE["surface_alt"], padx=14, pady=10)
        body.pack(side="left", fill="both", expand=True)
        body.columnconfigure(0, weight=1)

        self._history_conclusion_badge_var = tk.StringVar(value="")
        self._history_conclusion_badge = tk.Label(
            body, textvariable=self._history_conclusion_badge_var,
            bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9, "bold"), anchor="w")
        self._history_conclusion_badge.grid(row=0, column=0, sticky="w")

        self._history_conclusion_name_var = tk.StringVar(value="请选择策略")
        tk.Label(
            body, textvariable=self._history_conclusion_name_var,
            bg=PALETTE["surface_alt"], fg=PALETTE["text"],
            font=(_UI_FONT_FAMILY, 16, "bold"), anchor="w", justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(1, 6))

        self._history_conclusion_stats = tk.Frame(
            body, bg=PALETTE["surface_alt"])
        self._history_conclusion_stats.grid(row=2, column=0, sticky="w")

        self._build_history_action_bar(body)
        BacktestApp._refresh_history_conclusion_card(self)

    def _build_history_action_bar(self, body):
        """把结论用到当下的动作栏；逐段下钻另在图表下方，两者对象不同。"""
        actions = tk.Frame(body, bg=PALETTE["surface_alt"])
        actions.grid(row=0, column=1, rowspan=3, sticky="ne", padx=(18, 0))
        # 只留一个动作。原先并排的「应用并用当前行情回测」等于本按钮 + 左
        # 侧「运行回测」，两条路做同一件事，先要分辨点哪个本身就是负担。
        #
        # navigate=False 是关键：此前它会跳到回测摘要页，而那页显示的是上
        # 次留下的内容（多半是刚才下钻的那个分段）。刚点完「应用参数」就落
        # 在一页别人的数字上，会被当成本策略的结果读——这正是「感觉它把明
        # 细也加载了」的来源，其实它什么都没加载，只是跳了页。
        # command 必须自己接住异常。tkinter 的默认 report_callback_exception
        # 只把 Traceback 打到 stderr，打包成 .app 之后没人看得见——用户点下
        # 去的观感是「按钮坏了」。而 _apply_history_recommendation 有四条正
        # 常的 raise 路径（S0/σ 非法、缺固定时刻参数、缺 σ 参数、无法识别候
        # 选），全都会走到这里。结果池右键的同名动作一直是包着的，历史页这
        # 处是遗漏。
        ttk.Button(
            actions, text="应用策略", width=16,
            style="Accent.TButton",
            command=lambda: BacktestApp._apply_history_recommendation_safely(
                self),
        ).pack(fill="x")

    def _apply_history_recommendation_safely(self):
        """「应用策略」按钮的入口：失败必须让用户看见。"""
        try:
            return self._apply_history_recommendation(navigate=False)
        except Exception as exc:                       # noqa: BLE001
            messagebox.showerror("无法应用历史策略", str(exc))
            return None

    def _history_conclusion_stat_specs(self, row):
        """结论卡上的数字：优先给增量口径，缺失时退回损益波动原值。

        品种池模式（跨合约金额不可直接相加）下增量列整列为空，此时若照抄
        表格就是四个破折号——那等于把结论卡做成了装饰。
        """
        finite = BacktestApp._comparison_finite
        win_rate = finite(row.get("window_win_rate_vs_c2c"))
        paired = BacktestApp._comparison_safe_int(
            row.get("paired_windows", row.get("rolling_windows")), 0)
        baseline_windows = BacktestApp._comparison_safe_int(
            row.get("baseline_windows", paired), paired)
        # 第三个元素只驱动着色，规则是 >0 绿、<0 红。胜段占比是 [0,1] 的
        # 比例，中性点在 **0.5** 而不是 0——直接把它传下去，5 段里只赢 1 段
        # （20%，明显比基准差）也会被染成"优于基准"的绿色。减掉 0.5 之后
        # 才是"比一半多还是少"，与其余增量列"正=更好"的语义对齐。
        tail = (
            ("胜段占比", BacktestApp._format_comparison_value(
                win_rate, 0, percent=True),
             None if win_rate is None else win_rate - 0.5),
            ("可比样本", f"{paired}/{baseline_windows} 段", None),
        )
        if getattr(self, "_history_objectives_available", True):
            incremental_pnl = finite(row.get("incremental_pnl_vs_c2c"))
            incremental_sharpe = finite(row.get("incremental_sharpe_vs_c2c"))
            return (
                ("增量收益 vs 每日收盘",
                 BacktestApp._format_objective_value(incremental_pnl, 2),
                 incremental_pnl),
                ("增量信噪比",
                 BacktestApp._format_objective_value(incremental_sharpe, 4),
                 incremental_sharpe),
            ) + tail
        return (
            ("损益波动", BacktestApp._format_comparison_value(
                row.get("daily_net_pnl_rms"), 2), None),
            ("每日收盘", BacktestApp._format_comparison_value(
                row.get("baseline_daily_net_pnl_rms"), 2), None),
        ) + tail

    def _refresh_history_conclusion_card(self):
        """让结论卡始终等于排名表里当前选中的那一行。

        绑定到选中行而不是固定的周期冠军：动作按钮作用的对象就是选中行，
        卡片若一直显示冠军，用户选了别的策略再点按钮就会跑到另一个上面。
        表格默认选中 rank 1，因此默认态仍是本周期推荐。
        """
        name_var = getattr(self, "_history_conclusion_name_var", None)
        badge_var = getattr(self, "_history_conclusion_badge_var", None)
        stats = getattr(self, "_history_conclusion_stats", None)
        if name_var is None or badge_var is None or stats is None:
            return
        try:
            for widget in list(stats.winfo_children()):
                widget.destroy()
        except tk.TclError:
            return

        period_var = getattr(self, "_history_selected_period_var", None)
        period_item = (
            getattr(self, "_history_period_rows", {}) or {}).get(
                period_var.get() if period_var is not None else "", {})
        period_label = str(period_item.get("period", "") or "")
        row = self._selected_history_rank_row()
        if not row:
            badge_var.set(period_label)
            name_var.set("请选择策略")
            BacktestApp._set_history_conclusion_accent(
                self, PALETTE["border_soft"])
            return

        is_baseline = str(
            row.get("strategy_type", "")) == "close_to_close"
        recommendation_eligible = BacktestApp._comparison_safe_bool(
            row.get("recommendation_eligible"),
            BacktestApp._comparison_safe_bool(
                row.get("complete_window"), False))
        rank_val = BacktestApp._comparison_safe_int(row.get("rank"), 0)
        suffix = f" · {period_label}" if period_label else ""
        if is_baseline:
            badge_var.set(f"基准（每日收盘）{suffix}")
            accent = PALETTE["primary"]
        elif rank_val == 1 and recommendation_eligible:
            badge_var.set(f"🏆 本周期最优{suffix}")
            accent = PALETTE["success"]
        elif not recommendation_eligible:
            badge_var.set(f"第 {rank_val or '—'} 名 · 数据不足，仅供参考{suffix}")
            accent = PALETTE["warning"]
        else:
            badge_var.set(f"第 {rank_val or '—'} 名{suffix}")
            accent = PALETTE["border_soft"]
        name_var.set(str(row.get("strategy", "—")))
        BacktestApp._set_history_conclusion_accent(self, accent)

        for column, (caption, text, signed) in enumerate(
                self._history_conclusion_stat_specs(row)):
            tile = tk.Frame(stats, bg=PALETTE["surface_alt"])
            tile.grid(row=0, column=column, sticky="w", padx=(0, 26))
            tk.Label(
                tile, text=caption, bg=PALETTE["surface_alt"],
                fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 8),
                anchor="w",
            ).pack(anchor="w")
            if signed is None or is_baseline:
                value_fg = PALETTE["text"]
            elif signed > 0:
                value_fg = PALETTE["success"]
            elif signed < 0:
                value_fg = PALETTE["danger"]
            else:
                value_fg = PALETTE["text_muted"]
            tk.Label(
                tile, text=text, bg=PALETTE["surface_alt"], fg=value_fg,
                font=(_UI_FONT_FAMILY, 12, "bold"), anchor="w",
            ).pack(anchor="w")

    def _set_history_conclusion_accent(self, color):
        accent = getattr(self, "_history_conclusion_accent", None)
        if accent is None:
            return
        try:
            accent.configure(bg=color)
        except tk.TclError:
            pass

    def _build_history_chart_box(self, parent):
        """渲染整段接续的损益图表、口径切换与逐段下钻入口。

        图表口径固定为「完整回放累积路径」，不再提供单段 / 中位段视图：
        排名指标是把各段日损益接成一条后统计的（见 hedge_analysis 里的
        ``pd.concat(daily_windows)``），单段曲线与整段排名可以给出相反的
        印象，而界面无从提示这是两个口径。想看某一段的细节走下方的
        「查看某段明细」，那是下钻，不是换口径。

        与排名表一样，自己不做布局：它是分割区的一格，由调用方挂载。
        """
        chart_box = ttk.LabelFrame(
            parent,
            text=" 累计损益对比 ",
            padding=14,
        )
        controls = tk.Frame(
            chart_box, bg=PALETTE["surface_alt"],
            highlightbackground=PALETTE["border_soft"], highlightthickness=1,
            padx=12, pady=6)
        controls.pack(fill="x", pady=(0, 4))
        tk.Label(
            controls, text="指标:", bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9, "bold"),
        ).pack(side="left")
        self._history_chart_metric_var = tk.StringVar(
            value=HISTORY_CHART_METRIC_DISPLAY["net"])
        self._history_chart_metric_combo = ttk.Combobox(
            controls, textvariable=self._history_chart_metric_var,
            values=tuple(HISTORY_CHART_METRIC_DISPLAY.values()),
            width=11, state="readonly",
        )
        self._history_chart_metric_combo.pack(side="left", padx=(5, 12))
        self._history_chart_metric_combo.bind(
            "<<ComboboxSelected>>", self._update_history_chart_controls)
        self._history_chart_hint_var = tk.StringVar(value="")
        # 显式 9pt：不写就继承全局默认 10pt，比同一条里 9pt 加粗的「指标:」
        # 还大——整个结果区最长、最不重要的一句反而字号最大。
        tk.Label(
            controls, textvariable=self._history_chart_hint_var,
            bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        # figsize 在分割区里只是**请求**高度，实际高度由用户拖分割线决定。
        # 它必须和排名表那一格的请求高度相当：图表请求得越高，tk 分配时越
        # 偏袒图表，并且会在我们摆好分割线之后再把它拉回去——而反复覆盖
        # 正是把界面卡死的那条闭环。2.8in≈270px 与排名表基本持平。
        self._history_chart_figure = Figure(
            figsize=(7.2, 2.8), dpi=self._CHART_DPI,
            facecolor=PALETTE["surface"], constrained_layout=True,
        )
        self._history_chart_ax = self._history_chart_figure.add_subplot(111)
        self._history_chart_canvas = FigureCanvasTkAgg(
            self._history_chart_figure, master=chart_box)
        # 下钻栏必须先 pack 且 side="bottom"：pack 按顺序分配空间，画布带
        # expand=True 会把剩余空腔全部吃掉，排在它后面的控件在容器不够高
        # 时会被压成 0 高度直接消失——而且 tkinter 对此完全沉默。先占住自
        # 己那条，画布再吃剩下的，缩窗口时缩的是图不是这一行。
        self._build_history_replay_bar(chart_box)
        self._history_chart_canvas.get_tk_widget().pack(
            fill="both", expand=True)
        return chart_box

    def _build_history_replay_bar(self, chart_box):
        """逐段下钻入口：紧贴图表，因为选段这个动作是从图上发起的。

        展示页结构上只能装一次回测，而排名口径是各段合并——整段没法塞进
        展示页，所以下钻天然要选一段。

        调用方必须在 pack 画布**之前**调它，见那里的说明。
        """
        replay = tk.Frame(
            chart_box, bg=PALETTE["surface_alt"],
            highlightbackground=PALETTE["border_soft"], highlightthickness=1,
            padx=12, pady=6)
        replay.pack(side="bottom", fill="x", pady=(6, 0))
        tk.Label(
            replay, text="查看某段明细:", bg=PALETTE["surface_alt"],
            fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 9, "bold"),
        ).pack(side="left")
        self._history_replay_window_var = tk.StringVar(value="")
        self._history_replay_window_combo = ttk.Combobox(
            replay, textvariable=self._history_replay_window_var,
            values=(), width=40, state="readonly",
        )
        self._history_replay_window_combo.pack(side="left", padx=(6, 8))
        # 名字必须说「加载」而不是「回测」：优选跑完时全部分段的 bar 级
        # 明细就已落盘，这里是把它读回来（约 27 ms），不是重跑（620 ms）。
        # 叫「回测」会让人以为点一下要再算一遍，也和旁边的说明自相矛盾。
        # 只有保存的明细确实不在时才回退重算，那时状态栏会说明。
        self._history_replay_button = ttk.Button(
            replay, text="加载明细",
            command=self._load_history_window_backtest,
        )
        self._history_replay_button.pack(side="left")
        # 载入的结果包里没有 bar 级数组（161 个回测约 153 MB，不进包），
        # 逐段下钻因此不可用。必须把原因说出来并把控件禁掉——留一个点了没
        # 反应的按钮比没有按钮更糟。
        loaded = getattr(self, "_history_loaded_meta", None)
        if loaded and not loaded.get("replay_available"):
            # 旧版本的包不含重放配方（只存了结果表）。禁掉并说明原因，而
            # 不是留一个点了没反应的按钮。
            note = "这份结果不含逐 bar 明细，无法加载；重新运行一次优选即可"
            self._history_replay_window_combo.configure(state="disabled")
            self._history_replay_button.configure(state="disabled")
        elif loaded:
            note = ("加载这一段已保存的逐 bar 明细，"
                    "送进回测摘要 / 对冲图表 / 每日明细并跳转")
        else:
            # 说清去向与副作用：点了会切走标签页，而且这一段会成为「当前
            # 回测」——两者都是点击前看不出来的。正常是读盘（约 27 ms），
            # 只有保存的明细不在时才回退到重算，那时状态栏会说明。
            # 「跳转」必须写：点完整页会切走，这是点击前完全看不出的。放在
            # 句首是因为这条说明在窄窗口会从句尾开始丢字。
            note = ("加载这一段已保存的逐 bar 明细并跳转到回测摘要 /"
                    " 对冲图表 / 每日明细，同时成为当前回测")
        # anchor="w" + fill/expand：这条说明在最小窗口下放不下，居中截断会
        # 把句首也吃掉；左对齐后丢的是句尾的次要信息。字号同样显式 9pt。
        tk.Label(
            replay, text=note, anchor="w",
            bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9),
        ).pack(side="left", fill="x", expand=True, padx=(12, 0))

    def _history_chart_candidates(self, lookback=None):
        """按排名顺序返回当前周期勾选的图表候选，不包含固定 每日收盘。"""
        if lookback is None:
            lookback, _row = self._history_chart_selection()
        selected = set(getattr(
            self, "_history_chart_selected_by_period", {}).get(
                str(lookback), set()))
        if not selected:
            return []
        ordered = []
        tree = getattr(self, "_history_rank_tree", None)
        rows = getattr(self, "_history_rank_rows", {})
        children = tree.get_children() if tree is not None else ()
        for iid in children:
            row = rows.get(iid, {})
            strategy = str(row.get("strategy", "")).strip()
            if (strategy in selected
                    and str(row.get("strategy_type", "")) != "close_to_close"):
                ordered.append(strategy)
        return ordered[:MAX_HISTORY_CHART_CANDIDATES]

    def _history_chart_primary_candidate(self, candidates):
        """排名表焦点只控制主候选强调，不改变图表勾选集合。"""
        focused = self._selected_history_rank_row() or {}
        strategy = str(focused.get("strategy", "")).strip()
        return strategy if strategy in candidates else (
            candidates[0] if candidates else None)

    def _history_top_chart_candidates(self, limit=None):
        """默认勾选：前 ``limit`` 个名次里能与已选共同作图的候选。

        ``limit`` 数的是**排名行数**，不是候选个数。基准（每日收盘）也占
        一个名次，但它是固定基准、图上恒在，不参与勾选——所以基准排进前三
        名时默认只勾上两个候选，不会为了凑够三个再往下多抓一个名次。

        名次窗口内的候选仍要逐个试算共同分段交集：与已选没有交集的候选画
        不进同一张图，跳过它而不是往下顺延，默认勾选才始终对应"最靠前的
        那几名"。
        """
        if limit is None:
            limit = BacktestApp._HISTORY_DEFAULT_CHART_RANKS
        lookback, _row = self._history_chart_selection()
        if not lookback:
            return []
        tree = getattr(self, "_history_rank_tree", None)
        rows = getattr(self, "_history_rank_rows", {})
        if tree is None:
            return []
        mode, metric = self._history_chart_view_options()
        selected = []
        for position, iid in enumerate(tree.get_children()):
            if position >= limit:
                break
            row = rows.get(iid, {})
            if str(row.get("strategy_type", "")) == "close_to_close":
                continue          # 基准占名次但不勾选
            strategy = str(row.get("strategy", "")).strip()
            if not strategy:
                continue
            trial = [*selected, strategy]
            model = BacktestApp._history_multi_chart_model(
                getattr(self, "_history_window_summary", None),
                lookback, trial, mode=mode, metric=metric,
                pairs_cache=self._history_chart_pairs_cache(),
            )
            if model.get("state") == "ok":
                selected.append(strategy)
            if len(selected) >= MAX_HISTORY_CHART_CANDIDATES:
                break
        return selected

    def _refresh_history_chart_marks(self):
        tree = getattr(self, "_history_rank_tree", None)
        if tree is None:
            return
        lookback, _row = self._history_chart_selection()
        selected = set(self._history_chart_candidates(lookback))
        rows = getattr(self, "_history_rank_rows", {})
        for iid in tree.get_children():
            row = rows.get(iid, {})
            strategy = str(row.get("strategy", "")).strip()
            is_baseline = str(
                row.get("strategy_type", "")) == "close_to_close"
            paired = self._comparison_safe_int(
                row.get("paired_windows", row.get("rolling_windows")), 0)
            if is_baseline:
                # 基线恒定入图，没有可勾选的状态，这格留空即可。身份已由
                # 策略列的「（基准）」标明，勾选列再写一遍「基准」是同一行
                # 内的第二次重复。
                tree.item(iid, image="", text="")
            elif paired > 0:
                tree.item(iid, image=self._cb_sf_checked if strategy in selected else self._cb_sf_unchecked, text="")
            else:
                tree.item(iid, image="", text="—")

    def _popup_history_rank_menu(self, event):
        """排名行右键：先把该行设成作用对象，再弹菜单。

        空白处右键不弹——弹一份点了只会提示"请先选一行"的菜单比不弹更费解，
        与结果池那张表同一条规矩。右键只挪选择，不碰「显示到图上」的勾选：
        那是 ``<Button-1>`` 在 #0 列的职责。
        """
        tree = getattr(self, "_history_rank_tree", None)
        menu = getattr(self, "_history_rank_menu", None)
        if tree is None or menu is None:
            return None
        iid = tree.identify_row(event.y)
        if not iid:
            return None
        tree.selection_set(iid)
        tree.focus(iid)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _show_history_row_params(self):
        """右键菜单「参数详情」：列出选中候选这一行背后的全部输入。"""
        row = self._selected_history_rank_row()
        if not row:
            messagebox.showinfo(
                "请选择策略", "请先在周期排名表中点击一条策略。")
            return
        # 两条渲染路径设的属性不同：跑完一轮和载入结果包走 _latest_history_
        # state，而 _show_history_recommendation 自己记的是 _history_last_
        # state。两个都兜住，免得从某一条路进来就打不开。
        history_state = (
            getattr(self, "_latest_history_state", None)
            or getattr(self, "_history_last_state", None) or {})
        # 判据是"有没有冻结期权大类"而不是"dict 是不是空的"：旧结果包解出来
        # 的 state 可能只剩周期勾选那几项，非空却一个参数也说不出。
        if not str(history_state.get("cls_name", "") or "").strip():
            messagebox.showinfo(
                "缺少运行参数",
                "这份优选结果没有记录运行时的期权与行情参数"
                "（本功能上线前保存的结果包）。\n\n"
                "重新运行一次优选即可看到完整参数。")
            return
        BacktestApp._open_params_window(
            self,
            heading=str(row.get("strategy", "未命名候选")),
            subtitle=BacktestApp._history_params_subtitle(row, history_state),
            sections=BacktestApp._history_row_detail_sections(
                row, history_state),
            notes={"期权类型与参数": BacktestApp._HISTORY_CONTRACT_NOTE},
        )

    @staticmethod
    def _history_params_subtitle(row, history_state):
        """详情窗副标题：这一行属于哪个周期、跑在哪份行情上。"""
        parts = ["策略优选"]
        period = str(row.get("period", "") or "").strip()
        if period:
            parts.append(f"周期 {period}")
        source = BacktestApp._snapshot_source_label(history_state)
        if source:
            parts.append(source)
        return "  ·  ".join(parts)

    def _toggle_history_chart_click(self, event):
        tree = self._history_rank_tree
        iid = tree.identify_row(event.y)
        if not iid or tree.identify_column(event.x) != "#0":
            return None
        tree.selection_set(iid)
        tree.focus(iid)
        self._toggle_history_chart_candidate(iid)
        return "break"

    def _toggle_focused_history_chart_candidate(self, _event=None):
        tree = getattr(self, "_history_rank_tree", None)
        if tree is None:
            return "break"
        selection = tree.selection()
        iid = selection[0] if selection else tree.focus()
        if iid:
            self._toggle_history_chart_candidate(iid)
        return "break"

    def _toggle_history_chart_candidate(self, iid):
        row = getattr(self, "_history_rank_rows", {}).get(iid, {})
        strategy = str(row.get("strategy", "")).strip()
        lookback, _selected_row = self._history_chart_selection()
        if not strategy or not lookback:
            return
        if str(row.get("strategy_type", "")) == "close_to_close":
            self._set_status("每日收盘 是图表固定基准，无需勾选或取消。")
            self._update_history_rank_selection()
            return
        pairs = BacktestApp._history_chart_pairs(
            getattr(self, "_history_window_summary", None),
            lookback, strategy)
        if pairs.empty:
            self._set_status(
                f"『{strategy}』没有可与 每日收盘 严格配对的图表回测分段。")
            self._update_history_rank_selection()
            return
        states = getattr(self, "_history_chart_selected_by_period", None)
        if states is None:
            self._history_chart_selected_by_period = {}
            states = self._history_chart_selected_by_period
        selected = states.setdefault(str(lookback), set())
        if strategy in selected:
            selected.remove(strategy)
        elif len(selected) >= MAX_HISTORY_CHART_CANDIDATES:
            self._set_status(
                f"图表最多同时显示 {MAX_HISTORY_CHART_CANDIDATES} 个候选；"
                "请先取消一个候选。")
            self._update_history_rank_selection()
            return
        else:
            current_candidates = self._history_chart_candidates(lookback)
            mode, metric = self._history_chart_view_options()
            preview = BacktestApp._history_multi_chart_model(
                getattr(self, "_history_window_summary", None),
                lookback, [*current_candidates, strategy],
                mode=mode, metric=metric,
                pairs_cache=self._history_chart_pairs_cache(),
            )
            if preview.get("state") != "ok":
                reason = preview.get("message", "没有共同的严格配对回测分段")
                self._set_status(f"无法加入『{strategy}』：{reason}")
                self._update_history_rank_selection()
                return
            selected.add(strategy)
        self._refresh_history_chart_marks()
        self._update_history_rank_selection()

    def _clear_history_chart_candidates(self):
        lookback, _row = self._history_chart_selection()
        if not lookback:
            return
        self._history_chart_selected_by_period[str(lookback)] = set()
        self._refresh_history_chart_marks()
        self._update_history_chart_controls()
        self._set_status("已清空图表候选；每日收盘基准仍固定显示。")

    def _history_chart_selection(self):
        """返回当前历史周期与排名行。"""
        period_var = getattr(self, "_history_selected_period_var", None)
        if period_var is None:
            return None, None
        period = getattr(self, "_history_period_rows", {}).get(
            period_var.get(), {})
        return period.get("lookback"), self._selected_history_rank_row()

    def _history_chart_view_options(self):
        """返回绘图口径。模式恒为 ``full``，只有指标可切。

        排名指标统计的是各段日损益接成一条之后的整段，图表必须同口径。
        单段 / 中位段视图在后端仍然实现着，但不再从界面进入——同一个策略
        整段为正、某一段为负是常态，两个口径并列摆着只会误导。
        """
        metric_var = getattr(self, "_history_chart_metric_var", None)
        metric = HISTORY_CHART_METRIC_FROM_DISPLAY.get(
            metric_var.get() if metric_var is not None else "", "net")
        return "full", metric

    def _history_chart_pairs_cache(self):
        """返回本次结果页的策略配对缓存；结果页重建时由渲染入口清空。"""
        cache = self._history_pairs_cache
        if cache is None:
            cache = {}
            self._history_pairs_cache = cache
        return cache

    def _update_history_chart_controls(self, _event=None):
        """切换指标后重绘；图表口径固定，没有别的控件需要联动。"""
        self._history_chart_metric_combo.configure(state="readonly")
        self._draw_history_chart()

    def _update_history_rank_selection(self, _event=None):
        """排名行变化时同步刷新相对 每日收盘 结论与两种历史图表。"""
        prefix = getattr(self, "_history_period_detail_prefix", "")
        row = self._selected_history_rank_row()
        detail_var = getattr(self, "_history_detail_var", None)
        if detail_var is not None:
            if not row:
                detail_var.set(prefix or "请选择策略")
            else:
                strategy = str(row.get("strategy", "—"))
                is_baseline = str(
                    row.get("strategy_type", "")) == "close_to_close"
                rms_text = self._format_comparison_value(
                    row.get("daily_net_pnl_rms"), 2)
                baseline_text = self._format_comparison_value(
                    row.get("baseline_daily_net_pnl_rms"), 2)
                # 详情行只补表格没有的原始损益波动值。表格现在的四列是
                # 增量收益/增量信噪比/增量成本/最大回撤，改善率与胜率都
                # 不在其中——它们是刻意收敛掉的旧口径，不再单独渲染。
                uses_strict_metric = (
                    BacktestApp._history_row_uses_strict_metric(row))
                uses_product_pool = (
                    row.get("history_mode") == "product_contract_pool")
                rms_label = (
                    "损益波动(参考)" if uses_product_pool
                    else "损益波动" if uses_strict_metric
                    else "损益波动(旧版)")
                selected_detail = (
                    f"{rms_label} {rms_text}，"
                    f"每日收盘 {baseline_text}")
                if not uses_strict_metric:
                    improvement_text = (
                        "基准" if is_baseline else
                        self._format_comparison_value(
                            BacktestApp._history_row_improvement(row),
                            1, signed=True, percent=True))
                    selected_detail += (
                        f" · 旧版改善 {improvement_text}"
                        " · 建议重新运行")

                if row.get("history_mode") == "product_contract_pool":
                    code_text = BacktestApp._history_contract_codes_text(
                        row.get("paired_contract_codes", ()))
                    if code_text:
                        selected_detail += f" · 参与合约 {code_text}"
                    elif not uses_strict_metric:
                        selected_detail += " · 参与合约未记录"
                failure_reason = str(
                    row.get("failure_reason") or "").strip()
                if failure_reason:
                    if len(failure_reason) > 96:
                        failure_reason = failure_reason[:93] + "…"
                    selected_detail += f" · 失败原因：{failure_reason}"
                detail_var.set(
                    f"{prefix} · {selected_detail}" if prefix else selected_detail)
        BacktestApp._refresh_history_conclusion_card(self)
        self._refresh_history_replay_windows()
        self._update_history_chart_controls()

    @staticmethod
    def _history_replay_label(spec):
        """用分段自身的行情边界生成可读标签，不依赖排名表文案。"""
        metadata = dict(getattr(spec, "metadata", {}) or {})
        segment_no = metadata.get("segment_no")
        parts = [
            f"第 {segment_no} 段" if segment_no is not None
            else f"第 {spec.window_id} 段"]
        index = getattr(spec.external_path, "index", None)
        if index is not None and len(index):
            start = BacktestApp._format_detail_index(index[0])
            end = BacktestApp._format_detail_index(index[-1])
            parts.append(f"{start}~{end}")
        contract_code = str(metadata.get("contract_code") or "").strip()
        if contract_code:
            parts.append(contract_code)
        terminal_labels = {
            "expiry": "持有到期", "mark_to_market": "按市价结算"}
        terminal_mode = str(metadata.get("terminal_mode") or "")
        if terminal_mode:
            parts.append(terminal_labels.get(terminal_mode, terminal_mode))
        return " · ".join(parts)

    def _history_replay_options(self, strategy_name=None, lookback=None):
        """返回选中候选在当前周期内所有可重放的分段，按分段序号排列。"""
        if lookback is None:
            lookback, _row = self._history_chart_selection()
        if strategy_name is None:
            row = self._selected_history_rank_row() or {}
            strategy_name = str(row.get("strategy", "")).strip()
        strategy_name = str(strategy_name or "").strip()
        index = getattr(self, "_history_replay_index", None) or {}
        specs = index.get(str(lookback), {})
        options = []
        for window_id, spec in specs.items():
            if strategy_name not in getattr(spec, "strategies", {}):
                continue
            options.append({
                "window_id": str(window_id),
                "label": BacktestApp._history_replay_label(spec),
                "spec": spec,
            })
        options.sort(key=lambda item: BacktestApp._comparison_safe_int(
            dict(getattr(item["spec"], "metadata", {}) or {}).get(
                "segment_no"), 0))
        return options

    def _refresh_history_replay_windows(self):
        """按当前周期与候选刷新可加载分段；无可重放分段时清空下拉。"""
        combo = getattr(self, "_history_replay_window_combo", None)
        var = getattr(self, "_history_replay_window_var", None)
        if combo is None or var is None:
            return []
        options = self._history_replay_options()
        labels = [item["label"] for item in options]
        try:
            combo.configure(
                values=tuple(labels),
                state="readonly" if labels else "disabled")
        except tk.TclError:
            return options
        if var.get() not in labels:
            # 默认落在最近一段：段按时间升序排列，末项离当下最近，也是
            # 用户最可能想复核的那一笔。
            var.set(labels[-1] if labels else "")
        return options

    def _selected_history_replay_spec(self):
        """返回下拉选中的分段配方与候选名；缺任一条件时返回 None。"""
        row = self._selected_history_rank_row() or {}
        strategy_name = str(row.get("strategy", "")).strip()
        if not strategy_name:
            return None, ""
        var = getattr(self, "_history_replay_window_var", None)
        label = var.get() if var is not None else ""
        options = self._history_replay_options(strategy_name)
        if not options:
            return None, strategy_name
        # 与下拉的默认值保持一致：标签对不上时落到最近一段，不是最早那段。
        chosen = next(
            (item for item in options if item["label"] == label), options[-1])
        return chosen["spec"], strategy_name

    def _load_history_window_backtest(self):
        """把选中候选的单个历史分段重跑并加载到普通回测展示页。"""
        if getattr(self, "_active_job", None) is not None:
            messagebox.showinfo("任务运行中", "请等待当前任务完成后再加载分段。")
            return False
        spec, strategy_name = self._selected_history_replay_spec()
        if spec is None:
            messagebox.showinfo(
                "没有可加载的分段",
                "请先在周期排名表选中一个成功配对的候选；"
                "失败或未参与的候选没有保存下明细。"
                if strategy_name else "请先在周期排名表中选择一条策略。")
            return False
        label = BacktestApp._history_replay_label(spec)
        if not self._begin_job(
                "history_replay",
                f"正在加载『{strategy_name}』{label} 的明细…"):
            return False
        self._progress.configure(mode="indeterminate")
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.start(15)
        threading.Thread(
            target=self._history_replay_worker,
            args=(spec, strategy_name), daemon=True,
        ).start()
        return True

    @staticmethod
    def _replay_with_cache(spec, strategy_name):
        """取一段的 bar 级明细，返回 ``(回测对象, 是否重算过)``。

        **正常路径是「读已保存的数据」，不是重跑。** 一轮优选跑完时全部分
        段的 bar 级结果就已落盘（见 ``_cache_history_bars``），这里直接读
        回来，实测 27 ms；真跑一段要 620 ms。

        只有缓存确实不在时才回退到重算——缓存被清过、旧结果包、或者当初
        写盘失败。回退会算出同样的数字，但调用方要把这件事说出来：用户按
        的是「加载」，多等半秒总得有个交代。

        ``run()`` 只设置 ``_results``（见 hedge_backtest），所以命中时用
        ``build()`` 造出未运行的对象再把结果塞进去，与真跑出来的等价。
        """
        backtest = spec.build(strategy_name)
        cached = history_bar_cache.load(spec, strategy_name)
        if cached is not None:
            backtest._results = cached
            return backtest, False
        backtest.run()
        history_bar_cache.store(spec, strategy_name, backtest._results)
        return backtest, True

    @staticmethod
    def _cache_history_bars(window_results):
        """把一轮优选的全部分段结果写进磁盘缓存。

        缓存写失败绝不能影响主流程——它只省时间，没有它一切照常，所以这里
        整体兜底。不在这里按 LRU 裁容量是有意的：占用改由「最多保留 20 份
        优选记录」间接约束（见 history_bar_cache 模块头与 history_store.
        MAX_RESULTS）。
        """
        try:
            index = history_replay_index(window_results or {})
        except Exception:                              # noqa: BLE001
            return 0
        written = 0
        for _lookback, windows in index.items():
            for _window_id, spec in windows.items():
                try:
                    per_case = window_results[spec.lookback][spec.window_id]
                except Exception:                      # noqa: BLE001
                    continue
                for name in spec.strategy_names():
                    try:
                        result = per_case.get(name)
                    except Exception:                  # noqa: BLE001
                        continue
                    if not isinstance(result, Mapping):
                        continue
                    if history_bar_cache.store(spec, name, result):
                        written += 1
        return written

    @staticmethod
    def _replay_fidelity_error(result, spec, strategy_name, summary):
        """重放结果与包内逐日损益不符时返回一句描述，一致或无从比对返回 None。

        重建路径与原始运行是两套代码，任何一处对不上都会让下钻显示的数字
        与排名不符**且不会报错**。这个函数就是那道关——早前只在注释里写了
        它的名字却没实现，而它本该拦住的正是缓存 key 漏项、分段策略集合不
        对、预热参数丢失这几类问题。
        """
        if summary is None or getattr(summary, "empty", True):
            return None
        try:
            rows = summary[
                (summary["lookback"].astype(str) == str(spec.lookback))
                & (summary["window_id"].astype(str) == str(spec.window_id))
                & (summary["strategy"].astype(str) == str(strategy_name))
            ]
        except (KeyError, TypeError):
            return None
        if rows.empty or not bool(rows["success"].iloc[0]):
            return None
        expected = np.asarray(rows["daily_net_pnl"].iloc[0], dtype=float)
        try:
            actual = _aggregate_result_by_day(
                result, int(spec.steps_per_day))["net_pnl"].to_numpy()
        except Exception as exc:                       # noqa: BLE001
            return f"无法聚合重放结果：{type(exc).__name__} {exc}"
        if len(expected) != len(actual):
            return (f"重放天数 {len(actual)} 与记录的 {len(expected)} 不符")
        if not np.allclose(expected, actual, rtol=1e-9, atol=1e-12,
                           equal_nan=True):
            worst = float(np.max(np.abs(expected - actual)))
            return f"重放逐日损益与记录不符（最大差 {worst:.3e}）"
        return None

    def _history_replay_worker(self, spec, strategy_name):
        try:
            bt, recomputed = BacktestApp._replay_with_cache(
                spec, strategy_name)
            # 与包内逐日损益核对。不一致时照常展示（数据本身可能仍有诊断
            # 价值），但必须明说——静默显示对不上的数字才是最坏的结果。
            mismatch = BacktestApp._replay_fidelity_error(
                bt._results, spec, strategy_name,
                getattr(self, "_history_window_summary", None))
            self.after(
                0,
                lambda: self._deliver_history_replay(
                    bt, spec, strategy_name, mismatch=mismatch,
                    recomputed=recomputed),
            )
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(
                0,
                lambda message=err_msg: self._fail_history_replay(message))

    @staticmethod
    def _replay_scaled_params(params, option):
        """把分段实际使用的价格量纲参数写回，其余参数原样保留。

        缩放只动价格量纲那几项（各期权类自己声明在
        ``_PRICE_FIELDS_BY_CLS``），波动率、期限这些不受影响。
        """
        params = dict(params or {})
        if option is None:
            return params
        fields = ("s0",) + tuple(_PRICE_FIELDS_BY_CLS.get(
            type(option).__name__, ()))
        for field in fields:
            value = getattr(option, field, None)
            if value is None:
                continue
            number = BacktestApp._comparison_finite(value)
            if number is not None and field in params:
                params[field] = number
        return params

    def _history_replay_gui_state(self, spec, strategy_name):
        """把历史任务快照收敛成展示与快照保留都能消费的单次回测状态。"""
        history_state = getattr(self, "_latest_history_state", None) or {}
        state = BacktestApp._copy_snapshot_gui_state(history_state)
        metadata = dict(getattr(spec, "metadata", {}) or {})
        index = getattr(spec.external_path, "index", None)
        if index is not None and len(index):
            # 保留时分：日内滚动的相邻两段起止日相同，只到天的话两条快照
            # 的行情签名会一模一样，页面就会说"输入完全相同"。
            state["wind_start"] = BacktestApp._format_detail_index(
                index[0], include_time=True)
            state["wind_end"] = BacktestApp._format_detail_index(
                index[-1], include_time=True)
        contract_code = str(metadata.get("contract_code") or "").strip()
        if contract_code:
            state["wind_code"] = contract_code
        # 每段实际用的期权是按该段首价整体缩放过的（_rescale_option_to_real_s0），
        # 而 params 原样留着用户填的参考价——于是两段行权价实际差了 50%，
        # 快照里的期权签名却一模一样，对比页恒报"期权：相同"。
        state["params"] = BacktestApp._replay_scaled_params(
            state.get("params"), getattr(spec, "option", None))
        strategy = spec.strategies.get(strategy_name)
        state["strategy_name"] = getattr(strategy, "name", "unknown")
        # 策略的实际参数要从这一条候选的策略对象上取。原先只设了策略类型
        # 名，带宽阈值与波动率口径仍沿用整个优选任务的配置——于是同一段
        # 里重放两个不同候选（1σ 与 2σ），两条快照的策略签名一模一样。
        if isinstance(strategy, HedgeBandStrategy):
            state["interval_type"] = str(
                getattr(strategy, "band_type", "sigma"))
            state["price_interval"] = float(
                getattr(strategy, "threshold", 0.0))
            state["sigma_source"] = str(
                getattr(strategy, "sigma_source", "implied"))
            window_days = BacktestApp._comparison_safe_int(
                getattr(strategy, "window_days", None), 20)
            state["sigma_window"] = max(2, window_days)
        elif isinstance(strategy, FixedTimeStrategy):
            times = getattr(strategy, "requested_times", ())
            state["fixed_times"] = ",".join(
                value.strftime("%H:%M") if hasattr(value, "strftime")
                else str(value) for value in times)
        state["history_replay_strategy"] = strategy_name
        state["history_replay_lookback"] = spec.lookback
        state["history_replay_window_id"] = spec.window_id
        return state

    def _deliver_history_replay(self, bt, spec, strategy_name,
                                mismatch=None, recomputed=False):
        """在主线程渲染重放结果；成功后它也是可保留的当前回测。"""
        success = False
        try:
            state = self._history_replay_gui_state(spec, strategy_name)
            bt._gui_meta = {
                "cls_name": state.get("cls_name"),
                "subtype": state.get("subtype"),
                "source": state.get("source"),
                "wind_start": state.get("wind_start"),
                "wind_end": state.get("wind_end"),
                "wind_bar_size": state.get("wind_bar_size"),
                "wind_bar_size_requested": state.get(
                    "wind_bar_size_requested"),
                "wind_date_mode": state.get("wind_date_mode"),
            }
            timestamps = (bt._results or {}).get("timestamps")
            if timestamps is not None:
                bt._wind_meta = {"dates": timestamps}
            self._show_results(bt, None)
            self._latest_backtest = bt
            self._latest_backtest_state = state
            self._latest_retained_result_id = None
            success = True
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            messagebox.showerror("历史分段加载失败", err_msg)
        finally:
            self._finish_history_replay(
                success, spec, strategy_name)
            if success and recomputed:
                # 按钮叫「加载」，正常就该是秒开的读盘。走到重算说明保存
                # 的明细不在了，得说一声，否则用户只感觉「怎么卡了一下」。
                self._set_status(
                    f"已保存的明细不存在，已重新计算并补存："
                    f"{spec.lookback}/{spec.window_id}/{strategy_name}")
            if success and mismatch:
                # 数字对不上必须说出来。照常展示是因为它仍有诊断价值，但
                # 不说的话用户会把它当成排名的依据来读。
                messagebox.showwarning(
                    "重放结果与记录不一致",
                    f"{spec.lookback}/{spec.window_id}/{strategy_name}：\n"
                    f"{mismatch}\n\n"
                    "展示页的数字可能与排名表不同源，请重新运行一次策略优选。")
                self._set_status(f"⚠ 重放与记录不一致：{mismatch}")

    def _fail_history_replay(self, message):
        messagebox.showerror("历史分段回测失败", message)
        self._finish_history_replay(False)

    def _finish_history_replay(self, success=True, spec=None,
                               strategy_name=""):
        label = (
            BacktestApp._history_replay_label(spec)
            if spec is not None else "")
        detail = (
            f"已加载『{strategy_name}』{label}  |  "
            "可在每日明细查看逐次对冲，或保留到结果对比"
            if success and spec is not None else "历史分段加载完成")
        self._finish_job(
            "history_replay", success=success,
            success_text=detail,
            failure_text="分段加载失败  |  请查看错误信息",
        )

    def _draw_history_chart(self, _event=None):
        """绘制固定 每日收盘 加多个已勾选候选的严格配对回测分段图表。"""
        ax = getattr(self, "_history_chart_ax", None)
        canvas = getattr(self, "_history_chart_canvas", None)
        if ax is None or canvas is None:
            return
        ax.clear()
        lookback, _focused_row = self._history_chart_selection()
        candidates = self._history_chart_candidates(lookback)
        primary_strategy = self._history_chart_primary_candidate(candidates)
        mode, metric = self._history_chart_view_options()
        model = BacktestApp._history_multi_chart_model(
            getattr(self, "_history_window_summary", None),
            lookback, candidates, mode=mode, metric=metric,
            primary_strategy=primary_strategy,
            pairs_cache=self._history_chart_pairs_cache(),
        )

        if model.get("state") != "ok":
            message = model.get("message", "暂无可绘制的配对回测分段。")
            ax.text(
                0.5, 0.5, message, ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["text_muted"],
                fontsize=9, wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._history_chart_hint_var.set(message)
            canvas.draw_idle()
            return

        baseline_color = "#334155"
        fallback_colors = ("#2563EB", "#D97706", "#7C3AED")
        fallback_markers = ("o", "s", "^")
        color_map = getattr(self, "_history_chart_color_map", {})
        marker_map = getattr(self, "_history_chart_marker_map", {})

        def _candidate_style(strategy, index):
            return (
                color_map.get(strategy, fallback_colors[index % len(fallback_colors)]),
                marker_map.get(
                    strategy, fallback_markers[index % len(fallback_markers)]),
            )

        common_count = self._comparison_safe_int(
            model.get("common_window_count"), 0)
        scope_note = (
            "；可比段数少于单个策略各自的段数，与排名表不同"
            if model.get("sample_scope_differs") else "")
        # 口径固定为整段接续，只剩这一条绘制分支。
        candidate_index = 0
        for series in model["series"]:
            baseline = series["role"] == "baseline"
            strategy = str(series.get("strategy", series.get("label", "")))
            if baseline:
                color, marker = baseline_color, None
            else:
                color, marker = _candidate_style(strategy, candidate_index)
                candidate_index += 1
            emphasized = strategy == primary_strategy
            y_values = np.asarray(series["y"], dtype=float)
            ax.plot(
                series["x"], y_values, label=series["label"],
                color=color,
                linestyle="--" if baseline else "-",
                linewidth=(2.1 if baseline else 2.5 if emphasized else 1.8),
                marker=marker, markersize=3.2,
                markevery=(None if baseline else max(1, len(y_values) // 9)),
                alpha=1.0 if baseline or emphasized else 0.86,
            )
        # X 轴标签与下方图例位置冲突，且“横轴是交易日”一望即知。
        ax.set_xlabel("")
        ax.set_ylabel(model["metric_label"], fontsize=8)
        for boundary in model.get("segment_boundaries", ()):
            ax.axvline(
                float(boundary) + 0.5,
                color=PALETTE["text_muted"], linestyle=":",
                linewidth=0.7, alpha=0.55)
        common_days = self._comparison_safe_int(
            model.get("common_day_count"), 0)
        expected_days = self._comparison_safe_int(
            model.get("expected_day_count"), common_days)
        completeness = (
            "完整" if model.get("complete_evidence") else "部分重叠")
        self._history_chart_hint_var.set(
            f"{completeness} {common_days}/{expected_days} 日 · "
            f"每日收盘 + {len(candidates)} 个策略{scope_note}")
        if model.get("uses_normalized_notional"):
            self._history_chart_hint_var.set(
                self._history_chart_hint_var.get()
                + "；跨合约按各段 损益/(S0×乘数×|数量|) 安全归一化后拼接")

        ax.axhline(
            0.0, color=PALETTE["text_muted"],
            linewidth=0.8, alpha=0.65)
        ax.set_title(model.get("title", ""), fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.55)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            # 曲线通常贴着上下边缘，图内任何位置都会压住数据，因此把图例
            # 放到绘图区下方横排。
            ax.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, -0.11), frameon=False, fontsize=8,
                ncol=min(4, max(1, len(labels))), borderaxespad=0.0)
        ax.margins(x=0.02, y=0.12)
        canvas.draw_idle()

    def _select_history_period(self, key):
        """切换分析周期并同步分段控件的高亮。"""
        variable = getattr(self, "_history_selected_period_var", None)
        if variable is None or key not in getattr(
                self, "_history_period_rows", {}):
            return
        variable.set(key)
        BacktestApp._refresh_history_period_chips(self)
        self._update_history_selection()

    def _refresh_history_period_chips(self):
        """选中态用主题主色实心，未选中用次级表面色。"""
        chips = getattr(self, "_history_period_chips", None)
        variable = getattr(self, "_history_selected_period_var", None)
        if not chips or variable is None:
            return
        current = variable.get()
        for key, chip in chips.items():
            if key == current:
                chip.configure(
                    bg=PALETTE["primary"], fg="#FFFFFF",
                    font=(_UI_FONT_FAMILY, 10, "bold"))
            else:
                chip.configure(
                    bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
                    font=(_UI_FONT_FAMILY, 10))

    def _update_history_selection(self, _event=None):
        rank_tree = getattr(self, "_history_rank_tree", None)
        period_var = getattr(self, "_history_selected_period_var", None)
        if rank_tree is None or period_var is None:
            return
        period_item = self._history_period_rows.get(period_var.get(), {})
        if not period_item:
            return
        for child in rank_tree.get_children():
            rank_tree.delete(child)

        self._history_rank_rows = {}
        ranking = self._history_ranking
        group = None
        if ranking is not None and not getattr(ranking, "empty", True):
            selected = ranking[
                ranking["lookback"] == period_item.get("lookback")]
            if not selected.empty:
                group = selected.sort_values(
                    ["rank", "daily_net_pnl_rms", "strategy"], kind="stable")
        if group is not None:
            baseline = BacktestApp._comparison_baseline(group)
            for row_no, (_idx, row) in enumerate(group.iterrows()):
                rank_item = row.to_dict()
                is_baseline = str(
                    row.get("strategy_type", "")) == "close_to_close"
                paired = self._comparison_safe_int(
                    row.get("paired_windows", row.get("rolling_windows")), 0)
                baseline_windows = self._comparison_safe_int(
                    row.get("baseline_windows", paired), paired)
                comparison_eligible = self._comparison_safe_bool(
                    row.get("comparison_eligible"),
                    self._comparison_safe_bool(
                        row.get("complete_window"), False))
                recommendation_eligible = self._comparison_safe_bool(
                    row.get("recommendation_eligible"),
                    self._comparison_safe_bool(
                        row.get("complete_window"), False))
                if recommendation_eligible:
                    status = "可比"
                elif comparison_eligible:
                    status = "可比（仅参考）"
                elif paired:
                    status = "部分可比"
                else:
                    status = "不可比"
                strategy_label = str(row.get("strategy", "—"))
                if is_baseline:
                    strategy_label = (
                        BacktestApp._history_baseline_row_label(
                            strategy_label))
                rank_val = self._comparison_safe_int(
                    row.get("rank"), row_no + 1)
                rank_str = str(rank_val)
                if rank_val == 1:
                    rank_str = "🥇 " + rank_str
                elif rank_val == 2:
                    rank_str = "🥈 " + rank_str
                elif rank_val == 3:
                    rank_str = "🥉 " + rank_str

                values = BacktestApp._pad_history_row(
                    (
                        rank_str, strategy_label,
                        BacktestApp._format_objective_value(
                            row.get("incremental_pnl_vs_c2c"), 4),
                        BacktestApp._format_objective_value(
                            row.get("incremental_sharpe_vs_c2c"), 4),
                        BacktestApp._format_objective_value(
                            row.get("incremental_tc_vs_c2c"), 4),
                        BacktestApp._format_drawdown_value(
                            row.get("max_drawdown")),
                        BacktestApp._format_history_status(
                            status, paired, baseline_windows),
                    ),
                    rank_tree["columns"],
                )
                # 唯一的行标签：基准行加粗。其余一律不带标签——行底色只由
                # 选中态表达。
                tag = "baseline" if is_baseline else ""
                # 基线没有可勾选状态，这格留空；身份由策略列的「（基准）」
                # 标明。此前这里写「基准」，但 _refresh_history_chart_marks
                # 紧接着就把它无条件清成空串，写了也从不显示。
                image_val = "" if is_baseline else self._cb_sf_unchecked
                text_val = ""

                iid = f"history_rank_{row_no}"
                rank_tree.insert(
                    "", "end", iid=iid, image=image_val, text=text_val, values=values,
                    tags=(tag,) if tag else ())
                self._history_rank_rows[iid] = row.to_dict()

        lookback_key = str(period_item.get("lookback", ""))
        chart_states = getattr(
            self, "_history_chart_selected_by_period", None)
        if chart_states is None:
            self._history_chart_selected_by_period = {}
            chart_states = self._history_chart_selected_by_period
        if lookback_key not in chart_states:
            chart_states[lookback_key] = set(
                self._history_top_chart_candidates())
        self._refresh_history_chart_marks()

        rank_children = rank_tree.get_children()
        if rank_children:
            rank_tree.selection_set(rank_children[0])
            rank_tree.focus(rank_children[0])

        maturity = period_item.get("maturity_days")
        extra = []
        maturity_value = self._comparison_finite(maturity)
        if maturity_value is not None:
            extra.append(
                f"合约期限 {self._comparison_safe_int(maturity_value)} 日")
        warmup_days = self._comparison_safe_int(
            period_item.get("realized_sigma_warmup_days"), 0)
        if warmup_days:
            extra.append(f"波动率预热 {warmup_days} 日")
        evidence_days = self._comparison_safe_int(
            period_item.get("evidence_days"), 0)
        if evidence_days:
            extra.append(f"回放 {evidence_days} 日")
        elif period_item.get("requested", 0):
            extra.append(f"回放 {period_item.get('requested', 0)} 日")
        days_used = self._comparison_safe_int(
            period_item.get("days_used"), 0)
        # 与目标天数一致时不重复报数，只有实际短于目标才值得点出来。
        if days_used and days_used != evidence_days:
            extra.append(f"实际用到 {days_used} 日")
        segment_count = period_item.get("segment_count")
        if segment_count is not None:
            segment_count = self._comparison_safe_int(segment_count, 0)
            expiry_segments = self._comparison_safe_int(
                period_item.get("expiry_segments"), 0)
            mtm_segments = self._comparison_safe_int(
                period_item.get("mtm_segments"), 0)
            segment_text = f"分成 {segment_count} 段"
            if expiry_segments or mtm_segments:
                segment_text += (
                    f"（持有到期 {expiry_segments} / 按市价结算 {mtm_segments}）")
            extra.append(segment_text)
        # 残段在最老那一端（段边界从区间末端倒推对齐），不再是“末段”。
        terminal_labels = {
            "expiry": "持有到期",
            "mark_to_market": "按市价结算",
            "mixed": "到期 + 最老一段按市价结算",
        }
        terminal_mode = str(period_item.get("terminal_mode") or "")
        # 分段明细的括号里已经写了“持有到期 X / 按市价结算 Y”，此时再说一次
        # 结束方式就是重复；只有拿不到分段构成时才补这一句。
        if terminal_mode and segment_count is None:
            extra.append(
                "结束方式：" + terminal_labels.get(
                    terminal_mode, terminal_mode))
        sampling_mode = str(period_item.get("sampling_mode", ""))
        if not period_item.get("uses_strict_metric", False):
            extra.append("旧版取样方式，建议重新运行")
        elif sampling_mode and sampling_mode != "strict_contiguous":
            extra.append("取样方式与本次回放区间不一致")

        required = self._comparison_safe_int(
            period_item.get("required_history_days"), 0)
        available_history = self._comparison_safe_int(
            period_item.get("available_history_days"), 0)
        if required:
            extra.append(f"历史数据 {available_history}/{required} 日")
        elif period_item.get("requested", 0):
            extra.append(
                f"历史数据 {period_item.get('available', 0)}/"
                f"{period_item.get('requested', 0)} 日")
        if period_item.get("history_mode") == "product_contract_pool":
            effective_date = period_item.get("effective_asof_date")
            effective_date_text = (
                str(effective_date)[:10] if effective_date is not None else "—")
            extra.append(
                f"品种池={period_item.get('product_code', '—')}，"
                f"有效截止={effective_date_text}/"
                f"{period_item.get('effective_main_contract', '—')}")
            loaded = self._comparison_safe_int(
                period_item.get("pool_contracts_available"), 0)
            invalid = self._comparison_safe_int(
                period_item.get("pool_contracts_invalid"), 0)
            extra.append(f"历史合约={loaded}可用/{invalid}失败")
            code_text = BacktestApp._history_contract_codes_text(
                period_item.get("paired_contract_codes", ()))
            if code_text:
                extra.append("周期结论参与合约=" + code_text)
            mapping_dropped = self._comparison_safe_int(
                period_item.get("mapping_trailing_days_dropped"), 0)
            if mapping_dropped:
                extra.append(f"回退盘中主力映射={mapping_dropped}日")
        skipped_segments = self._comparison_safe_int(
            period_item.get("skipped"), 0)
        replayable = self._comparison_safe_int(period_item.get("eligible"), 0)
        effective_segments = self._comparison_safe_int(
            period_item.get("effective"), 0)
        # 可比段数已由表格“配对”列给出；这里只在出现失败段、或可回放段数
        # 与可比段数不一致时才补充说明，避免又抄一遍表格。
        if skipped_segments:
            extra.append(f"失败 {skipped_segments} 段")
        if replayable and replayable != effective_segments:
            extra.append(f"可回放 {replayable} 段")
        relative_value = period_item.get("relative_comparison_windows")
        relative_windows = (
            self._comparison_safe_int(relative_value, 0)
            if relative_value is not None else None)
        if (relative_windows is not None and period_item.get("effective", 0)
                and relative_windows != period_item.get("effective", 0)):
            extra.append(
                f"参与评分 {relative_windows}/"
                f"{period_item.get('effective', 0)} 段")
        if period_item.get("trailing_dropped", 0):
            extra.append(
                f"末尾剔除 {period_item['trailing_dropped']} 个未收盘交易日")
        # 增量指标缺失时，表格里那两列全是「—」，而真正的排序依据没有任何
        # 一列体现出来。必须说出来排的是什么，否则用户看到的是一张没有决
        # 策依据的排名表。
        if not getattr(self, "_history_objectives_available", True):
            extra.append(
                "本次按「各段日损益波动较每日收盘的改善」排名"
                "（跨合约金额不可直接相加，增量口径不可用）")
        if (group is not None
                and not period_item.get("has_comparable_candidate", False)):
            candidate_failures = group[
                ~group["strategy_type"].astype(str).eq("close_to_close")
            ].get("failure_reason")
            if candidate_failures is not None:
                reasons = [
                    str(reason).strip() for reason in candidate_failures
                    if str(reason).strip() and str(reason) != "nan"
                ]
                if reasons:
                    reason = reasons[0]
                    if len(reason) > 72:
                        reason = reason[:69] + "…"
                    extra.append(f"候选失败：{reason}")
        # 周期名与数据完整度已经是上方表格的两列，前缀不再重复它们。
        # extra 描述的是这个「周期」本身（合约期限、回放天数、分段构成…），
        # 归属上属于周期结论表，因此挂到那张表下方；排名表下面只留与选中
        # 策略有关的内容，两个层级不再混在一行里。
        period_context = getattr(self, "_history_period_context_var", None)
        if period_context is not None:
            period_context.set(" · ".join(extra))
            self._history_period_detail_prefix = ""
        else:
            self._history_period_detail_prefix = " · ".join(extra)
        self._history_detail_var.set(self._history_period_detail_prefix)
        self._update_history_rank_selection()

    def _selected_history_rank_row(self):
        tree = getattr(self, "_history_rank_tree", None)
        rows = getattr(self, "_history_rank_rows", {})
        if tree is None:
            return None
        selection = tree.selection()
        iid = selection[0] if selection else tree.focus()
        row = rows.get(iid)
        return copy.deepcopy(row) if row is not None else None

    def _apply_history_recommendation(self, row=None, *, navigate=True):
        """将选中历史候选明确写回单次回测控件。"""
        if row is None:
            row = self._selected_history_rank_row()
        elif hasattr(row, "to_dict"):
            row = row.to_dict()
        else:
            row = dict(row)
        if not row:
            messagebox.showinfo("请选择策略", "请先在周期排名表中选择一条策略。")
            return None

        strategy_name = str(
            row.get("meta_strategy_name")
            or row.get("strategy_type")
            or "").strip()
        display_name = str(row.get("strategy", "未命名策略"))
        applied = {"strategy_name": strategy_name, "strategy": display_name}

        if strategy_name == "close_to_close" or display_name == "每日收盘":
            strategy_name = "close_to_close"
            self._strategy_var.set(STRATEGY_DISPLAY[strategy_name])
        elif strategy_name == "fixed_times" or display_name.startswith("固定时刻("):
            strategy_name = "fixed_times"
            fixed_times = row.get("meta_fixed_times")
            if not isinstance(fixed_times, str) or not fixed_times.strip():
                if "(" in display_name and display_name.endswith(")"):
                    fixed_times = display_name.split("(", 1)[1][:-1]
                else:
                    raise ValueError("历史结果缺少固定时刻参数。")
            fixed_times = fixed_times.strip()
            FixedTimeStrategy(fixed_times)
            self._fixed_times_var.set(fixed_times)
            self._strategy_var.set(STRATEGY_DISPLAY[strategy_name])
            applied["fixed_times"] = fixed_times
        elif strategy_name == "hedge_band" or display_name.startswith("固定间隔("):
            strategy_name = "hedge_band"
            candidate_sigma = self._comparison_finite(
                row.get("meta_candidate_sigma"))
            if candidate_sigma is None or candidate_sigma <= 0:
                raise ValueError("历史结果缺少有效的固定间隔 σ 参数。")
            sigma_source = str(
                row.get("meta_sigma_source") or "implied").strip()
            if sigma_source not in SIGMA_SOURCE_DISPLAY:
                sigma_source = "implied"
            sigma_window = self._comparison_safe_int(
                row.get("meta_sigma_window"), 20)
            sigma_window = max(2, sigma_window)
            # 三个带宽口径必须恒等价，所以这一组写入要么全成、要么全不动。
            # strict 校验（S0 或 sigma 非法时抛错）发生在写入**之后**：此前
            # 失败会留下「σ 已改成候选值、绝对/相对还是旧值」的中间态，三者
            # 不再等价。配合按钮那边异常无人接管，用户看到的就是「点了没反
            # 应」外加一份自相矛盾的带宽。
            restore = {
                "abs": self._band_abs_var.get(),
                "rel": self._band_rel_var.get(),
                "sigma": self._band_sigma_var.get(),
                "interval": self._price_interval_var.get(),
                "interval_type": self._interval_type_var.get(),
                "sigma_src": self._sigma_src_var.get(),
                "sigma_win": self._sigma_win_var.get(),
                "last_edited": self._band_last_edited,
            }
            self._sigma_src_var.set(SIGMA_SOURCE_DISPLAY[sigma_source])
            self._sigma_win_var.set(str(sigma_window))
            self._band_sigma_var.set(f"{candidate_sigma:.10g}")
            self._mark_band_edited("sigma")
            try:
                self._sync_band_inputs("sigma", strict=True)
            except Exception:
                self._band_abs_var.set(restore["abs"])
                self._band_rel_var.set(restore["rel"])
                self._band_sigma_var.set(restore["sigma"])
                self._price_interval_var.set(restore["interval"])
                self._interval_type_var.set(restore["interval_type"])
                self._sigma_src_var.set(restore["sigma_src"])
                self._sigma_win_var.set(restore["sigma_win"])
                self._band_last_edited = restore["last_edited"]
                # 上面的 ``_mark_band_edited`` 顺手把历史页那个勾选项改写成
                # 了「编辑完成后自动换算」，它读的是同一份带宽状态却不在
                # ``restore`` 里。数值都退回去之后若不重算这一行，用户看到的
                # 就是「带宽还是旧值、标签却说正在等换算」——一个永远不会到
                # 来的换算，除非他再手动碰一次带宽输入框。
                self._refresh_history_current_band_label()
                raise
            self._strategy_var.set(STRATEGY_DISPLAY[strategy_name])
            applied.update({
                "candidate_sigma": candidate_sigma,
                "sigma_source": sigma_source,
                "sigma_window": sigma_window,
            })
        else:
            raise ValueError(f"无法识别历史候选策略：{display_name}")

        fallback_value = row.get("meta_force_day_close_hedge")
        if not isinstance(fallback_value, (bool, np.bool_)):
            history_state = (
                getattr(self, "_latest_history_state", None) or {})
            fallback_value = history_state.get(
                "force_day_close_hedge", False)
        force_day_close_hedge = BacktestApp._comparison_safe_bool(
            fallback_value, False)
        fallback_var = getattr(
            self, "_force_day_close_hedge_var", None)
        if fallback_var is not None:
            fallback_var.set(force_day_close_hedge)
        applied["force_day_close_hedge"] = force_day_close_hedge
        applied["strategy_name"] = strategy_name
        self._toggle_strategy()
        self._set_status(f"已应用历史候选『{display_name}』到单次回测参数")
        if navigate:
            self._nb.select(self._summary_tab)
        return applied

    def _show_results(self, bt, multi_stats=None):
        self._show_summary(bt, multi_stats)
        self._show_chart(bt)
        self._show_vol_chart(bt)
        self._show_dist_chart(multi_stats)
        self._show_table(bt)
        self._nb.select(0)

    @staticmethod
    def _result_greek_series(result, greek_name):
        """优先返回已按经济头寸签名的 Greek，兼容旧回测原始字段。"""
        portfolio_key = f"portfolio_{greek_name}"
        key = portfolio_key if portfolio_key in result else greek_name
        return np.asarray(result[key], dtype=float)

    def _show_summary(self, bt, multi_stats=None):
        self._hide_placeholder("summary")
        self._summary_text.pack(fill="both", expand=True, padx=8, pady=8)

        r = bt._results
        n = r['n_days']
        meta = bt._gui_meta
        strategy = getattr(bt, "strategy", None)
        strategy_name = getattr(strategy, "name", "unknown")
        if strategy_name == "close_to_close":
            strategy_lines = [
                "  对冲策略          :  close_to_close",
                "  触发方式          :  每交易日最后一个采样点",
            ]
        elif strategy_name == "fixed_times":
            requested_times = getattr(
                strategy, "requested_times", getattr(strategy, "times", ()))
            effective_times = getattr(
                strategy, "effective_times", getattr(strategy, "times", ()))
            skipped_times = getattr(strategy, "skipped_times", ())
            requested_text = ",".join(
                t.strftime("%H:%M") for t in requested_times) or "—"
            effective_text = ",".join(
                t.strftime("%H:%M") for t in effective_times) or "—"
            skipped_text = ",".join(
                t.strftime("%H:%M") for t in skipped_times) or "—"
            strategy_lines = [
                "  对冲策略          :  fixed_times",
                f"  请求时刻          :  {requested_text}",
                f"  实际生效时刻      :  {effective_text}",
                f"  已跳过（休市）    :  {skipped_text}",
            ]
            if not effective_times:
                strategy_lines.append(
                    "  执行提示          :  所选时刻全部休市，本策略没有固定时刻触发")
        elif strategy_name == "hedge_band":
            band_type = getattr(strategy, "band_type", "relative")
            threshold = float(getattr(strategy, "threshold", float("nan")))
            unit_names = {
                "absolute": "绝对价格", "relative": "相对价格", "sigma": "日波动率倍数",
            }
            strategy_lines = [
                "  对冲策略          :  hedge_band",
                f"  输入单位          :  {unit_names.get(band_type, band_type)}",
                f"  回测带宽          :  {threshold:>12.6g}",
            ]
            try:
                converted = HedgeBandStrategy.convert_threshold(
                    threshold, band_type, float(r["prices"][0]),
                    float(r["implied_vol"]),
                )
                strategy_lines.extend([
                    f"  期初等价绝对/相对 :  {converted['absolute']:.6g} / {converted['relative']:.6g}",
                    f"  期初等价日波动率倍数 :  {converted['sigma']:.6g}",
                ])
            except (TypeError, ValueError):
                pass
            sigma_strategy = getattr(strategy, "_sigma_strategy", None)
            sigma_src = getattr(sigma_strategy, "sigma_source", "implied")
            if band_type == "sigma":
                strategy_lines.append(f"  波动率来源        :  {sigma_src}")
                if sigma_src == "realized":
                    strategy_lines.append(
                        f"  历史波动率回看天数:  {getattr(sigma_strategy, 'window_days', 20)}")
        else:
            strategy_lines = [f"  对冲策略          :  {strategy_name}"]

        fallback_enabled = bool(r.get(
            "force_day_close_hedge",
            getattr(bt, "force_day_close_hedge", False),
        ))
        fallback_marks = np.asarray(
            r.get("day_close_fallback_triggered", []), dtype=bool)
        fallback_count = int(np.count_nonzero(fallback_marks))
        if fallback_enabled:
            strategy_lines.append(
                "  每日收盘保底      :  开启"
                f"（实际补触发 {fallback_count} 次；同采样点自动去重）")
        else:
            strategy_lines.append("  每日收盘保底      :  关闭")

        # ---- 使用富文本 tag 插入, 实现分段上色 ----
        tw = self._summary_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")

        def _ins(text, tag=None):
            tw.insert("end", text, (tag,) if tag else ())

        def _sep(char="─", width=54):
            _ins(char * width + "\n", "separator")

        def _section(title):
            _ins(f"  {title}\n", "section")

        def _kv(label_text, value_text, val_tag=None):
            _ins(f"  {label_text}  :  ", "label")
            _ins(f"{value_text}\n", val_tag)

        def _val_color(v):
            """根据数值正负返回 tag 名."""
            try:
                return "value_pos" if float(v) >= 0 else "value_neg"
            except (ValueError, TypeError):
                return None

        _ins("═" * 54 + "\n", "separator")
        _ins("            动态对冲回测结果摘要\n", "header")
        _ins("═" * 54 + "\n", "separator")
        _ins("\n")

        _kv("期权类型        ", meta['cls_name'])
        _kv("子类型          ", SUBTYPE_DISPLAY.get(meta['subtype'], meta['subtype']))
        _kv("数据来源        ", meta['source'])
        if meta.get("source") == "wind":
            _kv(
                "Wind 数据区间   ",
                f"{meta.get('wind_start', '—')} 至 {meta.get('wind_end', '—')}",
            )
            _kv("行情采样粒度  ", meta.get("wind_bar_size", "—"))
        _ins("\n")
        _sep()

        _kv("回测天数        ", f"{n:>10d}")
        _kv("每日模拟采样点数", f"{bt.steps_per_day:>10d}")
        if r.get("knocked_out"):
            _kv("敲出了结        ", f"第 {r['ko_day']} 日敲出, 结算票息 {r['ko_settle']:.4f}",
                "value_pos")
        for sl in strategy_lines:
            _ins(sl + "\n", "label")
        _kv("交易成本率      ", f"{bt.tc_rate * 100:.2f}%")
        _kv("头寸方向        ", "卖出" if bt.position == 1 else "买入")
        _kv("交易数量        ", f"{bt.quantity:>12.2f}")
        _kv("合约乘数        ", f"{bt.multiplier if bt.multiplier > 0 else '不取整':>12}")
        _sep()

        _kv("标的初始价格    ", f"{r['prices'][0]:>12.4f}")
        _kv("标的到期价格    ", f"{r['prices'][-1]:>12.4f}")
        chg = (r['prices'][-1] / r['prices'][0] - 1) * 100
        _kv("标的涨跌幅      ", f"{chg:>11.2f}%", _val_color(chg))
        _sep()

        # 真实数据回测时附加期权要素伸缩明细
        rescale_info = getattr(bt, "_rescale_info", None)
        if rescale_info is not None:
            _section("【期权要素伸缩 (S_ref → S_real)】")
            _ins(
                f"  S_ref={rescale_info['s_ref']:.4f}   "
                f"S_real={rescale_info['s_real']:.4f}   "
                f"ratio={rescale_info['ratio']:.6f}\n", "label"
            )

            def _fmt_rescale(v):
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v, dtype=float).reshape(-1)
                    if arr.size > 6:
                        head = ", ".join(f"{x:.4f}" for x in arr[:3])
                        tail = ", ".join(f"{x:.4f}" for x in arr[-2:])
                        return f"[{head}, ..., {tail}]"
                    return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"
                return f"{float(v):.4f}"

            for name, (old, new) in rescale_info["fields"].items():
                if name in ("s0", "sr"):
                    continue
                _ins(f"    {name:<8s}: {_fmt_rescale(old):>12s}  →  {_fmt_rescale(new)}\n")
            _sep()

        _kv("期权初始价值    ", f"{r['opt_value'][0]:>12.4f}")
        _kv("期权到期价值    ", f"{r['opt_value'][-1]:>12.4f}")
        _sep()

        _section("【单路径盈亏分解（种子路径）】")
        hedge_pnl = np.sum(r['hedge_daily'])
        opt_pnl   = np.sum(r['option_daily'])
        _kv("标的对冲盈亏    ", f"{hedge_pnl:>12.4f}", _val_color(hedge_pnl))
        _kv("期权市值盈亏   ", f"{opt_pnl:>12.4f}", _val_color(opt_pnl))
        _kv("累计交易成本    ", f"{r['total_tc']:>12.4f}", "value_neg")
        _kv("对冲误差        ", f"{r['hedging_error']:>12.4f}",
            _val_color(r['hedging_error']))
        _sep()

        _section("【波动率分析】")
        _kv("成交隐含波动率  ", f"{r['implied_vol'] * 100:>11.2f}%")
        _kv("已实现波动率    ", f"{r['realized_vol'] * 100:>11.2f}%")
        vs = r['vol_spread'] * 100
        _kv("波动率价差      ", f"{vs:>11.2f}%  (正=卖方优势)", _val_color(vs))
        _sep()

        _section("【Greeks 统计】")
        _ins(f"  {'':15s}  {'初始值':>10s}  {'均值':>10s}  {'最大|值|':>10s}\n", "label")
        for gn in ("delta", "gamma", "vega", "theta", "rho"):
            gv = BacktestApp._result_greek_series(r, gn)
            _ins(f"  {gn.capitalize():15s}  {gv[0]:>10.4f}  {np.mean(gv):>10.4f}  {np.max(np.abs(gv)):>10.4f}\n")
        # 回测只在调仓 bar 重算 Greeks，非调仓 bar 沿用上一次的值。存在
        # 非调仓 bar 时必须点明，否则这几行均值会被误读成日内全采样。
        triggered = np.asarray(r.get("hedge_triggered", ()), dtype=bool)
        if triggered.size and not triggered.all():
            _ins("  注：Greeks 仅在调仓 bar 重算，非调仓 bar 沿用上一次的值，\n"
                 "      故以上均值为调仓时点采样，不是日内全采样。\n",
                 "label")
        _sep()

        rebal = int(np.sum(np.abs(np.diff(r['shares'])) > 1e-10))
        _kv("调仓次数        ", f"{rebal:>10d}")
        _ins("═" * 54 + "\n", "separator")

        # 蒙特卡洛多路径统计
        if multi_stats is not None:
            ms = multi_stats
            pnl_all = np.asarray(ms['total_pnl'], dtype=float)
            valid = np.isfinite(pnl_all)
            pnl = pnl_all[valid]
            n_p = ms['n_paths']
            n_valid = int(np.sum(valid))
            n_failed = len(ms.get("failed_paths", []))
            pct = lambda arr, q: np.percentile(arr, q)

            _ins("\n")
            _ins("═" * 54 + "\n", "separator")
            _ins(f"       蒙特卡洛多路径统计 ({n_p} 条路径)\n", "monte_header")
            _ins("═" * 54 + "\n", "separator")
            _ins("\n")
            if n_failed:
                _kv("成功路径        ", f"{n_valid:>10d}/{n_p}")
                _kv("失败路径        ", f"{n_failed:>10d}", "value_neg")
                _sep()
            if pnl.size == 0:
                _ins("  无可用成功路径，无法统计分布。\n", "value_neg")
                tw.configure(state="disabled")
                return

            _section("【总盈亏分布】")
            mean_pnl = np.mean(pnl)
            _kv("期望盈亏(均值)  ", f"{mean_pnl:>12.4f}", _val_color(mean_pnl))
            _kv("盈亏标准差      ", f"{np.std(pnl):>12.4f}")
            _kv("盈亏中位数      ", f"{np.median(pnl):>12.4f}",
                _val_color(np.median(pnl)))
            _kv("盈利概率        ", f"{np.mean(pnl > 0) * 100:>11.2f}%",
                "value_pos" if np.mean(pnl > 0) > 0.5 else "value_neg")
            _sep()

            _section("【分位数】")
            for q_val in (1, 5, 25, 75, 95, 99):
                pv = pct(pnl, q_val)
                _kv(f"{q_val}%{'':2s}分位        ",
                    f"{pv:>12.4f}", _val_color(pv))
            _sep()

            _kv("最大盈利        ", f"{np.max(pnl):>12.4f}", "value_pos")
            _kv("最大亏损        ", f"{np.min(pnl):>12.4f}", "value_neg")
            _sep()

            _section("【波动率】")
            rv = np.asarray(ms['realized_vols'], dtype=float)[valid]
            _kv("隐含波动率      ", f"{ms['implied_vol'] * 100:>11.2f}%")
            _kv("已实现波动率均值", f"{np.mean(rv) * 100:>11.2f}%")
            _kv("已实现波动率标准差", f"{np.std(rv) * 100:>11.2f}%")
            _sep()
            if meta.get("cls_name") == "雪球期权 (Snowball)" and "knocked_out" in ms:
                ko_flags = np.asarray(ms["knocked_out"], dtype=bool)[valid]
                _section("【雪球敲出】")
                _kv("敲出路径占比    ", f"{np.mean(ko_flags) * 100:>11.2f}%")
                if np.any(ko_flags) and "ko_days" in ms:
                    _kv("平均敲出日      ", f"{np.nanmean(np.asarray(ms['ko_days'], dtype=float)[valid]):>12.2f}")
                _sep()
            _kv("平均交易成本    ", f"{np.mean(np.asarray(ms['total_tc'], dtype=float)[valid]):>12.4f}", "value_neg")
            _ins("═" * 54 + "\n", "separator")

        tw.configure(state="disabled")

    def _show_chart(self, bt):
        # 隐藏占位 + 清除旧图表
        self._hide_placeholder("chart")
        self._reset_figure_container("chart", self._chart_container)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        r = bt._results
        n = len(r['prices'])

        # 判断是否有日期索引
        if hasattr(bt, '_wind_meta') and bt._wind_meta is not None:
            days = bt._wind_meta['dates']
            x_label = '日期'
        else:
            days = np.arange(n)
            x_label = '交易日'

        fig = Figure(figsize=self._container_figsize(self._chart_container, fallback=(10, 9)),
                     dpi=self._CHART_DPI)

        # (1) 标的价格
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(days, r['prices'], 'b-', linewidth=1.2)
        ax1.set_title('标的价格路径', fontsize=10)
        ax1.set_xlabel(x_label, fontsize=8)
        ax1.set_ylabel('价格', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=30, labelsize=7)

        # (2) Delta 数量与持仓
        ax2 = fig.add_subplot(3, 2, 2)
        delta_qty = r['delta'] * bt.quantity * bt.position
        ax2.plot(days, delta_qty, 'r-', label='Delta持仓目标', linewidth=1.2)
        ax2.plot(days, r['shares'], 'b--', label='实际持仓', linewidth=1.0, alpha=0.7)
        ax2.set_title('Delta持仓目标 与 实际持仓', fontsize=10)
        ax2.set_xlabel(x_label, fontsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=30, labelsize=7)

        # (3) Gamma
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(
            days, BacktestApp._result_greek_series(r, "gamma"),
            'm-', linewidth=1.2)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
        ax3.set_title('Gamma', fontsize=10)
        ax3.set_xlabel(x_label, fontsize=8)
        ax3.set_ylabel('Gamma', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=30, labelsize=7)

        # (4) Vega
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(
            days, BacktestApp._result_greek_series(r, "vega"),
            'c-', linewidth=1.2)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
        ax4.set_title('Vega', fontsize=10)
        ax4.set_xlabel(x_label, fontsize=8)
        ax4.set_ylabel('Vega', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=30, labelsize=7)

        # (5) Theta
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(
            days, BacktestApp._result_greek_series(r, "theta"),
            color='orange', linewidth=1.2)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
        ax5.set_title('Theta', fontsize=10)
        ax5.set_xlabel(x_label, fontsize=8)
        ax5.set_ylabel('Theta', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=30, labelsize=7)

        # (6) 累计盈亏
        ax6 = fig.add_subplot(3, 2, 6)
        cpnl = r['cumulative_pnl']
        ax6.plot(days, cpnl, 'k-', linewidth=1.2, label='累计盈亏')
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.fill_between(days, 0, cpnl, where=cpnl >= 0, alpha=0.15, color='green')
        ax6.fill_between(days, 0, cpnl, where=cpnl < 0, alpha=0.15, color='red')
        ax6.set_title('累计对冲盈亏', fontsize=10)
        ax6.set_xlabel(x_label, fontsize=8)
        ax6.set_ylabel('盈亏', fontsize=8)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=30, labelsize=7)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._chart_figure = fig
        self._chart_canvas = canvas

    def _show_vol_chart(self, bt):
        """波动率分析图表"""
        self._hide_placeholder("vol")
        self._reset_figure_container("vol", self._vol_container)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        r = bt._results
        n = len(r['prices'])

        if hasattr(bt, '_wind_meta') and bt._wind_meta is not None:
            days = bt._wind_meta['dates']
            x_label = '日期'
        else:
            days = np.arange(n)
            x_label = '交易日'

        implied = r['implied_vol']
        rolling = r['rolling_realized']
        cum_real = r['cumulative_realized']

        fig = Figure(figsize=self._container_figsize(self._vol_container),
                     dpi=self._CHART_DPI)

        # (1) 滚动已实现波动率 vs 隐含波动率
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axhline(y=implied * 100, color='red', linestyle='--', linewidth=1.2,
                     label=f'隐含波动率 {implied*100:.1f}%')
        ax1.plot(days, rolling * 100, 'b-', linewidth=1.2,
                 label='滚动已实现波动率(20d)')
        ax1.fill_between(days, implied * 100, rolling * 100,
                         where=rolling > implied, alpha=0.15, color='red',
                         label='已实现 > 隐含')
        ax1.fill_between(days, implied * 100, rolling * 100,
                         where=rolling <= implied, alpha=0.15, color='green',
                         label='已实现 ≤ 隐含')
        ax1.set_title('滚动已实现波动率 vs 隐含波动率', fontsize=10)
        ax1.set_xlabel(x_label, fontsize=8)
        ax1.set_ylabel('波动率 (%)', fontsize=8)
        ax1.legend(fontsize=7, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=30, labelsize=7)

        # (2) 累计已实现波动率 vs 隐含波动率
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axhline(y=implied * 100, color='red', linestyle='--', linewidth=1.2,
                     label=f'隐含波动率 {implied*100:.1f}%')
        ax2.plot(days, cum_real * 100, 'g-', linewidth=1.2,
                 label='累计已实现波动率')
        ax2.set_title('累计已实现波动率收敛', fontsize=10)
        ax2.set_xlabel(x_label, fontsize=8)
        ax2.set_ylabel('波动率 (%)', fontsize=8)
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=30, labelsize=7)

        # (3) 波动率价差 (implied - rolling realized)
        ax3 = fig.add_subplot(2, 2, 3)
        vol_diff = (implied - rolling) * 100
        ax3.plot(days, vol_diff, 'k-', linewidth=1.0)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.fill_between(days, 0, vol_diff,
                         where=vol_diff >= 0, alpha=0.2, color='green')
        ax3.fill_between(days, 0, vol_diff,
                         where=vol_diff < 0, alpha=0.2, color='red')
        ax3.set_title('波动率价差 (隐含 − 滚动已实现)', fontsize=10)
        ax3.set_xlabel(x_label, fontsize=8)
        ax3.set_ylabel('价差 (%)', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=30, labelsize=7)

        # (4) 日收益率分布
        ax4 = fig.add_subplot(2, 2, 4)
        # 0 或负价格在这里是已知输入，不是异常：下面显式滤掉非有限收益，
        # 因此不必让 numpy 再为除零/取负对数刷一屏 RuntimeWarning。
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ret = np.log(r['prices'][1:] / r['prices'][:-1])
        # 价格里出现 0 或负数时对数收益是 ±inf / nan（CSV 与 Wind 都只校验
        # 条数，不保证价格为正），matplotlib 会直接抛「range is not finite」，
        # 异常顺着 _show_results 把整页结果一起打掉——和分布页那个 bug 同源。
        finite_ret = log_ret[np.isfinite(log_ret)]
        dropped = int(log_ret.size - finite_ret.size)
        notes = []
        if dropped:
            notes.append(f"剔除 {dropped} 个非有限收益")
        from scipy.stats import norm
        if finite_ret.size:
            returns_pct = finite_ret * 100
            # 边界复用分箱兜底：收益率恒定时它已按量级展开，正态曲线的
            # x 轴也就不会塌成一个点。
            edges = self._histogram_edges(
                returns_pct, max(15, r['n_days'] // 3))
            ax4.hist(returns_pct, bins=edges, edgecolor='black', alpha=0.7,
                     color='steelblue', density=True)
            x_range = np.linspace(float(edges[0]), float(edges[-1]), 200)
            daily_impl = implied / np.sqrt(ANNUAL_DAYS) * 100
            if daily_impl > 0:
                ax4.plot(x_range, norm.pdf(x_range, 0, daily_impl), 'r--',
                         linewidth=1.2,
                         label=f'隐含波动率正态 σ={daily_impl:.2f}%')
            daily_real = float(np.std(returns_pct))
            # σ=0 时 norm.pdf 整条返回 nan：曲线其实画不出来，图例却仍写着
            # 「σ=0.00%」，看上去像"画了但压成一条线"。不画就不要留标签。
            if daily_real > 0:
                ax4.plot(x_range,
                         norm.pdf(x_range, float(np.mean(returns_pct)),
                                  daily_real),
                         'b-', linewidth=1.2, alpha=0.7,
                         label=f'已实现正态 σ={daily_real:.2f}%')
            else:
                notes.append("收益率恒定")
        else:
            notes.append("无有效收益率")
        ax4.set_title(
            '日收益率分布' + (f"（{'；'.join(notes)}）" if notes else ''),
            fontsize=10)
        ax4.set_xlabel('日收益率 (%)', fontsize=8)
        ax4.set_ylabel('密度', fontsize=8)
        if ax4.get_legend_handles_labels()[0]:
            ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._vol_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._vol_figure = fig
        self._vol_canvas = canvas

    # 分箱兜底与后端 plot_error_dist 共用同一份实现（退化样本会让 numpy
    # 拒绝整数分箱数，见 pricing.hedge_backtest.histogram_bin_edges）。
    _histogram_edges = staticmethod(histogram_bin_edges)
    _padded_histogram_range = staticmethod(padded_histogram_range)

    def _show_dist_chart(self, multi_stats):
        """蒙特卡洛盈亏分布图表"""
        self._hide_placeholder("dist")
        self._reset_figure_container("dist", self._dist_container)

        if multi_stats is None:
            # 使用唯一的动态空状态，避免与 Tab 初始 placeholder 重叠。
            hint = ttk.Frame(self._dist_container, style="Surface.TFrame")
            hint.place(relx=0.5, rely=0.45, anchor="center")
            tk.Label(hint, text="🎲", font=(_UI_FONT_FAMILY, 36),
                     bg=PALETTE["surface"]).pack(pady=(0, 8))
            tk.Label(hint, text="暂无分布数据",
                     font=(_UI_FONT_FAMILY, 14, "bold"),
                     bg=PALETTE["surface"], fg=PALETTE["text"]).pack(pady=(0, 4))
            tk.Label(hint,
                     text="盈亏分布仅在模拟数据模式下显示（需路径数 > 1）",
                     font=(_UI_FONT_FAMILY, 10),
                     bg=PALETTE["surface"], fg=PALETTE["text_muted"],
                     wraplength=340, justify="center").pack()
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        ms = multi_stats
        pnl_all = np.asarray(ms['total_pnl'], dtype=float)
        errors_all = np.asarray(ms['errors'], dtype=float)
        rv_all = np.asarray(ms['realized_vols'], dtype=float)
        valid = np.isfinite(pnl_all) & np.isfinite(errors_all) & np.isfinite(rv_all)
        if not np.any(valid):
            hint = ttk.Frame(self._dist_container, style="Surface.TFrame")
            hint.place(relx=0.5, rely=0.45, anchor="center")
            tk.Label(hint, text="无可用成功路径",
                     font=(_UI_FONT_FAMILY, 13),
                     bg=PALETTE["surface"], fg=PALETTE["text"]).pack()
            return

        pnl = pnl_all[valid]
        errors = errors_all[valid]
        rv = rv_all[valid]
        iv = ms['implied_vol']
        n_paths = len(pnl)

        fig = Figure(figsize=self._container_figsize(self._dist_container),
                     dpi=self._CHART_DPI)
        preferred_bins = max(30, n_paths // 15)

        # (1) 总盈亏分布直方图
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(pnl, bins=self._histogram_edges(pnl, preferred_bins),
                 edgecolor='black',
                 alpha=0.7, color='steelblue', density=False)
        ax1.axvline(np.mean(pnl), color='red', linestyle='--', linewidth=1.5,
                    label=f'均值={np.mean(pnl):.2f}')
        ax1.axvline(0, color='gray', linestyle='-', alpha=0.6)
        ax1.axvline(np.percentile(pnl, 5), color='orange', linestyle=':',
                    linewidth=1.2, label=f'5%VaR={np.percentile(pnl, 5):.2f}')
        ax1.set_title(f'总盈亏分布 ({n_paths}条路径)', fontsize=10)
        ax1.set_xlabel('总盈亏', fontsize=8)
        ax1.set_ylabel('频次', fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # (2) 对冲误差分布
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(errors, bins=self._histogram_edges(errors, preferred_bins),
                 edgecolor='black',
                 alpha=0.7, color='salmon', density=False)
        ax2.axvline(np.mean(errors), color='red', linestyle='--', linewidth=1.5,
                    label=f'均值={np.mean(errors):.2f}')
        ax2.axvline(0, color='gray', linestyle='-', alpha=0.6)
        ax2.set_title('对冲误差分布', fontsize=10)
        ax2.set_xlabel('对冲误差', fontsize=8)
        ax2.set_ylabel('频次', fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # (3) 已实现波动率分布 vs 隐含波动率
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(rv * 100,
                 bins=self._histogram_edges(rv * 100, preferred_bins),
                 edgecolor='black',
                 alpha=0.7, color='mediumpurple', density=False)
        ax3.axvline(iv * 100, color='red', linestyle='--', linewidth=1.5,
                    label=f'隐含波动率={iv*100:.1f}%')
        ax3.axvline(np.mean(rv) * 100, color='blue', linestyle='--', linewidth=1.2,
                    label=f'已实现均值={np.mean(rv)*100:.1f}%')
        ax3.set_title('已实现波动率分布', fontsize=10)
        ax3.set_xlabel('年化波动率 (%)', fontsize=8)
        ax3.set_ylabel('频次', fontsize=8)
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # (4) 盈亏 vs 已实现波动率散点图
        ax4 = fig.add_subplot(2, 2, 4)
        vol_spread = iv - rv
        ax4.scatter(vol_spread * 100, pnl, s=8, alpha=0.4, c='teal')
        # 线性拟合：波动率价差全等时样本里根本没有斜率信息，polyfit 的最小
        # 范数解却会给出一个具体数字（实测全等样本写出「拟合斜率=2.8e13」），
        # 画成一个点上的直线还配着这个标签。只有价差真的散开时才拟合。
        spread_range = float(np.ptp(vol_spread)) if len(vol_spread) else 0.0
        fitted = (len(vol_spread) > 2 and np.isfinite(spread_range)
                  and spread_range > 0)
        if fitted:
            z = np.polyfit(vol_spread * 100, pnl, 1)
            x_fit = np.linspace(np.min(vol_spread) * 100, np.max(vol_spread) * 100, 100)
            ax4.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=1.5,
                     label=f'拟合斜率={z[0]:.2f}')
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('盈亏 vs 波动率价差', fontsize=10)
        ax4.set_xlabel('波动率价差 (隐含−已实现) %', fontsize=8)
        ax4.set_ylabel('总盈亏', fontsize=8)
        # 拟合线是这张图唯一带标签的图元；没拟合就没有图例可画。
        if fitted:
            ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._dist_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._dist_figure = fig
        self._dist_canvas = canvas

    @staticmethod
    def _hedge_trigger_detail_frame(bt):
        """返回只含对冲触发事件的明细及其原始 Bar 位置。"""
        df = bt.to_dataframe()
        result = getattr(bt, "_results", None)
        if not isinstance(result, Mapping) or "hedge_triggered" not in result:
            raise ValueError("回测结果缺少 hedge_triggered，无法生成对冲触发明细。")

        triggered = np.asarray(result["hedge_triggered"], dtype=bool)
        if triggered.ndim != 1 or len(triggered) != len(df):
            raise ValueError(
                "回测触发标记与每日明细长度不一致："
                f"hedge_triggered={triggered.size}, 明细={len(df)}。")

        optional_masks = {}
        for key in ("strategy_hedge_triggered",
                    "day_close_fallback_triggered"):
            values = result.get(key)
            if values is None:
                optional_masks[key] = np.zeros(len(df), dtype=bool)
                continue
            mask = np.asarray(values, dtype=bool)
            if mask.ndim != 1 or len(mask) != len(df):
                raise ValueError(
                    f"回测触发来源 {key} 与每日明细长度不一致："
                    f"{mask.size} != {len(df)}。")
            optional_masks[key] = mask

        source_positions = np.flatnonzero(triggered)
        detail = df.iloc[source_positions].copy()
        strategy_mask = optional_masks["strategy_hedge_triggered"]
        fallback_mask = optional_masks["day_close_fallback_triggered"]
        last_position = len(df) - 1
        knocked_out = bool(result.get("knocked_out", False))

        sources = []
        for position in source_positions:
            if position == 0:
                source = "初始建仓"
            elif position == last_position:
                if knocked_out:
                    source = "敲出平仓"
                elif result.get("terminal_mode") == "mark_to_market":
                    source = "评价期末平仓"
                else:
                    source = "到期平仓"
            elif strategy_mask[position]:
                source = "策略触发"
            elif fallback_mask[position]:
                source = "收盘保底"
            else:
                source = "引擎触发"
            sources.append(source)
        detail.insert(0, "触发来源", sources)
        return detail, source_positions

    @staticmethod
    def _format_detail_index(value, *, include_time=False):
        """格式化表格索引；日内触发必须保留时分。"""
        if hasattr(value, "strftime"):
            pattern = "%Y-%m-%d %H:%M" if include_time else "%Y-%m-%d"
            try:
                return value.strftime(pattern)
            except (TypeError, ValueError):
                pass
        return str(value)

    def _show_table(self, bt):
        self._hide_placeholder("table")
        # table 没有单独 container；占位符是 tab 的直接子控件。保留其
        # Python/Tk 生命周期，仅清理上一次生成的 toolbar 与 Treeview。
        self._clear_tab_content_preserving_placeholder(
            "table", self._table_tab)

        df, source_positions = self._hedge_trigger_detail_frame(bt)
        result = getattr(bt, "_results", {}) or {}
        total_rows = len(np.asarray(result["hedge_triggered"]))
        is_intraday = (
            int(result.get("steps_per_day", 1) or 1) > 1
            or any(
                any(getattr(value, field, 0) != 0
                    for field in ("hour", "minute", "second"))
                for value in df.index
            )
        )

        # 顶部工具栏 (导出 + 行数提示 + 统计信息)
        toolbar = ttk.Frame(self._table_tab, style="Surface.TFrame")
        toolbar.pack(fill="x", padx=10, pady=(10, 6))

        info_frame = ttk.Frame(toolbar, style="Surface.TFrame")
        info_frame.pack(side="left", fill="x")
        tk.Label(info_frame, text="📃",
                 font=(_UI_FONT_FAMILY, 14),
                 bg=PALETTE["surface"]).pack(side="left", padx=(0, 6))
        tk.Label(info_frame,
                 text=(f"共 {len(df)} 条对冲触发记录"
                       f"（原始 {total_rows} {'个采样点' if is_intraday else '行'}）"),
                 font=(_UI_FONT_FAMILY, 10, "bold"),
                 bg=PALETTE["surface"], fg=PALETTE["text"]).pack(side="left")
        tk.Label(info_frame,
                 text="  ·  盈亏字段为触发采样点当期值",
                 font=(_UI_FONT_FAMILY, 9),
                 bg=PALETTE["surface"],
                 fg=PALETTE["text_muted"]).pack(side="left")

        ttk.Button(toolbar, text="💾 导出 CSV", style="Accent.TButton",
                   command=lambda: self._export_csv(df)).pack(side="right")

        # Treeview
        columns = list(df.columns)
        tree_frame = ttk.Frame(self._table_tab, style="Surface.TFrame")
        tree_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        yscroll = ttk.Scrollbar(tree_frame, orient="vertical")

        tree = ttk.Treeview(tree_frame, columns=["day_no", "idx"] + columns,
                            show="headings", height=20,
                            xscrollcommand=xscroll.set,
                            yscrollcommand=yscroll.set)
        xscroll.config(command=tree.xview)
        yscroll.config(command=tree.yview)

        tree.heading("day_no", text="原始采样点" if is_intraday else "交易日")
        tree.column("day_no", width=60, anchor="center")
        tree.heading("idx", text=df.index.name or "日期")
        tree.column("idx", width=142 if is_intraday else 96, anchor="center")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(
                col,
                width=92,
                anchor="center" if col == "触发来源" else "e",
            )

        # 斑马行
        tree.tag_configure("odd",  background=PALETTE["surface"])
        tree.tag_configure("even", background=PALETTE["surface_alt"])

        for display_i, ((idx, row), source_position) in enumerate(
                zip(df.iterrows(), source_positions)):
            idx_str = self._format_detail_index(
                idx, include_time=is_intraday)
            values = [int(source_position), idx_str] + [
                f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                for v in row.values
            ]
            tag = "even" if display_i % 2 == 0 else "odd"
            tree.insert("", "end", values=values, tags=(tag,))

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

    # ---- 结构分析 ----
    def _plot_structure(self):
        try:
            gui_state = self._collect_gui_state()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        try:
            range_pct = float(self._struct_range_var.get()) / 100.0
            n_points = int(self._struct_npts_var.get())
            if n_points < 5 or n_points > 201:
                raise ValueError("扫描点数需在 5~201")
            if range_pct <= 0 or range_pct >= 1:
                raise ValueError("扫描 ±% 需在 (0, 100)")
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        if not gui_state["s0"]:
            messagebox.showerror("参数错误", "请先在左侧设置初始价格 s0")
            return

        if not self._begin_job("structure", "正在进行结构扫描…"):
            return
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.configure(mode="determinate", maximum=n_points, value=0)
        self._progress_label.configure(text=f"结构扫描: 0/{n_points}")
        self._progress_label.pack(fill="x")
        self._nb.select(self._struct_tab)
        threading.Thread(target=self._structure_worker,
                         args=(gui_state, range_pct, n_points),
                         daemon=True).start()

    def _structure_worker(self, gui_state, range_pct, n_points):
        success = False
        try:
            cfg = gui_state["cfg"]
            subtype = gui_state["subtype"]
            base_params = dict(gui_state["params"])

            s0_center = float(gui_state["s0"])
            # 限制 MC 路径数以加速扫描
            if "nPath" in base_params and base_params["nPath"] > 20000:
                base_params["nPath"] = 20000

            s_grid = np.linspace(s0_center * (1 - range_pct),
                                 s0_center * (1 + range_pct), n_points)
            prices = np.empty(n_points)
            deltas = np.empty(n_points)
            gammas = np.empty(n_points)
            vegas  = np.empty(n_points)
            thetas = np.empty(n_points)

            for i, s in enumerate(s_grid):
                p = dict(base_params)
                p["s0"] = float(s)
                opt = cfg["build"](subtype, p)
                price = opt.get_price()
                prices[i] = price if price is not None else 0.0
                g = opt.get_greeks()
                deltas[i] = g[0]
                gammas[i] = g[1]
                vegas[i]  = g[2]
                thetas[i] = g[3]

                done = i + 1
                self.after(0, lambda d=done: (
                    self._progress.configure(value=d),
                    self._progress_label.configure(
                        text=f"结构扫描: {d}/{n_points}"),
                ))

            self.after(0, lambda: self._show_structure(
                gui_state, s_grid, prices, deltas, gammas, vegas, thetas))
            success = True
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(0, lambda: messagebox.showerror("结构分析失败", err_msg))
        finally:
            self.after(0, lambda ok=success: self._finish_structure(ok))

    def _finish_structure(self, success=True):
        self._finish_job(
            "structure", success=success,
            success_text="结构扫描完成",
            failure_text="结构扫描失败  |  请查看错误信息",
        )

    def _show_structure(self, gui_state, s_grid, prices, deltas,
                        gammas, vegas, thetas):
        self._hide_placeholder("struct")
        for w in self._struct_container.winfo_children():
            w.destroy()

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        cls_name = gui_state["cls_name"]
        subtype = gui_state["subtype"]
        params = gui_state["params"]

        # 与下方回测的头寸方向联动：position=1 卖出, position=-1 买入
        position = gui_state.get("position", 1)
        sign = -1.0 if position == 1 else 1.0
        perspective = "卖方" if position == 1 else "买方"
        prices = prices * sign
        deltas = deltas * sign
        gammas = gammas * sign
        vegas  = vegas  * sign
        thetas = thetas * sign

        doc_key = (cls_name, subtype)
        doc_text = STRUCTURE_DOCS.get(doc_key, "(暂无结构说明)")
        subtype_disp = SUBTYPE_DISPLAY.get(subtype, subtype)
        header = (f"{cls_name}  /  {subtype_disp}   [视角: {perspective}]\n"
                  + "─" * 60 + "\n")

        def _fmt(v):
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)

        param_summary = "  ".join(
            f"{k}={_fmt(v)}"
            for k, v in params.items()
            if k != "nPath"
        )
        full_text = header + doc_text + "\n\n参数: " + param_summary

        # 顶部文本
        text_frame = ttk.Frame(self._struct_container, style="Surface.TFrame")
        text_frame.pack(fill="x", padx=8, pady=(8, 4))
        text_widget = tk.Text(
            text_frame, wrap="word", height=11,
            font=(_MONO_FONT_FAMILY, 10),
            bg=PALETTE["surface_alt"],
            fg=PALETTE["text"],
            relief="flat", borderwidth=0,
            padx=12, pady=10,
            selectbackground=PALETTE["selected"],
        )
        text_widget.insert("1.0", full_text)
        text_widget.configure(state="disabled")
        text_widget.pack(fill="x")

        # 图表
        chart_frame = ttk.Frame(self._struct_container, style="Surface.TFrame")
        chart_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        fig = Figure(figsize=self._container_figsize(chart_frame, fallback=(10, 6.2)),
                     dpi=self._CHART_DPI)

        def _has(key):
            v = params.get(key)
            return v is not None and v != 0

        markers = []
        if _has("K"):  markers.append(("K", params["K"], "red"))
        if _has("H"):  markers.append(("H", params["H"], "orange"))
        if _has("KI"): markers.append(("KI", params["KI"], "purple"))
        if _has("E") and subtype == "EnhanceAsian":
            markers.append(("E", params["E"], "green"))
        if _has("P"):  markers.append(("P", params["P"], "brown"))

        def add_markers(ax):
            for _, val, color in markers:
                if s_grid[0] <= val <= s_grid[-1]:
                    ax.axvline(val, color=color, linestyle=":",
                               linewidth=1.0, alpha=0.7)

        def style(ax, title, ylabel):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("标的价格 S", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
            add_markers(ax)
            ax.tick_params(labelsize=7)

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(s_grid, prices, "b-", linewidth=1.4)
        style(ax1, "期权价格", "Price")

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(s_grid, deltas, "r-", linewidth=1.4)
        style(ax2, "Delta", "Δ")

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(s_grid, gammas, "m-", linewidth=1.4)
        style(ax3, "Gamma", "Γ")

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(s_grid, vegas, "c-", linewidth=1.4)
        style(ax4, "Vega", "ν")

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(s_grid, thetas, color="orange", linewidth=1.4)
        style(ax5, "Theta", "Θ")

        # 第 6 格：参考线图例与扫描信息
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis("off")
        legend_lines = ["参考线 (虚线):"]
        label_map = {"K": "行权", "H": "障碍", "KI": "敲入",
                     "E": "增强", "P": "保障"}
        for name, val, color in markers:
            legend_lines.append(f"  {name}={val:g}  ({label_map.get(name, '')})")
        legend_lines.append("")
        legend_lines.append(f"扫描范围: [{s_grid[0]:.2f}, {s_grid[-1]:.2f}]")
        legend_lines.append(f"采样点数: {len(s_grid)}")
        if "nPath" in params:
            legend_lines.append(f"MC路径数:  {min(params['nPath'], 20000)}")
            if params['nPath'] > 20000:
                legend_lines.append("(已限制为 20000 以加速)")
        if _MPL_CJK_FP is not None:
            ax6.text(0.02, 0.98, "\n".join(legend_lines),
                     transform=ax6.transAxes, va="top", ha="left",
                     fontsize=9, fontproperties=_MPL_CJK_FP)
        else:
            ax6.text(0.02, 0.98, "\n".join(legend_lines),
                     transform=ax6.transAxes, va="top", ha="left",
                     fontsize=9)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _export_csv(self, df):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, encoding="utf-8-sig")
            messagebox.showinfo("导出成功", f"已保存至:\n{path}")


# ============================================================
#  入口
# ============================================================

def main():
    app = BacktestApp()
    app.mainloop()


if __name__ == "__main__":
    main()
