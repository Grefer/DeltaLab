# _*_ coding: utf-8 _*_
"""
DeltaLab - 期权对冲回测 GUI 应用

基于 tkinter 构建，支持选择不同期权类型、回测方式（模拟/历史数据），
并以图表和表格形式展示回测结果。
"""

import sys
import os
import copy
import hashlib
import platform
import threading
import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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
)
from pricing.constants import ANNUAL_DAYS
from pricing.hedge_analysis import (
    HISTORY_SELECTION_METRIC,
    LOOKBACK_DAYS,
    STRICT_LOOKBACK_SELECTION_METRIC,
    recommend_by_contract_history_pool,
    recommend_by_rolling_history,
)
from pricing.hedge_backtest import (
    _infer_intraday_steps,
    _rescale_strategy_to_real_s0,
    _validate_fixed_time_data,
)

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
    """构造雪球（act=1 交易日计息）。观察日按锁定期+固定间隔生成（全交易日口径）；
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
    "Eu":                    "欧式 (Eu)",
    "Opt_Decumulator":       "普通累计 (Opt_Decumulator)",
    "Opt_Decumulator_Back":  "回归累计 (Opt_Decumulator_Back)",
    "Opt_Decumulator_Fix":   "固定赔付回归累计 (Opt_Decumulator_Fix)",
    "Opt_Decumulator_Fix_E": "固赔到期结算累计 (Opt_Decumulator_Fix_E)",
    "Opt_EnDecumulator":     "增强回归累计 (Opt_EnDecumulator)",
    "Opt_EnDecumulator_Fix": "固定赔付增强累计 (Opt_EnDecumulator_Fix)",
    "Opt_ASGQ_call_put":     "到期熔断保障累计 (Opt_ASGQ_call_put)",
    "Opt_ASGQ_EP":           "熔断每日保障累计 (Opt_ASGQ_EP)",
    "Opt_ASGQ_EF":           "熔断每日固赔累计 (Opt_ASGQ_EF)",
    "Opt_ASGQ_EFF":          "熔断每日双固赔累计 (Opt_ASGQ_EFF)",
    "Opt_ASGQ_DP":           "每日熔断保障累计 (Opt_ASGQ_DP)",
    "Opt_ASGQ_DF":           "每日熔断固赔累计 (Opt_ASGQ_DF)",
    "Opt_ASGQ_DFF":          "每日熔断双固赔累计 (Opt_ASGQ_DFF)",
    "Asian":                 "标准亚式 (Asian)",
    "EnhanceAsian":          "增强亚式 (EnhanceAsian)",
    "Opt_Airbag":            "气囊 (Opt_Airbag)",
    "Opt_Snowball":          "雪球 (Opt_Snowball)",
}
SUBTYPE_FROM_DISPLAY = {v: k for k, v in SUBTYPE_DISPLAY.items()}

STRATEGY_DISPLAY = {
    "close_to_close": "每日收盘 (close-to-close)",
    "fixed_times": "每日固定时刻",
    "hedge_band": "价格 / σ 带宽调仓",
}
STRATEGY_FROM_DISPLAY = {v: k for k, v in STRATEGY_DISPLAY.items()}

# 历史择优的固定间隔候选统一用“日波动 σ 倍数”表达和执行；用户当前的
# absolute / relative / sigma 输入只在参考点换算成 σ 后加入。这样同一组候选
# 可跨标的复用，也不会把三种局部等价表达当成三套策略重复搜索。
DEFAULT_BAND_CANDIDATE_SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0)
# 按同一交易日的业务顺序排列：前一自然日夜盘 -> 次日日盘。
DEFAULT_FIXED_TIMES = "23:00,11:30,15:00"
MAX_BAND_CANDIDATES = 10
MAX_HISTORY_CHART_CANDIDATES = 3

# 历史周期的唯一 GUI 顺序与中文标签。每个周期都表示截至分析日、严格
# 连续回放的最近 L 个交易日；交易日长度继续由后端常量维护。
HISTORY_PERIOD_DEFS = (
    ("week", "近周"),
    ("month", "近月"),
    ("quarter", "近季"),
    ("half_year", "近半年"),
    ("year", "近年"),
)

# Wind 仍需要明确的起止边界和 BarSize；GUI 用“自动（推荐）”把这组底层
# 参数按单次回测 / 历史择优语义解析后，再传给既有后端 API。
WIND_AUTO_BAR_SIZE = "自动（推荐）"
WIND_BAR_SIZE_OPTIONS = (
    WIND_AUTO_BAR_SIZE, "日频", "60min", "30min", "15min", "5min", "1min",
)
_WIND_BAR_MINUTES = {
    "60min": 60, "30min": 30, "15min": 15, "5min": 5, "1min": 1,
}
_WIND_DATE_BUFFER_DAYS = 21

HISTORY_CHART_MODE_DISPLAY = {
    "full": "完整 L 日累计路径",
    "single": "单代理段路径",
    "typical": "多代理段中位路径",
}
HISTORY_CHART_METRIC_DISPLAY = {
    "net": "净 PnL",
    "gross": "成本前 PnL",
    "tc": "成本",
}
HISTORY_CHART_MODE_FROM_DISPLAY = {
    value: key for key, value in HISTORY_CHART_MODE_DISPLAY.items()
}
HISTORY_CHART_METRIC_FROM_DISPLAY = {
    value: key for key, value in HISTORY_CHART_METRIC_DISPLAY.items()
}


@dataclass
class SavedBacktestResult:
    """会话内保留的轻量回测快照，只保存实时对比所需字段。"""

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
    contract_key: tuple
    economics_key: tuple
    position: int

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
        # 会话内第一个 close-to-close 快照固定为回测对比基准；只有删除它后
        # 才会按保留顺序接替下一条 close-to-close，避免基准随勾选结果漂移。
        self._saved_comparison_baseline_id = None
        self._saved_backtest_sequence = 0
        self._latest_backtest = None
        self._latest_backtest_state = None
        self._latest_retained_result_id = None
        self._latest_history_state = None
        self._latest_history_source_label = None
        self._pending_history_retain_name = None
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
        style.configure("TEntry",
                        fieldbackground=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["border"],
                        darkcolor=PALETTE["border"],
                        padding=4)
        style.map("TEntry",
                  bordercolor=[("focus", PALETTE["primary"])],
                  lightcolor=[("focus", PALETTE["primary"])])

        style.configure("TCombobox",
                        fieldbackground=PALETTE["surface"],
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        arrowcolor=PALETTE["text_muted"],
                        padding=3)
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
                  background=[("active", PALETTE["surface"])],
                  foreground=[("active", PALETTE["primary"])])

        style.configure("TCheckbutton",
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        font=base_font,
                        focuscolor=PALETTE["surface"])

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

        # 主体：左侧参数 + 右侧结果
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        # ─── 左侧面板 (整体包一层 Canvas + Scrollbar, 解决低分辨率/高 DPI 下底部按钮被裁) ───
        left_outer = ttk.Frame(body)
        body.add(left_outer, weight=1)

        # 外层 Canvas 横向充满, Scrollbar 靠右
        # width 仅作为 PanedWindow 初始 sash 位置的参考, 不阻止后续缩放
        self._left_canvas = tk.Canvas(
            left_outer, highlightthickness=0, bd=0,
            bg=PALETTE["surface"], width=440,
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

        # 鼠标滚轮处理: 只在鼠标进入左侧面板时启用, 离开则取消, 避免劫持右侧图表/表格的滚轮.
        def _on_left_mousewheel(event):
            # Windows: event.delta 为 ±120 的倍数; 正值向上滚, 负值向下滚.
            self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def _on_left_mousewheel_linux(event):
            units = -3 if event.num == 4 else 3
            self._left_canvas.yview_scroll(units, "units")
            return "break"

        def _bind_left_wheel(_event=None):
            if _SYSTEM == "Linux":
                self._left_canvas.bind_all("<Button-4>", _on_left_mousewheel_linux)
                self._left_canvas.bind_all("<Button-5>", _on_left_mousewheel_linux)
            else:
                self._left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)

        def _unbind_left_wheel(_event=None):
            if _SYSTEM == "Linux":
                self._left_canvas.unbind_all("<Button-4>")
                self._left_canvas.unbind_all("<Button-5>")
            else:
                self._left_canvas.unbind_all("<MouseWheel>")

        # 保留引用, 其它地方如需手动启停滚轮可复用
        self._left_wheel_bind = _bind_left_wheel
        self._left_wheel_unbind = _unbind_left_wheel

        self._left_canvas.bind("<Enter>", _bind_left_wheel)
        self._left_canvas.bind("<Leave>", _unbind_left_wheel)
        self._left_inner.bind("<Enter>", _bind_left_wheel)
        self._left_inner.bind("<Leave>", _unbind_left_wheel)

        # 之后所有原本放在 left 里的控件, 父容器改为 left_inner
        left = self._left_inner

        # 1) 期权大类
        sec1 = ttk.LabelFrame(left, text=" 期权类型 ", padding=12)
        sec1.pack(fill="x", pady=(0, 8))

        ttk.Label(sec1, text="大类:", style="Surface.TLabel").grid(
            row=0, column=0, sticky="w", pady=4)
        self._class_var = tk.StringVar()
        class_cb = ttk.Combobox(sec1, textvariable=self._class_var, width=25,
                                values=list(OPTION_CLASSES.keys()), state="readonly")
        class_cb.grid(row=0, column=1, padx=(8, 0), pady=4, sticky="ew")
        class_cb.current(0)
        class_cb.bind("<<ComboboxSelected>>", self._on_option_class_change)

        ttk.Label(sec1, text="子类型:", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=4)
        self._subtype_var = tk.StringVar()
        self._subtype_cb = ttk.Combobox(sec1, textvariable=self._subtype_var,
                                        width=25, state="readonly")
        self._subtype_cb.grid(row=1, column=1, padx=(8, 0), pady=4, sticky="ew")
        self._subtype_cb.bind(
            "<<ComboboxSelected>>", lambda _event: self._schedule_band_reference_sync())
        sec1.columnconfigure(1, weight=1)

        # 2) 期权参数
        # 说明: 外层左侧已经有整体 Canvas+Scrollbar, 这里不再嵌套独立滚动容器,
        # 让参数区按内容自然撑开高度, 整体滚动由外层统一处理.
        sec2 = ttk.LabelFrame(left, text=" 期权参数 ", padding=12)
        sec2.pack(fill="x", pady=(0, 8))

        self._param_frame = ttk.Frame(sec2, style="Surface.TFrame")
        self._param_frame.pack(fill="x", expand=True)

        # 3) 回测设置
        sec3 = ttk.LabelFrame(left, text=" 回测设置 ", padding=12)
        sec3.pack(fill="x", pady=(0, 8))

        ttk.Label(sec3, text="数据来源:", style="Surface.TLabel").grid(
            row=0, column=0, sticky="w", pady=4)
        self._source_var = tk.StringVar(value="simulate")
        src_frame = ttk.Frame(sec3, style="Surface.TFrame")
        src_frame.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(src_frame, text="模拟", variable=self._source_var,
                        value="simulate", command=self._toggle_source).pack(side="left", padx=(0, 6))
        ttk.Radiobutton(src_frame, text="CSV", variable=self._source_var,
                        value="csv", command=self._toggle_source).pack(side="left", padx=6)
        ttk.Radiobutton(src_frame, text="Wind", variable=self._source_var,
                        value="wind", command=self._toggle_source).pack(side="left", padx=6)

        # 模拟参数
        self._sim_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sim_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        ttk.Label(self._sim_frame, text="种子:", style="Surface.TLabel").grid(
            row=0, column=0, sticky="w", pady=2)
        self._seed_var = tk.StringVar(value="42")
        ttk.Entry(self._sim_frame, textvariable=self._seed_var, width=10).grid(
            row=0, column=1, padx=(6, 0), pady=2, sticky="w")
        ttk.Label(self._sim_frame, text="已实现波动率:", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=2)
        self._real_vol_var = tk.StringVar(value="")
        rv_frame = ttk.Frame(self._sim_frame, style="Surface.TFrame")
        rv_frame.grid(row=1, column=1, columnspan=3, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(rv_frame, textvariable=self._real_vol_var, width=10).pack(side="left")
        ttk.Label(rv_frame, text=" 空=同隐含", style="SurfaceMuted.TLabel").pack(side="left")
        ttk.Label(self._sim_frame, text="回测路径数 (MC):", style="Surface.TLabel").grid(
            row=2, column=0, sticky="w", pady=2)
        self._npaths_var = tk.StringVar(value="10")
        ttk.Entry(self._sim_frame, textvariable=self._npaths_var, width=10).grid(
            row=2, column=1, padx=(6, 0), pady=2, sticky="w")
        ttk.Label(self._sim_frame, text="模拟采样 bar/日:",
                  style="Surface.TLabel").grid(
            row=3, column=0, sticky="w", pady=2)
        sim_spd_frame = ttk.Frame(self._sim_frame, style="Surface.TFrame")
        sim_spd_frame.grid(row=3, column=1, columnspan=3,
                           sticky="w", padx=(6, 0), pady=2)
        self._spd_var = tk.StringVar(value="1")
        self._spd_combo = ttk.Combobox(
            sim_spd_frame, textvariable=self._spd_var, width=6,
            values=("1", "4", "48", "240"), state="readonly")
        self._spd_combo.pack(side="left")
        ttk.Label(
            sim_spd_frame,
            text=" 每个交易日等分为 1 / 4 / 48 / 240 个采样点",
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(6, 0))

        # CSV 参数
        self._csv_frame = ttk.Frame(sec3, style="Surface.TFrame")
        ttk.Label(self._csv_frame, text="文件:", style="Surface.TLabel").grid(
            row=0, column=0, sticky="w", pady=2)
        self._csv_path_var = tk.StringVar()
        ttk.Entry(self._csv_frame, textvariable=self._csv_path_var, width=22).grid(
            row=0, column=1, padx=(6, 4), pady=2)
        ttk.Button(self._csv_frame, text="浏览…", width=6,
                   command=self._browse_csv).grid(row=0, column=2, pady=2)
        ttk.Label(self._csv_frame, text="价格列:", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=2)
        self._csv_col_var = tk.StringVar(value="close")
        ttk.Entry(self._csv_frame, textvariable=self._csv_col_var, width=12).grid(
            row=1, column=1, padx=(6, 0), pady=2, sticky="w")

        # Wind 参数
        self._wind_frame = ttk.Frame(sec3, style="Surface.TFrame")
        ttk.Label(self._wind_frame, text="代码:", style="Surface.TLabel").grid(
            row=0, column=0, sticky="w", pady=2)
        self._wind_code_var = tk.StringVar(value="510050.SH")
        ttk.Entry(self._wind_frame, textvariable=self._wind_code_var, width=15).grid(
            row=0, column=1, padx=(6, 0), pady=2)
        ttk.Label(self._wind_frame, text="建仓日:", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=2)
        _today = datetime.date.today()
        _wind_start_default = (_today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        _wind_end_default = _today.strftime("%Y-%m-%d")
        self._wind_start_var = tk.StringVar(value=_wind_start_default)
        ttk.Entry(self._wind_frame, textvariable=self._wind_start_var, width=15).grid(
            row=1, column=1, padx=(6, 8), pady=2)

        self._wind_auto_end_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._wind_frame, text="结束日按期限自动补足",
            variable=self._wind_auto_end_var,
            command=self._toggle_wind_auto_end,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(self._wind_frame, text="自定义结束日:",
                  style="Surface.TLabel").grid(
            row=2, column=2, sticky="w", pady=2)
        self._wind_end_var = tk.StringVar(value=_wind_end_default)
        self._wind_end_entry = ttk.Entry(
            self._wind_frame, textvariable=self._wind_end_var, width=15)
        self._wind_end_entry.grid(row=2, column=3, padx=(6, 0), pady=2)

        # Wind 的请求 BarSize 同时也是策略观察粒度，不再与自动推导的
        # steps_per_day / 每日 Bar 数混为一个概念。
        ttk.Label(self._wind_frame, text="行情采样粒度:",
                  style="Surface.TLabel").grid(
            row=3, column=0, sticky="w", pady=2)
        self._wind_bar_size_var = tk.StringVar(value=WIND_AUTO_BAR_SIZE)
        self._wind_bar_size_combo = ttk.Combobox(
            self._wind_frame, textvariable=self._wind_bar_size_var, width=13,
            values=WIND_BAR_SIZE_OPTIONS,
            state="readonly",
        )
        self._wind_bar_size_combo.grid(row=3, column=1, padx=(6, 8),
                                       pady=2, sticky="w")
        self._wind_bar_size_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_wind_frequency_hint(),
        )
        self._wind_frequency_hint_var = tk.StringVar()
        ttk.Label(
            self._wind_frame, textvariable=self._wind_frequency_hint_var,
            style="SurfaceMuted.TLabel",
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=2)
        self._toggle_wind_auto_end()
        self._refresh_wind_frequency_hint()
        self._wind_code_var.trace_add(
            "write", lambda *_args: self._refresh_wind_frequency_hint())

        # 轻分割线
        ttk.Separator(sec3, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 6))

        # 对冲参数
        row_h = 3
        ttk.Label(sec3, text="交易成本率(%):", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._tc_var = tk.StringVar(value="0.01")
        ttk.Entry(sec3, textvariable=self._tc_var, width=10).grid(
            row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)

        row_h += 1
        ttk.Label(sec3, text="头寸方向:", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._pos_var = tk.StringVar(value="1")
        pos_frame = ttk.Frame(sec3, style="Surface.TFrame")
        pos_frame.grid(row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)
        ttk.Radiobutton(pos_frame, text="卖出 (short)", variable=self._pos_var,
                        value="1").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(pos_frame, text="买入 (long)", variable=self._pos_var,
                        value="-1").pack(side="left")

        row_h += 1
        ttk.Label(sec3, text="交易数量:", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._qty_var = tk.StringVar(value="100")
        ttk.Entry(sec3, textvariable=self._qty_var, width=12).grid(
            row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)

        row_h += 1
        ttk.Label(sec3, text="合约乘数:", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._mult_var = tk.StringVar(value="5")
        mult_frame = ttk.Frame(sec3, style="Surface.TFrame")
        mult_frame.grid(row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)
        ttk.Entry(mult_frame, textvariable=self._mult_var, width=10).pack(side="left")
        ttk.Label(mult_frame, text=" 0=不取整", style="SurfaceMuted.TLabel").pack(side="left")

        # 轻分割线：高级对冲参数（策略 / intraday / 滑点）
        row_h += 1
        ttk.Separator(sec3, orient="horizontal").grid(
            row=row_h, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        row_h += 1
        ttk.Label(sec3, text="对冲策略:", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._strategy_var = tk.StringVar(
            value=STRATEGY_DISPLAY["close_to_close"])
        strat_frame = ttk.Frame(sec3, style="Surface.TFrame")
        strat_frame.grid(row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)
        self._strategy_combo = ttk.Combobox(
            strat_frame, textvariable=self._strategy_var, width=16,
            values=tuple(STRATEGY_DISPLAY.values()), state="readonly",
        )
        self._strategy_combo.pack(side="left")
        self._strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_strategy())

        # sigma 单位下的波动率来源参数；带宽数值统一使用下方输入框。
        row_h += 1
        self._sigma_band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sigma_band_frame.grid(row=row_h, column=0, columnspan=2, sticky="ew",
                                    padx=(0, 0), pady=2)
        self._k_var = tk.StringVar(value="1")  # 旧状态字段兼容，不再单独展示
        ttk.Label(self._sigma_band_frame, text="σ 来源:",
                  style="Surface.TLabel").grid(row=0, column=0, sticky="w", pady=2)
        self._sigma_src_var = tk.StringVar(value=SIGMA_SOURCE_DISPLAY["implied"])
        ttk.Combobox(self._sigma_band_frame, textvariable=self._sigma_src_var, width=14,
                     values=tuple(SIGMA_SOURCE_DISPLAY.values()), state="readonly").grid(
            row=0, column=1, padx=(8, 0), pady=2, sticky="w")
        ttk.Label(self._sigma_band_frame, text="HV 回看期 (日):",
                  style="Surface.TLabel").grid(row=1, column=0, sticky="w", pady=2)
        self._sigma_win_var = tk.StringVar(value="20")
        ttk.Entry(self._sigma_band_frame, textvariable=self._sigma_win_var, width=6).grid(
            row=1, column=1, padx=(8, 0), pady=2, sticky="w")

        row_h += 1
        self._fixed_time_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._fixed_time_frame.grid(row=row_h, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(self._fixed_time_frame, text="固定时刻(HH:MM,逗号分隔):",
                  style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        self._fixed_times_var = tk.StringVar(value=DEFAULT_FIXED_TIMES)
        self._fixed_times_entry = ttk.Entry(
            self._fixed_time_frame, textvariable=self._fixed_times_var, width=24)
        self._fixed_times_entry.grid(row=0, column=1, padx=(8, 0), sticky="w")

        self._band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._band_frame.grid(row=row_h, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(self._band_frame, text="绝对间隔:",
                  style="Surface.TLabel").grid(row=0, column=0, sticky="w", pady=2)
        self._band_abs_var = tk.StringVar(value="1")
        self._band_abs_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_abs_var, width=12)
        self._band_abs_entry.grid(row=0, column=1, padx=(8, 18), sticky="w")
        ttk.Label(self._band_frame, text="相对间隔 (0.01=1%):",
                  style="Surface.TLabel").grid(row=0, column=2, sticky="w", pady=2)
        self._band_rel_var = tk.StringVar(value="0.01")
        self._band_rel_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_rel_var, width=12)
        self._band_rel_entry.grid(row=0, column=3, padx=(8, 0), sticky="w")
        ttk.Label(self._band_frame, text="日波动 σ 倍数:",
                  style="Surface.TLabel").grid(row=1, column=0, sticky="w", pady=2)
        self._band_sigma_var = tk.StringVar(value="0.779423")
        self._band_sigma_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_sigma_var, width=12)
        self._band_sigma_entry.grid(row=1, column=1, padx=(8, 18), sticky="w")
        ttk.Label(self._band_frame, text="编辑任一项后自动换算",
                  style="SurfaceMuted.TLabel").grid(
            row=1, column=2, columnspan=2, sticky="w", pady=2)

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

        row_h += 1
        close_fallback_frame = ttk.Frame(sec3, style="Surface.TFrame")
        close_fallback_frame.grid(
            row=row_h, column=0, columnspan=2, sticky="w", pady=3)
        self._force_day_close_hedge_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            close_fallback_frame, text="每日收盘兜底对冲",
            variable=self._force_day_close_hedge_var,
            command=self._refresh_history_base_summary,
        ).pack(side="left")
        ttk.Label(
            close_fallback_frame,
            text="非终止交易日末至少对齐一次；同 Bar 自动去重",
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(8, 0))

        row_h += 1
        ttk.Label(sec3, text="滑点 (bps):", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._slip_var = tk.StringVar(value="0")
        ttk.Entry(sec3, textvariable=self._slip_var, width=10).grid(
            row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)

        sec3.columnconfigure(1, weight=1)

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
        self._retain_btn.pack(fill="x", ipady=2, pady=(6, 0))

        self._compare_btn = ttk.Button(
            btn_frame, text="⚖  回测结果对比 (0)", style="Accent.TButton",
            command=self._open_saved_comparison,
        )
        self._compare_btn.pack(fill="x", ipady=2, pady=(6, 0))

        self._struct_btn = ttk.Button(btn_frame, text="📊  绘制结构图",
                                      style="Accent.TButton",
                                      command=self._plot_structure)
        self._struct_btn.pack(fill="x", ipady=2, pady=(6, 0))

        struct_ctrl = ttk.Frame(btn_frame)
        struct_ctrl.pack(fill="x", pady=(6, 0))
        ttk.Label(struct_ctrl, text="扫描 ±%:", style="Muted.TLabel").pack(side="left")
        self._struct_range_var = tk.StringVar(value="30")
        ttk.Entry(struct_ctrl, textvariable=self._struct_range_var, width=6).pack(
            side="left", padx=(4, 10))
        ttk.Label(struct_ctrl, text="点数:", style="Muted.TLabel").pack(side="left")
        self._struct_npts_var = tk.StringVar(value="31")
        ttk.Entry(struct_ctrl, textvariable=self._struct_npts_var, width=6).pack(
            side="left", padx=(4, 0))

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
             "回测摘要", "运行回测后此处将展示详细的盈亏、Greeks 和波动率统计"),
            ("compare", " ⚖ 回测结果对比 ",
             "已保存回测结果对比",
             "回测完成后点击『保留当前结果到对比』；换策略或参数继续回测，再到此页勾选结果"),
            ("history", " ◷ 历史择优 ",
             "历史择优",
             "使用 CSV / Wind 真实历史行情搜索已勾选周期的领先对冲方案"),
            ("chart",   " 📈 对冲图表 ",
             "对冲图表", "运行回测后此处将展示标的价格、Delta、Gamma、累计盈亏等图表"),
            ("vol",     " 📊 波动率分析 ",
             "波动率分析", "运行回测后此处将展示隐含波动率与已实现波动率的对比分析"),
            ("dist",    " 🎲 盈亏分布 ",
             "盈亏分布", "在模拟模式下运行多路径回测后，此处将展示蒙特卡洛盈亏分布"),
            ("struct",  " 🔬 结构分析 ",
             "结构分析", "点击左侧『绘制结构图』按钮以生成期权结构的 Greeks 曲线"),
            ("table",   " 📃 每日明细 ",
             "每日明细",
             "运行回测后此处将只展示触发对冲的建仓、策略、收盘兜底与终止记录"),
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
                "summary": "📋", "compare": "⚖", "history": "◷",
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

        # 底部状态栏
        status_bar = ttk.Frame(self, style="Surface.TFrame")
        status_bar.pack(fill="x", side="bottom")
        ttk.Separator(status_bar, orient="horizontal").pack(fill="x")
        self._status_var = tk.StringVar(
            value="就绪  |  选择期权类型、设置参数后点击『运行回测』")
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Status.TLabel", anchor="w").pack(fill="x", padx=10)

        self._toggle_source()

    def _build_history_workspace(self):
        """构建独立历史择优工作区，避免占用回测结果对比容器。"""
        container = self._history_container
        for widget in container.winfo_children():
            widget.destroy()

        header = ttk.Frame(container, style="Surface.TFrame")
        header.pack(fill="x", padx=8, pady=(8, 4))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="历史择优 · 独立批量实验",
            style="Surface.TLabel", font=(_UI_FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=("这里只配置候选空间；左侧行情、期权、头寸和成本会在任务启动时"
                  "冻结为本次实验基准。历史汇总不会直接混入单路径回测排名。"),
            style="SurfaceMuted.TLabel", wraplength=880, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._history_config_visible = True
        self._history_config_toggle_btn = ttk.Button(
            header, text="收起候选配置", width=12,
            command=self._toggle_history_config_panel,
        )
        self._history_config_toggle_btn.grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))

        self._history_config_panel = ttk.Frame(
            container, style="Surface.TFrame")
        self._history_config_panel.pack(fill="x", padx=8, pady=(0, 5))

        self._history_base_summary_var = tk.StringVar(value="正在读取当前基准配置…")
        base_box = tk.Frame(
            self._history_config_panel,
            bg=PALETTE["primary_light"], padx=8, pady=5)
        base_box.pack(fill="x", pady=(0, 5))
        tk.Label(
            base_box, textvariable=self._history_base_summary_var,
            bg=PALETTE["primary_light"], fg=PALETTE["primary"],
            font=(_UI_FONT_FAMILY, 9), anchor="w", justify="left",
            wraplength=920,
        ).pack(fill="x")

        settings = ttk.LabelFrame(
            self._history_config_panel,
            text=" 候选空间（仅属于历史择优） ", padding=7)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        self._history_include_close_var = tk.BooleanVar(value=True)
        self._history_include_fixed_times_var = tk.BooleanVar(value=True)
        self._history_include_band_var = tk.BooleanVar(value=True)
        candidate_bar = ttk.Frame(settings, style="Surface.TFrame")
        candidate_bar.grid(
            row=0, column=0, columnspan=4, sticky="ew", pady=(0, 5))
        ttk.Label(
            candidate_bar, text="参与策略:", style="Surface.TLabel",
        ).pack(side="left", padx=(0, 8))
        self._history_close_baseline_check = ttk.Checkbutton(
            candidate_bar, text="每日收盘（固定基准）",
            variable=self._history_include_close_var,
            state="disabled",
        )
        self._history_close_baseline_check.pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            candidate_bar, text="固定时刻",
            variable=self._history_include_fixed_times_var,
            command=self._toggle_history_candidate_controls,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            candidate_bar, text="固定间隔",
            variable=self._history_include_band_var,
            command=self._toggle_history_candidate_controls,
        ).pack(side="left")

        period_bar = ttk.Frame(settings, style="Surface.TFrame")
        period_bar.grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(0, 5))
        ttk.Label(
            period_bar, text="分析周期:", style="Surface.TLabel",
        ).pack(side="left", padx=(0, 8))
        self._history_period_vars = {}
        for period_key, period_label in HISTORY_PERIOD_DEFS:
            variable = tk.BooleanVar(value=True)
            self._history_period_vars[period_key] = variable
            ttk.Checkbutton(
                period_bar, text=period_label, variable=variable,
                command=self._history_period_selection_changed,
            ).pack(side="left", padx=(0, 10))
        ttk.Label(
            period_bar, text="可多选；每档严格连续回放最近 L 日",
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(2, 0))

        ttk.Label(
            settings, text="固定时刻候选:", style="Surface.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=3)
        self._history_fixed_times_var = tk.StringVar(value=DEFAULT_FIXED_TIMES)
        self._history_fixed_times_entry = ttk.Entry(
            settings, textvariable=self._history_fixed_times_var, width=24)
        self._history_fixed_times_entry.grid(
            row=2, column=1, sticky="w", padx=(8, 18), pady=3)

        ttk.Label(
            settings, text="固定间隔候选 (σ):", style="Surface.TLabel",
        ).grid(row=2, column=2, sticky="w", pady=3)
        self._history_band_candidate_sigmas_var = tk.StringVar(
            value=",".join(
                f"{value:g}" for value in DEFAULT_BAND_CANDIDATE_SIGMAS))
        self._history_band_candidate_entry = ttk.Entry(
            settings, textvariable=self._history_band_candidate_sigmas_var,
            width=25,
        )
        self._history_band_candidate_entry.grid(
            row=2, column=3, sticky="w", padx=(8, 0), pady=3)

        self._history_include_current_band_var = tk.BooleanVar(value=True)
        self._history_current_band_label_var = tk.StringVar(
            value="加入当前回测带宽")
        self._history_current_band_check = ttk.Checkbutton(
            settings, textvariable=self._history_current_band_label_var,
            variable=self._history_include_current_band_var,
        )
        self._history_current_band_check.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(
            settings,
            text="候选列表可留空；包含当前带宽在内最多 10 档",
            style="SurfaceMuted.TLabel",
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=3)

        sigma_frame = ttk.Frame(settings, style="Surface.TFrame")
        sigma_frame.grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(2, 0))
        ttk.Label(
            sigma_frame, text="历史候选 σ 来源:", style="Surface.TLabel",
        ).pack(side="left")
        self._history_sigma_src_var = tk.StringVar(
            value=SIGMA_SOURCE_DISPLAY["implied"])
        self._history_sigma_src_combo = ttk.Combobox(
            sigma_frame, textvariable=self._history_sigma_src_var, width=14,
            values=tuple(SIGMA_SOURCE_DISPLAY.values()), state="readonly",
        )
        self._history_sigma_src_combo.pack(side="left", padx=(7, 18))
        self._history_sigma_src_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._toggle_history_candidate_controls(),
        )
        ttk.Label(
            sigma_frame, text="HV 回看期 (日):", style="Surface.TLabel",
        ).pack(side="left")
        self._history_sigma_win_var = tk.StringVar(value="20")
        self._history_sigma_win_entry = ttk.Entry(
            sigma_frame, textvariable=self._history_sigma_win_var, width=7)
        self._history_sigma_win_entry.pack(side="left", padx=(7, 0))

        self._history_wind_frame = ttk.LabelFrame(
            settings, text=" Wind 严格历史区间（独立于单次回测） ", padding=5)
        self._history_wind_frame.grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        ttk.Label(
            self._history_wind_frame, text="分析截至日:",
            style="Surface.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=2)
        history_asof_default = (datetime.date.today() - datetime.timedelta(
            days=1)).isoformat()
        self._history_wind_asof_var = tk.StringVar(
            value=history_asof_default)
        self._history_wind_asof_entry = ttk.Entry(
            self._history_wind_frame,
            textvariable=self._history_wind_asof_var, width=13,
        )
        self._history_wind_asof_entry.grid(
            row=0, column=1, sticky="w", padx=(7, 18), pady=2)

        ttk.Label(
            self._history_wind_frame, text="统一行情粒度:",
            style="Surface.TLabel",
        ).grid(row=0, column=2, sticky="w", pady=2)
        self._history_wind_bar_size_var = tk.StringVar(
            value=WIND_AUTO_BAR_SIZE)
        self._history_wind_bar_size_combo = ttk.Combobox(
            self._history_wind_frame,
            textvariable=self._history_wind_bar_size_var,
            values=WIND_BAR_SIZE_OPTIONS, width=13, state="readonly",
        )
        self._history_wind_bar_size_combo.grid(
            row=0, column=3, sticky="w", padx=(7, 0), pady=2)
        self._history_wind_bar_size_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._toggle_history_wind_controls(),
        )

        self._history_wind_auto_start_var = tk.BooleanVar(value=True)
        self._history_wind_auto_start_label_var = tk.StringVar()
        self._history_wind_auto_start_check = ttk.Checkbutton(
            self._history_wind_frame,
            textvariable=self._history_wind_auto_start_label_var,
            variable=self._history_wind_auto_start_var,
            command=self._toggle_history_wind_controls,
        )
        self._history_wind_auto_start_check.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(
            self._history_wind_frame, text="自定义起始日:",
            style="Surface.TLabel",
        ).grid(row=1, column=2, sticky="w", pady=2)
        history_start_default = (
            datetime.date.today() - datetime.timedelta(days=420)
        ).isoformat()
        self._history_wind_start_var = tk.StringVar(
            value=history_start_default)
        self._history_wind_start_entry = ttk.Entry(
            self._history_wind_frame,
            textvariable=self._history_wind_start_var, width=13,
        )
        self._history_wind_start_entry.grid(
            row=1, column=3, sticky="w", padx=(7, 0), pady=2)
        self._history_wind_hint_var = tk.StringVar()
        ttk.Label(
            self._history_wind_frame,
            textvariable=self._history_wind_hint_var,
            style="SurfaceMuted.TLabel",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self._wind_code_var.trace_add(
            "write",
            lambda *_args: (
                BacktestApp._refresh_history_wind_hint(self),
                BacktestApp._refresh_history_base_summary(self),
            ),
        )

        actions = ttk.Frame(container, style="Surface.TFrame")
        actions.pack(fill="x", padx=8, pady=(0, 5))
        self._history_actions_frame = actions
        self._history_source_hint_var = tk.StringVar()
        ttk.Label(
            actions, textvariable=self._history_source_hint_var,
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        self._history_btn = ttk.Button(
            actions, text="开始历史择优", style="Run.TButton",
            command=self._run_history_recommendation,
        )
        self._history_btn.pack(side="right", ipadx=10, ipady=2)

        self._history_results_container = ttk.Frame(
            container, style="Surface.TFrame")
        self._history_results_container.pack(
            fill="both", expand=True, padx=6, pady=(0, 6))
        self._show_empty_history_results()
        self._history_period_selection_changed()
        self._toggle_history_candidate_controls()
        self._sync_history_button_state()

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
            before = getattr(self, "_history_actions_frame", None)
            pack_options = {
                "fill": "x", "padx": 8, "pady": (0, 5),
            }
            if before is not None:
                pack_options["before"] = before
            panel.pack(**pack_options)
            self._history_config_visible = True
            button.configure(text="收起候选配置")

    def _show_empty_history_results(self):
        container = getattr(self, "_history_results_container", None)
        if container is None:
            return
        for widget in container.winfo_children():
            widget.destroy()
        placeholder = ttk.Frame(container, style="Surface.TFrame")
        placeholder.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(
            placeholder, text="尚未运行历史择优", style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 14, "bold"),
        ).pack(pady=(0, 5))
        ttk.Label(
            placeholder,
            text=("选择 CSV 或 Wind 真实历史行情，设置候选空间后开始。\n"
                  "结果仅显示本次勾选周期的严格连续排名与区间诊断。"),
            style="SurfaceMuted.TLabel", justify="center",
        ).pack()

    @staticmethod
    def _normalize_history_lookbacks(selection=None):
        """按固定顺序把周期勾选规范化为 ``lookback -> 交易日``。"""
        known = {key for key, _label in HISTORY_PERIOD_DEFS}
        if selection is None:
            selected = known
            unknown = set()
        elif isinstance(selection, Mapping):
            unknown = set(selection).difference(known)
            selected = {
                key for key, enabled in selection.items() if bool(enabled)
            }
        else:
            if isinstance(selection, str):
                selection = (selection,)
            selected = set(selection)
            unknown = selected.difference(known)
        if unknown:
            raise ValueError(
                "未知历史分析周期: " + ", ".join(sorted(map(str, unknown))))
        ordered = {
            key: int(LOOKBACK_DAYS[key])
            for key, _label in HISTORY_PERIOD_DEFS if key in selected
        }
        if not ordered:
            raise ValueError("请至少选择一个历史分析周期。")
        return ordered

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

    def _history_period_selection_changed(self):
        """同步周期勾选对 Wind 自动范围与基准摘要的影响。"""
        lookbacks = BacktestApp._history_lookbacks_from_controls(
            self, require_one=False)
        auto_label = getattr(
            self, "_history_wind_auto_start_label_var", None)
        if auto_label is not None:
            if lookbacks:
                longest = max(lookbacks.values())
                auto_label.set(
                    f"起始日自动覆盖所选最长连续区间 {longest} 日 + "
                    "Day 0 锚点 + HV 预热（如需）")
            else:
                auto_label.set("请先勾选至少一个分析周期")
        if getattr(self, "_history_wind_hint_var", None) is not None:
            BacktestApp._refresh_history_wind_hint(self)
        if getattr(self, "_history_base_summary_var", None) is not None:
            BacktestApp._refresh_history_base_summary(self)

    def _toggle_history_candidate_controls(self):
        """只联动历史页自己的候选输入，不影响单次回测控件。"""
        fixed_enabled = bool(self._history_include_fixed_times_var.get())
        band_enabled = bool(self._history_include_band_var.get())
        sigma_source = SIGMA_SOURCE_FROM_DISPLAY.get(
            self._history_sigma_src_var.get(),
            self._history_sigma_src_var.get(),
        )
        self._history_fixed_times_entry.configure(
            state="normal" if fixed_enabled else "disabled")
        self._history_band_candidate_entry.configure(
            state="normal" if band_enabled else "disabled")
        self._history_current_band_check.configure(
            state="normal" if band_enabled else "disabled")
        self._history_sigma_src_combo.configure(
            state="readonly" if band_enabled else "disabled")
        self._history_sigma_win_entry.configure(
            state=("normal" if band_enabled and sigma_source == "realized"
                   else "disabled"))
        BacktestApp._toggle_history_wind_controls(self)

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
            bar_var = getattr(self, "_history_wind_bar_size_var", None)
            start_var = getattr(self, "_history_wind_start_var", None)
            end_var = getattr(self, "_history_wind_asof_var", None)
            auto_start_var = getattr(
                self, "_history_wind_auto_start_var", None)
            code = code_var.get().strip() if code_var is not None else "—"
            bar = bar_var.get().strip() if bar_var is not None else "—"
            if auto_start_var is not None and auto_start_var.get():
                lookbacks = BacktestApp._history_lookbacks_from_controls(
                    self, require_one=False)
                start = (
                    f"自动覆盖最近{max(lookbacks.values())}日连续区间"
                    if lookbacks else "未选择分析周期")
            else:
                start = start_var.get().strip() if start_var is not None else "—"
            end = end_var.get().strip() if end_var is not None else "—"
            source_text = (
                f"Wind · {code or '—'} · {start or '—'} 至 {end or '—'}"
                f" · {bar or '—'}")
        else:
            source_text = "模拟行情（历史择优不可用）"
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
            f"收盘兜底 {close_fallback}  ·  "
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
        """进入历史页时刷新将被冻结的公共环境与当前带宽摘要。"""
        try:
            if self._nb.select() != str(self._history_tab):
                return
        except (AttributeError, tk.TclError):
            return
        BacktestApp._toggle_history_wind_controls(self)
        self._refresh_history_base_summary()
        self._refresh_history_current_band_label()

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
            label_widget = ttk.Label(self._param_frame, text=f"{label}:",
                                     style="Surface.TLabel")
            label_widget.grid(
                row=i, column=0, sticky="w", padx=(2, 8), pady=3)
            if choices:
                val_to_display = {v: k for k, v in choices.items()}
                default_display = val_to_display.get(
                    default, str(default) if editable else next(iter(choices)))
                var = tk.StringVar(value=default_display)
                cb = ttk.Combobox(self._param_frame, textvariable=var,
                                  values=list(choices.keys()),
                                  state="normal" if editable else "readonly",
                                  width=14)
                cb.grid(row=i, column=1, sticky="ew", pady=3, padx=(0, 2))
                input_widget = cb
                if key == "margin_call":
                    cb.bind("<<ComboboxSelected>>",
                            lambda _event: self._sync_snowball_margin_controls())
            else:
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(self._param_frame, textvariable=var, width=16)
                entry.grid(row=i, column=1, sticky="ew", pady=3, padx=(0, 2))
                input_widget = entry
            self._param_entries[key] = (var, dtype, choices)
            self._param_widgets[key] = {
                "label": label_widget,
                "input": input_widget,
                "base_label": label,
                "choices": choices,
            }
        self._param_frame.columnconfigure(1, weight=1)
        self._sync_snowball_margin_controls()
        self._bind_band_reference_inputs()

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

    def _toggle_wind_auto_end(self):
        entry = getattr(self, "_wind_end_entry", None)
        variable = getattr(self, "_wind_auto_end_var", None)
        if entry is None or variable is None:
            return
        entry.configure(state="disabled" if variable.get() else "normal")

    def _refresh_wind_frequency_hint(self):
        hint = getattr(self, "_wind_frequency_hint_var", None)
        requested_var = getattr(self, "_wind_bar_size_var", None)
        strategy_var = getattr(self, "_strategy_var", None)
        if hint is None or requested_var is None or strategy_var is None:
            return
        requested = requested_var.get().strip() or WIND_AUTO_BAR_SIZE
        strategy_name = STRATEGY_FROM_DISPLAY.get(
            strategy_var.get(), strategy_var.get())
        fixed_var = getattr(self, "_fixed_times_var", None)
        fixed_times = fixed_var.get().strip() if fixed_var is not None else ""
        wind_code_var = getattr(self, "_wind_code_var", None)
        wind_code = (
            wind_code_var.get().strip() if wind_code_var is not None else "")
        try:
            actual = BacktestApp._resolve_wind_bar_size(
                requested, strategy_name=strategy_name,
                fixed_times=fixed_times, wind_code=wind_code)
        except ValueError as exc:
            hint.set(str(exc))
            return
        if requested == WIND_AUTO_BAR_SIZE:
            reason = {
                "close_to_close": "收盘策略",
                "fixed_times": "覆盖目标时刻",
                "hedge_band": "观察盘中穿带",
            }.get(strategy_name, "策略需要")
            hint.set(f"自动采用 {actual}（{reason}）")
        elif strategy_name == "hedge_band" and actual == "日频":
            hint.set("手动日频：固定间隔仅检查收盘穿带")
        else:
            hint.set(f"手动采用 {actual}")

    def _toggle_history_wind_controls(self):
        source_var = getattr(self, "_source_var", None)
        is_wind = source_var is not None and source_var.get() == "wind"
        auto_var = getattr(self, "_history_wind_auto_start_var", None)
        auto_start = bool(auto_var.get()) if auto_var is not None else True

        asof_entry = getattr(self, "_history_wind_asof_entry", None)
        if asof_entry is not None:
            asof_entry.configure(state="normal" if is_wind else "disabled")
        combo = getattr(self, "_history_wind_bar_size_combo", None)
        if combo is not None:
            combo.configure(state="readonly" if is_wind else "disabled")
        check = getattr(self, "_history_wind_auto_start_check", None)
        if check is not None:
            check.configure(state="normal" if is_wind else "disabled")
        start_entry = getattr(self, "_history_wind_start_entry", None)
        if start_entry is not None:
            start_entry.configure(
                state="normal" if is_wind and not auto_start else "disabled")
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

        requested_var = getattr(
            self, "_history_wind_bar_size_var", None)
        requested = (
            requested_var.get().strip()
            if requested_var is not None else WIND_AUTO_BAR_SIZE)
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
        try:
            actual = BacktestApp._resolve_wind_bar_size(
                requested, fixed_times=fixed_times,
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
            range_text = (
                f"自动覆盖最近 {max(lookbacks.values())} 日连续区间"
                " + Day 0 锚点")
        else:
            range_text = "使用自定义历史起始日"
        sigma_source_var = getattr(self, "_history_sigma_src_var", None)
        sigma_window_var = getattr(self, "_history_sigma_win_var", None)
        sigma_source = SIGMA_SOURCE_FROM_DISPLAY.get(
            sigma_source_var.get() if sigma_source_var is not None else "",
            sigma_source_var.get() if sigma_source_var is not None else "",
        )
        if include_band and sigma_source == "realized":
            try:
                warmup_days = int(
                    sigma_window_var.get() if sigma_window_var is not None else 0)
            except (TypeError, ValueError):
                warmup_days = 0
            range_text += f" + HV预热 {warmup_days} 日"
        prefix = "自动统一采用" if requested == WIND_AUTO_BAR_SIZE else "统一采用"
        hint.set(f"{prefix} {actual}；{range_text}")

    def _toggle_source(self):
        src = self._source_var.get()
        self._sim_frame.grid_remove()
        self._csv_frame.grid_remove()
        self._wind_frame.grid_remove()
        if src == "simulate":
            self._sim_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        elif src == "csv":
            self._csv_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        elif src == "wind":
            self._wind_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        BacktestApp._toggle_strategy(self)
        BacktestApp._toggle_wind_auto_end(self)
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
                    f"{ANNUAL_DAYS} 日口径）")
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
                "每日固定时刻策略不支持 Wind 日频；请选择分钟频率。")

    @staticmethod
    def _validate_history_recommendation_source(gs):
        """历史择优只能基于用户提供的 CSV 或 Wind 真实行情。"""
        source = gs.get("source")
        if source not in ("csv", "wind"):
            raise ValueError(
                "历史择优必须使用 CSV 或 Wind 真实历史行情；模拟路径不可用。"
                "如需比较模拟结果，请分别运行回测并保留到结果池。")

    def _sync_history_button_state(self):
        """使历史择优入口始终与真实数据来源及后台任务状态一致。"""
        source_var = getattr(self, "_source_var", None)
        if source_var is None:
            return
        enabled = (
            getattr(self, "_active_job", None) is None
            and source_var.get() in ("csv", "wind")
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
            if source_var.get() in ("csv", "wind"):
                hint.set("真实历史来源已就绪；运行时会冻结当前基准参数")
            else:
                hint.set("历史择优仅支持 CSV / Wind，模拟行情下不可运行")

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
                "_run_btn", "_retain_btn", "_compare_btn", "_history_btn",
                "_struct_btn"):
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
        """用 CSV/Wind 真实历史执行独立的候选策略择优。"""
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
            messagebox.showerror("历史择优不可用", str(exc))
            return False

        period_labels = {
            key: label for key, label in HISTORY_PERIOD_DEFS}
        selected_text = "、".join(
            period_labels[key] for key in selected_lookbacks)
        if not self._begin_job(
                "history",
                f"正在使用真实历史行情执行批量择优（{selected_text}）…"):
            return False
        self._progress.configure(mode="indeterminate")
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.start(15)
        threading.Thread(
            target=self._history_recommendation_worker,
            args=(history_state,), daemon=True,
        ).start()
        return True

    # 保留旧私有入口的软兼容，但 UI 与任务语义均已切换到 history。
    def _run_strategy_comparison(self):
        return self._run_history_recommendation()

    @staticmethod
    def _parse_band_candidate_sigmas(raw_values):
        """解析历史择优的日波动 σ 候选，并按显示精度去重排序。"""
        if raw_values is None:
            raw_values = DEFAULT_BAND_CANDIDATE_SIGMAS
        if isinstance(raw_values, str):
            normalized = raw_values.translate(str.maketrans({
                "，": ",", "；": ",", ";": ",", "\n": ",",
            }))
            tokens = [token.strip() for token in normalized.split(",")]
            tokens = [token for token in tokens if token]
        else:
            try:
                tokens = list(raw_values)
            except TypeError as exc:
                raise ValueError("历史择优带宽候选必须是逗号分隔数值。") from exc

        candidates_by_key = {}
        for token in tokens:
            try:
                value = float(token)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"无效的历史择优 σ 候选: {token!r}") from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError("历史择优 σ 候选必须全部是大于 0 的有限数值。")
            # 名称也使用 .10g，先用同一规范键去重可从源头保证 case.name 唯一。
            key = f"{value:.10g}"
            candidates_by_key[key] = float(key)
        if len(candidates_by_key) > MAX_BAND_CANDIDATES:
            raise ValueError(
                f"历史择优固定间隔候选最多 {MAX_BAND_CANDIDATES} 档。")
        return tuple(sorted(candidates_by_key.values()))

    @staticmethod
    def _band_cases_for_history(gs):
        """把历史页的 σ 档位与可选当前带宽归一化为候选案例。"""
        band_type = gs.get("interval_type", "absolute")
        threshold = float(gs.get("price_interval", 1.0))
        force_day_close_hedge = bool(
            gs.get("force_day_close_hedge", False))
        params = gs.get("params", {})
        reference_price = float(params["s0"])
        sigma_annual = float(params["sigma"])
        include_current = bool(
            gs.get("history_include_current_band", True))
        current_equivalents = None
        current_key = None
        if include_current:
            current_equivalents = HedgeBandStrategy.convert_threshold(
                threshold, band_type, reference_price, sigma_annual)
            current_key = f"{current_equivalents['sigma']:.10g}"

        preset_values = BacktestApp._parse_band_candidate_sigmas(
            gs.get("band_candidate_sigmas"))
        preset_keys = {f"{value:.10g}" for value in preset_values}
        candidates_by_key = {
            f"{value:.10g}": value for value in preset_values
        }
        # 当前单次回测带宽只在历史页明确勾选时参与。
        if current_key is not None:
            candidates_by_key[current_key] = float(current_key)
        if len(candidates_by_key) > MAX_BAND_CANDIDATES:
            raise ValueError(
                f"历史择优固定间隔候选最多 {MAX_BAND_CANDIDATES} 档"
                "（包含勾选加入的当前带宽）。")

        cases = []
        for candidate_sigma in sorted(candidates_by_key.values()):
            key = f"{candidate_sigma:.10g}"
            is_current = current_key is not None and key == current_key
            equivalents = HedgeBandStrategy.convert_threshold(
                candidate_sigma, "sigma", reference_price, sigma_annual)
            origin = (
                "当前输入 / 常用候选" if is_current and key in preset_keys
                else "当前输入换算" if is_current else "常用候选"
            )
            marker = "·当前" if is_current else ""
            description = (
                f"{origin}；期初等价绝对 {equivalents['absolute']:.6g} / "
                f"相对 {equivalents['relative']:.4%} / "
                f"{candidate_sigma:.6g}σ；收盘兜底"
                f"{'开启' if force_day_close_hedge else '关闭'}"
            )
            cases.append(StrategyCase(
                f"固定间隔({key}σ{marker})",
                HedgeBandStrategy(
                    band_type="sigma",
                    threshold=candidate_sigma,
                    sigma_source=gs.get("sigma_source", "implied"),
                    window_days=gs.get("sigma_window", 20),
                ),
                {
                    "description": description,
                    "strategy_name": "hedge_band",
                    "candidate_origin": origin,
                    "candidate_sigma": candidate_sigma,
                    "equivalent_absolute": equivalents["absolute"],
                    "equivalent_relative": equivalents["relative"],
                    "input_band_type": band_type,
                    "input_threshold": threshold,
                    "is_current_band": is_current,
                    "sigma_source": gs.get("sigma_source", "implied"),
                    "sigma_window": gs.get("sigma_window", 20),
                    "force_day_close_hedge": force_day_close_hedge,
                },
            ))
        names = [case.name for case in cases]
        if len(names) != len(set(names)):
            raise RuntimeError("历史择优固定间隔候选名称重复。")
        return cases

    @staticmethod
    def _band_cases_for_comparison(gs):
        """旧私有名称的软兼容入口。"""
        return BacktestApp._band_cases_for_history(gs)

    def _strategy_cases_for_history(self, gs, base_bt):
        """根据历史页独立开关生成候选策略集。"""
        # close-to-close 始终作为历史图表和严格区间改善的固定基准，不允许
        # 通过旧状态或直接 API 调用移除。
        cases = [StrategyCase(
            "每日收盘", CloseToCloseStrategy(),
            {
                "description": (
                    "按真实交易日的最后一根 bar 调仓；固定比较基准；收盘兜底"
                    f"{'开启' if gs.get('force_day_close_hedge', False) else '关闭'}"
                ),
                "strategy_name": "close_to_close",
                "is_history_baseline": True,
                "force_day_close_hedge": bool(
                    gs.get("force_day_close_hedge", False)),
            },
        )]
        if gs.get("history_include_band", True):
            cases.extend(BacktestApp._band_cases_for_history(gs))
        skipped = []
        if gs.get("history_include_fixed_times", True):
            try:
                if gs.get("source") == "simulate":
                    raise ValueError("模拟路径没有真实时刻")
                if (gs.get("source") == "wind" and
                        gs.get("wind_bar_size", "日频") == "日频"):
                    raise ValueError("Wind 日频没有日内时刻")
                fixed = FixedTimeStrategy(
                    gs.get("fixed_times", DEFAULT_FIXED_TIMES))
                if gs.get("source") == "wind":
                    from pricing.wind_data import (
                        get_trading_session_clock_ranges,
                    )
                    fixed.set_trading_sessions(
                        get_trading_session_clock_ranges(
                            gs.get("wind_code", "")))
                # 此处只确认候选具备真实日内时间轴。具体目标时刻是否在
                # 每个交易日出现，必须由严格历史区间中的代理段逐一判断；
                # base_bt 只提供粒度上下文，不能据此全局剔除。
                BacktestApp._validate_fixed_time_history_granularity(base_bt)
                requested_label = ",".join(
                    t.strftime("%H:%M") for t in fixed.requested_times)
                effective_label = ",".join(
                    t.strftime("%H:%M") for t in fixed.effective_times)
                skipped_label = ",".join(
                    t.strftime("%H:%M") for t in fixed.skipped_times)
                execution_text = (
                    f"每日 {effective_label} 调仓"
                    if effective_label else "所选时刻均不在该品种交易时段")
                if skipped_label:
                    execution_text += f"；自动跳过非交易时刻 {skipped_label}"
                cases.append(StrategyCase(
                    f"固定时刻({requested_label})", fixed,
                    {
                        "description": (
                            f"{execution_text}；收盘兜底"
                            f"{'开启' if gs.get('force_day_close_hedge', False) else '关闭'}"
                        ),
                        "strategy_name": "fixed_times",
                        "fixed_times": requested_label,
                        "fixed_times_requested": requested_label,
                        "fixed_times_effective": effective_label,
                        "fixed_times_skipped": skipped_label,
                        "force_day_close_hedge": bool(
                            gs.get("force_day_close_hedge", False)),
                    },
                ))
            except (TypeError, ValueError) as exc:
                skipped.append(f"固定时刻策略未参与：{exc}")
        if len(cases) == 1:
            reason = f"（{'；'.join(skipped)}）" if skipped else ""
            raise ValueError(
                "历史择优只有每日收盘 C2C 基准，没有可执行的候选策略"
                f"{reason}")
        return cases, skipped

    # 保留旧私有名称的软兼容；新代码使用 history 语义。
    def _strategy_cases_for_comparison(self, gs, base_bt):
        return self._strategy_cases_for_history(gs, base_bt)

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
        # 区间之前最多只为 Day 0 锚点和 realized-HV 预热留出必要历史，
        # 不能再因代理期限 T 把行情扩到证据区间之外。
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

    @staticmethod
    def _validate_history_recommendation_payload(
            recommendations, ranking, window_results):
        """确认成功结果确实包含至少一个基于真实历史的可评估代理段。"""
        if recommendations is None:
            raise ValueError("历史择优未返回历史参考结果表。")
        if ranking is None or getattr(ranking, "empty", True):
            raise ValueError("历史择优未返回诊断排名。")
        if not isinstance(window_results, dict) or not window_results:
            raise ValueError("历史择优未返回严格区间代理段明细。")

        try:
            import pandas as pd
            rolling_windows = pd.to_numeric(
                ranking["rolling_windows"], errors="coerce").to_numpy(dtype=float)
            scores = pd.to_numeric(
                ranking["score"], errors="coerce").to_numpy(dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("历史择优排名缺少有效样本或评分字段。") from exc
        if not np.any((rolling_windows > 0) & np.isfinite(scores)):
            # fixed_times 的逐代理段失败已由历史分析层保留原始原因。若全部
            # 代理段都因目标时刻缺失而失败，应把真实数据问题直接反馈给
            # 用户，而不是误报成历史区间长度不足。
            try:
                strategy_types = ranking["strategy_type"].astype(str)
                failure_scopes = ranking["failure_scope"].astype(str)
                failure_reasons = ranking["failure_reason"].fillna("").astype(str)
                fixed_failures = failure_reasons[
                    strategy_types.eq("fixed_times")
                    & failure_scopes.eq("strategy")
                    & failure_reasons.str.strip().ne("")
                ]
            except (KeyError, TypeError, ValueError):
                fixed_failures = []
            if len(fixed_failures):
                first_failure = str(fixed_failures.iloc[0])
                raise ValueError(
                    "固定时刻策略没有形成任何可评估代理段；"
                    f"首个失败原因：{first_failure} "
                    "请确认行情粒度以及各交易日的 Bar 时间戳与设定时刻一致。"
                )
            try:
                endpoint_failures = failure_reasons[
                    failure_scopes.eq("endpoint")
                    & failure_reasons.str.strip().ne("")
                ]
            except (AttributeError, TypeError):
                endpoint_failures = []
            if len(endpoint_failures):
                first_failure = str(endpoint_failures.iloc[0])
                raise ValueError(
                    "历史具体合约池没有形成任何可评估代理段；"
                    f"首个失败原因：{first_failure}"
                )
            raise ValueError(
                "真实历史长度不足，尚未形成任何可评估严格区间；"
                "请扩大 CSV/Wind 历史区间后重试。")

    def _history_recommendation_worker(self, gs):
        try:
            BacktestApp._validate_history_recommendation_source(gs)
            history_lookbacks = BacktestApp._normalize_history_lookbacks(
                gs.get("history_lookbacks"))
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
                    )
                )
            BacktestApp._validate_history_recommendation_payload(
                recommendations, ranking, window_results)
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
            success_text="历史择优完成  |  可应用历史策略或在当前路径验证",
            failure_text="历史择优失败  |  请查看错误信息",
        )

    # 旧私有方法保留软兼容，不再代表对比页状态。
    def _strategy_comparison_worker(self, gs):
        return self._history_recommendation_worker(gs)

    def _deliver_strategy_comparison(
            self, _summary, _results, recommendations=None, ranking=None,
            notes=None, window_results=None, source_label=None):
        return self._deliver_history_recommendation(
            recommendations, ranking, notes, window_results, source_label)

    def _fail_strategy_comparison(self, message):
        return self._fail_history_recommendation(message)

    def _finish_comparison(self, success=True):
        return self._finish_history_recommendation(success)

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
        """返回历史候选在 Day 0 前必须具备的 realized-HV 预热日数。"""
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
    def _calendar_span_for_trading_days(trading_days):
        """把交易日需求换成带节假日缓冲的自然日拉取跨度。"""
        try:
            days = int(trading_days)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Wind 交易日需求必须是正整数。") from exc
        if days <= 0:
            raise ValueError("Wind 交易日需求必须是正整数。")
        return int(np.ceil(days * 365.0 / ANNUAL_DAYS)) + _WIND_DATE_BUFFER_DAYS

    @staticmethod
    def _recommended_fixed_time_bar_size(fixed_times, *, wind_code=None):
        """返回能稳健覆盖目标 HH:MM 的推荐分钟粒度。"""
        strategy = FixedTimeStrategy(fixed_times)
        if str(wind_code or "").strip():
            from pricing.wind_data import get_trading_session_clock_ranges
            strategy.set_trading_sessions(
                get_trading_session_clock_ranges(
                    str(wind_code).strip(), allow_wind=False))
        parsed = strategy.effective_times
        minute_marks = tuple(t.hour * 60 + t.minute for t in parsed)
        for label in ("15min", "5min", "1min"):
            bar_minutes = _WIND_BAR_MINUTES[label]
            if all(mark % bar_minutes == 0 for mark in minute_marks):
                return label
        return "1min"

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
        if requested != WIND_AUTO_BAR_SIZE:
            if requested == "日频" and needs_fixed:
                raise ValueError(
                    "固定时刻策略需要分钟行情；请把 Wind 行情采样粒度设为"
                    "“自动（推荐）”或分钟频率。")
            return requested

        recommendations = []
        if needs_band:
            # 固定间隔在日频上只能观察收盘穿带；15min 是准确性与一年历史
            # 批量回测成本之间的默认折中，用户仍可手动覆盖到更细粒度。
            recommendations.append("15min")
        if needs_fixed:
            recommendations.append(
                BacktestApp._recommended_fixed_time_bar_size(
                    fixed_times, wind_code=wind_code))
        if not recommendations:
            return "日频"
        return min(
            recommendations,
            key=lambda label: _WIND_BAR_MINUTES[label],
        )

    @staticmethod
    def _resolve_single_wind_state(state, *, today=None):
        """解析单次回测的建仓日、自动结束日和实际行情粒度。"""
        resolved = dict(state)
        if resolved.get("source") != "wind":
            return resolved
        if not str(resolved.get("wind_code", "")).strip():
            raise ValueError("Wind 代码不能为空。")

        today = today or datetime.date.today()
        if not isinstance(today, datetime.date):
            today = BacktestApp._parse_wind_date(today, "当前日期")
        start = BacktestApp._parse_wind_date(
            resolved.get("wind_start"), "Wind 建仓日")
        if start > today:
            raise ValueError("Wind 建仓日不能晚于当前日期。")

        maturity_days = BacktestApp._maturity_days_from_params(
            resolved.get("params", {}))
        auto_end = bool(resolved.get("wind_auto_end", False))
        if auto_end:
            span = BacktestApp._calendar_span_for_trading_days(
                maturity_days + 1)
            end = min(start + datetime.timedelta(days=span), today)
        else:
            end = BacktestApp._parse_wind_date(
                resolved.get("wind_end"), "Wind 数据结束日")
            if end > today:
                raise ValueError("Wind 数据结束日不能晚于当前日期。")
        if end <= start:
            raise ValueError("Wind 数据结束日必须晚于建仓日。")

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
            "wind_end": end.isoformat(),
            "wind_auto_end": auto_end,
            "wind_bar_size_requested": str(requested),
            "wind_bar_size": actual_bar_size,
            "wind_date_mode": "auto_maturity" if auto_end else "custom_range",
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
            span = BacktestApp._calendar_span_for_trading_days(
                required_trade_days)
            start = asof - datetime.timedelta(days=span)
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
        # 真实 CSV/Wind 的 bar 数由后端按时间索引和交易时段自动推导；
        # 只有模拟路径需要用户选择离散粒度。
        steps_per_day = self._gui_steps_per_day(source, self._spd_var.get())
        force_close_var = getattr(
            self, "_force_day_close_hedge_var", None)
        force_day_close_hedge = bool(
            force_close_var.get()) if force_close_var is not None else False
        wind_auto_end_var = getattr(self, "_wind_auto_end_var", None)
        wind_auto_end = bool(
            wind_auto_end_var.get()) if wind_auto_end_var is not None else False
        wind_bar_size_requested = self._wind_bar_size_var.get().strip()

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
            "real_vol": self._real_vol_var.get().strip(),
            "n_paths": self._npaths_var.get().strip(),
            "csv_path": self._csv_path_var.get().strip(),
            "csv_col": self._csv_col_var.get().strip() or "close",
            "wind_code": self._wind_code_var.get().strip(),
            "wind_start": self._wind_start_var.get().strip(),
            "wind_end": self._wind_end_var.get().strip(),
            "wind_auto_end": wind_auto_end,
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
        # 每日收盘是历史图表的固定 C2C 基准，UI 中只读且状态永远为真。
        include_close = True
        include_fixed_times = bool(
            self._history_include_fixed_times_var.get())
        include_band = bool(self._history_include_band_var.get())
        if not (include_fixed_times or include_band):
            raise ValueError(
                "历史择优会固定运行每日收盘 C2C 基准；"
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
        sigma_source = "implied"
        sigma_window = 20
        if include_band:
            band_candidates = BacktestApp._parse_band_candidate_sigmas(
                self._history_band_candidate_sigmas_var.get())
            if not band_candidates and not include_current_band:
                raise ValueError(
                    "固定间隔已勾选，请至少输入一个 σ 候选，"
                    "或勾选『加入当前回测带宽』。")
            sigma_source = SIGMA_SOURCE_FROM_DISPLAY.get(
                self._history_sigma_src_var.get(),
                self._history_sigma_src_var.get(),
            )
            if sigma_source not in SIGMA_SOURCE_DISPLAY:
                raise ValueError(f"未知历史候选 σ 来源: {sigma_source}")
            if sigma_source == "realized":
                sigma_window = int(self._history_sigma_win_var.get())
                if sigma_window < 2:
                    raise ValueError("历史候选 HV 回看期必须至少为 2 日。")
            else:
                sigma_window = 20

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
        })
        history_asof_var = getattr(self, "_history_wind_asof_var", None)
        history_start_var = getattr(self, "_history_wind_start_var", None)
        history_auto_start_var = getattr(
            self, "_history_wind_auto_start_var", None)
        history_bar_size_var = getattr(
            self, "_history_wind_bar_size_var", None)
        state.update({
            "history_wind_asof": (
                history_asof_var.get().strip()
                if history_asof_var is not None else state.get("wind_end", "")),
            "history_wind_start": (
                history_start_var.get().strip()
                if history_start_var is not None else state.get("wind_start", "")),
            # 旧测试替身 / API 调用没有历史控件时保留显式起始日；真实 GUI
            # 默认自动覆盖最长已选严格区间、HV 预热与 Day 0 锚点。
            "history_wind_auto_start": (
                bool(history_auto_start_var.get())
                if history_auto_start_var is not None else False),
            "history_wind_bar_size_requested": (
                history_bar_size_var.get().strip()
                if history_bar_size_var is not None
                else state.get("wind_bar_size_requested", "日频")),
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
        pending_history_name = getattr(
            self, "_pending_history_retain_name", None)
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
        self._pending_history_retain_name = None
        if success and pending_history_name:
            next_sequence = self._saved_backtest_sequence + 1
            auto_name = f"{pending_history_name} #{next_sequence:02d}"
            suffix = 2
            while any(
                    snapshot.name == auto_name
                    for snapshot in self._saved_backtests.values()):
                auto_name = f"{pending_history_name} ({suffix})"
                suffix += 1
            try:
                self._store_current_backtest(auto_name)
            except Exception as exc:
                messagebox.showerror("历史验证结果保留失败", str(exc))

    def _fail_backtest(self, message):
        self._pending_history_retain_name = None
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
            return "每日收盘", _with_fallback("每交易日最后一根 bar")
        if strategy == "fixed_times":
            times = str(gui_state.get("fixed_times", "")).strip() or "—"
            return "固定时刻", _with_fallback(times)
        if strategy == "hedge_band":
            band_type = gui_state.get("interval_type", "absolute")
            threshold = float(gui_state.get("price_interval", 0.0))
            if band_type == "relative":
                primary = f"相对 {threshold:.4%}"
            elif band_type == "sigma":
                primary = f"{threshold:g}σ（日波动）"
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
    def _saved_snapshot_strategy_type(snapshot):
        """读取不可被用户重命名影响的快照策略类型。"""
        summary_row = getattr(snapshot, "summary_row", {}) or {}
        return str(summary_row.get("strategy_type", "")).strip()

    @staticmethod
    def _saved_snapshot_position(snapshot):
        """读取快照的经济方向；显示名称和其它参数均不能覆盖它。"""
        return BacktestApp._normalize_position(snapshot.position)

    def _resolve_saved_comparison_baseline_id(self, position=None):
        """返回当前方向最早的 close-to-close 基准，禁止跨方向复用。"""
        saved = getattr(self, "_saved_backtests", {})
        target_position = (
            BacktestApp._normalize_position(position)
            if position is not None else None)
        if target_position is None:
            selected_ids = getattr(
                self, "_saved_comparison_selection", set())
            selected_positions = {
                BacktestApp._saved_snapshot_position(saved[result_id])
                for result_id in selected_ids if result_id in saved
            }
            if len(selected_positions) == 1:
                target_position = next(iter(selected_positions))
            elif len(selected_positions) > 1:
                # 防御旧状态或外部调用制造的混选；调用方随后会给出明确提示。
                self._saved_comparison_baseline_id = None
                return None

        preferred = getattr(self, "_saved_comparison_baseline_id", None)
        if (preferred in saved and BacktestApp._saved_snapshot_strategy_type(
                saved[preferred]) == "close_to_close"
                and (target_position is None
                     or BacktestApp._saved_snapshot_position(
                         saved[preferred]) == target_position)):
            return preferred

        baseline_id = next((
            result_id for result_id, snapshot in saved.items()
            if BacktestApp._saved_snapshot_strategy_type(snapshot)
            == "close_to_close"
            and (target_position is None
                 or BacktestApp._saved_snapshot_position(snapshot)
                 == target_position)
        ), None)
        self._saved_comparison_baseline_id = baseline_id
        return baseline_id

    @staticmethod
    def _make_saved_backtest_result(
            bt, gui_state, result_id, name, saved_at=None):
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
        params = gui_state.get("params", {})
        contract_key = BacktestApp._freeze_snapshot_value({
            "cls_name": gui_state.get("cls_name"),
            "subtype": gui_state.get("subtype"),
            "params": params,
        })
        economics_key = BacktestApp._freeze_snapshot_value({
            "position": snapshot_position,
            "quantity": gui_state.get("quantity"),
            "multiplier": gui_state.get("multiplier"),
            "tc_rate": gui_state.get("tc_rate"),
            "slippage_bps": gui_state.get("slippage_bps", 0.0),
            "steps_per_day": bt._results.get("steps_per_day"),
        })
        subtype = gui_state.get("subtype", "—")
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
            path_key=BacktestApp._backtest_path_key(bt, gui_state),
            contract_key=contract_key,
            economics_key=economics_key,
            position=snapshot_position,
        )

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

    def _update_saved_result_count(self):
        button = getattr(self, "_compare_btn", None)
        if button is not None:
            button.configure(
                text=f"⚖  回测结果对比 ({len(self._saved_backtests)})")

    def _store_current_backtest(self, name):
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
        snapshot = self._make_saved_backtest_result(
            bt, gui_state, result_id, cleaned_name)
        self._saved_backtest_sequence = next_sequence
        self._saved_backtests[result_id] = snapshot
        position = BacktestApp._saved_snapshot_position(snapshot)
        # 结果池可同时保存买卖两组，但当前视图始终只激活一个方向。
        self._saved_comparison_selection.intersection_update({
            saved_id for saved_id, saved_snapshot
            in self._saved_backtests.items()
            if BacktestApp._saved_snapshot_position(saved_snapshot)
            == position
        })
        self._saved_comparison_selection.add(result_id)
        baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(
            self, position)
        if baseline_id is not None:
            self._saved_comparison_selection.add(baseline_id)
        self._latest_retained_result_id = result_id
        self._update_saved_result_count()
        self._sync_retain_button_state()
        self._refresh_saved_comparison_if_visible()
        self._set_status(
            f"已保留『{cleaned_name}』  |  可修改策略或参数继续回测（共 "
            f"{len(self._saved_backtests)} 条）")
        return snapshot

    def _retain_current_backtest(self):
        bt = getattr(self, "_latest_backtest", None)
        gui_state = getattr(self, "_latest_backtest_state", None)
        if bt is None or gui_state is None:
            messagebox.showinfo("没有可保留结果", "请先成功运行一次回测。")
            return
        if getattr(self, "_latest_retained_result_id", None) is not None:
            messagebox.showinfo("结果已保留", "当前回测结果已经在对比结果池中。")
            return

        next_sequence = self._saved_backtest_sequence + 1
        strategy_label, parameters = self._strategy_snapshot_labels(gui_state)
        short_parameters = parameters.split("；", 1)[0]
        if gui_state.get("strategy_name") == "close_to_close":
            default_name = f"{strategy_label} #{next_sequence:02d}"
        else:
            default_name = (
                f"{strategy_label} · {short_parameters} #{next_sequence:02d}")
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

        try:
            self._store_current_backtest(name)
        except Exception as exc:
            messagebox.showerror("保留结果失败", str(exc))
            return

    @staticmethod
    def _saved_comparison_payload(snapshots, baseline_result_id=None):
        """只读取已完成快照生成排名与曲线数据，绝不重新运行回测。"""
        import pandas as pd
        snapshots = list(snapshots)
        positions = {
            BacktestApp._saved_snapshot_position(snapshot)
            for snapshot in snapshots
        }
        if len(positions) > 1:
            raise ValueError(
                "买入(long)与卖出(short)结果不能混选；"
                "请一次只比较同一头寸方向。")
        close_snapshots = [
            snapshot for snapshot in snapshots
            if BacktestApp._saved_snapshot_strategy_type(snapshot)
            == "close_to_close"
        ]
        close_ids = {snapshot.result_id for snapshot in close_snapshots}
        if baseline_result_id not in close_ids:
            baseline_result_id = (
                min(close_snapshots, key=lambda snapshot: (
                    snapshot.result_id, snapshot.saved_at)).result_id
                if close_snapshots else None
            )
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
                "meta_is_comparison_baseline": (
                    snapshot.result_id == baseline_result_id),
            })
            rows.append(row)
            daily_curves[snapshot.name] = snapshot.daily_frame
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary = summary.sort_values(
                ["score", "total_tc", "meta_result_id"], kind="stable"
            ).reset_index(drop=True)
            summary.insert(0, "rank", np.arange(1, len(summary) + 1))
        return summary, daily_curves

    @staticmethod
    def _saved_comparison_warnings(snapshots, baseline_result_id=None):
        snapshots = list(snapshots)
        if not snapshots:
            return []
        if len({
                BacktestApp._saved_snapshot_position(snapshot)
                for snapshot in snapshots
        }) > 1:
            return [
                "买入(long)与卖出(short)结果不能混选；"
                "请清空后选择同一头寸方向。"
            ]
        warnings = []
        close_snapshots = [
            snapshot for snapshot in snapshots
            if BacktestApp._saved_snapshot_strategy_type(snapshot)
            == "close_to_close"
        ]
        if not close_snapshots:
            warnings.append(
                "未包含 close-to-close 基准；请先保留或勾选每日收盘结果，"
                "相对基准指标暂不计算。")
        elif len(close_snapshots) > 1:
            close_ids = {snapshot.result_id for snapshot in close_snapshots}
            chosen_id = (
                baseline_result_id if baseline_result_id in close_ids else
                min(close_snapshots, key=lambda snapshot: (
                    snapshot.result_id, snapshot.saved_at)).result_id
            )
            chosen = next(
                snapshot for snapshot in close_snapshots
                if snapshot.result_id == chosen_id)
            warnings.append(
                f"包含多条 close-to-close 结果；固定使用最早设定的"
                f"『{chosen.name}』作为基准，其余作为普通对比项。")
        if len(snapshots) == 1:
            warnings.append(
                "当前只选择了 1 个结果；再勾选至少一个结果即可查看相对差异。")
            return warnings
        if len({snapshot.path_key for snapshot in snapshots}) > 1:
            warnings.append(
                "行情路径或时间索引不同：曲线可并列查看，但领先排名仅供场景对照。")
        if len({snapshot.contract_key for snapshot in snapshots}) > 1:
            warnings.append(
                "期权结构参数不同：结果同时包含产品参数差异，不应只归因于对冲策略。")
        if len({snapshot.economics_key for snapshot in snapshots}) > 1:
            warnings.append(
                "头寸、合约乘数、成本、滑点或交易日聚合口径不同：金额差异需谨慎解读。")
        return warnings

    def _rename_saved_backtest(self, result_id, new_name):
        snapshot = self._saved_backtests[result_id]
        snapshot.name = BacktestApp._validate_saved_result_name(
            self._saved_backtests, new_name, exclude_id=result_id)
        return snapshot

    def _delete_saved_backtest(self, result_id):
        snapshot = self._saved_backtests.pop(result_id)
        self._saved_comparison_selection.discard(result_id)
        if getattr(self, "_saved_comparison_baseline_id", None) == result_id:
            self._saved_comparison_baseline_id = None
        replacement_id = BacktestApp._resolve_saved_comparison_baseline_id(
            self)
        if self._saved_comparison_selection and replacement_id is not None:
            self._saved_comparison_selection.add(replacement_id)
        if getattr(self, "_latest_retained_result_id", None) == result_id:
            self._latest_retained_result_id = None
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
            "日频": None, "60min": "60", "30min": "30",
            "15min": "15", "5min": "5", "1min": "1",
        }
        if wind_bar_label not in _bar_label_to_size:
            raise ValueError(
                f"Wind 行情采样粒度尚未解析: {wind_bar_label!r}")
        wind_bar_size = _bar_label_to_size[wind_bar_label]

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

    @staticmethod
    def _validate_fixed_time_history_granularity(bt):
        """历史候选启动前只验证真实时间索引与日内粒度。"""
        timestamps = getattr(bt, "timestamps", None)
        if timestamps is None:
            raise ValueError(
                "固定时刻策略需要真实 pandas.DatetimeIndex 的日内行情。")
        try:
            import pandas as pd
            index = pd.DatetimeIndex(timestamps)
        except Exception as exc:
            raise ValueError("行情时间戳无法解析为 DatetimeIndex。") from exc
        if _infer_intraday_steps(index) <= 1:
            raise ValueError(
                "固定时刻策略仅支持真实日内行情；当前 DatetimeIndex "
                "每交易日只有一根 bar（日频）。")

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
    def _comparison_finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _comparison_safe_int(value, default=0):
        number = BacktestApp._comparison_finite(value)
        if number is None or not number.is_integer():
            return int(default)
        return int(number)

    @staticmethod
    def _comparison_safe_bool(value, default=False):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return bool(default)

    @staticmethod
    def _comparison_baseline(summary):
        """从对比摘要中稳定选择显式标记的 close-to-close 基准。"""
        if summary is None or getattr(summary, "empty", True):
            return None
        explicit = []
        close_rows = []
        for _index, row in summary.iterrows():
            item = row.to_dict()
            if item.get("strategy_type") != "close_to_close":
                continue
            close_rows.append(item)
            if BacktestApp._comparison_safe_bool(
                    item.get("meta_is_comparison_baseline"), False):
                explicit.append(item)

        candidates = explicit or close_rows
        if not candidates:
            return None

        def _stable_key(item):
            result_id = item.get("meta_result_id")
            if isinstance(result_id, str) and result_id:
                return 0, result_id
            rank = BacktestApp._comparison_finite(item.get("rank"))
            return 1, np.inf if rank is None else rank, str(
                item.get("strategy", ""))

        return min(candidates, key=_stable_key)

    @staticmethod
    def _comparison_rows_match(left, right):
        """用稳定快照 ID 判断同一结果，缺少 ID 时才回退到策略身份。"""
        if not left or not right:
            return False
        left_id = left.get("meta_result_id")
        right_id = right.get("meta_result_id")
        if (isinstance(left_id, str) and left_id
                and isinstance(right_id, str) and right_id):
            return left_id == right_id
        return (
            left.get("strategy_type") == right.get("strategy_type")
            and left.get("strategy") == right.get("strategy")
        )

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

    @staticmethod
    def _comparison_headline(summary):
        """返回当前路径冠军、次优及改善幅度，独立于 DataFrame 行顺序。"""
        candidates = []
        if summary is not None and not getattr(summary, "empty", True):
            for _index, row in summary.iterrows():
                item = row.to_dict()
                score = BacktestApp._comparison_finite(item.get("score"))
                if score is None:
                    continue
                cost = BacktestApp._comparison_finite(item.get("total_tc"))
                rank = BacktestApp._comparison_finite(item.get("rank"))
                stable_tie_breaker = (
                    (0, rank) if rank is not None else
                    (1, str(item.get("meta_result_id",
                                     item.get("strategy", ""))))
                )
                candidates.append((
                    score,
                    np.inf if cost is None else cost,
                    stable_tie_breaker,
                    item,
                ))
        candidates.sort(key=lambda item: item[:3])
        if not candidates:
            return {
                "best": None, "runner_up": None,
                "improvement_ratio": None, "strategy_count": 0,
            }
        best = candidates[0][3]
        runner = candidates[1][3] if len(candidates) > 1 else None
        improvement = None
        if runner is not None:
            best_score = BacktestApp._comparison_finite(best.get("score"))
            runner_score = BacktestApp._comparison_finite(runner.get("score"))
            if (best_score is not None and runner_score is not None
                    and runner_score > 0):
                improvement = (runner_score - best_score) / runner_score
        return {
            "best": best,
            "runner_up": runner,
            "improvement_ratio": improvement,
            "strategy_count": len(candidates),
        }

    @staticmethod
    def _comparison_relative_delta(selected, baseline, key):
        selected_value = BacktestApp._comparison_finite(selected.get(key))
        baseline_value = BacktestApp._comparison_finite(baseline.get(key))
        if selected_value is None or baseline_value is None:
            return {"value": selected_value, "delta": None, "ratio": None}
        delta = selected_value - baseline_value
        ratio = delta / abs(baseline_value) if baseline_value != 0 else None
        return {"value": selected_value, "delta": delta, "ratio": ratio}

    @staticmethod
    def _history_chart_pairs(summary, lookback, strategy):
        """按 ``(lookback, window_id)`` 配对候选与固定 C2C 基准。"""
        import pandas as pd

        if summary is None or getattr(summary, "empty", True) or not strategy:
            return pd.DataFrame()
        required = {
            "lookback", "window_id", "strategy", "strategy_type", "success",
        }
        if not required.issubset(summary.columns):
            return pd.DataFrame()
        rows = summary[summary["lookback"].astype(str) == str(lookback)].copy()
        if rows.empty:
            return pd.DataFrame()
        rows = rows[rows["success"].fillna(False).astype(bool)]
        baseline = rows[rows["strategy_type"].astype(str) == "close_to_close"]
        candidate = rows[rows["strategy"].astype(str) == str(strategy)]
        if baseline.empty or candidate.empty:
            return pd.DataFrame()
        baseline = baseline.drop_duplicates("window_id", keep="first")
        candidate = candidate.drop_duplicates("window_id", keep="first")
        pairs = candidate.merge(
            baseline, on=["lookback", "window_id"], how="inner",
            suffixes=("_candidate", "_baseline"), validate="one_to_one",
        )
        if pairs.empty:
            return pairs
        # 只保留段内交易日数完全一致的共同代理段；图表不得静默截断。
        def _daily_length(value):
            try:
                return len(np.asarray(value, dtype=float).reshape(-1))
            except (TypeError, ValueError):
                return -1

        candidate_daily = pairs.get("daily_net_pnl_candidate")
        baseline_daily = pairs.get("daily_net_pnl_baseline")
        if candidate_daily is None or baseline_daily is None:
            return pairs.iloc[:0].copy()
        same_length = np.fromiter((
            _daily_length(candidate_value) > 0
            and _daily_length(candidate_value) == _daily_length(baseline_value)
            for candidate_value, baseline_value in zip(
                candidate_daily, baseline_daily)
        ), dtype=bool, count=len(pairs))
        pairs = pairs.loc[same_length].copy()
        if pairs.empty:
            return pairs
        start_values = pairs.get("start_ts_candidate")
        if start_values is None:
            start_values = pairs.get("start_ts_baseline")
        end_values = pairs.get("end_ts_candidate")
        if end_values is None:
            end_values = pairs.get("end_ts_baseline")
        pairs = pairs.assign(
            _chart_start_ts=pd.to_datetime(start_values, errors="coerce"),
            _chart_end_ts=pd.to_datetime(end_values, errors="coerce"))
        return pairs.sort_values(
            ["_chart_start_ts", "_chart_end_ts", "window_id"], kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    @staticmethod
    def _history_chart_array(row, column, role):
        """安全读取 summary 单元格中的一维曲线数组。"""
        value = row.get(f"{column}_{role}")
        if value is None:
            return np.array([], dtype=float)
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.array([], dtype=float)
        return (
            array if array.size and np.any(np.isfinite(array))
            else np.array([], dtype=float)
        )

    @staticmethod
    def _history_chart_band(arrays):
        """按代理段内交易日对齐累计曲线并返回中位数与 P25/P75。

        提前敲出/结算后的累计值保持不变直至最长代理段，避免后半段只剩
        未敲出代理段而产生幸存者偏差。调用方只向这里传累计曲线。
        """
        arrays = [np.asarray(array, dtype=float).reshape(-1)
                  for array in arrays if len(array)]
        if not arrays:
            return None
        width = max(len(array) for array in arrays)
        matrix = np.full((len(arrays), width), np.nan, dtype=float)
        for row_no, array in enumerate(arrays):
            matrix[row_no, :len(array)] = array
            if len(array) < width and np.isfinite(array[-1]):
                matrix[row_no, len(array):] = array[-1]
        valid = np.any(np.isfinite(matrix), axis=0)
        if not np.any(valid):
            return None
        matrix = matrix[:, valid]
        return {
            "x": np.flatnonzero(valid).astype(float) + 1.0,
            "median": np.nanmedian(matrix, axis=0),
            "p25": np.nanpercentile(matrix, 25, axis=0),
            "p75": np.nanpercentile(matrix, 75, axis=0),
            "window_count": len(arrays),
            "show_interval": len(arrays) >= 2,
        }

    @staticmethod
    def _history_chart_model(
            summary, lookback, strategy, mode="single", metric="net",
            window_id=None, normalized=False):
        """生成历史图表的纯数据模型，不依赖 Tk，便于配对与空态测试。"""
        raw_metric_columns = {
            "net": ("cumulative_net_pnl", "累计净 PnL"),
            "gross": ("cumulative_gross_pnl", "累计成本前 PnL"),
            "tc": ("cumulative_tc", "累计成本"),
        }
        normalized_metric_columns = {
            "net": ("normalized_cumulative_net_pnl", "累计净 PnL / 期初名义金额"),
            "gross": (
                "normalized_cumulative_gross_pnl",
                "累计成本前 PnL / 期初名义金额",
            ),
            "tc": ("normalized_cumulative_tc", "累计成本 / 期初名义金额"),
        }
        metric_columns = (
            normalized_metric_columns if normalized else raw_metric_columns)
        import pandas as pd
        mode = str(mode or "single")
        metric = str(metric or "net")
        if mode not in {"single", "typical"}:
            return {"state": "empty", "message": "未知历史图表模式。", "mode": mode}
        if metric not in metric_columns:
            return {"state": "empty", "message": "未知历史图表指标。", "mode": mode}
        if not strategy:
            return {"state": "empty", "message": "请先选择一条历史策略。", "mode": mode}

        pairs = BacktestApp._history_chart_pairs(
            summary, lookback, strategy)
        if pairs.empty:
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": "所选周期没有可与每日收盘基准配对的成功代理段。",
                "window_options": [],
            }
        window_options = []
        for sample_no, (_index, row) in enumerate(pairs.iterrows(), start=1):
            end_ts = row.get("_chart_end_ts")
            date_text = (
                end_ts.strftime("%Y-%m-%d")
                if end_ts is not None and not getattr(end_ts, "isnat", False)
                and not pd.isna(end_ts)
                else "日期未知"
            )
            window_options.append({
                "window_id": str(row["window_id"]),
                "end_ts": end_ts,
                "label": f"代理段 {sample_no} · {date_text}",
            })
        base = {
            "state": "ok", "mode": mode, "metric": metric,
            "lookback": str(lookback), "strategy": str(strategy),
            "window_options": window_options,
            "uses_normalized_notional": bool(normalized),
        }

        curve_column, metric_label = metric_columns[metric]
        selected_is_baseline = bool(
            pairs["strategy_type_candidate"].astype(str)
            .eq("close_to_close").all())
        roles = [("baseline", "每日收盘（C2C基准）")]
        if not selected_is_baseline:
            roles.append(("candidate", str(strategy)))

        if mode == "single":
            option_ids = [item["window_id"] for item in window_options]
            selected_window = (
                str(window_id) if window_id is not None
                and str(window_id) in option_ids else option_ids[-1])
            row = pairs[pairs["window_id"].astype(str) == selected_window].iloc[0]
            series = []
            for role, label in roles:
                values = BacktestApp._history_chart_array(
                    row, curve_column, role)
                if not len(values):
                    return {
                        **base, "state": "empty",
                        "message": f"配对代理段缺少{metric_label}曲线。",
                        "selected_window_id": selected_window,
                    }
                series.append({
                    "role": role, "label": label,
                    "x": np.arange(1, len(values) + 1, dtype=float),
                    "y": values,
                })
            if len({len(item["y"]) for item in series}) != 1:
                return {
                    **base, "state": "empty",
                    "message": "配对代理段内交易日数不一致，无法绘图。",
                    "selected_window_id": selected_window,
                }
            return {
                **base, "selected_window_id": selected_window,
                "series": series, "metric_label": metric_label,
                "title": f"单代理段路径 · {metric_label}",
            }

        bands = []
        for role, label in roles:
            arrays = [
                BacktestApp._history_chart_array(row, curve_column, role)
                for _index, row in pairs.iterrows()
            ]
            band = BacktestApp._history_chart_band(arrays)
            if band is None:
                return {
                    **base, "state": "empty",
                    "message": f"配对代理段缺少{metric_label}曲线。",
                }
            bands.append({"role": role, "label": label, **band})
        return {
            **base, "bands": bands, "metric_label": metric_label,
            "title": (
                f"多代理段中位路径 · {metric_label}"
                "（中位数；P25/P75 为描述区间，非置信区间）"
            ),
        }

    @staticmethod
    def _history_multi_chart_model(
            summary, lookback, strategies, mode="single", metric="net",
            window_id=None, primary_strategy=None):
        """生成固定 C2C 基准加多个候选的严格配对代理段图表模型。"""
        import pandas as pd

        if isinstance(strategies, str):
            strategies = [strategies]
        try:
            strategies = list(dict.fromkeys(
                str(value).strip() for value in (strategies or ())
                if str(value).strip()))
        except TypeError:
            strategies = []
        mode = str(mode or "single")
        metric = str(metric or "net")
        if mode not in {"full", "single", "typical"}:
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": "未知历史图表模式。", "window_options": [],
                "strategies": strategies,
            }
        if metric not in {"net", "gross", "tc"}:
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": "未知历史图表指标。", "window_options": [],
                "strategies": strategies,
            }
        required = {
            "lookback", "window_id", "strategy", "strategy_type", "success",
        }
        if (summary is None or getattr(summary, "empty", True)
                or not required.issubset(summary.columns)):
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": "所选周期没有可绘制的历史代理段。",
                "window_options": [], "strategies": strategies,
            }

        period_rows = summary[
            summary["lookback"].astype(str) == str(lookback)].copy()
        if period_rows.empty:
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": "所选周期没有可绘制的历史代理段。",
                "window_options": [], "strategies": strategies,
            }
        successful = period_rows[
            period_rows["success"].fillna(False).astype(bool)]
        baseline_rows = successful[
            successful["strategy_type"].astype(str).eq("close_to_close")]
        if baseline_rows.empty:
            return {
                "state": "empty", "mode": str(mode), "metric": str(metric),
                "message": "所选周期缺少每日收盘 C2C 基准。",
                "window_options": [], "strategies": strategies,
            }
        baseline_strategy = str(baseline_rows.iloc[0]["strategy"])
        baseline_strategies = set(
            baseline_rows["strategy"].dropna().astype(str))
        candidates = [
            value for value in strategies if value not in baseline_strategies]
        primary_strategy = str(primary_strategy or "").strip()
        if primary_strategy not in candidates:
            primary_strategy = candidates[0] if candidates else baseline_strategy
        comparison_strategies = candidates or [baseline_strategy]

        contract_codes = set()
        if "history_contract_code" in successful.columns:
            contract_codes = {
                str(value).strip()
                for value in successful["history_contract_code"].dropna()
                if str(value).strip() and str(value).strip() != "nan"
            }
        uses_normalized_notional = (
            mode in {"full", "typical"} and len(contract_codes) > 1)

        pairs_by_strategy = {
            strategy: BacktestApp._history_chart_pairs(
                summary, lookback, strategy)
            for strategy in comparison_strategies
        }
        missing = [
            strategy for strategy, pairs in pairs_by_strategy.items()
            if pairs.empty]
        if missing:
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": (
                    "以下策略没有可与 C2C 严格配对的代理段："
                    + "、".join(missing)),
                "window_options": [], "strategies": candidates,
                "baseline_strategy": baseline_strategy,
            }

        raw_metric_columns = {
            "net": "cumulative_net_pnl",
            "gross": "cumulative_gross_pnl",
            "tc": "cumulative_tc",
        }
        normalized_metric_columns = {
            "net": "normalized_cumulative_net_pnl",
            "gross": "normalized_cumulative_gross_pnl",
            "tc": "normalized_cumulative_tc",
        }
        raw_daily_metric_columns = {
            "net": "daily_net_pnl",
            "gross": "daily_gross_pnl",
            "tc": "daily_tc",
        }
        normalized_daily_metric_columns = {
            "net": "normalized_daily_net_pnl",
            "gross": "normalized_daily_gross_pnl",
            "tc": "normalized_daily_tc",
        }
        metric_columns = (
            (normalized_daily_metric_columns
             if uses_normalized_notional else raw_daily_metric_columns)
            if mode == "full" else
            (normalized_metric_columns
             if uses_normalized_notional else raw_metric_columns)
        )
        required_curve_column = metric_columns[metric]
        if (uses_normalized_notional
                and required_curve_column not in successful.columns):
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": (
                    "跨合约历史路径缺少安全归一化数据；"
                    "请改用单代理段路径查看原始金额。"),
                "window_options": [], "strategies": candidates,
                "baseline_strategy": baseline_strategy,
                "uses_normalized_notional": True,
            }

        def _valid_window_ids(pairs):
            valid_ids = set()
            curve_column = metric_columns[metric]
            for _index, row in pairs.iterrows():
                candidate_curve = BacktestApp._history_chart_array(
                    row, curve_column, "candidate")
                baseline_curve = BacktestApp._history_chart_array(
                    row, curve_column, "baseline")
                if (not len(candidate_curve)
                        or len(candidate_curve) != len(baseline_curve)
                        or not np.all(np.isfinite(candidate_curve))
                        or not np.all(np.isfinite(baseline_curve))):
                    continue
                valid_ids.add(str(row["window_id"]))
            return valid_ids

        valid_ids_by_strategy = {
            strategy: _valid_window_ids(pairs)
            for strategy, pairs in pairs_by_strategy.items()
        }
        common_ids = None
        for window_ids in valid_ids_by_strategy.values():
            common_ids = (
                window_ids if common_ids is None
                else common_ids.intersection(window_ids))
        common_ids = common_ids or set()
        first_pairs = next(iter(pairs_by_strategy.values()))
        ordered_ids = [
            str(value) for value in first_pairs["window_id"]
            if str(value) in common_ids]
        if not ordered_ids:
            return {
                "state": "empty", "mode": mode, "metric": metric,
                "message": (
                    "跨合约历史路径没有可安全归一化的共同代理段；"
                    "请改用单代理段路径查看原始金额。"
                    if uses_normalized_notional else
                    "所选候选之间没有共同的严格配对代理段。"),
                "window_options": [], "strategies": candidates,
                "baseline_strategy": baseline_strategy,
                "uses_normalized_notional": uses_normalized_notional,
            }

        selected_period = summary["lookback"].astype(str).eq(str(lookback))
        selected_windows = summary["window_id"].astype(str).isin(ordered_ids)
        filtered = summary.loc[~selected_period | selected_windows].copy()
        option_pairs = first_pairs[
            first_pairs["window_id"].astype(str).isin(ordered_ids)]
        option_pairs = option_pairs.set_index(
            option_pairs["window_id"].astype(str), drop=False)
        window_options = []
        for sample_no, common_id in enumerate(ordered_ids, start=1):
            row = option_pairs.loc[common_id]
            if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            end_ts = row.get("_chart_end_ts")
            date_text = (
                end_ts.strftime("%Y-%m-%d")
                if end_ts is not None and not pd.isna(end_ts)
                else "日期未知")
            contract_code = str(
                row.get("history_contract_code_candidate", "") or ""
            ).strip()
            contract_text = f" · {contract_code}" if contract_code else ""
            window_options.append({
                "window_id": common_id,
                "end_ts": end_ts,
                "label": f"代理段 {sample_no} · {date_text}{contract_text}",
            })

        base = {
            "state": "ok", "mode": mode, "metric": metric,
            "lookback": str(lookback), "strategies": candidates,
            "baseline_strategy": baseline_strategy,
            "primary_strategy": primary_strategy,
            "window_options": window_options,
            "uses_normalized_notional": uses_normalized_notional,
            "common_window_count": len(ordered_ids),
            "individual_window_counts": {
                strategy: len(window_ids)
                for strategy, window_ids in valid_ids_by_strategy.items()
            },
            "sample_scope_differs": any(
                len(window_ids) != len(ordered_ids)
                for window_ids in valid_ids_by_strategy.values()),
        }
        if mode == "full":
            curve_column = metric_columns[metric]

            def _ordered_pair_rows(strategy):
                pairs = pairs_by_strategy[strategy].copy()
                pairs = pairs[
                    pairs["window_id"].astype(str).isin(ordered_ids)]
                rows_by_id = {
                    str(row["window_id"]): row
                    for _index, row in pairs.iterrows()
                }
                return [rows_by_id[common_id] for common_id in ordered_ids]

            first_rows = _ordered_pair_rows(comparison_strategies[0])
            baseline_parts = [
                BacktestApp._history_chart_array(
                    row, curve_column, "baseline")
                for row in first_rows
            ]
            if (not baseline_parts
                    or any(not len(part) for part in baseline_parts)):
                return {
                    **base, "state": "empty",
                    "message": "共同代理段缺少每日收盘 C2C 日级曲线。",
                }
            segment_lengths = [len(part) for part in baseline_parts]

            def _full_series(parts, role, label, strategy):
                if (len(parts) != len(segment_lengths)
                        or any(
                            not len(part)
                            or len(part) != expected_length
                            or not np.all(np.isfinite(part))
                            for part, expected_length in zip(
                                parts, segment_lengths))):
                    return None
                daily = np.concatenate(parts)
                return {
                    "role": role, "label": label, "strategy": strategy,
                    "x": np.arange(1, len(daily) + 1, dtype=float),
                    "y": np.cumsum(daily),
                }

            baseline_item = _full_series(
                baseline_parts, "baseline", "每日收盘（C2C基准）",
                baseline_strategy)
            if baseline_item is None:
                return {
                    **base, "state": "empty",
                    "message": "共同代理段内 C2C 日级曲线不完整；未补零。",
                }
            series = [baseline_item]
            for strategy in candidates:
                candidate_parts = [
                    BacktestApp._history_chart_array(
                        row, curve_column, "candidate")
                    for row in _ordered_pair_rows(strategy)
                ]
                item = _full_series(
                    candidate_parts, "candidate", strategy, strategy)
                if item is None:
                    return {
                        **base, "state": "empty",
                        "message": (
                            f"{strategy} 的共同代理段日级曲线不完整；未补零。"),
                    }
                series.append(item)

            common_day_count = int(sum(segment_lengths))
            expected_day_count = int(
                LOOKBACK_DAYS.get(str(lookback), common_day_count))
            complete_evidence = common_day_count == expected_day_count
            raw_labels = {
                "net": "累计净 PnL",
                "gross": "累计成本前 PnL",
                "tc": "累计成本",
            }
            normalized_labels = {
                "net": "累计净 PnL / 各代理段期初名义金额",
                "gross": "累计成本前 PnL / 各代理段期初名义金额",
                "tc": "累计成本 / 各代理段期初名义金额",
            }
            metric_label = (
                normalized_labels if uses_normalized_notional
                else raw_labels)[metric]
            title_prefix = (
                "完整 L 日累计路径"
                if complete_evidence else
                f"共同交集累计路径（{common_day_count}/{expected_day_count} 日）"
            )
            return {
                **base,
                "series": series,
                "metric_label": metric_label,
                "title": f"{title_prefix} · {metric_label}",
                "common_day_count": common_day_count,
                "expected_day_count": expected_day_count,
                "complete_evidence": complete_evidence,
                "segment_lengths": tuple(segment_lengths),
                "segment_boundaries": tuple(
                    int(value)
                    for value in np.cumsum(segment_lengths)[:-1]),
            }

        first_strategy = comparison_strategies[0]
        first_model = BacktestApp._history_chart_model(
            filtered, lookback, first_strategy, mode=mode, metric=metric,
            window_id=window_id, normalized=uses_normalized_notional)
        if first_model.get("state") != "ok":
            return first_model
        collection_key = "series" if mode == "single" else "bands"
        baseline_item = next(
            (item for item in first_model[collection_key]
             if item.get("role") == "baseline"), None)
        if baseline_item is None:
            return {
                **base, "state": "empty",
                "message": "共同代理段缺少每日收盘 C2C 曲线。",
            }
        collection = [{**baseline_item, "strategy": baseline_strategy}]
        for strategy in candidates:
            model = BacktestApp._history_chart_model(
                filtered, lookback, strategy, mode=mode, metric=metric,
                window_id=first_model.get("selected_window_id", window_id),
                normalized=uses_normalized_notional)
            if model.get("state") != "ok":
                return {
                    **base, "state": "empty",
                    "message": model.get(
                        "message", f"{strategy} 缺少共同代理段曲线。"),
                }
            candidate_item = next(
                (item for item in model[collection_key]
                 if item.get("role") == "candidate"), None)
            if candidate_item is not None:
                collection.append({
                    **candidate_item, "strategy": strategy,
                    "label": strategy,
                })
        title_prefix = (
            "单代理段路径" if mode == "single" else "多代理段中位路径")
        result = {
            **base, collection_key: collection,
            "metric_label": first_model.get("metric_label", "累计 PnL"),
            "title": (
                f"{title_prefix} · {first_model.get('metric_label', '累计 PnL')}"
                + ("（中位数与 P25/P75）" if mode == "typical" else "")),
        }
        if mode == "single":
            result["selected_window_id"] = first_model.get(
                "selected_window_id")
        return result

    @staticmethod
    def _history_row_improvement(row, baseline=None):
        """读取历史择优主指标；旧结果回退到合并 RMS 改善。"""
        if not row:
            return None
        # 新版严格区间与旧版逐窗等权结果都有各自正式的 selection 指标；
        # 二者都优先读取该指标，但只有新版常量可以驱动“严格 L 日”文案。
        if BacktestApp._history_row_uses_recognized_selection_metric(row):
            return BacktestApp._comparison_finite(
                row.get("selection_improvement_vs_c2c"))
        value = BacktestApp._comparison_finite(
            row.get("improvement_vs_c2c"))
        if value is not None:
            return value
        if str(row.get("strategy_type", "")) == "close_to_close":
            return 0.0 if BacktestApp._comparison_finite(
                row.get("score")) is not None else None
        candidate_score = BacktestApp._comparison_finite(row.get("score"))
        baseline_score = BacktestApp._comparison_finite(
            row.get("baseline_score"))
        if baseline_score is None and baseline:
            baseline_score = BacktestApp._comparison_finite(
                baseline.get("score"))
        if (candidate_score is None or baseline_score is None
                or np.isclose(baseline_score, 0.0, rtol=1e-12, atol=1e-12)):
            return None
        return (baseline_score - candidate_score) / abs(baseline_score)

    @staticmethod
    def _history_row_uses_recognized_selection_metric(row):
        """是否携带当前 GUI 能解释的新版或旧版选择指标。"""
        return bool(
            row
            and str(row.get("selection_metric", "")) in {
                HISTORY_SELECTION_METRIC,
                STRICT_LOOKBACK_SELECTION_METRIC,
            }
            and "selection_improvement_vs_c2c" in row)

    @staticmethod
    def _history_row_uses_strict_metric(row):
        """只有严格连续 L 日选择指标才允许展示 strict 口径文案。"""
        return bool(
            row
            and str(row.get("selection_metric", ""))
            == STRICT_LOOKBACK_SELECTION_METRIC
            and "selection_improvement_vs_c2c" in row)

    @staticmethod
    def _history_row_uses_window_equal_metric(row):
        """识别升级前的逐历史终点等权选择指标。"""
        return bool(
            row
            and str(row.get("selection_metric", "")) == HISTORY_SELECTION_METRIC
            and "selection_improvement_vs_c2c" in row)

    @staticmethod
    def _history_metric_labels(*, uses_strict_metric, uses_product_pool):
        """返回与历史数据路由一致的表头，避免把品种池诊断值写成主指标。"""
        if uses_strict_metric and uses_product_pool:
            return {
                "score": "汇总RMS(诊断)↓",
                "baseline": "C2C RMS(诊断)",
                "improvement": "较C2C改善",
            }
        if uses_strict_metric:
            return {
                "score": "区间RMS↓",
                "baseline": "C2C区间RMS",
                "improvement": "较C2C改善",
            }
        return {
            "score": "汇总RMS↓",
            "baseline": "C2C汇总RMS",
            "improvement": "旧口径改善",
        }

    @staticmethod
    def _history_contract_codes_text(values, *, limit=6):
        """把历史评分实际参与合约压缩成适合详情栏的文本。"""
        if isinstance(values, str):
            values = (values,)
        try:
            codes = tuple(dict.fromkeys(
                str(code).strip() for code in values
                if str(code).strip() and str(code).strip() != "nan"))
        except TypeError:
            return ""
        shown = codes[:max(1, int(limit))]
        result = "、".join(shown)
        if len(codes) > len(shown):
            result += f" 等{len(codes)}个"
        return result

    @staticmethod
    def _comparison_recommendation_rows(
            recommendations, ranking, lookbacks=None):
        """按固定 C2C 基准整理本次已选周期的历史参考展示模型。"""
        selected = BacktestApp._normalize_history_lookbacks(lookbacks)
        period_defs = tuple(
            (key, label) for key, label in HISTORY_PERIOD_DEFS
            if key in selected)

        def _subset(frame, key):
            if frame is None or getattr(frame, "empty", True):
                return None
            selected = frame[frame["lookback"] == key]
            return selected if not selected.empty else None

        def _sampling_context(item):
            item = item or {}
            segment_count = item.get(
                "segment_count", item.get("proxy_segments"))
            return {
                "maturity_days": item.get("maturity_days"),
                "evaluation_mode": item.get("evaluation_mode"),
                "evidence_days": BacktestApp._comparison_safe_int(
                    item.get("evidence_days"), 0),
                "days_used": BacktestApp._comparison_safe_int(
                    item.get("days_used"), 0),
                "segment_count": (
                    BacktestApp._comparison_safe_int(segment_count, 0)
                    if segment_count is not None else None),
                "expiry_segments": BacktestApp._comparison_safe_int(
                    item.get("expiry_segments"), 0),
                "mtm_segments": BacktestApp._comparison_safe_int(
                    item.get("mtm_segments"), 0),
                "terminal_mode": item.get("terminal_mode"),
                "realized_sigma_warmup_days": (
                    BacktestApp._comparison_safe_int(
                        item.get("realized_sigma_warmup_days"), 0)),
                "warmup_eligible_endpoints": (
                    BacktestApp._comparison_safe_int(
                        item.get("warmup_eligible_endpoints"), 0)),
                "sampling_mode": item.get("sampling_mode", "fixed_step"),
                "step_days": item.get("step_days"),
                "required_history_days": BacktestApp._comparison_safe_int(
                    item.get("required_history_days"), 0),
                "available_history_days": BacktestApp._comparison_safe_int(
                    item.get("available_history_days"), 0),
                "history_complete": BacktestApp._comparison_safe_bool(
                    item.get("history_complete"), False),
                "history_mode": item.get("history_mode"),
                "product_code": item.get("product_code"),
                "main_contract_asof": item.get("main_contract_asof"),
                "effective_asof_date": item.get("effective_asof_date"),
                "effective_main_contract": item.get(
                    "effective_main_contract"),
                "pool_contracts_available": BacktestApp._comparison_safe_int(
                    item.get("pool_contracts_available"), 0),
                "pool_contracts_invalid": BacktestApp._comparison_safe_int(
                    item.get("pool_contracts_invalid"), 0),
                "mapping_trailing_days_dropped": (
                    BacktestApp._comparison_safe_int(
                        item.get("mapping_trailing_days_dropped"), 0)),
                "selection_metric": item.get("selection_metric"),
                "uses_strict_metric": (
                    BacktestApp._history_row_uses_strict_metric(item)),
                "uses_window_equal_metric": (
                    BacktestApp._history_row_uses_window_equal_metric(item)),
                "relative_comparison_windows": (
                    BacktestApp._comparison_safe_int(
                        item.get("relative_comparison_windows"), 0)
                    if "relative_comparison_windows" in item else None),
                "paired_contract_codes": item.get(
                    "paired_contract_codes", ()),
            }

        rows = []
        for key, display in period_defs:
            rec_group = _subset(recommendations, key)
            rank_group = _subset(ranking, key)
            formal_row = (
                rec_group.iloc[0].to_dict()
                if rec_group is not None else None)
            baseline = BacktestApp._comparison_baseline(rank_group)
            context_row = baseline
            if context_row is None and rank_group is not None:
                context_row = rank_group.iloc[0].to_dict()

            comparable_candidates = []
            if rank_group is not None:
                for _index, rank_row in rank_group.iterrows():
                    item = rank_row.to_dict()
                    if str(item.get("strategy_type", "")) == "close_to_close":
                        continue
                    # 新结果显式携带 comparison_eligible；旧快照仅在完整行
                    # 上回退为可比，避免把部分代理段结果包装成诊断冠军。
                    if "comparison_eligible" in item:
                        comparable = BacktestApp._comparison_safe_bool(
                            item.get("comparison_eligible"), False)
                    else:
                        comparable = BacktestApp._comparison_safe_bool(
                            item.get("complete_window"), False)
                    if comparable:
                        comparable_candidates.append(item)

            has_comparable_candidate = bool(comparable_candidates)
            if formal_row is not None:
                leader = formal_row
            elif has_comparable_candidate and rank_group is not None:
                comparable_names = {
                    str(item.get("strategy", ""))
                    for item in comparable_candidates
                }
                eligible_group = rank_group[
                    rank_group["strategy"].astype(str).isin(comparable_names)
                    | rank_group["strategy_type"].astype(str).eq(
                        "close_to_close")
                ]
                leader = eligible_group.sort_values(
                    ["rank", "score", "strategy"],
                    kind="stable").iloc[0].to_dict()
            else:
                leader = baseline

            leader_score = (
                BacktestApp._comparison_finite(leader.get("score"))
                if leader is not None else None)
            leader_effective = (
                BacktestApp._comparison_safe_int(
                    leader.get("rolling_windows"), 0)
                if leader is not None else 0)
            if leader_score is None or leader_effective <= 0:
                leader = None

            if leader is None:
                context = _sampling_context(context_row)
                rows.append({
                    "lookback": key, "period": display, "strategy": "—",
                    "strategy_label": "—", "score": None,
                    "baseline_score": None, "improvement_vs_c2c": None,
                    "selection_improvement_vs_c2c": None,
                    "gap_ratio": None, "window_win_rate_vs_c2c": None,
                    "paired": 0, "baseline_windows": 0,
                    "effective": 0,
                    "eligible": BacktestApp._comparison_safe_int(
                        (context_row or {}).get("eligible_endpoints"), 0),
                    "skipped": BacktestApp._comparison_safe_int(
                        (context_row or {}).get("skipped_endpoints"), 0),
                    "available": BacktestApp._comparison_safe_int(
                        (context_row or {}).get(
                            "history_days_available", 0), 0),
                    "requested": BacktestApp._comparison_safe_int(
                        (context_row or {}).get("lookback_days", 0), 0),
                    "status": "无可评估代理段", "formal": False,
                    "best_is_baseline": False,
                    "has_comparable_candidate": False,
                    "trailing_dropped": BacktestApp._comparison_safe_int(
                        (context_row or {}).get(
                            "trailing_partial_groups_dropped"), 0),
                    **context,
                })
                continue

            formal = formal_row is not None
            strategy = str(leader.get("strategy", "—"))
            best_is_baseline = (
                str(leader.get("strategy_type", "")) == "close_to_close")
            baseline_score = BacktestApp._comparison_finite(
                leader.get("baseline_score"))
            if baseline_score is None and baseline:
                baseline_score = BacktestApp._comparison_finite(
                    baseline.get("score"))
            improvement = BacktestApp._history_row_improvement(
                leader, baseline)
            uses_strict_metric = (
                BacktestApp._history_row_uses_strict_metric(leader))
            paired = BacktestApp._comparison_safe_int(
                leader.get("paired_windows", leader.get("rolling_windows")), 0)
            baseline_windows = BacktestApp._comparison_safe_int(
                leader.get("baseline_windows", paired), paired)
            win_rate = BacktestApp._comparison_finite(
                leader.get("window_win_rate_vs_c2c"))
            if not has_comparable_candidate:
                strategy_label = "仅基准（无完整配对候选）"
                status = "无完整配对候选"
            elif best_is_baseline:
                strategy_label = "每日收盘（基准最优）"
                if not formal:
                    strategy_label = f"诊断：{strategy_label}"
                if uses_strict_metric:
                    status = "严格区间完整" if formal else "区间不足，仅诊断"
                else:
                    status = "旧口径样本完整" if formal else "旧口径，仅诊断"
            else:
                strategy_label = strategy if formal else f"诊断领先：{strategy}"
                if uses_strict_metric:
                    status = "严格区间完整" if formal else "区间不足，仅诊断"
                else:
                    status = "旧口径样本完整" if formal else "旧口径，仅诊断"
            rows.append({
                "lookback": key,
                "period": display,
                "strategy": strategy,
                "strategy_label": strategy_label,
                "score": leader_score,
                "baseline_score": baseline_score,
                "improvement_vs_c2c": improvement,
                "selection_improvement_vs_c2c": (
                    improvement if uses_strict_metric else None),
                # 旧展示模型字段保留同值软兼容；其语义已固定为较 C2C 改善。
                "gap_ratio": improvement,
                "window_win_rate_vs_c2c": win_rate,
                "paired": paired,
                "baseline_windows": baseline_windows,
                "effective": paired or leader_effective,
                "eligible": BacktestApp._comparison_safe_int(
                    leader.get("eligible_endpoints"), 0),
                "skipped": BacktestApp._comparison_safe_int(
                    leader.get("skipped_endpoints"), 0),
                "available": BacktestApp._comparison_safe_int(leader.get(
                    "history_days_available", leader.get("days_used")), 0),
                "requested": BacktestApp._comparison_safe_int(
                    leader.get("lookback_days"), 0),
                "status": status,
                "formal": formal,
                "best_is_baseline": best_is_baseline,
                "has_comparable_candidate": has_comparable_candidate,
                "trailing_dropped": BacktestApp._comparison_safe_int(
                    leader.get("trailing_partial_groups_dropped"), 0),
                **_sampling_context(leader),
            })
        return rows

    @staticmethod
    def _comparison_delta_display(selected, baseline, key, *, digits=2,
                                  percent=True):
        metric = BacktestApp._comparison_relative_delta(
            selected, baseline, key)
        value_text = BacktestApp._format_comparison_value(
            metric["value"], digits)
        delta = metric["delta"]
        if delta is None:
            return value_text
        is_baseline = BacktestApp._comparison_rows_match(selected, baseline)
        if is_baseline:
            return f"{value_text}（基准）"
        if abs(delta) < 0.5 * 10 ** (-digits):
            return f"{value_text}（与基准相同）"
        delta_text = BacktestApp._format_comparison_value(
            delta, digits, signed=True)
        ratio_text = BacktestApp._format_comparison_value(
            metric["ratio"], 1, signed=True, percent=True)
        return (f"{value_text}  |  {delta_text}"
                + (f" ({ratio_text})" if percent and ratio_text != "—" else ""))

    @staticmethod
    def _make_comparison_stat_card(parent, column, title, value, subtitle="",
                                   *, highlight=False, baseline=False):
        if baseline:
            bg, border = PALETTE["primary_light"], PALETTE["primary"]
        elif highlight:
            bg, border = PALETTE["success_light"], PALETTE["success"]
        else:
            bg, border = PALETTE["surface_alt"], PALETTE["border_soft"]
        card = tk.Frame(
            parent, bg=bg, highlightbackground=border, highlightthickness=1,
            padx=9, pady=5,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 4, 0))
        tk.Label(
            card, text=title, bg=bg, fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 8), anchor="w",
        ).pack(fill="x")
        tk.Label(
            card, text=value, bg=bg, fg=PALETTE["text"],
            font=(_UI_FONT_FAMILY, 11, "bold"), anchor="w",
            justify="left", wraplength=165,
        ).pack(fill="x")
        if subtitle:
            tk.Label(
                card, text=subtitle, bg=bg, fg=PALETTE["text_muted"],
                font=(_UI_FONT_FAMILY, 8), anchor="w",
                justify="left", wraplength=165,
            ).pack(fill="x")

    def _build_comparison_stat_strip(
            self, parent, summary, *, lead_title="固定基准",
            count_label="可比策略"):
        headline = self._comparison_headline(summary)
        best = headline["best"] or {}
        baseline = self._comparison_baseline(summary) or {}
        self._comparison_best_row = best
        self._comparison_baseline_row = baseline

        stat_strip = ttk.Frame(parent, style="Surface.TFrame")
        stat_strip.pack(fill="x", padx=8, pady=(8, 4))
        for column in range(4):
            stat_strip.columnconfigure(column, weight=1, uniform="comparison_stats")
        baseline_name = str(baseline.get(
            "strategy", "缺少 close-to-close 结果"))
        self._make_comparison_stat_card(
            stat_strip, 0, lead_title, baseline_name,
            (f"close-to-close · 共 {headline['strategy_count']} 个{count_label}"
             if baseline else "请先运行并保留每日收盘回测"),
            baseline=True)
        self._make_comparison_stat_card(
            stat_strip, 1, "基准日净 PnL RMS（已含成本）",
            self._format_comparison_value(baseline.get("score"), 2),
            "金额口径 · 越低越好")
        baseline_score = self._comparison_finite(baseline.get("score"))
        best_score = self._comparison_finite(best.get("score"))
        improvement = None
        if (baseline_score is not None and best_score is not None
                and baseline_score != 0):
            improvement = (baseline_score - best_score) / abs(baseline_score)
        best_name = str(best.get("strategy", "无可比较结果"))
        if not baseline:
            improvement_note = "缺少基准，无法计算改善"
        elif self._comparison_rows_match(best, baseline):
            improvement_note = "基准即当前最优"
        else:
            improvement_note = (
                "较基准改善 "
                + self._format_comparison_value(
                    improvement, 1, signed=True, percent=True)
            )
        self._make_comparison_stat_card(
            stat_strip, 2, "当前最优", best_name, improvement_note,
            highlight=True)
        baseline_cost = self._format_comparison_value(
            baseline.get("total_tc"), 2)
        baseline_rehedges = self._comparison_safe_int(
            baseline.get("rehedge_count"), 0)
        baseline_actual_trades = self._comparison_safe_int(
            baseline.get("actual_trade_count"), 0)
        self._make_comparison_stat_card(
            stat_strip, 3, "基准总成本 / 再触发",
            f"{baseline_cost} / {baseline_rehedges}",
            f"实际成交 {baseline_actual_trades} 次")
        return headline

    def _open_saved_comparison(self):
        """打开会话结果池；勾选只读取快照，不触发任何回测。"""
        if getattr(self, "_active_job", None) is not None:
            messagebox.showinfo("任务运行中", "请等待当前任务完成后再打开结果对比。")
            return
        self._show_saved_comparison_page()

    def _show_saved_comparison_page(self):
        self._hide_placeholder("compare")
        old_figure = getattr(self, "_comparison_chart_figure", None)
        if old_figure is not None:
            plt.close(old_figure)
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
            font=(_UI_FONT_FAMILY, 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=("仅保存在当前应用会话；同方向最早保留的 close-to-close "
                  "结果固定为基准，勾选结果不会重新回测"),
            style="SurfaceMuted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        pool = ttk.LabelFrame(container, text=" 已保留结果 ", padding=5)
        pool.pack(fill="x", padx=8, pady=(3, 4))
        toolbar = ttk.Frame(pool, style="Surface.TFrame")
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            toolbar,
            text=("选择其它结果时会自动带入同方向固定基准；买卖方向分组"
                  "展示，点击其它列聚焦后可重命名或删除"),
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            toolbar, text="全选", width=6,
            command=self._select_all_saved_backtests,
        ).pack(side="left", padx=(4, 0))
        ttk.Button(
            toolbar, text="清空", width=6,
            command=self._clear_saved_backtest_selection,
        ).pack(side="left", padx=(4, 0))
        ttk.Button(
            toolbar, text="重命名", width=7,
            command=self._prompt_rename_saved_backtest,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            toolbar, text="删除", width=6,
            command=self._prompt_delete_saved_backtest,
        ).pack(side="left", padx=(4, 0))

        columns = ("shown", "name", "strategy", "parameters", "source", "saved_at")
        headings = {
            "shown": "显示", "name": "结果名称", "strategy": "策略",
            "parameters": "策略参数", "source": "行情来源", "saved_at": "保留时间",
        }
        widths = {
            "shown": 50, "name": 150, "strategy": 112,
            "parameters": 238, "source": 145, "saved_at": 115,
        }
        tree_frame = ttk.Frame(pool, style="Surface.TFrame")
        tree_frame.pack(fill="x")
        tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=5,
            selectmode="browse",
        )
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(
                column, width=widths[column], minwidth=45,
                anchor="center" if column in ("shown", "strategy", "saved_at") else "w",
                stretch=column in ("name", "parameters"),
            )
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")
        tree.bind("<Button-1>", self._toggle_saved_backtest_click)
        tree.bind("<space>", self._toggle_focused_saved_backtest)
        self._saved_pool_tree = tree

        self._saved_comparison_notice = tk.Frame(
            container, bg=PALETTE["primary_light"], padx=8, pady=4)
        self._saved_comparison_notice.pack(fill="x", padx=8, pady=(0, 3))
        self._saved_comparison_notice_label = tk.Label(
            self._saved_comparison_notice, text="", anchor="w", justify="left",
            bg=PALETTE["primary_light"], fg=PALETTE["primary"],
            font=(_UI_FONT_FAMILY, 8), wraplength=850,
        )
        self._saved_comparison_notice_label.pack(fill="x")

        self._saved_comparison_content = ttk.Frame(
            container, style="Surface.TFrame")
        self._saved_comparison_content.pack(
            fill="both", expand=True, padx=2, pady=(0, 6))
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()
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

    def _refresh_saved_pool_tree(self):
        tree = getattr(self, "_saved_pool_tree", None)
        if tree is None:
            return
        baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(self)
        focus = tree.focus()
        for item in tree.get_children():
            tree.delete(item)
        for snapshot in self._saved_backtests.values():
            tree.insert(
                "", "end", iid=snapshot.result_id,
                values=(
                    "☑" if snapshot.result_id in self._saved_comparison_selection else "☐",
                    snapshot.name,
                    (f"{snapshot.strategy_label} · 基准"
                     if snapshot.result_id == baseline_id
                     else snapshot.strategy_label),
                    snapshot.parameter_summary, snapshot.source_label,
                    snapshot.saved_at.strftime("%m-%d %H:%M:%S"),
                ),
            )
        children = tree.get_children()
        if focus in children:
            tree.selection_set(focus)
            tree.focus(focus)
        elif children:
            tree.selection_set(children[-1])
            tree.focus(children[-1])
        count_var = getattr(self, "_saved_pool_count_var", None)
        if count_var is not None:
            count_var.set(
                f"回测结果池 · 已保存 {len(self._saved_backtests)} 条 · "
                f"当前显示 {len(self._saved_comparison_selection)} 条")

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
        if not result_id or tree.identify_column(event.x) != "#1":
            return None
        tree.selection_set(result_id)
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

    def _toggle_saved_backtest_selection(self, result_id):
        baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(self)
        if result_id in self._saved_comparison_selection:
            other_selected = self._saved_comparison_selection - {result_id}
            if result_id == baseline_id and other_selected:
                self._set_status(
                    "close-to-close 是固定基准；请先取消其它对比项，或点击清空。")
            else:
                self._saved_comparison_selection.remove(result_id)
        elif result_id in self._saved_backtests:
            target_position = BacktestApp._saved_snapshot_position(
                self._saved_backtests[result_id])
            selected_positions = {
                BacktestApp._saved_snapshot_position(
                    self._saved_backtests[selected_id])
                for selected_id in self._saved_comparison_selection
                if selected_id in self._saved_backtests
            }
            if selected_positions and selected_positions != {target_position}:
                # 点击另一方向时切换整个视图分组，绝不沿用上一组的 C2C。
                self._saved_comparison_selection.clear()
                direction = (
                    "卖出(short)" if target_position == 1
                    else "买入(long)")
                self._set_status(
                    f"已切换到{direction}结果组；买卖方向不混合比较。")
            self._saved_comparison_selection.add(result_id)
            baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(
                self, target_position)
            if baseline_id is not None:
                self._saved_comparison_selection.add(baseline_id)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _select_all_saved_backtests(self):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        if not self._saved_backtests:
            self._saved_comparison_selection.clear()
            self._refresh_saved_pool_tree()
            self._refresh_saved_comparison_view()
            return
        focused_id = BacktestApp._focused_saved_backtest_id(self)
        if focused_id in self._saved_backtests:
            target_position = BacktestApp._saved_snapshot_position(
                self._saved_backtests[focused_id])
        else:
            selected_positions = {
                BacktestApp._saved_snapshot_position(
                    self._saved_backtests[result_id])
                for result_id in self._saved_comparison_selection
                if result_id in self._saved_backtests
            }
            target_position = (
                next(iter(selected_positions))
                if len(selected_positions) == 1 else
                BacktestApp._saved_snapshot_position(
                    next(iter(self._saved_backtests.values())))
            )
        same_direction = {
            result_id for result_id, snapshot in self._saved_backtests.items()
            if BacktestApp._saved_snapshot_position(snapshot)
            == target_position
        }
        skipped = len(self._saved_backtests) - len(same_direction)
        self._saved_comparison_selection = same_direction
        BacktestApp._resolve_saved_comparison_baseline_id(
            self, target_position)
        if skipped:
            direction = (
                "卖出(short)" if target_position == 1 else "买入(long)")
            self._set_status(
                f"已全选{direction}结果；另 {skipped} 条相反方向结果未混入。")
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _clear_saved_backtest_selection(self):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        self._saved_comparison_selection.clear()
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _focused_saved_backtest_id(self):
        tree = getattr(self, "_saved_pool_tree", None)
        if tree is None:
            return None
        selection = tree.selection()
        return selection[0] if selection else (tree.focus() or None)

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

    def _prompt_delete_saved_backtest(self):
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshot = self._saved_backtests[result_id]
        if not messagebox.askyesno(
                "删除回测结果", f"确定从当前会话删除『{snapshot.name}』吗？"):
            return
        self._delete_saved_backtest(result_id)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()
        self._set_status(
            f"已删除『{snapshot.name}』  |  剩余 {len(self._saved_backtests)} 条")

    def _selected_saved_backtests(self):
        selected_positions = {
            BacktestApp._saved_snapshot_position(
                self._saved_backtests[result_id])
            for result_id in self._saved_comparison_selection
            if result_id in self._saved_backtests
        }
        if len(selected_positions) > 1:
            raise ValueError(
                "买入(long)与卖出(short)结果不能混选；"
                "请清空后选择同一头寸方向。")
        selected_position = (
            next(iter(selected_positions)) if selected_positions else None)
        baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(
            self, selected_position)
        if self._saved_comparison_selection and baseline_id is not None:
            self._saved_comparison_selection.add(baseline_id)
        return [
            snapshot for result_id, snapshot in self._saved_backtests.items()
            if result_id in self._saved_comparison_selection
        ]

    def _refresh_saved_comparison_notice(self, snapshots):
        baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(self)
        warnings = self._saved_comparison_warnings(
            snapshots, baseline_result_id=baseline_id)
        if warnings:
            text = "⚠ " + "  ".join(warnings)
            mismatch = len(snapshots) > 1
            bg = PALETTE["warning_light"] if mismatch else PALETTE["primary_light"]
            fg = PALETTE["warning"] if mismatch else PALETTE["primary"]
        elif len(snapshots) >= 2:
            text = "✓ 所选结果的行情路径、期权参数及金额/成本口径一致。"
            bg, fg = PALETTE["success_light"], PALETTE["success"]
        else:
            text = "点击结果池“显示”列的复选框，选择要放入图表和排名的回测结果。"
            bg, fg = PALETTE["primary_light"], PALETTE["primary"]
        notice = self._saved_comparison_notice
        label = self._saved_comparison_notice_label
        notice.configure(bg=bg)
        label.configure(text=text, bg=bg, fg=fg)

    def _refresh_saved_comparison_view(self):
        try:
            snapshots = self._selected_saved_backtests()
        except Exception as exc:
            messagebox.showerror("选择已保存结果失败", str(exc))
            return
        self._refresh_saved_comparison_notice(snapshots)
        content = self._saved_comparison_content
        if not snapshots:
            old_figure = getattr(self, "_comparison_chart_figure", None)
            if old_figure is not None:
                plt.close(old_figure)
            for widget in content.winfo_children():
                widget.destroy()
            empty = ttk.Frame(content, style="Surface.TFrame")
            empty.place(relx=0.5, rely=0.42, anchor="center")
            ttk.Label(
                empty, text="尚未选择对比结果", style="Surface.TLabel",
                font=(_UI_FONT_FAMILY, 14, "bold"),
            ).pack(pady=(0, 5))
            ttk.Label(
                empty,
                text=("先运行回测并保留结果，或在上方结果池勾选已有结果。\n"
                      "修改策略或参数后的下一次回测不会覆盖已保留快照。"),
                style="SurfaceMuted.TLabel", justify="center",
            ).pack()
            return

        try:
            baseline_id = BacktestApp._resolve_saved_comparison_baseline_id(
                self)
            summary, daily_curves = self._saved_comparison_payload(
                snapshots, baseline_result_id=baseline_id)
        except Exception as exc:
            messagebox.showerror("读取已保存结果失败", str(exc))
            return
        old_figure = getattr(self, "_comparison_chart_figure", None)
        if old_figure is not None:
            plt.close(old_figure)
        for widget in content.winfo_children():
            widget.destroy()
        self._comparison_results = {}
        self._comparison_ranking = None
        self._build_comparison_stat_strip(
            content, summary, lead_title="固定基准", count_label="已选结果")
        current = ttk.Frame(content, style="Surface.TFrame")
        current.pack(fill="both", expand=True, padx=8, pady=(1, 0))
        self._build_current_comparison_view(
            current, summary, {}, show_curve_controls=False,
            ranking_title="已选结果排名（口径不一致时仅作场景对照）",
            daily_curves=daily_curves)

    def _show_history_recommendation(
            self, recommendations, ranking, notes=None, source_label=None,
            window_results=None, history_state=None):
        """只在历史择优页渲染严格区间结论与排名。"""
        container = self._history_results_container
        view_attrs = (
            "_history_ranking", "_history_period_rows",
            "_history_period_tree", "_history_detail_var",
            "_history_rank_tree", "_history_rank_rows",
            "_history_window_summary",
            "_history_chart_figure", "_history_chart_ax",
            "_history_chart_canvas", "_history_chart_mode_var",
            "_history_chart_metric_var", "_history_chart_window_var",
            "_history_chart_mode_combo", "_history_chart_metric_combo",
            "_history_chart_window_combo", "_history_chart_window_labels",
            "_history_chart_hint_var",
            "_history_chart_selected_by_period",
            "_history_chart_color_map", "_history_chart_marker_map",
            "_history_lookbacks", "_history_uses_strict_metric",
            "_history_uses_window_equal_metric",
        )
        missing = object()
        old_view_state = {
            name: getattr(self, name, missing) for name in view_attrs
        }
        staging = ttk.Frame(container, style="Surface.TFrame")
        self._history_ranking = ranking
        mode_bar = ttk.Frame(staging, style="Surface.TFrame")
        mode_bar.pack(fill="x", padx=4, pady=(3, 0))
        ttk.Label(
            mode_bar,
            text=("严格区间历史结果"
                  + (f"  ·  {source_label}" if source_label else "")),
            style="SurfaceMuted.TLabel",
        ).pack(side="left")
        if notes:
            first_note = str(notes[0])
            short_note = first_note if len(first_note) <= 92 else first_note[:89] + "…"
            note_bar = tk.Frame(staging, bg=PALETTE["warning_light"], padx=7, pady=3)
            note_bar.pack(fill="x", padx=8, pady=(0, 4))
            tk.Label(
                note_bar, text=f"⚠ {len(notes)} 条说明：{short_note}",
                bg=PALETTE["warning_light"], fg=PALETTE["warning"],
                font=(_UI_FONT_FAMILY, 8), anchor="w",
            ).pack(side="left", fill="x", expand=True)
            ttk.Button(
                note_bar, text="查看全部", width=8,
                command=lambda: messagebox.showinfo(
                    "历史择优说明", "\n\n".join(str(note) for note in notes)),
            ).pack(side="right", padx=(6, 0))

        history_body = ttk.Frame(staging, style="Surface.TFrame")
        history_body.pack(fill="both", expand=True, padx=4, pady=(2, 2))
        history_lookbacks = (
            history_state.get("history_lookbacks")
            if isinstance(history_state, Mapping) else None)
        try:
            self._build_history_comparison_view(
                history_body, recommendations, ranking, window_results,
                history_lookbacks=history_lookbacks)
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

        # 新视图完整构建后再一次性替换，渲染失败不破坏上次成功页面。
        old_figure = old_view_state.get("_history_chart_figure", missing)
        if (old_figure is not missing and old_figure is not None
                and old_figure is not self._history_chart_figure):
            plt.close(old_figure)
        for widget in list(container.winfo_children()):
            if widget is not staging:
                widget.destroy()
        staging.pack(fill="both", expand=True)
        self._nb.select(self._history_tab)

    # 旧私有渲染入口不再使用 summary/results 覆盖回测对比页。
    def _show_strategy_comparison(
            self, _summary, _results, recommendations=None, ranking=None,
            notes=None, window_results=None, source_label=None):
        return self._show_history_recommendation(
            recommendations, ranking, notes, source_label, window_results)

    def _build_current_comparison_view(
            self, parent, summary, results, *, show_curve_controls=True,
            ranking_title="当前路径排名（金额与持仓口径一致）",
            daily_curves=None):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)

        chart_box = ttk.LabelFrame(
            parent, text=" 累计净 PnL（按真实交易日汇总，已扣成本） ", padding=5)
        chart_box.grid(row=0, column=0, sticky="nsew", pady=(4, 3))
        controls = ttk.Frame(chart_box, style="Surface.TFrame")
        controls.pack(fill="x", pady=(0, 2))
        ttk.Label(
            controls,
            text=("显示曲线:" if show_curve_controls else
                  f"已选择 {len(daily_curves or {})} 条保存结果："),
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(2, 5))
        curve_choices = None
        if show_curve_controls:
            curve_choices = ttk.Frame(chart_box, style="Surface.TFrame")
            curve_choices.pack(fill="x", padx=2, pady=(0, 2))
            for column in range(2):
                curve_choices.columnconfigure(column, weight=1, uniform="curve_choices")

        if daily_curves is None:
            self._comparison_daily_curves = {
                name: result_daily_frame(result)
                for name, result in (results or {}).items()
            }
        else:
            self._comparison_daily_curves = dict(daily_curves)
        color_cycle = [
            PALETTE["primary"], PALETTE["accent"], PALETTE["warning"],
            PALETTE["success"], PALETTE["danger"],
            "#7C3AED", "#0F766E", "#DB2777", "#475569", "#65A30D",
        ]
        self._comparison_color_map = {
            name: color_cycle[index % len(color_cycle)]
            for index, name in enumerate(self._comparison_daily_curves)
        }
        self._comparison_curve_vars = {}
        for index, name in enumerate(self._comparison_daily_curves):
            variable = tk.BooleanVar(value=True)
            self._comparison_curve_vars[name] = variable
            if show_curve_controls:
                tk.Checkbutton(
                    curve_choices, text=name, variable=variable,
                    command=self._draw_comparison_cumulative_chart,
                    bg=PALETTE["surface"], fg=self._comparison_color_map[name],
                    activebackground=PALETTE["surface"],
                    activeforeground=self._comparison_color_map[name],
                    selectcolor=PALETTE["surface"],
                    font=(_UI_FONT_FAMILY, 8), padx=2,
                    anchor="w",
                ).grid(
                    row=index // 2, column=index % 2,
                    sticky="w", padx=(0, 8), pady=(0, 1),
                )
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
            lower, text=f" {ranking_title} ", padding=5)
        result_box.grid(row=0, column=0, sticky="nsew")
        columns = (
            "rank", "strategy", "score", "gap", "total_net_pnl",
            "total_tc", "rehedge_count", "max_drawdown",
        )
        headings = {
            "rank": "#", "strategy": "策略 / 参数",
            "score": "日净PnL RMS↓", "gap": "相对C2C基准",
            "total_net_pnl": "期末净PnL", "total_tc": "总成本↓",
            "rehedge_count": "再触发/成交", "max_drawdown": "最大回撤↓",
        }
        widths = {
            "rank": 36, "strategy": 145, "score": 78, "gap": 92,
            "total_net_pnl": 78, "total_tc": 62,
            "rehedge_count": 68, "max_drawdown": 70,
        }
        tree = ttk.Treeview(
            result_box, columns=columns, show="headings",
            height=max(3, min(5, len(summary))), selectmode="browse",
        )
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(
                col, width=widths[col], minwidth=35,
                anchor="w" if col == "strategy" else "e",
                stretch=(col == "strategy"),
            )
        tree.pack(fill="both", expand=True)
        tree.tag_configure("best", background=PALETTE["success_light"])
        tree.tag_configure("baseline", background=PALETTE["primary_light"])
        tree.tag_configure("baseline_best", background=PALETTE["success_light"])
        tree.tag_configure("even", background=PALETTE["surface_alt"])

        self._comparison_rows = {}
        baseline = getattr(self, "_comparison_baseline_row", {}) or {}
        baseline_score = self._comparison_finite(baseline.get("score"))
        baseline_iid = None
        for row_no, (_idx, row) in enumerate(summary.iterrows()):
            item = row.to_dict()
            iid = f"strategy_{row_no}"
            score = self._comparison_finite(item.get("score"))
            gap = None
            if score is not None and baseline_score not in (None, 0):
                gap = (score - baseline_score) / abs(baseline_score)
            is_baseline = self._comparison_rows_match(item, baseline)
            is_best = self._comparison_rows_match(
                item, self._comparison_best_row)
            gap_text = (
                "基准" if is_baseline
                else self._format_comparison_value(
                    gap, 1, signed=True, percent=True))
            values = (
                self._comparison_safe_int(
                    item.get("rank"), row_no + 1), item.get("strategy", ""),
                self._format_comparison_value(item.get("score"), 2), gap_text,
                self._format_comparison_value(item.get("total_net_pnl"), 2),
                self._format_comparison_value(item.get("total_tc"), 2),
                (f"{self._comparison_safe_int(item.get('rehedge_count'), 0)}/"
                 f"{self._comparison_safe_int(item.get('actual_trade_count'), 0)}"),
                self._format_comparison_value(item.get("max_drawdown"), 2),
            )
            tag = (
                "baseline_best" if is_baseline and is_best else
                "baseline" if is_baseline else
                "best" if is_best else
                "even" if row_no % 2 == 0 else ""
            )
            tree.insert("", "end", iid=iid, values=values,
                        tags=(tag,) if tag else ())
            self._comparison_rows[iid] = item
            if is_baseline:
                baseline_iid = iid
        self._comparison_tree = tree
        tree.bind("<<TreeviewSelect>>", self._update_comparison_selection)

        detail_box = ttk.LabelFrame(
            lower, text=" 所选策略相对 close-to-close 基准 ", padding=5)
        detail_box.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._comparison_detail_header_var = tk.StringVar(value="请选择策略")
        ttk.Label(
            detail_box, textvariable=self._comparison_detail_header_var,
            style="Surface.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 3))
        detail_specs = (
            ("score", "日净PnL RMS"), ("total_net_pnl", "期末净PnL"),
            ("total_tc", "总成本"), ("max_drawdown", "最大回撤"),
            ("rehedge_count", "再触发 / 成交"), ("turnover", "换手额"),
        )
        self._comparison_detail_vars = {}
        for index, (key, label) in enumerate(detail_specs):
            column = index % 3
            row_no = 1 + index // 3
            detail_box.columnconfigure(column, weight=1, uniform="comparison_detail")
            cell = ttk.Frame(detail_box, style="Surface.TFrame")
            cell.grid(
                row=row_no, column=column, sticky="ew",
                padx=(0 if column == 0 else 8, 0), pady=(0, 3),
            )
            ttk.Label(cell, text=label, style="SurfaceMuted.TLabel").pack(anchor="w")
            variable = tk.StringVar(value="—")
            self._comparison_detail_vars[key] = variable
            ttk.Label(cell, textvariable=variable, style="Surface.TLabel").pack(anchor="w")

        children = tree.get_children()
        if children:
            initial_iid = baseline_iid or children[0]
            tree.selection_set(initial_iid)
            tree.focus(initial_iid)
        self._update_comparison_selection()

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
        baseline_name = (getattr(
            self, "_comparison_baseline_row", {}) or {}).get("strategy")

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
            is_baseline = name == baseline_name
            ax.plot(
                x, y, label=name, color=self._comparison_color_map[name],
                linewidth=2.8 if focused else (2.2 if is_baseline else 1.7),
                alpha=1.0 if focused or focused_name is None else 0.58,
                linestyle="--" if is_baseline else "-",
                marker="o", markersize=3.5 if focused else 2.5,
            )
            plotted += 1

        ax.axhline(0.0, color=PALETTE["text_muted"], linewidth=0.8, alpha=0.7)
        ax.set_xlabel("交易日", fontsize=8)
        ax.set_ylabel("累计净 PnL", fontsize=8)
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
        tree = getattr(self, "_comparison_tree", None)
        if tree is None or not tree.selection():
            self._draw_comparison_cumulative_chart()
            return
        selected = self._comparison_rows.get(tree.selection()[0], {})
        baseline = getattr(self, "_comparison_baseline_row", {}) or {}
        description = selected.get("meta_description", "")
        header = (
            f"{selected.get('strategy', '—')}  ·  排名 "
            f"{self._comparison_safe_int(selected.get('rank'), 0)}")
        if description:
            header += f"  ·  {description}"
        self._comparison_detail_header_var.set(header)
        for key in (
                "score", "total_net_pnl", "total_tc", "max_drawdown",
                "turnover"):
            self._comparison_detail_vars[key].set(
                self._comparison_delta_display(selected, baseline, key))
        selected_rehedges = self._comparison_safe_int(
            selected.get("rehedge_count"), 0)
        baseline_rehedges = self._comparison_safe_int(
            baseline.get("rehedge_count"), 0)
        selected_actual = self._comparison_safe_int(
            selected.get("actual_trade_count"), 0)
        baseline_actual = self._comparison_safe_int(
            baseline.get("actual_trade_count"), 0)
        rehedge_delta = selected_rehedges - baseline_rehedges
        actual_delta = selected_actual - baseline_actual
        is_baseline = self._comparison_rows_match(selected, baseline)
        if not baseline:
            rehedge_text = f"{selected_rehedges}/{selected_actual}"
        elif is_baseline:
            rehedge_text = f"{selected_rehedges}/{selected_actual}（基准）"
        elif rehedge_delta == 0 and actual_delta == 0:
            rehedge_text = f"{selected_rehedges}/{selected_actual}（与基准相同）"
        else:
            rehedge_text = (
                f"{selected_rehedges}/{selected_actual}  |  "
                f"{rehedge_delta:+d}/{actual_delta:+d}")
        self._comparison_detail_vars["rehedge_count"].set(rehedge_text)
        self._draw_comparison_cumulative_chart()

    def _build_history_comparison_view(
            self, parent, recommendations, ranking, window_results=None,
            history_lookbacks=None):
        self._history_ranking = ranking
        self._history_lookbacks = BacktestApp._normalize_history_lookbacks(
            history_lookbacks)
        # 原始逐代理段结果含完整 bar 级价格、Greeks 与 PnL 数组；历史分钟数据下
        # 深拷贝并长期保留会占用大量内存。图表只保存压缩后的独立日级摘要，
        # worker 投递的原始结果在本次渲染结束后即可释放。
        self._history_window_summary = history_window_summary(
            window_results or {})
        self._history_chart_selected_by_period = {}
        chart_colors = (
            "#2563EB", "#D97706", "#7C3AED", "#0F766E",
            "#DB2777", "#DC2626", "#0891B2", "#65A30D",
            "#9333EA", "#C2410C", "#4F46E5", "#047857",
        )
        chart_markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
        candidate_names = []
        if (ranking is not None and not getattr(ranking, "empty", True)
                and {"strategy", "strategy_type"}.issubset(ranking.columns)):
            candidate_names = sorted(set(
                ranking.loc[
                    ~ranking["strategy_type"].astype(str).eq("close_to_close"),
                    "strategy",
                ].dropna().astype(str)
            ))
        self._history_chart_color_map = {
            strategy: chart_colors[index % len(chart_colors)]
            for index, strategy in enumerate(candidate_names)
        }
        self._history_chart_marker_map = {
            strategy: chart_markers[index % len(chart_markers)]
            for index, strategy in enumerate(candidate_names)
        }
        uses_strict_metric = bool(
            ranking is not None
            and not getattr(ranking, "empty", True)
            and "selection_metric" in ranking
            and "selection_improvement_vs_c2c" in ranking
            and ranking["selection_metric"].fillna("").astype(str).eq(
                STRICT_LOOKBACK_SELECTION_METRIC).all())
        uses_product_pool = bool(
            ranking is not None
            and not getattr(ranking, "empty", True)
            and "history_mode" in ranking
            and ranking["history_mode"].fillna("").astype(str).eq(
                "product_contract_pool").any())
        self._history_uses_strict_metric = uses_strict_metric
        self._history_uses_window_equal_metric = bool(
            ranking is not None
            and not getattr(ranking, "empty", True)
            and "selection_metric" in ranking
            and "selection_improvement_vs_c2c" in ranking
            and ranking["selection_metric"].fillna("").astype(str).eq(
                HISTORY_SELECTION_METRIC).all())
        selected_periods = [
            (key, label) for key, label in HISTORY_PERIOD_DEFS
            if key in self._history_lookbacks
        ]
        period_names = " / ".join(label for _key, label in selected_periods)
        if uses_strict_metric:
            scope_explanation = (
                f"本次所选严格连续区间：{period_names}。每档都从共同分析"
                "截止日向前回放最近 L 日；期权期限 T 只决定代理合约每段"
                "最长存续期，L>T 时自动续接代理段，最后不足 T 的尾段按"
                "剩余期限公允价值 MTM。realized σ 候选另要求区间前完整"
                " V 日 HV 预热；排名只使用同一严格区间内与 C2C 成功配对"
                "的证据日；")
            if uses_product_pool:
                scope_explanation += (
                    "品种池主指标为各非重叠代理段的有界 C2C RMS 改善按"
                    "实际评分日数加权，汇总金额 RMS 只作诊断；")
            else:
                scope_explanation += (
                    "主指标为完整 L 日合并日净 PnL RMS 的有界 C2C 改善；")
        else:
            scope_explanation = (
                f"本次所选周期：{period_names}。该结果来自升级前的历史终点"
                "抽样口径，不能解释为严格连续最近 L 日；建议重新运行以生成"
                "新版严格区间排名；")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(3, weight=2)
        ttk.Label(
            parent,
            text=(scope_explanation +
                  "短周期更贴近当前，长周期覆盖更多市场阶段、更稳健但反应更慢。"
                  "各周期结论仅作历史参考，不代表未来表现。"),
            style="SurfaceMuted.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(7, 3))

        rec_box = ttk.LabelFrame(parent, text=" 周期结论 ", padding=5)
        rec_box.grid(row=1, column=0, sticky="ew", padx=2, pady=3)
        columns = (
            "period", "strategy", "score", "baseline", "improvement",
            "win_rate", "windows", "status",
        )
        metric_labels = BacktestApp._history_metric_labels(
            uses_strict_metric=uses_strict_metric,
            uses_product_pool=uses_product_pool,
        )
        headings = {
            "period": "分析周期", "strategy": "历史最优参考",
            "score": metric_labels["score"],
            "baseline": metric_labels["baseline"],
            "improvement": metric_labels["improvement"],
            "win_rate": "代理段胜率",
            "windows": "配对代理段/C2C", "status": "历史资格",
        }
        widths = {
            "period": 58, "strategy": 170, "score": 98,
            "baseline": 98, "improvement": 88, "win_rate": 84,
            "windows": 100, "status": 105,
        }
        tree = ttk.Treeview(
            rec_box, columns=columns, show="headings",
            height=max(1, len(selected_periods)),
            selectmode="browse",
        )
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(
                col, width=widths[col], minwidth=40,
                anchor="w" if col in ("strategy", "status") else "center",
                stretch=(col == "strategy"),
            )
        tree.pack(fill="x")
        tree.tag_configure("formal", background=PALETTE["success_light"])
        tree.tag_configure("diagnostic", background=PALETTE["warning_light"])
        tree.tag_configure("empty", background=PALETTE["surface_alt"])

        self._history_period_rows = {}
        for row_no, item in enumerate(self._comparison_recommendation_rows(
                recommendations, ranking, self._history_lookbacks)):
            iid = f"history_{item['lookback']}"
            values = (
                item["period"], item["strategy_label"],
                self._format_comparison_value(item["score"], 2),
                self._format_comparison_value(
                    item["baseline_score"], 2),
                ("—" if not item["has_comparable_candidate"] else
                 "基准" if item["best_is_baseline"] else
                 self._format_comparison_value(
                     item["improvement_vs_c2c"], 1,
                     signed=True, percent=True)),
                ("—" if not item["has_comparable_candidate"] else
                 "基准" if item["best_is_baseline"] else
                 self._format_comparison_value(
                     item["window_win_rate_vs_c2c"], 1, percent=True)),
                f"{item['paired']}/{item['baseline_windows']}",
                item["status"],
            )
            tag = "formal" if item["formal"] else (
                "empty" if item["strategy"] == "—" else "diagnostic")
            tree.insert("", "end", iid=iid, values=values, tags=(tag,))
            self._history_period_rows[iid] = item
        self._history_period_tree = tree

        chart_box = ttk.LabelFrame(
            parent,
            text=(" 多策略共同代理段 PnL 图表（C2C 固定；"
                  f"候选最多 {MAX_HISTORY_CHART_CANDIDATES} 项） "),
            padding=5,
        )
        chart_box.grid(row=3, column=0, sticky="nsew", padx=2, pady=(3, 2))
        controls = ttk.Frame(chart_box, style="Surface.TFrame")
        controls.pack(fill="x", pady=(0, 2))
        ttk.Label(
            controls, text="模式:", style="SurfaceMuted.TLabel",
        ).pack(side="left")
        self._history_chart_mode_var = tk.StringVar(
            value=HISTORY_CHART_MODE_DISPLAY["full"])
        self._history_chart_mode_combo = ttk.Combobox(
            controls, textvariable=self._history_chart_mode_var,
            values=tuple(HISTORY_CHART_MODE_DISPLAY.values()),
            width=15, state="readonly",
        )
        self._history_chart_mode_combo.pack(side="left", padx=(5, 12))
        self._history_chart_mode_combo.bind(
            "<<ComboboxSelected>>", self._update_history_chart_controls)

        ttk.Label(
            controls, text="口径:", style="SurfaceMuted.TLabel",
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

        ttk.Label(
            controls, text="代理段:", style="SurfaceMuted.TLabel",
        ).pack(side="left")
        self._history_chart_window_var = tk.StringVar(value="")
        self._history_chart_window_combo = ttk.Combobox(
            controls, textvariable=self._history_chart_window_var,
            values=(), width=35, state="readonly",
        )
        self._history_chart_window_combo.pack(side="left", padx=(5, 8))
        self._history_chart_window_combo.bind(
            "<<ComboboxSelected>>", self._draw_history_chart)
        self._history_chart_window_labels = {}
        self._history_chart_hint_var = tk.StringVar(value="")
        ttk.Label(
            controls, textvariable=self._history_chart_hint_var,
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._history_chart_figure = Figure(
            figsize=(7.2, 2.15), dpi=self._CHART_DPI,
            facecolor=PALETTE["surface"], constrained_layout=True,
        )
        self._history_chart_ax = self._history_chart_figure.add_subplot(111)
        self._history_chart_canvas = FigureCanvasTkAgg(
            self._history_chart_figure, master=chart_box)
        self._history_chart_canvas.get_tk_widget().pack(
            fill="both", expand=True)

        detail_box = ttk.LabelFrame(
            parent, text=" 周期内策略 vs C2C 排名 ", padding=5)
        detail_box.grid(row=2, column=0, sticky="nsew", padx=2, pady=(3, 2))
        detail_box.columnconfigure(0, weight=1)
        detail_box.rowconfigure(2, weight=1)
        self._history_detail_var = tk.StringVar(value="请选择周期")
        ttk.Label(
            detail_box, textvariable=self._history_detail_var,
            style="SurfaceMuted.TLabel", justify="left", wraplength=1100,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        chart_selection_bar = ttk.Frame(
            detail_box, style="Surface.TFrame")
        chart_selection_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        ttk.Label(
            chart_selection_bar,
            text=("图表显示：点击首列勾选；C2C 固定，"
                  f"候选最多 {MAX_HISTORY_CHART_CANDIDATES} 项"),
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            chart_selection_bar, text="图表前2名", width=9,
            command=self._select_top_history_chart_candidates,
        ).pack(side="right", padx=(4, 0))
        ttk.Button(
            chart_selection_bar, text="清空候选", width=8,
            command=self._clear_history_chart_candidates,
        ).pack(side="right", padx=(8, 0))
        rank_columns = (
            "shown", "rank", "strategy", "score", "baseline", "improvement",
            "win_rate", "windows", "status",
        )
        rank_headings = {
            "shown": "图表", "rank": "#", "strategy": "策略 / 参数",
            "score": metric_labels["score"],
            "baseline": metric_labels["baseline"],
            "improvement": metric_labels["improvement"],
            "win_rate": "代理段胜率", "windows": "配对代理段/C2C",
            "status": "比较资格",
        }
        rank_widths = {
            "shown": 48, "rank": 34, "strategy": 175, "score": 98,
            "baseline": 98,
            "improvement": 88, "win_rate": 84, "windows": 100,
            "status": 92,
        }
        rank_tree = ttk.Treeview(
            detail_box, columns=rank_columns, show="headings", height=6,
            selectmode="browse",
        )
        for col in rank_columns:
            rank_tree.heading(col, text=rank_headings[col])
            rank_tree.column(
                col, width=rank_widths[col], minwidth=40,
                anchor="w" if col in ("strategy", "status") else "center",
                stretch=(col == "strategy"),
            )
        rank_tree.grid(row=2, column=0, sticky="nsew")
        rank_scrollbar = ttk.Scrollbar(
            detail_box, orient="vertical", command=rank_tree.yview)
        rank_scrollbar.grid(row=2, column=1, sticky="ns")
        rank_tree.configure(yscrollcommand=rank_scrollbar.set)
        rank_tree.tag_configure("leader", background=PALETTE["success_light"])
        rank_tree.tag_configure("baseline", background=PALETTE["primary_light"])
        rank_tree.tag_configure("incomplete", background=PALETTE["warning_light"])
        self._history_rank_tree = rank_tree
        rank_tree.bind("<Button-1>", self._toggle_history_chart_click)
        rank_tree.bind("<space>", self._toggle_focused_history_chart_candidate)
        rank_tree.bind(
            "<<TreeviewSelect>>", self._update_history_rank_selection)

        result_actions = ttk.Frame(detail_box, style="Surface.TFrame")
        result_actions.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(
            result_actions,
            text=("历史参考只说明同一严格区间内代理段相对 C2C 的结果；"
                  "若当前期限已修改，路径验证会按当前 T 重新运行。"),
            style="SurfaceMuted.TLabel",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            result_actions, text="应用选中历史策略",
            command=self._apply_history_recommendation,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            result_actions, text="当前路径验证并保留到对比",
            style="Accent.TButton",
            command=self._run_history_selection_on_current_path,
        ).pack(side="left", padx=(6, 0))

        tree.bind("<<TreeviewSelect>>", self._update_history_selection)
        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            tree.focus(children[0])
            self._update_history_selection()

    def _history_chart_candidates(self, lookback=None):
        """按排名顺序返回当前周期勾选的图表候选，不包含固定 C2C。"""
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

    def _history_top_chart_candidates(self, limit=2):
        """选择排名靠前且能保持共同代理段交集的候选。"""
        lookback, _row = self._history_chart_selection()
        if not lookback:
            return []
        tree = getattr(self, "_history_rank_tree", None)
        rows = getattr(self, "_history_rank_rows", {})
        if tree is None:
            return []
        mode_var = getattr(self, "_history_chart_mode_var", None)
        metric_var = getattr(self, "_history_chart_metric_var", None)
        mode = HISTORY_CHART_MODE_FROM_DISPLAY.get(
            mode_var.get() if mode_var is not None else "", "full")
        metric = HISTORY_CHART_METRIC_FROM_DISPLAY.get(
            metric_var.get() if metric_var is not None else "", "net")
        selected = []
        for iid in tree.get_children():
            row = rows.get(iid, {})
            if str(row.get("strategy_type", "")) == "close_to_close":
                continue
            strategy = str(row.get("strategy", "")).strip()
            if not strategy:
                continue
            trial = [*selected, strategy]
            model = BacktestApp._history_multi_chart_model(
                getattr(self, "_history_window_summary", None),
                lookback, trial, mode=mode, metric=metric,
            )
            if model.get("state") == "ok":
                selected.append(strategy)
            if len(selected) >= min(limit, MAX_HISTORY_CHART_CANDIDATES):
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
            mark = (
                "基准" if is_baseline else
                "☑" if strategy in selected else
                "☐" if paired > 0 else "—"
            )
            tree.set(iid, "shown", mark)

    def _toggle_history_chart_click(self, event):
        tree = self._history_rank_tree
        iid = tree.identify_row(event.y)
        if not iid or tree.identify_column(event.x) != "#1":
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
            self._set_status("每日收盘 C2C 是图表固定基准，无需勾选或取消。")
            self._update_history_rank_selection()
            return
        pairs = BacktestApp._history_chart_pairs(
            getattr(self, "_history_window_summary", None),
            lookback, strategy)
        if pairs.empty:
            self._set_status(
                f"『{strategy}』没有可与 C2C 严格配对的图表代理段。")
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
            mode = HISTORY_CHART_MODE_FROM_DISPLAY.get(
                self._history_chart_mode_var.get(), "full")
            metric = HISTORY_CHART_METRIC_FROM_DISPLAY.get(
                self._history_chart_metric_var.get(), "net")
            preview = BacktestApp._history_multi_chart_model(
                getattr(self, "_history_window_summary", None),
                lookback, [*current_candidates, strategy],
                mode=mode, metric=metric,
            )
            if preview.get("state") != "ok":
                reason = preview.get("message", "没有共同的严格配对代理段")
                self._set_status(f"无法加入『{strategy}』：{reason}")
                self._update_history_rank_selection()
                return
            selected.add(strategy)
        self._refresh_history_chart_marks()
        self._update_history_rank_selection()

    def _select_top_history_chart_candidates(self):
        lookback, _row = self._history_chart_selection()
        if not lookback:
            return
        selected = set(self._history_top_chart_candidates(limit=2))
        self._history_chart_selected_by_period[str(lookback)] = selected
        self._refresh_history_chart_marks()
        self._update_history_chart_controls()
        self._set_status(
            f"图表已显示 C2C 与排名靠前的 {len(selected)} 个可比候选。")

    def _clear_history_chart_candidates(self):
        lookback, _row = self._history_chart_selection()
        if not lookback:
            return
        self._history_chart_selected_by_period[str(lookback)] = set()
        self._refresh_history_chart_marks()
        self._update_history_chart_controls()
        self._set_status("已清空图表候选；每日收盘 C2C 基准仍固定显示。")

    def _history_chart_selection(self):
        """返回当前历史周期与排名行。"""
        period_tree = getattr(self, "_history_period_tree", None)
        period_rows = getattr(self, "_history_period_rows", {})
        if period_tree is None or not period_tree.selection():
            return None, None
        period = period_rows.get(period_tree.selection()[0], {})
        return period.get("lookback"), self._selected_history_rank_row()

    def _update_history_chart_controls(self, _event=None):
        """按当前周期及图表候选刷新共同代理段和控件可用性。"""
        mode_display = self._history_chart_mode_var.get()
        # 旧热更新控件可能仍残留已删除的“逐窗优势”显示值；统一降级到
        # 与严格区间排名证据一致的完整 L 日累计路径。
        mode = HISTORY_CHART_MODE_FROM_DISPLAY.get(mode_display, "full")
        metric = HISTORY_CHART_METRIC_FROM_DISPLAY.get(
            self._history_chart_metric_var.get(), "net")
        lookback, _row = self._history_chart_selection()
        candidates = self._history_chart_candidates(lookback)
        model = BacktestApp._history_multi_chart_model(
            getattr(self, "_history_window_summary", None),
            lookback, candidates, mode=mode, metric=metric,
            primary_strategy=self._history_chart_primary_candidate(candidates),
        )
        labels = {
            str(item["label"]): str(item["window_id"])
            for item in model.get("window_options", ())
        }
        self._history_chart_window_labels = labels
        window_combo = self._history_chart_window_combo
        window_combo.configure(values=tuple(labels))
        current = self._history_chart_window_var.get()
        if current not in labels:
            self._history_chart_window_var.set(next(reversed(labels), ""))
        window_combo.configure(
            state="readonly" if mode == "single" and labels else "disabled")
        self._history_chart_metric_combo.configure(state="readonly")
        self._draw_history_chart()

    def _update_history_rank_selection(self, _event=None):
        """排名行变化时同步刷新相对 C2C 结论与两种历史图表。"""
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
                score_text = self._format_comparison_value(
                    row.get("score"), 2)
                baseline_text = self._format_comparison_value(
                    row.get("baseline_score"), 2)
                improvement = BacktestApp._history_row_improvement(row)
                improvement_text = (
                    "基准" if is_baseline else
                    self._format_comparison_value(
                        improvement, 1, signed=True, percent=True))
                win_rate_text = (
                    "基准" if is_baseline else
                    self._format_comparison_value(
                        row.get("window_win_rate_vs_c2c"), 1,
                        percent=True))
                paired = self._comparison_safe_int(
                    row.get("paired_windows", row.get("rolling_windows")), 0)
                baseline_windows = self._comparison_safe_int(
                    row.get("baseline_windows", paired), paired)
                uses_strict_metric = (
                    BacktestApp._history_row_uses_strict_metric(row))
                uses_product_pool = (
                    row.get("history_mode") == "product_contract_pool")
                rms_label = (
                    "汇总RMS(诊断)" if uses_product_pool
                    else "区间RMS" if uses_strict_metric
                    else "汇总RMS")
                selected_detail = (
                    f"所选={strategy} · {rms_label}="
                    f"{score_text}/C2C {baseline_text}")
                if uses_strict_metric:
                    improvement_label = (
                        "较C2C改善(逐段按实际天数加权)"
                        if uses_product_pool
                        else "较C2C改善(完整L日合并RMS)")
                    selected_detail += (
                        f" · {improvement_label}={improvement_text}"
                        f" · 代理段胜率={win_rate_text}")
                else:
                    selected_detail += (
                        f" · 旧结果RMS改善={improvement_text}"
                        " · 建议重新运行获取严格区间排名")
                selected_detail += (
                    f" · 配对代理段={paired}/C2C {baseline_windows}")
                if row.get("history_mode") == "product_contract_pool":
                    code_text = BacktestApp._history_contract_codes_text(
                        row.get("paired_contract_codes", ()))
                    if code_text:
                        selected_detail += f" · 所选参与合约={code_text}"
                    elif not uses_strict_metric:
                        selected_detail += " · 所选参与合约=旧结果未记录"
                failure_reason = str(
                    row.get("failure_reason") or "").strip()
                if failure_reason:
                    if len(failure_reason) > 96:
                        failure_reason = failure_reason[:93] + "…"
                    selected_detail += f" · 首个失败原因={failure_reason}"
                detail_var.set(
                    f"{prefix} · {selected_detail}" if prefix else selected_detail)
        self._update_history_chart_controls()

    def _draw_history_chart(self, _event=None):
        """绘制固定 C2C 加多个已勾选候选的严格配对代理段图表。"""
        ax = getattr(self, "_history_chart_ax", None)
        canvas = getattr(self, "_history_chart_canvas", None)
        if ax is None or canvas is None:
            return
        ax.clear()
        lookback, _focused_row = self._history_chart_selection()
        candidates = self._history_chart_candidates(lookback)
        primary_strategy = self._history_chart_primary_candidate(candidates)
        mode = HISTORY_CHART_MODE_FROM_DISPLAY.get(
            self._history_chart_mode_var.get(), "full")
        metric = HISTORY_CHART_METRIC_FROM_DISPLAY.get(
            self._history_chart_metric_var.get(), "net")
        window_label = self._history_chart_window_var.get()
        window_id = getattr(
            self, "_history_chart_window_labels", {}).get(window_label)
        model = BacktestApp._history_multi_chart_model(
            getattr(self, "_history_window_summary", None),
            lookback, candidates, mode=mode, metric=metric,
            window_id=window_id,
            primary_strategy=primary_strategy,
        )

        if model.get("state") != "ok":
            message = model.get("message", "暂无可绘制的配对代理段。")
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
            "；共同代理段少于个别策略的独立配对代理段，与排名表口径不同"
            if model.get("sample_scope_differs") else "")
        if mode in {"full", "single"}:
            candidate_index = 0
            for series in model["series"]:
                baseline = series["role"] == "baseline"
                strategy = str(series.get("strategy", series.get("label", "")))
                if baseline:
                    color, marker = baseline_color, None
                else:
                    color, marker = _candidate_style(
                        strategy, candidate_index)
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
            ax.set_xlabel(
                "严格区间交易日" if mode == "full" else "代理段内交易日",
                fontsize=8)
            ax.set_ylabel(model["metric_label"], fontsize=8)
            if mode == "full":
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
                    "完整" if model.get("complete_evidence")
                    else "部分共同交集")
                self._history_chart_hint_var.set(
                    f"共同代理段 {common_count} 个 · {completeness} "
                    f"{common_days}/{expected_days} 日 · "
                    f"C2C + {len(candidates)} 个候选；"
                    f"失败段不补零{scope_note}")
                if model.get("uses_normalized_notional"):
                    self._history_chart_hint_var.set(
                        self._history_chart_hint_var.get()
                        + "；跨合约按各段 "
                        "PnL/(S0×乘数×|数量|) 安全归一化后拼接")
            else:
                self._history_chart_hint_var.set(
                    f"共同代理段 {common_count} 个 · "
                    f"当前={window_label or '—'} · "
                    f"C2C + {len(candidates)} 个候选；"
                    f"净PnL = 成本前PnL - 成本{scope_note}")
        elif mode == "typical":
            candidate_index = 0
            for band in model["bands"]:
                baseline = band["role"] == "baseline"
                strategy = str(band.get("strategy", band.get("label", "")))
                if baseline:
                    color, marker = baseline_color, None
                else:
                    color, marker = _candidate_style(
                        strategy, candidate_index)
                    candidate_index += 1
                emphasized = strategy == primary_strategy
                show_interval = bool(
                    band.get("show_interval")
                    and (baseline or emphasized))
                if show_interval:
                    ax.fill_between(
                        band["x"], band["p25"], band["p75"],
                        color=color, alpha=0.13, linewidth=0,
                    )
                ax.plot(
                    band["x"], band["median"], label=band["label"],
                    color=color, linestyle="--" if baseline else "-",
                    linewidth=(2.1 if baseline else 2.5 if emphasized else 1.8),
                    marker=marker, markersize=3.2,
                    markevery=(None if baseline else max(
                        1, len(band["median"]) // 9)),
                    alpha=1.0 if baseline or emphasized else 0.86,
                )
            ax.set_xlabel("代理段内交易日", fontsize=8)
            ax.set_ylabel(model["metric_label"], fontsize=8)
            interval_text = (
                "阴影仅显示 C2C 与主候选的 P25/P75（非置信区间）"
                if common_count >= 2 else "仅1个共同代理段，不绘制区间带")
            self._history_chart_hint_var.set(
                f"共同代理段 {common_count} 个 · "
                f"C2C + {len(candidates)} 个候选；"
                f"{interval_text}；提前结算后累计值保持不变{scope_note}")
            if model.get("uses_normalized_notional"):
                self._history_chart_hint_var.set(
                    self._history_chart_hint_var.get()
                    + "；品种池多代理段路径按 "
                    "PnL/(S0×乘数×|数量|) 归一化")

        ax.axhline(
            0.0, color=PALETTE["text_muted"],
            linewidth=0.8, alpha=0.65)
        ax.set_title(model.get("title", ""), fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.55)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(
                handles, labels, loc="best", frameon=False, fontsize=8,
                ncol=2 if len(labels) > 2 else 1)
        ax.margins(x=0.02, y=0.12)
        canvas.draw_idle()

    def _update_history_selection(self, _event=None):
        tree = getattr(self, "_history_period_tree", None)
        rank_tree = getattr(self, "_history_rank_tree", None)
        if tree is None or rank_tree is None or not tree.selection():
            return
        period_item = self._history_period_rows.get(tree.selection()[0], {})
        for child in rank_tree.get_children():
            rank_tree.delete(child)

        self._history_rank_rows = {}
        ranking = self._history_ranking
        group = None
        if ranking is not None and not getattr(ranking, "empty", True):
            selected = ranking[
                ranking["lookback"] == period_item.get("lookback")]
            if not selected.empty:
                group = selected.sort_values(["rank", "score", "strategy"], kind="stable")
        if group is not None:
            baseline = BacktestApp._comparison_baseline(group)
            for row_no, (_idx, row) in enumerate(group.iterrows()):
                score = self._comparison_finite(row.get("score"))
                rank_item = row.to_dict()
                is_baseline = str(
                    row.get("strategy_type", "")) == "close_to_close"
                improvement = BacktestApp._history_row_improvement(
                    rank_item, baseline)
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
                    status = "完整配对"
                elif comparison_eligible:
                    status = "配对诊断"
                elif paired:
                    status = "部分配对"
                else:
                    status = "无配对"
                strategy_label = str(row.get("strategy", "—"))
                if is_baseline:
                    strategy_label += "（C2C基准）"
                values = (
                    "基准" if is_baseline else "☐",
                    self._comparison_safe_int(
                        row.get("rank"), row_no + 1), strategy_label,
                    self._format_comparison_value(score, 2),
                    self._format_comparison_value(
                        row.get("baseline_score"), 2),
                    ("基准" if is_baseline else
                     self._format_comparison_value(
                         improvement, 1, signed=True, percent=True)),
                    ("基准" if is_baseline else
                     self._format_comparison_value(
                         row.get("window_win_rate_vs_c2c"), 1,
                         percent=True)),
                    f"{paired}/{baseline_windows}", status,
                )
                if is_baseline:
                    tag = "baseline"
                elif row_no == 0 and comparison_eligible:
                    tag = "leader"
                elif not comparison_eligible:
                    tag = "incomplete"
                else:
                    tag = ""
                iid = f"history_rank_{row_no}"
                rank_tree.insert(
                    "", "end", iid=iid, values=values,
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
                self._history_top_chart_candidates(limit=2))
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
                f"代理期限T={self._comparison_safe_int(maturity_value)}日")
        warmup_days = self._comparison_safe_int(
            period_item.get("realized_sigma_warmup_days"), 0)
        if warmup_days:
            extra.append(f"HV预热V={warmup_days}日")
        evidence_days = self._comparison_safe_int(
            period_item.get("evidence_days"), 0)
        if evidence_days:
            extra.append(f"目标L={evidence_days}日")
        elif period_item.get("requested", 0):
            extra.append(f"目标观察期={period_item.get('requested', 0)}日")
        days_used = self._comparison_safe_int(
            period_item.get("days_used"), 0)
        if evidence_days or period_item.get("requested", 0) or days_used:
            extra.append(f"实际评分={days_used}日")
        segment_count = period_item.get("segment_count")
        if segment_count is not None:
            segment_count = self._comparison_safe_int(segment_count, 0)
            expiry_segments = self._comparison_safe_int(
                period_item.get("expiry_segments"), 0)
            mtm_segments = self._comparison_safe_int(
                period_item.get("mtm_segments"), 0)
            segment_text = f"代理段数={segment_count}"
            if expiry_segments or mtm_segments:
                segment_text += (
                    f"（到期{expiry_segments} / MTM {mtm_segments}）")
            extra.append(segment_text)
        terminal_labels = {
            "expiry": "到期结算",
            "mark_to_market": "尾段MTM",
            "mixed": "到期+尾段MTM",
        }
        terminal_mode = str(period_item.get("terminal_mode") or "")
        if terminal_mode:
            extra.append(
                "终止口径=" + terminal_labels.get(
                    terminal_mode, terminal_mode))
        sampling_mode = str(period_item.get("sampling_mode", ""))
        if not period_item.get("uses_strict_metric", False):
            extra.append("旧版终点采样口径（建议重新运行）")
        elif sampling_mode and sampling_mode != "strict_contiguous":
            extra.append("采样元数据与严格区间口径不一致")

        required = self._comparison_safe_int(
            period_item.get("required_history_days"), 0)
        available_history = self._comparison_safe_int(
            period_item.get("available_history_days"), 0)
        if required:
            extra.append(f"历史={available_history}/{required}日")
        elif period_item.get("requested", 0):
            extra.append(
                f"历史覆盖={period_item.get('available', 0)}/"
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
        extra.append(
            f"成功配对代理段={period_item.get('effective', 0)} · "
            f"可回放代理段={period_item.get('eligible', 0)} · "
            f"失败代理段={period_item.get('skipped', 0)}")
        relative_value = period_item.get("relative_comparison_windows")
        relative_windows = (
            self._comparison_safe_int(relative_value, 0)
            if relative_value is not None else None)
        if (relative_windows is not None and period_item.get("effective", 0)
                and relative_windows != period_item.get("effective", 0)):
            extra.append(
                f"可评分代理段={relative_windows}/"
                f"{period_item.get('effective', 0)}")
        if period_item.get("trailing_dropped", 0):
            extra.append(
                f"剔除尾部不完整组={period_item['trailing_dropped']}")
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
                    extra.append(f"候选失败={reason}")
        self._history_period_detail_prefix = (
            f"{period_item.get('period', '—')} · "
            f"{period_item.get('status', '—')} · "
            + " · ".join(extra))
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
            self._sigma_src_var.set(SIGMA_SOURCE_DISPLAY[sigma_source])
            self._sigma_win_var.set(str(sigma_window))
            self._band_sigma_var.set(f"{candidate_sigma:.10g}")
            self._mark_band_edited("sigma")
            self._sync_band_inputs("sigma", strict=True)
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

    def _run_history_selection_on_current_path(self):
        """应用选中候选，运行普通回测并在成功后自动保留快照。"""
        if getattr(self, "_active_job", None) is not None:
            messagebox.showinfo("任务运行中", "请等待当前任务完成后再验证。")
            return False
        row = self._selected_history_rank_row()
        if not row:
            messagebox.showinfo("请选择策略", "请先在周期排名表中选择一条策略。")
            return False
        position_var = getattr(self, "_pos_var", None)
        launch_position = (
            BacktestApp._normalize_position(position_var.get())
            if position_var is not None else None)
        try:
            self._apply_history_recommendation(row, navigate=False)
            # 历史行只携带策略参数；当前路径验证始终沿用点击启动时的
            # 左侧方向，不能被历史任务快照或未来新增回填字段覆盖。
            if position_var is not None:
                position_var.set(launch_position)
        except Exception as exc:
            messagebox.showerror("无法应用历史策略", str(exc))
            return False
        self._pending_history_retain_name = (
            f"历史验证 · {str(row.get('strategy', '策略')).strip()}")
        started = bool(self._run_backtest())
        if not started:
            self._pending_history_retain_name = None
        return started

    # 旧回调名保留软兼容，不再读写 comparison 状态。
    def _update_comparison_history_selection(self, _event=None):
        return self._update_history_selection(_event)

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
                "  触发方式          :  每交易日最后一根 bar",
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
                "absolute": "绝对价格", "relative": "相对价格", "sigma": "日波动σ倍数",
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
                    f"  期初等价日波动σ倍数 :  {converted['sigma']:.6g}",
                ])
            except (TypeError, ValueError):
                pass
            sigma_strategy = getattr(strategy, "_sigma_strategy", None)
            sigma_src = getattr(sigma_strategy, "sigma_source", "implied")
            if band_type == "sigma":
                strategy_lines.append(f"  σ 来源            :  {sigma_src}")
                if sigma_src == "realized":
                    strategy_lines.append(
                        f"  HV 回看期 (日)    :  {getattr(sigma_strategy, 'window_days', 20)}")
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
                "  每日收盘兜底      :  开启"
                f"（实际补触发 {fallback_count} 次；同 Bar 自动去重）")
        else:
            strategy_lines.append("  每日收盘兜底      :  关闭")

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
        _kv("实际采样 bar/日 ", f"{bt.steps_per_day:>10d}")
        if r.get("knocked_out"):
            _kv("敲出了结        ", f"第 {r['ko_day']} 日敲出, 结算票息 {r['ko_settle']:.4f}",
                "value_pos")
        for sl in strategy_lines:
            _ins(sl + "\n", "label")
        _kv("交易成本率      ", f"{bt.tc_rate * 100:.2f}%")
        _kv("头寸方向        ", '卖出(short)' if bt.position == 1 else '买入(long)')
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
        _kv("期权 MtM 盈亏   ", f"{opt_pnl:>12.4f}", _val_color(opt_pnl))
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
        log_ret = np.log(r['prices'][1:] / r['prices'][:-1])
        ax4.hist(log_ret * 100, bins=max(15, r['n_days'] // 3),
                 edgecolor='black', alpha=0.7, color='steelblue', density=True)
        # 叠加正态分布（隐含波动率）
        x_range = np.linspace(log_ret.min() * 100, log_ret.max() * 100, 200)
        daily_impl = implied / np.sqrt(ANNUAL_DAYS) * 100
        from scipy.stats import norm
        ax4.plot(x_range, norm.pdf(x_range, 0, daily_impl), 'r--', linewidth=1.2,
                 label=f'隐含波动率正态 σ={daily_impl:.2f}%')
        daily_real = np.std(log_ret) * 100
        ax4.plot(x_range, norm.pdf(x_range, np.mean(log_ret) * 100, daily_real),
                 'b-', linewidth=1.2, alpha=0.7,
                 label=f'已实现正态 σ={daily_real:.2f}%')
        ax4.set_title('日收益率分布', fontsize=10)
        ax4.set_xlabel('日收益率 (%)', fontsize=8)
        ax4.set_ylabel('密度', fontsize=8)
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._vol_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._vol_figure = fig
        self._vol_canvas = canvas

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

        # (1) 总盈亏分布直方图
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(pnl, bins=max(30, n_paths // 15), edgecolor='black',
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
        ax2.hist(errors, bins=max(30, n_paths // 15), edgecolor='black',
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
        ax3.hist(rv * 100, bins=max(30, n_paths // 15), edgecolor='black',
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
        # 线性拟合
        if len(vol_spread) > 2:
            z = np.polyfit(vol_spread * 100, pnl, 1)
            x_fit = np.linspace(np.min(vol_spread) * 100, np.max(vol_spread) * 100, 100)
            ax4.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=1.5,
                     label=f'拟合斜率={z[0]:.2f}')
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('盈亏 vs 波动率价差', fontsize=10)
        ax4.set_xlabel('波动率价差 (隐含−已实现) %', fontsize=8)
        ax4.set_ylabel('总盈亏', fontsize=8)
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
                source = "收盘兜底"
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
                       f"（原始 {total_rows} {'个 Bar' if is_intraday else '行'}）"),
                 font=(_UI_FONT_FAMILY, 10, "bold"),
                 bg=PALETTE["surface"], fg=PALETTE["text"]).pack(side="left")
        tk.Label(info_frame,
                 text="  ·  盈亏字段为触发 Bar 当期值",
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

        tree.heading("day_no", text="原始 Bar" if is_intraday else "交易日")
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
        perspective = "卖方(short)" if position == 1 else "买方(long)"
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
