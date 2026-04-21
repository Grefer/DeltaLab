# _*_ coding: utf-8 _*_
"""
DeltaLab - 期权对冲回测 GUI 应用

基于 tkinter 构建，支持选择不同期权类型、回测方式（模拟/历史数据），
并以图表和表格形式展示回测结果。
"""

import sys
import os
import copy
import platform
import threading
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
    Option_AB, Option_AS, Option_DE, Option_Vanilla, HedgeBacktest,
    FixedFreqStrategy, SigmaBandStrategy,
)
from pricing.constants import ANNUAL_DAYS

# ============================================================
#  期权类型注册表
# ============================================================

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
            "Opt_Decumulator_Back", "Opt_Decumulator_Fix",
            "Opt_EnDecumulator", "Opt_EnDecumulator_Fix",
            "Opt_ASGQ_call_put", "Opt_ASGQ_EP", "Opt_ASGQ_EF",
            "Opt_ASGQ_DP", "Opt_ASGQ_DF",
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
}


# ============================================================
#  GUI 显示名 ↔ 后端内部键 映射
#  说明：后端 (hedge_backtest / Option_* 类) 使用英文/方法名做字符串匹配，
#  这里仅影响界面显示；读取 Combobox 值后需通过 *_FROM_DISPLAY 反向映射
#  还原为内部键再传给后端。
# ============================================================

SUBTYPE_DISPLAY = {
    "Eu":                    "欧式 (Eu)",
    "Opt_Decumulator_Back":  "回归累计 (Opt_Decumulator_Back)",
    "Opt_Decumulator_Fix":   "固定赔付回归累计 (Opt_Decumulator_Fix)",
    "Opt_EnDecumulator":     "增强回归累计 (Opt_EnDecumulator)",
    "Opt_EnDecumulator_Fix": "固定赔付增强累计 (Opt_EnDecumulator_Fix)",
    "Opt_ASGQ_call_put":     "到期熔断保障累计 (Opt_ASGQ_call_put)",
    "Opt_ASGQ_EP":           "熔断每日保障累计 (Opt_ASGQ_EP)",
    "Opt_ASGQ_EF":           "熔断每日固赔累计 (Opt_ASGQ_EF)",
    "Opt_ASGQ_DP":           "每日熔断保障累计 (Opt_ASGQ_DP)",
    "Opt_ASGQ_DF":           "每日熔断固赔累计 (Opt_ASGQ_DF)",
    "Asian":                 "标准亚式 (Asian)",
    "EnhanceAsian":          "增强亚式 (EnhanceAsian)",
    "Opt_Airbag":            "气囊 (Opt_Airbag)",
}
SUBTYPE_FROM_DISPLAY = {v: k for k, v in SUBTYPE_DISPLAY.items()}

STRATEGY_DISPLAY = {
    "fixed_freq": "固定频率调仓",
    "sigma_band": "σ 带宽调仓",
}
STRATEGY_FROM_DISPLAY = {v: k for k, v in STRATEGY_DISPLAY.items()}

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
        self._apply_window_icon()
        self._setup_styles()
        self._build_ui()
        self._param_entries = {}
        self._on_option_class_change(None)

    # ---- 窗口图标 ----
    def _apply_window_icon(self):
        ico_path = _resource_path("assets", "deltalab.ico")
        png_path = _resource_path("assets", "deltalab.png")

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

        def _bind_left_wheel(_event=None):
            self._left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)

        def _unbind_left_wheel(_event=None):
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
        ttk.Label(self._wind_frame, text="起始日:", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=2)
        _today = datetime.date.today()
        _wind_start_default = (_today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        _wind_end_default = _today.strftime("%Y-%m-%d")
        self._wind_start_var = tk.StringVar(value=_wind_start_default)
        ttk.Entry(self._wind_frame, textvariable=self._wind_start_var, width=15).grid(
            row=1, column=1, padx=(6, 8), pady=2)
        ttk.Label(self._wind_frame, text="结束日:", style="Surface.TLabel").grid(
            row=1, column=2, sticky="w", pady=2)
        self._wind_end_var = tk.StringVar(value=_wind_end_default)
        ttk.Entry(self._wind_frame, textvariable=self._wind_end_var, width=15).grid(
            row=1, column=3, padx=(6, 0), pady=2)

        # Wind intraday bar_size 选择
        ttk.Label(self._wind_frame, text="频率:", style="Surface.TLabel").grid(
            row=2, column=0, sticky="w", pady=2)
        self._wind_bar_size_var = tk.StringVar(value="日频")
        self._wind_bar_size_combo = ttk.Combobox(
            self._wind_frame, textvariable=self._wind_bar_size_var, width=10,
            values=("日频", "60min", "30min", "15min", "5min", "1min"),
            state="readonly",
        )
        self._wind_bar_size_combo.grid(row=2, column=1, padx=(6, 8),
                                       pady=2, sticky="w")
        self._wind_bar_size_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._toggle_source())

        # 轻分割线
        ttk.Separator(sec3, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 6))

        # 对冲参数
        row_h = 3
        ttk.Label(sec3, text="调仓频率(天):", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._freq_var = tk.StringVar(value="1")
        ttk.Entry(sec3, textvariable=self._freq_var, width=10).grid(
            row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)

        row_h += 1
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
        self._strategy_var = tk.StringVar(value=STRATEGY_DISPLAY["fixed_freq"])
        strat_frame = ttk.Frame(sec3, style="Surface.TFrame")
        strat_frame.grid(row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)
        self._strategy_combo = ttk.Combobox(
            strat_frame, textvariable=self._strategy_var, width=16,
            values=tuple(STRATEGY_DISPLAY.values()), state="readonly",
        )
        self._strategy_combo.pack(side="left")
        self._strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_strategy())

        # x-sigma 带专用参数（默认隐藏，选 sigma_band 时显示）
        row_h += 1
        self._sigma_band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sigma_band_frame.grid(row=row_h, column=0, columnspan=2, sticky="ew",
                                    padx=(0, 0), pady=2)
        ttk.Label(self._sigma_band_frame, text="调仓带宽 k·σ:",
                  style="Surface.TLabel").grid(row=0, column=0, sticky="w", pady=2)
        self._k_var = tk.StringVar(value="0.5")
        ttk.Entry(self._sigma_band_frame, textvariable=self._k_var, width=8).grid(
            row=0, column=1, padx=(6, 12), pady=2, sticky="w")
        ttk.Label(self._sigma_band_frame, text="σ 来源:",
                  style="Surface.TLabel").grid(row=0, column=2, sticky="w", pady=2)
        self._sigma_src_var = tk.StringVar(value=SIGMA_SOURCE_DISPLAY["implied"])
        ttk.Combobox(self._sigma_band_frame, textvariable=self._sigma_src_var, width=14,
                     values=tuple(SIGMA_SOURCE_DISPLAY.values()), state="readonly").grid(
            row=0, column=3, padx=(6, 12), pady=2, sticky="w")
        ttk.Label(self._sigma_band_frame, text="历史波动率窗口 N (日):",
                  style="Surface.TLabel").grid(row=0, column=4, sticky="w", pady=2)
        self._sigma_win_var = tk.StringVar(value="20")
        ttk.Entry(self._sigma_band_frame, textvariable=self._sigma_win_var, width=6).grid(
            row=0, column=5, padx=(6, 0), pady=2, sticky="w")

        row_h += 1
        ttk.Label(sec3, text="每日 bar 数:", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        spd_frame = ttk.Frame(sec3, style="Surface.TFrame")
        spd_frame.grid(row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)
        self._spd_var = tk.StringVar(value="1")
        self._spd_combo = ttk.Combobox(spd_frame, textvariable=self._spd_var, width=6,
                                       values=("1", "4", "48", "240"), state="readonly")
        self._spd_combo.pack(side="left")
        self._spd_hint_label = ttk.Label(
            spd_frame, text=" 1=日频 / 4=60分 / 48=5分 / 240=1分",
            style="SurfaceMuted.TLabel")
        self._spd_hint_label.pack(side="left", padx=(6, 0))

        row_h += 1
        ttk.Label(sec3, text="滑点 (bps):", style="Surface.TLabel").grid(
            row=row_h, column=0, sticky="w", pady=4)
        self._slip_var = tk.StringVar(value="0")
        ttk.Entry(sec3, textvariable=self._slip_var, width=10).grid(
            row=row_h, column=1, sticky="w", padx=(8, 0), pady=4)

        sec3.columnconfigure(1, weight=1)

        # 依据默认策略（fixed_freq）初始化 sigma 带控件可见性
        self._toggle_strategy()

        # 运行按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(2, 6))
        self._run_btn = ttk.Button(btn_frame, text="▶  运行回测", style="Run.TButton",
                                   command=self._run_backtest)
        self._run_btn.pack(fill="x", ipady=4)

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
            ("chart",   " 📈 对冲图表 ",
             "对冲图表", "运行回测后此处将展示标的价格、Delta、Gamma、累计盈亏等图表"),
            ("vol",     " 📊 波动率分析 ",
             "波动率分析", "运行回测后此处将展示隐含波动率与已实现波动率的对比分析"),
            ("dist",    " 🎲 盈亏分布 ",
             "盈亏分布", "在模拟模式下运行多路径回测后，此处将展示蒙特卡洛盈亏分布"),
            ("struct",  " 🔬 结构分析 ",
             "结构分析", "点击左侧『绘制结构图』按钮以生成期权结构的 Greeks 曲线"),
            ("table",   " 📃 每日明细 ",
             "每日明细", "运行回测后此处将展示逐日对冲持仓、盈亏与 Greeks 明细表"),
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

            # ==== 占位 / 欢迎界面 ====
            placeholder = ttk.Frame(tab, style="Surface.TFrame")
            placeholder.place(relx=0.5, rely=0.45, anchor="center")
            setattr(self, f"_{suffix}_placeholder", placeholder)

            # 大图标
            icon_map = {
                "summary": "📋", "chart": "📈", "vol": "📊",
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

        # 底部状态栏
        status_bar = ttk.Frame(self, style="Surface.TFrame")
        status_bar.pack(fill="x", side="bottom")
        ttk.Separator(status_bar, orient="horizontal").pack(fill="x")
        self._status_var = tk.StringVar(
            value="就绪  |  选择期权类型、设置参数后点击『运行回测』")
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Status.TLabel", anchor="w").pack(fill="x", padx=10)

        self._toggle_source()

    # ---- 事件回调 ----
    def _on_option_class_change(self, event):
        cls_name = self._class_var.get()
        cfg = OPTION_CLASSES[cls_name]
        display_values = [SUBTYPE_DISPLAY[s] if s in SUBTYPE_DISPLAY else str(s)
                          for s in cfg["subtypes"]]
        self._subtype_cb.configure(values=display_values)
        self._subtype_cb.current(0)
        self._rebuild_params(cfg["params"])

    def _rebuild_params(self, params):
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_entries = {}
        for i, spec in enumerate(params):
            key, label, dtype, default = spec[:4]
            choices = spec[4] if len(spec) > 4 else None
            ttk.Label(self._param_frame, text=f"{label}:",
                      style="Surface.TLabel").grid(
                row=i, column=0, sticky="w", padx=(2, 8), pady=3)
            if choices:
                val_to_display = {v: k for k, v in choices.items()}
                default_display = val_to_display.get(default, next(iter(choices)))
                var = tk.StringVar(value=default_display)
                cb = ttk.Combobox(self._param_frame, textvariable=var,
                                  values=list(choices.keys()),
                                  state="readonly", width=14)
                cb.grid(row=i, column=1, sticky="ew", pady=3, padx=(0, 2))
            else:
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(self._param_frame, textvariable=var, width=16)
                entry.grid(row=i, column=1, sticky="ew", pady=3, padx=(0, 2))
            self._param_entries[key] = (var, dtype, choices)
        self._param_frame.columnconfigure(1, weight=1)

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
        # CSV 始终日频；Wind 在 bar_size=日频 时锁 spd=1，其它分钟级解锁并按映射自动填值。
        if hasattr(self, "_spd_combo"):
            bar_label = getattr(self, "_wind_bar_size_var", None)
            bar_label = bar_label.get() if bar_label is not None else "日频"
            wind_intraday = (src == "wind" and bar_label != "日频")

            if src == "csv" or (src == "wind" and not wind_intraday):
                self._spd_var.set("1")
                self._spd_combo.configure(state="disabled")
                if src == "csv":
                    self._spd_hint_label.configure(
                        text=" CSV 模式仅支持日频 (spd=1)")
                else:
                    self._spd_hint_label.configure(
                        text=" Wind 日频模式 (spd=1)；切到分钟频率可解锁")
            elif wind_intraday:
                # 根据 bar_size 自动推导 spd（A 股 240 分钟日盘）
                bar_min_map = {"60min": 4, "30min": 8, "15min": 16,
                               "5min": 48, "1min": 240}
                auto_spd = bar_min_map.get(bar_label, 1)
                # 每次切换都覆盖一次；避免上轮 simulate/其它 bar_size 残留值
                self._spd_var.set(str(auto_spd))
                # 允许用户手动改写（商品期货含夜盘需要自定义 spd）
                self._spd_combo.configure(state="normal")
                self._spd_hint_label.configure(
                    text=f" Wind intraday {bar_label} -> spd={auto_spd}（可手动覆盖）")
            else:
                # simulate 模式
                self._spd_combo.configure(state="readonly")
                self._spd_hint_label.configure(
                    text=" 1=日频 / 4=60分 / 48=5分 / 240=1分")

    def _toggle_strategy(self):
        """对冲策略切换：sigma_band 显示 k / σ 源 / N，fixed_freq 隐藏。"""
        strategy_key = STRATEGY_FROM_DISPLAY.get(
            self._strategy_var.get(), self._strategy_var.get())
        if strategy_key == "sigma_band":
            self._sigma_band_frame.grid()
        else:
            self._sigma_band_frame.grid_remove()

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._csv_path_var.set(path)

    # ---- 回测核心 ----
    def _run_backtest(self):
        # 在主线程中收集所有 GUI 参数（tkinter 非线程安全）
        try:
            gui_state = self._collect_gui_state()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._run_btn.configure(state="disabled")
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.start(15)
        self._set_status("正在运行回测…")
        threading.Thread(target=self._backtest_worker, args=(gui_state,),
                         daemon=True).start()

    def _collect_gui_state(self):
        """在主线程中读取所有 tkinter 变量，返回纯 Python dict"""
        cls_name = self._class_var.get()
        cfg = OPTION_CLASSES[cls_name]
        subtype_display = self._subtype_var.get()
        subtype = SUBTYPE_FROM_DISPLAY.get(subtype_display, subtype_display)

        params = {}
        for key, (var, dtype, choices) in self._param_entries.items():
            val_str = var.get().strip()
            if choices:
                params[key] = choices[val_str]
            elif not val_str:
                params[key] = 0
            elif dtype == float:
                params[key] = float(val_str)
            elif dtype == int:
                params[key] = int(val_str)
            else:
                params[key] = val_str

        return {
            "cls_name": cls_name,
            "cfg": cfg,
            "subtype": subtype,
            "params": params,
            "source": self._source_var.get(),
            "hedge_freq": int(self._freq_var.get()),
            "tc_rate": float(self._tc_var.get()) / 100.0,
            "position": int(self._pos_var.get()),
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
            "wind_bar_size": self._wind_bar_size_var.get().strip(),
            # --- 新增：对冲策略与 intraday / 滑点 ---
            "strategy_name": STRATEGY_FROM_DISPLAY.get(
                self._strategy_var.get(), self._strategy_var.get()),
            "k_band": float(self._k_var.get() or 0.5),
            "sigma_source": SIGMA_SOURCE_FROM_DISPLAY.get(
                self._sigma_src_var.get(), self._sigma_src_var.get()),
            "sigma_window": int(self._sigma_win_var.get() or 20),
            "steps_per_day": int(self._spd_var.get() or 1),
            "slippage_bps": float(self._slip_var.get() or 0.0),
        }

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

            self.after(0, lambda: self._show_results(bt, multi_stats))
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(0, lambda: messagebox.showerror("回测失败", err_msg))
        finally:
            self.after(0, self._finish_run)

    def _set_status(self, text):
        """更新底部状态栏文字 (线程安全: 调用方需保证在主线程或通过 after)。"""
        try:
            self._status_var.set(text)
        except Exception:
            pass

    def _finish_run(self):
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.pack_forget()
        self._progress_label.pack_forget()
        self._progress_label.configure(text="")
        self._run_btn.configure(state="normal")
        self._set_status("回测完成  |  切换上方 Tab 查看各项结果")

    def _build_backtest(self, gs):
        """根据已收集的 GUI 状态构建 HedgeBacktest 实例（可在任意线程调用）"""
        cfg = gs["cfg"]
        subtype = gs["subtype"]
        params = gs["params"]
        src = gs["source"]
        hedge_freq = gs["hedge_freq"]
        tc_rate = gs["tc_rate"]
        position = gs["position"]
        quantity = gs["quantity"]
        multiplier = gs["multiplier"]
        slippage_bps = gs.get("slippage_bps", 0.0)
        steps_per_day = int(gs.get("steps_per_day", 1))

        # 组装对冲策略对象
        strat_name = gs.get("strategy_name", "fixed_freq")
        if strat_name == "sigma_band":
            strategy = SigmaBandStrategy(
                k=gs.get("k_band", 0.5),
                sigma_source=gs.get("sigma_source", "implied"),
                window_days=gs.get("sigma_window", 20),
            )
        else:
            strategy = FixedFreqStrategy(hedge_freq=hedge_freq)

        # 判断 Wind intraday 频率（None/日频 走日频分支，其它走 wsi）
        wind_bar_label = gs.get("wind_bar_size", "日频") or "日频"
        _bar_label_to_size = {
            "日频": None, "60min": "60", "30min": "30",
            "15min": "15", "5min": "5", "1min": "1",
        }
        wind_bar_size = _bar_label_to_size.get(wind_bar_label, None)

        # CSV 始终日频；Wind 日频下同样强制 spd=1；Wind intraday 保留 spd。
        if src == "csv" and steps_per_day != 1:
            steps_per_day = 1
        if src == "wind" and wind_bar_size is None and steps_per_day != 1:
            steps_per_day = 1

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
            bt = HedgeBacktest(option, prices, hedge_freq=hedge_freq,
                               tc_rate=tc_rate, position=position,
                               quantity=quantity, multiplier=multiplier,
                               strategy=strategy, steps_per_day=steps_per_day,
                               slippage_bps=slippage_bps)

        elif src == "csv":
            filepath = gs["csv_path"]
            if not filepath:
                raise ValueError("请选择 CSV 文件")
            price_col = gs["csv_col"]
            # 期权参数中的 s0 作为参考价 S_ref，
            # from_csv 会按 ratio = 真实起始价 / S_ref 自动缩放价格量纲要素。
            option = cfg["build"](subtype, params)
            bt = HedgeBacktest.from_csv(option, filepath, price_col=price_col,
                                        hedge_freq=hedge_freq, tc_rate=tc_rate,
                                        position=position,
                                        quantity=quantity, multiplier=multiplier,
                                        strategy=strategy,
                                        steps_per_day=steps_per_day,
                                        slippage_bps=slippage_bps)

        elif src == "wind":
            code = gs["wind_code"]
            start = gs["wind_start"]
            end = gs["wind_end"]
            # 期权参数中的 s0 作为参考价 S_ref，
            # from_wind 会按 ratio = 真实起始价 / S_ref 自动缩放价格量纲要素。
            option = cfg["build"](subtype, params)
            # wind_bar_size=None -> 日频；否则 '1'/'5'/'60' 等字符串触发 intraday 分支。
            # intraday 下 steps_per_day 由 from_wind 自动推导，这里仍显式传入以防覆盖。
            bt = HedgeBacktest.from_wind(option, code, start, end,
                                         hedge_freq=hedge_freq, tc_rate=tc_rate,
                                         position=position,
                                         quantity=quantity, multiplier=multiplier,
                                         strategy=strategy,
                                         steps_per_day=steps_per_day,
                                         slippage_bps=slippage_bps,
                                         bar_size=wind_bar_size)
        else:
            raise ValueError(f"未知数据来源: {src}")

        # 保存元信息用于展示（不再依赖 tkinter 变量）
        bt._gui_meta = {
            "cls_name": gs["cls_name"],
            "subtype": subtype,
            "source": src,
        }
        return bt

    # ---- 隐藏占位符工具方法 ----
    def _hide_placeholder(self, suffix):
        """隐藏指定 tab 的占位符 (place_forget), 以便展示真正的内容控件."""
        ph = getattr(self, f"_{suffix}_placeholder", None)
        if ph is not None:
            ph.place_forget()

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

    # ---- 结果展示 ----
    def _show_results(self, bt, multi_stats=None):
        self._show_summary(bt, multi_stats)
        self._show_chart(bt)
        self._show_vol_chart(bt)
        self._show_dist_chart(multi_stats)
        self._show_table(bt)
        self._nb.select(0)

    def _show_summary(self, bt, multi_stats=None):
        self._hide_placeholder("summary")
        self._summary_text.pack(fill="both", expand=True, padx=8, pady=8)

        r = bt._results
        n = r['n_days']
        meta = bt._gui_meta
        strategy = getattr(bt, "strategy", None)
        strategy_name = getattr(strategy, "name", "fixed_freq")
        if strategy_name == "sigma_band":
            k_val = getattr(strategy, "k", float("nan"))
            sigma_src = getattr(strategy, "sigma_source", "implied")
            win_days = getattr(strategy, "window_days", None)
            strategy_lines = [
                f"  对冲策略          :  sigma_band",
                f"  σ 带 k            :  {k_val:>12.4f}",
                f"  σ 来源            :  {sigma_src:>12s}",
            ]
            if sigma_src == "realized" and win_days is not None:
                strategy_lines.append(f"  HV 窗口 (日)      :  {int(win_days):>12d}")
        else:
            strategy_lines = [
                f"  对冲策略          :  fixed_freq",
                f"  调仓频率          :  每 {bt.hedge_freq} 天",
            ]

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
        _ins("\n")
        _sep()

        _kv("回测天数        ", f"{n:>10d}")
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
            for name, (old, new) in rescale_info["fields"].items():
                if name in ("s0", "sr"):
                    continue
                _ins(f"    {name:<8s}: {old:>12.4f}  →  {new:.4f}\n")
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
            gv = r[gn]
            _ins(f"  {gn.capitalize():15s}  {gv[0]:>10.4f}  {np.mean(gv):>10.4f}  {np.max(np.abs(gv)):>10.4f}\n")
        _sep()

        rebal = int(np.sum(np.abs(np.diff(r['shares'])) > 1e-10))
        _kv("调仓次数        ", f"{rebal:>10d}")
        _ins("═" * 54 + "\n", "separator")

        # 蒙特卡洛多路径统计
        if multi_stats is not None:
            ms = multi_stats
            pnl = ms['total_pnl']
            n_p = ms['n_paths']
            pct = lambda arr, q: np.percentile(arr, q)

            _ins("\n")
            _ins("═" * 54 + "\n", "separator")
            _ins(f"       蒙特卡洛多路径统计 ({n_p} 条路径)\n", "monte_header")
            _ins("═" * 54 + "\n", "separator")
            _ins("\n")

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
            _kv("隐含波动率      ", f"{ms['implied_vol'] * 100:>11.2f}%")
            _kv("已实现波动率均值", f"{np.mean(ms['realized_vols']) * 100:>11.2f}%")
            _kv("已实现波动率标准差", f"{np.std(ms['realized_vols']) * 100:>11.2f}%")
            _sep()
            _kv("平均交易成本    ", f"{np.mean(ms['total_tc']):>12.4f}", "value_neg")
            _ins("═" * 54 + "\n", "separator")

        tw.configure(state="disabled")

    def _show_chart(self, bt):
        # 隐藏占位 + 清除旧图表
        self._hide_placeholder("chart")
        for w in self._chart_container.winfo_children():
            w.destroy()

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
        ax3.plot(days, r['gamma'], 'm-', linewidth=1.2)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
        ax3.set_title('Gamma', fontsize=10)
        ax3.set_xlabel(x_label, fontsize=8)
        ax3.set_ylabel('Gamma', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=30, labelsize=7)

        # (4) Vega
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(days, r['vega'], 'c-', linewidth=1.2)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
        ax4.set_title('Vega', fontsize=10)
        ax4.set_xlabel(x_label, fontsize=8)
        ax4.set_ylabel('Vega', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=30, labelsize=7)

        # (5) Theta
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(days, r['theta'], color='orange', linewidth=1.2)
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

    def _show_vol_chart(self, bt):
        """波动率分析图表"""
        self._hide_placeholder("vol")
        for w in self._vol_container.winfo_children():
            w.destroy()

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

    def _show_dist_chart(self, multi_stats):
        """蒙特卡洛盈亏分布图表"""
        for w in self._dist_container.winfo_children():
            w.destroy()

        if multi_stats is None:
            # 显示优雅的空状态提示 (不隐藏占位符, 让它保持可见)
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
        self._hide_placeholder("dist")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        ms = multi_stats
        pnl = ms['total_pnl']
        errors = ms['errors']
        rv = ms['realized_vols']
        iv = ms['implied_vol']
        n_paths = ms['n_paths']

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

    def _show_table(self, bt):
        self._hide_placeholder("table")
        for w in self._table_tab.winfo_children():
            w.destroy()

        df = bt.to_dataframe()

        # 顶部工具栏 (导出 + 行数提示 + 统计信息)
        toolbar = ttk.Frame(self._table_tab, style="Surface.TFrame")
        toolbar.pack(fill="x", padx=10, pady=(10, 6))

        info_frame = ttk.Frame(toolbar, style="Surface.TFrame")
        info_frame.pack(side="left", fill="x")
        tk.Label(info_frame, text="📃",
                 font=(_UI_FONT_FAMILY, 14),
                 bg=PALETTE["surface"]).pack(side="left", padx=(0, 6))
        tk.Label(info_frame,
                 text=f"共 {len(df)} 行 × {len(df.columns)} 列",
                 font=(_UI_FONT_FAMILY, 10, "bold"),
                 bg=PALETTE["surface"], fg=PALETTE["text"]).pack(side="left")

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

        tree.heading("day_no", text="交易日")
        tree.column("day_no", width=60, anchor="center")
        tree.heading("idx", text=df.index.name or "日期")
        tree.column("idx", width=96, anchor="center")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=92, anchor="e")

        # 斑马行
        tree.tag_configure("odd",  background=PALETTE["surface"])
        tree.tag_configure("even", background=PALETTE["surface_alt"])

        for i, (idx, row) in enumerate(df.iterrows()):
            idx_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
            values = [i, idx_str] + [f"{v:.4f}" if isinstance(v, float) else str(v)
                                     for v in row.values]
            tag = "even" if i % 2 == 0 else "odd"
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

        self._struct_btn.configure(state="disabled")
        self._run_btn.configure(state="disabled")
        self._progress.pack(fill="x", pady=(6, 0))
        self._progress.configure(mode="determinate", maximum=n_points, value=0)
        self._progress_label.configure(text=f"结构扫描: 0/{n_points}")
        self._progress_label.pack(fill="x")
        self._set_status("正在进行结构扫描…")
        self._nb.select(self._struct_tab)
        threading.Thread(target=self._structure_worker,
                         args=(gui_state, range_pct, n_points),
                         daemon=True).start()

    def _structure_worker(self, gui_state, range_pct, n_points):
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
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(0, lambda: messagebox.showerror("结构分析失败", err_msg))
        finally:
            self.after(0, self._finish_structure)

    def _finish_structure(self):
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.pack_forget()
        self._progress_label.pack_forget()
        self._progress_label.configure(text="")
        self._struct_btn.configure(state="normal")
        self._run_btn.configure(state="normal")
        self._set_status("结构扫描完成")

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
