# _*_ coding: utf-8 _*_
"""
期权对冲回测 GUI 应用

基于 tkinter 构建，支持选择不同期权类型、回测方式（模拟/历史数据），
并以图表和表格形式展示回测结果。
"""

import sys
import os
import copy
import platform
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

# 确保 pricing 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing import Option_AB, Option_AS, Option_DE, Option_Vanilla, HedgeBacktest
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
            ("cp",     "方向(1看涨/-1看跌)", int, 1),
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
            ("cp",     "方向(1看涨/-1看跌)", int, 1),
            ("fix",    "固定赔付(可选)",  float, 0.0),
            ("P",      "保障价格(可选)",  float, 0.0),
            ("amount", "固定金额(可选)",  float, 0.0),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
            ("nPath",  "模拟路径数",     int,   100000),
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
            ("cp",     "方向(1看涨/-1看跌)", int, 1),
            ("minPay", "最低赔付",       float, 0.0),
            ("maxPay", "最高赔付",       float, 999999.0),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
            ("nPath",  "模拟路径数",     int,   100000),
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
            ("cp",    "方向(1看涨/-1看跌)", int, 1),
            ("r",     "无风险利率",     float, 0.03),
            ("q",     "分红率",        float, 0.03),
            ("nPath", "模拟路径数",     int,   100000),
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
        self.title("期权对冲回测系统")
        self.geometry("1500x950")
        self.minsize(1200, 800)
        self._setup_styles()
        self._build_ui()
        self._param_entries = {}
        self._on_option_class_change(None)

    # ---- 样式 ----
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=(_UI_FONT_FAMILY, 14, "bold"))
        style.configure("Header.TLabel", font=(_UI_FONT_FAMILY, 10, "bold"))
        style.configure("Run.TButton", font=(_UI_FONT_FAMILY, 11, "bold"),
                        foreground="white", background="#2563EB")
        style.map("Run.TButton",
                  background=[("active", "#1D4ED8"), ("pressed", "#1E40AF")])

    # ---- 界面构建 ----
    def _build_ui(self):
        # 顶部标题
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(title_frame, text="期权动态对冲回测系统",
                  style="Title.TLabel").pack(side="left")

        # 主体：左侧参数 + 右侧结果
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=5)

        # ─── 左侧面板 ───
        left = ttk.Frame(body, width=380)
        body.add(left, weight=1)

        # 1) 期权大类
        sec1 = ttk.LabelFrame(left, text="期权类型", padding=8)
        sec1.pack(fill="x", pady=(0, 5))

        ttk.Label(sec1, text="大类:").grid(row=0, column=0, sticky="w")
        self._class_var = tk.StringVar()
        class_cb = ttk.Combobox(sec1, textvariable=self._class_var, width=25,
                                values=list(OPTION_CLASSES.keys()), state="readonly")
        class_cb.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        class_cb.current(0)
        class_cb.bind("<<ComboboxSelected>>", self._on_option_class_change)

        ttk.Label(sec1, text="子类型:").grid(row=1, column=0, sticky="w")
        self._subtype_var = tk.StringVar()
        self._subtype_cb = ttk.Combobox(sec1, textvariable=self._subtype_var,
                                        width=25, state="readonly")
        self._subtype_cb.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        sec1.columnconfigure(1, weight=1)

        # 2) 期权参数（可滚动）
        sec2 = ttk.LabelFrame(left, text="期权参数", padding=8)
        sec2.pack(fill="both", expand=True, pady=(0, 5))

        canvas = tk.Canvas(sec2, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sec2, orient="vertical", command=canvas.yview)
        self._param_frame = ttk.Frame(canvas)
        self._param_frame.bind("<Configure>",
                               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._param_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 3) 回测设置
        sec3 = ttk.LabelFrame(left, text="回测设置", padding=8)
        sec3.pack(fill="x", pady=(0, 5))

        ttk.Label(sec3, text="数据来源:").grid(row=0, column=0, sticky="w")
        self._source_var = tk.StringVar(value="simulate")
        src_frame = ttk.Frame(sec3)
        src_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(src_frame, text="模拟数据", variable=self._source_var,
                        value="simulate", command=self._toggle_source).pack(side="left")
        ttk.Radiobutton(src_frame, text="CSV文件", variable=self._source_var,
                        value="csv", command=self._toggle_source).pack(side="left")
        ttk.Radiobutton(src_frame, text="Wind", variable=self._source_var,
                        value="wind", command=self._toggle_source).pack(side="left")

        # 模拟参数
        self._sim_frame = ttk.Frame(sec3)
        self._sim_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(self._sim_frame, text="种子:").grid(row=0, column=0, sticky="w")
        self._seed_var = tk.StringVar(value="42")
        ttk.Entry(self._sim_frame, textvariable=self._seed_var, width=8).grid(
            row=0, column=1, padx=3, sticky="w")
        ttk.Label(self._sim_frame, text="已实现波动率:").grid(row=1, column=0, sticky="w")
        self._real_vol_var = tk.StringVar(value="")
        rv_frame = ttk.Frame(self._sim_frame)
        rv_frame.grid(row=1, column=1, columnspan=3, sticky="w", padx=3)
        ttk.Entry(rv_frame, textvariable=self._real_vol_var, width=8).pack(side="left")
        ttk.Label(rv_frame, text="(空=同隐含波动率)").pack(side="left", padx=3)
        ttk.Label(self._sim_frame, text="模拟路径数:").grid(row=2, column=0, sticky="w")
        self._npaths_var = tk.StringVar(value="10")
        ttk.Entry(self._sim_frame, textvariable=self._npaths_var, width=8).grid(
            row=2, column=1, padx=3, sticky="w")

        # CSV 参数
        self._csv_frame = ttk.Frame(sec3)
        ttk.Label(self._csv_frame, text="文件:").grid(row=0, column=0, sticky="w")
        self._csv_path_var = tk.StringVar()
        ttk.Entry(self._csv_frame, textvariable=self._csv_path_var, width=22).grid(
            row=0, column=1, padx=3)
        ttk.Button(self._csv_frame, text="浏览...", width=6,
                   command=self._browse_csv).grid(row=0, column=2)
        ttk.Label(self._csv_frame, text="价格列:").grid(row=1, column=0, sticky="w")
        self._csv_col_var = tk.StringVar(value="close")
        ttk.Entry(self._csv_frame, textvariable=self._csv_col_var, width=12).grid(
            row=1, column=1, padx=3, sticky="w")

        # Wind 参数
        self._wind_frame = ttk.Frame(sec3)
        ttk.Label(self._wind_frame, text="代码:").grid(row=0, column=0, sticky="w")
        self._wind_code_var = tk.StringVar(value="510050.SH")
        ttk.Entry(self._wind_frame, textvariable=self._wind_code_var, width=15).grid(
            row=0, column=1, padx=3)
        ttk.Label(self._wind_frame, text="起始日:").grid(row=1, column=0, sticky="w")
        self._wind_start_var = tk.StringVar(value="2026-01-02")
        ttk.Entry(self._wind_frame, textvariable=self._wind_start_var, width=15).grid(
            row=1, column=1, padx=3)
        ttk.Label(self._wind_frame, text="结束日:").grid(row=1, column=2, sticky="w")
        self._wind_end_var = tk.StringVar(value="2026-02-07")
        ttk.Entry(self._wind_frame, textvariable=self._wind_end_var, width=15).grid(
            row=1, column=3, padx=3)

        # 对冲参数
        row_h = 3
        ttk.Label(sec3, text="调仓频率(天):").grid(row=row_h, column=0, sticky="w")
        self._freq_var = tk.StringVar(value="1")
        ttk.Entry(sec3, textvariable=self._freq_var, width=8).grid(
            row=row_h, column=1, sticky="w", padx=3)

        row_h += 1
        ttk.Label(sec3, text="交易成本率(%):").grid(row=row_h, column=0, sticky="w")
        self._tc_var = tk.StringVar(value="0.01")
        ttk.Entry(sec3, textvariable=self._tc_var, width=8).grid(
            row=row_h, column=1, sticky="w", padx=3)

        row_h += 1
        ttk.Label(sec3, text="头寸方向:").grid(row=row_h, column=0, sticky="w")
        self._pos_var = tk.StringVar(value="1")
        pos_frame = ttk.Frame(sec3)
        pos_frame.grid(row=row_h, column=1, sticky="w")
        ttk.Radiobutton(pos_frame, text="卖出(short)", variable=self._pos_var,
                        value="1").pack(side="left")
        ttk.Radiobutton(pos_frame, text="买入(long)", variable=self._pos_var,
                        value="-1").pack(side="left")

        row_h += 1
        ttk.Label(sec3, text="交易数量:").grid(row=row_h, column=0, sticky="w")
        self._qty_var = tk.StringVar(value="100")
        ttk.Entry(sec3, textvariable=self._qty_var, width=12).grid(
            row=row_h, column=1, sticky="w", padx=3)

        row_h += 1
        ttk.Label(sec3, text="合约乘数:").grid(row=row_h, column=0, sticky="w")
        self._mult_var = tk.StringVar(value="5")
        mult_frame = ttk.Frame(sec3)
        mult_frame.grid(row=row_h, column=1, sticky="w")
        ttk.Entry(mult_frame, textvariable=self._mult_var, width=8).pack(side="left")
        ttk.Label(mult_frame, text="(0=不取整)").pack(side="left", padx=3)

        sec3.columnconfigure(1, weight=1)

        # 运行按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=5)
        self._run_btn = ttk.Button(btn_frame, text="▶  运行回测", style="Run.TButton",
                                   command=self._run_backtest)
        self._run_btn.pack(fill="x", ipady=4)

        self._struct_btn = ttk.Button(btn_frame, text="📊  绘制结构图",
                                      command=self._plot_structure)
        self._struct_btn.pack(fill="x", ipady=3, pady=(3, 0))

        struct_ctrl = ttk.Frame(btn_frame)
        struct_ctrl.pack(fill="x", pady=(2, 0))
        ttk.Label(struct_ctrl, text="扫描 ±%:").pack(side="left")
        self._struct_range_var = tk.StringVar(value="30")
        ttk.Entry(struct_ctrl, textvariable=self._struct_range_var, width=5).pack(side="left", padx=(2, 8))
        ttk.Label(struct_ctrl, text="点数:").pack(side="left")
        self._struct_npts_var = tk.StringVar(value="31")
        ttk.Entry(struct_ctrl, textvariable=self._struct_npts_var, width=5).pack(side="left", padx=2)

        self._progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self._progress_label = ttk.Label(btn_frame, text="", anchor="center")

        # ─── 右侧面板 ───
        right = ttk.Frame(body)
        body.add(right, weight=2)

        # Notebook for results
        self._nb = ttk.Notebook(right)
        self._nb.pack(fill="both", expand=True)

        # Tab 1: 摘要
        self._summary_tab = ttk.Frame(self._nb)
        self._nb.add(self._summary_tab, text="  回测摘要  ")
        self._summary_text = tk.Text(self._summary_tab, wrap="word",
                                     font=("Consolas", 10), state="disabled",
                                     bg="#FAFAFA")
        self._summary_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 2: 图表
        self._chart_tab = ttk.Frame(self._nb)
        self._nb.add(self._chart_tab, text="  对冲图表  ")
        self._chart_container = ttk.Frame(self._chart_tab)
        self._chart_container.pack(fill="both", expand=True)

        # Tab 3: 波动率分析
        self._vol_tab = ttk.Frame(self._nb)
        self._nb.add(self._vol_tab, text="  波动率分析  ")
        self._vol_container = ttk.Frame(self._vol_tab)
        self._vol_container.pack(fill="both", expand=True)

        # Tab 4: 盈亏分布（蒙特卡洛）
        self._dist_tab = ttk.Frame(self._nb)
        self._nb.add(self._dist_tab, text="  盈亏分布  ")
        self._dist_container = ttk.Frame(self._dist_tab)
        self._dist_container.pack(fill="both", expand=True)

        # Tab 5: 结构分析
        self._struct_tab = ttk.Frame(self._nb)
        self._nb.add(self._struct_tab, text="  结构分析  ")
        self._struct_container = ttk.Frame(self._struct_tab)
        self._struct_container.pack(fill="both", expand=True)

        # Tab 6: 明细表
        self._table_tab = ttk.Frame(self._nb)
        self._nb.add(self._table_tab, text="  每日明细  ")

        self._toggle_source()

    # ---- 事件回调 ----
    def _on_option_class_change(self, event):
        cls_name = self._class_var.get()
        cfg = OPTION_CLASSES[cls_name]
        self._subtype_cb.configure(values=cfg["subtypes"])
        self._subtype_cb.current(0)
        self._rebuild_params(cfg["params"])

    def _rebuild_params(self, params):
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_entries = {}
        for i, (key, label, dtype, default) in enumerate(params):
            ttk.Label(self._param_frame, text=f"{label}:").grid(
                row=i, column=0, sticky="w", padx=(0, 5), pady=1)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(self._param_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky="ew", pady=1)
            self._param_entries[key] = (var, dtype)
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
        self._progress.pack(fill="x", pady=(3, 0))
        self._progress.start(15)
        threading.Thread(target=self._backtest_worker, args=(gui_state,),
                         daemon=True).start()

    def _collect_gui_state(self):
        """在主线程中读取所有 tkinter 变量，返回纯 Python dict"""
        cls_name = self._class_var.get()
        cfg = OPTION_CLASSES[cls_name]
        subtype = self._subtype_var.get()

        params = {}
        for key, (var, dtype) in self._param_entries.items():
            val_str = var.get().strip()
            if not val_str:
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
                    paths = HedgeBacktest.simulate_multi_paths(
                        s0, sigma_real, T_days, n_paths=n_paths,
                        r=r, q=q, seed=seed)

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

    def _finish_run(self):
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.pack_forget()
        self._progress_label.pack_forget()
        self._progress_label.configure(text="")
        self._run_btn.configure(state="normal")

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
            prices = HedgeBacktest.simulate_prices(s0, sigma_real, T_days, r=r, q=q, seed=seed)
            bt = HedgeBacktest(option, prices, hedge_freq=hedge_freq,
                               tc_rate=tc_rate, position=position,
                               quantity=quantity, multiplier=multiplier)

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
                                        quantity=quantity, multiplier=multiplier)

        elif src == "wind":
            code = gs["wind_code"]
            start = gs["wind_start"]
            end = gs["wind_end"]
            # 期权参数中的 s0 作为参考价 S_ref，
            # from_wind 会按 ratio = 真实起始价 / S_ref 自动缩放价格量纲要素。
            option = cfg["build"](subtype, params)
            bt = HedgeBacktest.from_wind(option, code, start, end,
                                         hedge_freq=hedge_freq, tc_rate=tc_rate,
                                         position=position,
                                         quantity=quantity, multiplier=multiplier)
        else:
            raise ValueError(f"未知数据来源: {src}")

        # 保存元信息用于展示（不再依赖 tkinter 变量）
        bt._gui_meta = {
            "cls_name": gs["cls_name"],
            "subtype": subtype,
            "source": src,
        }
        return bt

    # ---- 结果展示 ----
    def _show_results(self, bt, multi_stats=None):
        self._show_summary(bt, multi_stats)
        self._show_chart(bt)
        self._show_vol_chart(bt)
        self._show_dist_chart(multi_stats)
        self._show_table(bt)
        self._nb.select(0)

    def _show_summary(self, bt, multi_stats=None):
        r = bt._results
        n = r['n_days']
        meta = bt._gui_meta
        lines = [
            "═" * 54,
            "            动态对冲回测结果摘要",
            "═" * 54,
            "",
            f"  期权类型          :  {meta['cls_name']}",
            f"  子类型            :  {meta['subtype']}",
            f"  数据来源          :  {meta['source']}",
            "",
            "─" * 54,
            f"  回测天数          :  {n:>10d}",
            f"  调仓频率          :  每 {bt.hedge_freq} 天",
            f"  交易成本率        :  {bt.tc_rate * 100:.2f}%",
            f"  头寸方向          :  {'卖出(short)' if bt.position == 1 else '买入(long)'}",
            f"  交易数量          :  {bt.quantity:>12.2f}",
            f"  合约乘数          :  {bt.multiplier if bt.multiplier > 0 else '不取整':>12}",
            "─" * 54,
            f"  标的初始价格      :  {r['prices'][0]:>12.4f}",
            f"  标的到期价格      :  {r['prices'][-1]:>12.4f}",
            f"  标的涨跌幅        :  {(r['prices'][-1] / r['prices'][0] - 1) * 100:>11.2f}%",
            "─" * 54,
        ]
        # 真实数据回测时附加期权要素伸缩明细
        rescale_info = getattr(bt, "_rescale_info", None)
        if rescale_info is not None:
            lines.append("  【期权要素伸缩 (S_ref → S_real)】")
            lines.append(
                f"  S_ref={rescale_info['s_ref']:.4f}   "
                f"S_real={rescale_info['s_real']:.4f}   "
                f"ratio={rescale_info['ratio']:.6f}"
            )
            for name, (old, new) in rescale_info["fields"].items():
                if name in ("s0", "sr"):
                    continue
                lines.append(f"    {name:<8s}: {old:>12.4f}  →  {new:.4f}")
            lines.append("─" * 54)
        lines += [
            f"  期权初始价值      :  {r['opt_value'][0]:>12.4f}",
            f"  期权到期价值      :  {r['opt_value'][-1]:>12.4f}",
            "─" * 54,
            "  【单路径盈亏分解（种子路径）】",
            f"  标的对冲盈亏      :  {np.sum(r['hedge_daily']):>12.4f}",
            f"  期权 MtM 盈亏     :  {np.sum(r['option_daily']):>12.4f}",
            f"  累计交易成本      :  {r['total_tc']:>12.4f}",
            f"  对冲误差          :  {r['hedging_error']:>12.4f}",
            "─" * 54,
            "  【波动率分析】",
            f"  成交隐含波动率    :  {r['implied_vol'] * 100:>11.2f}%",
            f"  已实现波动率      :  {r['realized_vol'] * 100:>11.2f}%",
            f"  波动率价差        :  {r['vol_spread'] * 100:>11.2f}%  (正=卖方优势)",
            "─" * 54,
            "  【Greeks 统计】",
            f"  {'':15s}  {'初始值':>10s}  {'均值':>10s}  {'最大|值|':>10s}",
            f"  {'Delta':15s}  {r['delta'][0]:>10.4f}  {np.mean(r['delta']):>10.4f}  {np.max(np.abs(r['delta'])):>10.4f}",
            f"  {'Gamma':15s}  {r['gamma'][0]:>10.4f}  {np.mean(r['gamma']):>10.4f}  {np.max(np.abs(r['gamma'])):>10.4f}",
            f"  {'Vega':15s}  {r['vega'][0]:>10.4f}  {np.mean(r['vega']):>10.4f}  {np.max(np.abs(r['vega'])):>10.4f}",
            f"  {'Theta':15s}  {r['theta'][0]:>10.4f}  {np.mean(r['theta']):>10.4f}  {np.max(np.abs(r['theta'])):>10.4f}",
            f"  {'Rho':15s}  {r['rho'][0]:>10.4f}  {np.mean(r['rho']):>10.4f}  {np.max(np.abs(r['rho'])):>10.4f}",
            "─" * 54,
            f"  调仓次数          :  {int(np.sum(np.abs(np.diff(r['shares'])) > 1e-10)):>10d}",
            "═" * 54,
        ]

        # 蒙特卡洛多路径统计
        if multi_stats is not None:
            ms = multi_stats
            pnl = ms['total_pnl']
            n_p = ms['n_paths']
            pct = lambda arr, q: np.percentile(arr, q)
            lines += [
                "",
                "═" * 54,
                f"       蒙特卡洛多路径统计 ({n_p} 条路径)",
                "═" * 54,
                "",
                "  【总盈亏分布】",
                f"  期望盈亏(均值)    :  {np.mean(pnl):>12.4f}",
                f"  盈亏标准差        :  {np.std(pnl):>12.4f}",
                f"  盈亏中位数        :  {np.median(pnl):>12.4f}",
                f"  盈利概率          :  {np.mean(pnl > 0) * 100:>11.2f}%",
                "─" * 54,
                "  【分位数】",
                f"  1%  分位          :  {pct(pnl, 1):>12.4f}",
                f"  5%  分位          :  {pct(pnl, 5):>12.4f}",
                f"  25% 分位          :  {pct(pnl, 25):>12.4f}",
                f"  75% 分位          :  {pct(pnl, 75):>12.4f}",
                f"  95% 分位          :  {pct(pnl, 95):>12.4f}",
                f"  99% 分位          :  {pct(pnl, 99):>12.4f}",
                "─" * 54,
                f"  最大盈利          :  {np.max(pnl):>12.4f}",
                f"  最大亏损          :  {np.min(pnl):>12.4f}",
                "─" * 54,
                "  【波动率】",
                f"  隐含波动率        :  {ms['implied_vol'] * 100:>11.2f}%",
                f"  已实现波动率均值  :  {np.mean(ms['realized_vols']) * 100:>11.2f}%",
                f"  已实现波动率标准差:  {np.std(ms['realized_vols']) * 100:>11.2f}%",
                "─" * 54,
                f"  平均交易成本      :  {np.mean(ms['total_tc']):>12.4f}",
                "═" * 54,
            ]

        self._summary_text.configure(state="normal")
        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("1.0", "\n".join(lines))
        self._summary_text.configure(state="disabled")

    def _show_chart(self, bt):
        # 清除旧图表
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

        fig = Figure(figsize=(10, 9), dpi=96)

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

        fig = Figure(figsize=(10, 7), dpi=96)

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
        daily_impl = implied / np.sqrt(243.0) * 100
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
            lbl = ttk.Label(self._dist_container,
                            text="盈亏分布仅在模拟数据模式下显示（需路径数 > 1）",
                            font=("", 11))
            lbl.pack(expand=True)
            return

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        ms = multi_stats
        pnl = ms['total_pnl']
        errors = ms['errors']
        rv = ms['realized_vols']
        iv = ms['implied_vol']
        n_paths = ms['n_paths']

        fig = Figure(figsize=(10, 7), dpi=96)

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
        for w in self._table_tab.winfo_children():
            w.destroy()

        df = bt.to_dataframe()

        # Treeview
        columns = list(df.columns)
        tree_frame = ttk.Frame(self._table_tab)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        yscroll = ttk.Scrollbar(tree_frame, orient="vertical")

        tree = ttk.Treeview(tree_frame, columns=["day_no", "idx"] + columns,
                            show="headings", height=20,
                            xscrollcommand=xscroll.set,
                            yscrollcommand=yscroll.set)
        xscroll.config(command=tree.xview)
        yscroll.config(command=tree.yview)

        tree.heading("day_no", text="交易日")
        tree.column("day_no", width=55, anchor="center")
        tree.heading("idx", text=df.index.name or "日期")
        tree.column("idx", width=90, anchor="center")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=85, anchor="e")

        for i, (idx, row) in enumerate(df.iterrows()):
            idx_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
            values = [i, idx_str] + [f"{v:.4f}" if isinstance(v, float) else str(v)
                                     for v in row.values]
            tree.insert("", "end", values=values)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # 导出按钮
        btn_frame = ttk.Frame(self._table_tab)
        btn_frame.pack(fill="x", padx=5, pady=3)
        ttk.Button(btn_frame, text="导出 CSV", command=lambda: self._export_csv(df)).pack(
            side="right")

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
        self._progress.pack(fill="x", pady=(3, 0))
        self._progress.configure(mode="determinate", maximum=n_points, value=0)
        self._progress_label.configure(text=f"结构扫描: 0/{n_points}")
        self._progress_label.pack(fill="x")
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

    def _show_structure(self, gui_state, s_grid, prices, deltas,
                        gammas, vegas, thetas):
        for w in self._struct_container.winfo_children():
            w.destroy()

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        cls_name = gui_state["cls_name"]
        subtype = gui_state["subtype"]
        params = gui_state["params"]

        # 与下方回测的头寸方向联动：position=1 卖出, position=-1 买入
        position = gui_state.get("position", -1)
        sign = -1.0 if position == 1 else 1.0
        perspective = "卖方(short)" if position == 1 else "买方(long)"
        prices = prices * sign
        deltas = deltas * sign
        gammas = gammas * sign
        vegas  = vegas  * sign
        thetas = thetas * sign

        doc_key = (cls_name, subtype)
        doc_text = STRUCTURE_DOCS.get(doc_key, "(暂无结构说明)")
        header = (f"{cls_name}  /  {subtype}   [视角: {perspective}]\n"
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
        text_frame = ttk.Frame(self._struct_container)
        text_frame.pack(fill="x", padx=5, pady=(5, 3))
        text_widget = tk.Text(text_frame, wrap="word", height=11,
                              font=("Consolas", 9), bg="#FAFAFA")
        text_widget.insert("1.0", full_text)
        text_widget.configure(state="disabled")
        text_widget.pack(fill="x")

        # 图表
        chart_frame = ttk.Frame(self._struct_container)
        chart_frame.pack(fill="both", expand=True, padx=5, pady=3)

        fig = Figure(figsize=(10, 6.2), dpi=96)

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
