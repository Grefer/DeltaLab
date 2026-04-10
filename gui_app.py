# _*_ coding: utf-8 _*_
"""
期权对冲回测 GUI 应用

基于 tkinter 构建，支持选择不同期权类型、回测方式（模拟/历史数据），
并以图表和表格形式展示回测结果。
"""

import sys
import os
import copy
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保 pricing 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing import Option_AB, Option_AS, Option_DE, HedgeBacktest
from pricing.constants import ANNUAL_DAYS

# ============================================================
#  期权类型注册表
# ============================================================

OPTION_CLASSES = {
    "气囊期权 (Airbag)": {
        "class": Option_AB,
        "subtypes": ["Opt_Airbag"],
        "params": [
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
    "亚式期权 (Asian)": {
        "class": Option_AS,
        "subtypes": ["Asian", "EnhanceAsian"],
        "params": [
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
    "累计期权 (Decumulator)": {
        "class": Option_DE,
        "subtypes": [
            "Opt_Decumulator_Back", "Opt_Decumulator_Fix",
            "Opt_EnDecumulator", "Opt_EnDecumulator_Fix",
            "Opt_ASGQ_call_put", "Opt_ASGQ_EP", "Opt_ASGQ_EF",
            "Opt_ASGQ_DP", "Opt_ASGQ_DF",
        ],
        "params": [
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
}


# ============================================================
#  主窗口
# ============================================================

class BacktestApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("期权对冲回测系统")
        self.geometry("1100x750")
        self.minsize(900, 650)
        self._setup_styles()
        self._build_ui()
        self._param_entries = {}
        self._on_option_class_change(None)

    # ---- 样式 ----
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Run.TButton", font=("Microsoft YaHei UI", 11, "bold"),
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
        ttk.Label(self._sim_frame, text="初始价格:").grid(row=0, column=0, sticky="w")
        self._s0_var = tk.StringVar(value="100")
        ttk.Entry(self._sim_frame, textvariable=self._s0_var, width=12).grid(
            row=0, column=1, padx=3)
        ttk.Label(self._sim_frame, text="种子:").grid(row=0, column=2, sticky="w")
        self._seed_var = tk.StringVar(value="42")
        ttk.Entry(self._sim_frame, textvariable=self._seed_var, width=8).grid(
            row=0, column=3, padx=3)

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
        self._wind_start_var = tk.StringVar(value="2025-01-02")
        ttk.Entry(self._wind_frame, textvariable=self._wind_start_var, width=15).grid(
            row=1, column=1, padx=3)
        ttk.Label(self._wind_frame, text="结束日:").grid(row=1, column=2, sticky="w")
        self._wind_end_var = tk.StringVar(value="2025-02-07")
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
        self._tc_var = tk.StringVar(value="0.1")
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
        self._qty_var = tk.StringVar(value="1")
        ttk.Entry(sec3, textvariable=self._qty_var, width=12).grid(
            row=row_h, column=1, sticky="w", padx=3)

        row_h += 1
        ttk.Label(sec3, text="合约乘数:").grid(row=row_h, column=0, sticky="w")
        self._mult_var = tk.StringVar(value="0")
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

        self._progress = ttk.Progressbar(btn_frame, mode="indeterminate")

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

        # Tab 3: Greeks 时序
        self._greeks_tab = ttk.Frame(self._nb)
        self._nb.add(self._greeks_tab, text="  Greeks  ")
        self._greeks_container = ttk.Frame(self._greeks_tab)
        self._greeks_container.pack(fill="both", expand=True)

        # Tab 4: 明细表
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
            "s0": self._s0_var.get().strip(),
            "seed": self._seed_var.get().strip(),
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
            self.after(0, lambda: self._show_results(bt))
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg, file=sys.stderr)
            self.after(0, lambda: messagebox.showerror("回测失败", err_msg))
        finally:
            self.after(0, self._finish_run)

    def _finish_run(self):
        self._progress.stop()
        self._progress.pack_forget()
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
            s0 = float(gs["s0"])
            seed = int(gs["seed"])
            params["s0"] = s0

            option = cfg["build"](subtype, params)

            # 获取期限天数
            T_days = params.get("T_days") or params.get("T") or 20
            sigma = params.get("sigma", 0.18)
            r = params.get("r", 0.03)
            q = params.get("q", 0.03)
            prices = HedgeBacktest.simulate_prices(s0, sigma, T_days, r=r, q=q, seed=seed)
            bt = HedgeBacktest(option, prices, hedge_freq=hedge_freq,
                               tc_rate=tc_rate, position=position,
                               quantity=quantity, multiplier=multiplier)

        elif src == "csv":
            filepath = gs["csv_path"]
            if not filepath:
                raise ValueError("请选择 CSV 文件")
            price_col = gs["csv_col"]
            params["s0"] = 0  # 占位，from_csv 会覆盖
            option = cfg["build"](subtype, params)
            bt = HedgeBacktest.from_csv(option, filepath, price_col=price_col,
                                        hedge_freq=hedge_freq, tc_rate=tc_rate,
                                        position=position,
                                        quantity=quantity, multiplier=multiplier)

        elif src == "wind":
            code = gs["wind_code"]
            start = gs["wind_start"]
            end = gs["wind_end"]
            params["s0"] = 0
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
    def _show_results(self, bt):
        self._show_summary(bt)
        self._show_chart(bt)
        self._show_greeks(bt)
        self._show_table(bt)
        self._nb.select(0)

    def _show_summary(self, bt):
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
            f"  期权初始价值      :  {r['opt_value'][0]:>12.4f}",
            f"  期权到期价值      :  {r['opt_value'][-1]:>12.4f}",
            "─" * 54,
            "  【盈亏分解】",
            f"  标的对冲盈亏      :  {np.sum(r['hedge_daily']):>12.4f}",
            f"  期权 MtM 盈亏     :  {np.sum(r['option_daily']):>12.4f}",
            f"  利息收入          :  {np.sum(r['interest_daily']):>12.4f}",
            f"  累计交易成本      :  {r['total_tc']:>12.4f}",
            f"  对冲误差          :  {r['hedging_error']:>12.4f}",
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

    def _show_greeks(self, bt):
        """专用 Greeks 时序表格"""
        for w in self._greeks_container.winfo_children():
            w.destroy()

        r = bt._results
        n = len(r['prices'])

        # 使用 Treeview 展示 Greeks 时序
        greeks_cols = ["Day", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
        tree_frame = ttk.Frame(self._greeks_container)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

        yscroll = ttk.Scrollbar(tree_frame, orient="vertical")
        tree = ttk.Treeview(tree_frame, columns=greeks_cols, show="headings",
                            height=20, yscrollcommand=yscroll.set)
        yscroll.config(command=tree.yview)

        col_widths = {"Day": 55, "Price": 90, "Delta": 90, "Gamma": 90,
                      "Vega": 90, "Theta": 100, "Rho": 90}
        for col in greeks_cols:
            tree.heading(col, text=col)
            tree.column(col, width=col_widths.get(col, 90), anchor="e")

        # 如果有 rho 数据
        rho_arr = r.get('rho', np.zeros(n))

        # 判断日期
        if hasattr(bt, '_wind_meta') and bt._wind_meta is not None:
            date_labels = [str(d)[:10] for d in bt._wind_meta['dates']]
        else:
            date_labels = [str(i) for i in range(n)]

        for i in range(n):
            values = [
                date_labels[i],
                f"{r['prices'][i]:.4f}",
                f"{r['delta'][i]:.6f}",
                f"{r['gamma'][i]:.6f}",
                f"{r['vega'][i]:.4f}",
                f"{r['theta'][i]:.4f}",
                f"{rho_arr[i]:.6f}",
            ]
            tree.insert("", "end", values=values)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

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

        tree = ttk.Treeview(tree_frame, columns=["idx"] + columns,
                            show="headings", height=20,
                            xscrollcommand=xscroll.set,
                            yscrollcommand=yscroll.set)
        xscroll.config(command=tree.xview)
        yscroll.config(command=tree.yview)

        tree.heading("idx", text=df.index.name or "日期")
        tree.column("idx", width=90, anchor="center")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=85, anchor="e")

        for idx, row in df.iterrows():
            idx_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
            values = [idx_str] + [f"{v:.4f}" if isinstance(v, float) else str(v)
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
