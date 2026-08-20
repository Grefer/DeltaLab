# _*_ coding: utf-8 _*_
"""策略优选的配置区，以及结果包的存取。

跑之前要填的那一整块——候选空间、周期勾选、行情区间与粒度、基准摘要条——都在
这里；跑完之后的呈现在 ``deltalab_ui.view_history``。结果包的保存/载入/管理窗
口也归这里：它们复用同一批配置控件，载入一份旧结果就是把控件回填回去再交给
结果页渲染。

``_load_history_result`` 用显式类级调用 ``HistoryViewMixin._show_history_
recommendation(self, ...)``，与原文件里的 ``BacktestApp._show_history_
recommendation(self, ...)`` 语义一致（都不走实例查找，因此不会被替身覆盖）。

**它对宿主类的要求**：``_set_status``、``_on_option_class_change``，以及
``_build_ui`` 建立的那批 ``_history_*`` 控件与 Tk 变量。
"""

import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import tkinter.font as tkfont

import numpy as np

import history_selection
import history_store
from history_selection import (
    DEFAULT_BAND_CANDIDATE_SIGMAS,
    DEFAULT_FIXED_TIMES,
    HISTORY_PERIOD_DEFS,
)
from pricing import CloseToCloseStrategy, FixedTimeStrategy, HedgeBandStrategy
from pricing.hedge_analysis import (
    DEFAULT_SELECTION_OBJECTIVE,
    SELECTION_OBJECTIVES,
    HistoryReplaySpec,
    rerank_history,
)
from pricing.hedge_backtest import _rescale_option_to_real_s0

from deltalab_ui import snapshot_detail, widgets, wind_resolve
from deltalab_ui.constants import (
    HISTORY_OBJECTIVE_DISPLAY,
    HISTORY_OBJECTIVE_FROM_DISPLAY,
    OPTION_CLASSES,
    SUBTYPE_DISPLAY,
    WIND_AUTO_BAR_SIZE,
)
from deltalab_ui.theme import PALETTE, _UI_FONT_FAMILY
from deltalab_ui.view_history import HistoryViewMixin


class HistorySetupMixin:
    """策略优选的配置区与结果包存取；混入 ``BacktestApp``，不单独实例化。"""

    def _build_history_workspace(self):
        """构建独立历史择优工作区，避免占用回测结果对比容器。"""
        container = self._history_container
        for widget in container.winfo_children():
            widget.destroy()

        header = ttk.Frame(container, style="Surface.TFrame")
        header.pack(fill="x", padx=10, pady=(10, 8))
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
            header, text="📂 结果库", width=9, style="Ghost.TButton",
            command=self._open_history_result_store,
        )
        self._history_open_store_btn.grid(
            row=0, column=1, sticky="e", padx=(12, 0))
        self._history_save_btn = ttk.Button(
            header, text="💾 保存", width=8, state="disabled",
            command=self._save_history_result,
        )
        self._history_save_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))

        self._history_config_visible = True
        self._history_config_toggle_btn = ttk.Button(
            header, text="▲ 收起配置", width=10, style="Ghost.TButton",
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
            header, text="▶ 开始优选", style="Run.TButton",
            command=self._run_history_recommendation,
        )
        self._history_btn.grid(
            row=0, column=4, sticky="e", padx=(6, 0))

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
        widgets.track_wraplength(base_summary)

        settings = ttk.LabelFrame(
            self._history_config_panel,
            text=" 候选空间（仅属于策略优选） ", padding=(16, 10))
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
            font=(_UI_FONT_FAMILY, 8),
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
                HistorySetupMixin._refresh_history_auto_wind_start(self),
                HistorySetupMixin._refresh_history_base_summary(self),
            ),
        )
        self._wind_code_var.trace_add(
            "write",
            lambda *_args: (
                HistorySetupMixin._refresh_history_wind_hint(self),
                HistorySetupMixin._refresh_history_base_summary(self),
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
        HistorySetupMixin._update_history_header_summary(self)

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
                font=(_UI_FONT_FAMILY, 9, "bold"),
                highlightbackground=PALETTE["primary"], highlightthickness=1,
                padx=10, pady=3,
            )
            loaded_pill.pack(side="left", padx=(0, 6))

        hint = getattr(self, "_history_source_hint_var", None)
        hint_text = hint.get().strip() if hint is not None else ""
        if hint_text and not loaded:
            pill = tk.Label(
                container, text=f"⚠ {hint_text}",
                bg=PALETTE["warning_light"], fg=PALETTE["warning"],
                font=(_UI_FONT_FAMILY, 9, "bold"),
                highlightbackground=PALETTE["warning"], highlightthickness=1,
                padx=10, pady=3,
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
                font=(_UI_FONT_FAMILY, 9),
                highlightbackground=PALETTE["border_soft"], highlightthickness=1,
                padx=10, pady=3,
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
                font=(_UI_FONT_FAMILY, 9),
                highlightbackground=PALETTE["border_soft"], highlightthickness=1,
                padx=10, pady=3,
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
            button.configure(text="▼ 展开配置")
        else:
            before = getattr(self, "_history_results_container", None)
            pack_options = {
                "fill": "x", "padx": 8, "pady": (0, 5),
            }
            if before is not None:
                pack_options["before"] = before
            panel.pack(**pack_options)
            self._history_config_visible = True
            button.configure(text="▲ 收起配置")

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
            HistoryViewMixin._show_history_recommendation(
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
        self._latest_history_state = snapshot_detail.copy_snapshot_gui_state(
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
            sigma = history_selection.finite_value(
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
            base_option = HistorySetupMixin._option_from_state(state)
        if base_option is None:
            return {}
        strategies = {}
        for _index, row in ranking.iterrows():
            item = row.to_dict()
            built = HistorySetupMixin._strategy_from_ranking_row(
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
                        strategies=HistorySetupMixin._segment_strategies(
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
                self._on_option_class_change(None)
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
        # 子类型用中文名，最长「敲出计零·区间固赔·到期杠杆累计」在这张表的
        # TkDefaultFont 下实测 189px，加单元格内边距按 210 留（旧的 9 汉字名
        # 只要 150）。八列之和 1029px，窗口 1180 扣掉 padding 与滚动条后还有
        # 约 100px 富余，都归 stretch=True 的「名称」列。
        widths = {"label": 170, "code": 92, "option": 92, "subtype": 210,
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
            HistorySetupMixin._sync_history_store_buttons(
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
            # delete_result 对「本来就不存在」和「删不掉」都返回 False，真正
            # 的失败判据是「没删成且文件还在」。refresh() 会把删不掉的那条重
            # 新列出来，但列表里凭空多回一行不会被读成"删除失败"，得明说。
            if (not history_store.delete_result(item["path"])
                    and os.path.exists(item["path"])):
                messagebox.showerror(
                    "删除失败",
                    f"删不掉「{name}」的结果文件（可能是目录只读，或文件正被"
                    f"别的程序占用）：\n\n{item['path']}",
                    parent=window)
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
            HistorySetupMixin._sync_history_store_buttons(
                tree, by_iid, load_btn, rename_btn, delete_btn)
            item = selected()
            # 标的、周期构成、大小、来源都不单独占列——它们是「确认这是不
            # 是我要的那份」时才看一眼的细节，放在选中后的这一行足够。
            self._history_store_hint_var.set(
                HistorySetupMixin._history_store_detail_line(item) if item else "")

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

    def _show_empty_history_results(self):
        container = getattr(self, "_history_results_container", None)
        if container is None:
            return
        for widget in container.winfo_children():
            widget.destroy()
        placeholder = tk.Frame(container, bg=PALETTE["surface"])
        placeholder.place(relx=0.5, rely=0.45, anchor="center")

        icon_lbl = tk.Label(
            placeholder, text="🎯",
            font=(_UI_FONT_FAMILY, 42),
            bg=PALETTE["surface"], fg=PALETTE["text_muted"],
        )
        icon_lbl.pack(pady=(0, 10), padx=(14, 0))

        title_lbl = tk.Label(
            placeholder, text="策略优选",
            font=(_UI_FONT_FAMILY, 16, "bold"),
            bg=PALETTE["surface"], fg=PALETTE["text"],
        )
        title_lbl.pack(pady=(0, 6))

        desc_lbl = tk.Label(
            placeholder,
            text="设置上方候选参数与分析周期，点击「▶ 开始策略优选」生成多周期回测排名与最优方案",
            font=(_UI_FONT_FAMILY, 10),
            bg=PALETTE["surface"], fg=PALETTE["text_muted"],
            wraplength=620, justify="center",
        )
        desc_lbl.pack(pady=(0, 0))

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
            return history_selection.normalize_lookbacks()
        flags = {
            key: bool(variable.get())
            for key, variable in variables.items()
        }
        if require_one:
            return history_selection.normalize_lookbacks(flags)
        try:
            return history_selection.normalize_lookbacks(flags)
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
        lookbacks = HistorySetupMixin._history_lookbacks_from_controls(
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
            asof = wind_resolve.parse_wind_date(
                asof_var.get(), "历史分析截至日")
        except (ValueError, tk.TclError):
            # 截至日正在输入、还不是合法日期时不猜；用户填完自然会再触发。
            return
        include_band_var = getattr(self, "_history_include_band_var", None)
        warmup_days = wind_resolve.history_realized_warmup_days({
            "history_include_band": (
                bool(include_band_var.get())
                if include_band_var is not None else False),
            # 本页恒用输入波动率，与 _collect_history_state 写入的一致。
            "sigma_source": "implied",
        })
        required = max(lookbacks.values()) + warmup_days + 1
        start = wind_resolve.history_auto_wind_start(
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
        HistorySetupMixin._refresh_history_auto_wind_start(self)
        if getattr(self, "_history_wind_hint_var", None) is not None:
            HistorySetupMixin._refresh_history_wind_hint(self)
        if getattr(self, "_history_base_summary_var", None) is not None:
            HistorySetupMixin._refresh_history_base_summary(self)
        # 勾选数为 0 时主按钮要禁用，所以每次改动都得重算按钮状态。
        HistorySetupMixin._sync_history_button_state(self)

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
        HistorySetupMixin._toggle_history_wind_controls(self)

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
            return wind_resolve.resolve_wind_bar_size(
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
            bar = HistorySetupMixin._history_summary_bar_label(self)
            # 自动模式也直接写推算出的起始日，而不是"自动覆盖最近 243 日连续
            # 区间"这种规则描述。这一行讲的是"本次将冻结的取数区间"，规则描
            # 述放在这里读者还得自己换算成日期；而起始日已经实时回填进旁边那
            # 个输入框，两处显示同一个日期才对得上。周期没勾时才退回提示语。
            start = start_var.get().strip() if start_var is not None else "—"
            if auto_start_var is not None and auto_start_var.get():
                lookbacks = HistorySetupMixin._history_lookbacks_from_controls(
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
        selected_lookbacks = HistorySetupMixin._history_lookbacks_from_controls(
            self, require_one=False)
        period_labels = {
            key: label for key, label in HISTORY_PERIOD_DEFS}
        selected_period_text = "、".join(
            period_labels[key] for key in selected_lookbacks)
        if not selected_period_text:
            selected_period_text = "未选择"
        variable.set(
            f"📌 基准参照 ｜ 【行情】{source_text} ｜ 【合约】{subtype or '—'}（期限 T={maturity or '—'}日） ｜ "
            f"【周期】{selected_period_text} ｜ 【头寸】{position} 数量{quantity} 乘数{multiplier} ｜ "
            f"【成本】费率{tc}% 滑点{slippage}bps 保底{close_fallback} ｜ 左侧期权参数")

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
        HistorySetupMixin._refresh_history_auto_wind_start(self)
        HistorySetupMixin._refresh_history_wind_hint(self)
        if getattr(self, "_history_base_summary_var", None) is not None:
            HistorySetupMixin._refresh_history_base_summary(self)

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
            wind_resolve.resolve_wind_bar_size(
                WIND_AUTO_BAR_SIZE, fixed_times=fixed_times,
                include_fixed_times=include_fixed,
                include_band=include_band, wind_code=wind_code)
        except ValueError as exc:
            hint.set(str(exc))
            return
        auto_start_var = getattr(
            self, "_history_wind_auto_start_var", None)
        if auto_start_var is None or auto_start_var.get():
            lookbacks = HistorySetupMixin._history_lookbacks_from_controls(
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

    def _sync_history_button_state(self):
        """使策略优选入口始终与真实数据来源及后台任务状态一致。"""
        source_var = getattr(self, "_source_var", None)
        if source_var is None:
            return
        # 一个周期都没勾时也要禁用：此前按钮照常可点，点下去才弹
        # 「请至少选择一个历史分析周期。」的模态框——用模态错误代替禁用态，
        # 等于让用户先撞一次才知道不能按。
        try:
            has_period = bool(HistorySetupMixin._history_lookbacks_from_controls(
                self, require_one=False))
        except (AttributeError, tk.TclError):
            has_period = True      # 控件还没建好时不误禁
        # 优选跑起来之后这个按钮改当"停止"用：一轮优选要跑十几分钟，
        # 此前它只是变灰，用户除了等就只能杀进程。中止请求在段/候选边界
        # 生效，已跑完的部分照常释放资源。
        running = getattr(self, "_active_job", None) == "history"
        cancelling = running and bool(
            getattr(self, "_history_cancel_event", None) is not None
            and self._history_cancel_event.is_set())
        enabled = (running and not cancelling) or (
            getattr(self, "_active_job", None) is None
            and source_var.get() in ("csv", "wind")
            and has_period
        )
        state = "normal" if enabled else "disabled"
        if running:
            btn_text = "■ 停止优选"
            btn_command = getattr(
                self, "_cancel_history_recommendation", None)
        else:
            btn_text = "▶ 开始优选"
            btn_command = getattr(
                self, "_run_history_recommendation", None)
        for attr in ("_history_btn",):
            button = getattr(self, attr, None)
            if button is None:
                continue
            try:
                exists = getattr(button, "winfo_exists", None)
                if exists is not None and not exists():
                    setattr(self, attr, None)
                    continue
                button.configure(state=state, text=btn_text)
                if btn_command is not None:
                    button.configure(command=btn_command)
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
        HistorySetupMixin._update_history_header_summary(self)
