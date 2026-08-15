# _*_ coding: utf-8 _*_
"""左侧参数面板：控件联动、输入校验、状态收集与回测对象构造。

用户在左边填的每一个格子最终要变成一个 ``HedgeBacktest``，这条路上的四段都在
这里：换期权类型时重建参数行、各控件之间的联动与置灰、把控件读成一份
``gui_state`` dict、再由 ``_build_backtest`` 把 dict 组装成回测对象。控件**本
身**由 ``BacktestApp._build_ui`` 创建，这里只负责它们之间的关系。

拆分时的注意点与其它 Mixin 相同：组里有七个方法（``_build_backtest`` /
``_collect_gui_state*`` / ``_sync_band_inputs`` 等）在测试里被传
``SimpleNamespace`` 假 self 调用，因此凡是原先写作 ``BacktestApp._x(...)`` 的
类级调用都保持类级——纯函数改直调模块，跨 Mixin 的改成显式
``HistorySetupMixin._x(self)``，一处都没有降级成 ``self._x()``。

**它对宿主类的要求**：``_set_status``、``after`` / ``after_cancel``，
``_build_ui`` 建立的整套左侧控件与 Tk 变量，以及策略优选配置区那三个联动入口
（由 ``HistorySetupMixin`` 提供）。
"""

import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np

import history_selection
from history_selection import DEFAULT_FIXED_TIMES
from pricing import (
    CloseToCloseStrategy,
    FixedTimeStrategy,
    HedgeBacktest,
    HedgeBandStrategy,
)
from pricing.constants import ANNUAL_DAYS
from pricing.hedge_backtest import _infer_intraday_steps, _validate_fixed_time_data

from deltalab_ui import wind_resolve
from deltalab_ui.constants import (
    OPTION_CLASSES,
    SIGMA_SOURCE_FROM_DISPLAY,
    STRATEGY_DISPLAY,
    STRATEGY_FROM_DISPLAY,
    SUBTYPE_DISPLAY,
    SUBTYPE_FROM_DISPLAY,
    WIND_AUTO_BAR_SIZE,
    _WIND_BAR_MINUTES,
)
from deltalab_ui.history_setup import HistorySetupMixin
from deltalab_ui.theme import (
    FORM_ENTRY_CHARS,
    _form_grid,
    _form_input,
    _form_label,
)


class FormPanelMixin:
    """左侧参数面板的行为层；混入 ``BacktestApp``，不单独实例化。"""

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
        FormPanelMixin._sync_wind_entry_date(self)

    def _maturity_days_from_controls(self):
        """读取期权参数区的剩余期限；输入未完成时返回 None（只用于联动）。"""
        entries = getattr(self, "_param_entries", {}) or {}
        entry = entries.get("T_days") or entries.get("T")
        if entry is None:
            return None
        try:
            return wind_resolve.maturity_days_from_params(
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
        maturity_days = FormPanelMixin._maturity_days_from_controls(self)
        if maturity_days is None:
            return
        asof_var = getattr(self, "_wind_end_var", None)
        try:
            asof = wind_resolve.parse_wind_date(
                asof_var.get().strip() if asof_var is not None else "",
                "Wind 数据截止日")
            start, _asof_anchor = wind_resolve.entry_date_from_asof(
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
            asof = wind_resolve.parse_wind_date(
                asof_var.get().strip(), "Wind 数据截止日")
        except ValueError:
            return
        anchor = wind_resolve.latest_trading_day(asof)
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
            actual = wind_resolve.resolve_wind_bar_size(
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
        FormPanelMixin._toggle_strategy(self)
        FormPanelMixin._toggle_wind_auto_start(self)
        HistorySetupMixin._toggle_history_wind_controls(self)
        self._refresh_history_base_summary()
        HistorySetupMixin._sync_history_button_state(self)

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
        FormPanelMixin._refresh_wind_frequency_hint(self)

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
        if src == "csv" and FormPanelMixin._csv_is_daily(gs.get("csv_path")):
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
                nrows=FormPanelMixin._CSV_GRANULARITY_PROBE_ROWS)
            index = frame.index
            if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
                return False
            return int(_infer_intraday_steps(index)) <= 1
        except Exception:                                  # noqa: BLE001
            return False

    def _collect_gui_state(self):
        """收集单次回测状态；不读取或校验任何历史择优控件。"""
        state = FormPanelMixin._collect_gui_state_for_strategy(self)
        return wind_resolve.resolve_single_wind_state(state)

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

        wind_resolve.validate_sigma_input(
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
            wind_resolve.validate_sigma_input(real_vol, "已实现波动率")
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
            "position": wind_resolve.normalize_position(
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
        history_lookbacks = HistorySetupMixin._history_lookbacks_from_controls(self)
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
        state = FormPanelMixin._collect_gui_state_for_strategy(
            self, "close_to_close")
        history_selection.validate_source(state)
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
            band_candidates = history_selection.parse_band_candidate_sigmas(
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
            "history_objective": HistorySetupMixin._history_objective_from_controls(
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
        state = wind_resolve.resolve_history_wind_state(state)
        if include_band:
            # 在启动线程前完成当前带宽换算、去重和 10 档上限校验。
            history_selection.band_cases(state)
        return state

    def _build_backtest(self, gs):
        """根据已收集的 GUI 状态构建 HedgeBacktest 实例（可在任意线程调用）"""
        if (gs.get("source") == "wind"
                and gs.get("wind_bar_size") == WIND_AUTO_BAR_SIZE):
            # 兼容直接调用 _build_backtest 的 API / 测试；正常 GUI 路径在
            # 主线程收集状态时已经完成解析。
            gs = wind_resolve.resolve_single_wind_state(gs)
        cfg = gs["cfg"]
        subtype = gs["subtype"]
        params = gs["params"]
        src = gs["source"]
        tc_rate = gs["tc_rate"]
        position = wind_resolve.normalize_position(gs["position"])
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
