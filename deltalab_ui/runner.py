# _*_ coding: utf-8 _*_
"""后台执行：跑一次回测，或跑一轮策略优选。

两条链路形状相同——主线程收参数 → 起工作线程 → 结果经 ``after`` 回投主线程
渲染 → 统一收尾。放在一起是因为它们共用同一套任务状态（``_begin_job`` /
``_finish_job`` 仍在 ``BacktestApp``，全局只允许一个任务在跑），也共用取行情
那几步。

**测试补丁点在本模块，不在 gui_app。** ``recommend_by_rolling_history`` /
``recommend_by_contract_history_pool`` 这两个名字是从**这里**的全局取的，
测试要替身就得打在 ``deltalab_ui.runner`` 上；补在 ``gui_app`` 上不报错，只
会静默走真实计算，然后超时或出怪错。
"""

import copy
import datetime
import os
import sys
import threading
from types import SimpleNamespace
from tkinter import messagebox

import numpy as np

import history_selection
from history_selection import HISTORY_PERIOD_DEFS
from pricing import ContractHistoryPool, HedgeBacktest, StrategyCase
from pricing.hedge_analysis import (
    DEFAULT_SELECTION_OBJECTIVE,
    SELECTION_OBJECTIVES,
    recommend_by_contract_history_pool,
    recommend_by_rolling_history,
)
from pricing.hedge_backtest import (
    _infer_intraday_steps,
    _rescale_strategy_to_real_s0,
)

from deltalab_ui import snapshot_detail, wind_resolve
from deltalab_ui.view_history import HistoryViewMixin


class RunnerMixin:
    """回测与策略优选的执行链路；混入 ``BacktestApp``，不单独实例化。"""

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
                history_selection.validate_source(
                    {"source": source_var.get()})
            history_state = self._collect_history_state()
            selected_lookbacks = history_selection.normalize_lookbacks(
                history_state.get("history_lookbacks"))
            history_state["history_lookbacks"] = selected_lookbacks
            self._refresh_history_base_summary()
            history_selection.validate_source(history_state)
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
            "position": wind_resolve.normalize_position(bt.position),
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
        start_date = wind_resolve.parse_wind_date(
            gs["wind_start"], "历史行情起始日")
        end_date = wind_resolve.parse_wind_date(
            gs["wind_end"], "历史分析截至日")
        mapping = get_main_contract_history(
            product_code, start_date.isoformat(), end_date.isoformat())
        # 严格区间只加载本次所选最长证据区间内真实出现的主力合约。
        # 区间之前只为 Day 0 锚点留出必要历史，不能再因代理期限 T 把行情扩
        # 到证据区间之外。下面仍加上 warmup_days 是留给引擎的 realized 通
        # 路——本页 σ 恒取输入值，它恒为 0。
        history_lookbacks = history_selection.normalize_lookbacks(
            gs.get("history_lookbacks"))
        evidence_days = max(history_lookbacks.values())
        evidence_mapping = mapping.iloc[-min(evidence_days, len(mapping)):]
        contract_codes = list(dict.fromkeys(evidence_mapping.astype(str)))
        warmup_days = wind_resolve.history_realized_warmup_days(gs)
        prehistory_span = wind_resolve.calendar_span_for_trading_days(
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
            position=wind_resolve.normalize_position(
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
        history_selection.validate_source(gs)
        if gs.get("source") == "wind":
            from pricing.wind_data import classify_wind_history_code
            classification = classify_wind_history_code(gs.get("wind_code"))
            if classification["mode"] == "product_pool":
                return RunnerMixin._load_wind_contract_history_pool(gs)
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
        history_selection.validate_source(gs)
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
            history_selection.validate_source(gs)
            history_lookbacks = history_selection.normalize_lookbacks(
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
                base_bt = RunnerMixin._contract_pool_backtest_context(gs, history)
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
            history_selection.validate_payload(
                recommendations, ranking, window_results)
            # 趁 bar 级结果还在手里落盘：渲染之后它们就被释放了，之后想看
            # 某段明细只能重跑（620 ms/段）。现在写一次约 10 ms/段，读回
            # 只要 3 ms。仍在 worker 线程里，不挡界面。
            HistoryViewMixin._cache_history_bars(window_results)
            source_label = RunnerMixin._history_recommendation_source_label(
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
            history_selection.validate_payload(
                recommendations, ranking, window_results)
            self._show_history_recommendation(
                recommendations, ranking, notes, source_label,
                window_results, history_state)
            self._latest_history_state = (
                snapshot_detail.copy_snapshot_gui_state(history_state)
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

    def _deliver_backtest_result(self, bt, multi_stats, gui_state):
        """在主线程原子地渲染并登记最新结果，避免假成功状态。"""
        success = False
        try:
            state_copy = snapshot_detail.copy_snapshot_gui_state(gui_state)
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

    def _finish_run(self, success=True):
        self._finish_job(
            "backtest", success=success,
            success_text="回测完成  |  可保留当前结果，随后修改策略或参数继续回测",
            failure_text="回测失败  |  请查看错误信息",
        )
