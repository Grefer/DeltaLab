# _*_ coding: utf-8 _*_
"""单次回测的结果呈现：摘要文本、四张图、明细表与结构分析。

``BacktestApp`` 跑完一次回测后的全部输出都在这里。这一块是整个界面里最独立
的一段——它只往下写（往 notebook 的几个页签里塞控件和 Figure），不参与参数
联动，也不回头改左侧表单，因此第一个被拆成 Mixin。

**它对宿主类的要求**（都由 ``BacktestApp.__init__`` / ``_build_ui`` 建立）：

* 页签与容器：``_nb``、``_summary_text``、``_table_tab``、``_struct_tab``、
  ``_chart_container`` / ``_vol_container`` / ``_dist_container`` /
  ``_struct_container``，以及与之配对的 ``_*_figure`` / ``_*_canvas`` 句柄
* 结构分析的输入控件：``_struct_range_var``、``_struct_npts_var``
* 进度与任务状态：``_progress``、``_progress_label``、``_begin_job``、
  ``_finish_job``
* 画布尺寸与占位符：``_CHART_DPI``、``_container_figsize``、
  ``_reset_figure_container``、``_hide_placeholder``、
  ``_clear_tab_content_preserving_placeholder``
* 参数收集：``_collect_gui_state``

Python 不会替我们检查这些，所以别把清单当装饰——加新依赖时同步更新它，是
后来人判断"这个 Mixin 能不能单独复用"的唯一线索。
"""

import sys
import threading
from collections.abc import Mapping
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from pricing import HedgeBandStrategy
from pricing.constants import ANNUAL_DAYS
from pricing.hedge_backtest import histogram_bin_edges, padded_histogram_range

from deltalab_ui.constants import SUBTYPE_DISPLAY
from deltalab_ui.structure_docs import STRUCTURE_DOCS
from deltalab_ui.theme import (
    PALETTE,
    _MONO_FONT_FAMILY,
    _MPL_CJK_FP,
    _UI_FONT_FAMILY,
)


class ResultsMixin:
    """回测结果的呈现层；混入 ``BacktestApp``，不单独实例化。"""

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
            gv = ResultsMixin._result_greek_series(r, gn)
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
            days, ResultsMixin._result_greek_series(r, "gamma"),
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
            days, ResultsMixin._result_greek_series(r, "vega"),
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
            days, ResultsMixin._result_greek_series(r, "theta"),
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
