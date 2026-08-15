# _*_ coding: utf-8 _*_
"""策略优选的结果页：排名表、周期切换、结论卡、曲线图与分段重放。

跑完一轮历史择优之后的全部呈现都在这里——左边一张按周期分组的候选排名表，
右边是选中候选的诊断明细与累计曲线，底部那条是"把某一段重新跑成 bar 级明细"
的重放条。配置区（跑之前填的那些控件）不在这里，仍在 ``BacktestApp``。

原文件里这些方法大量写作 ``BacktestApp._x(...)`` 的类级调用，而且是**有意**
的：好几个测试传一个 ``SimpleNamespace`` 假 self 进来，只有类级调用才走得通。
Mixin 里够不着 ``BacktestApp`` 这个名字，所以按被调对象分了三种改法——

* 本来就是纯函数的（``_format_comparison_value``、``_strategy_style_key``、
  ``_attach_tooltip`` 等）：连同实现一起下沉到 ``deltalab_ui`` 的
  ``formatting`` / ``widgets`` / ``snapshot_detail``，这里直接调模块函数，
  假 self 那条路照样通。
* 指向纯逻辑层的别名：改成直接调 ``history_selection.*``。
* 真正的实例方法（``_open_params_window`` 等四个）：改成 ``self._x()``。这几处
  的调用方在测试里传的都是真 ``BacktestApp`` 实例，语义不变。

**它对宿主类的要求**：

* 结果数据：``_history_recommendations``、``_history_ranking``、
  ``_history_details``、``_history_state``、``_latest_backtest`` 等运行结果
* 页面控件：``_history_container``、排名 Treeview、周期 chip、结论卡与图表画布
* 宿主方法：``_begin_job`` / ``_finish_job``、``_set_status``、``_show_results``、
  ``_strategy_style``、``_toggle_strategy``、``_sync_band_inputs``、
  ``_mark_band_edited``、``_refresh_history_current_band_label``、
  ``_open_params_window``、``_update_history_header_summary``、
  ``_sync_history_save_button``

七张 ``_HISTORY_*`` 常量表随方法一起搬进本类；它们在 MRO 上，类外仍按
``BacktestApp._HISTORY_METRIC_COLUMNS`` 访问得到。
"""

import copy
import sys
import threading
from collections.abc import Mapping
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import history_bar_cache
import history_selection
from history_selection import HISTORY_PERIOD_DEFS, MAX_HISTORY_CHART_CANDIDATES
from pricing import (
    FixedTimeStrategy,
    HedgeBandStrategy,
    history_replay_index,
    history_window_summary,
)
from pricing.hedge_analysis import (
    DEFAULT_SELECTION_OBJECTIVE,
    SELECTION_OBJECTIVES,
    _aggregate_result_by_day,
    rerank_history,
)
from pricing.hedge_backtest import _PRICE_FIELDS_BY_CLS

from deltalab_ui import formatting, snapshot_detail, widgets
from deltalab_ui.constants import (
    HISTORY_CHART_METRIC_DISPLAY,
    HISTORY_CHART_METRIC_FROM_DISPLAY,
    SIGMA_SOURCE_DISPLAY,
    STRATEGY_DISPLAY,
    _OBJECTIVE_COLUMN_KEYS,
    _OBJECTIVE_RANKING_COLUMNS,
)
from deltalab_ui.theme import PALETTE, _MONO_FONT_FAMILY, _UI_FONT_FAMILY
# theme 在模块顶部选定 matplotlib 后端，pyplot 必须晚于它 import。
import matplotlib.pyplot as plt


class HistoryViewMixin:
    """策略优选结果页；混入 ``BacktestApp``，不单独实例化。"""

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
            "_history_sash_placer",
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
        self._update_history_header_summary()
        self._sync_history_save_button()
        if notes:
            first_note = str(notes[0])
            short_note = first_note if len(first_note) <= 92 else first_note[:89] + "…"
            note_bar = tk.Frame(
                staging, bg=PALETTE["warning_light"], bd=0,
                highlightbackground=PALETTE["warning"], highlightthickness=1)
            note_bar.pack(fill="x", padx=4, pady=(0, 6))
            tk.Frame(note_bar, bg=PALETTE["warning"], width=4).pack(side="left", fill="y")
            inner = tk.Frame(note_bar, bg=PALETTE["warning_light"], padx=12, pady=6)
            inner.pack(side="left", fill="x", expand=True)
            tk.Label(
                inner, text=f"⚠ {len(notes)} 条说明：{short_note}",
                bg=PALETTE["warning_light"], fg=PALETTE["warning"],
                font=(_UI_FONT_FAMILY, 9, "bold"), anchor="w",
            ).pack(side="left", fill="x", expand=True)
            ttk.Button(
                inner, text="🔍 查看全部", width=9,
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
        self._update_history_header_summary()

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
            self._toggle_history_config_panel()
        self._nb.select(self._history_tab)

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
        return HistoryViewMixin._HISTORY_COLUMN_ANCHORS.get(
            key, HistoryViewMixin._HISTORY_NUMERIC_ANCHOR)

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
            anchor = HistoryViewMixin._history_column_anchor(key)
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
        number = history_selection.finite_value(value)
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
        number = history_selection.finite_value(value)
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
            + HistoryViewMixin._HISTORY_METRIC_COLUMNS
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
            anchor = HistoryViewMixin._history_column_anchor(key)
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
                anchor=HistoryViewMixin._history_column_anchor(key),
                stretch=True,
            )
        return tree

    def _build_history_comparison_view(
            self, parent, recommendations, ranking, window_results=None,
            history_lookbacks=None, details=None):
        """按“区间说明 / 周期结论 / 周期排名与动作 / 图表”四段装配结果页。"""
        self._history_ranking = ranking
        self._history_lookbacks = history_selection.normalize_lookbacks(
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
        HistoryViewMixin._build_history_scope_note(
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
            # 填完才知道排名表的请求高度，这时补摆一次分割线——上面那次是在
            # 空表上算的，会被填充后的请求高度顶下去。
            splitter.after_idle(self._history_sash_placer)

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
                    identities[name] = formatting.history_row_style_key(item)
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
            HistoryViewMixin._refresh_history_period_chips(self)
        self._update_history_selection()

    def _history_rerender(self, recommendations, ranking, details):
        """用新的排名结果重建结果页，沿用本次已有的分段明细。

        切排名口径走的也是这条路。载入标记必须原样带过去——它描述的是「这
        份结果哪来的」，换个看法不会把载入的结果变成刚跑的。
        """
        if getattr(self, "_history_results_container", None) is None:
            return
        HistoryViewMixin._show_history_recommendation(
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
            return history_selection.safe_int(item.get("paired", 0), 0)

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
        note, state = HistoryViewMixin._history_consensus_note(items)

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
                bg=PALETTE["surface_alt"], fg=PALETTE["text"],
                highlightbackground=PALETTE["border_soft"], highlightthickness=1,
            )
            chip.pack(side="left", padx=(0, 6))
            chip.bind(
                "<Button-1>",
                lambda _event, k=key: HistoryViewMixin._select_history_period(
                    self, k))
            best = str(item.get("strategy", "") or "").strip()
            widgets.attach_tooltip(
                chip,
                f"{item.get('period', key)} 最优：{best}" if best and best != "—"
                else f"{item.get('period', key)}：无可比结论")
            self._history_period_chips[key] = chip
        if self._history_period_rows:
            self._history_selected_period_var.set(
                HistoryViewMixin._default_history_period(self._history_period_rows))
            HistoryViewMixin._refresh_history_period_chips(self)

        pill_bg = {
            "agree": PALETTE["success_light"],
            "disagree": PALETTE["warning_light"],
        }.get(state, PALETTE["surface_alt"])
        pill_fg = {
            "agree": PALETTE["success"],
            "disagree": PALETTE["warning"],
        }.get(state, PALETTE["text_muted"])
        pill_border = {
            "agree": "#BBF7D0",
            "disagree": "#FDE68A",
        }.get(state, PALETTE["border_soft"])
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
            font=(_UI_FONT_FAMILY, 9, "bold"), padx=12, pady=4,
            highlightbackground=pill_border, highlightthickness=1,
        )
        consensus_pill.pack(side="right")
        widgets.attach_tooltip(consensus_pill, note)

        # 周期本身的取样口径（合约期限、回放天数、分段构成…）另起一行：
        # 它常长到七八项，挤在周期按钮同一行会把 pill 顶出可视区。
        self._history_period_context_var = tk.StringVar(value="")
        period_context = ttk.Label(
            box, textvariable=self._history_period_context_var,
            style="SurfaceMuted.TLabel", justify="left")
        period_context.pack(fill="x", pady=(0, 2))
        widgets.track_wraplength(period_context)

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
            chart_selection_bar, text="✕ 清空勾选", width=9,
            style="Ghost.TButton",
            command=self._clear_history_chart_candidates,
        ).pack(side="right", padx=(8, 0))

        rank_tree = HistoryViewMixin._build_history_metric_tree(
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

        所以**目标位只在真实尺寸变化时重算**：相对幅度小于
        ``_HISTORY_SASH_ECHO_TOLERANCE`` 的高度变化是自己引起的回波，不重
        算；用户拉窗口那种大幅变化才按比例取新目标。另有两道兜底——已经在
        目标位就不动（minsize 钳住目标时不会反复试），以及一个硬性次数预
        算。平时也不需要持续跟随：两格都是 ``stretch="always"``，窗口缩放
        时 tk 自己会按比例分配。

        **但"不重算目标"不等于"不回到目标"。** 分割线还会被另一件事推走：
        排名表是在本函数之后才填充的，填完那一刻它的请求高度从空表的两三
        行涨到十来行，PanedWindow 据此重排，分割线被顶下去——而 splitter
        自己的高度一点没变，于是上面那条回波判据把这次纠正也一起挡掉了。
        表现就是默认窗口下载入一份优选结果，图表被压到只剩三成高。所以只
        要还没交给用户，每次 ``<Configure>`` 都把分割线拉回当前目标位，重
        算与否只影响目标是多少。这不会重新引出自激：拉回之后分割线就等于
        目标位，下一次 ``<Configure>`` 在第一道判断就返回了。
        """
        state = {"owned": False, "height": None, "target": None,
                 "left": HistoryViewMixin._HISTORY_SASH_PLACEMENT_BUDGET,
                 "pending": False}

        def _apply():
            # 真正摆分割线的动作推到 idle 才做。在 ``<Configure>`` 里同步摆会
            # 被这一轮尚未走完的几何计算覆盖掉：窗口拉高时实测摆到 355 之后
            # 仍然停在 437，因为重排是在事件回调之后才结束的。
            state["pending"] = False
            if state["owned"] or state["left"] <= 0 or state["target"] is None:
                return
            target = state["target"]
            try:
                if abs(splitter.sash_coord(0)[1] - target) <= 2:
                    return
                state["left"] -= 1
                splitter.sash_place(0, 0, target)
            except tk.TclError:
                pass

        def _place(_event=None):
            if state["owned"] or state["left"] <= 0:
                return
            height = splitter.winfo_height()
            if height <= 1:
                return
            previous = state["height"]
            resized = previous is None or abs(height - previous) > (
                previous * HistoryViewMixin._HISTORY_SASH_ECHO_TOLERANCE)
            if resized:
                state["height"] = height
                state["target"] = max(150, int(height * ratio))
            elif state["target"] is None:
                return
            # pending 标志防止一串 <Configure>（拖窗口时每帧都有）堆出一串
            # 回调；它们要做的是同一件事。
            if not state["pending"]:
                state["pending"] = True
                splitter.after_idle(_apply)

        # 排名表填完之后要再摆一次，见上面第二段。这里存下入口而不是等
        # ``<Configure>`` 自己来：填充是否会顺带触发一次事件取决于行数有没有
        # 真的改变控件尺寸，赌它必然发生就会剩下一个偶发的压扁。
        self._history_sash_placer = _place

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
        card.grid(row=2, column=0, sticky="ew", padx=2, pady=(4, 2))
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
            font=(_UI_FONT_FAMILY, 15, "bold"), anchor="w", justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(1, 6))

        self._history_conclusion_stats = tk.Frame(
            body, bg=PALETTE["surface_alt"])
        self._history_conclusion_stats.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        self._build_history_action_bar(body)
        HistoryViewMixin._refresh_history_conclusion_card(self)

    def _build_history_action_bar(self, body):
        """把结论用到当下的动作栏；逐段下钻另在图表下方，两者对象不同。"""
        actions = tk.Frame(body, bg=PALETTE["surface_alt"])
        actions.grid(row=0, column=1, rowspan=3, sticky="ne", padx=(18, 0), pady=(4, 0))
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
            actions, text="应用策略", width=14,
            style="Accent.TButton",
            command=lambda: HistoryViewMixin._apply_history_recommendation_safely(
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
        finite = history_selection.finite_value
        win_rate = finite(row.get("window_win_rate_vs_c2c"))
        paired = history_selection.safe_int(
            row.get("paired_windows", row.get("rolling_windows")), 0)
        baseline_windows = history_selection.safe_int(
            row.get("baseline_windows", paired), paired)
        # 第三个元素只驱动着色，规则是 >0 绿、<0 红。胜段占比是 [0,1] 的
        # 比例，中性点在 **0.5** 而不是 0——直接把它传下去，5 段里只赢 1 段
        # （20%，明显比基准差）也会被染成"优于基准"的绿色。减掉 0.5 之后
        # 才是"比一半多还是少"，与其余增量列"正=更好"的语义对齐。
        tail = (
            ("胜段占比", formatting.format_comparison_value(
                win_rate, 0, percent=True),
             None if win_rate is None else win_rate - 0.5),
            ("可比样本", f"{paired}/{baseline_windows} 段", None),
        )
        if getattr(self, "_history_objectives_available", True):
            incremental_pnl = finite(row.get("incremental_pnl_vs_c2c"))
            incremental_sharpe = finite(row.get("incremental_sharpe_vs_c2c"))
            return (
                ("增量收益 vs 每日收盘",
                 HistoryViewMixin._format_objective_value(incremental_pnl, 2),
                 incremental_pnl),
                ("增量信噪比",
                 HistoryViewMixin._format_objective_value(incremental_sharpe, 4),
                 incremental_sharpe),
            ) + tail
        return (
            ("损益波动", formatting.format_comparison_value(
                row.get("daily_net_pnl_rms"), 2), None),
            ("每日收盘", formatting.format_comparison_value(
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
            HistoryViewMixin._set_history_conclusion_accent(
                self, PALETTE["border_soft"])
            return

        is_baseline = str(
            row.get("strategy_type", "")) == "close_to_close"
        recommendation_eligible = history_selection.safe_bool(
            row.get("recommendation_eligible"),
            history_selection.safe_bool(
                row.get("complete_window"), False))
        rank_val = history_selection.safe_int(row.get("rank"), 0)
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
        HistoryViewMixin._set_history_conclusion_accent(self, accent)

        stat_specs = list(self._history_conclusion_stat_specs(row))
        for column, (caption, text, signed) in enumerate(stat_specs):
            stats.columnconfigure(column, weight=1, uniform="stat_tile", minsize=130)
            tile = tk.Frame(
                stats, bg=PALETTE["surface"],
                highlightbackground=PALETTE["border_soft"], highlightthickness=1,
                padx=12, pady=7)
            tile.grid(
                row=0, column=column, sticky="nsew",
                padx=(0, 8 if column < len(stat_specs) - 1 else 0),
            )
            tk.Label(
                tile, text=caption, bg=PALETTE["surface"],
                fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 8),
                anchor="w",
            ).pack(fill="x", anchor="w")
            if signed is None or is_baseline:
                value_fg = PALETTE["text"]
            elif signed > 0:
                value_fg = PALETTE["success"]
            elif signed < 0:
                value_fg = PALETTE["danger"]
            else:
                value_fg = PALETTE["text_muted"]
            tk.Label(
                tile, text=text, bg=PALETTE["surface"], fg=value_fg,
                font=(_UI_FONT_FAMILY, 12, "bold"), anchor="w",
            ).pack(fill="x", anchor="w", pady=(2, 0))

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
            limit = HistoryViewMixin._HISTORY_DEFAULT_CHART_RANKS
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
            model = history_selection.multi_chart_model(
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
        self._open_params_window(heading=str(row.get("strategy", "未命名候选")),
            subtitle=HistoryViewMixin._history_params_subtitle(row, history_state),
            sections=snapshot_detail.history_row_detail_sections(
                row, history_state),
            notes={"期权类型与参数": HistoryViewMixin._HISTORY_CONTRACT_NOTE},
        )

    @staticmethod
    def _history_params_subtitle(row, history_state):
        """详情窗副标题：这一行属于哪个周期、跑在哪份行情上。"""
        parts = ["策略优选"]
        period = str(row.get("period", "") or "").strip()
        if period:
            parts.append(f"周期 {period}")
        source = formatting.snapshot_source_label(history_state)
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
        pairs = history_selection.chart_pairs(
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
            preview = history_selection.multi_chart_model(
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
                rms_text = formatting.format_comparison_value(
                    row.get("daily_net_pnl_rms"), 2)
                baseline_text = formatting.format_comparison_value(
                    row.get("baseline_daily_net_pnl_rms"), 2)
                # 详情行只补表格没有的原始损益波动值。表格现在的四列是
                # 增量收益/增量信噪比/增量成本/最大回撤，改善率与胜率都
                # 不在其中——它们是刻意收敛掉的旧口径，不再单独渲染。
                uses_strict_metric = (
                    history_selection.row_uses_strict_metric(row))
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
                        formatting.format_comparison_value(
                            history_selection.row_improvement(row),
                            1, signed=True, percent=True))
                    selected_detail += (
                        f" · 旧版改善 {improvement_text}"
                        " · 建议重新运行")

                if row.get("history_mode") == "product_contract_pool":
                    code_text = history_selection.contract_codes_text(
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
        HistoryViewMixin._refresh_history_conclusion_card(self)
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
            start = formatting.format_detail_index(index[0])
            end = formatting.format_detail_index(index[-1])
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
                "label": HistoryViewMixin._history_replay_label(spec),
                "spec": spec,
            })
        options.sort(key=lambda item: history_selection.safe_int(
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
        label = HistoryViewMixin._history_replay_label(spec)
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
            bt, recomputed = HistoryViewMixin._replay_with_cache(
                spec, strategy_name)
            # 与包内逐日损益核对。不一致时照常展示（数据本身可能仍有诊断
            # 价值），但必须明说——静默显示对不上的数字才是最坏的结果。
            mismatch = HistoryViewMixin._replay_fidelity_error(
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
            number = history_selection.finite_value(value)
            if number is not None and field in params:
                params[field] = number
        return params

    def _history_replay_gui_state(self, spec, strategy_name):
        """把历史任务快照收敛成展示与快照保留都能消费的单次回测状态。"""
        history_state = getattr(self, "_latest_history_state", None) or {}
        state = snapshot_detail.copy_snapshot_gui_state(history_state)
        metadata = dict(getattr(spec, "metadata", {}) or {})
        index = getattr(spec.external_path, "index", None)
        if index is not None and len(index):
            # 保留时分：日内滚动的相邻两段起止日相同，只到天的话两条快照
            # 的行情签名会一模一样，页面就会说"输入完全相同"。
            state["wind_start"] = formatting.format_detail_index(
                index[0], include_time=True)
            state["wind_end"] = formatting.format_detail_index(
                index[-1], include_time=True)
        contract_code = str(metadata.get("contract_code") or "").strip()
        if contract_code:
            state["wind_code"] = contract_code
        # 每段实际用的期权是按该段首价整体缩放过的（_rescale_option_to_real_s0），
        # 而 params 原样留着用户填的参考价——于是两段行权价实际差了 50%，
        # 快照里的期权签名却一模一样，对比页恒报"期权：相同"。
        state["params"] = HistoryViewMixin._replay_scaled_params(
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
            window_days = history_selection.safe_int(
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
            HistoryViewMixin._history_replay_label(spec)
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
        model = history_selection.multi_chart_model(
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
        HistoryViewMixin._refresh_history_period_chips(self)
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
                    font=(_UI_FONT_FAMILY, 10, "bold"),
                    highlightbackground=PALETTE["primary"],
                    highlightthickness=1,
                )
            else:
                chip.configure(
                    bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
                    font=(_UI_FONT_FAMILY, 10),
                    highlightbackground=PALETTE["border_soft"],
                    highlightthickness=1,
                )

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
            baseline = history_selection.ranking_baseline(group)
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
                        formatting.history_baseline_row_label(
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

                values = HistoryViewMixin._pad_history_row(
                    (
                        rank_str, strategy_label,
                        HistoryViewMixin._format_objective_value(
                            row.get("incremental_pnl_vs_c2c"), 4),
                        HistoryViewMixin._format_objective_value(
                            row.get("incremental_sharpe_vs_c2c"), 4),
                        HistoryViewMixin._format_objective_value(
                            row.get("incremental_tc_vs_c2c"), 4),
                        HistoryViewMixin._format_drawdown_value(
                            row.get("max_drawdown")),
                        HistoryViewMixin._format_history_status(
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
            code_text = history_selection.contract_codes_text(
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
        formatted_context = "  ·  ".join(extra) if extra else ""
        if period_context is not None:
            period_context.set(f"📅 取样口径：{formatted_context}" if formatted_context else "")
            self._history_period_detail_prefix = ""
        else:
            self._history_period_detail_prefix = formatted_context
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
        force_day_close_hedge = history_selection.safe_bool(
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
