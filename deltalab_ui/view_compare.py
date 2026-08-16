# _*_ coding: utf-8 _*_
"""结果池表格与结果对比页。

两块内容合成一个 Mixin，因为它们共享同一份选中态：上半部分是结果池的
Treeview（勾选、右键菜单、列宽自适应、改名/删除），下半部分是被勾中的那几条
快照的指标表与累计曲线图。拆成两个 Mixin 的话，``_saved_comparison_selection``
和 ``_saved_backtests`` 会横跨两个文件，反而更难读。

``_show_history_recommendation`` 在原文件里正夹在这两段中间，但它属于策略优选
页，没有跟着搬——所以这里的方法在 gui_app.py 里原本不是一整段连续的。

**它对宿主类的要求**：

* 结果池数据与选中态：``_saved_backtests``、``_saved_comparison_selection``、
  ``_saved_pool_tree`` 及其字体/宽度缓存
* 对比页控件：``_comparison_tree``、``_comparison_frame``、
  ``_comparison_chart_figure`` / ``_comparison_chart_canvas``、
  ``_saved_comparison_content`` 以及一批 ``_comparison_*`` 的 Tk 变量
* 宿主方法：``_hide_placeholder``、``_set_status``、``_strategy_style``、
  ``_format_comparison_value``、``_saved_comparison_payload``、
  ``_saved_comparison_warnings``、``_rename_saved_backtest``、
  ``_delete_saved_backtest``、``_saved_snapshot_origin_label``、
  ``_apply_selected_snapshot_to_form``、``_load_saved_snapshot_detail``、
  ``_show_saved_snapshot_params``、``_export_saved_comparison``

结果池表的列定义等十三张常量表随方法一起搬进本类。它们仍以
``BacktestApp._SAVED_POOL_COLUMNS`` 的写法被类外代码和测试访问——Mixin 在 MRO
上，类属性照常查得到，不需要额外的别名。
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import tkinter.font as tkfont

import numpy as np

import backtest_pool_store
import history_selection

from deltalab_ui import snapshot_detail, widgets
from deltalab_ui.constants import (
    MAX_COMPARISON_CHART_CURVES,
    STRATEGY_CHART_DASHES,
    STRATEGY_CHART_SHADES,
)
from deltalab_ui.theme import PALETTE, _MONO_FONT_FAMILY, _UI_FONT_FAMILY


class ComparisonMixin:
    """结果池与结果对比页；混入 ``BacktestApp``，不单独实例化。"""

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
        ComparisonMixin._discard_comparison_frame(self)
        container = self._compare_container
        for widget in container.winfo_children():
            widget.destroy()

        header = ttk.Frame(container, style="Surface.TFrame")
        header.pack(fill="x", padx=8, pady=(8, 3))
        header.columnconfigure(0, weight=1)

        title_row = tk.Frame(header, bg=PALETTE["surface"])
        title_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            title_row, text="已保留结果对比",
            style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 12, "bold"),
        ).pack(side="left")

        # 初值与 _refresh_saved_pool_tree 里的刷新文案同一个模子，免得首帧
        # 先闪一下另一种格式；上限也从 store 取，不再写死。
        self._saved_pool_badge = tk.Label(
            title_row,
            text=f"0 显示 / 0 保留 (上限 {backtest_pool_store.MAX_RESULTS})",
            bg=PALETTE["surface_alt"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9, "bold"),
            padx=8, pady=1,
            highlightbackground=PALETTE["border_soft"], highlightthickness=1,
        )
        self._saved_pool_badge.pack(side="left", padx=(10, 0))

        ttk.Label(
            header,
            text="✦ 本地持久化保存 · 勾选行即时对比 · 拖动中间分割条自由调节高度",
            style="SurfaceMuted.TLabel",
            font=(_UI_FONT_FAMILY, 9),
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
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
            toolbar, text="☑ 全选", width=7, style="Ghost.TButton",
            command=self._select_all_saved_backtests,
        )
        self._saved_pool_show_all_btn.pack(side="left", padx=(4, 0))
        self._saved_pool_hide_all_btn = ttk.Button(
            toolbar, text="☐ 清选", width=7, style="Ghost.TButton",
            command=self._clear_saved_backtest_selection,
        )
        self._saved_pool_hide_all_btn.pack(side="left", padx=(2, 0))
        # 条数写进按钮文案，这样"要删几条"在点下去之前就已经在屏幕上；宽度
        # 按带条数时的最长文案定死，免得每次改选都抖一下版面。
        self._saved_pool_delete_btn = ttk.Button(
            toolbar, text="🗑 删除", width=7, style="Danger.TButton",
            command=self._prompt_delete_saved_backtest,
        )
        self._saved_pool_delete_btn.pack(side="left", padx=(14, 0))
        self._saved_pool_clear_btn = ttk.Button(
            toolbar, text="🧹 清空", width=7, style="Danger.TButton",
            command=self._clear_saved_backtest_pool,
        )
        self._saved_pool_clear_btn.pack(side="left", padx=(2, 0))
        # 导出收在最右，与两个红按钮隔开：它是这一排唯一不改变任何状态的
        # 动作，挨着删除放会让"点错一格"的代价过大。
        self._saved_pool_export_btn = ttk.Button(
            toolbar, text="📊 导出", width=7,
            command=self._export_saved_comparison,
        )
        self._saved_pool_export_btn.pack(side="left", padx=(14, 0))

        pool_columns = ComparisonMixin._SAVED_POOL_COLUMNS
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
        # 隔行底色，与指标表、优选排名表同一条规则。少了这一行，插入时打的
        # "even" tag 就只是个没配过样式的名字，一点效果都没有。
        tree.tag_configure("even", background=PALETTE["surface_alt"])
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
        self._saved_pool_menu = ComparisonMixin._build_saved_pool_menu(self)

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
        for index, snapshot in enumerate(self._saved_backtests.values()):
            is_selected = snapshot.result_id in self._saved_comparison_selection
            tree.insert(
                "", "end", iid=snapshot.result_id,
                image=self._cb_sf_checked if is_selected else self._cb_sf_unchecked,
                values=ComparisonMixin._pad_tree_cells(
                    (
                        snapshot.name,
                        self._saved_snapshot_origin_label(snapshot),
                        snapshot.option_label,
                        snapshot.strategy_label,
                        snapshot.parameter_summary, snapshot.source_label,
                        snapshot.saved_at.strftime("%m-%d %H:%M:%S"),
                    ),
                    [anchor for _key, _text, _width, anchor
                     in ComparisonMixin._SAVED_POOL_COLUMNS],
                ),
                tags=("even",) if index % 2 == 0 else (),
            )
        children = tree.get_children()
        survivors = [item for item in picked if item in children]
        if not survivors and focus in children:
            survivors = [focus]
        if not survivors and children:
            neighbour = ComparisonMixin._neighbour_pool_row(
                previous, set(picked) or {focus}, children)
            survivors = [neighbour] if neighbour else []
        if survivors:
            tree.selection_set(survivors)
            # 焦点优先留在原处：打勾不该把焦点拽到选区的第一行去。
            tree.focus(focus if focus in children else survivors[0])
        pooled = len(self._saved_backtests)
        shown = len(self._saved_comparison_selection)
        badge = getattr(self, "_saved_pool_badge", None)
        if badge is not None:
            badge.configure(
                text=f"{shown} 显示 / {pooled} 保留 (上限 {backtest_pool_store.MAX_RESULTS})",
                bg=PALETTE["primary_light"] if shown > 0 else PALETTE["surface_alt"],
                fg=PALETTE["primary"] if shown > 0 else PALETTE["text_muted"],
            )
        ComparisonMixin._autosize_saved_pool_columns(self)
        ComparisonMixin._sync_saved_pool_buttons(self)

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
            cell_font, head_font = ComparisonMixin._saved_pool_fonts(self)
            rows = tree.get_children()
            widths, floors = {}, {}
            for key, text, base, _anchor in ComparisonMixin._SAVED_POOL_COLUMNS:
                # 表头两侧的留白比单元格宽：排序标记之外还有 Treeview 自己的
                # 表头内边距。
                need = head_font.measure(text) + 26
                for iid in rows:
                    need = max(need, cell_font.measure(tree.set(iid, key)) + 16)
                ceiling = ComparisonMixin._SAVED_POOL_COLUMN_MAX.get(key, base)
                floor = max(44, base - 30)
                # 下限用 minwidth 而不是基准宽：内容短的列（产生方式、策略、
                # 保留时间）占着基准宽不放，省下的空间本该归内容长的那几列。
                widths[key] = max(floor, min(need, ceiling))
                keep = ComparisonMixin._SAVED_POOL_COLUMN_KEEP.get(key)
                floors[key] = (
                    max(floor, min(widths[key], keep)) if keep else floor)
            widths = ComparisonMixin._fit_pool_columns_to_width(
                widths, floors,
                tree.winfo_width() - ComparisonMixin._SAVED_POOL_CHECK_WIDTH
                - ComparisonMixin._SAVED_POOL_TABLE_CHROME)
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
        ComparisonMixin._autosize_saved_pool_columns(self)

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
        picked = len(ComparisonMixin._picked_saved_backtest_ids(self))
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
                    text=f"🗑 删除 ({picked})" if picked > 1 else "🗑 删除")
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
        if not ComparisonMixin._saved_pool_actions_allowed(self):
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
        picked = ComparisonMixin._picked_saved_backtest_ids(self)
        multiple = len(picked) > 1
        menu.entryconfigure(
            0,
            label=("取消显示" if result_id in self._saved_comparison_selection
                   else "显示"))
        for index in ComparisonMixin._POOL_MENU_SINGLE_INDEXES:
            menu.entryconfigure(
                index, state=("disabled" if multiple else "normal"))
        # 没有重放配方就置灰：本功能上线前保留的快照只有汇总层，点了只能
        # 弹一句"没有配方"，不如直接让它点不动。
        snapshot = self._saved_backtests.get(result_id)
        if not multiple:
            menu.entryconfigure(
                ComparisonMixin._POOL_MENU_LOAD_DETAIL_INDEX,
                state=("normal" if getattr(snapshot, "replay", None)
                       else "disabled"))
        # 末项文案跟着选中条数走，这样"这一下要删几条"在菜单里就已经写着。
        menu.entryconfigure(
            ComparisonMixin._POOL_MENU_DELETE_INDEX,
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
        if not ComparisonMixin._saved_pool_actions_allowed(self):
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
        if not ComparisonMixin._saved_pool_actions_allowed(self):
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
        ComparisonMixin._persist_pool_view(self)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _select_all_saved_backtests(self):
        if not ComparisonMixin._saved_pool_actions_allowed(self):
            return
        # 原地改而不是重新绑定属性：「取消全选」走的是 .clear()，两个入口对
        # 同一个集合一个换对象一个改内容，拿着旧引用的地方就会看到两种结果。
        self._saved_comparison_selection.clear()
        self._saved_comparison_selection.update(self._saved_backtests)
        ComparisonMixin._persist_pool_view(self)
        self._refresh_saved_pool_tree()
        self._refresh_saved_comparison_view()

    def _clear_saved_backtest_selection(self):
        if not ComparisonMixin._saved_pool_actions_allowed(self):
            return
        self._saved_comparison_selection.clear()
        ComparisonMixin._persist_pool_view(self)
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
        if not ComparisonMixin._saved_pool_actions_allowed(self):
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
        if not ComparisonMixin._saved_pool_actions_allowed(self):
            return
        picked = ComparisonMixin._picked_saved_backtest_ids(self)
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
                f"{ComparisonMixin._saved_pool_name_digest(snapshots)}\n\n"
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
        digest = ComparisonMixin._saved_pool_name_digest(deleted)
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
        if not ComparisonMixin._saved_pool_actions_allowed(self):
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
        self._comparison_status_pill = None
        self._comparison_expandable_container = None
        self._comparison_toggle_btn = None
        self._comparison_caveat_frame = None

    def _render_saved_comparison_empty(self, title, detail):
        """把内容区换成居中的空状态说明，并释放上一张图表。

        图标由标题反推，不做成参数：四个调用点各自的标题就是这三种状态，
        多一个参数只会让调用处多写一份跟标题重复的信息。
        """
        content = self._saved_comparison_content
        ComparisonMixin._discard_comparison_frame(self)
        for widget in content.winfo_children():
            widget.destroy()
        placeholder = tk.Frame(content, bg=PALETTE["surface"])
        placeholder.place(relx=0.5, rely=0.45, anchor="center")

        if "没有显示" in str(title):
            icon = "🔍"
        elif "无法" in str(title):
            icon = "⚠"
        else:
            icon = "🆚"

        icon_lbl = ttk.Label(
            placeholder, text=icon, style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 42),
        )
        icon_lbl.pack(pady=(0, 10))

        title_lbl = ttk.Label(
            placeholder, text=title, style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 16, "bold"),
        )
        title_lbl.pack(pady=(0, 6))

        desc_lbl = ttk.Label(
            placeholder, text=detail, style="SurfaceMuted.TLabel",
            font=(_UI_FONT_FAMILY, 10), justify="center",
            # 这里也会显示异常原文，长度不可控，必须折行。
            wraplength=600,
        )
        desc_lbl.pack(pady=(0, 0))

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
                        f"已保留的 {pooled} 条结果都在上方列表，勾选「显示」框或按「全选」即可放上来对比。")
                elif load_error:
                    self._render_saved_comparison_empty(
                        "没有可显示的结果", load_error)
                else:
                    self._render_saved_comparison_empty(
                        "结果对比",
                        "回测完成后点击『＋ 保留当前结果到对比』，即可在此跨策略与参数勾选对比。")
                return
            content = self._saved_comparison_content
            summary, daily_curves = self._saved_comparison_payload(snapshots)
        except Exception as exc:
            self._render_saved_comparison_empty("无法生成对比", str(exc))
            return
        ComparisonMixin._ensure_comparison_frame(self, content)
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

    # 说明卡与图表、指标表共用同一块高度，而 pack 是先足额兑现卡片的请求高
    # 度、剩下的才轮到下面那半（见 _ensure_comparison_frame 的 pack 顺序）。
    # 所以卡片必须自己封顶：字段行数等于五组属性摊平后的**全部**差异字段，
    # 本身没有上限——香草期权五组全变就是 17 行、卡片实测 390px，换成 Wind
    # 行情还要再多四行，而下半部分（图表 225px + 指标表）请求 449px，挤下去
    # 先没的是图表。指标表早给自己封过顶（_populate_comparison_view 里的
    # ``min(8, len(rows))``），这里补上同一道闸。
    # 6 行的依据：同时变两三组属性时字段数多在四到六行之间，原地能看完；再
    # 多的属于批量扫参，本来就得逐项对，交给详情窗比挤在卡片里强。
    _COMPARISON_FIELD_ROW_LIMIT = 6
    # 列数等于当前显示的结果条数，而结果池最多 20 条。20 列的网格请求宽度实
    # 测 2682px、容器只有约 1240px——超出的宽度会顺着几何传播把整个窗口撑
    # 开，不是被裁掉就算了。
    _COMPARISON_FIELD_COLUMN_LIMIT = 6

    def _build_comparison_variable_card(self, parent):
        """对比说明卡：智能折叠/展开结构，保证多要素时不挤压图表。

        - 顶部栏（折叠态）：高度仅约 34px，展示状态徽章、核心结论、受控项及展开按钮；
        - 可展开区：展示对齐紧凑的差异矩阵表格与看数限制警示条；
        - 智能折叠策略：差异项 <= 2 时自动展开，差异项 >= 3（多变量）时默认折叠。
        """
        # 说明卡外层：白底微卡片，左侧色条指示对比状态
        card = tk.Frame(
            parent, bg=PALETTE["surface"],
            highlightbackground=PALETTE["border_soft"], highlightthickness=1)
        card.pack(fill="x", padx=8, pady=(4, 4))

        self._comparison_variable_accent = tk.Frame(
            card, bg=PALETTE["border_soft"], width=4)
        self._comparison_variable_accent.pack(side="left", fill="y")

        body = tk.Frame(card, bg=PALETTE["surface"], padx=14, pady=8)
        body.pack(side="left", fill="both", expand=True)
        body.columnconfigure(0, weight=1)

        # ── 顶部栏：左侧 Badge Pill + 变量标题；右侧折叠按钮 ──
        top_bar = tk.Frame(body, bg=PALETTE["surface"], cursor="hand2")
        top_bar.grid(row=0, column=0, sticky="ew")

        self._comparison_status_pill = tk.Label(
            top_bar, text="ℹ 等待选择", bg=PALETTE["surface_alt"],
            fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 9, "bold"),
            padx=8, pady=2,
            highlightbackground=PALETTE["border_soft"], highlightthickness=1,
        )
        self._comparison_status_pill.pack(side="left", padx=(0, 10))

        self._comparison_variable_var = tk.StringVar(value="")
        self._comparison_variable_label = tk.Label(
            top_bar, textvariable=self._comparison_variable_var,
            bg=PALETTE["surface"], fg=PALETTE["text"],
            font=(_UI_FONT_FAMILY, 11, "bold"), anchor="w", justify="left",
        )
        self._comparison_variable_label.pack(side="left", fill="x", expand=True)
        widgets.track_wraplength(self._comparison_variable_label)

        self._comparison_toggle_btn = tk.Label(
            top_bar, text="▼ 展开明细",
            bg=PALETTE["surface_alt"], fg=PALETTE["primary"],
            font=(_UI_FONT_FAMILY, 9, "bold"),
            padx=10, pady=2, cursor="hand2",
            highlightbackground=PALETTE["border_soft"], highlightthickness=1,
        )
        self._comparison_toggle_btn.pack(side="right", padx=(8, 0))

        def _on_hover_toggle(event):
            try:
                event.widget.configure(
                    bg=PALETTE["primary_light"], highlightbackground=PALETTE["primary"])
            except tk.TclError:
                pass

        def _on_leave_toggle(event):
            try:
                event.widget.configure(
                    bg=PALETTE["surface_alt"], highlightbackground=PALETTE["border_soft"])
            except tk.TclError:
                pass

        top_bar.bind("<Button-1>", lambda _e: self._toggle_comparison_card())
        self._comparison_toggle_btn.bind("<Button-1>", lambda _e: self._toggle_comparison_card())
        self._comparison_toggle_btn.bind("<Enter>", _on_hover_toggle)
        self._comparison_toggle_btn.bind("<Leave>", _on_leave_toggle)

        # ── 可展开明细容器 ──
        expandable = tk.Frame(body, bg=PALETTE["surface"])
        expandable.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        expandable.columnconfigure(0, weight=1)
        self._comparison_expandable_container = expandable

        # 逐字段的取值网格
        self._comparison_field_grid = tk.Frame(
            expandable, bg=PALETTE["surface"])
        self._comparison_field_grid.grid(row=0, column=0, sticky="w", pady=(0, 4))

        # 受控变量（其余一致项）
        self._comparison_same_var = tk.StringVar(value="")
        self._comparison_same_label = tk.Label(
            expandable, textvariable=self._comparison_same_var,
            bg=PALETTE["surface"], fg=PALETTE["text_muted"],
            font=(_UI_FONT_FAMILY, 9), anchor="w", justify="left",
        )
        self._comparison_same_label.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        widgets.track_wraplength(self._comparison_same_label)

        # 看数限制/警告条：挂在 body 上而不是折叠区里。它说的是「这几条数
        # 字不能直接横着比」（交易日数不同、买卖方向混在一起），而这两种情
        # 况几乎必然带来三项以上差异——正好是折叠区默认收起的那一档，藏进
        # 去等于最该提醒的时候不提醒。
        # 放出来不会把膨胀问题带回来：saved_comparison_warnings 的产出就是
        # 写死的那两条，实测一条 25px、两条 34px 封顶；真正会涨的是字段网
        # 格，那道闸由 _COMPARISON_FIELD_ROW_LIMIT / _COLUMN_LIMIT 单独把着。
        self._comparison_caveat_frame = tk.Frame(
            body, bg=PALETTE["warning_light"],
            highlightbackground=PALETTE["warning"], highlightthickness=1,
            padx=10, pady=4,
        )
        self._comparison_caveat_var = tk.StringVar(value="")
        self._comparison_caveat_label = tk.Label(
            self._comparison_caveat_frame, textvariable=self._comparison_caveat_var,
            bg=PALETTE["warning_light"], fg=PALETTE["warning"],
            font=(_UI_FONT_FAMILY, 9), anchor="w", justify="left",
        )
        self._comparison_caveat_label.pack(side="left", fill="x", expand=True)
        widgets.track_wraplength(self._comparison_caveat_label)
        # row 2 = 折叠区（row 1）之下，与 top_bar 同属 body，收起明细带不走它。
        self._comparison_caveat_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._comparison_caveat_frame.grid_remove()

        self._comparison_card_expanded = False
        self._comparison_card_user_toggled = False

    def _toggle_comparison_card(self):
        """用户点击切换说明卡的折叠/展开状态。"""
        self._comparison_card_user_toggled = True
        self._comparison_card_expanded = not getattr(self, "_comparison_card_expanded", False)
        self._sync_comparison_card_expansion()

    def _sync_comparison_card_expansion(self):
        """根据当前展开状态同步 UI。"""
        container = getattr(self, "_comparison_expandable_container", None)
        toggle_btn = getattr(self, "_comparison_toggle_btn", None)
        if container is None or toggle_btn is None:
            return
        # 折叠区里只剩字段网格与「其余一致」，告警条已挂到 body 上，它的显
        # 隐由 _refresh_comparison_variable_card 直接管，不进这道判断。
        fields = getattr(self, "_comparison_field_rows", [])
        if not fields:
            toggle_btn.pack_forget()
            container.grid_remove()
            return
        toggle_btn.pack(side="right", padx=(8, 0))
        expanded = getattr(self, "_comparison_card_expanded", False)
        if expanded:
            toggle_btn.configure(text="▲ 收起明细")
            container.grid()
        else:
            num = len(fields)
            toggle_btn.configure(text=f"▼ 展开明细 ({num}项)" if num > 0 else "▼ 展开明细")
            container.grid_remove()

    def _fill_comparison_field_grid(self, fields):
        """一行一个差异字段：字段名 + 各结果的取值，取值按 #序号 排开。

        序号与指标表首列一致，所以三条以上也认得出谁是谁——此前这些值拼成
        "A vs B vs C" 塞在标题里，既分不清对应关系，又能把标题撑到七八十字。

        行数与列数都按 ``_COMPARISON_FIELD_ROW_LIMIT`` /
        ``_COMPARISON_FIELD_COLUMN_LIMIT`` 封顶，超出的收进详情窗——铺满会
        把卡片撑到几百像素，而那几百像素是从图表身上拿的。
        """
        grid = getattr(self, "_comparison_field_grid", None)
        if grid is None:
            return
        try:
            for widget in list(grid.winfo_children()):
                widget.destroy()
        except tk.TclError:
            return
        # 详情窗要的是全量，不是网格上还剩下的那几行。
        self._comparison_field_rows = list(fields)
        row_limit = ComparisonMixin._COMPARISON_FIELD_ROW_LIMIT
        column_limit = ComparisonMixin._COMPARISON_FIELD_COLUMN_LIMIT
        for row, (label, values) in enumerate(fields[:row_limit]):
            tk.Label(
                grid, text=label, bg=PALETTE["surface"],
                fg=PALETTE["text_muted"], font=(_UI_FONT_FAMILY, 9),
                anchor="w", width=12,
            ).grid(row=row, column=0, sticky="w", pady=(0, 2))
            if values is None:
                # 价格序列摘要之类：只说不同，没有可读的值。
                tk.Label(
                    grid, text="（不同）", bg=PALETTE["surface"],
                    fg=PALETTE["text"], font=(_UI_FONT_FAMILY, 10),
                    anchor="w",
                ).grid(row=row, column=1, sticky="w", pady=(0, 2))
                continue
            for column, value in enumerate(values[:column_limit]):
                cell = tk.Frame(
                    grid, bg=PALETTE["surface_alt"],
                    highlightbackground=PALETTE["border_soft"],
                    highlightthickness=1, padx=6, pady=1,
                )
                cell.grid(row=row, column=column + 1, sticky="w",
                          padx=(0, 8), pady=(0, 2))
                tk.Label(
                    cell, text=f"#{column + 1}", bg=PALETTE["surface_alt"],
                    fg=PALETTE["primary"], font=(_UI_FONT_FAMILY, 8, "bold"),
                ).pack(side="left", padx=(0, 3))
                tk.Label(
                    cell, text=str(value), bg=PALETTE["surface_alt"],
                    fg=PALETTE["text"],
                    font=(_UI_FONT_FAMILY, 9, "bold"),
                ).pack(side="left")
            if len(values) > column_limit:
                # 说的是"这一行还有几条没摆出来"，不是又一个字段。
                tk.Label(
                    grid, text=f"…另 {len(values) - column_limit} 条",
                    bg=PALETTE["surface"], fg=PALETTE["text_muted"],
                    font=(_UI_FONT_FAMILY, 9), anchor="w",
                ).grid(row=row, column=column_limit + 1, sticky="w",
                       pady=(0, 2))
        # 截断必须有去处，而**列**超限同样是截断。只挂"行超限"这一个入口
        # 的话，最常见的那种批量扫参就没救了：20 条结果只差一个参数时行数
        # 根本不超限，被截掉的是 14 条取值，而那 14 条在界面上再也点不出来。
        hidden_rows = max(0, len(fields) - row_limit)
        hidden_columns = max(
            (len(values) - column_limit
             for _label, values in fields if values is not None),
            default=0)
        self._comparison_field_more = None
        if hidden_rows or hidden_columns > 0:
            if hidden_rows and hidden_columns > 0:
                text = (f"＋ 还有 {hidden_rows} 项差异、"
                        f"每项另有 {hidden_columns} 条取值，点此逐项查看")
            elif hidden_rows:
                text = f"＋ 还有 {hidden_rows} 项差异，点此逐项查看"
            else:
                text = f"＋ 另有 {hidden_columns} 条结果的取值未列出，点此逐项查看"
            more = tk.Label(
                grid, text=text,
                bg=PALETTE["surface"], fg=PALETTE["primary"],
                font=(_UI_FONT_FAMILY, 9), anchor="w", cursor="hand2")
            # 摆在最后一行字段的下面。写死 row_limit 的话，只有列超限时上面
            # 会空出好几行——那时铺出来的字段数根本没到 row_limit。
            more.grid(row=min(len(fields), row_limit), column=0,
                      columnspan=column_limit + 2, sticky="w", pady=(4, 0))
            more.bind(
                "<Button-1>",
                lambda _event: ComparisonMixin._show_comparison_field_detail(
                    self))
            more.bind("<Enter>", lambda _e: more.configure(fg=PALETTE["primary_hov"], font=(_UI_FONT_FAMILY, 9, "underline")))
            more.bind("<Leave>", lambda _e: more.configure(fg=PALETTE["primary"], font=(_UI_FONT_FAMILY, 9)))
            self._comparison_field_more = more

    def _show_comparison_field_detail(self):
        """把全部差异字段摆进参数详情窗。

        卡片上只留得下前几行，而"到底差在哪"有时就得逐项看完。复用结果池
        那扇参数详情窗：正文是只读 ``Text``，长了自带滚动，取值还能选中复
        制——把差异贴给别人是这个窗口本来就有的第二个用途。
        """
        rows = list(getattr(self, "_comparison_field_rows", ()) or ())
        if not rows:
            return None
        return self._open_params_window(
            heading="本次对比的差异字段",
            subtitle=f"共 {len(rows)} 项；#序号与说明卡、指标表首列一致",
            sections=[("全部差异字段", [
                (label,
                  "（各条不同，没有可读的取值）" if values is None
                  else "   ".join(
                      f"#{index + 1} {value}"
                      for index, value in enumerate(values)))
                for label, values in rows
            ])],
        )

    def _refresh_comparison_variable_card(self, snapshots):
        """把变量结论与看数提醒写进说明卡。"""
        variable_var = getattr(self, "_comparison_variable_var", None)
        accent = getattr(self, "_comparison_variable_accent", None)
        if variable_var is None or accent is None:
            return
        same_var = self._comparison_same_var
        caveat_var = self._comparison_caveat_var
        status_pill = getattr(self, "_comparison_status_pill", None)
        caveat_frame = getattr(self, "_comparison_caveat_frame", None)
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
            if status_pill is not None:
                status_pill.configure(
                    text="ℹ 等待选择", bg=PALETTE["surface_alt"],
                    fg=PALETTE["text_muted"])
            if caveat_frame is not None:
                caveat_frame.grid_remove()
            ComparisonMixin._fill_comparison_field_grid(self, [])
            self._sync_comparison_card_expansion()
            return

        summary = snapshot_detail.comparison_variable_summary(snapshots)
        state = summary["state"]
        variable_var.set(summary["headline"])
        same_var.set(summary["rest"])
        fields = summary["fields"]
        ComparisonMixin._fill_comparison_field_grid(self, fields)
        accent.configure(bg={
            "single": PALETTE["success"],
            "identical": PALETTE["primary"],
            "multiple": PALETTE["warning"],
        }[state])
        if status_pill is not None:
            if state == "single":
                status_pill.configure(
                    text="✓ 单变量对比", bg=PALETTE["success_light"], fg=PALETTE["success"])
            elif state == "identical":
                status_pill.configure(
                    text="≡ 配置完全相同", bg=PALETTE["primary_light"], fg=PALETTE["primary"])
            else:
                status_pill.configure(
                    text="⚠ 多变量对比", bg=PALETTE["warning_light"], fg=PALETTE["warning"])
        caveats = self._saved_comparison_warnings(snapshots)
        if caveats:
            caveat_var.set("\n".join(f"· {caveat}" for caveat in caveats))
            if caveat_frame is not None:
                caveat_frame.grid()
        else:
            caveat_var.set("")
            if caveat_frame is not None:
                caveat_frame.grid_remove()

        # 智能折叠策略：若用户未手动点击过折叠/展开，单变量 (<=2项差异) 自动展开，多变量 (>2项) 默认折叠以解放图表空间
        if not getattr(self, "_comparison_card_user_toggled", False):
            self._comparison_card_expanded = (len(fields) <= 2)
        self._sync_comparison_card_expansion()

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
        return ComparisonMixin._COMPARISON_COLUMN_ANCHORS.get(key, "e")

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
        return ComparisonMixin._pad_tree_cells(values, [
            ComparisonMixin._comparison_column_anchor(key)
            for key, _text, _width in
            ComparisonMixin._COMPARISON_RANKING_COLUMNS
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
            value = history_selection.finite_value(
                item.get(ComparisonMixin._COMPARISON_SORT_KEYS.get(
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
                column in ComparisonMixin._COMPARISON_SORT_DESCENDING_FIRST)
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
        for key, text, _width in ComparisonMixin._COMPARISON_RANKING_COLUMNS:
            if key in ComparisonMixin._COMPARISON_UNSORTABLE_COLUMNS:
                marker = ""
            elif key == active:
                marker = " ▼" if descending else " ▲"
            else:
                marker = " ⇅"
            try:
                tree.heading(
                    key, text=f"{text}{marker}",
                    anchor=ComparisonMixin._comparison_column_anchor(key))
            except tk.TclError:
                return

    def _init_comparison_splitter_ratio(self, splitter, ratio=0.55):
        """按比例摆放曲线图与指标表的初始分割线。"""
        state = {"owned": False, "height": None, "target": None, "budget": 6, "pending": False}

        def _apply():
            state["pending"] = False
            if state["owned"] or state["budget"] <= 0 or state["target"] is None:
                return
            target = state["target"]
            try:
                if abs(splitter.sash_coord(0)[1] - target) <= 2:
                    return
                state["budget"] -= 1
                splitter.sash_place(0, 0, target)
            except tk.TclError:
                pass

        def _place(_event=None):
            if state["owned"] or state["budget"] <= 0:
                return
            height = splitter.winfo_height()
            if height <= 1:
                return
            previous = state["height"]
            resized = previous is None or abs(height - previous) > (previous * 0.15)
            if resized:
                state["height"] = height
                state["target"] = max(180, int(height * ratio))
            elif state["target"] is None:
                return
            if not state["pending"]:
                state["pending"] = True
                splitter.after_idle(_apply)

        def _hand_over(_event=None):
            state["owned"] = True

        splitter.bind("<Configure>", _place, add="+")
        splitter.bind("<ButtonPress-1>", _hand_over, add="+")
        splitter.after_idle(_place)

    def _build_current_comparison_view(
            self, parent, *, show_curve_controls=True,
            ranking_title="已选结果指标（点列头换排序）"):
        """建结果区骨架：图表画布、排名表、详情格。

        这里一个数据字段都不读。勾选一次就把整块 destroy 重建，代价是每次
        新造一个 matplotlib Figure 和整张表——批量验证十条候选，就是主线程
        上十次这样的重建。骨架与数据分开之后，刷新只剩下改值。
        """
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        splitter = tk.PanedWindow(
            parent, orient="vertical", bg=PALETTE["border_soft"],
            sashwidth=6, sashrelief="flat", borderwidth=0,
            opaqueresize=False)
        splitter.grid(row=0, column=0, sticky="nsew", pady=(2, 0))

        chart_box = ttk.LabelFrame(
            splitter, text=" 累计净损益（按真实交易日汇总，已扣成本） ", padding=10)
        controls = ttk.Frame(chart_box, style="Surface.TFrame")
        controls.pack(fill="x", pady=(0, 2))
        self._comparison_curve_hint_var = tk.StringVar(
            value="显示曲线:" if show_curve_controls else "")
        ttk.Label(
            controls, textvariable=self._comparison_curve_hint_var,
            style="SurfaceMuted.TLabel",
            font=(_UI_FONT_FAMILY, 9),
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
        splitter.add(chart_box, minsize=180, stretch="always")

        result_box = ttk.LabelFrame(
            splitter, text=f" {ranking_title} ", padding=10)
        tree_frame = ttk.Frame(result_box, style="Surface.TFrame")
        tree_frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(
            tree_frame,
            columns=[key for key, _text, _width in
                     ComparisonMixin._COMPARISON_RANKING_COLUMNS],
            show="headings", height=3, selectmode="browse",
        )
        for key, text, width in ComparisonMixin._COMPARISON_RANKING_COLUMNS:
            anchor = ComparisonMixin._comparison_column_anchor(key)
            if key in ComparisonMixin._COMPARISON_UNSORTABLE_COLUMNS:
                tree.heading(key, text=text, anchor=anchor)
            else:
                tree.heading(
                    key, text=text, anchor=anchor,
                    command=lambda k=key: self._sort_comparison_by(k))
            tree.column(
                key, width=width, minwidth=max(40, width - 30),
                anchor=anchor, stretch=True,
            )
        ranking_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ranking_scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        ranking_scroll.pack(side="right", fill="y")
        tree.tag_configure("even", background=PALETTE["surface_alt"])
        self._comparison_tree = tree
        self._comparison_rows = {}
        tree.bind("<<TreeviewSelect>>", self._update_comparison_selection)
        splitter.add(result_box, minsize=120, stretch="always")
        ComparisonMixin._init_comparison_splitter_ratio(self, splitter)

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
            ComparisonMixin._comparison_curve_styles(self, style_keys))

        tree = self._comparison_tree
        rows = {}
        # 重填后把用户原来盯着的那一行选回来。行号会随排序和勾选变化，只有
        # 快照 ID 是稳的；不记住它的话，换个列排序选中行就跳到第一行去了。
        previous_id = ""
        if tree.selection():
            previous_id = str(getattr(self, "_comparison_rows", {}).get(
                tree.selection()[0], {}).get("meta_result_id", "") or "")
        ordered = ComparisonMixin._comparison_sorted_rows(
            summary, getattr(self, "_comparison_sort_column", None),
            getattr(self, "_comparison_sort_descending", False))
        # 曲线闸门要在排序之后定：图上画的是"按当前排序靠前的那几条"。
        self._rebuild_comparison_curve_toggles(
            ComparisonMixin._comparison_chartable_names(ordered))
        # 清表会让选中行消失并发出 TreeviewSelect；重填期间抑制回调，填完再
        # 主动刷新一次，避免拿着半张表去画图。
        self._comparison_populating = True
        try:
            tree.delete(*tree.get_children())
            for row_no, item in enumerate(ordered):
                iid = f"strategy_{row_no}"
                values = ComparisonMixin._pad_comparison_row((
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
            ComparisonMixin._refresh_comparison_sort_markers(self)
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
        ComparisonMixin._refresh_comparison_variable_card(self, [
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
            colors[name] = ComparisonMixin._shift_hex(base, index)
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
        # 图例挂在 figure 上而不是 ax 上，ax.clear() 带不走它，重画前得自己摘
        # 干净，否则每刷一次就叠一层。
        for legend in list(ax.figure.legends):
            legend.remove()
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

        ax.axhline(0.0, color=PALETTE["text_muted"], linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_facecolor(PALETTE["surface"])
        ax.set_xlabel("交易日", fontsize=8.5, color=PALETTE["text"])
        ax.set_ylabel("累计净损益", fontsize=8.5, color=PALETTE["text"])
        ax.tick_params(labelsize=8, colors=PALETTE["text_muted"])
        ax.grid(True, linestyle="--", linewidth=0.6, color=PALETTE["border_soft"], alpha=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(PALETTE["border"])
        if plotted:
            # 曲线常贴着上下边缘，图内放哪儿都会压住数据（"best" 也只是挑一
            # 处压得少的），所以照策略优选的做法横排到绘图区下方。用
            # figure 级的 outside 位（constrained_layout 才认这个位），让它
            # 在轴外单占一条：轴自动让出高度，与 X 轴标签互不相干，用户拖分
            # 割线改变图高时留白也不会跟着放大——换成 ax.legend 配
            # bbox_to_anchor 的话，锚点是轴高的百分比，高图上就会豁开一道。
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(
                handles, labels, loc="outside lower center",
                frameon=False, fontsize=8, ncol=min(4, plotted))
            ax.margins(x=0.02, y=0.12)
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
