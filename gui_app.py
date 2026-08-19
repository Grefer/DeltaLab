# _*_ coding: utf-8 _*_
"""
DeltaLab - 期权对冲回测 GUI 应用

基于 tkinter 构建，支持选择不同期权类型、回测方式（模拟/历史数据），
并以图表和表格形式展示回测结果。
"""

import sys
import os

# 仓库根加入 sys.path。pricing / deltalab_ui / history_* 都按顶层模块引用，
# 从别的工作目录启动、以及 PyInstaller 打包态都靠这一行兜底。它必须早于下面
# 任何一个项目内 import——原先压在 matplotlib 配置之后还能用，是因为那时第一
# 个项目内 import 出现得更晚；现在 deltalab_ui 也走这条路径，位置就不能再往
# 后放了。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
import numpy as np

# theme 必须早于 matplotlib.pyplot 被 import：后端选择（含 headless 回退）在
# 它的模块顶部完成，晚于 pyplot 的首次 import 就不生效了。这里已不再直接用
# pyplot，但 theme 的 import 仍然要留着——本文件是 GUI 的入口，后端必须在
# 任何画图模块之前定下来。
from deltalab_ui.theme import (
    FORM_ENTRY_CHARS, FORM_HINT_GAP, FORM_ROW_PADY, FORM_SECTION_GAP,
    FORM_SECTION_PAD,
    PALETTE,
    _MONO_FONT_FAMILY, _SYSTEM, _UI_FONT_FAMILY,
    _form_grid, _form_hint, _form_input, _form_label, _form_separator,
    _resource_path,
)

# 部分是 re-export：测试与外部脚本一直按 ``gui_app.XXX`` 取值，拆分不该动它们
# 的取值路径。
#
# 但**执行链路用到的那几个名字不再从这里 re-export**（recommend_by_* /
# summarize_strategy_result / result_daily_frame）：实现已随 worker 搬进
# deltalab_ui/runner.py 与 snapshot_store.py，它们从各自模块的全局取值。留在
# 这里只会变成陷阱——往 gui_app 上打补丁不报错、静默失效。删掉之后误打的地方
# 会直接 AttributeError。
#
# compare_strategies 也一并去掉了：多路径统计早已改由 ``bt.run_multi`` 产出，
# 它在界面层一个取用点都没有；此前三条挂在它上面的负向守卫（"不得重新运行回
# 测" / "不应再运行当前期限路径对比"）因此从来不会触发，同样已删除。
from pricing import HedgeBacktest, FixedTimeStrategy
from pricing.hedge_analysis import LOOKBACK_DAYS
import history_selection
from history_selection import (
    # 前两个本文件不用，是 re-export：测试写的是
    # ``from gui_app import DEFAULT_BAND_CANDIDATE_SIGMAS, MAX_BAND_CANDIDATES``。
    DEFAULT_BAND_CANDIDATE_SIGMAS,
    DEFAULT_FIXED_TIMES,
    HISTORY_PERIOD_DEFS,
    MAX_BAND_CANDIDATES,
    MAX_HISTORY_CHART_CANDIDATES,
)
import history_store
import history_bar_cache
import backtest_pool_store


from deltalab_ui.constants import (
    BASELINE_STRATEGY_STYLE_KEY,
    HISTORY_CHART_METRIC_DISPLAY, HISTORY_CHART_METRIC_FROM_DISPLAY,
    HISTORY_OBJECTIVE_DISPLAY, HISTORY_OBJECTIVE_FROM_DISPLAY,
    MAX_COMPARISON_CHART_CURVES,
    OPTION_CLASSES,
    SIGMA_SOURCE_DISPLAY, SIGMA_SOURCE_FROM_DISPLAY,
    SNAPSHOT_ORIGIN_DISPLAY, SNAPSHOT_ORIGIN_HISTORY_REPLAY,
    SNAPSHOT_ORIGIN_MANUAL,
    STRATEGY_CHART_COLORS, STRATEGY_CHART_DASHES, STRATEGY_CHART_MARKERS,
    STRATEGY_CHART_SHADES,
    STRATEGY_DISPLAY, STRATEGY_FROM_DISPLAY,
    SUBTYPE_DISPLAY, SUBTYPE_FROM_DISPLAY,
    WIND_AUTO_BAR_SIZE, WIND_BAR_SIZE_OPTIONS,
    _OBJECTIVE_COLUMN_KEYS, _OBJECTIVE_RANKING_COLUMNS,
    _WIND_BAND_BAR_LABEL, _WIND_BAR_MINUTES, _WIND_DATE_BUFFER_DAYS,
    _WIND_FIXED_TIME_BAR_LABELS,
    _build_snowball, _snowball_ko_observ,
)
from deltalab_ui.snapshot_store import SavedBacktestResult
from deltalab_ui.structure_docs import STRUCTURE_DOCS
from deltalab_ui import (
    formatting,
    history_setup,
    panel_form,
    runner,
    widgets,
    snapshot_detail,
    snapshot_store,
    view_compare,
    view_history,
    view_results,
    wind_resolve,
)


# ============================================================
#  主窗口
# ============================================================


class BacktestApp(panel_form.FormPanelMixin,
                  history_setup.HistorySetupMixin,
                  runner.RunnerMixin,
                  view_history.HistoryViewMixin,
                  snapshot_store.SnapshotStoreMixin,
                  view_compare.ComparisonMixin,
                  view_results.ResultsMixin,
                  tk.Tk):

    # `tkinter.Misc.__getattr__` 会把未定义属性转发给 self.tk，对尚未渲染
    # 结果页的实例用 getattr(..., None) 兜底会递归。策略优选结果页按需创建
    # 的状态统一在类级给出 None 默认值。
    _history_pairs_cache = None

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
        self._saved_backtest_sequence = 0
        self._latest_backtest = None
        self._latest_backtest_state = None
        self._latest_retained_result_id = None
        self._latest_history_state = None
        self._latest_history_source_label = None
        # 策略配色登记表：结果对比与策略优选共用，使同一策略跨页同色。
        # 仍是会话级——跨会话的配色稳定性从没承诺过，结果池持久化后同一批
        # 结果重开可能换色，这一点写在 GUI_USAGE 的 6.7 里。
        self._strategy_style_registry = None
        # 载入本机已保存的结果池。放在 _build_ui 之前：计数行与占位符在构建
        # 时就要拿到真实条数，否则要么显示 0、要么得再刷一次。
        self._saved_pool_load_error = ""
        self._load_saved_pool()
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
        brand_font   = (_UI_FONT_FAMILY, 12, "bold")
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
        # 左上角品牌标 (与右侧页签行同处一条水平带, 不再单独占一条横幅)
        style.configure("Brand.TLabel",
                        background=PALETTE["bg"],
                        foreground=PALETTE["text"],
                        font=brand_font)
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
        # Entry 与 Combobox 的纵向内边距取同一个值, 两者才等高;
        # 横向留 6 让文字不贴边 (Combobox 右侧还要放下箭头, 少 2)。
        style.configure("TEntry",
                        fieldbackground=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["border"],
                        darkcolor=PALETTE["border"],
                        padding=(6, 4))
        style.map("TEntry",
                  bordercolor=[("focus", PALETTE["primary"])],
                  lightcolor=[("focus", PALETTE["primary"])])

        style.configure("TCombobox",
                        fieldbackground=PALETTE["surface"],
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        arrowcolor=PALETTE["text_muted"],
                        padding=(6, 4))
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
                  background=[("active", PALETTE["surface"]),
                              ("disabled", PALETTE["surface"])],
                  foreground=[("disabled", PALETTE["text_light"]),
                              ("active", PALETTE["primary"]),
                              ("selected", PALETTE["text"])])

        # 专门定制 clam 下的原生 Checkbutton 样式（采用无锯齿 Pixel Art 彻底消除 macOS Retina 拉伸模糊）
        try:
            bg_checked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAZElEQVR4nGP8/PXnfwYKABMlmhkYGBhYYAyjvE8kaTw3iY86LqC/AbfmiJBvALpmrAZgU4QsrpbyhrAL0A3BpRmrATBFME34NON0AbohuDTjNABZEz7NeA0gRjNBA4gBjAOeGwEXWyXhhSE6UgAAAABJRU5ErkJggg=="
            bg_unchecked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAASElEQVR4nGP8/PXnfwYKABMlmhkYGBhYYIynL9+TpFFaXBDVAAYGBgZ1RXGiNN+8/xLOptgLowaMGsDAgJYSkVMYsYBxwHMjAKdYEIYsHHJgAAAAAElFTkSuQmCC"
            self._cb_bg_checked = tk.PhotoImage(data=bg_checked_crisp)
            self._cb_bg_unchecked = tk.PhotoImage(data=bg_unchecked_crisp)
            style.element_create("Bg.Indicator", "image", self._cb_bg_unchecked,
                ("selected", self._cb_bg_checked), border=0, sticky="w")

            sf_checked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAY0lEQVR4nGP8////fwYKABMlmhkYGBhYYAy1lDckabw1R4Q6LqC/ATCnk2UAumasBmBThCyOHthYXYBuCC7NWA2AKYJpwqcZpwvQDcGXRnAGIkwToQSGNxaISZ0UJyTGAc+NAJGvJhltVByMAAAAAElFTkSuQmCC"
            sf_unchecked_crisp = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAQklEQVR4nGP8////fwYKABMlmhkYGBhYYIyb91+SpFFdURzVAGRBQgDZMoq9MGrAqAEMDGgpkdTkzMDAwMA44LkRACG7EL4AADlaAAAAAElFTkSuQmCC"
            self._cb_sf_checked = tk.PhotoImage(data=sf_checked_crisp)
            self._cb_sf_unchecked = tk.PhotoImage(data=sf_unchecked_crisp)
            style.element_create("Surface.Indicator", "image", self._cb_sf_unchecked,
                ("selected", self._cb_sf_checked), border=0, sticky="w")

            style.layout("TCheckbutton", [
                ('Checkbutton.padding', {'sticky': 'nswe', 'children': [
                    ('Bg.Indicator', {'side': 'left', 'sticky': ''}),
                    ('Checkbutton.focus', {'side': 'left', 'sticky': '', 'children': [
                        ('Checkbutton.label', {'sticky': 'nswe'})
                    ]})
                ]})
            ])
            style.layout("Surface.TCheckbutton", [
                ('Checkbutton.padding', {'sticky': 'nswe', 'children': [
                    ('Surface.Indicator', {'side': 'left', 'sticky': ''}),
                    ('Checkbutton.focus', {'side': 'left', 'sticky': '', 'children': [
                        ('Checkbutton.label', {'sticky': 'nswe'})
                    ]})
                ]})
            ])
        except Exception:
            pass

        def _config_checkbutton(style_name, bg_color):
            style.configure(style_name,
                            background=bg_color,
                            foreground=PALETTE["text"],
                            font=base_font,
                            focuscolor=bg_color)
            style.map(style_name,
                      background=[("active", bg_color),
                                  ("disabled", bg_color)],
                      foreground=[("disabled", PALETTE["text_light"]),
                                  ("active", PALETTE["primary"]),
                                  ("selected", PALETTE["text"])])

        _config_checkbutton("TCheckbutton", PALETTE["bg"])
        _config_checkbutton("Surface.TCheckbutton", PALETTE["surface"])

        # ---- 按钮 ----
        style.configure("TButton",
                        font=btn_font,
                        background=PALETTE["surface"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["surface"],
                        darkcolor=PALETTE["border"],
                        focusthickness=0,
                        padding=(10, 5),
                        relief="flat")
        style.map("TButton",
                  background=[("active", PALETTE["primary_light"]),
                              ("pressed", PALETTE["selected"]),
                              ("disabled", PALETTE["surface_alt"])],
                  foreground=[("active", PALETTE["primary"]),
                              ("disabled", PALETTE["text_light"])],
                  bordercolor=[("active", PALETTE["primary"]),
                               ("disabled", PALETTE["border_soft"])],
                  lightcolor=[("active", PALETTE["primary_light"]),
                              ("disabled", PALETTE["border_soft"])],
                  darkcolor=[("active", PALETTE["primary"]),
                             ("disabled", PALETTE["border_soft"])])

        # 行内小按钮 (如 CSV 的「浏览…」): 纵向内边距与输入框对齐, 同高
        style.configure("Field.TButton", padding=(10, 4))

        # 主色按钮 (运行)
        style.configure("Run.TButton",
                        font=run_font,
                        foreground="white",
                        background=PALETTE["primary"],
                        bordercolor=PALETTE["primary"],
                        padding=(12, 5),
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
                        padding=(10, 5),
                        relief="flat")
        style.map("Accent.TButton",
                  background=[("active", PALETTE["accent_hov"]),
                              ("pressed", PALETTE["accent_hov"]),
                              ("disabled", "#9CA3AF")],
                  foreground=[("disabled", "#E5E7EB")])

        # 幽灵按钮：统一白底与浅描边，文字为次要色，悬停浮出主色底
        style.configure("Ghost.TButton",
                        font=btn_font,
                        background=PALETTE["surface"],
                        foreground=PALETTE["text_muted"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["surface"],
                        darkcolor=PALETTE["border"],
                        focusthickness=0,
                        padding=(10, 5),
                        relief="flat")
        style.map("Ghost.TButton",
                  background=[("active", PALETTE["primary_light"]),
                              ("pressed", PALETTE["selected"]),
                              ("disabled", PALETTE["surface_alt"])],
                  foreground=[("active", PALETTE["primary"]),
                              ("disabled", PALETTE["text_light"])],
                  bordercolor=[("active", PALETTE["primary"]),
                               ("disabled", PALETTE["border_soft"])],
                  lightcolor=[("active", PALETTE["primary_light"]),
                              ("disabled", PALETTE["border_soft"])],
                  darkcolor=[("active", PALETTE["primary"]),
                             ("disabled", PALETTE["border_soft"])])

        # 危险按钮：删数据且不可撤销的那一类。做成红字描边而不是实心红——它
        # 不是主行动，实心红摆在一排常规按钮里会比「运行回测」还抢眼；红字
        # 加悬停浅红底，足够在点下去之前把性质说清楚。
        style.configure("Danger.TButton",
                        font=btn_font,
                        background=PALETTE["surface"],
                        foreground=PALETTE["danger"],
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["surface"],
                        darkcolor=PALETTE["surface"],
                        focusthickness=0,
                        padding=(10, 5),
                        relief="flat")
        style.map("Danger.TButton",
                  background=[("active", PALETTE["danger_light"]),
                              ("pressed", PALETTE["danger_light"]),
                              ("disabled", PALETTE["surface"])],
                  foreground=[("disabled", PALETTE["text_light"])],
                  bordercolor=[("active", PALETTE["danger"]),
                               ("disabled", PALETTE["border_soft"])],
                  lightcolor=[("active", PALETTE["danger_light"]),
                              ("pressed", PALETTE["danger_light"])],
                  darkcolor=[("active", PALETTE["danger_light"]),
                             ("pressed", PALETTE["danger_light"])])

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
    def _build_brand_row(self, parent):
        """左侧参数栏正上方的品牌标 (图标 + 名称 + 副标题).

        高度按右侧页签行配平 (图标 32px + 上下各 8px 留白), 于是窗口顶部
        这条水平带左右两侧都有内容, 宽屏下不会再空出一整条横幅.
        """
        row = ttk.Frame(parent)
        # padx=0: 图标左边缘与下方各 LabelFrame 的左边缘对齐 (Canvas 也在 x=0)
        row.pack(side="top", fill="x", padx=0, pady=8)

        # Tk 的 PhotoImage 只能按整数倍抽样缩小 (最近邻, 圆角会发毛), 所以
        # 直接读按尺寸渲染好的 32px 图. 缺图/解码失败时只是没有图标, 不影响文字.
        icon_path = _resource_path("assets", "deltalab_32.png")
        self._brand_photo = None
        if os.path.exists(icon_path):
            try:
                self._brand_photo = tk.PhotoImage(file=icon_path)
            except tk.TclError:
                pass
            else:
                ttk.Label(row, image=self._brand_photo,
                          style="Brand.TLabel").pack(side="left")

        text_col = ttk.Frame(row)
        text_col.pack(side="left", padx=(8, 0))
        ttk.Label(text_col, text="DeltaLab", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(text_col, text="期权动态对冲回测系统",
                  style="Muted.TLabel").pack(anchor="w")

    def _build_ui(self):
        # 品牌标不再单独占一条通栏横幅：窗口标题栏已经写着同一句话，通栏
        # 只是在宽屏上留出一条几乎全空的白带。改放到左侧参数栏正上方
        # （见下面的 _build_brand_row），与右侧页签行同处一条水平带。

        # 底部状态栏
        # 必须排在 body 之前 pack：pack 按调用顺序分配空间，body 带 expand=True
        # 会先吃满自己的请求高度；一旦右侧图表 + 表格的请求高度超过窗口可用高度
        # （最大化到较矮的屏幕、或策略优选页有图有表时），后 pack 的状态栏就会被
        # 挤出窗口底部只露半行。先占位则状态栏恒为 24px，压缩的是可伸缩的 body。
        status_bar = ttk.Frame(self, style="Surface.TFrame")
        status_bar.pack(fill="x", side="bottom")
        ttk.Separator(status_bar, orient="horizontal").pack(fill="x")
        self._status_var = tk.StringVar(
            value="就绪  |  选择期权类型、设置参数后点击『运行回测』")
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Status.TLabel", anchor="w").pack(fill="x", padx=10)

        # 主体：左侧参数 + 右侧结果
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        # ─── 左侧面板 (整体包一层 Canvas + Scrollbar, 解决低分辨率/高 DPI 下底部按钮被裁) ───
        left_outer = ttk.Frame(body)
        body.add(left_outer, weight=1)

        # 品牌标：必须先 pack, 才能拿到 left_outer 顶部这条; 之后的滚动条与
        # Canvas 分掉剩下的空腔。它不进 Canvas —— 进去会跟着参数一起滚走。
        self._build_brand_row(left_outer)

        # 外层 Canvas 横向充满, Scrollbar 靠右
        # width 仅作为 PanedWindow 初始 sash 位置的参考, 不阻止后续缩放;
        # 取表单三列 (标签 + 输入 + 说明) 的自然宽度, 初次打开时右侧说明
        # 文字不会被 sash 截断
        self._left_canvas = tk.Canvas(
            left_outer, highlightthickness=0, bd=0,
            bg=PALETTE["surface"], width=480,
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

        # ─── 鼠标滚轮 ───
        # 旧实现是 <Enter> 时 bind_all、<Leave> 时 unbind_all. 问题在于指针只要
        # 从 Canvas 挪到面板内任意子控件 (LabelFrame / Label / Entry / Combobox…),
        # Canvas 就会收到 <Leave> 把滚轮解绑 —— 而参数区表面几乎全被子控件盖住,
        # 于是表现为「鼠标停在参数上滚轮没反应, 只有压着右侧滚动条才滚得动」
        # (滚动条能滚是 Tk 自带的 Scrollbar 类绑定, 与这里无关).
        #
        # 改成常驻一条全局绑定, 每次事件按**指针位置**判断是否落在左侧面板内:
        # 落在外面就原样放行, 不会劫持右侧图表/表格的滚轮.
        # 用指针位置而不是 event.widget, 是因为 Windows 上 Tk 把 <MouseWheel>
        # 投递给**焦点控件**而非指针下的控件, 只看 event.widget 会判错.
        _self_scrolling = {"Text", "Listbox", "Treeview"}

        def _wheel_over_left_panel(event):
            try:
                widget = self.winfo_containing(event.x_root, event.y_root)
                while widget is not None:
                    if widget is self._left_canvas:
                        return True
                    # 指针下方是自带滚动的控件, 让它自己处理, 不抢事件
                    if widget.winfo_class() in _self_scrolling:
                        return False
                    widget = getattr(widget, "master", None)
            except (KeyError, tk.TclError):
                # 指针不在本程序窗口内 / 控件刚被销毁
                return False
            return False

        def _on_left_mousewheel(event):
            if not _wheel_over_left_panel(event):
                return None
            # Windows: delta 为 ±120 的倍数; macOS: 每格 ±1 (触控板还更小).
            step = -event.delta if _SYSTEM == "Darwin" else -event.delta / 120
            units = int(step)
            if units == 0 and step:
                units = 1 if step > 0 else -1
            self._left_canvas.yview_scroll(units, "units")
            return "break"

        def _on_left_mousewheel_linux(event):
            if not _wheel_over_left_panel(event):
                return None
            self._left_canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
            return "break"

        # 供测试/其它代码直接调用的入口 (指向本平台实际生效的那个处理器)
        self._left_wheel_handler = (
            _on_left_mousewheel_linux if _SYSTEM == "Linux" else _on_left_mousewheel)

        if _SYSTEM == "Linux":
            self.bind_all("<Button-4>", _on_left_mousewheel_linux, add="+")
            self.bind_all("<Button-5>", _on_left_mousewheel_linux, add="+")
        else:
            self.bind_all("<MouseWheel>", _on_left_mousewheel, add="+")

        # ttk 的 Combobox/Spinbox 自带滚轮类绑定, 悬停其上滚动会直接改选中值 ——
        # 参数区里全是这种控件, 既会误改「大类」「子类型」等参数, 又会让人以为
        # 面板滚不动. 覆盖成空处理: 不返回 "break", 事件继续走到上面的全局绑定,
        # 于是滚轮在整个界面里统一是「滚面板」而不是「改值」.
        for _wheel_cls in ("TCombobox", "TSpinbox"):
            self.bind_class(_wheel_cls, "<MouseWheel>", lambda _event: None)
            if _SYSTEM == "Linux":
                self.bind_class(_wheel_cls, "<Button-4>", lambda _event: None)
                self.bind_class(_wheel_cls, "<Button-5>", lambda _event: None)

        # 之后所有原本放在 left 里的控件, 父容器改为 left_inner
        left = self._left_inner

        # 1) 期权大类
        sec1 = ttk.LabelFrame(left, text=" 期权类型 ", padding=FORM_SECTION_PAD)
        sec1.pack(fill="x", pady=(0, FORM_SECTION_GAP))
        _form_grid(sec1)

        _form_label(sec1, "大类:", 0)
        self._class_var = tk.StringVar()
        class_cb = ttk.Combobox(sec1, textvariable=self._class_var,
                                width=FORM_ENTRY_CHARS,
                                values=list(OPTION_CLASSES.keys()), state="readonly")
        _form_input(class_cb, 0)
        class_cb.current(0)
        class_cb.bind("<<ComboboxSelected>>", self._on_option_class_change)

        _form_label(sec1, "子类型:", 1)
        self._subtype_var = tk.StringVar()
        self._subtype_cb = ttk.Combobox(sec1, textvariable=self._subtype_var,
                                        width=FORM_ENTRY_CHARS, state="readonly")
        _form_input(self._subtype_cb, 1)
        self._subtype_cb.bind(
            "<<ComboboxSelected>>", lambda _event: self._schedule_band_reference_sync())

        # 2) 期权参数
        # 说明: 外层左侧已经有整体 Canvas+Scrollbar, 这里不再嵌套独立滚动容器,
        # 让参数区按内容自然撑开高度, 整体滚动由外层统一处理.
        sec2 = ttk.LabelFrame(left, text=" 期权参数 ", padding=FORM_SECTION_PAD)
        sec2.pack(fill="x", pady=(0, FORM_SECTION_GAP))

        self._param_frame = ttk.Frame(sec2, style="Surface.TFrame")
        self._param_frame.pack(fill="x", expand=True)
        _form_grid(self._param_frame)

        # 3) 回测设置
        sec3 = ttk.LabelFrame(left, text=" 回测设置 ", padding=FORM_SECTION_PAD)
        sec3.pack(fill="x", pady=(0, FORM_SECTION_GAP))
        _form_grid(sec3)

        _form_label(sec3, "数据来源:", 0)
        self._source_var = tk.StringVar(value="simulate")
        src_frame = ttk.Frame(sec3, style="Surface.TFrame")
        # 单选组比输入列宽, 让它跨到说明列, 免得把输入列撑变形
        _form_input(src_frame, 0, columnspan=2, sticky="w")
        ttk.Radiobutton(src_frame, text="模拟", variable=self._source_var,
                        value="simulate", command=self._toggle_source).pack(side="left")
        ttk.Radiobutton(src_frame, text="CSV", variable=self._source_var,
                        value="csv", command=self._toggle_source).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(src_frame, text="Wind", variable=self._source_var,
                        value="wind", command=self._toggle_source).pack(side="left", padx=(12, 0))

        # 模拟 / CSV / Wind 三个子面板共用同一套列宽, 切换数据来源时
        # 输入框不会左右跳动。
        # 模拟参数
        self._sim_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sim_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        _form_grid(self._sim_frame)
        _form_label(self._sim_frame, "种子:", 0)
        self._seed_var = tk.StringVar(value="42")
        _form_input(ttk.Entry(self._sim_frame, textvariable=self._seed_var,
                              width=FORM_ENTRY_CHARS), 0)
        _form_label(self._sim_frame, "已实现波动率:", 1)
        self._real_vol_var = tk.StringVar(value="")
        _form_input(ttk.Entry(self._sim_frame, textvariable=self._real_vol_var,
                              width=FORM_ENTRY_CHARS), 1)
        _form_hint(self._sim_frame, 1, "空=同隐含")
        _form_label(self._sim_frame, "回测路径数 (MC):", 2)
        self._npaths_var = tk.StringVar(value="10")
        _form_input(ttk.Entry(self._sim_frame, textvariable=self._npaths_var,
                              width=FORM_ENTRY_CHARS), 2)
        _form_label(self._sim_frame, "每日模拟采样点数:", 3)
        self._spd_var = tk.StringVar(value="1")
        self._spd_combo = ttk.Combobox(
            self._sim_frame, textvariable=self._spd_var, width=FORM_ENTRY_CHARS,
            values=("1", "4", "48", "240"), state="readonly")
        _form_input(self._spd_combo, 3)
        _form_hint(self._sim_frame, 3, "每个交易日等分的采样点数")

        # CSV 参数
        self._csv_frame = ttk.Frame(sec3, style="Surface.TFrame")
        _form_grid(self._csv_frame)
        _form_label(self._csv_frame, "文件:", 0)
        self._csv_path_var = tk.StringVar()
        _form_input(ttk.Entry(self._csv_frame, textvariable=self._csv_path_var,
                              width=FORM_ENTRY_CHARS), 0)
        ttk.Button(self._csv_frame, text="浏览…", width=6,
                   style="Field.TButton", command=self._browse_csv).grid(
            row=0, column=2, sticky="w",
            padx=(FORM_HINT_GAP, 0), pady=FORM_ROW_PADY)
        _form_label(self._csv_frame, "价格列:", 1)
        self._csv_col_var = tk.StringVar(value="close")
        _form_input(ttk.Entry(self._csv_frame, textvariable=self._csv_col_var,
                              width=FORM_ENTRY_CHARS), 1)

        # Wind 参数
        self._wind_frame = ttk.Frame(sec3, style="Surface.TFrame")
        _form_grid(self._wind_frame)
        _form_label(self._wind_frame, "代码:", 0)
        self._wind_code_var = tk.StringVar(value="510050.SH")
        _form_input(ttk.Entry(self._wind_frame, textvariable=self._wind_code_var,
                              width=FORM_ENTRY_CHARS), 0)
        # 建仓日排在最上面：它才是读结果时关心的那一天。勾选倒推时这一格
        # 由「数据截止日」按期权期限沿交易日历往前算出来并回填（置灰只读），
        # 开关紧跟其后说明这格是谁算的；下一行才是作为主控的截止日。
        _form_label(self._wind_frame, "建仓日:", 1)
        _today = datetime.date.today()
        # 截止日默认落在最近一个「已收盘」交易日：当天盘中拉行情，最后一个
        # 交易日组只有半天 bar，分钟粒度的策略会被判成盘中残段而直接拒绝。
        # 这也与历史择优默认截至日（不含当天）保持同一口径。
        _wind_asof_default = BacktestApp._latest_trading_day(
            _today - datetime.timedelta(days=1)).isoformat()
        # 只是取消勾选前的占位：勾选倒推时首次刷新就会用倒推结果覆盖它。
        _wind_start_default = (_today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        self._wind_start_var = tk.StringVar(value=_wind_start_default)
        self._wind_start_entry = ttk.Entry(
            self._wind_frame, textvariable=self._wind_start_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._wind_start_entry, 1)
        self._wind_auto_start_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._wind_frame, text="按数据截止日倒推",
            variable=self._wind_auto_start_var,
            command=self._toggle_wind_auto_start,
            style="Surface.TCheckbutton",
        ).grid(row=1, column=2, sticky="w",
               padx=(FORM_HINT_GAP, 0), pady=FORM_ROW_PADY)

        _form_label(self._wind_frame, "数据截止日:", 2)
        self._wind_end_var = tk.StringVar(value=_wind_asof_default)
        self._wind_end_entry = ttk.Entry(
            self._wind_frame, textvariable=self._wind_end_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._wind_end_entry, 2)
        # 界面上不再单列"实际区间"，两个日期框就是本次真实取数区间，因此
        # 截止日填成休市日时也要落到倒推真正用的那个锚点（不晚于它的最近
        # 交易日）。逐次击键就落值会打断输入，只在提交时对齐。
        self._wind_end_entry.bind(
            "<FocusOut>", lambda _event: self._commit_wind_asof())
        self._wind_end_entry.bind(
            "<Return>", lambda _event: self._commit_wind_asof())

        # 行情采样粒度不再由用户选择：每种策略的正确粒度是唯一确定的
        # （收盘=日频、固定时刻=能覆盖目标时刻的最粗、价格波动=1 分钟），
        # 手动调粗只会静默让结论偏乐观。这里只展示本次将要使用的粒度。
        _form_label(self._wind_frame, "行情采样粒度:", 3)
        self._wind_frequency_hint_var = tk.StringVar()
        # 这一格是只读结论, 不是输入框, 因此贴着输入列起点直接铺开
        ttk.Label(
            self._wind_frame, textvariable=self._wind_frequency_hint_var,
            style="SurfaceMuted.TLabel",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=FORM_ROW_PADY)
        self._toggle_wind_auto_start()
        self._refresh_wind_frequency_hint()
        self._wind_code_var.trace_add(
            "write", lambda *_args: self._refresh_wind_frequency_hint())
        self._wind_end_var.trace_add(
            "write", lambda *_args: self._sync_wind_entry_date())

        # 轻分割线
        _form_separator(sec3, 2)

        # 对冲参数
        # 顺序：成本（成本率 + 滑点）→ 仓位（方向 / 数量 / 乘数）→ 策略。
        # 滑点与成本率是同一类按成交量计的执行成本，排在一起；也因此不能留在
        # 策略块之后——那里的专属参数随策略显隐，会带着滑点上下跳一百多像素。
        row_h = 3
        _form_label(sec3, "交易成本率(%):", row_h)
        self._tc_var = tk.StringVar(value="0.01")
        _form_input(ttk.Entry(sec3, textvariable=self._tc_var,
                              width=FORM_ENTRY_CHARS), row_h)

        row_h += 1
        _form_label(sec3, "滑点 (基点):", row_h)
        self._slip_var = tk.StringVar(value="0")
        _form_input(ttk.Entry(sec3, textvariable=self._slip_var,
                              width=FORM_ENTRY_CHARS), row_h)

        row_h += 1
        _form_label(sec3, "头寸方向:", row_h)
        self._pos_var = tk.StringVar(value="1")
        pos_frame = ttk.Frame(sec3, style="Surface.TFrame")
        _form_input(pos_frame, row_h, columnspan=2, sticky="w")
        ttk.Radiobutton(pos_frame, text="卖出 (Sell)", variable=self._pos_var,
                        value="1").pack(side="left")
        ttk.Radiobutton(pos_frame, text="买入 (Buy)", variable=self._pos_var,
                        value="-1").pack(side="left", padx=(12, 0))

        row_h += 1
        _form_label(sec3, "交易数量:", row_h)
        self._qty_var = tk.StringVar(value="100")
        _form_input(ttk.Entry(sec3, textvariable=self._qty_var,
                              width=FORM_ENTRY_CHARS), row_h)

        row_h += 1
        _form_label(sec3, "合约乘数:", row_h)
        self._mult_var = tk.StringVar(value="5")
        _form_input(ttk.Entry(sec3, textvariable=self._mult_var,
                              width=FORM_ENTRY_CHARS), row_h)
        _form_hint(sec3, row_h, "0=不取整")

        # 轻分割线：高级对冲参数（策略 / intraday / 滑点）
        row_h += 1
        _form_separator(sec3, row_h)

        row_h += 1
        _form_label(sec3, "对冲策略:", row_h)
        self._strategy_var = tk.StringVar(
            value=STRATEGY_DISPLAY["close_to_close"])
        self._strategy_combo = ttk.Combobox(
            sec3, textvariable=self._strategy_var, width=FORM_ENTRY_CHARS,
            values=tuple(STRATEGY_DISPLAY.values()), state="readonly",
        )
        _form_input(self._strategy_combo, row_h)
        self._strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_strategy())

        # 紧跟策略下拉：它只修饰所选策略的触发时点，不是第四种策略。原先排在
        # 固定时刻 / 带宽参数之后，会随策略切换在面板里上下跳。
        # 默认开启：日内触发策略若不在收盘补一次，隔夜会裸露一整晚的
        # Δ 敞口，而回测把这段风险记为零成本，结果偏乐观。
        row_h += 1
        self._force_day_close_hedge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sec3, text="每日收盘保底对冲（重复则跳过）",
            variable=self._force_day_close_hedge_var,
            command=self._refresh_history_base_summary,
            style="Surface.TCheckbutton",
        ).grid(row=row_h, column=1, columnspan=2, sticky="w", pady=FORM_ROW_PADY)

        # sigma 单位下的波动率来源参数；带宽数值统一使用下方输入框。
        row_h += 1
        self._sigma_band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._sigma_band_frame.grid(row=row_h, column=0, columnspan=3, sticky="ew")
        _form_grid(self._sigma_band_frame)
        self._k_var = tk.StringVar(value="1")  # 旧状态字段兼容，不再单独展示
        _form_label(self._sigma_band_frame, "波动率来源:", 0)
        self._sigma_src_var = tk.StringVar(value=SIGMA_SOURCE_DISPLAY["implied"])
        _form_input(ttk.Combobox(
            self._sigma_band_frame, textvariable=self._sigma_src_var,
            width=FORM_ENTRY_CHARS,
            values=tuple(SIGMA_SOURCE_DISPLAY.values()), state="readonly"), 0)
        _form_label(self._sigma_band_frame, "历史波动率回看天数:", 1)
        self._sigma_win_var = tk.StringVar(value="20")
        _form_input(ttk.Entry(self._sigma_band_frame,
                              textvariable=self._sigma_win_var,
                              width=FORM_ENTRY_CHARS), 1)

        row_h += 1
        self._fixed_time_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._fixed_time_frame.grid(row=row_h, column=0, columnspan=3, sticky="ew")
        _form_grid(self._fixed_time_frame)
        _form_label(self._fixed_time_frame, "固定时刻:", 0)
        self._fixed_times_var = tk.StringVar(value=DEFAULT_FIXED_TIMES)
        self._fixed_times_entry = ttk.Entry(
            self._fixed_time_frame, textvariable=self._fixed_times_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._fixed_times_entry, 0)
        _form_hint(self._fixed_time_frame, 0, "HH:MM，逗号分隔")

        # 三种带宽单位互为换算，逐行排布才能和上面的参数共用同一条对齐线。
        self._band_frame = ttk.Frame(sec3, style="Surface.TFrame")
        self._band_frame.grid(row=row_h, column=0, columnspan=3, sticky="ew")
        _form_grid(self._band_frame)
        _form_label(self._band_frame, "绝对间隔:", 0)
        self._band_abs_var = tk.StringVar(value="1")
        self._band_abs_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_abs_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._band_abs_entry, 0)
        _form_label(self._band_frame, "相对间隔:", 1)
        self._band_rel_var = tk.StringVar(value="0.01")
        self._band_rel_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_rel_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._band_rel_entry, 1)
        _form_hint(self._band_frame, 1, "0.01 = 1%")
        _form_label(self._band_frame, "日波动率倍数:", 2)
        self._band_sigma_var = tk.StringVar(value="0.779423")
        self._band_sigma_entry = ttk.Entry(
            self._band_frame, textvariable=self._band_sigma_var,
            width=FORM_ENTRY_CHARS)
        _form_input(self._band_sigma_entry, 2)
        _form_hint(self._band_frame, 2, "编辑任一项后自动换算")
        # 策略专属参数是本组最后一行：它随策略显隐，后面不能再排别的控件。

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
        self._retain_btn.pack(fill="x", ipady=2, pady=(FORM_SECTION_GAP - 2, 0))

        # 左侧不再有「结果对比」按钮：同名标签页是等效入口，切过去时
        # _on_notebook_tab_changed 会补建页面。已保留条数改挂到标签页标题上，
        # 从任何页面都看得见。
        self._struct_btn = ttk.Button(btn_frame, text="📊  绘制结构图",
                                      style="Accent.TButton",
                                      command=self._plot_structure)
        self._struct_btn.pack(fill="x", ipady=2, pady=(FORM_SECTION_GAP - 2, 0))

        struct_ctrl = ttk.Frame(btn_frame)
        struct_ctrl.pack(fill="x", pady=(FORM_SECTION_GAP - 2, 0))
        ttk.Label(struct_ctrl, text="扫描 ±%:", style="Muted.TLabel").pack(side="left")
        self._struct_range_var = tk.StringVar(value="30")
        ttk.Entry(struct_ctrl, textvariable=self._struct_range_var, width=6).pack(
            side="left", padx=(6, 16))
        ttk.Label(struct_ctrl, text="点数:", style="Muted.TLabel").pack(side="left")
        self._struct_npts_var = tk.StringVar(value="31")
        ttk.Entry(struct_ctrl, textvariable=self._struct_npts_var, width=6).pack(
            side="left", padx=(6, 0))

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
             "回测摘要", "运行回测后此处将展示详细的盈亏、Greeks (希腊字母) 和波动率统计"),
            ("chart",   " 📈 对冲图表 ",
             "对冲图表", "运行回测后此处将展示标的价格、Delta、Gamma、累计盈亏等图表"),
            ("table",   " 📃 每日明细 ",
             "每日明细",
             "运行回测后此处将只展示触发对冲的建仓、策略、收盘保底与终止记录"),
            ("struct",  " 🔬 结构分析 ",
             "结构分析", "点击左侧『绘制结构图』按钮以生成期权结构的 Greeks (希腊字母) 曲线"),
            ("vol",     " 📊 波动率分析 ",
             "波动率分析", "运行回测后此处将展示隐含波动率与已实现波动率的对比分析"),
            ("dist",    " 🎲 盈亏分布 ",
             "盈亏分布", "在模拟模式下运行多路径回测后，此处将展示蒙特卡洛盈亏分布"),
            ("compare", BacktestApp._COMPARE_TAB_TITLE,
             "结果对比",
             "回测完成后点击『＋ 保留当前结果到对比』，即可在此跨策略与参数勾选对比"),
            ("history", " 🎯 策略优选 ",
             "策略优选",
             "使用 CSV / Wind 真实历史行情搜索已勾选周期的领先对冲方案"),
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
                "summary": "📋", "compare": "🆚", "history": "🎯",
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
                                wraplength=620, justify="center")
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

        self._toggle_source()

    # ---- 跨页共用的纯函数与控件小工具 ----
    # 这几个原本就写成 @staticmethod 供类级调用（测试会传 SimpleNamespace
    # 假 self 进来），页面拆成 Mixin 之后必须落到模块里：Mixin 里的静态方法
    # 拿不到 self，够不着别的 Mixin 上的同类函数。别名保住 BacktestApp._x 的调法。
    _attach_tooltip = staticmethod(widgets.attach_tooltip)
    _copy_snapshot_gui_state = staticmethod(snapshot_detail.copy_snapshot_gui_state)
    _format_comparison_value = staticmethod(formatting.format_comparison_value)
    _history_baseline_row_label = staticmethod(formatting.history_baseline_row_label)
    _history_row_style_key = staticmethod(formatting.history_row_style_key)
    _strategy_style_key = staticmethod(formatting.strategy_style_key)
    _track_wraplength = staticmethod(widgets.track_wraplength)

    # ---- 策略优选纯逻辑的 GUI 侧入口 ----
    # 候选空间构造、任务入参校验、排名展示模型与图表模型都不依赖 tkinter，
    # 实现集中在 history_selection.py。这里按同名 staticmethod 暴露，使
    # 界面回调、后台 worker 与既有调用点仍以统一的 BacktestApp 名字访问。
    _normalize_history_lookbacks = staticmethod(
        history_selection.normalize_lookbacks)
    _validate_history_recommendation_source = staticmethod(
        history_selection.validate_source)
    _validate_fixed_time_history_granularity = staticmethod(
        history_selection.validate_fixed_time_granularity)
    _validate_history_recommendation_payload = staticmethod(
        history_selection.validate_payload)
    _parse_band_candidate_sigmas = staticmethod(
        history_selection.parse_band_candidate_sigmas)
    _band_cases_for_history = staticmethod(history_selection.band_cases)
    _strategy_cases_for_history = staticmethod(
        history_selection.strategy_cases)
    _comparison_finite = staticmethod(history_selection.finite_value)
    _comparison_safe_int = staticmethod(history_selection.safe_int)
    _comparison_safe_bool = staticmethod(history_selection.safe_bool)
    _comparison_baseline = staticmethod(history_selection.ranking_baseline)
    _history_row_improvement = staticmethod(history_selection.row_improvement)
    _history_row_uses_recognized_selection_metric = staticmethod(
        history_selection.row_uses_recognized_metric)
    _history_row_uses_strict_metric = staticmethod(
        history_selection.row_uses_strict_metric)
    _history_row_uses_window_equal_metric = staticmethod(
        history_selection.row_uses_window_equal_metric)
    _history_contract_codes_text = staticmethod(
        history_selection.contract_codes_text)
    _comparison_recommendation_rows = staticmethod(
        history_selection.recommendation_rows)
    _history_chart_pairs = staticmethod(history_selection.chart_pairs)
    _history_chart_array = staticmethod(history_selection.chart_array)
    _history_chart_band = staticmethod(history_selection.chart_band)
    _history_chart_model = staticmethod(history_selection.chart_model)
    _history_multi_chart_model = staticmethod(
        history_selection.multi_chart_model)

    # ---- 事件回调 ----
    def _on_notebook_tab_changed(self, _event=None):
        """按落地页刷新它依赖的会话状态。

        历史页要刷新将被冻结的公共环境与当前带宽摘要；结果对比页此前只由
        左侧按钮构建，从标签直接进来的用户会看到「先保留结果」占位符——哪
        怕按钮上已经写着结果数。
        """
        try:
            current = self._nb.select()
        except (AttributeError, tk.TclError):
            return
        if current == str(self._history_tab):
            BacktestApp._toggle_history_wind_controls(self)
            self._refresh_history_base_summary()
            self._refresh_history_current_band_label()
            return
        if current == str(getattr(self, "_compare_tab", None)):
            BacktestApp._ensure_saved_comparison_page(self)

    def _ensure_saved_comparison_page(self):
        """结果池非空却还没构建过页面时补建一次，不改变任何勾选状态。

        只在页面缺失时动手：已经构建过的页面由结果池自己的刷新链路维护，
        重建会丢掉用户当前的聚焦行。池子为空时保持占位符——那句提示正是
        此时该说的话。
        """
        tree = getattr(self, "_saved_pool_tree", None)
        try:
            if tree is not None and tree.winfo_exists():
                return
        except tk.TclError:
            # widget 已随旧渲染销毁，按未构建处理。
            pass
        if not getattr(self, "_saved_backtests", None):
            return
        self._show_saved_comparison_page(navigate=False)

    def destroy(self):
        """先撤掉挂起的带宽换算定时器，再销毁窗口。

        Tk 的 destroy 清 widget、也清本实例注册的 Tcl 命令，唯独不动 after
        队列。带宽输入的防抖定时器几乎总是挂着一个，窗口关掉后它仍排在事件
        队列里，指向一个已经被删掉的命令名。同一个进程里随后再开一个窗口
        时，那次事件循环就会执行到这条失效的 after 脚本 —— 报
        ``invalid command name``，并且不再返回。
        """
        pending = getattr(self, "_band_reference_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except (tk.TclError, ValueError):
                pass
            self._band_reference_after_id = None
        super().destroy()

    # ---- 任务状态 ----
    # 全局只允许一个任务在跑；执行链路本身在 deltalab_ui/runner.py。

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
                "_run_btn", "_retain_btn", "_history_btn", "_struct_btn"):
            button = getattr(self, attr, None)
            if button is not None:
                buttons.append(button)
        return buttons

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

    # ---- Wind 取数区间 / 交易日历 / 行情粒度解析的 GUI 侧入口 ----
    # 这一组同样是不碰 tkinter 的纯函数，实现集中在 deltalab_ui/wind_resolve.py。
    # 沿用上面 history_selection 那组的做法按同名 staticmethod 暴露：界面回调、
    # 后台 worker 与测试一直按 BacktestApp 名字访问，改成模块路径要动上百处，
    # 而这些函数原本挂在窗口类上本来就只是历史原因。
    _parse_wind_date = staticmethod(wind_resolve.parse_wind_date)
    _maturity_days_from_params = staticmethod(
        wind_resolve.maturity_days_from_params)
    _history_realized_warmup_days = staticmethod(
        wind_resolve.history_realized_warmup_days)
    _history_auto_wind_start = staticmethod(wind_resolve.history_auto_wind_start)
    _validate_sigma_input = staticmethod(wind_resolve.validate_sigma_input)
    _calendar_span_for_trading_days = staticmethod(
        wind_resolve.calendar_span_for_trading_days)
    _CALENDAR_FIX_HINT = wind_resolve._CALENDAR_FIX_HINT
    _trading_calendar_days = staticmethod(wind_resolve.trading_calendar_days)
    _latest_trading_day = staticmethod(wind_resolve.latest_trading_day)
    _entry_date_from_asof = staticmethod(wind_resolve.entry_date_from_asof)
    _local_trading_sessions = staticmethod(wind_resolve.local_trading_sessions)
    _recommended_fixed_time_bar_size = staticmethod(
        wind_resolve.recommended_fixed_time_bar_size)
    _recommended_band_bar_size = staticmethod(
        wind_resolve.recommended_band_bar_size)
    _resolve_wind_bar_size = staticmethod(wind_resolve.resolve_wind_bar_size)
    _resolve_single_wind_state = staticmethod(
        wind_resolve.resolve_single_wind_state)
    _resolve_history_wind_state = staticmethod(
        wind_resolve.resolve_history_wind_state)
    _gui_steps_per_day = staticmethod(wind_resolve.gui_steps_per_day)
    _normalize_position = staticmethod(wind_resolve.normalize_position)

    def _set_status(self, text):
        """更新底部状态栏文字 (线程安全: 调用方需保证在主线程或通过 after)。"""
        try:
            self._status_var.set(text)
        except Exception:
            pass

    def _strategy_style(self, style_key):
        """按会话内首次出现顺序固定策略配色；两页共用同一登记表。

        登记表随会话保留，因此先在策略优选页看到的候选，之后作为快照进入
        结果对比页时颜色不变。
        """
        registry = getattr(self, "_strategy_style_registry", None)
        if registry is None:
            # 基准先占位，保证 每日收盘 在两页永远是同一个颜色。
            registry = {
                BASELINE_STRATEGY_STYLE_KEY: (
                    STRATEGY_CHART_COLORS[0], STRATEGY_CHART_MARKERS[0]),
            }
            self._strategy_style_registry = registry
        key = str(style_key or BASELINE_STRATEGY_STYLE_KEY)
        if key not in registry:
            index = len(registry)
            registry[key] = (
                STRATEGY_CHART_COLORS[index % len(STRATEGY_CHART_COLORS)],
                STRATEGY_CHART_MARKERS[index % len(STRATEGY_CHART_MARKERS)],
            )
        return registry[key]

    # ---- 跨页共用的小格式化函数 ----
    # 实现在 deltalab_ui/formatting.py。它们的调用点散在结果页、结果池与策略
    # 优选三处，页面各自成 Mixin 之后就没有共同归属了——Mixin 里的
    # @staticmethod 拿不到 self，够不着别的 Mixin 上的同类函数。
    _snapshot_source_label = staticmethod(formatting.snapshot_source_label)
    _format_detail_index = staticmethod(formatting.format_detail_index)

    # ---- 快照签名摊平 / 明细分节 / 差异比对的 GUI 侧入口 ----
    # 实现集中在 deltalab_ui/snapshot_detail.py：结果池详情、结果对比、策略优选
    # 参数详情三处共用同一份摊平与排版逻辑。随方法一起搬走的还有它们独占的
    # 八张字段表（_COMPARISON_ASPECTS 等），那些表在闭包外没有引用点。
    _bar_size_detail_text = staticmethod(snapshot_detail.bar_size_detail_text)
    _comparison_aspect_diff = staticmethod(
        snapshot_detail.comparison_aspect_diff)
    _comparison_variable_summary = staticmethod(
        snapshot_detail.comparison_variable_summary)
    _differing_field_names = staticmethod(
        snapshot_detail.differing_field_names)
    _flatten_signature = staticmethod(snapshot_detail.flatten_signature)
    _format_detail_value = staticmethod(snapshot_detail.format_detail_value)
    _format_signature_value = staticmethod(
        snapshot_detail.format_signature_value)
    _freeze_snapshot_value = staticmethod(
        snapshot_detail.freeze_snapshot_value)
    _history_row_detail_sections = staticmethod(
        snapshot_detail.history_row_detail_sections)
    _history_row_form_state = staticmethod(
        snapshot_detail.history_row_form_state)
    _intraday_steps_detail_text = staticmethod(
        snapshot_detail.intraday_steps_detail_text)
    _option_param_labels = staticmethod(snapshot_detail.option_param_labels)
    _saved_comparison_warnings = staticmethod(
        snapshot_detail.saved_comparison_warnings)
    _saved_snapshot_position = staticmethod(
        snapshot_detail.saved_snapshot_position)
    _signature_cls_name = staticmethod(snapshot_detail.signature_cls_name)
    _signature_keys_from_state = staticmethod(
        snapshot_detail.signature_keys_from_state)
    _snapshot_detail_sections = staticmethod(
        snapshot_detail.snapshot_detail_sections)
    _snapshot_source = staticmethod(snapshot_detail.snapshot_source)
    _snapshot_strategy_signature = staticmethod(
        snapshot_detail.snapshot_strategy_signature)

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

    # 导出用的列与表头。排名表给的是格式化过的展示串，这里导原始数值，
    # 落到表格软件里才能继续算。
    _COMPARISON_EXPORT_COLUMNS = (
        ("strategy", "结果名称"),
        ("meta_description", "策略 / 参数 / 期权 / 行情来源"),
        ("total_net_pnl", "期末净损益"),
        ("total_tc", "总成本"),
        ("max_drawdown", "最大回撤"),
        ("rehedge_count", "再触发次数"),
        ("actual_trade_count", "实际成交次数"),
        ("turnover", "换手额"),
        ("n_trade_days", "交易日数"),
        ("meta_origin", "产生方式"),
        ("meta_origin_detail", "来源溯源"),
    )

    def _comparison_export_frames(self):
        """把当前对比整理成两张可导出的表：指标与曲线。

        导的是原始数值而不是表里那串格式化文本，落到表格软件里才能继续算。
        """
        import pandas as pd
        summary = getattr(self, "_comparison_summary", None)
        if summary is None or getattr(summary, "empty", True):
            raise ValueError("当前没有可导出的对比结果。")
        ranking = pd.DataFrame([
            {
                heading: row.to_dict().get(key)
                for key, heading in BacktestApp._COMPARISON_EXPORT_COLUMNS
            }
            for _index, row in summary.iterrows()
        ])
        # 逐条建 Series 再按索引横向拼，不能用 dict-of-arrays 直接建表：
        # 后者要求各列等长，而"各条交易日数不同"恰恰是本页显式支持的状态
        # （_saved_comparison_warnings 有专门提示、绘图侧对每条各自算
        # x = arange(1, len+1)），换区间做对比更是本页头号用途。等长要求会
        # 让整个导出抛 ValueError，被调用方转成"没有可导出的结果"，连同已
        # 经拼好的排名表一起丢掉——结果明明就在屏幕上。
        # 索引即交易日序号，短的那几条自动补 NaN，与图表的对齐方式一致。
        columns = [
            pd.Series(
                np.asarray(
                    daily.get("cumulative_net_pnl", []), dtype=float),
                name=name,
            ).rename_axis(None)
            for name, daily in getattr(
                self, "_comparison_daily_curves", {}).items()
        ]
        for column in columns:
            column.index = np.arange(1, len(column) + 1)
        curves = pd.concat(columns, axis=1) if columns else pd.DataFrame()
        curves.index.name = "交易日序号"
        return ranking, curves

    def _export_saved_comparison(self):
        """导出当前对比：排名一张表，累计净损益曲线另一张。"""
        try:
            ranking, curves = BacktestApp._comparison_export_frames(self)
        except Exception as exc:
            messagebox.showinfo("没有可导出的结果", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="导出结果对比", defaultextension=".csv",
            initialfile="结果对比.csv",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        base, extension = os.path.splitext(path)
        curves_path = f"{base}_曲线{extension or '.csv'}"
        try:
            ranking.to_csv(path, index=False, encoding="utf-8-sig")
            curves.to_csv(curves_path, encoding="utf-8-sig")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self._set_status(
            f"已导出 {len(ranking)} 条排名与曲线  |  "
            f"{os.path.basename(path)} / {os.path.basename(curves_path)}")
        messagebox.showinfo(
            "导出成功",
            f"排名已保存至:\n{path}\n\n累计净损益曲线已保存至:\n{curves_path}")

    def _apply_snapshot_strategy_to_form(self, snapshot):
        """把快照的对冲策略输入写回左侧表单，不动其它任何一组参数。"""
        form_state = dict(getattr(snapshot, "form_state", None) or {})
        strategy_name = str(form_state.get("strategy_name", "") or "").strip()
        if strategy_name not in STRATEGY_DISPLAY:
            raise ValueError(
                "这条快照保存于本功能之前，没有可回填的策略参数；"
                "重新运行一次回测并保留即可。")
        if strategy_name == "fixed_times":
            fixed_times = str(form_state.get("fixed_times", "") or "").strip()
            if not fixed_times:
                raise ValueError("快照缺少固定时刻参数。")
            FixedTimeStrategy(fixed_times)
            self._fixed_times_var.set(fixed_times)
        elif strategy_name == "hedge_band":
            band_type = str(
                form_state.get("interval_type", "") or "absolute").strip()
            threshold = BacktestApp._comparison_finite(
                form_state.get("price_interval"))
            if band_type not in ("absolute", "relative", "sigma"):
                raise ValueError(f"快照的带宽单位无法识别：{band_type}")
            if threshold is None or threshold <= 0:
                raise ValueError("快照缺少有效的固定间隔带宽。")
            # 波动率口径要先写回：带宽的三口径换算依赖它，顺序反了会按旧
            # 口径换算出另外两种表示。
            sigma_source = str(
                form_state.get("sigma_source", "") or "implied")
            if sigma_source in SIGMA_SOURCE_DISPLAY:
                self._sigma_src_var.set(SIGMA_SOURCE_DISPLAY[sigma_source])
            sigma_window = BacktestApp._comparison_safe_int(
                form_state.get("sigma_window"), 20)
            self._sigma_win_var.set(str(max(2, sigma_window)))
            # 按保存时的那一种输入单位写回，另外两种由既有换算链路补齐；
            # 换算过一手再写回去，显示值会和当初跑的那次差一个舍入。
            {
                "absolute": self._band_abs_var,
                "relative": self._band_rel_var,
                "sigma": self._band_sigma_var,
            }[band_type].set(f"{threshold:.10g}")
            self._mark_band_edited(band_type)
            self._sync_band_inputs(band_type, strict=True)
        self._strategy_var.set(STRATEGY_DISPLAY[strategy_name])
        fallback_var = getattr(self, "_force_day_close_hedge_var", None)
        if fallback_var is not None:
            fallback_var.set(
                bool(form_state.get("force_day_close_hedge", False)))
        self._toggle_strategy()
        return strategy_name

    def _apply_selected_snapshot_to_form(self):
        """把结果池里聚焦那条快照的对冲参数送回左侧继续调。

        与同排的重命名、删除同一个作用对象：结果池里点中的那一行。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshot = self._saved_backtests[result_id]
        try:
            self._apply_snapshot_strategy_to_form(snapshot)
        except Exception as exc:
            messagebox.showerror("无法应用策略", str(exc))
            return
        self._set_status(
            f"已把『{snapshot.name}』的对冲策略应用到左侧参数  |  "
            "期权结构、方向与行情来源保持不变")

    def _show_saved_snapshot_params(self):
        """右键菜单「参数详情」：只读列出聚焦那条快照的全部原始输入。

        与「加载明细」的分工：那一条要的是**结果**（逐日损益、对冲图表），
        因此必须重放，代价是顶掉当前回测并切页，旧快照没有配方还点不动；
        这一条要的是**输入**，四个签名里全存着，读一读就有——不重放、不动
        当前回测、不依赖 ``replay``，上线前保留的快照照样打得开。
        """
        if not BacktestApp._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        BacktestApp._open_snapshot_params_window(
            self, self._saved_backtests[result_id])

    def _open_snapshot_params_window(self, snapshot):
        """结果池那条快照的参数详情窗。"""
        meta = [
            BacktestApp._saved_snapshot_origin_label(snapshot),
            snapshot.saved_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        detail = BacktestApp._snapshot_origin_detail(snapshot)
        if detail:
            meta.append(detail)
        return BacktestApp._open_params_window(
            self,
            heading=snapshot.name,
            subtitle="  ·  ".join(meta),
            sections=BacktestApp._snapshot_detail_sections(snapshot),
        )

    def _open_params_window(self, *, heading, subtitle, sections, notes=None):
        """摆出参数详情窗。结果池与策略优选共用这一套渲染。

        正文用只读 ``Text`` 而不是一排 ``Label``：内容长度随期权大类差着数
        倍（香草 7 个合约参数，雪球 18 个），Text 自带滚动，而且文本可以选中
        复制——把一条结果的参数贴给别人是这个窗口的第二个用途。

        ``notes`` 是 ``{组名: 说明}``，用于给整组加一句限定。策略优选那边靠它
        说明「期权参数是参考值，每段按段初价缩放」——那句话属于整组，挂在某
        一行上都不对。
        """
        notes = dict(notes or {})
        window = tk.Toplevel(self)
        window.title(f"参数详情 — {heading}")
        window.geometry("560x620")
        window.transient(self)
        body = ttk.Frame(window, style="Surface.TFrame", padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body, text=heading, style="Surface.TLabel",
            font=(_UI_FONT_FAMILY, 13, "bold"), anchor="w",
        ).pack(fill="x")
        ttk.Label(
            body, text=subtitle, style="SurfaceMuted.TLabel",
            anchor="w", wraplength=520, justify="left",
        ).pack(fill="x", pady=(2, 8))

        text_frame = ttk.Frame(body, style="Surface.TFrame")
        text_frame.pack(fill="both", expand=True)
        text = tk.Text(
            text_frame, wrap="word", relief="flat", padx=10, pady=8,
            bg=PALETTE["surface_alt"], fg=PALETTE["text"],
            font=(_MONO_FONT_FAMILY, 10), highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text.tag_configure(
            "group", foreground=PALETTE["primary"],
            font=(_UI_FONT_FAMILY, 11, "bold"), spacing1=10, spacing3=4)
        text.tag_configure("label", foreground=PALETTE["text_muted"])
        text.tag_configure("value", foreground=PALETTE["text"])
        text.tag_configure("empty", foreground=PALETTE["text_light"])
        text.tag_configure(
            "note", foreground=PALETTE["warning"], lmargin1=16, lmargin2=16,
            spacing3=2)

        # 取值全部起在同一竖列上。用制表位按**像素**对齐，不能按字符数补空格：
        # 等宽字体里一个汉字占两格而 len() 只算一个，`初始价格 S0`（7 字、11
        # 格）和 `期限(交易日)`（8 字、12 格）补完反而错开。
        # 制表位按全窗最长的字段名算，不是逐组算——逐组算的话五组的取值会各起
        # 各的位置。
        name_font = tkfont.Font(font=(_MONO_FONT_FAMILY, 10))
        stop = 14 + max(
            (name_font.measure(name)
             for _label, rows in sections for name, _v in rows),
            default=0) + 18
        text.configure(tabs=(stop,))
        for label, rows in sections:
            text.insert("end", f"{label}\n", "group")
            note = str(notes.get(label, "") or "").strip()
            if note:
                text.insert("end", f"{note}\n", "note")
            if not rows:
                text.insert("end", "  （本组没有已记录的输入）\n", "empty")
                continue
            for name, shown in rows:
                text.insert("end", f"  {name}\t", "label")
                text.insert("end", f"{shown}\n", "value")
        # 插完再锁：disabled 状态下 insert 是静默无效的。
        text.configure(state="disabled")

        ttk.Button(
            body, text="关闭", width=10, command=window.destroy,
        ).pack(anchor="e", pady=(10, 0))
        return window


# ============================================================
#  入口
# ============================================================


def main():
    app = BacktestApp()
    app.mainloop()


if __name__ == "__main__":
    main()
