# _*_ coding: utf-8 _*_
"""界面主题：matplotlib 后端、中文字体、配色、左侧表单的统一度量。

本模块在 **import 期**就把全局绘图环境配好（后端 → 字体 → rcParams），并
提供一组无状态的表单布局助手。它不 import 项目里的任何其他模块，因此界面
层任何一处都能安全引入，不会形成环。

import 顺序是有意的：``matplotlib.use()`` 必须早于 ``import
matplotlib.pyplot``。后端选择因此放在本模块最顶部——只要谁先 import 了
``deltalab_ui.theme``，后端就已经定死，其余模块再 import pyplot 都是安全的。
"""

import os
import platform
import sys
from tkinter import ttk

import matplotlib
# 拿不到窗口服务器时（CI 的 headless Linux、无 DISPLAY 的批处理调用），
# TkAgg 会抛 ImportError："Cannot load backend 'TkAgg' ... as 'headless' is
# currently running"。回退到 Agg 只为让 **import 本身**成功——真要画图的那批
# 测试仍在 xvfb 下跑，那时 DISPLAY 有效，选中的照样是 TkAgg。
#
# 这里不回退的话，`pytest -m "not gui"` 也救不了：marker 要把模块 import 完
# 才读得到，而炸点恰恰就在 import gui_app 这一步，于是连一条纯逻辑测试都收集
# 不起来（实测 CI 上 4 个模块全部 collection error）。
try:
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 打包态解压到 sys._MEIPASS; 开发态以**仓库根**为准。本文件在 deltalab_ui/
# 包内，比入口脚本深一层，所以要比原先多退一级——少退一级的话 assets/ 会被
# 拼成 deltalab_ui/assets/，图标就静默加载不到。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resource_path(*parts: str) -> str:
    base = getattr(sys, "_MEIPASS", _PROJECT_ROOT)
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

# ---- 左侧参数面板的统一度量 (像素) ----
# 三个分组 (期权类型 / 期权参数 / 回测设置) 以及它们内部的子面板共用同一套
# 列宽, 输入框的左右边缘才能跨分组对齐; 列宽由 grid 的 minsize 强制, 各控件
# 自己的 width= 只是字符数下限, 因此统一改这里就能整体调节表单尺寸。
FORM_LABEL_W     = 144   # 标签列宽 (含标签右侧留白; 容得下最长的雪球保证金标签)
FORM_INPUT_W     = 168   # 输入列宽: 所有 Entry / Combobox 等宽
FORM_LABEL_GAP   = 10    # 标签与输入框之间的留白
FORM_HINT_GAP    = 8     # 输入框与右侧说明文字之间的留白
FORM_ROW_PADY    = 4     # 每行上下留白 => 相邻控件间隔恒为 8
FORM_SECTION_PAD = 12    # 分组内边距
FORM_SECTION_GAP = 10    # 分组之间的间距
FORM_ENTRY_CHARS = 8     # 字符宽度只作下限, 实际宽度取 FORM_INPUT_W


def _form_grid(container):
    """把容器配成统一的三列表单: 标签 | 等宽输入 | 说明文字。

    只有第三列可伸缩, 面板拉宽时输入框不会跟着变形, 跨分组始终等宽对齐。
    """
    container.columnconfigure(0, minsize=FORM_LABEL_W, weight=0)
    container.columnconfigure(1, minsize=FORM_INPUT_W, weight=0)
    container.columnconfigure(2, weight=1)


def _form_label(parent, text, row, column=0, columnspan=1,
                style="Surface.TLabel"):
    """表单标签: 统一右侧留白与行距。"""
    widget = ttk.Label(parent, text=text, style=style)
    widget.grid(row=row, column=column, columnspan=columnspan, sticky="w",
                padx=(0, FORM_LABEL_GAP), pady=FORM_ROW_PADY)
    return widget


def _form_input(widget, row, column=1, columnspan=1, sticky="ew"):
    """表单输入: 贴住输入列左边缘, 宽度由列宽统一。

    多控件组合 (单选组、带勾选框的行) 传 columnspan=2 + sticky="w",
    让它们向右溢出到说明列, 而不是把输入列撑宽。
    """
    widget.grid(row=row, column=column, columnspan=columnspan,
                sticky=sticky, pady=FORM_ROW_PADY)
    return widget


def _form_hint(parent, row, text, column=2, columnspan=1):
    """输入框右侧的浅色说明文字。"""
    widget = ttk.Label(parent, text=text, style="SurfaceMuted.TLabel")
    widget.grid(row=row, column=column, columnspan=columnspan, sticky="w",
                padx=(FORM_HINT_GAP, 0), pady=FORM_ROW_PADY)
    return widget


def _form_separator(parent, row, columnspan=3):
    """分组内的轻分割线: 上下留白一致。"""
    widget = ttk.Separator(parent, orient="horizontal")
    widget.grid(row=row, column=0, columnspan=columnspan, sticky="ew",
                pady=(FORM_ROW_PADY + 4, FORM_ROW_PADY + 2))
    return widget


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
