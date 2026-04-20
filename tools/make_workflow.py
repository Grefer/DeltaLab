"""生成 README 工作流示意图 (assets/workflow.png).

三层结构 (从左到右):
  1. 输入 (Inputs):    期权类型 / 参数 / 数据源 / 对冲策略
  2. 引擎 (Engine):    HedgeBacktest + MC 路径生成 + 策略调度
  3. 输出 (Outputs):   6 个结果 Tab (摘要/对冲图表/波动率/盈亏分布/结构/明细)

使用 matplotlib 手绘, 与项目整体风格 (PALETTE) 保持一致.
用法: python tools/make_workflow.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

# ---- 同 gui_app.py PALETTE 对齐 ----
BG          = "#F3F5F9"
SURFACE     = "#FFFFFF"
BORDER      = "#D8DEE8"
TEXT        = "#1F2937"
TEXT_MUTED  = "#6B7280"
PRIMARY     = "#2563EB"
PRIMARY_ACT = "#1E40AF"
PRIMARY_LIT = "#EFF6FF"
ACCENT      = "#0EA5E9"
ACCENT_LIT  = "#E0F2FE"
GOLD        = "#B8860B"
SUCCESS     = "#16A34A"
SUCCESS_LIT = "#F0FDF4"

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


# ------------------------------------------------------------------
#  字体 (CJK 自动挑选)
# ------------------------------------------------------------------
def _pick_cjk_font() -> str:
    from matplotlib import font_manager
    candidates = [
        "PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
        "Songti SC", "Microsoft YaHei", "SimHei", "SimSun",
        "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Sans"


# ------------------------------------------------------------------
#  画矩形卡片
# ------------------------------------------------------------------
def _card(ax, xy, size, title, items, fill, edge, title_color="#FFFFFF",
          item_color=None):
    x, y = xy
    w, h = size
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=1.5, facecolor=fill, edgecolor=edge, zorder=2,
    )
    ax.add_patch(box)
    # 标题条
    header_h = 0.55
    header = FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=0, facecolor=edge, zorder=3,
    )
    ax.add_patch(header)
    # 为让标题条下缘平齐, 再盖一个矩形遮住下半圆角
    ax.add_patch(mpatches.Rectangle(
        (x, y + h - header_h), w, header_h / 2,
        linewidth=0, facecolor=edge, zorder=3,
    ))
    ax.text(x + w / 2, y + h - header_h / 2, title,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=title_color, zorder=4)

    # 列出条目
    if item_color is None:
        item_color = TEXT
    n = len(items)
    inner_top = y + h - header_h - 0.25
    inner_bot = y + 0.25
    if n > 0:
        step = (inner_top - inner_bot) / n
        for i, it in enumerate(items):
            ty = inner_top - step * (i + 0.5)
            ax.text(x + 0.35, ty, it, ha="left", va="center",
                    fontsize=10, color=item_color, zorder=4)


def _arrow(ax, xy_from, xy_to, color=PRIMARY_ACT):
    x1, y1 = xy_from
    x2, y2 = xy_to
    ax.add_patch(FancyArrow(
        x1, y1, x2 - x1, y2 - y1,
        width=0.05, head_width=0.28, head_length=0.28,
        length_includes_head=True,
        facecolor=color, edgecolor=color, zorder=1,
    ))


# ------------------------------------------------------------------
#  主绘制
# ------------------------------------------------------------------
def render_workflow() -> plt.Figure:
    plt.rcParams["font.sans-serif"] = [_pick_cjk_font(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ---- 标题 ----
    ax.text(10, 8.5, "DeltaLab 回测工作流",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=TEXT)
    ax.text(10, 8.0,
            "参数配置 → 价格序列 → 动态对冲引擎 → 多维结果可视化",
            ha="center", va="center", fontsize=11, color=TEXT_MUTED)

    # ================== 左列: 输入 ==================
    # 期权类型/参数
    _card(ax, (0.6, 4.9), (4.4, 2.5),
          "① 期权类型 & 参数",
          ["• 香草 / 累计 / 亚式 / 气囊",
           "• 13 种子类型",
           "• s0 / K / T / σ / cp / r / q ..."],
          fill=SURFACE, edge=PRIMARY)

    # 数据源
    _card(ax, (0.6, 1.9), (4.4, 2.5),
          "② 数据来源",
          ["• 模拟 (MC) — 种子可控",
           "• CSV — 本地历史行情",
           "• Wind — 实时终端接入"],
          fill=SURFACE, edge=ACCENT)

    # 对冲设置 (底部, 低调一些)
    _card(ax, (0.6, -0.3), (4.4, 1.7),
          "③ 对冲策略",
          ["• fixed_freq / sigma_band",
           "• 日频 · 60分 · 5分 · 1分"],
          fill=SURFACE, edge=GOLD)

    # ================== 中列: 引擎 ==================
    _card(ax, (7.4, 2.5), (5.2, 4.6),
          "HedgeBacktest 引擎",
          ["• GBM 路径 + 反对称采样",
           "• per-path MC seed",
           "• Δ/Γ/ν/Θ/ρ (有限差分)",
           "• 滑点 / 手续费 / 乘数取整",
           "• 单路径 + run_multi 并行",
           "• 实盘模式自动缩放 S_ref→S_real"],
          fill=PRIMARY_LIT, edge=PRIMARY_ACT, title_color="#FFFFFF")

    # ================== 右列: 输出 ==================
    _card(ax, (15.0, 5.8), (4.4, 2.2),
          "④ 单路径结果",
          ["• 回测摘要 (盈亏分解 + Greeks)",
           "• 对冲图表 (6 宫格)",
           "• 波动率分析 (RV vs IV)"],
          fill=SURFACE, edge=SUCCESS)

    _card(ax, (15.0, 3.2), (4.4, 2.2),
          "⑤ 多路径分析",
          ["• 盈亏 / 对冲误差分布",
           "• RV 分布 · 5% VaR",
           "• 盈亏 vs (IV − RV) 回归"],
          fill=SURFACE, edge=SUCCESS)

    _card(ax, (15.0, 0.6), (4.4, 2.2),
          "⑥ 结构 & 明细",
          ["• 价格/Greeks 扫描曲线",
           "• 关键价位标记 (K/H/KI/E/P)",
           "• 每日明细表格 · CSV 导出"],
          fill=SURFACE, edge=SUCCESS)

    # ================== 箭头 ==================
    # 左 -> 中
    for y in (6.15, 3.15, 0.55):
        _arrow(ax, (5.1, y), (7.3, 4.8), color=PRIMARY_ACT)
    # 中 -> 右
    for y in (6.9, 4.3, 1.7):
        _arrow(ax, (12.7, 4.8), (14.9, y), color=PRIMARY_ACT)

    # 底部脚注
    ax.text(10, -0.8,
            "配置项定义于 gui_app.py:OPTION_CLASSES · "
            "引擎在 pricing/hedge_backtest.py · "
            "GUI 入口 gui_app.py:main()",
            ha="center", va="top", fontsize=9, color=TEXT_MUTED, style="italic")

    fig.tight_layout(pad=0.5)
    return fig


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig = render_workflow()
    out = ASSETS / "workflow.png"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
