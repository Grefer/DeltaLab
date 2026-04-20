"""生成 README 工作流示意图 (assets/workflow.png).

三列布局 (从左到右):
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
GOLD        = "#B8860B"
SUCCESS     = "#16A34A"

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


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


def _card(ax, xy, size, title, items, edge, fill=SURFACE,
          title_color="#FFFFFF"):
    """画一张带顶部标题条的圆角卡片.

    xy : 左下角
    size : (w, h)
    title : 顶部标题 (反色)
    items : 条目列表 (每个一行)
    edge : 主题色 (同时用于标题条底色)
    """
    x, y = xy
    w, h = size
    header_h = 0.9
    radius = 0.25

    # 卡片主体
    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.6, facecolor=fill, edgecolor=edge, zorder=2,
    )
    ax.add_patch(body)

    # 顶部标题条: 先画全宽圆角矩形(作为标题底), 再用普通矩形遮住下半圆角
    header_bg = FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=edge, zorder=3,
    )
    ax.add_patch(header_bg)
    ax.add_patch(mpatches.Rectangle(
        (x, y + h - header_h), w, header_h - radius,
        linewidth=0, facecolor=edge, zorder=3,
    ))
    ax.text(x + w / 2, y + h - header_h / 2, title,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=title_color, zorder=4)

    # 条目: 均匀铺开
    n = len(items)
    inner_top = y + h - header_h - 0.35
    inner_bot = y + 0.35
    if n > 0:
        step = (inner_top - inner_bot) / n
        for i, it in enumerate(items):
            ty = inner_top - step * (i + 0.5)
            ax.text(x + 0.45, ty, it, ha="left", va="center",
                    fontsize=11, color=TEXT, zorder=4)


def _arrow(ax, xy_from, xy_to, color=PRIMARY_ACT, lw=2.0):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>,head_length=0.35,head_width=0.28",
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def render_workflow() -> plt.Figure:
    plt.rcParams["font.sans-serif"] = [_pick_cjk_font(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 数据坐标系: (0..32) x (0..19)
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=150)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 19)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ---- 顶部标题 ----
    ax.text(16, 18.1, "DeltaLab 回测工作流",
            ha="center", va="center", fontsize=19, fontweight="bold",
            color=TEXT)
    ax.text(16, 17.2,
            "参数配置 → 价格序列 → 动态对冲引擎 → 多维结果可视化",
            ha="center", va="center", fontsize=12, color=TEXT_MUTED)

    # 三列顶栏标签
    ax.text(5.25, 15.7, "① 配置输入",  ha="center", va="center",
            fontsize=12, fontweight="bold", color=PRIMARY_ACT)
    ax.text(16,   15.7, "② 回测引擎",  ha="center", va="center",
            fontsize=12, fontweight="bold", color=PRIMARY_ACT)
    ax.text(26.75, 15.7, "③ 结果输出",  ha="center", va="center",
            fontsize=12, fontweight="bold", color=PRIMARY_ACT)

    # ================== 左列: 输入 ==================
    CARD_W = 8.5
    LEFT_X = 1.0

    # 期权 类型 & 参数
    _card(ax, (LEFT_X, 11.0), (CARD_W, 4.0),
          "期权类型 & 参数",
          ["• 香草 / 累计 / 亚式 / 气囊",
           "• 13 种子类型",
           "• s0 / K / T / σ / cp / r / q ..."],
          edge=PRIMARY)

    # 数据源
    _card(ax, (LEFT_X, 6.2), (CARD_W, 4.0),
          "数据来源",
          ["• 模拟 (MC) — 种子可控",
           "• CSV — 本地历史行情",
           "• Wind — 实时终端接入"],
          edge=ACCENT)

    # 对冲设置
    _card(ax, (LEFT_X, 1.4), (CARD_W, 4.0),
          "对冲策略 & 频率",
          ["• fixed_freq / sigma_band",
           "• 日频 · 60 分 · 5 分 · 1 分",
           "• 滑点 / 手续费 / 乘数取整"],
          edge=GOLD)

    # ================== 中列: 引擎 ==================
    ENGINE_X = 11.5
    ENGINE_W = 9.0
    _card(ax, (ENGINE_X, 4.5), (ENGINE_W, 10.5),
          "HedgeBacktest 引擎",
          ["• GBM 路径 + 反对称采样",
           "• per-path MC seed",
           "• Δ / Γ / ν / Θ / ρ (有限差分)",
           "• fixed_freq / sigma_band 策略调度",
           "• 日频 / 日内多频率 (spd ∈ 1·4·48·240)",
           "• 滑点 · 手续费 · 乘数取整",
           "• run() 单路径 + run_multi() 并行",
           "• S_ref → S_real 自动缩放"],
          edge=PRIMARY_ACT, fill=PRIMARY_LIT)

    # ================== 右列: 输出 ==================
    RIGHT_X = 22.5
    _card(ax, (RIGHT_X, 11.0), (CARD_W, 4.0),
          "单路径结果",
          ["• 回测摘要 (盈亏分解 + Greeks)",
           "• 对冲图表 (6 宫格)",
           "• 波动率分析 (RV vs IV)"],
          edge=SUCCESS)

    _card(ax, (RIGHT_X, 6.2), (CARD_W, 4.0),
          "多路径分析",
          ["• 盈亏 / 对冲误差分布",
           "• RV 分布 · 5% VaR",
           "• 盈亏 vs (IV − RV) 回归"],
          edge=SUCCESS)

    _card(ax, (RIGHT_X, 1.4), (CARD_W, 4.0),
          "结构 & 明细",
          ["• 价格 / Greeks 扫描曲线",
           "• 关键价位标记 (K / H / KI / E / P)",
           "• 每日明细表格 · CSV 导出"],
          edge=SUCCESS)

    # ================== 箭头 ==================
    # 左 -> 引擎: 每张卡片右中 → 引擎左侧对应 y
    left_right_x = LEFT_X + CARD_W
    engine_left_x = ENGINE_X
    for y_src, y_dst in [(13.0, 12.5), (8.2, 9.75), (3.4, 7.0)]:
        _arrow(ax, (left_right_x + 0.1, y_src),
               (engine_left_x - 0.1, y_dst))

    # 引擎 -> 右: 引擎右侧对应 y → 每张卡片左中
    engine_right_x = ENGINE_X + ENGINE_W
    right_left_x = RIGHT_X
    for y_src, y_dst in [(12.5, 13.0), (9.75, 8.2), (7.0, 3.4)]:
        _arrow(ax, (engine_right_x + 0.1, y_src),
               (right_left_x - 0.1, y_dst))

    # 底部脚注
    ax.text(16, 0.55,
            "配置定义: gui_app.py:OPTION_CLASSES    ·    "
            "引擎: pricing/hedge_backtest.py    ·    "
            "GUI 入口: gui_app.py:main()",
            ha="center", va="center", fontsize=10,
            color=TEXT_MUTED, style="italic")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig = render_workflow()
    out = ASSETS / "workflow.png"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3,
                dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
