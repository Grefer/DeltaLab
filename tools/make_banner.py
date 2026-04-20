"""生成 README 顶部横向 banner (assets/banner.png).

设计:
  - 左: deltalab.png 图标 (圆角蓝底 + Δ)
  - 中: 标题 "DeltaLab" 副标题 "Options Delta Dynamic Hedging Lab"
  - 右: 一条半透明的累计盈亏曲线, 强化"回测"观感
  - 底色: 与图标一致的品牌蓝渐变

用法: python tools/make_banner.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_icon import _find_font, _gradient_bg, _rounded_mask  # 复用


def _find_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    """找一个含简体中文字形的字体, 用于 banner 中文 tagline."""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

W, H = 1600, 420
RADIUS = 36

BLUE_TOP    = (37, 99, 235)
BLUE_BOTTOM = (30, 64, 175)
WHITE       = (255, 255, 255)
GOLD        = (184, 134, 11)
SOFT        = (219, 234, 254)   # 淡蓝, 用于副标题


def _render_banner() -> Image.Image:
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # 渐变底 + 圆角
    bg = _gradient_bg(max(W, H)).crop((0, 0, W, H)).convert("RGBA")
    # 让渐变水平方向微微变化: 在右侧盖一层淡化层
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for x in range(W):
        a = int(40 * (x / W))  # 0 -> 40
        sd.line([(x, 0), (x, H)], fill=(255, 255, 255, a))
    bg.alpha_composite(shade)
    mask = _rounded_mask(W, RADIUS) if W == H else None
    # 矩形圆角 mask (非正方形, 需要独立生成)
    rect_mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(rect_mask).rounded_rectangle(
        (0, 0, W - 1, H - 1), radius=RADIUS, fill=255
    )
    canvas.paste(bg, (0, 0), rect_mask)

    # ---- 左侧图标 ----
    icon_path = ASSETS / "deltalab.png"
    icon_size = 260
    if icon_path.exists():
        icon = Image.open(icon_path).convert("RGBA")
        icon = icon.resize((icon_size, icon_size), Image.LANCZOS)
        canvas.alpha_composite(icon, (70, (H - icon_size) // 2))

    # ---- 中部文字 ----
    draw = ImageDraw.Draw(canvas)
    title_font    = _find_font(132)
    subtitle_font = _find_font(36)
    tagline_font  = _find_cjk_font(28)

    text_x = 70 + icon_size + 60
    # 主标题
    draw.text((text_x, 110), "DeltaLab", font=title_font, fill=WHITE)
    # 英文副标题
    draw.text((text_x, 250), "Options Delta Dynamic Hedging Lab",
              font=subtitle_font, fill=SOFT)
    # 中文 tag
    draw.text((text_x, 300),
              "期权 · 对冲 · 回测 · 蒙特卡洛",
              font=tagline_font, fill=(207, 220, 244))

    # ---- 右侧累计盈亏曲线 ----
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    curve_left  = int(W * 0.55)
    curve_right = int(W * 0.96)
    baseline_y  = int(H * 0.55)
    amp         = int(H * 0.22)

    points = []
    n = 140
    for i in range(n + 1):
        t = i / n
        x = curve_left + t * (curve_right - curve_left)
        # 多频率叠加 + 轻微上行漂移, 像一条"卖方累计盈亏"曲线
        y = baseline_y \
            - math.sin(t * math.pi * 1.4) * amp * 0.55 \
            - math.sin(t * math.pi * 3.7 + 0.6) * amp * 0.25 \
            - t * amp * 0.35
        points.append((x, y))

    # 曲线发光效果
    od.line(points, fill=(255, 255, 255, 60), width=14, joint="curve")
    od.line(points, fill=(255, 255, 255, 130), width=8, joint="curve")
    od.line(points, fill=(255, 255, 255, 230), width=4, joint="curve")

    # 底部水平网格线 (像 matplotlib 图表)
    grid_y = [int(H * 0.25), int(H * 0.50), int(H * 0.75)]
    for gy in grid_y:
        od.line([(curve_left, gy), (curve_right, gy)],
                fill=(255, 255, 255, 40), width=1)
    # 左右竖线
    od.line([(curve_left, int(H * 0.15)), (curve_left, int(H * 0.85))],
            fill=(255, 255, 255, 70), width=2)

    # 端点小圆
    for (px, py) in (points[0], points[-1]):
        od.ellipse((px - 6, py - 6, px + 6, py + 6),
                   fill=(255, 255, 255, 235))

    # 金色 accent 在曲线终点
    end_x, end_y = points[-1]
    od.ellipse((end_x - 12, end_y - 12, end_x + 12, end_y + 12),
               outline=GOLD + (255,), width=3)

    canvas.alpha_composite(overlay)

    # ---- 底部金色细线 (整条 banner 下方) ----
    gold_y = H - 14
    draw.rounded_rectangle(
        (int(W * 0.08), gold_y, int(W * 0.92), gold_y + 4),
        radius=2, fill=GOLD + (255,)
    )

    # 再裁一次圆角, 保证 overlay 没超出
    final = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    final.paste(canvas, (0, 0), rect_mask)
    return final


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    banner = _render_banner()
    out = ASSETS / "banner.png"
    banner.save(out, format="PNG")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
